"""Build the reviewed TabM submission from hash-bound existing GPU evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import importlib.metadata
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Mapping
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from submission.contract import PackageRequest, PackageResult
from submission.package import build_submission_package
from submission.tabm_candidate import (
    CANDIDATE_ID,
    load_imported_candidate,
    render_validation_script,
)
from submission.tabm_existing_evidence import (
    build_reused_gpu_acceptance,
    verify_existing_gpu_evidence,
)


class SubmissionBuildError(ValueError):
    """Raised before an unverified submission can be handed to the user."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_python311_environment() -> dict[str, object]:
    """Require the official Python patch and submitted dependency versions."""

    if sys.version_info[:3] != (3, 11, 15):
        raise SubmissionBuildError(
            "final build requires the official Python 3.11.15 environment, "
            f"found {'.'.join(str(value) for value in sys.version_info[:3])}"
        )
    try:
        tabm_version = importlib.metadata.version("tabm")
        embeddings_version = importlib.metadata.version("rtdl-num-embeddings")
        torch_version = importlib.metadata.version("torch")
        pandas_version = importlib.metadata.version("pandas")
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError as error:
        raise SubmissionBuildError(f"required package is missing: {error.name}") from error
    if tabm_version != "0.0.3" or embeddings_version != "0.0.12":
        raise SubmissionBuildError("submitted dependency versions differ")
    if (
        torch_version.split("+", 1)[0] != "2.7.1"
        or pandas_version != "2.0.3"
        or numpy_version != "1.26.4"
    ):
        raise SubmissionBuildError("official base library versions differ")
    return {
        "status": "passed",
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "tabm": tabm_version,
        "rtdl_num_embeddings": embeddings_version,
        "torch": torch_version,
        "pandas": pandas_version,
        "numpy": numpy_version,
    }


def load_rendered_runtime(runtime: bytes) -> ModuleType:
    """Load the exact bytes that will become script.py."""

    if not isinstance(runtime, bytes) or not runtime:
        raise SubmissionBuildError("rendered runtime is empty")
    module = ModuleType("tabm_final_submission_runtime")
    module.__file__ = "script.py"
    try:
        exec(compile(runtime, "script.py", "exec"), module.__dict__)
    except Exception as error:
        raise SubmissionBuildError("rendered runtime cannot be loaded") from error
    required = ("load_frozen_predictor", "validate_inputs", "main")
    if any(not callable(getattr(module, name, None)) for name in required):
        raise SubmissionBuildError("rendered runtime contract differs")
    return module


def load_official_sample(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the official five-row local evaluator sample."""

    root = Path(data_dir).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise SubmissionBuildError("official data directory is unsafe")
    test_path = root / "test.csv"
    sample_path = root / "sample_submission.csv"
    if any(path.is_symlink() or not path.is_file() for path in (test_path, sample_path)):
        raise SubmissionBuildError("official sample input is missing or unsafe")
    test = pd.read_csv(test_path, dtype={"row_id": "string"})
    sample = pd.read_csv(sample_path, dtype={"row_id": "string"})
    if len(test) != 5 or len(sample) != 5:
        raise SubmissionBuildError("official local sample must contain five rows")
    if sample.columns.tolist() != ["row_id", "control_success"]:
        raise SubmissionBuildError("sample submission columns differ")
    if (
        test["row_id"].isna().any()
        or sample["row_id"].isna().any()
        or test["row_id"].duplicated().any()
        or set(test["row_id"].astype(str)) != set(sample["row_id"].astype(str))
    ):
        raise SubmissionBuildError("official sample row IDs differ")
    return test, sample


def _project_sizes(
    runtime: bytes, requirements: bytes, model_dir: Path
) -> tuple[int, int]:
    members = {"script.py": runtime, "requirements.txt": requirements}
    for path in sorted(model_dir.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise SubmissionBuildError("model directory contains an unsafe member")
        members[f"model/{path.name}"] = path.read_bytes()
    import io

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, value in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, value)
    return len(stream.getvalue()), sum(len(value) for value in members.values())


def verify_created_package(
    archive_path: str | Path,
    *,
    expected_runtime: bytes,
    expected_requirements: bytes,
    expected_model: Mapping[str, bytes],
) -> None:
    """Verify exact final bytes and reject additional archive members."""

    path = Path(archive_path)
    expected_names = {
        "script.py",
        "requirements.txt",
        *(f"model/{name}" for name in expected_model),
    }
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise SubmissionBuildError("created submission layout differs")
            if archive.read("script.py") != expected_runtime:
                raise SubmissionBuildError("created script.py differs")
            if archive.read("requirements.txt") != expected_requirements:
                raise SubmissionBuildError("created requirements.txt differs")
            for name, value in expected_model.items():
                if archive.read(f"model/{name}") != value:
                    raise SubmissionBuildError(f"created model member differs: {name}")
            bad = archive.testzip()
            if bad is not None:
                raise SubmissionBuildError(f"created submission member is corrupt: {bad}")
    except zipfile.BadZipFile as error:
        raise SubmissionBuildError("created submission is not a ZIP") from error


def run_build(args: argparse.Namespace) -> PackageResult:
    """Run all gates and create the final local archive exactly once."""

    project = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).expanduser().resolve(strict=False)
    try:
        output.relative_to(project)
    except ValueError as error:
        raise SubmissionBuildError("output directory is outside project root") from error
    if output.exists():
        raise SubmissionBuildError(f"output directory already exists: {output}")

    candidate = load_imported_candidate(args.candidate_root)
    gpu = verify_existing_gpu_evidence(
        stage_c_delivery=args.stage_c_delivery,
        stage_d_delivery=args.stage_d_delivery,
        candidate=candidate,
    )
    runtime = render_validation_script(candidate)
    environment = require_python311_environment()
    runtime_module = load_rendered_runtime(runtime)
    test, sample = load_official_sample(args.official_data)
    predictor = runtime_module.load_frozen_predictor(candidate.model_dir, device="cpu")
    probabilities = np.asarray(
        predictor.predict_batch(test.copy(deep=True), batch_size=2048), dtype="float64"
    )
    if probabilities.shape != (5,) or not np.isfinite(probabilities).all():
        raise SubmissionBuildError("Python 3.11 sample predictions are invalid")
    environment["probabilities"] = [float(value) for value in probabilities]

    requirements_path = (
        project / "artifacts/tabm_submission_validation_handoff_v2/requirements.txt"
    )
    requirements = requirements_path.read_bytes()
    if requirements != b"tabm==0.0.3\nrtdl-num-embeddings==0.0.12\n":
        raise SubmissionBuildError("submitted requirements differ")
    package_bytes, extracted_bytes = _project_sizes(
        runtime, requirements, candidate.model_dir
    )
    package_time = datetime.now(ZoneInfo("Asia/Seoul"))
    acceptance = build_reused_gpu_acceptance(
        project_root=project,
        candidate=candidate,
        gpu_evidence=gpu,
        test_frame=test,
        sample_frame=sample,
        runtime_bytes=runtime,
        output_dir=output / "evidence",
        load_predictor=lambda: runtime_module.load_frozen_predictor(
            candidate.model_dir, device="cpu"
        ),
        python_probe=environment,
        package_bytes=package_bytes,
        extracted_bytes=extracted_bytes,
        policy_path=project / "competition_rules/policy.json",
        policy_review_path=project
        / "reports/rules/2026-08-15-final-policy-review.json",
        package_time=package_time,
    )
    request = PackageRequest(
        project_root=project,
        policy_path=project / "competition_rules/policy.json",
        policy_review_path=project
        / "reports/rules/2026-08-15-final-policy-review.json",
        acceptance_path=acceptance.acceptance_path,
        full_audit_manifest_path=acceptance.audit_manifest,
        runtime_benchmark_path=acceptance.benchmark_path,
        model_dir=candidate.model_dir,
        requirements_path=requirements_path,
        adapter_id=CANDIDATE_ID,
        archive_path=output / "submit.zip",
        receipt_path=output / "submission_receipt.json",
        package_time=package_time,
    )
    result = build_submission_package(request)
    expected_model = {
        name: (candidate.model_dir / name).read_bytes()
        for name in sorted(candidate.member_sha256)
    }
    verify_created_package(
        result.archive_path,
        expected_runtime=runtime,
        expected_requirements=requirements,
        expected_model=expected_model,
    )
    if _sha256_file(result.archive_path) != result.archive_sha256:
        raise SubmissionBuildError("created submission SHA-256 differs from receipt")
    print(
        "SUBMISSION_PACKAGE_READY "
        f"archive={result.archive_path} receipt={result.receipt_path} "
        f"archive_sha256={result.archive_sha256}",
        flush=True,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the hash-gated Version D TabM DACON submission"
    )
    parser.add_argument("--stage-c-delivery", type=Path, required=True)
    parser.add_argument("--stage-d-delivery", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--official-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    run_build(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
