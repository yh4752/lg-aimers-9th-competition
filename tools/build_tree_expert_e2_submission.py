"""Build the accepted Tree Expert E2 submission through the sole packager."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import importlib.metadata
from pathlib import Path
import stat
import sys
from types import MappingProxyType, ModuleType
from typing import Mapping
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from experiments.tree_expert.e2_full_fit import AcceptedForFullFit
from experiments.tree_expert.e2_inference import load_inference_runtime
from submission.contract import PackageRequest, PackageResult
from submission.package import build_submission_package
from submission.tree_expert_e2_candidate import (
    TREE_E2_ADAPTER_ID,
    candidate_metadata,
    import_tree_expert_e2_candidate,
    render_validation_script,
)
from submission.tree_expert_e2_existing_evidence import build_tree_e2_acceptance


class TreeE2SubmissionBuildError(ValueError):
    """Raised before an invalid Tree E2 submission can be handed to the user."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_python311_environment() -> dict[str, object]:
    if sys.version_info[:3] != (3, 11, 15):
        raise TreeE2SubmissionBuildError(
            "final build requires Python 3.11.15, found "
            + ".".join(str(value) for value in sys.version_info[:3])
        )
    try:
        versions = {
            "catboost": importlib.metadata.version("catboost"),
            "pandas": importlib.metadata.version("pandas"),
            "numpy": importlib.metadata.version("numpy"),
        }
    except importlib.metadata.PackageNotFoundError as error:
        raise TreeE2SubmissionBuildError(f"required package is missing: {error.name}") from error
    expected = {"catboost": "1.2.10", "pandas": "2.0.3", "numpy": "1.26.4"}
    if versions != expected:
        raise TreeE2SubmissionBuildError(f"official package versions differ: {versions}")
    return {
        "status": "passed",
        "python": "3.11.15",
        **versions,
    }


def load_official_sample(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise TreeE2SubmissionBuildError("official data directory is unsafe")
    test_path = root / "test.csv"
    sample_path = root / "sample_submission.csv"
    if any(path.is_symlink() or not path.is_file() for path in (test_path, sample_path)):
        raise TreeE2SubmissionBuildError("official sample input is missing or unsafe")
    test = pd.read_csv(test_path, dtype={"row_id": "string"})
    sample = pd.read_csv(sample_path, dtype={"row_id": "string"})
    if len(test) != 5 or len(sample) != 5:
        raise TreeE2SubmissionBuildError("official local sample must contain five rows")
    if sample.columns.tolist() != ["row_id", "control_success"]:
        raise TreeE2SubmissionBuildError("sample submission columns differ")
    test_ids = test["row_id"].astype("string")
    sample_ids = sample["row_id"].astype("string")
    if (
        test_ids.isna().any()
        or sample_ids.isna().any()
        or test_ids.astype(str).duplicated().any()
        or sample_ids.astype(str).duplicated().any()
        or set(test_ids.astype(str)) != set(sample_ids.astype(str))
    ):
        raise TreeE2SubmissionBuildError("official sample row IDs differ")
    return test, sample


def load_rendered_runtime(source: bytes) -> ModuleType:
    if not isinstance(source, bytes) or not source:
        raise TreeE2SubmissionBuildError("rendered runtime is empty")
    module = ModuleType("tree_expert_e2_final_runtime")
    module.__file__ = "script.py"
    try:
        exec(compile(source, "script.py", "exec"), module.__dict__)
    except Exception as error:
        raise TreeE2SubmissionBuildError("rendered runtime cannot be loaded") from error
    for name in ("load_frozen_predictor", "validate_inputs", "main"):
        if not callable(getattr(module, name, None)):
            raise TreeE2SubmissionBuildError("rendered runtime contract differs")
    return module


def _project_sizes(
    runtime: bytes, requirements: bytes, model_dir: Path
) -> tuple[int, int]:
    members = {"script.py": runtime, "requirements.txt": requirements}
    for path in sorted(model_dir.rglob("*")):
        if path.is_symlink():
            raise TreeE2SubmissionBuildError("model directory contains a symlink")
        if path.is_file():
            members[f"model/{path.relative_to(model_dir).as_posix()}"] = path.read_bytes()
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
    path: str | Path,
    *,
    expected_runtime: bytes,
    expected_requirements: bytes,
    expected_model: Mapping[str, bytes],
) -> None:
    expected_names = {
        "script.py",
        "requirements.txt",
        *(f"model/{name}" for name in expected_model),
    }
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise TreeE2SubmissionBuildError("created submission layout differs")
            if archive.read("script.py") != expected_runtime:
                raise TreeE2SubmissionBuildError("created script.py differs")
            if archive.read("requirements.txt") != expected_requirements:
                raise TreeE2SubmissionBuildError("created requirements.txt differs")
            for name, value in expected_model.items():
                if archive.read(f"model/{name}") != value:
                    raise TreeE2SubmissionBuildError(f"created model member differs: {name}")
            bad = archive.testzip()
            if bad is not None:
                raise TreeE2SubmissionBuildError(f"created submission member is corrupt: {bad}")
    except zipfile.BadZipFile as error:
        raise TreeE2SubmissionBuildError("created submission is not a ZIP") from error


def _trusted_probabilities(candidate, test: pd.DataFrame) -> np.ndarray:
    token = AcceptedForFullFit(
        candidate_id=candidate.candidate_id,
        predictor="catboost",
        seeds=candidate.seeds,
        iterations=MappingProxyType(dict(candidate.iterations)),
        decision_sha256=sha256(
            json_bytes(candidate.acceptance_decision)
        ).hexdigest(),
    )
    runtime = load_inference_runtime(
        token=token,
        frozen_state_dir=candidate.model_dir / "frozen_state",
        catboost_model_dir=candidate.model_dir / "models",
        blend_method="catboost",
        catboost_weight=1.0,
        device="cpu",
    )
    return np.asarray(runtime.predict(test.copy(deep=True)), dtype="float64")


def json_bytes(value: Mapping[str, object]) -> bytes:
    import json

    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def run_build(args: argparse.Namespace) -> PackageResult:
    project = PROJECT_ROOT.resolve(strict=True)
    output = Path(args.output_dir).expanduser().resolve(strict=False)
    try:
        output.relative_to(project)
    except ValueError as error:
        raise TreeE2SubmissionBuildError("output directory is outside project root") from error
    if output.exists():
        raise TreeE2SubmissionBuildError(f"output directory already exists: {output}")

    environment = require_python311_environment()
    candidate = import_tree_expert_e2_candidate(args.handoff, output / "candidate")
    runtime_bytes = render_validation_script(candidate)
    runtime_module = load_rendered_runtime(runtime_bytes)
    test, sample = load_official_sample(args.official_data)
    predictor = runtime_module.load_frozen_predictor(candidate.model_dir)
    current = np.asarray(predictor.predict_batch(test.copy(deep=True)), dtype="float64")
    trusted = _trusted_probabilities(candidate, test)
    if (
        current.shape != (5,)
        or trusted.shape != (5,)
        or not np.isfinite(current).all()
        or not np.array_equal(current, trusted)
    ):
        raise TreeE2SubmissionBuildError("standalone and reviewed E2 predictions differ")

    requirements = b"catboost==1.2.10\n"
    requirements_path = output / "requirements.txt"
    requirements_path.parent.mkdir(parents=True, exist_ok=True)
    requirements_path.write_bytes(requirements)
    package_bytes, extracted_bytes = _project_sizes(
        runtime_bytes, requirements, candidate.model_dir
    )
    package_time = datetime.now(ZoneInfo("Asia/Seoul"))
    acceptance = build_tree_e2_acceptance(
        project_root=project,
        candidate=candidate,
        test_frame=test,
        sample_frame=sample,
        runtime_bytes=runtime_bytes,
        output_dir=output / "evidence",
        load_predictor=lambda: runtime_module.load_frozen_predictor(candidate.model_dir),
        python_probe=environment,
        package_bytes=package_bytes,
        extracted_bytes=extracted_bytes,
        policy_path=project / "competition_rules/policy.json",
        policy_review_path=project / "reports/rules/2026-08-27-final-policy-review.json",
        package_time=package_time,
    )
    request = PackageRequest(
        project_root=project,
        policy_path=project / "competition_rules/policy.json",
        policy_review_path=project / "reports/rules/2026-08-27-final-policy-review.json",
        acceptance_path=acceptance.acceptance_path,
        full_audit_manifest_path=acceptance.audit_manifest,
        runtime_benchmark_path=acceptance.benchmark_path,
        model_dir=candidate.model_dir,
        requirements_path=requirements_path,
        adapter_id=TREE_E2_ADAPTER_ID,
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
        expected_runtime=runtime_bytes,
        expected_requirements=requirements,
        expected_model=expected_model,
    )
    if _sha256_file(result.archive_path) != result.archive_sha256:
        raise TreeE2SubmissionBuildError("created submission SHA-256 differs from receipt")
    print(
        "TREE_E2_SUBMISSION_READY "
        f"archive={result.archive_path} sha256={result.archive_sha256} "
        f"bytes={result.archive_bytes} candidate={candidate.candidate_id} "
        f"handoff_sha256={candidate.handoff_sha256}",
        flush=True,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the hash-gated Tree Expert E2 DACON submission"
    )
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--official-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    run_build(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
