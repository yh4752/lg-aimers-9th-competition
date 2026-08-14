"""Validation contracts for the frozen TabM evaluator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
from math import isfinite
import os
from pathlib import Path
import platform
import resource
import shutil
import stat
import subprocess
import sys
from time import monotonic
from typing import Mapping
import zipfile

import numpy as np
import pandas as pd

from competition_rules.contract import load_policy
from competition_rules.evidence_gate import AuditIdentity, run_phased_independence_audit
from submission.tabm_candidate import (
    CANDIDATE_ID,
    load_imported_candidate,
    render_validation_script,
)


AUDIT_SCOPE = "official_sample_plus_synthetic_scale"
CAPACITY_ROWS = 245_789


class TabMValidationError(ValueError):
    """Raised when compatibility or capacity evidence is insufficient."""


@dataclass(frozen=True)
class ValidationBundles:
    review_path: Path
    resume_path: Path
    review_sha256: str
    resume_sha256: str


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _zip_bytes(members: Mapping[str, bytes]) -> bytes:
    import io

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, value in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, value)
    return stream.getvalue()


def _write_new(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(value)
    except FileExistsError as error:
        raise TabMValidationError(f"validation output already exists: {path}") from error


def build_validation_bundles(
    evidence_root: str | Path,
    report: Mapping[str, object],
    output_dir: str | Path,
) -> ValidationBundles:
    """Package review evidence without model files or official input rows."""

    validate_report(report)
    root = Path(evidence_root)
    required = {
        "environment.json",
        "install.log",
        "validation.log",
        "full_audit/full_audit_manifest.json",
    }
    available = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if not required.issubset(available):
        raise TabMValidationError("validation evidence is incomplete")
    if any(
        name.startswith("model/") or name.endswith((".pt", ".csv"))
        for name in available
    ):
        raise TabMValidationError("validation evidence contains model or input data")
    evidence = {
        name: (root / name).read_bytes()
        for name in sorted(available)
        if name in {"environment.json", "install.log", "validation.log"}
        or name.startswith("full_audit/")
    }
    review_members = dict(evidence)
    review_members["validation_report.json"] = _canonical_json(dict(report))
    review_manifest = {
        "schema_version": 1,
        "artifact_kind": "tabm_submission_validation_review",
        "audit_scope": AUDIT_SCOPE,
        "identity": report["identity"],
        "members": {
            name: sha256(value).hexdigest()
            for name, value in sorted(review_members.items())
        },
    }
    review_members["manifest.json"] = _canonical_json(review_manifest)

    resume_members = dict(evidence)
    resume_members["resume_metadata.json"] = _canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "tabm_submission_validation_resume",
            "audit_scope": AUDIT_SCOPE,
            "identity": report["identity"],
            "members": {
                name: sha256(value).hexdigest()
                for name, value in sorted(resume_members.items())
            },
        }
    )
    destination = Path(output_dir)
    review_path = destination / "tabm_submission_validation_review_bundle.zip"
    resume_path = destination / "tabm_submission_validation_resume_bundle.zip"
    review_value = _zip_bytes(review_members)
    resume_value = _zip_bytes(resume_members)
    _write_new(review_path, review_value)
    _write_new(resume_path, resume_value)
    return ValidationBundles(
        review_path=review_path,
        resume_path=resume_path,
        review_sha256=sha256(review_value).hexdigest(),
        resume_sha256=sha256(resume_value).hexdigest(),
    )


def _version(probe: Mapping[str, object], name: str) -> str:
    value = probe.get(name)
    if not isinstance(value, str) or not value:
        raise TabMValidationError(f"{name} version is missing")
    return value


def validate_exact_probe(probe: Mapping[str, object]) -> None:
    """Require the evaluator's documented Python and package versions."""

    if not isinstance(probe, Mapping) or probe.get("status") != "passed":
        raise TabMValidationError("exact compatibility probe did not pass")
    if _version(probe, "python") != "3.11.15":
        raise TabMValidationError("exact probe requires Python 3.11.15")
    if not _version(probe, "torch").startswith("2.7.1"):
        raise TabMValidationError("exact probe requires PyTorch 2.7.1")
    expected = {
        "pandas": "2.0.3",
        "numpy": "1.26.4",
        "tabm": "0.0.3",
        "rtdl_num_embeddings": "0.0.12",
    }
    for name, value in expected.items():
        if _version(probe, name) != value:
            raise TabMValidationError(f"exact probe requires {name} {value}")
    probabilities = probe.get("probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise TabMValidationError("exact probe probabilities are missing")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0
        or float(value) > 1
        for value in probabilities
    ):
        raise TabMValidationError("exact probe probabilities are invalid")


def compare_probe_predictions(
    exact: Mapping[str, object],
    gpu: Mapping[str, object],
    *,
    tolerance: float = 1e-6,
) -> None:
    validate_exact_probe(exact)
    if not isinstance(gpu, Mapping) or gpu.get("status") != "passed":
        raise TabMValidationError("GPU probe did not pass")
    left = np.asarray(exact["probabilities"], dtype="float64")
    right = np.asarray(gpu.get("probabilities"), dtype="float64")
    if left.shape != right.shape or not np.isfinite(right).all():
        raise TabMValidationError("probe predictions are not aligned")
    if float(np.max(np.abs(left - right))) > tolerance:
        raise TabMValidationError("exact and GPU probe predictions differ")


def build_capacity_frames(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    row_count: int = CAPACITY_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat official sample rows solely for evaluator capacity measurement."""

    if (
        not isinstance(test, pd.DataFrame)
        or not isinstance(sample, pd.DataFrame)
        or test.empty
        or "row_id" not in test
        or sample.columns.tolist() != ["row_id", "control_success"]
    ):
        raise TabMValidationError("capacity source frames are invalid")
    if type(row_count) is not int or row_count < 1:
        raise TabMValidationError("capacity row_count must be positive")
    positions = np.arange(row_count, dtype="int64") % len(test)
    scale_test = test.iloc[positions].reset_index(drop=True).copy()
    width = max(6, len(str(row_count - 1)))
    ids = [f"SCALE_{index:0{width}d}" for index in range(row_count)]
    scale_test["row_id"] = ids
    scale_sample = pd.DataFrame(
        {"row_id": ids, "control_success": np.full(row_count, 0.5)}
    )
    return scale_test, scale_sample


def _finite_number(report: Mapping[str, object], name: str) -> float:
    value = report.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TabMValidationError(f"validation report {name} is invalid")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise TabMValidationError(f"validation report {name} is invalid")
    return number


def validate_report(report: Mapping[str, object]) -> None:
    """Apply stricter-than-platform capacity thresholds."""

    if (
        not isinstance(report, Mapping)
        or report.get("schema_version") != 1
        or report.get("status") != "validation_passed"
    ):
        raise TabMValidationError("validation report status differs")
    if report.get("audit_scope") != AUDIT_SCOPE:
        raise TabMValidationError("validation report audit scope differs")
    identity = report.get("identity")
    if not isinstance(identity, dict) or not identity:
        raise TabMValidationError("validation report identity is missing")
    install = _finite_number(report, "install_seconds")
    inference = _finite_number(report, "inference_seconds")
    ram = _finite_number(report, "peak_ram_bytes")
    vram = _finite_number(report, "peak_vram_bytes")
    package = _finite_number(report, "package_bytes")
    extracted = _finite_number(report, "extracted_bytes")
    if install > 480:
        raise TabMValidationError("submitted requirements installation exceeds safety threshold")
    if inference > 480:
        raise TabMValidationError("runtime safety threshold exceeded")
    if ram >= 22 * 1024**3:
        raise TabMValidationError("peak RAM exceeds safety threshold")
    if vram >= 20 * 1024**3:
        raise TabMValidationError("peak VRAM exceeds safety threshold")
    if package > 10_000_000_000 or extracted > 32_000_000_000:
        raise TabMValidationError("package size projection exceeds official limit")


def _combined_sha256(files: Mapping[str, Path]) -> str:
    digest = sha256()
    for name, path in sorted(files.items()):
        value = path.read_bytes()
        digest.update(name.encode("utf-8") + b"\0" + sha256(value).digest())
    return digest.hexdigest()


def _load_runtime(path: Path) -> object:
    name = f"tabm_validation_runtime_{sha256(path.read_bytes()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TabMValidationError("cannot load rendered evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise TabMValidationError(f"required package is missing: {distribution}") from error


def _gpu_environment() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise TabMValidationError("GPU validation requires CUDA")
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "tabm": _package_version("tabm"),
        "rtdl_num_embeddings": _package_version("rtdl-num-embeddings"),
    }


_EXACT_PROBE_SOURCE = r'''
from pathlib import Path
import importlib.metadata
import importlib.util
import json
import platform
import sys
import numpy as np
import pandas as pd
import torch

runtime_path, model_path, test_path = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("exact_tabm_runtime", runtime_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
frame = pd.read_csv(test_path, dtype={"row_id": "string"})
predictor = module.load_frozen_predictor(model_path, device="cpu")
probabilities = predictor.predict_batch(frame, batch_size=2048).tolist()
payload = {
    "status": "passed",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "tabm": importlib.metadata.version("tabm"),
    "rtdl_num_embeddings": importlib.metadata.version("rtdl-num-embeddings"),
    "probabilities": probabilities,
}
print("EXACT_PROBE_JSON=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
'''


def _run_exact_probe(
    python: Path,
    runtime_path: Path,
    model_dir: Path,
    test_path: Path,
) -> tuple[dict[str, object], str]:
    if not python.is_file():
        raise TabMValidationError(f"exact-probe Python is missing: {python}")
    completed = subprocess.run(
        [
            str(python),
            "-c",
            _EXACT_PROBE_SOURCE,
            str(runtime_path),
            str(model_dir),
            str(test_path),
        ],
        text=True,
        capture_output=True,
        timeout=480,
        check=False,
    )
    log = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise TabMValidationError("exact compatibility probe failed")
    lines = [line for line in completed.stdout.splitlines() if line.startswith("EXACT_PROBE_JSON=")]
    if len(lines) != 1:
        raise TabMValidationError("exact compatibility probe output differs")
    try:
        probe = json.loads(lines[0].split("=", 1)[1])
    except json.JSONDecodeError as error:
        raise TabMValidationError("exact compatibility probe JSON is invalid") from error
    validate_exact_probe(probe)
    return probe, log


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _input_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    if any(path.is_symlink() or not path.is_file() for path in (test_path, sample_path)):
        raise TabMValidationError("official sample input is missing or unsafe")
    test = pd.read_csv(test_path, dtype={"row_id": "string"})
    sample = pd.read_csv(sample_path, dtype={"row_id": "string"})
    if len(test) != 5 or len(sample) != 5:
        raise TabMValidationError("local official sample must contain exactly five rows")
    if test["row_id"].isna().any() or test["row_id"].duplicated().any():
        raise TabMValidationError("official sample row_id is invalid")
    if sample.columns.tolist() != ["row_id", "control_success"]:
        raise TabMValidationError("official sample submission schema differs")
    if set(test["row_id"].astype(str)) != set(sample["row_id"].astype(str)):
        raise TabMValidationError("official sample IDs differ")
    return test, sample, test_path, sample_path


def _audit_identity(
    *,
    candidate: object,
    test_path: Path,
    sample_path: Path,
    runtime: bytes,
) -> AuditIdentity:
    project = Path(__file__).resolve().parents[1]
    policy = load_policy(project / "competition_rules/policy.json", project_root=project)
    source_files = {
        "submission/tabm_candidate.py": project / "submission/tabm_candidate.py",
        "submission/tabm_validation.py": project / "submission/tabm_validation.py",
        "competition_rules/evidence_gate.py": project / "competition_rules/evidence_gate.py",
    }
    adapter_source = project / "submission/tabm_version_d_script.py"
    config = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "audit_scope": AUDIT_SCOPE,
        "capacity_rows": CAPACITY_ROWS,
        "probability_tolerance": 1e-6,
    }
    data_digest = sha256()
    for name, path in (("test.csv", test_path), ("sample_submission.csv", sample_path)):
        value = path.read_bytes()
        data_digest.update(name.encode() + b"\0" + sha256(value).digest())
    return AuditIdentity(
        policy_version=str(policy["policy_version"]),
        candidate_id=CANDIDATE_ID,
        data_sha256=data_digest.hexdigest(),
        code_sha256=_combined_sha256(source_files),
        config_sha256=sha256(_canonical_json(config)).hexdigest(),
        preprocessing_sha256=str(candidate.member_sha256["preprocessing_state.json"]),
        model_sha256=str(candidate.model_sha256),
        adapter_sha256=sha256(adapter_source.read_bytes()).hexdigest(),
        runtime_sha256=sha256(runtime).hexdigest(),
    )


class _Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: object) -> None:
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        line = event if not detail else f"{event} {detail}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _copy_model(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in source.iterdir():
        shutil.copy2(path, destination / path.name)


def _size_projection(runtime: bytes, requirements: bytes, model_dir: Path) -> tuple[int, int]:
    members = {"script.py": runtime, "requirements.txt": requirements}
    for path in model_dir.iterdir():
        members[f"model/{path.name}"] = path.read_bytes()
    return len(_zip_bytes(members)), sum(len(value) for value in members.values())


def run_validation(args: argparse.Namespace) -> ValidationBundles:
    """Run compatibility and capacity checks; never create a submission archive."""

    output = args.output_dir.resolve()
    if output.exists():
        raise TabMValidationError(f"validation output already exists: {output}")
    evidence = output / "evidence"
    evidence.mkdir(parents=True)
    logger = _Logger(evidence / "validation.log")
    candidate = load_imported_candidate(args.candidate_root)
    test, sample, test_path, sample_path = _input_frames(args.data_dir)
    runtime_value = render_validation_script(candidate)
    runtime_dir = output / "runtime"
    runtime_dir.mkdir()
    runtime_path = runtime_dir / "script.py"
    runtime_path.write_bytes(runtime_value)
    identity = _audit_identity(
        candidate=candidate,
        test_path=test_path,
        sample_path=sample_path,
        runtime=runtime_value,
    )
    logger.write(
        "VALIDATION_CODE_READY",
        candidate=CANDIDATE_ID,
        runtime_sha256=identity.runtime_sha256,
    )
    logger.write(
        "VALIDATION_INPUTS_VERIFIED",
        rows=len(test),
        scope=AUDIT_SCOPE,
        model_sha256=candidate.model_sha256,
    )
    exact, exact_log = _run_exact_probe(
        args.compat_python.resolve(), runtime_path, candidate.model_dir, test_path
    )
    with (evidence / "exact_probe.log").open("x", encoding="utf-8") as handle:
        handle.write(exact_log)

    runtime = _load_runtime(runtime_path)
    gpu_environment = _gpu_environment()
    predictor = runtime.load_frozen_predictor(candidate.model_dir)
    gpu_probabilities = predictor.predict_batch(test, batch_size=2048).tolist()
    gpu_probe = {"status": "passed", **gpu_environment, "probabilities": gpu_probabilities}
    compare_probe_predictions(exact, gpu_probe)
    environment = {"exact_probe": exact, "gpu_host": gpu_probe}
    (evidence / "environment.json").write_bytes(_canonical_json(environment))
    logger.write(
        "VALIDATION_ENVIRONMENT_READY",
        exact_python=exact["python"],
        exact_torch=exact["torch"],
        host_python=gpu_environment["python"],
        host_torch=gpu_environment["torch"],
        gpu=gpu_environment["gpu"],
    )
    del predictor

    for phase in (
        "baseline",
        "reverse",
        "shuffle",
        "batch_257",
        "batch_2048",
        "singleton_canaries",
    ):
        logger.write("VALIDATION_PROGRESS", phase=phase, status="scheduled")
    audit = run_phased_independence_audit(
        frame=test,
        output_dir=evidence / "full_audit",
        identity=identity,
        load_predictor=lambda: runtime.load_frozen_predictor(candidate.model_dir),
        singleton_count=len(test),
    )
    if audit.status != "passed":
        raise TabMValidationError("row-independence audit is incomplete")
    logger.write("VALIDATION_PROGRESS", phase="row_independence", status="completed")

    scale_test, scale_sample = build_capacity_frames(test, sample)
    sandbox = output / "capacity_sandbox"
    (sandbox / "data").mkdir(parents=True)
    _copy_model(candidate.model_dir, sandbox / "model")
    (sandbox / "script.py").write_bytes(runtime_value)
    scale_test.to_csv(sandbox / "data/test.csv", index=False)
    scale_sample.to_csv(sandbox / "data/sample_submission.csv", index=False)
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    previous = Path.cwd()
    started = monotonic()
    try:
        os.chdir(sandbox)
        runtime.main()
    finally:
        os.chdir(previous)
    inference_seconds = monotonic() - started
    peak_vram = int(torch.cuda.max_memory_allocated())
    peak_ram = _peak_rss_bytes()
    result = pd.read_csv(
        sandbox / "output/submission.csv", dtype={"row_id": "string"}
    )
    if result.columns.tolist() != ["row_id", "control_success"] or len(result) != CAPACITY_ROWS:
        raise TabMValidationError("capacity output schema or row count differs")
    if result["row_id"].tolist() != scale_sample["row_id"].tolist():
        raise TabMValidationError("capacity output row order differs")
    values = result["control_success"].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise TabMValidationError("capacity output probabilities are invalid")
    output_sha256 = sha256((sandbox / "output/submission.csv").read_bytes()).hexdigest()
    logger.write(
        "VALIDATION_OUTPUT_VERIFIED",
        rows=len(result),
        output_sha256=output_sha256,
        elapsed_seconds=f"{inference_seconds:.3f}",
    )

    requirements_path = args.requirements.resolve()
    requirements = requirements_path.read_bytes()
    if requirements != b"tabm==0.0.3\nrtdl-num-embeddings==0.0.12\n":
        raise TabMValidationError("submitted requirements differ")
    install_log = args.install_log.resolve()
    if not install_log.is_file():
        raise TabMValidationError("install log is missing")
    shutil.copy2(install_log, evidence / "install.log")
    package_bytes, extracted_bytes = _size_projection(
        runtime_value, requirements, candidate.model_dir
    )
    report = {
        "schema_version": 1,
        "status": "validation_passed",
        "audit_scope": AUDIT_SCOPE,
        "identity": asdict(identity),
        "candidate_id": CANDIDATE_ID,
        "official_sample_rows": len(test),
        "capacity_rows": CAPACITY_ROWS,
        "install_seconds": float(args.install_seconds),
        "inference_seconds": inference_seconds,
        "peak_ram_bytes": peak_ram,
        "peak_vram_bytes": peak_vram,
        "package_bytes": package_bytes,
        "extracted_bytes": extracted_bytes,
        "output_sha256": output_sha256,
        "gpu_host_is_official_evaluator": False,
        "hidden_evaluation_rows_inspected": False,
    }
    validate_report(report)
    logger.write(
        "VALIDATION_GATES_PASSED",
        audit_scope=AUDIT_SCOPE,
        hidden_rows_inspected=False,
    )
    bundles = build_validation_bundles(evidence, report, output / "bundles")
    logger.write(
        "VALIDATION_SUCCESS",
        review=bundles.review_path,
        resume=bundles.resume_path,
    )
    return bundles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the frozen TabM evaluator")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compat-python", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--install-log", type=Path, required=True)
    parser.add_argument("--install-seconds", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_validation(args)
    except Exception as error:
        message = str(error).replace(" ", "_").replace("\n", "_")
        print(
            f"VALIDATION_ERROR stage=run type={type(error).__name__} message={message}",
            flush=True,
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
