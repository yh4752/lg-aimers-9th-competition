from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping

import pandas as pd

from .contracts import E1Job, load_e1_contract
from .e2_contracts import E2Contract, E2Job, load_e2_contract
from .e2_inputs import file_sha256
from .inputs import PREDICTION_COLUMNS, VerifiedOfficialData
from .training import FoldResult, run_e1_job


class E2TrainingError(ValueError):
    """Raised when an E2 fold job or its restart evidence differs."""


FoldRunner = Callable[..., FoldResult]
_IDENTITY_KEYS = {
    "campaign_id",
    "contract_sha256",
    "code_sha256",
    "train_sha256",
    "history_sha256",
    "input_manifest_sha256",
    "job_id",
    "candidate_id",
    "train_end_year",
    "valid_year",
    "seed",
    "objective",
    "use_trackman",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _code_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    names = (
        "experiments/tree_expert/e2_training.py",
        "experiments/tree_expert/training.py",
        "experiments/tree_expert/features.py",
        "experiments/tree_expert/failure_labels.py",
        "experiments/temporal_portfolio/seasonal_features.py",
        "experiments/temporal_portfolio/trackman_pitcher.py",
    )
    digest = sha256()
    for name in names:
        path = root / name
        if not path.is_file():
            raise E2TrainingError(f"training source is absent: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def expected_job_identity(
    job: E2Job,
    data: VerifiedOfficialData,
    input_manifest_sha256: str,
    *,
    contract: E2Contract | None = None,
) -> dict[str, object]:
    active = load_e2_contract() if contract is None else contract
    if not _valid_sha(input_manifest_sha256):
        raise E2TrainingError("input manifest SHA-256 is invalid")
    identity: dict[str, object] = {
        "campaign_id": active.campaign_id,
        "contract_sha256": file_sha256(active.source_path),
        "code_sha256": _code_sha256(),
        "train_sha256": data.train_sha256,
        "history_sha256": data.history_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "job_id": job.job_id,
        "candidate_id": job.candidate_id,
        "train_end_year": job.train_end_year,
        "valid_year": job.valid_year,
        "seed": job.seed,
        "objective": job.objective,
        "use_trackman": job.use_trackman,
    }
    if set(identity) != _IDENTITY_KEYS:
        raise E2TrainingError("job identity keys differ")
    for key in (
        "contract_sha256",
        "code_sha256",
        "train_sha256",
        "history_sha256",
        "input_manifest_sha256",
    ):
        if not _valid_sha(identity[key]):
            raise E2TrainingError(f"job identity SHA-256 is invalid: {key}")
    return identity


def _mapped_e1_job(job: E2Job) -> E1Job:
    return E1Job(
        job_id=job.job_id,
        candidate_id=job.candidate_id,
        train_end_year=job.train_end_year,
        valid_year=job.valid_year,
        seed=job.seed,
        objective=job.objective,
        use_trackman=job.use_trackman,
        use_failure_labels=False,
    )


def _e1_training_contract(job: E2Job, contract: E2Contract):
    e1 = load_e1_contract()
    if dict(e1.catboost_parameters) != dict(contract.catboost):
        raise E2TrainingError("E1 and E2 CatBoost parameters differ")
    return replace(
        e1,
        fold=(job.train_end_year, job.valid_year),
        seed=job.seed,
        catboost_parameters=contract.catboost,
    )


def _artifact_evidence(path: Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
        raise E2TrainingError(f"completed artifact differs: {source.name}")
    return {"size": source.stat().st_size, "sha256": file_sha256(source)}


def _validate_predictions(path: Path) -> None:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise E2TrainingError(f"cannot read fold predictions: {error}") from error
    if (
        tuple(frame.columns) != PREDICTION_COLUMNS
        or frame.empty
        or frame["row_id"].isna().any()
        or not frame["row_id"].is_unique
    ):
        raise E2TrainingError("fold prediction schema or row identity differs")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if probability.isna().any() or not probability.between(0, 1).all():
        raise E2TrainingError("fold prediction probabilities differ")


def _worker_payload(result: FoldResult) -> dict[str, object]:
    return {
        "job_id": result.job_id,
        "candidate_id": result.candidate_id,
        "status": result.status,
        "brier": result.brier,
        "model": result.model_path.name if result.model_path else None,
        "predictions": result.predictions_path.name if result.predictions_path else None,
        "snapshot": result.snapshot_path.name if result.snapshot_path else None,
        "failure": result.failure,
    }


def run_e2_fold_job(
    *,
    job: E2Job,
    data: VerifiedOfficialData,
    baseline: pd.DataFrame,
    output_dir: Path,
    absolute_deadline: float,
    gpu_id: int,
    input_manifest_sha256: str,
    contract: E2Contract | None = None,
    fold_runner: FoldRunner = run_e1_job,
) -> FoldResult:
    active = load_e2_contract() if contract is None else contract
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = expected_job_identity(
        job,
        data,
        input_manifest_sha256,
        contract=active,
    )
    result = fold_runner(
        job=_mapped_e1_job(job),
        data=data,
        baseline=baseline,
        output_dir=output,
        absolute_deadline=absolute_deadline,
        gpu_id=gpu_id,
        contract=_e1_training_contract(job, active),
    )
    if result.job_id != job.job_id or result.candidate_id != job.candidate_id:
        raise E2TrainingError("fold result identity differs")
    artifacts: dict[str, dict[str, object]] = {}
    if result.status == "completed":
        if result.model_path is None or result.predictions_path is None:
            raise E2TrainingError("completed fold result has no artifacts")
        if (
            Path(result.model_path).resolve() != (output / "model.cbm").resolve()
            or Path(result.predictions_path).resolve()
            != (output / "predictions.csv").resolve()
        ):
            raise E2TrainingError("completed fold artifact paths differ")
        _validate_predictions(result.predictions_path)
        artifacts = {
            "model.cbm": _artifact_evidence(result.model_path),
            "predictions.csv": _artifact_evidence(result.predictions_path),
        }
    binding = {
        "schema_version": 1,
        "identity": identity,
        "artifacts": artifacts,
    }
    _atomic_bytes(output / "e2_job_binding.json", _canonical_json(binding))
    _atomic_bytes(output / "worker_result.json", _canonical_json(_worker_payload(result)))
    return result


def reusable_completed_job(
    output_dir: Path,
    job: E2Job,
    expected_identity: Mapping[str, object],
) -> FoldResult | None:
    output = Path(output_dir)
    binding_path = output / "e2_job_binding.json"
    worker_path = output / "worker_result.json"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (binding_path, worker_path)
    ):
        return None
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        type(binding) is not dict
        or set(binding) != {"schema_version", "identity", "artifacts"}
        or binding["schema_version"] != 1
        or binding["identity"] != dict(expected_identity)
        or set(binding["identity"]) != _IDENTITY_KEYS
        or type(binding["artifacts"]) is not dict
        or set(binding["artifacts"]) != {"model.cbm", "predictions.csv"}
    ):
        return None
    if (
        type(worker) is not dict
        or worker.get("job_id") != job.job_id
        or worker.get("candidate_id") != job.candidate_id
        or worker.get("status") != "completed"
        or worker.get("model") != "model.cbm"
        or worker.get("predictions") != "predictions.csv"
    ):
        return None
    for name, evidence in binding["artifacts"].items():
        path = output / name
        if (
            type(evidence) is not dict
            or set(evidence) != {"size", "sha256"}
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != evidence["size"]
            or file_sha256(path) != evidence["sha256"]
        ):
            return None
    try:
        _validate_predictions(output / "predictions.csv")
        brier = float(worker["brier"])
    except (E2TrainingError, TypeError, ValueError, KeyError):
        return None
    if not math.isfinite(brier):
        return None
    snapshot = worker.get("snapshot")
    snapshot_path = output / snapshot if isinstance(snapshot, str) else None
    return FoldResult(
        job_id=job.job_id,
        candidate_id=job.candidate_id,
        status="completed",
        brier=brier,
        model_path=output / "model.cbm",
        predictions_path=output / "predictions.csv",
        snapshot_path=snapshot_path if snapshot_path and snapshot_path.is_file() else None,
        failure=None,
    )
