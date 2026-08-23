"""Common temporal expert dispatch and fail-closed artifact publication."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.independent_dl.training import fit_candidate

from .catboost_training import (
    CATBOOST_PREFIXES,
    CatBoostResult,
    TemporalTrainingJob,
    run_catboost_job,
)
from .lupi_teacher import TeacherOOF, build_teacher_vector
from .tabm_training import TemporalTabMAdapter


class WorkerPublicationError(RuntimeError):
    """Raised when worker output is incomplete, unsafe, or unbound."""


FINAL_WORKER_STATUSES = (
    "completed",
    "budget_inconclusive",
    "failed",
    "rule_blocked",
)
_REQUIRED_ARTIFACTS = frozenset(
    {"metrics.json", "predictions.csv", "checkpoint_meta.json"}
)
_MANIFEST = "worker_result.json"


@dataclass(frozen=True)
class WorkerBackendDispatcher:
    """Inject independent fixture or production backends without importing GPU code."""

    catboost_backend: object | None = None
    tabm_backend: object | None = None
    fit_function: Callable[..., object] | None = None


def run_worker(
    job: TemporalTrainingJob,
    output_dir: str | Path,
    *,
    backend: object,
) -> Path:
    """Dispatch one validated job and publish completion only after full validation."""

    if type(job) is not TemporalTrainingJob:
        raise WorkerPublicationError("worker job has an invalid type")
    root = _safe_root(output_dir)
    if any(root.iterdir()):
        raise WorkerPublicationError("worker output directory must start empty")
    dispatcher = backend if type(backend) is WorkerBackendDispatcher else None

    if job.expert == "catboost":
        runtime = dispatcher.catboost_backend if dispatcher is not None else backend
        if runtime is None:
            raise WorkerPublicationError("CatBoost backend is missing")
        result = run_catboost_job(job, backend=runtime)
        predictions = result.prefix_predictions[max(CATBOOST_PREFIXES)]
        prediction_frame = _prediction_frame(job, predictions)
        for prefix, values in result.prefix_predictions.items():
            prediction_frame[f"probability_prefix_{prefix}"] = values
        metrics = {
            "schema_version": 1,
            "job_id": job.job_id,
            "expert": job.expert,
            "fixed_inference_prefix": max(CATBOOST_PREFIXES),
            "prefixes": list(CATBOOST_PREFIXES),
            "backend_calls": _plain_json(result.backend_calls),
            "valid_row_order_sha256": result.valid_row_order_sha256,
        }
        model_path = root / "model.cbm"
        _save_catboost_model(runtime, result, model_path)
        checkpoint_meta = {
            "schema_version": 1,
            "job_id": job.job_id,
            "training_identity_sha256": job.identity.sha256,
            "checkpoint": model_path.name,
            "iterations": max(CATBOOST_PREFIXES),
            "early_stopping": False,
        }
    else:
        runtime = dispatcher.tabm_backend if dispatcher is not None else backend
        fit_function = (
            dispatcher.fit_function
            if dispatcher is not None and dispatcher.fit_function is not None
            else fit_candidate
        )
        teacher_probability = None
        if job.expert == "lupi":
            teacher_probability = _verified_teacher_vector(job)
        adapter = TemporalTabMAdapter(
            sample_weight=np.array(job.sample_weight, copy=True),
            loss_name=str(job.identity.payload["loss"]),
            teacher_probability=teacher_probability,
            teacher_lambda=job.teacher_lambda,
        )
        trained = fit_function(job.train_request, adapter, root, backend=runtime)
        predictions = _finite_probabilities(
            getattr(trained, "predictions", None), job.valid_rows
        )
        prediction_frame = _prediction_frame(job, predictions)
        checkpoint = _bound_checkpoint(root, getattr(trained, "checkpoint", None))
        metrics = {
            "schema_version": 1,
            "job_id": job.job_id,
            "expert": job.expert,
            "best_epoch": int(getattr(trained, "best_epoch")),
            "best_brier": float(getattr(trained, "best_brier")),
            "teacher_oof_sha256": job.teacher_oof_sha256,
        }
        meta_path = root / "checkpoint_meta.json"
        if meta_path.exists() or meta_path.is_symlink():
            checkpoint_meta = _read_json(meta_path, "checkpoint metadata")
        else:
            checkpoint_meta = {"schema_version": 1}
        checkpoint_meta.update(
            {
                "job_id": job.job_id,
                "training_identity_sha256": job.identity.sha256,
                "checkpoint": checkpoint.name,
            }
        )

    _atomic_bytes(root / "predictions.csv", prediction_frame.to_csv(index=False).encode("utf-8"))
    _atomic_bytes(root / "metrics.json", _canonical_json(metrics))
    _atomic_bytes(root / "checkpoint_meta.json", _canonical_json(checkpoint_meta))
    artifacts = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() or path.is_symlink()
    )
    return publish_worker_result(root, job, artifacts, status="completed")


def publish_worker_result(
    root: str | Path,
    job: TemporalTrainingJob,
    artifacts: Sequence[Path],
    status: str,
) -> Path:
    """Atomically bind exactly every regular output file except the manifest itself."""

    directory = _safe_root(root)
    if type(job) is not TemporalTrainingJob:
        raise WorkerPublicationError("worker job has an invalid type")
    if type(status) is not str or status not in FINAL_WORKER_STATUSES:
        raise WorkerPublicationError("worker status is not an exact final status")
    manifest_path = directory / _MANIFEST
    if manifest_path.exists() or manifest_path.is_symlink():
        raise WorkerPublicationError("worker result already exists")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in artifacts:
        candidate = Path(raw)
        try:
            relative = candidate.relative_to(directory)
        except ValueError as error:
            raise WorkerPublicationError("artifact path escapes worker root") from error
        name = _normalized_relative(relative.as_posix())
        if name == _MANIFEST or name in seen:
            raise WorkerPublicationError("artifact path is duplicated or reserved")
        seen.add(name)
        metadata = _regular_file(candidate)
        records.append(
            {
                "path": name,
                "size_bytes": metadata.st_size,
                "sha256": _file_sha256(candidate),
            }
        )
    records.sort(key=lambda item: str(item["path"]))
    discovered = _discover_regular_paths(directory, exclude_manifest=True)
    if seen != discovered:
        raise WorkerPublicationError("artifact set leaves missing or unbound files")
    if status == "completed" and not _REQUIRED_ARTIFACTS.issubset(seen):
        raise WorkerPublicationError("completed worker lacks required artifacts")
    payload = {
        "schema_version": 1,
        "job_id": job.job_id,
        "status": status,
        "training_identity_sha256": job.identity.sha256,
        "artifacts": records,
    }
    _atomic_bytes(manifest_path, _canonical_json(payload))
    verify_worker_result(directory)
    return manifest_path


def verify_worker_result(root: str | Path) -> Mapping[str, object]:
    """Re-open and validate a closed worker result directory."""

    directory = _safe_root(root, create=False)
    manifest_path = directory / _MANIFEST
    _regular_file(manifest_path)
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerPublicationError("worker result JSON is invalid") from error
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "job_id",
        "status",
        "training_identity_sha256",
        "artifacts",
    }:
        raise WorkerPublicationError("worker result schema differs")
    if raw != _canonical_json(payload):
        raise WorkerPublicationError("worker result is not canonical JSON")
    if payload["schema_version"] != 1:
        raise WorkerPublicationError("worker result schema version differs")
    if type(payload["job_id"]) is not str or not payload["job_id"]:
        raise WorkerPublicationError("worker result job_id is invalid")
    if payload["status"] not in FINAL_WORKER_STATUSES:
        raise WorkerPublicationError("worker result status differs")
    if not _is_sha256(payload["training_identity_sha256"]):
        raise WorkerPublicationError("training identity SHA-256 is invalid")
    artifacts = payload["artifacts"]
    if type(artifacts) is not list:
        raise WorkerPublicationError("worker artifact list is invalid")
    seen: set[str] = set()
    for record in artifacts:
        if type(record) is not dict or set(record) != {"path", "size_bytes", "sha256"}:
            raise WorkerPublicationError("worker artifact record differs")
        name = _normalized_relative(record["path"])
        if name == _MANIFEST or name in seen:
            raise WorkerPublicationError("worker artifact path is duplicated")
        seen.add(name)
        candidate = directory.joinpath(*PurePosixPath(name).parts)
        metadata = _regular_file(candidate)
        if type(record["size_bytes"]) is not int or metadata.st_size != record["size_bytes"]:
            raise WorkerPublicationError(f"artifact size differs: {name}")
        if not _is_sha256(record["sha256"]) or _file_sha256(candidate) != record["sha256"]:
            raise WorkerPublicationError(f"artifact SHA-256 differs: {name}")
    if seen != _discover_regular_paths(directory, exclude_manifest=True):
        raise WorkerPublicationError("artifact set leaves missing or unbound files")
    if payload["status"] == "completed" and not _REQUIRED_ARTIFACTS.issubset(seen):
        raise WorkerPublicationError("completed worker lacks required artifacts")
    return MappingProxyType(payload)


def _verified_teacher_vector(job: TemporalTrainingJob) -> np.ndarray:
    teacher = job.teacher_oof
    if type(teacher) is not TeacherOOF:
        raise WorkerPublicationError("LUPI teacher OOF has an invalid sealed type")
    try:
        teacher.probability
        observed = teacher._sha256
    except (AttributeError, ValueError) as error:
        raise WorkerPublicationError("LUPI teacher OOF integrity validation failed") from error
    if observed != job.teacher_oof_sha256:
        raise WorkerPublicationError("LUPI teacher OOF hash differs")
    row_ids = tuple(
        value.item() if isinstance(value, np.generic) else value
        for value in job.train_request.train.row_id
    )
    return build_teacher_vector(row_ids, teacher)


def _prediction_frame(job: TemporalTrainingJob, probability: object) -> pd.DataFrame:
    values = _finite_probabilities(probability, job.valid_rows)
    frame = job.audit_frame.copy(deep=True)
    if tuple(frame["row_id"].tolist()) != job.valid_row_ids:
        raise WorkerPublicationError("prediction row order differs from source")
    frame.insert(2, "probability", values)
    return frame


def _finite_probabilities(value: object, rows: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype="float64")
    except (TypeError, ValueError, OverflowError) as error:
        raise WorkerPublicationError("worker probabilities must be numeric") from error
    if result.shape != (rows,):
        raise WorkerPublicationError("worker probability row count differs")
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise WorkerPublicationError("worker probabilities must be finite and in [0, 1]")
    return np.array(result, copy=True)


def _save_catboost_model(backend: object, result: CatBoostResult, path: Path) -> None:
    saver = getattr(backend, "save_model", None)
    backend_saver = callable(saver)
    if not callable(saver):
        saver = getattr(result.model, "save_model", None)
    if not callable(saver):
        raise WorkerPublicationError("CatBoost backend cannot save its model checkpoint")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        saver(result.model, temporary) if backend_saver else saver(temporary)
        _regular_file(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _bound_checkpoint(root: Path, value: object) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise WorkerPublicationError("TabM checkpoint path is invalid")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        path.relative_to(root)
    except ValueError as error:
        raise WorkerPublicationError("TabM checkpoint escapes worker root") from error
    _regular_file(path)
    return path


def _safe_root(value: str | Path, *, create: bool = True) -> Path:
    requested = Path(value)
    if requested.exists() and requested.is_symlink():
        raise WorkerPublicationError("worker root must not be a symlink")
    if create:
        requested.mkdir(parents=True, exist_ok=True)
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise WorkerPublicationError("worker root is missing or invalid") from error
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise WorkerPublicationError("worker root must be a real directory")
    return root


def _normalized_relative(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise WorkerPublicationError("artifact path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise WorkerPublicationError("artifact path is unsafe")
    if path.as_posix() != value:
        raise WorkerPublicationError("artifact path is not normalized")
    return value


def _regular_file(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise WorkerPublicationError(f"artifact is missing: {path.name}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise WorkerPublicationError(f"artifact must be a regular non-symlink file: {path.name}")
    if metadata.st_nlink != 1:
        raise WorkerPublicationError(f"artifact hard links are forbidden: {path.name}")
    return metadata


def _discover_regular_paths(root: Path, *, exclude_manifest: bool) -> set[str]:
    paths: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise WorkerPublicationError(f"symlink artifact is forbidden: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise WorkerPublicationError(f"non-regular or hard-linked artifact: {relative}")
        if not (exclude_manifest and relative == _MANIFEST):
            paths.add(_normalized_relative(relative))
    return paths


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            _plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise WorkerPublicationError("worker JSON value is invalid") from error


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain_json(item) for item in value]
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path, label: str) -> dict[str, object]:
    _regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerPublicationError(f"{label} is invalid") from error
    if type(value) is not dict:
        raise WorkerPublicationError(f"{label} must be an object")
    return value


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise WorkerPublicationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
