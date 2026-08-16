"""Resumable, review-only orchestration for the sealed Stage P proxy."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping
from zipfile import ZipFile

from .artifacts import (
    ArtifactError,
    BundlePaths,
    StageEvidence,
    _validate_archive_entries,
    verify_resume_bundle,
    write_stage_bundles,
)
from .row_feature_contracts import (
    DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
    RowFeatureProxyContract,
    _read_contract_bytes,
    load_row_feature_proxy_contract,
    row_feature_contract_sha256,
)
from .row_feature_decisions import (
    ProxyDecision,
    ProxyMetric,
    decide_proxy_survivors,
    proxy_decision_json,
)
from .runner import CampaignJob, CampaignJobResult, CampaignRuntime
from .worker import (
    _job_payload,
    _job_sha,
    _result_from_payload as _worker_result_from_payload,
    _training_source_sha256,
    _valid_completed_result,
)


class RowFeatureProxyError(RuntimeError):
    """Raised when Stage P cannot advance from trusted evidence."""


@dataclass(frozen=True)
class RowFeatureProxyRun:
    version: str
    bundles: BundlePaths
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    inconclusive: tuple[str, ...]
    decision: ProxyDecision
    state_path: Path


_VERSION = "P"
_STATE_SCHEMA_VERSION = 1
_CONFIG_MEMBER = "config/row_feature_proxy_v1.json"
_METRICS_MEMBER = "metrics/job_results.json"
_DECISION_MEMBER = "decisions/proxy_decision.json"
_LOG_MEMBER = "logs/stage.log"
_STATE_MEMBER = "state/stage_state.json"
_BASE_MEMBERS = {_CONFIG_MEMBER, _METRICS_MEMBER, _LOG_MEMBER, _STATE_MEMBER}
_ALLOWED_JOB_ARTIFACTS = {
    "job.json",
    "worker_result.json",
    "worker.log",
    "predictions.csv",
    "checkpoint.pt",
    "checkpoint_meta.json",
    "best_checkpoint.pt",
}
_CHECKPOINT_NAMES = {"checkpoint.pt", "best_checkpoint.pt"}
_STATUSES = {"completed", "failed", "inconclusive"}
_DISPOSITIONS = {"completed", "failed", "inconclusive", "not_started"}
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_GENERATED_CELL_PREFIXES = ("COLAB_", "KAGGLE_")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _official_train_file_sha256(path: Path) -> str:
    """Separate seam for tiny fixture tests; production hashes the whole file."""

    return _file_sha256(path)


def _regular_file(path: Path) -> bool:
    try:
        return not path.is_symlink() and stat.S_ISREG(path.stat().st_mode)
    except OSError:
        return False


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _find_official_train(data_dir: Path) -> Path:
    try:
        lexical = Path(os.path.abspath(os.fspath(data_dir)))
    except (OSError, TypeError, ValueError) as error:
        raise RowFeatureProxyError("data_dir path is invalid") from error
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise RowFeatureProxyError(
                "data_dir path must not contain a symlink ancestor"
            )
    data_dir = lexical
    if not data_dir.is_dir():
        raise RowFeatureProxyError("data_dir must be a regular directory, not a symlink")
    candidates: list[Path] = []
    for root, directories, files in os.walk(data_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        directories[:] = sorted(
            name for name in directories if not (root_path / name).is_symlink()
        )
        if "train.csv" in files:
            candidate = root_path / "train.csv"
            if _regular_file(candidate):
                candidates.append(candidate)
    if len(candidates) != 1:
        raise RowFeatureProxyError(
            f"data_dir must contain exactly one regular non-symlink train.csv; found={len(candidates)}"
        )
    return candidates[0]


def _code_file_paths(experiments_root: Path | None = None) -> tuple[Path, ...]:
    """Return the conservative Python source closure for Stage P execution."""

    root = (
        Path(__file__).resolve().parents[1]
        if experiments_root is None
        else Path(experiments_root).resolve()
    )
    paths = list((root / "independent_dl").rglob("*.py"))
    paths.extend(
        path
        for path in (root / "tabm_campaign").rglob("*.py")
        if not path.name.startswith(_GENERATED_CELL_PREFIXES)
    )
    result = tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))
    if not result:
        raise RowFeatureProxyError("Stage P code source closure is empty")
    return result


def _code_sha256(experiments_root: Path | None = None) -> str:
    root = (
        Path(__file__).resolve().parents[1]
        if experiments_root is None
        else Path(experiments_root).resolve()
    )
    digest = sha256()
    for path in _code_file_paths(experiments_root):
        if not _regular_file(path):
            raise RowFeatureProxyError(
                f"bound source file is not regular: {path.relative_to(root)}"
            )
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_proxy_jobs(contract: RowFeatureProxyContract) -> tuple[CampaignJob, ...]:
    """Build the sealed baseline-first, same-seed paired Stage P job grid."""

    if not isinstance(contract, RowFeatureProxyContract):
        raise RowFeatureProxyError("contract must be a RowFeatureProxyContract")
    baseline = contract.baseline
    jobs: list[CampaignJob] = []
    for bundle in (None, *contract.feature_bundles):
        label = "baseline" if bundle is None else bundle
        for seed in contract.seeds:
            jobs.append(
                CampaignJob(
                    candidate_id=f"rfp__{label}__s{seed}",
                    capacity=baseline.capacity,
                    k=baseline.k,
                    width=baseline.width,
                    blocks=baseline.blocks,
                    dropout=baseline.dropout,
                    num_embedding=baseline.num_embedding,
                    loss=baseline.loss,
                    scheduler=baseline.scheduler,
                    learning_rate=baseline.learning_rate,
                    seed=seed,
                    train_end_year=contract.fold.train_end_year,
                    valid_year=contract.fold.valid_year,
                    sample_mode=contract.sample.mode,
                    max_epochs=contract.training.max_epochs,
                    min_epochs=contract.training.min_epochs,
                    patience=contract.training.patience,
                    feature_bundle=bundle,
                )
            )
    return tuple(jobs)


def _job_grid(jobs: tuple[CampaignJob, ...]) -> list[dict[str, str]]:
    return [
        {"candidate_id": job.candidate_id, "job_sha256": _job_sha(job)}
        for job in jobs
    ]


def _json_resource(value: object, label: str = "resource_evidence") -> object:
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str) and PurePosixPath(value).is_absolute():
            raise RowFeatureProxyError(f"{label} must not contain an absolute path")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RowFeatureProxyError(f"{label} must contain finite values")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RowFeatureProxyError(f"{label} keys must be strings")
            result[key] = _json_resource(item, f"{label}.{key}")
        return result
    if type(value) in {list, tuple}:
        return [_json_resource(item, label) for item in value]
    raise RowFeatureProxyError(f"{label} contains a non-JSON value")


def _artifact_entry(job_dir: Path, path: Path) -> dict[str, str]:
    if path.parent.resolve() != job_dir.resolve():
        raise RowFeatureProxyError("result artifact must be inside its current job directory")
    if path.name not in _ALLOWED_JOB_ARTIFACTS or not _regular_file(path):
        raise RowFeatureProxyError(f"result artifact is not an allowlisted regular file: {path.name}")
    return {
        "path": f"jobs/{job_dir.name}/{path.name}",
        "sha256": _file_sha256(path),
    }


def _collect_job_artifacts(job_dir: Path) -> dict[str, dict[str, str]]:
    if not job_dir.exists():
        return {}
    if job_dir.is_symlink() or not job_dir.is_dir():
        raise RowFeatureProxyError("job artifact directory must be a regular directory")
    artifacts: dict[str, dict[str, str]] = {}
    for path in sorted(job_dir.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise RowFeatureProxyError(f"job artifact must not be a symlink: {path.name}")
        if path.is_dir():
            raise RowFeatureProxyError(f"unexpected job artifact directory: {path.name}")
        if path.name not in _ALLOWED_JOB_ARTIFACTS:
            raise RowFeatureProxyError(f"unexpected job artifact name: {path.name}")
        if not _regular_file(path):
            raise RowFeatureProxyError(f"job artifact must be a regular file: {path.name}")
        artifacts[path.name] = _artifact_entry(job_dir, path)
    return artifacts


def _read_json_file(path: Path, label: str) -> dict[str, object]:
    if not _regular_file(path):
        raise RowFeatureProxyError(f"{label} must be a regular file")
    value = _read_json_bytes(path.read_bytes(), label)
    if type(value) is not dict:
        raise RowFeatureProxyError(f"{label} must contain an object")
    return value


def _same_worker_result(
    archived: CampaignJobResult,
    returned: CampaignJobResult,
) -> bool:
    checkpoint_name = (
        None if archived.checkpoint is None else archived.checkpoint.name
    )
    returned_checkpoint_name = (
        None if returned.checkpoint is None else returned.checkpoint.name
    )
    prediction_name = (
        None if archived.predictions_path is None else archived.predictions_path.name
    )
    returned_prediction_name = (
        None if returned.predictions_path is None else returned.predictions_path.name
    )
    return (
        archived.candidate_id == returned.candidate_id
        and archived.status == returned.status
        and archived.brier == returned.brier
        and archived.best_epoch == returned.best_epoch
        and archived.completed_epochs == returned.completed_epochs
        and checkpoint_name == returned_checkpoint_name
        and prediction_name == returned_prediction_name
        and _json_resource(dict(archived.resource_evidence))
        == _json_resource(dict(returned.resource_evidence))
        and archived.failure == returned.failure
    )


def _validate_checkpoint_meta(
    job: CampaignJob,
    result: CampaignJobResult,
    job_dir: Path,
) -> None:
    meta = _read_json_file(job_dir / "checkpoint_meta.json", "checkpoint metadata")
    expected_keys = {
        "candidate_id",
        "epoch",
        "checkpoint",
        "adapter_state",
        "checkpoint_binding",
    }
    if set(meta) != expected_keys:
        raise RowFeatureProxyError("checkpoint metadata keys are invalid")
    epoch = meta["epoch"]
    if (
        meta["candidate_id"] != job.candidate_id
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
        or result.completed_epochs != epoch + 1
        or meta["checkpoint"] != "checkpoint.pt"
        or (
            meta["adapter_state"] is not None
            and type(meta["adapter_state"]) is not dict
        )
    ):
        raise RowFeatureProxyError(
            "checkpoint metadata job, model, or epoch binding is invalid"
        )
    binding = meta["checkpoint_binding"]
    cache_digest = result.resource_evidence.get("cache_digest")
    deadline_fallback = (
        result.status == "inconclusive"
        and result.failure == "worker_exceeded_deadline_grace"
        and dict(result.resource_evidence) == {}
    )
    if cache_digest is None and deadline_fallback and isinstance(binding, dict):
        cache_digest = binding.get("cache_sha256")
    expected_binding = {
        "config_sha256": _job_sha(job),
        "cache_sha256": cache_digest,
        "training_source_sha256": _training_source_sha256(),
    }
    if (
        type(cache_digest) is not str
        or _SHA_RE.fullmatch(cache_digest) is None
        or binding != expected_binding
    ):
        raise RowFeatureProxyError(
            "checkpoint metadata config, cache, optimizer, scheduler, or code binding is invalid"
        )
    if not _regular_file(job_dir / "checkpoint.pt"):
        raise RowFeatureProxyError("checkpoint metadata references a missing checkpoint")
    _validate_checkpoint_payload(
        job,
        result,
        job_dir / "checkpoint.pt",
        meta_epoch=epoch,
        meta_adapter_state=meta["adapter_state"],
    )


def _validate_checkpoint_payload(
    job: CampaignJob,
    result: CampaignJobResult,
    path: Path,
    *,
    meta_epoch: int,
    meta_adapter_state: object,
) -> None:
    """Safely validate the restart state written by independent_dl.training."""

    try:
        import numpy as np
        import torch

        safe_types = [
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint32)),
        ]
        safe_globals = torch.serialization.safe_globals
        with safe_globals(safe_types):
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
    except Exception as error:
        raise RowFeatureProxyError(
            f"checkpoint payload cannot be loaded safely: {error}"
        ) from error
    required = {
        "candidate_id",
        "epoch",
        "best_epoch",
        "best_brier",
        "validation_curve",
        "validation_time_curve",
        "elapsed_seconds",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "adapter_state",
    }
    if type(payload) is not dict or set(payload) != required:
        raise RowFeatureProxyError("checkpoint payload keys are invalid")
    epoch = payload["epoch"]
    best_epoch = payload["best_epoch"]
    best_brier = payload["best_brier"]
    elapsed = payload["elapsed_seconds"]
    if (
        payload["candidate_id"] != job.candidate_id
        or type(epoch) is not int
        or epoch != meta_epoch
        or type(best_epoch) is not int
        or not 0 <= best_epoch <= epoch
        or (
            result.best_epoch is not None
            and result.best_epoch != best_epoch
        )
        or type(best_brier) is not float
        or not math.isfinite(best_brier)
        or type(elapsed) is not float
        or not math.isfinite(elapsed)
        or elapsed < 0.0
    ):
        raise RowFeatureProxyError(
            "checkpoint payload candidate, epoch, best epoch, or metric is invalid"
        )
    validation_curve = payload["validation_curve"]
    time_curve = payload["validation_time_curve"]
    curves_valid = (
        type(validation_curve) is list
        and len(validation_curve) == epoch + 1
        and type(time_curve) is list
        and len(time_curve) == epoch + 1
    )
    if curves_valid:
        for expected_epoch, (metric_item, time_item) in enumerate(
            zip(validation_curve, time_curve, strict=True)
        ):
            if (
                type(metric_item) not in {list, tuple}
                or len(metric_item) != 2
                or type(metric_item[0]) is not int
                or metric_item[0] != expected_epoch
                or type(metric_item[1]) is not float
                or not math.isfinite(metric_item[1])
                or not 0.0 <= metric_item[1] <= 1.0
                or type(time_item) not in {list, tuple}
                or len(time_item) != 3
                or type(time_item[0]) is not int
                or time_item[0] != expected_epoch
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in time_item[1:]
                )
                or time_item[1] < 0.0
                or time_item[2] != metric_item[1]
            ):
                curves_valid = False
                break
    if not curves_valid:
        raise RowFeatureProxyError("checkpoint validation curves are invalid")
    expected_best_epoch, expected_best_brier = min(
        validation_curve, key=lambda item: item[1]
    )
    if (
        best_epoch != expected_best_epoch
        or not math.isclose(best_brier, expected_best_brier, rel_tol=0.0, abs_tol=1e-15)
        or any(
            current[1] < previous[1]
            for previous, current in zip(time_curve, time_curve[1:])
        )
        or not math.isclose(elapsed, time_curve[-1][1], rel_tol=0.0, abs_tol=1e-9)
    ):
        raise RowFeatureProxyError("checkpoint best metric does not match its curve")
    if (
        not isinstance(payload["model"], Mapping)
        or not payload["model"]
        or not isinstance(payload["optimizer"], Mapping)
        or not {"state", "param_groups"}.issubset(payload["optimizer"])
        or not isinstance(payload["scheduler"], Mapping)
        or not isinstance(payload["scaler"], Mapping)
        or type(payload["python_rng"]) is not tuple
        or type(payload["numpy_rng"]) is not tuple
        or not isinstance(payload["torch_rng"], torch.Tensor)
        or type(payload["cuda_rng"]) is not list
        or payload["adapter_state"] != meta_adapter_state
    ):
        raise RowFeatureProxyError(
            "checkpoint model, optimizer, scheduler, RNG, or adapter state is invalid"
        )


def _validate_semantic_evidence(
    job: CampaignJob,
    result: CampaignJobResult,
    job_dir: Path,
) -> None:
    try:
        job_payload = _read_json_file(job_dir / "job.json", "job evidence")
        if job_payload != _job_payload(job):
            raise RowFeatureProxyError("job evidence differs from the scheduled job")

        worker_payload = _read_json_file(
            job_dir / "worker_result.json", "worker result evidence"
        )
        if worker_payload.get("job_sha256") != _job_sha(job):
            raise RowFeatureProxyError("worker result job SHA-256 differs")
        parsed = _worker_result_from_payload(worker_payload)
        if not _same_worker_result(parsed, result):
            raise RowFeatureProxyError(
                "worker result identity, status, metric, or resource evidence differs"
            )

        if result.checkpoint is not None or result.completed_epochs > 0:
            _validate_checkpoint_meta(job, result, job_dir)

        has_prediction_evidence = (
            result.brier is not None
            and result.checkpoint is not None
            and result.predictions_path is not None
        )
        if has_prediction_evidence and not _valid_completed_result(
            job_dir, job, result
        ):
            raise RowFeatureProxyError(
                "prediction schema, recomputed Brier, checkpoint, code, or artifact binding differs"
            )
        if result.status == "completed" and not has_prediction_evidence:
            raise RowFeatureProxyError("completed result evidence is incomplete")
    except (RowFeatureProxyError, KeyError, OSError, TypeError, ValueError) as error:
        raise RowFeatureProxyError(
            f"completed evidence is untrusted for {job.candidate_id}: {error}"
        ) from error


def _normalize_worker_result(
    job: CampaignJob,
    result: CampaignJobResult,
    job_dir: Path,
) -> None:
    """Remove machine-specific absolute paths after semantic validation."""

    payload = {
        "job_sha256": _job_sha(job),
        "candidate_id": result.candidate_id,
        "status": result.status,
        "brier": result.brier,
        "best_epoch": result.best_epoch,
        "completed_epochs": result.completed_epochs,
        "checkpoint": (
            None if result.checkpoint is None else result.checkpoint.name
        ),
        "predictions_path": (
            None
            if result.status != "completed" or result.predictions_path is None
            else result.predictions_path.name
        ),
        "resource_evidence": dict(result.resource_evidence),
        "failure": result.failure,
    }
    _atomic_bytes(job_dir / "worker_result.json", _canonical_json(payload))


def _validate_runtime_result(
    job: CampaignJob,
    result: CampaignJobResult,
    job_dir: Path,
) -> dict[str, object]:
    if not isinstance(result, CampaignJobResult):
        raise RowFeatureProxyError("runtime returned a foreign result type")
    if result.candidate_id != job.candidate_id:
        raise RowFeatureProxyError("runtime result candidate identity differs from its job")
    if result.status not in _STATUSES:
        raise RowFeatureProxyError(f"runtime result status is invalid: {result.status}")
    if (
        isinstance(result.completed_epochs, bool)
        or not isinstance(result.completed_epochs, int)
        or result.completed_epochs < 0
    ):
        raise RowFeatureProxyError("completed_epochs must be a nonnegative integer")
    if result.best_epoch is not None and (
        isinstance(result.best_epoch, bool)
        or not isinstance(result.best_epoch, int)
        or result.best_epoch < 0
    ):
        raise RowFeatureProxyError("best_epoch must be a nonnegative integer or None")
    if result.failure is not None and not isinstance(result.failure, str):
        raise RowFeatureProxyError("result failure must be a string or None")
    if result.brier is not None and (
        type(result.brier) is not float
        or not math.isfinite(result.brier)
        or not 0.0 <= result.brier <= 1.0
    ):
        raise RowFeatureProxyError("result brier must be a finite float in [0, 1] or None")
    if result.checkpoint is not None:
        if result.checkpoint.name not in _CHECKPOINT_NAMES:
            raise RowFeatureProxyError("result checkpoint name is not allowlisted")
        _artifact_entry(job_dir, result.checkpoint)
    if result.status == "completed":
        if result.brier is None or result.checkpoint is None or result.predictions_path is None:
            raise RowFeatureProxyError("completed result requires brier, checkpoint, and predictions")
        if result.predictions_path.name != "predictions.csv":
            raise RowFeatureProxyError("completed predictions name must be predictions.csv")
        _artifact_entry(job_dir, result.predictions_path)
    _validate_semantic_evidence(job, result, job_dir)
    _normalize_worker_result(job, result, job_dir)
    artifacts = _collect_job_artifacts(job_dir)
    if result.status != "completed":
        artifacts.pop("predictions.csv", None)
    if result.status == "completed":
        assert result.checkpoint is not None and result.predictions_path is not None
        if result.checkpoint.name not in artifacts or "predictions.csv" not in artifacts:
            raise RowFeatureProxyError("completed result artifacts are missing")
    return {
        "candidate_id": job.candidate_id,
        "feature_bundle": job.feature_bundle,
        "seed": job.seed,
        "job_sha256": _job_sha(job),
        "status": result.status,
        "disposition": result.status,
        "brier": result.brier,
        "best_epoch": result.best_epoch,
        "completed_epochs": result.completed_epochs,
        "checkpoint_name": None if result.checkpoint is None else result.checkpoint.name,
        "predictions_name": (
            result.predictions_path.name if result.status == "completed" else None
        ),
        "resource_evidence": _json_resource(dict(result.resource_evidence)),
        "failure": result.failure,
        "artifacts": artifacts,
    }


def _pending_row(job: CampaignJob) -> dict[str, object]:
    return {
        "candidate_id": job.candidate_id,
        "feature_bundle": job.feature_bundle,
        "seed": job.seed,
        "job_sha256": _job_sha(job),
        "status": "inconclusive",
        "disposition": "not_started",
        "brier": None,
        "best_epoch": None,
        "completed_epochs": 0,
        "checkpoint_name": None,
        "predictions_name": None,
        "resource_evidence": {},
        "failure": "not_started",
        "artifacts": {},
    }


def _state_payload(
    *,
    contract_sha: str,
    train_sha: str,
    code_sha: str,
    jobs: tuple[CampaignJob, ...],
    rows: Mapping[str, dict[str, object]],
    prior_manifest_sha: str | None,
    decision: ProxyDecision,
) -> dict[str, object]:
    ordered = [rows.get(job.candidate_id, _pending_row(job)) for job in jobs]
    terminal = decision.status in {"complete", "blocked"}
    return {
        "schema_version": _STATE_SCHEMA_VERSION,
        "version": _VERSION,
        "stage_complete": terminal,
        "campaign_config_sha256": contract_sha,
        "official_train_sha256": train_sha,
        "code_sha256": code_sha,
        "prior_manifest_sha256": prior_manifest_sha,
        "jobs": _job_grid(jobs),
        "results": ordered,
    }


def _write_local_state(state_path: Path, state: dict[str, object]) -> None:
    _atomic_bytes(state_path, _canonical_json(state))


def _parse_sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise RowFeatureProxyError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_state_header(
    state: object,
    *,
    contract_sha: str,
    train_sha: str,
    code_sha: str,
    jobs: tuple[CampaignJob, ...],
) -> dict[str, object]:
    if type(state) is not dict:
        raise RowFeatureProxyError("stage state root must be an object")
    expected_keys = {
        "schema_version",
        "version",
        "stage_complete",
        "campaign_config_sha256",
        "official_train_sha256",
        "code_sha256",
        "prior_manifest_sha256",
        "jobs",
        "results",
    }
    if set(state) != expected_keys:
        raise RowFeatureProxyError("stage state keys are invalid")
    if state["schema_version"] != _STATE_SCHEMA_VERSION or state["version"] != _VERSION:
        raise RowFeatureProxyError("stage state version is invalid")
    if type(state["stage_complete"]) is not bool:
        raise RowFeatureProxyError("stage_complete must be boolean")
    bindings = {
        "campaign_config_sha256": contract_sha,
        "official_train_sha256": train_sha,
        "code_sha256": code_sha,
    }
    for label, expected in bindings.items():
        if _parse_sha(state[label], label) != expected:
            raise RowFeatureProxyError(f"stage state {label} binding differs")
    prior = state["prior_manifest_sha256"]
    if prior is not None:
        _parse_sha(prior, "prior_manifest_sha256")
    if state["jobs"] != _job_grid(jobs):
        raise RowFeatureProxyError("stage state job hash grid differs")
    if type(state["results"]) is not list or len(state["results"]) != len(jobs):
        raise RowFeatureProxyError("stage state must contain one result per job")
    return state


def _validate_artifact_mapping(
    value: object,
    *,
    candidate_id: str,
) -> dict[str, dict[str, str]]:
    if type(value) is not dict:
        raise RowFeatureProxyError("result artifacts must be an object")
    result: dict[str, dict[str, str]] = {}
    for name, binding in value.items():
        if type(name) is not str or name not in _ALLOWED_JOB_ARTIFACTS:
            raise RowFeatureProxyError("state contains an unexpected artifact name")
        if type(binding) is not dict or set(binding) != {"path", "sha256"}:
            raise RowFeatureProxyError("artifact binding keys are invalid")
        expected_path = f"jobs/{candidate_id}/{name}"
        if binding["path"] != expected_path:
            raise RowFeatureProxyError("artifact path binding is invalid")
        digest = _parse_sha(binding["sha256"], "artifact sha256")
        result[name] = {"path": expected_path, "sha256": digest}
    return result


def _validated_rows(
    state: dict[str, object],
    jobs: tuple[CampaignJob, ...],
) -> dict[str, dict[str, object]]:
    expected_keys = {
        "candidate_id",
        "feature_bundle",
        "seed",
        "job_sha256",
        "status",
        "disposition",
        "brier",
        "best_epoch",
        "completed_epochs",
        "checkpoint_name",
        "predictions_name",
        "resource_evidence",
        "failure",
        "artifacts",
    }
    parsed: dict[str, dict[str, object]] = {}
    for job, raw in zip(jobs, state["results"]):  # type: ignore[arg-type]
        if type(raw) is not dict or set(raw) != expected_keys:
            raise RowFeatureProxyError("result state keys are invalid")
        if raw["candidate_id"] != job.candidate_id or raw["candidate_id"] in parsed:
            raise RowFeatureProxyError("duplicate, missing, or foreign result identity")
        if raw["feature_bundle"] != job.feature_bundle or raw["seed"] != job.seed:
            raise RowFeatureProxyError("result feature or seed identity differs")
        if raw["job_sha256"] != _job_sha(job):
            raise RowFeatureProxyError("result job SHA-256 differs")
        status = raw["status"]
        disposition = raw["disposition"]
        if status not in _STATUSES or disposition not in _DISPOSITIONS:
            raise RowFeatureProxyError("result status or disposition is invalid")
        if disposition == "not_started" and status != "inconclusive":
            raise RowFeatureProxyError("not-started result must be inconclusive")
        if disposition != "not_started" and disposition != status:
            raise RowFeatureProxyError("result disposition contradicts status")
        brier = raw["brier"]
        if status == "completed":
            if type(brier) is not float or not math.isfinite(brier) or not 0.0 <= brier <= 1.0:
                raise RowFeatureProxyError("completed state brier is invalid")
        elif brier is not None and (
            type(brier) is not float
            or not math.isfinite(brier)
            or not 0.0 <= brier <= 1.0
        ):
            raise RowFeatureProxyError(
                "non-completed state brier must be finite and in [0, 1] or None"
            )
        completed_epochs = raw["completed_epochs"]
        if (
            isinstance(completed_epochs, bool)
            or not isinstance(completed_epochs, int)
            or completed_epochs < 0
        ):
            raise RowFeatureProxyError(
                "result completed_epochs must be a nonnegative integer"
            )
        best_epoch = raw["best_epoch"]
        if best_epoch is not None and (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or best_epoch < 0
        ):
            raise RowFeatureProxyError(
                "result best_epoch must be a nonnegative integer or None"
            )
        failure = raw["failure"]
        if failure is not None and not isinstance(failure, str):
            raise RowFeatureProxyError("result failure must be a string or None")
        artifacts = _validate_artifact_mapping(
            raw["artifacts"], candidate_id=job.candidate_id
        )
        checkpoint_name = raw["checkpoint_name"]
        predictions_name = raw["predictions_name"]
        if checkpoint_name is not None and checkpoint_name not in _CHECKPOINT_NAMES:
            raise RowFeatureProxyError("state checkpoint name is invalid")
        if predictions_name is not None and predictions_name != "predictions.csv":
            raise RowFeatureProxyError("state predictions name is invalid")
        if checkpoint_name is not None and checkpoint_name not in artifacts:
            raise RowFeatureProxyError("state checkpoint artifact is missing")
        if status == "completed" and (
            checkpoint_name not in artifacts or "predictions.csv" not in artifacts
        ):
            raise RowFeatureProxyError("completed state artifacts are incomplete")
        if status == "completed" and failure is not None:
            raise RowFeatureProxyError("completed state failure must be None")
        if status != "completed" and predictions_name is not None:
            raise RowFeatureProxyError(
                "non-completed state predictions name must be None"
            )
        if disposition == "not_started" and any(
            (
                brier is not None,
                best_epoch is not None,
                completed_epochs != 0,
                checkpoint_name is not None,
                predictions_name is not None,
                raw["resource_evidence"] != {},
                failure != "not_started",
                artifacts != {},
            )
        ):
            raise RowFeatureProxyError("not-started result state is not canonical")
        normalized = dict(raw)
        normalized["resource_evidence"] = _json_resource(raw["resource_evidence"])
        normalized["artifacts"] = artifacts
        parsed[job.candidate_id] = normalized
    return parsed


def _verify_local_artifacts(
    output_dir: Path,
    rows: Mapping[str, dict[str, object]],
) -> None:
    for row in rows.values():
        for binding in row["artifacts"].values():  # type: ignore[union-attr]
            path = output_dir / str(binding["path"])
            if not _regular_file(path) or _file_sha256(path) != binding["sha256"]:
                raise RowFeatureProxyError(f"local artifact hash differs: {binding['path']}")


def _read_json_bytes(value: bytes, label: str) -> object:
    try:
        return json.loads(value.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RowFeatureProxyError(f"{label} is not valid UTF-8 JSON") from error


def _load_local_state(
    state_path: Path,
    *,
    contract_sha: str,
    train_sha: str,
    code_sha: str,
    jobs: tuple[CampaignJob, ...],
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if state_path.is_symlink() or not _regular_file(state_path):
        raise RowFeatureProxyError("local stage state must be a regular non-symlink file")
    state = _validate_state_header(
        _read_json_bytes(state_path.read_bytes(), "local stage state"),
        contract_sha=contract_sha,
        train_sha=train_sha,
        code_sha=code_sha,
        jobs=jobs,
    )
    rows = _validated_rows(state, jobs)
    _verify_local_artifacts(output_dir, rows)
    return state, rows


def _resume_expected_members(rows: Mapping[str, dict[str, object]]) -> set[str]:
    expected = set(_BASE_MEMBERS)
    for row in rows.values():
        artifacts = row["artifacts"]
        expected.update(binding["path"] for binding in artifacts.values())  # type: ignore[union-attr]
        if row["status"] == "completed":
            expected.add(f"predictions/{row['candidate_id']}.csv")
    return expected


def _restore_resume(
    resume_path: Path,
    *,
    output_dir: Path,
    config_bytes: bytes,
    contract_sha: str,
    train_sha: str,
    code_sha: str,
    contract: RowFeatureProxyContract,
    jobs: tuple[CampaignJob, ...],
) -> tuple[dict[str, object], dict[str, dict[str, object]], str]:
    try:
        verified = verify_resume_bundle(resume_path)
    except ArtifactError as error:
        raise RowFeatureProxyError(f"resume bundle is untrusted: {error}") from error
    if verified.version != _VERSION or verified.campaign_config_sha256 != contract_sha:
        raise RowFeatureProxyError("resume version or contract SHA-256 differs")
    try:
        with ZipFile(resume_path, "r") as archive:
            names = _validate_archive_entries(archive, label="resume")
            expected_names = set(verified.member_sha256) | {"manifest.json"}
            if set(names) != expected_names:
                raise RowFeatureProxyError(
                    "resume member set changed after verification"
                )
            members: dict[str, bytes] = {}
            for name, expected_hash in verified.member_sha256.items():
                value = archive.read(name)
                if sha256(value).hexdigest() != expected_hash:
                    raise RowFeatureProxyError(
                        f"resume member changed after verification: {name}"
                    )
                members[name] = value
    except ArtifactError as error:
        raise RowFeatureProxyError(f"resume bundle is untrusted: {error}") from error
    if members.get(_CONFIG_MEMBER) != config_bytes:
        raise RowFeatureProxyError("resume contains the wrong exact contract bytes")
    if _STATE_MEMBER not in members:
        raise RowFeatureProxyError("resume stage state is missing")
    state = _validate_state_header(
        _read_json_bytes(members[_STATE_MEMBER], "resume stage state"),
        contract_sha=contract_sha,
        train_sha=train_sha,
        code_sha=code_sha,
        jobs=jobs,
    )
    if state["prior_manifest_sha256"] != verified.prior_manifest_sha256:
        raise RowFeatureProxyError("resume state prior manifest binding differs")
    rows = _validated_rows(state, jobs)
    resume_decision = _decision(jobs, rows, contract)
    if state["stage_complete"] != (
        resume_decision.status in {"complete", "blocked"}
    ):
        raise RowFeatureProxyError(
            "resume stage_complete contradicts its results"
        )
    expected = _resume_expected_members(rows)
    if set(members) != expected:
        raise RowFeatureProxyError("resume member set differs from its explicit state schema")
    if members[_METRICS_MEMBER] != _canonical_json(state["results"]):
        raise RowFeatureProxyError("resume metrics differ from stage state")
    if members[_LOG_MEMBER] != _stage_log(state, resume_decision):
        raise RowFeatureProxyError("resume log differs from stage state")
    for row in rows.values():
        artifacts = row["artifacts"]
        for binding in artifacts.values():  # type: ignore[union-attr]
            member = binding["path"]
            value = members[member]
            if sha256(value).hexdigest() != binding["sha256"]:
                raise RowFeatureProxyError(f"resume artifact binding differs: {member}")
        if row["status"] == "completed":
            prediction = f"predictions/{row['candidate_id']}.csv"
            artifact = artifacts["predictions.csv"]  # type: ignore[index]
            if members[prediction] != members[artifact["path"]]:
                raise RowFeatureProxyError("resume completed prediction copies differ")
    jobs_root = output_dir / "jobs"
    if jobs_root.exists() and (jobs_root.is_symlink() or not jobs_root.is_dir()):
        raise RowFeatureProxyError("resume jobs root must be a regular directory")
    jobs_root.mkdir(parents=True, exist_ok=True)
    for row in rows.values():
        for binding in row["artifacts"].values():  # type: ignore[union-attr]
            target = output_dir / binding["path"]
            if target.parent.exists() and (
                target.parent.is_symlink() or not target.parent.is_dir()
            ):
                raise RowFeatureProxyError("resume candidate directory is unsafe")
            _atomic_bytes(target, members[binding["path"]])
    _verify_local_artifacts(output_dir, rows)
    return state, rows, verified.manifest_sha256


def _result_from_row(
    row: dict[str, object],
    output_dir: Path,
) -> CampaignJobResult:
    job_dir = output_dir / "jobs" / str(row["candidate_id"])
    checkpoint_name = row["checkpoint_name"]
    predictions_name = row["predictions_name"]
    return CampaignJobResult(
        candidate_id=str(row["candidate_id"]),
        status=str(row["status"]),
        brier=None if row["brier"] is None else float(row["brier"]),
        best_epoch=None if row["best_epoch"] is None else int(row["best_epoch"]),
        completed_epochs=int(row["completed_epochs"]),
        checkpoint=None if checkpoint_name is None else job_dir / str(checkpoint_name),
        predictions_path=None if predictions_name is None else job_dir / str(predictions_name),
        resource_evidence=dict(row["resource_evidence"]),  # type: ignore[arg-type]
        failure=None if row["failure"] is None else str(row["failure"]),
    )


def _validate_recovered_rows(
    rows: Mapping[str, dict[str, object]],
    jobs: tuple[CampaignJob, ...],
    output_dir: Path,
) -> None:
    for job in jobs:
        row = rows.get(job.candidate_id)
        if row is None or row["disposition"] == "not_started":
            continue
        recovered = _result_from_row(row, output_dir)
        rebuilt = _validate_runtime_result(
            job,
            recovered,
            output_dir / "jobs" / job.candidate_id,
        )
        if rebuilt != row:
            raise RowFeatureProxyError(
                f"recovered result state differs: {job.candidate_id}"
            )


def _now_value(now: Callable[[], float]) -> float:
    value = now()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RowFeatureProxyError("now() must return a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RowFeatureProxyError("now() must return a finite number")
    return result


def _decision(
    jobs: tuple[CampaignJob, ...], rows: Mapping[str, dict[str, object]], contract: RowFeatureProxyContract
) -> ProxyDecision:
    return decide_proxy_survivors(
        (
            ProxyMetric(
                job.feature_bundle,
                job.seed,
                str(rows.get(job.candidate_id, _pending_row(job))["status"]),
                rows.get(job.candidate_id, _pending_row(job))["brier"],  # type: ignore[arg-type]
            )
            for job in jobs
        ),
        contract,
    )


def _stage_log(state: dict[str, object], decision: ProxyDecision) -> bytes:
    rows = state["results"]
    counts = {
        status: sum(row["status"] == status for row in rows)  # type: ignore[index]
        for status in ("completed", "failed", "inconclusive")
    }
    not_started = sum(row["disposition"] == "not_started" for row in rows)  # type: ignore[index]
    return (
        f"version=P\n"
        f"completed={counts['completed']}\n"
        f"failed={counts['failed']}\n"
        f"inconclusive={counts['inconclusive']}\n"
        f"not_started={not_started}\n"
        f"decision={decision.status}\n"
        f"reason={decision.reason or 'none'}\n"
    ).encode("utf-8")


def _bundle_members(
    *,
    output_dir: Path,
    config_bytes: bytes,
    state: dict[str, object],
    rows: Mapping[str, dict[str, object]],
    decision: ProxyDecision,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    state_bytes = _canonical_json(state)
    metrics_bytes = _canonical_json(state["results"])
    log_bytes = _stage_log(state, decision)
    review = {
        _CONFIG_MEMBER: config_bytes,
        _METRICS_MEMBER: metrics_bytes,
        _LOG_MEMBER: log_bytes,
        _STATE_MEMBER: state_bytes,
    }
    resume = dict(review)
    if decision.status in {"complete", "blocked"}:
        review[_DECISION_MEMBER] = proxy_decision_json(decision)
    for row in rows.values():
        artifacts = row["artifacts"]
        for binding in artifacts.values():  # type: ignore[union-attr]
            path = output_dir / binding["path"]
            if not _regular_file(path) or _file_sha256(path) != binding["sha256"]:
                raise RowFeatureProxyError(f"artifact changed before publication: {binding['path']}")
            resume[binding["path"]] = path.read_bytes()
        if row["status"] == "completed":
            prediction_binding = artifacts["predictions.csv"]  # type: ignore[index]
            prediction_path = output_dir / prediction_binding["path"]
            name = f"predictions/{row['candidate_id']}.csv"
            value = prediction_path.read_bytes()
            review[name] = value
            resume[name] = value
    return review, resume


def run_row_feature_proxy(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    resume_bundle: str | Path | None = None,
    runtime: CampaignRuntime | None = None,
    gpu_count: int = 1,
    wall_deadline: float,
    contract_path: str | Path = DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
    now: Callable[[], float] = time.time,
) -> RowFeatureProxyRun:
    """Advance Stage P sequentially and publish deterministic review/resume evidence."""

    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise RowFeatureProxyError("gpu_count must be a positive integer")
    if isinstance(wall_deadline, bool) or not isinstance(wall_deadline, (int, float)):
        raise RowFeatureProxyError("wall_deadline must be a finite number")
    wall_deadline = float(wall_deadline)
    if not math.isfinite(wall_deadline):
        raise RowFeatureProxyError("wall_deadline must be finite")
    current = _now_value(now)
    if wall_deadline <= current:
        raise RowFeatureProxyError("wall_deadline is expired")

    try:
        contract = load_row_feature_proxy_contract(contract_path)
        contract_sha = row_feature_contract_sha256(contract_path)
        config_bytes = _read_contract_bytes(contract_path)
    except Exception as error:
        raise RowFeatureProxyError(f"cannot trust the Stage P contract: {error}") from error
    if sha256(config_bytes).hexdigest() != contract_sha:
        raise RowFeatureProxyError("exact contract bytes changed during validation")
    maximum_deadline = current + contract.budget.wall_seconds
    if wall_deadline > maximum_deadline:
        raise RowFeatureProxyError(
            "wall_deadline exceeds now() + contract budget.wall_seconds"
        )
    train_path = _find_official_train(Path(data_dir))
    train_sha = _official_train_file_sha256(train_path)
    if train_sha != contract.official_train_sha256:
        raise RowFeatureProxyError("official train.csv SHA-256 differs from the contract")
    code_sha = _code_sha256()
    jobs = build_proxy_jobs(contract)
    job_deadline = wall_deadline - contract.budget.new_job_guard_seconds
    root = Path(output_dir)
    jobs_root = root / "jobs"
    state_path = root / "stage_state.json"
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise RowFeatureProxyError("output_dir must be a regular directory")
    root.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, object]] = {}
    prior_manifest_sha: str | None = None
    recovered_state: dict[str, object] | None = None
    if resume_bundle is not None and state_path.exists():
        raise RowFeatureProxyError("cannot mix a resume bundle with existing local stage state")
    if resume_bundle is not None:
        recovered_state, rows, prior_manifest_sha = _restore_resume(
            Path(resume_bundle),
            output_dir=root,
            config_bytes=config_bytes,
            contract_sha=contract_sha,
            train_sha=train_sha,
            code_sha=code_sha,
            contract=contract,
            jobs=jobs,
        )
    elif state_path.exists() or state_path.is_symlink():
        recovered_state, rows = _load_local_state(
            state_path,
            contract_sha=contract_sha,
            train_sha=train_sha,
            code_sha=code_sha,
            jobs=jobs,
            output_dir=root,
        )
        prior_manifest_sha = recovered_state["prior_manifest_sha256"]  # type: ignore[assignment]

    _validate_recovered_rows(rows, jobs, root)

    if runtime is None:
        from .worker import SubprocessCampaignRuntime

        runtime = SubprocessCampaignRuntime(Path(data_dir))

    decision = _decision(jobs, rows, contract)
    if recovered_state is not None and recovered_state["stage_complete"] != (
        decision.status in {"complete", "blocked"}
    ):
        raise RowFeatureProxyError("recovered stage_complete contradicts its results")
    state = _state_payload(
        contract_sha=contract_sha,
        train_sha=train_sha,
        code_sha=code_sha,
        jobs=jobs,
        rows=rows,
        prior_manifest_sha=prior_manifest_sha,
        decision=decision,
    )
    _write_local_state(state_path, state)

    for job in jobs:
        if decision.status == "blocked":
            break
        previous = rows.get(job.candidate_id)
        if previous is not None and previous["status"] in {"completed", "failed"}:
            continue
        if _now_value(now) >= job_deadline:
            break
        returned = runtime.run_jobs(
            _VERSION,
            (job,),
            jobs_root,
            gpu_count=gpu_count,
            job_deadline=job_deadline,
        )
        if type(returned) is not tuple or len(returned) != 1:
            raise RowFeatureProxyError("runtime must return exactly one result for each one-job call")
        row = _validate_runtime_result(job, returned[0], jobs_root / job.candidate_id)
        rows[job.candidate_id] = row
        decision = _decision(jobs, rows, contract)
        state = _state_payload(
            contract_sha=contract_sha,
            train_sha=train_sha,
            code_sha=code_sha,
            jobs=jobs,
            rows=rows,
            prior_manifest_sha=prior_manifest_sha,
            decision=decision,
        )
        _write_local_state(state_path, state)
        if row["status"] == "inconclusive":
            break
        if job.feature_bundle is None and row["status"] == "failed":
            break

    decision = _decision(jobs, rows, contract)
    state = _state_payload(
        contract_sha=contract_sha,
        train_sha=train_sha,
        code_sha=code_sha,
        jobs=jobs,
        rows=rows,
        prior_manifest_sha=prior_manifest_sha,
        decision=decision,
    )
    _write_local_state(state_path, state)
    review, resume = _bundle_members(
        output_dir=root,
        config_bytes=config_bytes,
        state=state,
        rows=rows,
        decision=decision,
    )
    evidence = StageEvidence(
        version=_VERSION,
        campaign_config_sha256=contract_sha,
        prior_manifest_sha256=prior_manifest_sha,
        review_members=review,
        resume_members=resume,
    )
    bundles = write_stage_bundles(
        root,
        evidence,
        bundle_prefix="tabm_row_feature_stage",
    )
    ordered_rows = [rows.get(job.candidate_id, _pending_row(job)) for job in jobs]
    completed = tuple(row["candidate_id"] for row in ordered_rows if row["status"] == "completed")
    failed = tuple(row["candidate_id"] for row in ordered_rows if row["status"] == "failed")
    inconclusive = tuple(
        row["candidate_id"] for row in ordered_rows if row["status"] == "inconclusive"
    )
    return RowFeatureProxyRun(
        version=_VERSION,
        bundles=bundles,
        completed=completed,
        failed=failed,
        inconclusive=inconclusive,
        decision=decision,
        state_path=state_path,
    )
