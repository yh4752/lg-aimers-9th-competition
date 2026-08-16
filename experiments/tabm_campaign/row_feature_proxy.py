"""Resumable, review-only orchestration for the sealed Stage P proxy."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import sys
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
    _StagePFile,
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
_MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024
_MAX_CONTROL_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_TABM_NUMERIC_FEATURES = 4096
_MAX_TABM_PIECEWISE_BINS = 48
_MAX_TABM_INPUT_WIDTH = 131_072
_MAX_TABM_TENSOR_NUMEL = 50_000_000


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
        checkpoint_size = path.stat().st_size
    except OSError as error:
        raise RowFeatureProxyError("checkpoint payload size cannot be read") from error
    if checkpoint_size <= 0 or checkpoint_size > _MAX_CHECKPOINT_BYTES:
        raise RowFeatureProxyError("checkpoint payload exceeds the Stage P size limit")
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
    if result.brier is not None and not math.isclose(
        result.brier, best_brier, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RowFeatureProxyError("checkpoint best Brier differs from result Brier")
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
    _validate_restart_state(payload, torch, np)
    if payload["adapter_state"] != meta_adapter_state:
        raise RowFeatureProxyError("checkpoint adapter state is invalid")
    parameter_tensors = _validate_tabm_model_state(job, payload["model"], torch)
    optimizer_lr = _validate_adamw_state(
        job,
        payload["optimizer"],
        parameter_tensors,
        completed_epochs=epoch + 1,
        torch=torch,
    )
    _validate_plateau_scheduler_state(
        job,
        payload["scheduler"],
        epoch=epoch,
        validation_curve=validation_curve,
        optimizer_lr=optimizer_lr,
    )


def _valid_finite_tensor(
    value: object,
    shape: tuple[int, ...],
    torch: object,
    *,
    dtype: object,
) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.dtype == dtype
        and tuple(value.shape) == shape
        and value.numel() <= _MAX_TABM_TENSOR_NUMEL
        and value.is_contiguous()
        and value.untyped_storage().nbytes() >= value.numel() * value.element_size()
        and bool(torch.isfinite(value).all().item())
    )


def _validate_restart_state(payload: Mapping[str, object], torch: object, np: object) -> None:
    scaler = payload["scaler"]
    scaler_keys = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if (
        not isinstance(scaler, Mapping)
        or set(scaler) != scaler_keys
        or type(scaler["scale"]) is not float
        or not math.isfinite(scaler["scale"])
        or scaler["scale"] <= 0.0
        or scaler["growth_factor"] != 2.0
        or scaler["backoff_factor"] != 0.5
        or scaler["growth_interval"] != 2000
        or type(scaler["_growth_tracker"]) is not int
        or not 0 <= scaler["_growth_tracker"] < 2000
    ):
        raise RowFeatureProxyError("checkpoint AMP scaler state is invalid")
    try:
        isolated_scaler = torch.amp.GradScaler("cpu", enabled=True)
        isolated_scaler.load_state_dict(dict(scaler))
    except Exception as error:
        raise RowFeatureProxyError("checkpoint AMP scaler cannot be restored") from error
    try:
        import random

        random.Random().setstate(payload["python_rng"])
        np.random.RandomState().set_state(payload["numpy_rng"])
    except Exception as error:
        raise RowFeatureProxyError("checkpoint Python or NumPy RNG cannot be restored") from error
    torch_rng = payload["torch_rng"]
    reference_rng = torch.Generator(device="cpu").get_state()
    if (
        not isinstance(torch_rng, torch.Tensor)
        or torch_rng.device.type != "cpu"
        or torch_rng.dtype != torch.uint8
        or torch_rng.layout != torch.strided
        or not torch_rng.is_contiguous()
        or tuple(torch_rng.shape) != tuple(reference_rng.shape)
        or torch_rng.untyped_storage().nbytes() < torch_rng.numel()
    ):
        raise RowFeatureProxyError("checkpoint Torch RNG state is invalid")
    try:
        torch.Generator(device="cpu").set_state(torch_rng)
    except Exception as error:
        raise RowFeatureProxyError("checkpoint Torch RNG cannot be restored") from error
    cuda_rng = payload["cuda_rng"]
    if (
        type(cuda_rng) not in {list, tuple}
        or len(cuda_rng) != 1
        or any(
            not isinstance(state, torch.Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
            or state.layout != torch.strided
            or state.ndim != 1
            or state.numel() not in {8, 16}
            or not state.is_contiguous()
            or state.untyped_storage().nbytes() < state.numel()
            for state in cuda_rng
        )
    ):
        raise RowFeatureProxyError("checkpoint CUDA RNG state is invalid")
    for state in cuda_rng:
        raw = bytes(state.tolist())
        if len(raw) == 16:
            offset = int.from_bytes(raw[8:], byteorder=sys.byteorder, signed=True)
            if offset < 0 or offset % 4:
                raise RowFeatureProxyError("checkpoint CUDA RNG state is invalid")
        if torch.cuda.is_available():
            try:
                torch.Generator(device="cuda:0").set_state(state)
            except Exception as error:
                raise RowFeatureProxyError(
                    "checkpoint CUDA RNG cannot be restored"
                ) from error


def _validate_tabm_model_state(
    job: CampaignJob,
    value: object,
    torch: object,
) -> tuple[object, ...]:
    if job.num_embedding != "piecewise_linear" or not isinstance(value, Mapping):
        raise RowFeatureProxyError("checkpoint TabM model state is invalid")
    state = value
    numeric_keys = {
        "model.num_module.linear0.weight",
        "model.num_module.linear0.bias",
        "model.num_module.impl.weight",
        "model.num_module.impl.bias",
        "model.num_module.linear.weight",
    }
    block_keys = {
        f"model.backbone.blocks.{block}.0.{suffix}"
        for block in range(job.blocks)
        for suffix in ("weight", "r", "s", "bias")
    }
    required_keys = numeric_keys | block_keys | {
        "model.output.weight",
        "model.output.bias",
    }
    optional_keys = {"model.num_module.impl.mask"}
    if not required_keys.issubset(state) or set(state) - required_keys - optional_keys:
        raise RowFeatureProxyError("checkpoint TabM model keys are invalid")

    linear0 = state["model.num_module.linear0.weight"]
    impl = state["model.num_module.impl.weight"]
    first_block = state["model.backbone.blocks.0.0.weight"]
    if (
        not isinstance(linear0, torch.Tensor)
        or linear0.ndim != 2
        or linear0.shape[0] <= 0
        or tuple(linear0.shape[1:]) != (32,)
        or not isinstance(impl, torch.Tensor)
        or impl.ndim != 2
        or impl.shape[0] != linear0.shape[0]
        or impl.shape[1] <= 0
        or not isinstance(first_block, torch.Tensor)
        or first_block.ndim != 2
    ):
        raise RowFeatureProxyError("checkpoint TabM numerical embedding is invalid")
    n_num = int(linear0.shape[0])
    n_bins = int(impl.shape[1])
    first_width = int(first_block.shape[1])
    if (
        n_num > _MAX_TABM_NUMERIC_FEATURES
        or n_bins > _MAX_TABM_PIECEWISE_BINS
        or first_width > _MAX_TABM_INPUT_WIDTH
        or first_width < n_num * 32
    ):
        raise RowFeatureProxyError("checkpoint TabM input width is invalid")

    float_shapes = {
        "model.num_module.linear0.weight": (n_num, 32),
        "model.num_module.linear0.bias": (n_num, 32),
        "model.num_module.impl.weight": (n_num, n_bins),
        "model.num_module.impl.bias": (n_num, n_bins),
        "model.num_module.linear.weight": (n_num, n_bins, 32),
        "model.output.weight": (job.k, job.width, 1),
        "model.output.bias": (job.k, 1),
    }
    for block in range(job.blocks):
        block_input = first_width if block == 0 else job.width
        prefix = f"model.backbone.blocks.{block}.0"
        float_shapes.update(
            {
                f"{prefix}.weight": (job.width, block_input),
                f"{prefix}.r": (job.k, block_input),
                f"{prefix}.s": (job.k, job.width),
                f"{prefix}.bias": (job.k, job.width),
            }
        )
    if sum(math.prod(shape) for shape in float_shapes.values()) > _MAX_TABM_TENSOR_NUMEL:
        raise RowFeatureProxyError("checkpoint TabM tensors exceed the size limit")
    if any(
        not _valid_finite_tensor(
            state[name], shape, torch, dtype=torch.float32
        )
        for name, shape in float_shapes.items()
    ):
        raise RowFeatureProxyError("checkpoint TabM tensor shape or dtype is invalid")
    mask = state.get("model.num_module.impl.mask")
    if mask is not None and (
        not isinstance(mask, torch.Tensor)
        or mask.layout != torch.strided
        or mask.device.type != "cpu"
        or mask.dtype != torch.bool
        or tuple(mask.shape) != (n_num, n_bins)
        or not mask.is_contiguous()
        or mask.untyped_storage().nbytes() < mask.numel()
    ):
        raise RowFeatureProxyError("checkpoint TabM embedding mask is invalid")

    parameter_names = [
        "model.num_module.linear0.weight",
        "model.num_module.linear0.bias",
        "model.num_module.linear.weight",
    ]
    for block in range(job.blocks):
        prefix = f"model.backbone.blocks.{block}.0"
        parameter_names.extend(
            f"{prefix}.{suffix}" for suffix in ("weight", "r", "s", "bias")
        )
    parameter_names.extend(("model.output.weight", "model.output.bias"))
    return tuple(state[name] for name in parameter_names)


def _validate_adamw_state(
    job: CampaignJob,
    value: object,
    parameters: tuple[object, ...],
    *,
    completed_epochs: int,
    torch: object,
) -> float:
    if not isinstance(value, Mapping) or set(value) != {"state", "param_groups"}:
        raise RowFeatureProxyError("checkpoint AdamW state keys are invalid")
    state = value["state"]
    groups = value["param_groups"]
    group_keys = {
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
    }
    if (
        not isinstance(state, Mapping)
        or type(groups) is not list
        or len(groups) != 1
        or not isinstance(groups[0], Mapping)
        or set(groups[0]) != group_keys
    ):
        raise RowFeatureProxyError("checkpoint AdamW parameter groups are invalid")
    group = groups[0]
    parameter_ids = group["params"]
    expected_ids = list(range(len(parameters)))
    if (
        type(parameter_ids) is not list
        or parameter_ids != expected_ids
        or set(state) != set(expected_ids)
        or group["betas"] != (0.9, 0.999)
        or group["eps"] != 1e-8
        or group["weight_decay"] != 0.0001
        or group["amsgrad"] is not False
        or group["maximize"] is not False
        or group["foreach"] is not None
        or group["capturable"] is not False
        or group["differentiable"] is not False
        or group["fused"] is not None
        or group["decoupled_weight_decay"] is not True
        or type(group["lr"]) is not float
        or not math.isclose(
            group["lr"], job.learning_rate, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise RowFeatureProxyError("checkpoint AdamW configuration is invalid")

    optimizer_steps: set[int] = set()
    for parameter_id, parameter in enumerate(parameters):
        item = state[parameter_id]
        if not isinstance(item, Mapping) or set(item) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RowFeatureProxyError("checkpoint AdamW parameter state is invalid")
        step = item["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.dtype != torch.float32
            or step.numel() != 1
            or not math.isfinite(float(step.item()))
            or not float(step.item()).is_integer()
            or int(step.item()) < completed_epochs
        ):
            raise RowFeatureProxyError("checkpoint AdamW step is invalid")
        optimizer_steps.add(int(step.item()))
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = item[moment_name]
            if not _valid_finite_tensor(
                moment,
                tuple(parameter.shape),
                torch,
                dtype=parameter.dtype,
            ):
                raise RowFeatureProxyError(
                    "checkpoint AdamW moment shape or dtype is invalid"
                )
    if len(optimizer_steps) != 1:
        raise RowFeatureProxyError("checkpoint AdamW parameter steps differ")
    return group["lr"]


def _validate_plateau_scheduler_state(
    job: CampaignJob,
    value: object,
    *,
    epoch: int,
    validation_curve: list[tuple[int, float]],
    optimizer_lr: float,
) -> None:
    required = {
        "factor",
        "default_min_lr",
        "min_lrs",
        "patience",
        "cooldown",
        "cooldown_counter",
        "mode",
        "threshold",
        "threshold_mode",
        "eps",
        "last_epoch",
        "_last_lr",
        "mode_worse",
        "best",
        "num_bad_epochs",
    }
    if job.scheduler != "plateau" or not isinstance(value, Mapping) or set(value) != required:
        raise RowFeatureProxyError("checkpoint plateau scheduler keys are invalid")
    scheduler_best = math.inf
    num_bad_epochs = 0
    for _, brier in validation_curve:
        if brier < scheduler_best * (1.0 - 0.0001):
            scheduler_best = brier
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
    expected = {
        "factor": 0.1,
        "default_min_lr": 0,
        "min_lrs": [0],
        "patience": 10,
        "cooldown": 0,
        "cooldown_counter": 0,
        "mode": "min",
        "threshold": 0.0001,
        "threshold_mode": "rel",
        "eps": 1e-8,
        "last_epoch": epoch + 1,
        "_last_lr": [optimizer_lr],
        "mode_worse": math.inf,
        "best": scheduler_best,
        "num_bad_epochs": num_bad_epochs,
    }
    if dict(value) != expected:
        raise RowFeatureProxyError("checkpoint plateau scheduler state is invalid")


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


def _read_zip_control_member(
    archive: ZipFile,
    name: str,
    expected_sha256: str,
) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > _MAX_CONTROL_MEMBER_BYTES:
        raise RowFeatureProxyError(f"resume control member exceeds the size limit: {name}")
    digest = sha256()
    value = bytearray()
    with archive.open(info, "r") as source:
        while chunk := source.read(1024 * 1024):
            value.extend(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RowFeatureProxyError(f"resume member changed after verification: {name}")
    return bytes(value)


def _zip_member_sha256(archive: ZipFile, name: str) -> str:
    digest = sha256()
    with archive.open(name, "r") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_restore_member(
    archive: ZipFile,
    name: str,
    target: Path,
    expected_sha256: str,
) -> None:
    info = archive.getinfo(name)
    if name.endswith(("/checkpoint.pt", "/best_checkpoint.pt")) and (
        info.file_size <= 0 or info.file_size > _MAX_CHECKPOINT_BYTES
    ):
        raise RowFeatureProxyError(f"resume checkpoint exceeds the size limit: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}-", dir=target.parent
    )
    try:
        digest = sha256()
        with archive.open(info, "r") as source, os.fdopen(descriptor, "wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha256:
            raise RowFeatureProxyError(
                f"resume member changed after verification: {name}"
            )
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


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
        with ZipFile(resume_path, "r") as archive:
            _validate_archive_entries(archive, label="resume")
            if any(
                info.filename.endswith(("/checkpoint.pt", "/best_checkpoint.pt"))
                and (info.file_size <= 0 or info.file_size > _MAX_CHECKPOINT_BYTES)
                for info in archive.infolist()
            ):
                raise ArtifactError("resume checkpoint exceeds the Stage P size limit")
        verified = verify_resume_bundle(resume_path)
    except ArtifactError as error:
        raise RowFeatureProxyError(f"resume bundle is untrusted: {error}") from error
    except Exception as error:
        raise RowFeatureProxyError(f"resume bundle cannot be inspected: {error}") from error
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
            manifest_bytes = _read_zip_control_member(
                archive,
                "manifest.json",
                verified.manifest_sha256,
            )
            if sha256(manifest_bytes).hexdigest() != verified.manifest_sha256:
                raise RowFeatureProxyError("resume manifest changed after verification")
            if not _BASE_MEMBERS.issubset(verified.member_sha256):
                raise RowFeatureProxyError(
                    "resume member set differs from its explicit state schema"
                )
            controls = {
                name: _read_zip_control_member(
                    archive, name, verified.member_sha256[name]
                )
                for name in _BASE_MEMBERS
            }
            if controls[_CONFIG_MEMBER] != config_bytes:
                raise RowFeatureProxyError(
                    "resume contains the wrong exact contract bytes"
                )
            state = _validate_state_header(
                _read_json_bytes(controls[_STATE_MEMBER], "resume stage state"),
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
            if set(verified.member_sha256) != expected:
                raise RowFeatureProxyError(
                    "resume member set differs from its explicit state schema"
                )
            if controls[_METRICS_MEMBER] != _canonical_json(state["results"]):
                raise RowFeatureProxyError("resume metrics differ from stage state")
            if controls[_LOG_MEMBER] != _stage_log(state, resume_decision):
                raise RowFeatureProxyError("resume log differs from stage state")

            jobs_root = output_dir / "jobs"
            if jobs_root.exists() and (
                jobs_root.is_symlink() or not jobs_root.is_dir()
            ):
                raise RowFeatureProxyError(
                    "resume jobs root must be a regular directory"
                )
            jobs_root.mkdir(parents=True, exist_ok=True)
            for row in rows.values():
                artifacts = row["artifacts"]
                for binding in artifacts.values():  # type: ignore[union-attr]
                    member = binding["path"]
                    if verified.member_sha256[member] != binding["sha256"]:
                        raise RowFeatureProxyError(
                            f"resume artifact binding differs: {member}"
                        )
                    target = output_dir / member
                    if target.parent.exists() and (
                        target.parent.is_symlink() or not target.parent.is_dir()
                    ):
                        raise RowFeatureProxyError(
                            "resume candidate directory is unsafe"
                        )
                    _atomic_restore_member(
                        archive,
                        member,
                        target,
                        binding["sha256"],
                    )
                if row["status"] == "completed":
                    prediction = f"predictions/{row['candidate_id']}.csv"
                    artifact = artifacts["predictions.csv"]  # type: ignore[index]
                    if (
                        verified.member_sha256[prediction]
                        != artifact["sha256"]
                        or _zip_member_sha256(archive, prediction)
                        != artifact["sha256"]
                    ):
                        raise RowFeatureProxyError(
                            "resume completed prediction copies differ"
                        )
    except RowFeatureProxyError:
        raise
    except ArtifactError as error:
        raise RowFeatureProxyError(f"resume bundle is untrusted: {error}") from error
    except Exception as error:
        raise RowFeatureProxyError(f"cannot restore resume bundle: {error}") from error
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
) -> tuple[dict[str, bytes | _StagePFile], dict[str, bytes | _StagePFile]]:
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
            source = _StagePFile(path, binding["sha256"])
            resume[binding["path"]] = source
        if row["status"] == "completed":
            prediction_binding = artifacts["predictions.csv"]  # type: ignore[index]
            prediction_path = output_dir / prediction_binding["path"]
            name = f"predictions/{row['candidate_id']}.csv"
            prediction_source = _StagePFile(
                prediction_path, prediction_binding["sha256"]
            )
            review[name] = prediction_source
            resume[name] = prediction_source
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
