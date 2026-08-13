from __future__ import annotations

import json
import math
import os
import resource
import tempfile
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Protocol


class TrainingCampaignError(ValueError):
    """Raised when a restartable training request is malformed."""


class CheckpointBindingError(TrainingCampaignError):
    """Raised when a checkpoint belongs to different code, config, or data."""


@dataclass(frozen=True)
class RestartableRequest:
    candidate_id: str
    max_epochs: int
    min_epochs: int
    patience: int
    absolute_deadline: float | None
    config_sha256: str
    source_sha256: str
    cache_sha256: str


@dataclass(frozen=True)
class EpochResult:
    metric: float
    checkpoint_bytes: bytes


@dataclass(frozen=True)
class RestartableResult:
    candidate_id: str
    status: str
    completed_epochs: int
    best_epoch: int | None
    best_metric: float | None
    checkpoint_sha256: str | None
    validation_curve: tuple[tuple[int, float], ...]
    reason: str


@dataclass(frozen=True)
class PreflightResult:
    candidate_id: str
    status: str
    failure_type: str | None
    message: str | None
    elapsed_seconds: float
    process_peak_rss_bytes: int
    evidence: Mapping[str, object]


class EpochBackend(Protocol):
    def run_epoch(self, epoch: int) -> EpochResult: ...


class EarlyStopper:
    def __init__(self, *, min_epochs: int, patience: int) -> None:
        if min_epochs <= 0 or patience <= 0:
            raise TrainingCampaignError("min_epochs and patience must be positive")
        self.min_epochs = min_epochs
        self.patience = patience
        self.completed = 0
        self.best = math.inf
        self.stale = 0

    def restore(self, curve: tuple[tuple[int, float], ...]) -> None:
        for _, metric in curve:
            self.update(metric)

    def update(self, metric: float) -> bool:
        if not math.isfinite(metric):
            raise TrainingCampaignError("validation metric must be finite")
        self.completed += 1
        if metric < self.best:
            self.best = metric
            self.stale = 0
        else:
            self.stale += 1
        return self.completed >= self.min_epochs and self.stale >= self.patience


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _atomic_bytes(path, encoded)


def _binding(request: RestartableRequest) -> dict[str, str]:
    bindings = {
        "candidate_id": request.candidate_id,
        "config_sha256": request.config_sha256,
        "source_sha256": request.source_sha256,
        "cache_sha256": request.cache_sha256,
    }
    for name, value in bindings.items():
        if not value:
            raise TrainingCampaignError(f"{name} must not be empty")
        if name.endswith("sha256") and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            raise TrainingCampaignError(f"{name} must be a lowercase SHA-256")
    return bindings


def _load_resume(root: Path, request: RestartableRequest) -> tuple[tuple[tuple[int, float], ...], str | None]:
    metadata_path = root / "checkpoint_meta.json"
    if not metadata_path.exists():
        return (), None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointBindingError(f"checkpoint metadata is invalid: {exc}") from exc
    if metadata.get("binding") != _binding(request):
        raise CheckpointBindingError("checkpoint binding differs from this request")
    checkpoint = root / "checkpoint.bin"
    expected = metadata.get("checkpoint_sha256")
    if not checkpoint.is_file() or _sha256_bytes(checkpoint.read_bytes()) != expected:
        raise CheckpointBindingError("checkpoint SHA-256 differs")
    curve = tuple((int(item[0]), float(item[1])) for item in metadata.get("validation_curve", ()))
    if [epoch for epoch, _ in curve] != list(range(len(curve))):
        raise CheckpointBindingError("checkpoint validation curve is not contiguous")
    if any(not math.isfinite(metric) for _, metric in curve):
        raise CheckpointBindingError("checkpoint validation curve is non-finite")
    return curve, str(expected)


def train_restartable(
    request: RestartableRequest,
    output_dir: str | Path,
    *,
    backend: EpochBackend,
    clock: Callable[[], float] = time.time,
) -> RestartableResult:
    """Advance complete epochs and return a terminal typed status at deadlines."""

    if request.max_epochs <= 0 or request.min_epochs <= 0 or request.patience <= 0:
        raise TrainingCampaignError("epoch and patience values must be positive")
    if request.min_epochs > request.max_epochs:
        raise TrainingCampaignError("min_epochs must not exceed max_epochs")
    _binding(request)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    curve, checkpoint_sha = _load_resume(root, request)
    stopper = EarlyStopper(min_epochs=request.min_epochs, patience=request.patience)
    stopper.restore(curve)

    reason = "max_epochs_reached"
    status = "completed"
    for epoch in range(len(curve), request.max_epochs):
        if request.absolute_deadline is not None and clock() >= request.absolute_deadline:
            status = "inconclusive"
            reason = "absolute_deadline_reached"
            break
        epoch_result = backend.run_epoch(epoch)
        metric = float(epoch_result.metric)
        if not math.isfinite(metric) or not epoch_result.checkpoint_bytes:
            status = "failed"
            reason = "invalid_epoch_result"
            break
        checkpoint_sha = _sha256_bytes(epoch_result.checkpoint_bytes)
        curve = (*curve, (epoch, metric))
        _atomic_bytes(root / "checkpoint.bin", epoch_result.checkpoint_bytes)
        _atomic_json(
            root / "checkpoint_meta.json",
            {
                "binding": _binding(request),
                "epoch": epoch,
                "checkpoint_sha256": checkpoint_sha,
                "validation_curve": curve,
            },
        )
        if stopper.update(metric):
            reason = "early_stopping"
            break

    if not curve:
        return RestartableResult(request.candidate_id, status, 0, None, None, None, (), reason)
    best_epoch, best_metric = min(curve, key=lambda item: (item[1], item[0]))
    return RestartableResult(
        request.candidate_id,
        status,
        len(curve),
        best_epoch,
        best_metric,
        checkpoint_sha,
        curve,
        reason,
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def preflight(candidate_id: str, operation: Callable[[], object]) -> PreflightResult:
    """Run a one-batch architecture probe without aborting sibling candidates."""

    started = time.monotonic()
    try:
        value = operation()
        evidence = dict(value) if isinstance(value, Mapping) else {}
        loss = evidence.get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            raise RuntimeError("non-finite loss")
        return PreflightResult(
            candidate_id,
            "completed",
            None,
            None,
            time.monotonic() - started,
            _peak_rss_bytes(),
            evidence,
        )
    except Exception as exc:  # preflight must isolate one candidate failure
        message = str(exc)
        lowered = message.casefold()
        failure_type = (
            "cuda_oom"
            if "cuda" in lowered and "out of memory" in lowered
            else "non_finite_loss"
            if "non-finite" in lowered
            else "preflight_error"
        )
        return PreflightResult(
            candidate_id,
            "failed",
            failure_type,
            message,
            time.monotonic() - started,
            _peak_rss_bytes(),
            {},
        )
