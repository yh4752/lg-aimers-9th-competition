"""Shared restartable trainer for independent DL candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from .features import FeatureBatch
from .models.common import (
    ModelAdapter,
    attach_progress_reporter,
    import_runtime_module,
    metadata_from_train,
)
from .progress import ProgressReporter


class TrainingContractError(ValueError):
    """Raised before a malformed training request can start GPU work."""


class TrainingTimeBudgetReached(RuntimeError):
    """Raised after preserving the last complete epoch checkpoint."""


def session_deadline_reached(deadline: float | None) -> bool:
    """Return whether a training boundary has reached its absolute deadline."""

    return deadline is not None and time.time() >= deadline


def enforce_session_deadline(deadline: float | None, *, boundary: str) -> None:
    """Stop before more GPU work when the user-owned session budget expires."""

    if session_deadline_reached(deadline):
        raise TrainingTimeBudgetReached(
            f"session time budget reached at {boundary}; resume from the last complete epoch"
        )


def progress_message(
    *,
    candidate_id: str,
    epoch: int,
    epochs: int,
    batch: int,
    batches: int,
    elapsed_seconds: float,
) -> str:
    """Format one machine- and human-readable training heartbeat."""

    completed_fraction = max(batch / max(batches, 1), 1e-12)
    eta = max(0, round(elapsed_seconds / completed_fraction - elapsed_seconds))
    return (
        "TRAINING_PROGRESS "
        f"candidate={candidate_id} epoch={epoch + 1}/{epochs} "
        f"batch={batch}/{batches} elapsed_seconds={round(elapsed_seconds)} "
        f"epoch_eta_seconds={eta}"
    )


def require_finite_validation_brier(
    *, candidate_id: str, epoch: int, brier: float
) -> None:
    """Fail before persisting a checkpoint with non-finite validation evidence."""

    if not math.isfinite(brier):
        raise RuntimeError(
            "non-finite validation Brier "
            f"candidate={candidate_id} epoch={epoch + 1} brier={brier}"
        )


def _session_deadline() -> float | None:
    raw = os.environ.get("PREPROCESSING_SESSION_DEADLINE_UNIX")
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise TrainingContractError(
            "PREPROCESSING_SESSION_DEADLINE_UNIX must be numeric"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise TrainingContractError(
            "PREPROCESSING_SESSION_DEADLINE_UNIX must be finite and positive"
        )
    return value


@dataclass(frozen=True)
class TrainRequest:
    candidate_id: str
    family: str
    seed: int
    epochs: int
    model_config: Mapping[str, object]
    training_config: Mapping[str, object]
    train: FeatureBatch
    valid: FeatureBatch


@dataclass(frozen=True)
class BackendAttemptResult:
    best_epoch: int
    best_brier: float
    checkpoint: Path
    predictions: np.ndarray
    hardware: Mapping[str, object] = field(default_factory=dict)
    completed_epochs: int = 0
    validation_curve: tuple[tuple[int, float], ...] = ()
    validation_time_curve: tuple[tuple[int, float, float], ...] = ()


@dataclass(frozen=True)
class TrainResult:
    candidate_id: str
    best_epoch: int
    best_brier: float
    checkpoint: Path
    predictions: np.ndarray
    attempted_micro_batches: tuple[int, ...]
    effective_batch_size: int
    model_config: Mapping[str, object]
    started_epoch: int
    hardware: Mapping[str, object] = field(default_factory=dict)
    completed_epochs: int = 0
    validation_curve: tuple[tuple[int, float], ...] = ()
    validation_time_curve: tuple[tuple[int, float, float], ...] = ()


class TrainingBackend(Protocol):
    def run_attempt(
        self,
        *,
        request: TrainRequest,
        adapter: ModelAdapter,
        output_dir: Path,
        micro_batch_size: int,
        accumulation_steps: int,
        activation_checkpointing: bool,
        resume_epoch: int,
        reporter: ProgressReporter,
    ) -> BackendAttemptResult: ...


def inspect_cuda_hardware(torch: object) -> dict[str, object]:
    """Describe visible CUDA devices and the device this backend actually uses."""

    count = int(torch.cuda.device_count())
    devices = tuple(
        {
            "index": index,
            "name": str(torch.cuda.get_device_name(index)),
            "vram_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        }
        for index in range(count)
    )
    return {
        "device_count": count,
        "devices": devices,
        "training_mode": "single_gpu" if count else "cpu",
        "training_device_indices": (0,) if count else (),
    }


def prepare_adapter_context(adapter: object, train: FeatureBatch) -> None:
    fit_context = getattr(adapter, "fit_context", None)
    if fit_context is not None:
        fit_context(train)


def refresh_retrieval_cache(adapter: object, model: object, device: str) -> None:
    refresh = getattr(adapter, "refresh_retrieval_cache", None)
    if refresh is not None:
        refresh(model, device)


def configure_adapter_output(adapter: object, output_dir: Path) -> None:
    setter = getattr(adapter, "set_output_dir", None)
    if setter is not None:
        setter(output_dir)


def prepare_initial_retrieval_cache(
    adapter: object, model: object, device: str
) -> None:
    prepare = getattr(adapter, "prepare_initial_retrieval_cache", None)
    if prepare is None:
        refresh_retrieval_cache(adapter, model, device)
    else:
        prepare(model, device)


def on_adapter_epoch_start(
    adapter: object, model: object, epoch: int, device: str
) -> None:
    hook = getattr(adapter, "on_epoch_start", None)
    if hook is not None:
        hook(model, epoch, device)


def on_adapter_epoch_end(
    adapter: object, model: object, epoch: int, device: str
) -> None:
    hook = getattr(adapter, "on_epoch_end", None)
    if hook is not None:
        hook(model, epoch, device)


def adapter_checkpoint_state(adapter: object) -> object | None:
    getter = getattr(adapter, "checkpoint_state", None)
    return None if getter is None else getter()


def restore_adapter_checkpoint_state(
    adapter: object, payload: object, model: object, device: str
) -> bool:
    restore = getattr(adapter, "restore_checkpoint_state", None)
    if restore is None:
        return False
    return bool(restore(payload, model, device))


def _fold_label(request: TrainRequest) -> str:
    train_years = np.asarray(request.train.season, dtype="int64")
    valid_years = np.unique(np.asarray(request.valid.season, dtype="int64"))
    if len(train_years) == 0 or len(valid_years) != 1:
        raise TrainingContractError("training fold requires one validation season")
    return f"{int(train_years.max())}->{int(valid_years[0])}"


def _gpu_memory(torch: object) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_bytes": int(torch.cuda.max_memory_allocated()),
    }


def report_training_window(
    *,
    reporter: object,
    torch: object,
    completed_rows: int,
    total_rows: int,
    started_at: float,
    epoch: int,
    epochs: int,
    batch: int,
    batches: int,
    loss: object,
) -> None:
    """Synchronize and report only when a meaningful heartbeat is due."""

    stream = f"training_epoch_{epoch}"
    if not reporter.should_emit(completed_rows=completed_rows, stream=stream):
        return
    torch.cuda.synchronize()
    detached = getattr(loss, "detach", None)
    loss_value = float(detached() if detached is not None else loss)
    reporter.progress(
        "TRAINING_PROGRESS",
        completed_rows=completed_rows,
        total_rows=total_rows,
        started_at=started_at,
        gpu=_gpu_memory(torch),
        epoch=epoch,
        epochs=epochs,
        batch=batch,
        batches=batches,
        loss=loss_value,
        stream=stream,
    )


def call_adapter_loss(
    adapter: ModelAdapter,
    model: object,
    x_num: object,
    x_cat: object,
    y: object,
    *,
    row_indices: object,
) -> object:
    return adapter.loss(
        model, x_num, x_cat, y, row_indices=row_indices
    )


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingContractError(f"{label} must be a positive integer")
    return value


def _validate_request(request: TrainRequest) -> tuple[int, int]:
    if not request.candidate_id:
        raise TrainingContractError("candidate_id must not be empty")
    if request.train.y is None or request.valid.y is None:
        raise TrainingContractError("train and validation targets are required")
    if len(request.train.x_num) != len(request.train.y):
        raise TrainingContractError("training features and targets are not aligned")
    if len(request.valid.x_num) != len(request.valid.y):
        raise TrainingContractError("validation features and targets are not aligned")
    effective = _positive_integer(
        request.training_config.get("effective_batch_size"), "effective_batch_size"
    )
    micro = _positive_integer(
        request.training_config.get("micro_batch_size"), "micro_batch_size"
    )
    if effective < micro or effective % micro:
        raise TrainingContractError(
            "effective_batch_size must be an exact multiple of micro_batch_size"
        )
    return effective, micro


def _resume_epoch(output_dir: Path, candidate_id: str) -> int:
    path = output_dir / "checkpoint_meta.json"
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError(f"cannot read checkpoint metadata: {error}") from error
    if not isinstance(payload, dict) or payload.get("candidate_id") != candidate_id:
        raise TrainingContractError("checkpoint candidate_id does not match the request")
    epoch = payload.get("epoch")
    checkpoint = payload.get("checkpoint")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise TrainingContractError("checkpoint epoch is invalid")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise TrainingContractError("checkpoint path is invalid")
    if not (output_dir / checkpoint).is_file():
        raise TrainingContractError("checkpoint file is missing")
    return epoch + 1


def _is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return "cuda" in message and "out of memory" in message


def fit_candidate(
    request: TrainRequest,
    adapter: ModelAdapter,
    output_dir: str | Path,
    *,
    backend: TrainingBackend | None = None,
) -> TrainResult:
    """Fit one unchanged model, adapting only memory execution parameters on OOM."""

    effective, micro = _validate_request(request)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(
        request.candidate_id,
        request.family,
        _fold_label(request),
        root,
    )
    attach_progress_reporter(adapter, reporter)
    configure_adapter_output(adapter, root)
    started_epoch = _resume_epoch(root, request.candidate_id)
    current_resume_epoch = started_epoch
    runtime = TorchTrainingBackend() if backend is None else backend
    attempted: list[int] = []
    activation_checkpointing = False

    try:
        while True:
            attempted.append(micro)
            try:
                attempt = runtime.run_attempt(
                    request=request,
                    adapter=adapter,
                    output_dir=root,
                    micro_batch_size=micro,
                    accumulation_steps=effective // micro,
                    activation_checkpointing=activation_checkpointing,
                    resume_epoch=current_resume_epoch,
                    reporter=reporter,
                )
                break
            except RuntimeError as error:
                if not _is_cuda_oom(error):
                    raise
                current_resume_epoch = _resume_epoch(root, request.candidate_id)
                if micro > 1:
                    micro //= 2
                    while effective % micro:
                        micro //= 2
                    continue
                if not activation_checkpointing:
                    activation_checkpointing = True
                    continue
                raise

        predictions = np.asarray(attempt.predictions, dtype="float64")
        if predictions.shape != (len(request.valid.row_id),):
            raise TrainingContractError("validation predictions have the wrong shape")
        if not np.isfinite(predictions).all() or (
            (predictions < 0) | (predictions > 1)
        ).any():
            raise TrainingContractError(
                "validation predictions must be finite probabilities"
            )
        if not math.isfinite(float(attempt.best_brier)):
            raise TrainingContractError("best_brier must be finite")
        result = TrainResult(
            candidate_id=request.candidate_id,
            best_epoch=attempt.best_epoch,
            best_brier=float(attempt.best_brier),
            checkpoint=Path(attempt.checkpoint),
            predictions=predictions,
            attempted_micro_batches=tuple(attempted),
            effective_batch_size=effective,
            model_config=MappingProxyType(dict(request.model_config)),
            started_epoch=started_epoch,
            hardware=MappingProxyType(dict(attempt.hardware)),
            completed_epochs=(
                int(attempt.completed_epochs)
                if int(attempt.completed_epochs) > 0
                else int(attempt.best_epoch) + 1
            ),
            validation_curve=tuple(
                (int(epoch), float(brier))
                for epoch, brier in attempt.validation_curve
            ),
            validation_time_curve=tuple(
                (int(epoch), float(elapsed), float(brier))
                for epoch, elapsed, brier in attempt.validation_time_curve
            ),
        )
        reporter.emit(
            "CANDIDATE_COMPLETED",
            best_epoch=result.best_epoch,
            best_brier=result.best_brier,
            completed_epochs=result.completed_epochs,
            checkpoint=result.checkpoint,
        )
        return result
    except Exception as error:
        try:
            reporter.emit(
                "CANDIDATE_FAILED",
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exc(),
            )
        except Exception:
            pass
        raise


def _atomic_torch_save(torch: object, payload: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_resume_payload(torch: object, path: Path) -> object:
    """Load optimizer and RNG state on CPU before restoring the CUDA model."""

    return torch.load(path, map_location="cpu", weights_only=False)


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, path)


def _make_scheduler(
    torch: object,
    optimizer: object,
    request: TrainRequest,
    updates_per_epoch: int,
) -> tuple[object, str]:
    name = str(request.training_config["scheduler"])
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min"), "metric"
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=request.epochs
        ), "epoch"
    if name == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(request.training_config["learning_rate"]),
            total_steps=request.epochs * updates_per_epoch,
        ), "update"
    if name == "cosine_warmup":
        warmup = max(1, request.epochs // 10)

        def schedule(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, request.epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), "epoch"
    raise TrainingContractError(f"unsupported scheduler: {name}")


class TorchTrainingBackend:
    """CUDA backend imported only when the user starts a real candidate."""

    def run_attempt(
        self,
        *,
        request: TrainRequest,
        adapter: ModelAdapter,
        output_dir: Path,
        micro_batch_size: int,
        accumulation_steps: int,
        activation_checkpointing: bool,
        resume_epoch: int,
        reporter: ProgressReporter,
    ) -> BackendAttemptResult:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch = import_runtime_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU가 필요합니다. Colab 런타임 유형에서 GPU를 선택하세요.")
        device = "cuda"
        hardware = inspect_cuda_hardware(torch)
        random.seed(request.seed)
        np.random.seed(request.seed)
        torch.manual_seed(request.seed)
        torch.cuda.manual_seed_all(request.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        metadata = metadata_from_train(request.train)
        prepare_adapter_context(adapter, request.train)
        model = adapter.build(request.model_config, metadata, device)
        if activation_checkpointing and hasattr(model, "enable_activation_checkpointing"):
            model.enable_activation_checkpointing()
        optimizer = adapter.optimizer(model, request.training_config)
        effective_batch = micro_batch_size * accumulation_steps
        updates_per_epoch = math.ceil(len(request.train.row_id) / effective_batch)
        scheduler, scheduler_mode = _make_scheduler(
            torch, optimizer, request, updates_per_epoch
        )
        amp_enabled = bool(request.training_config["amp"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        reporter.emit(
            "CANDIDATE_RUNTIME_READY",
            device=device,
            hardware=hardware,
            train_rows=len(request.train.row_id),
            validation_rows=len(request.valid.row_id),
            epochs=request.epochs,
            micro_batch_size=micro_batch_size,
            accumulation_steps=accumulation_steps,
            activation_checkpointing=activation_checkpointing,
        )
        checkpoint_path = output_dir / "checkpoint.pt"
        best_path = output_dir / "best_checkpoint.pt"
        best_brier = math.inf
        best_epoch = -1
        validation_curve: list[tuple[int, float]] = []
        validation_time_curve: list[tuple[int, float, float]] = []
        elapsed_before_resume = 0.0
        if resume_epoch:
            payload = load_resume_payload(torch, checkpoint_path)
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            scaler.load_state_dict(payload["scaler"])
            best_brier = float(payload["best_brier"])
            best_epoch = int(payload["best_epoch"])
            validation_curve = [
                (int(item[0]), float(item[1]))
                for item in payload.get("validation_curve", ())
            ]
            validation_time_curve = [
                (int(item[0]), float(item[1]), float(item[2]))
                for item in payload.get("validation_time_curve", ())
            ]
            elapsed_before_resume = float(payload.get("elapsed_seconds", 0.0))
            random.setstate(payload["python_rng"])
            np.random.set_state(payload["numpy_rng"])
            torch.set_rng_state(payload["torch_rng"])
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
            restore_adapter_checkpoint_state(
                adapter, payload.get("adapter_state"), model, device
            )

        prepare_initial_retrieval_cache(adapter, model, device)
        patience = int(request.training_config["patience"])
        stale_epochs = 0 if best_epoch < 0 else max(0, resume_epoch - 1 - best_epoch)
        deadline = _session_deadline()
        attempt_started = time.monotonic()
        budget_reached = False
        for epoch in range(resume_epoch, request.epochs):
            if epoch > 0 and session_deadline_reached(deadline):
                budget_reached = True
                break
            epoch_started = time.monotonic()
            on_adapter_epoch_start(adapter, model, epoch, device)
            model.train()
            order = np.random.default_rng(request.seed + epoch).permutation(
                len(request.train.row_id)
            )
            total_windows = math.ceil(len(order) / effective_batch)
            for window_number, window_start in enumerate(
                range(0, len(order), effective_batch), start=1
            ):
                if epoch > 0 and session_deadline_reached(deadline):
                    budget_reached = True
                    break
                window = order[window_start : window_start + effective_batch]
                n_micro = math.ceil(len(window) / micro_batch_size)
                optimizer.zero_grad(set_to_none=True)
                window_loss = None
                for start in range(0, len(window), micro_batch_size):
                    indices = window[start : start + micro_batch_size]
                    x_num = torch.as_tensor(
                        np.asarray(request.train.x_num[indices]),
                        dtype=torch.float32,
                        device=device,
                    )
                    x_cat = torch.as_tensor(
                        np.asarray(request.train.x_cat[indices]),
                        dtype=torch.long,
                        device=device,
                    )
                    y = torch.as_tensor(
                        np.asarray(request.train.y[indices]),
                        dtype=torch.float32,
                        device=device,
                    )
                    row_indices = torch.as_tensor(
                        indices, dtype=torch.long, device=device
                    )
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=amp_enabled
                    ):
                        loss = call_adapter_loss(
                            adapter,
                            model,
                            x_num,
                            x_cat,
                            y,
                            row_indices=row_indices,
                        ) / n_micro
                    scaler.scale(loss).backward()
                    detached_loss = loss.detach()
                    window_loss = (
                        detached_loss
                        if window_loss is None
                        else window_loss + detached_loss
                    )
                scaler.step(optimizer)
                scaler.update()
                if scheduler_mode == "update":
                    scheduler.step()
                if window_loss is None:
                    raise RuntimeError("training window did not contain a microbatch")
                report_training_window(
                    reporter=reporter,
                    torch=torch,
                    completed_rows=window_start + len(window),
                    total_rows=len(order),
                    started_at=epoch_started,
                    epoch=epoch,
                    epochs=request.epochs,
                    batch=window_number,
                    batches=total_windows,
                    loss=window_loss,
                )

            if budget_reached:
                reporter.emit(
                    "TRAINING_TIME_BUDGET_REACHED",
                    completed_epochs=len(validation_curve),
                )
                break

            refresh_retrieval_cache(adapter, model, device)
            predictions = self._predict(
                torch,
                request.valid,
                model,
                adapter,
                micro_batch_size,
                amp_enabled,
                device,
                reporter=reporter,
                stream=f"validation_epoch_{epoch}",
            )
            target = np.asarray(request.valid.y, dtype="float64")
            brier = float(np.mean(np.square(predictions - target)))
            require_finite_validation_brier(
                candidate_id=request.candidate_id, epoch=epoch, brier=brier
            )
            validation_curve.append((epoch, brier))
            elapsed_seconds = elapsed_before_resume + time.monotonic() - attempt_started
            validation_time_curve.append((epoch, elapsed_seconds, brier))
            improved = brier < best_brier
            if improved:
                best_brier = brier
                best_epoch = epoch
                stale_epochs = 0
                _atomic_torch_save(
                    torch, {"model": model.state_dict(), "epoch": epoch}, best_path
                )
            else:
                stale_epochs += 1
            if scheduler_mode == "metric":
                scheduler.step(brier)
            elif scheduler_mode == "epoch":
                scheduler.step()
            on_adapter_epoch_end(adapter, model, epoch, device)
            current_adapter_state = adapter_checkpoint_state(adapter)

            state = {
                "candidate_id": request.candidate_id,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_brier": best_brier,
                "validation_curve": validation_curve,
                "validation_time_curve": validation_time_curve,
                "elapsed_seconds": elapsed_seconds,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "adapter_state": current_adapter_state,
            }
            _atomic_torch_save(torch, state, checkpoint_path)
            _atomic_json(
                {
                    "candidate_id": request.candidate_id,
                    "epoch": epoch,
                    "checkpoint": checkpoint_path.name,
                    "adapter_state": current_adapter_state,
                },
                output_dir / "checkpoint_meta.json",
            )
            reporter.emit(
                "EPOCH_CHECKPOINTED",
                epoch=epoch,
                epochs=request.epochs,
                brier=brier,
                best_brier=best_brier,
                best_epoch=best_epoch,
                epoch_seconds=time.monotonic() - epoch_started,
                checkpoint=checkpoint_path,
            )
            if stale_epochs >= patience:
                break

        if best_epoch < 0 or not best_path.is_file():
            raise RuntimeError("training did not produce a valid checkpoint")
        best_payload = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best_payload["model"])
        refresh_retrieval_cache(adapter, model, device)
        predictions = self._predict(
            torch,
            request.valid,
            model,
            adapter,
            micro_batch_size,
            amp_enabled,
            device,
            reporter=reporter,
            stream="validation_best_checkpoint",
        )
        return BackendAttemptResult(
            best_epoch=best_epoch,
            best_brier=best_brier,
            checkpoint=best_path,
            predictions=predictions,
            hardware=hardware,
            completed_epochs=len(validation_curve),
            validation_curve=tuple(validation_curve),
            validation_time_curve=tuple(validation_time_curve),
        )

    @staticmethod
    def _predict(
        torch: object,
        batch: FeatureBatch,
        model: object,
        adapter: ModelAdapter,
        micro_batch_size: int,
        amp_enabled: bool,
        device: str,
        *,
        reporter: object | None = None,
        stream: str = "validation",
    ) -> np.ndarray:
        model.eval()
        outputs: list[np.ndarray] = []
        validation_started = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(batch.row_id), micro_batch_size):
                stop = start + micro_batch_size
                x_num = torch.as_tensor(
                    np.asarray(batch.x_num[start:stop]),
                    dtype=torch.float32,
                    device=device,
                )
                x_cat = torch.as_tensor(
                    np.asarray(batch.x_cat[start:stop]),
                    dtype=torch.long,
                    device=device,
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=amp_enabled
                ):
                    probabilities = adapter.probabilities(model, x_num, x_cat)
                outputs.append(
                    probabilities.detach().to(dtype=torch.float64, device="cpu").numpy()
                )
                completed_rows = min(stop, len(batch.row_id))
                if reporter is not None and reporter.should_emit(
                    completed_rows=completed_rows, stream=stream
                ):
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    reporter.progress(
                        "VALIDATION_PROGRESS",
                        completed_rows=completed_rows,
                        total_rows=len(batch.row_id),
                        started_at=validation_started,
                        gpu=_gpu_memory(torch),
                        stream=stream,
                    )
        return np.concatenate(outputs).reshape(-1)
