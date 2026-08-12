"""Shared restartable trainer for independent DL candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import time
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from .features import FeatureBatch
from .models.common import ModelAdapter, import_runtime_module, metadata_from_train


class TrainingContractError(ValueError):
    """Raised before a malformed training request can start GPU work."""


class TrainingTimeBudgetReached(RuntimeError):
    """Raised after preserving the last complete epoch checkpoint."""


def enforce_session_deadline(deadline: float | None, *, boundary: str) -> None:
    """Stop before more GPU work when the user-owned session budget expires."""

    if deadline is not None and time.time() >= deadline:
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
    started_epoch = _resume_epoch(root, request.candidate_id)
    current_resume_epoch = started_epoch
    runtime = TorchTrainingBackend() if backend is None else backend
    attempted: list[int] = []
    activation_checkpointing = False

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
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise TrainingContractError("validation predictions must be finite probabilities")
    if not math.isfinite(float(attempt.best_brier)):
        raise TrainingContractError("best_brier must be finite")
    return TrainResult(
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
    )


def _atomic_torch_save(torch: object, payload: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
        checkpoint_path = output_dir / "checkpoint.pt"
        best_path = output_dir / "best_checkpoint.pt"
        best_brier = math.inf
        best_epoch = -1
        if resume_epoch:
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            scaler.load_state_dict(payload["scaler"])
            best_brier = float(payload["best_brier"])
            best_epoch = int(payload["best_epoch"])
            random.setstate(payload["python_rng"])
            np.random.set_state(payload["numpy_rng"])
            torch.set_rng_state(payload["torch_rng"])
            torch.cuda.set_rng_state_all(payload["cuda_rng"])

        refresh_retrieval_cache(adapter, model, device)
        patience = int(request.training_config["patience"])
        stale_epochs = 0 if best_epoch < 0 else max(0, resume_epoch - 1 - best_epoch)
        last_epoch = resume_epoch - 1
        deadline = _session_deadline()
        for epoch in range(resume_epoch, request.epochs):
            if epoch > 0:
                enforce_session_deadline(deadline, boundary=f"before_epoch_{epoch + 1}")
            last_epoch = epoch
            epoch_started = time.monotonic()
            model.train()
            order = np.random.default_rng(request.seed + epoch).permutation(
                len(request.train.row_id)
            )
            total_windows = math.ceil(len(order) / effective_batch)
            heartbeat_every = max(1, total_windows // 20)
            for window_number, window_start in enumerate(
                range(0, len(order), effective_batch), start=1
            ):
                if epoch > 0:
                    enforce_session_deadline(
                        deadline,
                        boundary=f"epoch_{epoch + 1}_batch_{window_number}",
                    )
                window = order[window_start : window_start + effective_batch]
                n_micro = math.ceil(len(window) / micro_batch_size)
                optimizer.zero_grad(set_to_none=True)
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
                scaler.step(optimizer)
                scaler.update()
                if scheduler_mode == "update":
                    scheduler.step()
                if (
                    window_number == 1
                    or window_number == total_windows
                    or window_number % heartbeat_every == 0
                ):
                    print(
                        progress_message(
                            candidate_id=request.candidate_id,
                            epoch=epoch,
                            epochs=request.epochs,
                            batch=window_number,
                            batches=total_windows,
                            elapsed_seconds=time.monotonic() - epoch_started,
                        ),
                        flush=True,
                    )

            refresh_retrieval_cache(adapter, model, device)
            predictions = self._predict(
                torch,
                request.valid,
                model,
                adapter,
                micro_batch_size,
                amp_enabled,
                device,
            )
            target = np.asarray(request.valid.y, dtype="float64")
            brier = float(np.mean(np.square(predictions - target)))
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

            state = {
                "candidate_id": request.candidate_id,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_brier": best_brier,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }
            _atomic_torch_save(torch, state, checkpoint_path)
            _atomic_json(
                {
                    "candidate_id": request.candidate_id,
                    "epoch": epoch,
                    "checkpoint": checkpoint_path.name,
                },
                output_dir / "checkpoint_meta.json",
            )
            print(
                "EPOCH_CHECKPOINTED "
                f"candidate={request.candidate_id} epoch={epoch + 1}/{request.epochs} "
                f"brier={brier:.10f} best_brier={best_brier:.10f} "
                f"elapsed_seconds={round(time.monotonic() - epoch_started)}",
                flush=True,
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
        )
        del last_epoch
        return BackendAttemptResult(
            best_epoch=best_epoch,
            best_brier=best_brier,
            checkpoint=best_path,
            predictions=predictions,
            hardware=hardware,
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
    ) -> np.ndarray:
        model.eval()
        outputs: list[np.ndarray] = []
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
        return np.concatenate(outputs).reshape(-1)
