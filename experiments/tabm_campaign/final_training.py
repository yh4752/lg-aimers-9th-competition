from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd


class FinalTrainingError(RuntimeError):
    """Raised when official-train-only fixed-epoch fitting cannot finish safely."""


def save_epoch_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_epoch_checkpoint(
    path: Path,
    expected_identity: Mapping[str, str],
) -> dict[str, object]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise FinalTrainingError("checkpoint schema differs")
    if value.get("identity") != dict(expected_identity):
        raise FinalTrainingError("checkpoint identity differs")
    completed = value.get("completed_epochs")
    if isinstance(completed, bool) or not isinstance(completed, int) or not 1 <= completed <= 3:
        raise FinalTrainingError("checkpoint completed epoch differs")
    return value


def resume_start_epoch(
    checkpoint: Mapping[str, object] | None,
    epochs: int,
) -> int:
    completed = 0 if checkpoint is None else checkpoint.get("completed_epochs")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= epochs
    ):
        raise FinalTrainingError("checkpoint epoch is outside final epoch range")
    return completed


def validate_final_fit_policy(
    candidate: Mapping[str, object],
    *,
    seed: int,
    epochs: int,
) -> None:
    if seed != 3407:
        raise FinalTrainingError("final seed differs")
    if epochs != 3:
        raise FinalTrainingError("final epoch count differs")
    if candidate.get("scheduler") != "constant":
        raise FinalTrainingError("final scheduler differs")


def resolve_final_candidate(
    selected_candidate: Mapping[str, object],
    final_fit: Mapping[str, object],
) -> dict[str, object]:
    if selected_candidate.get("seed") != 3407:
        raise FinalTrainingError("selected candidate seed differs")
    if selected_candidate.get("scheduler") != "plateau":
        raise FinalTrainingError("selected candidate scheduler differs")
    if final_fit.get("epochs") != 3 or final_fit.get("scheduler") != "constant":
        raise FinalTrainingError("final fit policy differs")
    if final_fit.get("learning_rate") != selected_candidate.get("learning_rate"):
        raise FinalTrainingError("final learning rate differs")
    if (
        final_fit.get("weight_decay") != 0.0001
        or final_fit.get("effective_batch_size") != 4096
        or final_fit.get("micro_batch_size") != 512
    ):
        raise FinalTrainingError("final optimizer policy differs")
    return {**selected_candidate, "scheduler": "constant"}


def build_inference_manifest(
    *,
    row_count: int,
    preprocessing_state: str,
    member: Mapping[str, object],
    files: Mapping[str, str],
    identity: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "fit_scope": "official_train_2019_2024_only",
        "row_count": row_count,
        "epochs": 3,
        "seeds": [3407],
        "scheduler": "constant",
        "preprocessing_state": preprocessing_state,
        "members": [dict(member)],
        "files": dict(files),
        "identity": dict(identity),
    }


def _find_train(root: Path) -> Path:
    direct = root / "train.csv"
    candidates = [direct] if direct.is_file() else sorted(root.rglob("train.csv"))
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise FinalTrainingError(f"official train.csv candidate count must be 1; found={len(candidates)}")
    return candidates[0]


def _atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _restore_rng(checkpoint: Mapping[str, object], torch: object) -> None:
    python_rng = checkpoint.get("python_rng")
    numpy_rng = checkpoint.get("numpy_rng")
    torch_rng = checkpoint.get("torch_rng")
    cuda_rng = checkpoint.get("cuda_rng")
    if python_rng is not None:
        random.setstate(python_rng)
    if numpy_rng is not None:
        np.random.set_state(numpy_rng)
    if torch_rng is not None:
        torch.set_rng_state(torch_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)


def fit_final_member(
    *,
    data_dir: Path,
    artifact_root: Path,
    candidate: Mapping[str, object],
    seed: int,
    epochs: int,
    absolute_deadline: float,
    checkpoint_path: Path,
    checkpoint_identity: Mapping[str, str],
    on_epoch_checkpoint: Callable[[Path, int], None] | None = None,
) -> dict[str, object]:
    """Fit the sealed member on official 2019-2024 training rows."""

    validate_final_fit_policy(candidate, seed=seed, epochs=epochs)
    required_identity = {
        "contract_sha256",
        "data_archive_sha256",
        "train_sha256",
        "runtime_sha256",
        "training_source_sha256",
    }
    if set(checkpoint_identity) != required_identity or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in checkpoint_identity.values()
    ):
        raise FinalTrainingError("final checkpoint identity differs")
    import torch

    from experiments.independent_dl.features import (
        PreprocessedFeatureState,
        _fit_categories,
        _preprocessed_batch,
        _preprocessed_state_payload,
    )
    from experiments.independent_dl.models.common import metadata_from_train
    from experiments.independent_dl.models.tabm import TabMAdapter
    from experiments.independent_dl.preprocessing import PreprocessingSpec, fit_preprocessor
    from .model_state import fit_numeric_embedding_state, save_numeric_embedding_state
    from .version_d import file_sha256

    if not torch.cuda.is_available():
        raise FinalTrainingError("Version D final fitting requires a CUDA GPU")
    train = pd.read_csv(_find_train(data_dir))
    seasons = pd.to_numeric(train["season"], errors="raise").astype("int64")
    if train.empty or seasons.min() < 2019 or seasons.max() > 2024:
        raise FinalTrainingError("official training seasons must stay inside 2019-2024")
    spec = PreprocessingSpec("dl_standard", ("hand_matchup",))
    preprocessing, prepared = fit_preprocessor(train, spec)
    maps = _fit_categories(prepared, preprocessing.categorical_columns)
    state = PreprocessedFeatureState(
        "raw_typed",
        2024,
        preprocessing.numeric_columns,
        preprocessing.categorical_columns,
        maps,
        preprocessing,
        None,
        None,
    )
    batch = _preprocessed_batch(train, prepared, state)
    artifact_root.mkdir(parents=True, exist_ok=True)
    state_path = artifact_root / "preprocessing_state.json"
    _atomic_json(state_path, _preprocessed_state_payload(state))

    started = time.monotonic()
    model_config = {
        "architecture": "tabm",
        "k": int(candidate["k"]),
        "width": int(candidate["width"]),
        "blocks": int(candidate["blocks"]),
        "dropout": float(candidate["dropout"]),
        "num_embedding": str(candidate["num_embedding"]),
    }
    numeric_state = fit_numeric_embedding_state(
        str(candidate["num_embedding"]), np.asarray(batch.x_num)
    )
    numeric_path = artifact_root / "numeric_embedding_0.json"
    save_numeric_embedding_state(numeric_state, numeric_path)
    preprocessing_sha = file_sha256(state_path)
    numeric_sha = file_sha256(numeric_path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    adapter = TabMAdapter(loss_name=str(candidate["loss"]))
    model = adapter.build(model_config, metadata_from_train(batch), "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=0.0001,
    )
    effective, micro = 4096, 512
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    checkpoint: dict[str, object] | None = None
    if checkpoint_path.is_file():
        checkpoint = load_epoch_checkpoint(checkpoint_path, checkpoint_identity)
        if (
            checkpoint.get("preprocessing_sha256") != preprocessing_sha
            or checkpoint.get("numeric_embedding_sha256") != numeric_sha
        ):
            raise FinalTrainingError("checkpoint preprocessing state differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint, torch)
    start_epoch = resume_start_epoch(checkpoint, epochs)
    for epoch in range(start_epoch, epochs):
        if time.time() >= absolute_deadline:
            raise FinalTrainingError("Version D absolute training deadline reached")
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(batch.row_id))
        for window_start in range(0, len(order), effective):
            if time.time() >= absolute_deadline:
                raise FinalTrainingError("Version D absolute training deadline reached")
            window = order[window_start : window_start + effective]
            optimizer.zero_grad(set_to_none=True)
            n_micro = math.ceil(len(window) / micro)
            for start in range(0, len(window), micro):
                indices = window[start : start + micro]
                x_num = torch.as_tensor(
                    np.asarray(batch.x_num[indices]).copy(),
                    dtype=torch.float32,
                    device="cuda",
                )
                x_cat = torch.as_tensor(
                    np.asarray(batch.x_cat[indices]).copy(),
                    dtype=torch.long,
                    device="cuda",
                )
                target = torch.as_tensor(
                    np.asarray(batch.y[indices]).copy(),
                    dtype=torch.float32,
                    device="cuda",
                )
                row_indices = torch.as_tensor(
                    indices.copy(), dtype=torch.long, device="cuda"
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    loss = adapter.loss(
                        model,
                        x_num,
                        x_cat,
                        target,
                        row_indices=row_indices,
                    ) / n_micro
                if not torch.isfinite(loss):
                    raise FinalTrainingError(
                        f"non-finite final loss seed={seed} epoch={epoch}"
                    )
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        save_epoch_checkpoint(
            checkpoint_path,
            {
                "schema_version": 1,
                "identity": dict(checkpoint_identity),
                "completed_epochs": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "preprocessing_sha256": preprocessing_sha,
                "numeric_embedding_sha256": numeric_sha,
            },
        )
        print(
            f"FINAL_TRAINING_PROGRESS seed={seed} epoch={epoch + 1}/{epochs}",
            flush=True,
        )
        if on_epoch_checkpoint is not None:
            on_epoch_checkpoint(checkpoint_path, epoch + 1)

    weights = artifact_root / f"tabm_member_0_seed_{seed}.pt"
    temporary = weights.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, weights)
    member = {
        "seed": seed,
        "weights": weights.name,
        "numeric_state": numeric_path.name,
        "model_config": model_config,
    }
    files = {
        path.name: file_sha256(path)
        for path in (state_path, numeric_path, weights)
    }
    manifest = build_inference_manifest(
        row_count=len(train),
        preprocessing_state=state_path.name,
        member=member,
        files=files,
        identity=checkpoint_identity,
    )
    _atomic_json(artifact_root / "inference_manifest.json", manifest)
    return {
        "status": "completed",
        "rows": len(train),
        "epochs": epochs,
        "seeds": [seed],
        "scheduler": "constant",
        "resumed_from_epoch": start_epoch,
        "elapsed_seconds": time.monotonic() - started,
        "manifest_sha256": sha256((artifact_root / "inference_manifest.json").read_bytes()).hexdigest(),
    }
