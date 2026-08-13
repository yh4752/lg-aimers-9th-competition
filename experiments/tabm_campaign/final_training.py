from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


class FinalTrainingError(RuntimeError):
    """Raised when official-train-only fixed-epoch fitting cannot finish safely."""


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


def fit_final_members(
    *,
    data_dir: Path,
    artifact_root: Path,
    champion: Mapping[str, object],
    member_seeds: list[int],
    epochs: int,
) -> dict[str, object]:
    """Fit only on official 2019-2024 training rows for a fixed epoch count."""

    if not 1 <= len(member_seeds) <= 3 or not 2 <= epochs <= 40:
        raise FinalTrainingError("final training requires 1-3 seeds and 2-40 epochs")
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

    model_config = {
        "architecture": "tabm",
        "k": int(champion["k"]),
        "width": int(champion["width"]),
        "blocks": int(champion["blocks"]),
        "dropout": float(champion["dropout"]),
        "num_embedding": str(champion["num_embedding"]),
    }
    numeric_state = fit_numeric_embedding_state(str(champion["num_embedding"]), np.asarray(batch.x_num))
    numeric_path = artifact_root / "numeric_embedding.json"
    save_numeric_embedding_state(numeric_state, numeric_path)
    members: list[dict[str, object]] = []
    started = time.monotonic()
    for seed in member_seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        adapter = TabMAdapter(loss_name=str(champion["loss"]))
        model = adapter.build(model_config, metadata_from_train(batch), "cuda")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(champion["learning_rate"]),
            weight_decay=0.0001,
        )
        effective, micro = 4096, 512
        updates_per_epoch = math.ceil(len(batch.row_id) / effective)
        scheduler_name = str(champion["scheduler"])
        scheduler = (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(champion["learning_rate"]),
                total_steps=epochs * updates_per_epoch,
            )
            if scheduler_name == "one_cycle"
            else None
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        for epoch in range(epochs):
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(batch.row_id))
            for window_start in range(0, len(order), effective):
                window = order[window_start : window_start + effective]
                optimizer.zero_grad(set_to_none=True)
                n_micro = math.ceil(len(window) / micro)
                for start in range(0, len(window), micro):
                    indices = window[start : start + micro]
                    x_num = torch.as_tensor(np.asarray(batch.x_num[indices]), dtype=torch.float32, device="cuda")
                    x_cat = torch.as_tensor(np.asarray(batch.x_cat[indices]), dtype=torch.long, device="cuda")
                    target = torch.as_tensor(np.asarray(batch.y[indices]), dtype=torch.float32, device="cuda")
                    row_indices = torch.as_tensor(indices, dtype=torch.long, device="cuda")
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        loss = adapter.loss(model, x_num, x_cat, target, row_indices=row_indices) / n_micro
                    if not torch.isfinite(loss):
                        raise FinalTrainingError(f"non-finite final loss seed={seed} epoch={epoch}")
                    scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            print(f"FINAL_TRAINING_PROGRESS seed={seed} epoch={epoch + 1}/{epochs}", flush=True)
        weights = artifact_root / f"tabm_seed_{seed}.pt"
        temporary = weights.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, weights)
        members.append(
            {
                "seed": seed,
                "weights": weights.name,
                "numeric_state": numeric_path.name,
                "model_config": model_config,
            }
        )

    files = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in artifact_root.iterdir()
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "fit_scope": "official_train_2019_2024_only",
        "row_count": len(train),
        "preprocessing_state": state_path.name,
        "members": members,
        "files": files,
    }
    _atomic_json(artifact_root / "inference_manifest.json", manifest)
    return {
        "status": "completed",
        "rows": len(train),
        "epochs": epochs,
        "seeds": member_seeds,
        "elapsed_seconds": time.monotonic() - started,
        "manifest_sha256": sha256((artifact_root / "inference_manifest.json").read_bytes()).hexdigest(),
    }
