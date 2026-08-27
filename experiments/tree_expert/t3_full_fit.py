from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .e2_full_fit import export_frozen_tree_state
from .features import TreeFeatureBatch, fit_tree_features
from .t3_contracts import load_t3_contract
from .t3_decisions import T3AcceptanceDecision, acceptance_payload
from .t3_temporal import temporal_training_weights


class T3FullFitError(ValueError):
    pass


class TrainableModel(Protocol):
    def fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs: object) -> object: ...
    def save_model(self, path: str) -> None: ...


ModelFactory = Callable[[dict[str, object]], TrainableModel]
FeatureBuilder = Callable[..., tuple[object, TreeFeatureBatch]]
StateExporter = Callable[..., Path]


@dataclass(frozen=True)
class T3FullFitResult:
    root: Path
    frozen_state: Path
    model_paths: Mapping[str, Path]
    manifest_path: Path


def _default_model_factory(parameters: dict[str, object]) -> TrainableModel:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise T3FullFitError("catboost==1.2.10 is required") from error
    return CatBoostRegressor(**parameters)


def full_fit_iterations(best_iterations: Sequence[int]) -> int:
    values = tuple(best_iterations)
    if len(values) != 3 or any(type(value) is not int or value < 0 for value in values):
        raise T3FullFitError("three fold iterations are required")
    return min(400, max(50, int(statistics.median(values)) + 1))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _save_model(model: TrainableModel, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        model.save_model(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise T3FullFitError("full-fit model is empty")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _iteration_map(evidence: Mapping[str, Mapping[int, Sequence[int]]]) -> dict[str, dict[int, int]]:
    if set(evidence) != {"recent", "multi"}:
        raise T3FullFitError("iteration evidence heads differ")
    result: dict[str, dict[int, int]] = {}
    for head in ("recent", "multi"):
        if set(evidence[head]) != {42, 2026, 3407}:
            raise T3FullFitError("iteration evidence seeds differ")
        result[head] = {
            seed: full_fit_iterations(tuple(evidence[head][seed]))
            for seed in (42, 2026, 3407)
        }
    return result


def full_fit_t3(
    *,
    decision: T3AcceptanceDecision,
    train: pd.DataFrame,
    output_dir: Path,
    iteration_evidence: Mapping[str, Mapping[int, Sequence[int]]],
    gpu_ids: tuple[int, int] = (0, 1),
    model_factory: ModelFactory = _default_model_factory,
    feature_builder: FeatureBuilder = fit_tree_features,
    state_exporter: StateExporter = export_frozen_tree_state,
) -> T3FullFitResult:
    if type(decision) is not T3AcceptanceDecision or decision.status != "accepted":
        raise T3FullFitError("candidate is not accepted")
    if type(train) is not pd.DataFrame or train.empty or "season" not in train or "control_success" not in train:
        raise T3FullFitError("full-fit training rows differ")
    seasons = pd.to_numeric(train["season"], errors="raise")
    if int(seasons.min()) != 2019 or int(seasons.max()) != 2024:
        raise T3FullFitError("full-fit seasons differ")
    if len(gpu_ids) != 2 or any(type(item) is not int or item < 0 for item in gpu_ids):
        raise T3FullFitError("full-fit GPU ids differ")
    iterations = _iteration_map(iteration_evidence)
    state, batch = feature_builder(train, None, valid_year=2025, use_trackman=False)
    if batch.target is None or len(batch.target) != len(train):
        raise T3FullFitError("full-fit target differs")
    weights = {
        "recent": temporal_training_weights(seasons, valid_year=2025, head="recent"),
        "multi": temporal_training_weights(seasons, valid_year=2025, head="multi", decay=decision.decay),
    }
    output = Path(output_dir)
    if output.exists() and (output.is_symlink() or any(output.iterdir())):
        raise T3FullFitError("full-fit output is not empty")
    output.mkdir(parents=True, exist_ok=True)
    frozen = state_exporter(state, output / "frozen_state", candidate_id="c1_anchor_residual")
    model_root = output / "models"
    model_root.mkdir()
    active = load_t3_contract()
    model_paths: dict[str, Path] = {}
    residual = np.asarray(batch.target, dtype="float64") - np.asarray(batch.anchor, dtype="float64")
    for head in ("recent", "multi"):
        selected = weights[head] > 0
        for index, seed in enumerate((42, 2026, 3407)):
            parameters = dict(active.catboost)
            parameters.update(
                iterations=iterations[head][seed], random_seed=seed,
                devices=str(gpu_ids[index % len(gpu_ids)]),
                train_dir=str(output / f"catboost_info/{head}_{seed}"),
                loss_function="RMSE", eval_metric="RMSE",
            )
            model = model_factory(parameters)
            model.fit(
                batch.frame.loc[selected], residual[selected],
                sample_weight=weights[head][selected],
                cat_features=list(getattr(state, "categorical_columns", ())),
                use_best_model=False, verbose=50,
            )
            name = f"{head}_seed_{seed}.cbm"
            path = model_root / name
            _save_model(model, path)
            model_paths[f"{head}:{seed}"] = path
    manifest = {
        "schema_version": 1, "artifact_kind": "tree_expert_t3_full_fit_v1",
        "candidate_id": "t3_temporal_dual", "status": "accepted",
        "decision": acceptance_payload(decision), "iterations": iterations,
        "models": {
            key: {"path": path.relative_to(output).as_posix(), "sha256": _sha(path), "size": path.stat().st_size}
            for key, path in sorted(model_paths.items())
        },
    }
    manifest_path = output / "full_fit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return T3FullFitResult(output, frozen, MappingProxyType(model_paths), manifest_path)
