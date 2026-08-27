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
from .rf_contracts import RFContract, load_rf_contract
from .rf_decisions import RFAcceptanceDecision, acceptance_payload


class RFFullFitError(ValueError):
    pass


class TrainableModel(Protocol):
    def fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs: object) -> object: ...
    def save_model(self, path: str) -> None: ...


ModelFactory = Callable[[dict[str, object]], TrainableModel]
FeatureBuilder = Callable[..., tuple[object, TreeFeatureBatch]]
StateExporter = Callable[..., Path]


@dataclass(frozen=True)
class RFFullFitToken:
    f_head: str
    include_r: bool
    alpha_r: float
    alpha_f: float
    seeds: tuple[int, ...]
    iterations: Mapping[str, Mapping[int, int]]


@dataclass(frozen=True)
class RFFullFitResult:
    root: Path
    frozen_states: Mapping[str, Path]
    model_paths: Mapping[str, Path]
    manifest_path: Path
    token: RFFullFitToken


def _default_model_factory(parameters: dict[str, object]) -> TrainableModel:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise RFFullFitError("catboost==1.2.10 is required") from error
    return CatBoostRegressor(**parameters)


def full_fit_iterations(best_iterations: Sequence[int], *, maximum: int) -> int:
    values = tuple(best_iterations)
    if (
        len(values) != 3
        or type(maximum) is not int
        or maximum <= 0
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise RFFullFitError("three fold iterations are required")
    return min(maximum, max(1, int(statistics.median(values)) + 1))


def create_rf_full_fit_token(
    decision: RFAcceptanceDecision,
    iteration_evidence: Mapping[str, Mapping[int, Sequence[int]]],
    contract: RFContract | None = None,
) -> RFFullFitToken:
    if type(decision) is not RFAcceptanceDecision or decision.status != "accepted":
        raise RFFullFitError("accepted decision is required")
    active = load_rf_contract() if contract is None else contract
    heads = (decision.f_head, "r_expert") if decision.include_r else (decision.f_head,)
    if set(iteration_evidence) != set(heads):
        raise RFFullFitError("iteration evidence heads differ")
    seeds = tuple(sorted({active.structure_seed, *active.confirmation_seeds}))
    iterations: dict[str, Mapping[int, int]] = {}
    for head in heads:
        if set(iteration_evidence[head]) != set(seeds):
            raise RFFullFitError("iteration evidence seeds differ")
        maximum = int(active.experts[head]["iterations"])
        iterations[head] = MappingProxyType(
            {
                seed: full_fit_iterations(iteration_evidence[head][seed], maximum=maximum)
                for seed in seeds
            }
        )
    return RFFullFitToken(
        f_head=decision.f_head,
        include_r=decision.include_r,
        alpha_r=decision.alpha_r,
        alpha_f=decision.alpha_f,
        seeds=seeds,
        iterations=MappingProxyType(iterations),
    )


def _save_model(model: TrainableModel, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        model.save_model(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RFFullFitError("full-fit model is empty")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def full_fit_rf(
    *,
    decision: RFAcceptanceDecision,
    train: pd.DataFrame,
    output_dir: Path,
    iteration_evidence: Mapping[str, Mapping[int, Sequence[int]]],
    gpu_ids: tuple[int, int] = (0, 1),
    contract: RFContract | None = None,
    model_factory: ModelFactory = _default_model_factory,
    feature_builder: FeatureBuilder = fit_tree_features,
    state_exporter: StateExporter = export_frozen_tree_state,
) -> RFFullFitResult:
    active = load_rf_contract() if contract is None else contract
    token = create_rf_full_fit_token(decision, iteration_evidence, active)
    required = {"row_id", "season", "game_type", "control_success"}
    if type(train) is not pd.DataFrame or train.empty or not required.issubset(train.columns):
        raise RFFullFitError("full-fit training rows differ")
    if not train["game_type"].astype(str).isin(["R", "F"]).all():
        raise RFFullFitError("full-fit game_type differs")
    if len(gpu_ids) != 2 or any(type(item) is not int or item < 0 for item in gpu_ids):
        raise RFFullFitError("full-fit GPU ids differ")
    output = Path(output_dir)
    if output.exists() and (output.is_symlink() or any(output.iterdir())):
        raise RFFullFitError("full-fit output is not empty")
    output.mkdir(parents=True, exist_ok=True)
    heads = (token.f_head, "r_expert") if token.include_r else (token.f_head,)
    frozen_states: dict[str, Path] = {}
    model_paths: dict[str, Path] = {}
    for head in heads:
        segment = str(active.experts[head]["segment"])
        rows = train.loc[train["game_type"].astype(str).eq(segment)].copy()
        if len(rows) < active.gates.minimum_segment_rows:
            raise RFFullFitError("full-fit segment rows are below minimum")
        state, batch = feature_builder(rows, None, valid_year=2025, use_trackman=False)
        if batch.target is None or len(batch.target) != len(rows):
            raise RFFullFitError("full-fit target differs")
        head_root = output / head
        frozen = state_exporter(
            state,
            head_root / "frozen_state",
            candidate_id="c1_anchor_residual",
        )
        frozen_states[head] = frozen
        model_root = head_root / "models"
        model_root.mkdir(parents=True)
        residual = np.asarray(batch.target, dtype="float64") - np.asarray(batch.anchor, dtype="float64")
        for index, seed in enumerate(token.seeds):
            parameters = dict(active.catboost_common)
            parameters.update(active.experts[head])
            parameters.pop("segment", None)
            parameters.update(
                iterations=token.iterations[head][seed],
                random_seed=seed,
                devices=str(gpu_ids[index % len(gpu_ids)]),
                train_dir=str(head_root / f"catboost_info/seed_{seed}"),
                loss_function="RMSE",
                eval_metric="RMSE",
            )
            model = model_factory(parameters)
            model.fit(
                batch.frame,
                residual,
                cat_features=list(getattr(state, "categorical_columns", ())),
                use_best_model=False,
                verbose=50,
            )
            path = model_root / f"seed_{seed}.cbm"
            _save_model(model, path)
            model_paths[f"{head}:{seed}"] = path
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tree_expert_rf_full_fit_v1",
        "status": "accepted",
        "decision": acceptance_payload(decision),
        "token": {
            "f_head": token.f_head,
            "include_r": token.include_r,
            "alpha_r": token.alpha_r,
            "alpha_f": token.alpha_f,
            "seeds": list(token.seeds),
            "iterations": {
                head: {str(seed): count for seed, count in values.items()}
                for head, values in token.iterations.items()
            },
        },
        "models": {
            key: {
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha(path),
                "size": path.stat().st_size,
            }
            for key, path in sorted(model_paths.items())
        },
    }
    manifest_path = output / "full_fit_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return RFFullFitResult(
        root=output,
        frozen_states=MappingProxyType(frozen_states),
        model_paths=MappingProxyType(model_paths),
        manifest_path=manifest_path,
        token=token,
    )
