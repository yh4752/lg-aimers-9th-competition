"""Accepted-only full-data fitting for privileged residual students."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from experiments.tree_expert.e2_full_fit import export_frozen_tree_state
from experiments.tree_expert.inputs import VerifiedOfficialData, file_sha256

from .contracts import load_contract
from .decisions import CandidateDecision
from .features import CandidateFeatureBatch, CandidateFeatureState, fit_candidate_features
from .profiles import ProfileState, ProfileStrengths


class PrivilegedFullFitError(ValueError):
    pass


@dataclass(frozen=True)
class FullFitToken:
    decision: CandidateDecision
    bindings: Mapping[str, str]
    seeds: tuple[int, ...]
    iterations: Mapping[int, int]
    strengths: ProfileStrengths


@dataclass(frozen=True)
class FullFitSources:
    data: VerifiedOfficialData
    e2_delivery: Path
    e2_handoff_sha256: str


@dataclass(frozen=True)
class FullFitResult:
    candidate_id: str
    root: Path
    models: Mapping[int, Path]
    frozen_state: Path
    nested_e2_delivery: Path
    manifest: Path
    state: CandidateFeatureState
    model_objects: tuple[object, ...]


def _expected_bindings() -> dict[str, str]:
    contract = load_contract()
    return {
        "official_train_sha256": contract.inputs["official_train_sha256"],
        "official_history_sha256": contract.inputs["official_history_sha256"],
        "e2_handoff_sha256": contract.inputs["e2_handoff_sha256"],
    }


def accepted_full_fit_token(
    decision: CandidateDecision,
    *,
    bindings: Mapping[str, str],
    best_iterations: Mapping[int, tuple[int, int, int]],
    strengths: ProfileStrengths,
) -> FullFitToken:
    if type(decision) is not CandidateDecision or decision.status != "accepted":
        raise PrivilegedFullFitError("accepted decision is required")
    if dict(bindings) != _expected_bindings():
        raise PrivilegedFullFitError("decision bindings differ")
    if set(best_iterations) != {42, 2026, 3407}:
        raise PrivilegedFullFitError("full-fit seed evidence differs")
    iterations = {}
    for seed, values in best_iterations.items():
        if len(values) != 3 or any(type(value) is not int or value < 0 for value in values):
            raise PrivilegedFullFitError("best-iteration evidence differs")
        iterations[seed] = max(50, min(1200, int(np.median(values)) + 1))
    return FullFitToken(decision, MappingProxyType(dict(bindings)), (42, 2026, 3407),
                        MappingProxyType(iterations), strengths)


def _profile_payload(state: ProfileState | None) -> bytes:
    if state is None:
        return b"null"
    value = {
        "cutoff_year": state.cutoff_year,
        "strengths": {"identity": state.strengths.identity, "interaction": state.strengths.interaction,
                      "matchup": state.strengths.matchup},
        "global_count": state.global_count, "global_rate": state.global_rate,
        "tables": {name: {"keys": list(table.keys), "rows": [list(row) for row in table.rows]}
                   for name, table in state.tables.items()},
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _default_factory(parameters: dict[str, object]):
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise PrivilegedFullFitError("catboost==1.2.10 is required") from error
    return CatBoostRegressor(**parameters)


def fit_accepted_candidate(
    token: FullFitToken,
    sources: FullFitSources,
    output_dir: Path,
    *,
    gpu_ids: tuple[int, ...] = (0, 1),
    model_factory: Callable[[dict[str, object]], object] = _default_factory,
    feature_builder=fit_candidate_features,
) -> FullFitResult:
    if type(token) is not FullFitToken or token.decision.status != "accepted":
        raise PrivilegedFullFitError("accepted decision is required")
    if dict(token.bindings) != _expected_bindings():
        raise PrivilegedFullFitError("decision bindings differ")
    if type(sources) is not FullFitSources:
        raise PrivilegedFullFitError("full-fit sources differ")
    if (sources.data.train_sha256 != token.bindings["official_train_sha256"]
            or sources.data.history_sha256 != token.bindings["official_history_sha256"]
            or sources.e2_handoff_sha256 != token.bindings["e2_handoff_sha256"]):
        raise PrivilegedFullFitError("source bindings differ")
    if Path(sources.e2_delivery).is_symlink() or not Path(sources.e2_delivery).is_file():
        raise PrivilegedFullFitError("accepted E2 delivery is absent")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise PrivilegedFullFitError("full-fit output is not empty")
    root.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(sources.data.train); history = pd.read_csv(sources.data.history)
    if rows.empty or int(rows["season"].max()) != 2024:
        raise PrivilegedFullFitError("full-fit seasons differ")
    state, batch = feature_builder(rows, history, valid_year=2025,
                                   candidate_id=token.decision.candidate_id, strengths=token.strengths)
    if type(batch) is not CandidateFeatureBatch or batch.soft_target is None:
        raise PrivilegedFullFitError("full-fit soft target differs")
    models: dict[int, Path] = {}
    model_objects_by_seed: dict[int, object] = {}
    contract = load_contract().catboost
    def train_seed(index_seed: tuple[int, int]) -> tuple[int, Path, object]:
        index, seed = index_seed
        parameters = {
            "iterations": token.iterations[seed], "depth": contract.depth,
            "learning_rate": float(contract.learning_rate), "l2_leaf_reg": float(contract.l2_leaf_reg),
            "random_strength": float(contract.random_strength), "bagging_temperature": float(contract.bagging_temperature),
            "border_count": contract.border_count, "max_ctr_complexity": contract.max_ctr_complexity,
            "loss_function": "RMSE", "random_seed": seed, "task_type": "GPU",
            "devices": str(gpu_ids[index % len(gpu_ids)]), "allow_writing_files": False, "verbose": 50,
        }
        model = model_factory(parameters)
        model.fit(batch.frame, batch.soft_target - batch.anchor,
                  cat_features=list(state.categorical_columns), use_best_model=False)
        path = root / f"student_seed_{seed}.cbm"; temporary = root / f".student_seed_{seed}.tmp"
        model.save_model(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise PrivilegedFullFitError("full-fit model output is empty")
        os.replace(temporary, path)
        return seed, path, model
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        for seed, path, model in pool.map(train_seed, enumerate(token.seeds)):
            models[seed] = path
            model_objects_by_seed[seed] = model
    frozen = root / "frozen_state"; frozen.mkdir()
    tree_root = frozen / "tree_state"
    export_frozen_tree_state(state.tree_state, tree_root, candidate_id="c1_anchor_residual")
    (frozen / "profile_state.json").write_bytes(_profile_payload(state.profile_state))
    (frozen / "candidate_state.json").write_text(json.dumps({
        "candidate_id": state.candidate_id, "feature_columns": list(state.feature_columns),
        "categorical_columns": list(state.categorical_columns), "profile_columns": list(state.profile_columns),
        "teacher_evidence_hashes": dict(state.teacher_evidence_hashes),
    }, sort_keys=True), encoding="utf-8")
    nested = root / "e2_model_delivery.zip"; shutil.copyfile(sources.e2_delivery, nested)
    manifest = root / "full_fit_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "campaign_id": "tree_privileged_profile_v1",
        "candidate_id": token.decision.candidate_id, "bindings": dict(token.bindings),
        "seeds": list(token.seeds), "iterations": {str(k): v for k, v in token.iterations.items()},
        "models": {str(seed): file_sha256(path) for seed, path in models.items()},
        "e2_delivery_sha256": file_sha256(nested),
        "profile_state_sha256": sha256((frozen / "profile_state.json").read_bytes()).hexdigest(),
    }, sort_keys=True), encoding="utf-8")
    return FullFitResult(token.decision.candidate_id, root, MappingProxyType(models), frozen, nested,
                         manifest, state, tuple(model_objects_by_seed[seed] for seed in token.seeds))
