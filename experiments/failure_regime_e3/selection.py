from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

import numpy as np
import pandas as pd

from .contracts import load_contract


class E3SelectionError(ValueError):
    pass


class GateEstimator(Protocol):
    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> None: ...
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...


GateFactory = Callable[[int], GateEstimator]
_META = {"row_id", "target", "oof_year", "pitcher_id"}
_WEIGHTS = {2022: 0.60, 2023: 0.75, 2024: 0.90}


@dataclass(frozen=True)
class GateRecipe:
    e2_weight: float
    gate_strength: float
    selection_years: tuple[int, ...]
    identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        contract = load_contract()
        if (
            self.e2_weight not in contract.e2_weights
            or self.gate_strength not in contract.gate_strengths
            or self.selection_years != contract.selection_years
        ):
            raise E3SelectionError("gate recipe differs")
        payload = json.dumps(
            {
                "e2_weight": self.e2_weight,
                "gate_strength": self.gate_strength,
                "selection_years": list(self.selection_years),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(self, "identity_sha256", sha256(payload).hexdigest())


@dataclass(frozen=True)
class FrozenRecipe:
    recipe: GateRecipe
    selection_probability: np.ndarray
    selection_brier: float


@dataclass(frozen=True)
class CandidateEvidence:
    fold_gains: Mapping[int, float]
    weighted_gain: float
    latest_gain: float
    minimum_fold_gain: float
    bootstrap_lower: float
    maximum_segment_regression: float
    r_gain: float
    f_gain: float
    non_worse_seed_count: int
    finite_probabilities: bool


@dataclass(frozen=True)
class CandidateDecision:
    status: str
    failed_gates: tuple[str, ...]


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or frame.empty or not _META.issubset(frame.columns):
        raise E3SelectionError("meta evidence differs")
    columns = [name for name in frame.columns if name not in _META]
    if "game_type" not in columns or "p_e2" not in columns or "p_s_global" not in columns:
        raise E3SelectionError("gate feature columns differ")
    return frame.loc[:, columns].copy(deep=True)


def gate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed row-local gate matrix without target metadata."""
    return _features(frame)


def _fold_id(value: str, folds: int = 5) -> int:
    return int.from_bytes(sha256(value.encode("utf-8")).digest()[:4], "big") % folds


def _gate_probability(model: GateEstimator, frame: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(_features(frame)), dtype="float64")
    if raw.shape != (len(frame), 2) or not np.isfinite(raw).all():
        raise E3SelectionError("gate probability differs")
    probability = raw[:, 1]
    if np.any((probability < 0) | (probability > 1)):
        raise E3SelectionError("gate probability range differs")
    return probability


def _crossfit(frame: pd.DataFrame, factory: GateFactory) -> np.ndarray:
    groups = frame["pitcher_id"].astype(str).map(_fold_id).to_numpy(dtype="int8")
    target = frame["target"].to_numpy(dtype="int8")
    output = np.full(len(frame), np.nan, dtype="float64")
    for fold in range(5):
        valid = groups == fold
        train = ~valid
        if not valid.any() or not train.any():
            raise E3SelectionError("gate crossfit split differs")
        model = factory(3407 + fold)
        model.fit(_features(frame.loc[train]), target[train])
        output[valid] = _gate_probability(model, frame.loc[valid])
    if not np.isfinite(output).all():
        raise E3SelectionError("gate crossfit is incomplete")
    return output


def crossfit_gate_probability(frame: pd.DataFrame, *, estimator_factory: GateFactory) -> np.ndarray:
    return _crossfit(frame, estimator_factory)


def apply_recipe(recipe: GateRecipe, e2: np.ndarray, gate: np.ndarray) -> np.ndarray:
    anchor = np.asarray(e2, dtype="float64")
    direct = np.asarray(gate, dtype="float64")
    if anchor.shape != direct.shape or anchor.ndim != 1 or not np.isfinite(anchor).all() or not np.isfinite(direct).all():
        raise E3SelectionError("recipe arrays differ")
    scale = (1.0 - recipe.e2_weight) * recipe.gate_strength
    return np.clip(anchor + scale * (direct - anchor), 1e-6, 1.0 - 1e-6)


def fit_frozen_recipe(frame: pd.DataFrame, *, estimator_factory: GateFactory) -> FrozenRecipe:
    contract = load_contract()
    years = pd.to_numeric(frame["oof_year"], errors="coerce")
    tuning = frame.loc[years.isin(contract.selection_years)].copy(deep=True).reset_index(drop=True)
    if set(tuning["oof_year"].astype(int)) != set(contract.selection_years):
        raise E3SelectionError("selection years differ")
    gate = _crossfit(tuning, estimator_factory)
    target = tuning["target"].to_numpy(dtype="float64")
    e2 = tuning["p_e2"].to_numpy(dtype="float64")
    best: tuple[float, float, float, GateRecipe, np.ndarray] | None = None
    for e2_weight in contract.e2_weights:
        for strength in contract.gate_strengths:
            recipe = GateRecipe(e2_weight, strength, contract.selection_years)
            probability = apply_recipe(recipe, e2, gate)
            brier = float(np.mean(np.square(target - probability)))
            candidate = (brier, -strength, -e2_weight, recipe, probability)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    assert best is not None
    probability = best[4].copy()
    probability.setflags(write=False)
    return FrozenRecipe(best[3], probability, best[0])


def fit_confirmation_gate(
    frame: pd.DataFrame,
    *,
    estimator_factory: GateFactory,
    seed: int = 3407,
) -> GateEstimator:
    contract = load_contract()
    training = frame.loc[frame["oof_year"].isin(contract.selection_years)].copy(deep=True)
    model = estimator_factory(seed)
    model.fit(_features(training), training["target"].to_numpy(dtype="int8"))
    return model


def _prediction_gain(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype="float64")
    anchor = pd.to_numeric(frame["p_e2"], errors="coerce").to_numpy(dtype="float64")
    candidate = np.asarray(probability, dtype="float64")
    if candidate.shape != target.shape or not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 1)):
        raise E3SelectionError("candidate probability differs")
    return np.square(target - anchor) - np.square(target - candidate)


def _fold_gains(frame: pd.DataFrame, probability: np.ndarray) -> dict[int, float]:
    gain = _prediction_gain(frame, probability)
    years = frame["oof_year"].to_numpy(dtype="int64")
    return {int(year): float(gain[years == year].mean()) for year in sorted(np.unique(years))}


def _weighted(folds: Mapping[int, float]) -> float:
    values = [(_WEIGHTS[year], gain) for year, gain in folds.items() if year in _WEIGHTS]
    if not values:
        raise E3SelectionError("fold weights differ")
    return float(sum(weight * gain for weight, gain in values) / sum(weight for weight, _ in values))


def _bootstrap(frame: pd.DataFrame, gain: np.ndarray, repetitions: int) -> float:
    groups = [part.index.to_numpy(dtype="int64") for _, part in frame.groupby("pitcher_id", sort=True)]
    sums = np.asarray([gain[index].sum() for index in groups], dtype="float64")
    counts = np.asarray([len(index) for index in groups], dtype="float64")
    rng = np.random.default_rng(3407)
    sampled = np.empty(repetitions, dtype="float64")
    for index in range(repetitions):
        choice = rng.integers(0, len(groups), size=len(groups))
        sampled[index] = sums[choice].sum() / counts[choice].sum()
    return float(np.quantile(sampled, 0.025))


def candidate_evidence(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    seed_probabilities: Mapping[int, np.ndarray],
    bootstrap_repetitions: int = 1000,
    minimum_segment_rows: int | None = None,
) -> CandidateEvidence:
    if type(frame) is not pd.DataFrame or frame.empty or not {"target", "p_e2", "oof_year", "pitcher_id", "game_type"}.issubset(frame.columns):
        raise E3SelectionError("candidate evidence columns differ")
    gain = _prediction_gain(frame, probability)
    folds = _fold_gains(frame, probability)
    contract = load_contract()
    threshold = int(minimum_segment_rows or contract.gates["minimum_segment_rows"])
    segment_regressions = []
    for _, part in frame.assign(_gain=gain).groupby("game_type", sort=True):
        if len(part) >= threshold:
            segment_regressions.append(max(0.0, -float(part["_gain"].mean())))
    def segment(active: str) -> float:
        mask = frame["game_type"].eq(active).to_numpy()
        return float(gain[mask].mean()) if mask.any() else 0.0
    non_worse = sum(_weighted(_fold_gains(frame, values)) >= 0.0 for values in seed_probabilities.values())
    return CandidateEvidence(
        fold_gains=MappingProxyType(folds),
        weighted_gain=_weighted(folds),
        latest_gain=folds[max(folds)],
        minimum_fold_gain=min(folds.values()),
        bootstrap_lower=_bootstrap(frame, gain, bootstrap_repetitions),
        maximum_segment_regression=max(segment_regressions, default=0.0),
        r_gain=segment("R"),
        f_gain=segment("F"),
        non_worse_seed_count=int(non_worse),
        finite_probabilities=bool(np.isfinite(probability).all()),
    )


def decide(evidence: CandidateEvidence) -> CandidateDecision:
    gates = load_contract().gates
    checks = (
        ("weighted_gain", evidence.weighted_gain >= gates["weighted_gain"]),
        ("latest_gain", evidence.latest_gain >= gates["latest_gain"]),
        ("minimum_fold_gain", evidence.minimum_fold_gain >= gates["minimum_fold_gain"]),
        ("bootstrap_lower", evidence.bootstrap_lower >= gates["bootstrap_lower"]),
        ("maximum_segment_regression", evidence.maximum_segment_regression <= gates["maximum_segment_regression"] and evidence.r_gain >= 0.0 and evidence.f_gain >= 0.0),
        ("minimum_non_worse_seeds", evidence.non_worse_seed_count >= gates["minimum_non_worse_seeds"]),
        ("finite_probabilities", evidence.finite_probabilities),
    )
    failed = tuple(name for name, passed in checks if not passed)
    return CandidateDecision("accepted" if not failed else "rejected", failed)
