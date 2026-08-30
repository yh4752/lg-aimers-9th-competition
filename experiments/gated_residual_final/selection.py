from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .contracts import load_contract
from .inputs import canonical_json


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateConfig:
    archetype: str
    alpha: float
    k: int | None
    beta: float | None
    ridge: int | None

    @property
    def candidate_id(self) -> str:
        payload = canonical_json({
            "archetype": self.archetype, "alpha": self.alpha, "k": self.k,
            "beta": self.beta, "ridge": self.ridge,
        })
        return f"{self.archetype}_{sha256(payload).hexdigest()[:10]}"


@dataclass(frozen=True)
class FrozenCandidate:
    config: CandidateConfig
    locked_years: tuple[int, ...]
    tuning_objective: float

    @property
    def archetype(self) -> str:
        return self.config.archetype


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    fold_gains: Mapping[int, float]
    weighted_gain: float
    latest_gain: float
    minimum_fold_gain: float
    bootstrap_lower: float
    non_worse_seed_count: int
    latest_non_worse_seed_count: int
    maximum_segment_regression: float
    finite_probabilities: bool


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    status: str
    failed_gates: tuple[str, ...]


Predictor = Callable[[pd.DataFrame, CandidateConfig], np.ndarray]
_WEIGHTS = {2022: 0.60, 2023: 0.75, 2024: 0.90}


def _prediction_gain(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype="float64")
    anchor = pd.to_numeric(frame["p_anchor"], errors="coerce").to_numpy(dtype="float64")
    candidate = np.asarray(probability, dtype="float64")
    if candidate.shape != target.shape:
        raise SelectionError("candidate prediction shape differs")
    if (
        not np.isfinite(target).all() or not np.isfinite(anchor).all() or not np.isfinite(candidate).all()
        or np.any((anchor < 0) | (anchor > 1)) or np.any((candidate < 0) | (candidate > 1))
    ):
        raise SelectionError("candidate prediction values differ")
    return np.square(target - anchor) - np.square(target - candidate)


def _fold_gains(frame: pd.DataFrame, probability: np.ndarray) -> dict[int, float]:
    gain = _prediction_gain(frame, probability)
    years = pd.to_numeric(frame["oof_year"], errors="coerce").to_numpy(dtype="int64")
    return {int(year): float(gain[years == year].mean()) for year in sorted(np.unique(years))}


def _weighted(folds: Mapping[int, float]) -> float:
    items = [(_WEIGHTS[year], value) for year, value in folds.items() if year in _WEIGHTS]
    if not items:
        raise SelectionError("fold weights have no overlap")
    return float(sum(weight * value for weight, value in items) / sum(weight for weight, _ in items))


def _objective(frame: pd.DataFrame, probability: np.ndarray, *, years: tuple[int, ...]) -> float:
    mask = frame["oof_year"].isin(years).to_numpy()
    folds = _fold_gains(frame.loc[mask], np.asarray(probability)[mask])
    worst = min(folds.values())
    return _weighted(folds) - 2.0 * max(0.0, -worst)


def freeze_archetypes(frame: pd.DataFrame, *, predictor: Predictor) -> tuple[FrozenCandidate, ...]:
    required = {"target", "p_anchor", "oof_year"}
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns):
        raise SelectionError("tuning columns differ")
    tuning = frame.loc[frame["oof_year"].isin((2022, 2023))].copy()
    if set(pd.to_numeric(tuning["oof_year"]).astype(int).unique()) != {2022, 2023}:
        raise SelectionError("tuning years differ")
    contract = load_contract()

    def choose(configs: list[CandidateConfig], years: tuple[int, ...]) -> FrozenCandidate:
        scored = []
        for config in configs:
            probability = predictor(tuning.copy(deep=True), config)
            scored.append((_objective(tuning, probability, years=years), config.candidate_id, config))
        score, _, config = max(scored, key=lambda item: (item[0], item[1]))
        return FrozenCandidate(config, (2022, 2023), float(score))

    g0 = choose([CandidateConfig("G0", alpha, None, None, None) for alpha in contract.alpha_grid], (2022, 2023))
    g1 = choose([
        CandidateConfig("G1", alpha, k, None, None)
        for alpha in contract.alpha_grid for k in contract.k_grid
    ], (2022, 2023))
    calibration = {}
    for archetype in ("G2", "G3"):
        calibration[archetype] = choose([
            CandidateConfig(archetype, g1.config.alpha, g1.config.k, beta, ridge)
            for beta in contract.beta_grid for ridge in contract.lambda_grid
        ], (2023,))
    return (g0, g1, calibration["G2"], calibration["G3"])


def _validated_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "row_id", "target", "p_anchor", "probability", "oof_year", "pitcher_id", "game_type",
    }
    if type(frame) is not pd.DataFrame or frame.empty or not required.issubset(frame.columns):
        raise SelectionError("evidence columns differ")
    output = frame.copy(deep=True).sort_values("row_id", kind="stable").reset_index(drop=True)
    if output["row_id"].isna().any() or not output["row_id"].is_unique:
        raise SelectionError("evidence row identity differs")
    for column in ("target", "p_anchor", "probability", "oof_year"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if (
        output[["target", "p_anchor", "probability", "oof_year"]].isna().any().any()
        or not output["target"].isin((0, 1)).all()
        or not output["p_anchor"].between(0, 1).all()
        or not output["probability"].between(0, 1).all()
        or not np.isfinite(output["probability"]).all()
    ):
        raise SelectionError("evidence values differ")
    return output


def _bootstrap_lower(frame: pd.DataFrame, gain: np.ndarray, repetitions: int) -> float:
    group_positions = [group.index.to_numpy(dtype="int64") for _, group in frame.groupby("pitcher_id", sort=True)]
    if not group_positions:
        raise SelectionError("pitcher clusters are absent")
    cluster_sums = np.asarray([gain[positions].sum() for positions in group_positions], dtype="float64")
    cluster_counts = np.asarray([len(positions) for positions in group_positions], dtype="float64")
    rng = np.random.default_rng(3407)
    sampled = np.empty(int(repetitions), dtype="float64")
    for index in range(len(sampled)):
        choices = rng.integers(0, len(group_positions), size=len(group_positions))
        sampled[index] = float(cluster_sums[choices].sum() / cluster_counts[choices].sum())
    return float(np.quantile(sampled, 0.025))


def evidence_from_predictions(
    candidate_id: str,
    predictions: pd.DataFrame,
    *,
    seed_predictions: Mapping[int, pd.DataFrame],
    minimum_segment_rows: int | None = None,
    bootstrap_repetitions: int = 1000,
) -> CandidateEvidence:
    frame = _validated_predictions(predictions)
    gain = _prediction_gain(frame, frame["probability"].to_numpy())
    folds = _fold_gains(frame, frame["probability"].to_numpy())
    latest_year = max(folds)
    non_worse = 0
    latest_non_worse = 0
    for seed, raw in sorted(seed_predictions.items()):
        seed_frame = _validated_predictions(raw)
        if seed_frame["row_id"].astype(str).tolist() != frame["row_id"].astype(str).tolist():
            raise SelectionError(f"seed row identity differs: {seed}")
        seed_folds = _fold_gains(seed_frame, seed_frame["probability"].to_numpy())
        non_worse += int(_weighted(seed_folds) >= 0.0)
        latest_non_worse += int(seed_folds.get(latest_year, float("-inf")) >= 0.0)
    threshold = int(minimum_segment_rows or load_contract().gates["minimum_segment_rows"])
    regressions: list[float] = []
    segment_columns = [column for column in ("game_type", "hand_matchup") if column in frame]
    work = frame.assign(_gain=gain)
    for column in segment_columns:
        for _, group in work.groupby(column, sort=True, observed=True):
            if len(group) >= threshold:
                regressions.append(max(0.0, -float(group["_gain"].mean())))
    return CandidateEvidence(
        candidate_id=str(candidate_id),
        fold_gains=MappingProxyType(folds),
        weighted_gain=_weighted(folds),
        latest_gain=folds[latest_year],
        minimum_fold_gain=min(folds.values()),
        bootstrap_lower=_bootstrap_lower(frame, gain, bootstrap_repetitions),
        non_worse_seed_count=non_worse,
        latest_non_worse_seed_count=latest_non_worse,
        maximum_segment_regression=max(regressions, default=0.0),
        finite_probabilities=bool(np.isfinite(frame["probability"]).all()),
    )


def decide(evidence: CandidateEvidence) -> CandidateDecision:
    gates = load_contract().gates
    checks = (
        ("weighted_gain", evidence.weighted_gain >= gates["weighted_gain"]),
        ("latest_gain", evidence.latest_gain >= gates["latest_gain"]),
        ("minimum_fold_gain", evidence.minimum_fold_gain >= gates["minimum_fold_gain"]),
        ("bootstrap_lower", evidence.bootstrap_lower >= gates["bootstrap_lower"]),
        ("non_worse_seed_count", evidence.non_worse_seed_count >= gates["minimum_non_worse_seeds"]),
        ("latest_non_worse_seed_count", evidence.latest_non_worse_seed_count >= gates["minimum_latest_non_worse_seeds"]),
        ("maximum_segment_regression", evidence.maximum_segment_regression <= gates["maximum_segment_regression"]),
        ("finite_probabilities", evidence.finite_probabilities),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return CandidateDecision(evidence.candidate_id, "accepted" if not failures else "rejected", failures)
