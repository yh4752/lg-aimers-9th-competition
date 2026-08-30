"""OOF-only screening, R/F blending and acceptance gates."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import load_contract


class PrivilegedDecisionError(ValueError):
    pass


_FOLD_WEIGHTS = {2022: 0.20, 2023: 0.30, 2024: 0.50}


@dataclass(frozen=True)
class ScreenCandidate:
    candidate_id: str
    weighted_gain: float
    latest_gain: float
    worst_fold_gain: float
    first_two_gains: tuple[float, float]


@dataclass(frozen=True)
class ScreenEvidence:
    candidates: Mapping[str, ScreenCandidate]
    correlations: Mapping[tuple[str, str], float]


@dataclass(frozen=True)
class RfBlendDecision:
    candidate_id: str
    alpha_r: float
    alpha_f: float
    selection_brier: float


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    status: str
    failed_gates: tuple[str, ...]
    observed: Mapping[str, float | int | bool]
    rf_blend: RfBlendDecision | None = None


def _required(frame: object) -> pd.DataFrame:
    columns = {"candidate_id", "valid_year", "seed", "row_id", "target", "baseline_probability", "probability", "game_type"}
    if type(frame) is not pd.DataFrame or not columns.issubset(frame.columns):
        raise PrivilegedDecisionError("decision evidence schema differs")
    result = frame.copy(deep=True)
    for name in ("target", "baseline_probability", "probability"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    if result[["target", "baseline_probability", "probability"]].isna().any().any():
        raise PrivilegedDecisionError("decision probabilities are not finite")
    if not result["target"].isin([0, 1]).all() or not result[["baseline_probability", "probability"]].apply(lambda col: col.between(0, 1).all()).all():
        raise PrivilegedDecisionError("decision values are out of range")
    return result


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(target - probability)))


def _gain(frame: pd.DataFrame) -> float:
    target = frame["target"].to_numpy(dtype="float64")
    return _brier(target, frame["baseline_probability"].to_numpy(dtype="float64")) - _brier(target, frame["probability"].to_numpy(dtype="float64"))


def summarize_screen(evidence: pd.DataFrame) -> ScreenEvidence:
    frame = _required(evidence)
    frame = frame.loc[frame["seed"].eq(load_contract().screen_seed)]
    candidates: dict[str, ScreenCandidate] = {}
    predictions: dict[str, pd.Series] = {}
    for candidate_id, group in frame.groupby("candidate_id", sort=True):
        fold_gains = {int(year): _gain(part) for year, part in group.groupby("valid_year")}
        if set(fold_gains) != set(_FOLD_WEIGHTS):
            raise PrivilegedDecisionError("screen fold evidence differs")
        weighted = sum(_FOLD_WEIGHTS[year] * fold_gains[year] for year in _FOLD_WEIGHTS)
        candidates[str(candidate_id)] = ScreenCandidate(
            str(candidate_id), weighted, fold_gains[2024], min(fold_gains.values()),
            (fold_gains[2022], fold_gains[2023]),
        )
        predictions[str(candidate_id)] = group.sort_values(["valid_year", "row_id"], kind="stable").set_index(["valid_year", "row_id"])["probability"]
    correlations: dict[tuple[str, str], float] = {}
    names = sorted(predictions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            joined = pd.concat([predictions[left], predictions[right]], axis=1, join="inner")
            if len(joined) != len(predictions[left]) or len(joined) != len(predictions[right]):
                raise PrivilegedDecisionError("candidate row alignment differs")
            correlation = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            correlations[(left, right)] = correlation
    return ScreenEvidence(MappingProxyType(candidates), MappingProxyType(correlations))


def _correlation(evidence: ScreenEvidence, left: str, right: str) -> float:
    if left == right:
        return 1.0
    return float(evidence.correlations.get(tuple(sorted((left, right))), np.nan))


def select_confirmation_candidates(evidence: ScreenEvidence | pd.DataFrame) -> tuple[str, ...]:
    summary = summarize_screen(evidence) if type(evidence) is pd.DataFrame else evidence
    if type(summary) is not ScreenEvidence:
        raise PrivilegedDecisionError("screen evidence type differs")
    eligible = [item for item in summary.candidates.values() if not (item.first_two_gains[0] < 0 and item.first_two_gains[1] < 0)]
    simplicity = {"P": 0, "D15": 1, "D35": 2, "PD15": 3, "PD35": 4, "PD_RF": 5}
    ranked = sorted(eligible, key=lambda item: (
        -item.weighted_gain, -item.latest_gain, -item.worst_fold_gain, simplicity.get(item.candidate_id, 99), item.candidate_id,
    ))
    selected: list[str] = []
    threshold = float(load_contract().gates.maximum_screen_correlation)
    for item in ranked:
        if any(np.isfinite(_correlation(summary, item.candidate_id, chosen))
               and _correlation(summary, item.candidate_id, chosen) > threshold for chosen in selected):
            continue
        selected.append(item.candidate_id)
        if len(selected) == load_contract().maximum_confirmed_candidates:
            break
    return tuple(selected)


def select_rf_blend(evidence: pd.DataFrame, *, candidate_id: str | None = None) -> RfBlendDecision:
    frame = _required(evidence)
    if candidate_id is None:
        names = sorted(set(frame["candidate_id"]))
        if len(names) != 1:
            raise PrivilegedDecisionError("R/F blend requires one candidate")
        candidate_id = str(names[0])
    source = frame.loc[frame["candidate_id"].eq(candidate_id) & frame["valid_year"].isin([2022, 2023])]
    if source.empty:
        raise PrivilegedDecisionError("R/F selection evidence is empty")
    alphas = tuple(float(value) for value in load_contract().rf_alphas)
    selected: dict[str, tuple[float, float]] = {}
    for game_type in ("R", "F"):
        segment = source.loc[source["game_type"].eq(game_type)]
        if segment.empty:
            selected[game_type] = (0.0, np.nan)
            continue
        target = segment["target"].to_numpy(dtype="float64")
        baseline = segment["baseline_probability"].to_numpy(dtype="float64")
        candidate = segment["probability"].to_numpy(dtype="float64")
        scored = [(_brier(target, np.clip(baseline + alpha * (candidate - baseline), 0, 1)), alpha) for alpha in alphas]
        score, alpha = min(scored, key=lambda item: (item[0], item[1]))
        selected[game_type] = (alpha, score)
    finite_scores = [value[1] for value in selected.values() if np.isfinite(value[1])]
    return RfBlendDecision(candidate_id, selected["R"][0], selected["F"][0], float(np.mean(finite_scores)))


def apply_rf_blend(frame: pd.DataFrame, decision: RfBlendDecision) -> np.ndarray:
    source = _required(frame)
    alpha = np.where(source["game_type"].eq("F"), decision.alpha_f, decision.alpha_r)
    baseline = source["baseline_probability"].to_numpy(dtype="float64")
    candidate = source["probability"].to_numpy(dtype="float64")
    return np.clip(baseline + alpha * (candidate - baseline), 0, 1)


def decide_candidate(evidence: pd.DataFrame, *, rf_blend: RfBlendDecision | None = None) -> CandidateDecision:
    frame = _required(evidence)
    names = sorted(set(frame["candidate_id"]));
    if len(names) != 1:
        raise PrivilegedDecisionError("acceptance requires one candidate")
    candidate_id = str(names[0])
    if rf_blend is not None:
        frame = frame.copy(); frame["probability"] = apply_rf_blend(frame, rf_blend)
    by_fold = {int(year): _gain(part) for year, part in frame.groupby("valid_year")}
    if set(by_fold) != set(_FOLD_WEIGHTS):
        raise PrivilegedDecisionError("acceptance fold evidence differs")
    weighted_gain = sum(_FOLD_WEIGHTS[year] * by_fold[year] for year in _FOLD_WEIGHTS)
    improved_folds = sum(value > 0 for value in by_fold.values())
    seed_gains = {int(seed): _gain(part) for seed, part in frame.groupby("seed")}
    non_worse_seeds = sum(value >= 0 for value in seed_gains.values())
    regressions = []
    for _, segment in frame.groupby(["valid_year", "game_type"], dropna=False):
        regressions.append(max(0.0, -_gain(segment)))
    maximum_regression = max(regressions, default=0.0)
    finite_bounded = bool(np.isfinite(frame["probability"]).all() and frame["probability"].between(0, 1).all())
    gates = load_contract().gates
    checks = {
        "weighted_gain": weighted_gain >= float(gates.weighted_gain),
        "minimum_improved_folds": improved_folds >= gates.minimum_improved_folds,
        "latest_min_gain": by_fold[2024] >= float(gates.latest_min_gain),
        "maximum_segment_regression": maximum_regression <= float(gates.maximum_segment_regression),
        "minimum_non_worse_seeds": non_worse_seeds >= gates.minimum_non_worse_seeds,
        "finite_bounded_probabilities": finite_bounded,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    observed: dict[str, float | int | bool] = {
        "weighted_gain": weighted_gain, "latest_gain": by_fold[2024], "improved_folds": improved_folds,
        "maximum_segment_regression": maximum_regression, "non_worse_seeds": non_worse_seeds,
        "finite_bounded_probabilities": finite_bounded,
    }
    return CandidateDecision(candidate_id, "accepted" if not failed else "rejected", failed,
                             MappingProxyType(observed), rf_blend)


def select_ensemble(evidence: pd.DataFrame, accepted: tuple[str, ...]) -> tuple[str, ...]:
    if len(accepted) != 2:
        return accepted
    frame = _required(evidence)
    pivots = []
    for name in accepted:
        part = frame.loc[frame["candidate_id"].eq(name)].sort_values(["valid_year", "seed", "row_id"])
        pivots.append(part)
    if len(pivots[0]) != len(pivots[1]) or not pivots[0][["valid_year", "seed", "row_id"]].reset_index(drop=True).equals(pivots[1][["valid_year", "seed", "row_id"]].reset_index(drop=True)):
        raise PrivilegedDecisionError("ensemble rows differ")
    target = pivots[0]["target"].to_numpy(dtype="float64")
    best = min(_brier(target, part["probability"].to_numpy(dtype="float64")) for part in pivots)
    averaged = (pivots[0]["probability"].to_numpy() + pivots[1]["probability"].to_numpy()) / 2
    gain = best - _brier(target, averaged)
    return accepted if gain >= float(load_contract().gates.ensemble_incremental_gain) else (accepted[0],)

