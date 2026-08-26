from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .e2_contracts import BlendCandidate, E2Contract
from .inputs import PREDICTION_COLUMNS


class E2DecisionError(ValueError):
    """Raised when E2 prediction evidence cannot support a decision."""


FOLD_ORDER = ((2021, 2022), (2022, 2023), (2023, 2024))


@dataclass(frozen=True)
class FoldScore:
    fold: tuple[int, int]
    row_count: int
    baseline_brier: float
    candidate_brier: float
    gain: float
    best_iteration: int


@dataclass(frozen=True)
class StructureEvidence:
    candidate_id: str
    folds: tuple[FoldScore, ...]


@dataclass(frozen=True)
class StructureDecision:
    status: str
    selected: str | None
    weighted_gain: float | None
    f3_gain: float | None
    worst_fold_gain: float | None
    reason: str


@dataclass(frozen=True)
class SeedEvidence:
    seed: int
    folds: tuple[FoldScore, ...]


@dataclass(frozen=True)
class SeedDecision:
    status: str
    candidate_id: str
    ensemble_weighted_gain: float
    seed_weighted_gains: Mapping[int, float]
    non_worse_seed_counts: Mapping[tuple[int, int], int]
    reason: str


@dataclass(frozen=True)
class BlendDecision:
    status: str
    predictor: str
    method: str
    catboost_weight: float
    f2_selection_fold: tuple[int, int]
    f2_applied_method: str
    f2_catboost_weight: float
    f3_selection_folds: tuple[tuple[int, int], ...]
    f3_applied_method: str
    f3_catboost_weight: float
    sequential_gain: float
    fold_gains_vs_catboost: Mapping[tuple[int, int], float]
    reason: str


@dataclass(frozen=True)
class AcceptanceDecision:
    status: str
    predictor: str
    weighted_gain: float
    f3_gain: float
    worst_fold_gain: float
    bootstrap_lower: float
    bootstrap_upper: float
    maximum_segment_regression: float
    performance_grade: str
    reason: str


def _frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or tuple(frame.columns) != PREDICTION_COLUMNS:
        raise E2DecisionError(f"{label} prediction schema differs")
    if frame.empty or frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise E2DecisionError(f"{label} row_id values differ")
    output = frame.copy(deep=True)
    output["target"] = pd.to_numeric(output["target"], errors="coerce")
    output["probability"] = pd.to_numeric(output["probability"], errors="coerce")
    if (
        not output["target"].isin([0, 1]).all()
        or output["probability"].isna().any()
        or not output["probability"].between(0, 1).all()
        or output.loc[:, PREDICTION_COLUMNS[3:]].isna().any().any()
    ):
        raise E2DecisionError(f"{label} prediction values differ")
    return output


def _aligned(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _frame(baseline, "baseline")
    trial = _frame(candidate, "candidate")
    if base["row_id"].astype(str).tolist() != trial["row_id"].astype(str).tolist():
        raise E2DecisionError("row_id alignment differs")
    if not np.array_equal(base["target"].to_numpy(), trial["target"].to_numpy()):
        raise E2DecisionError("target alignment differs")
    for column in PREDICTION_COLUMNS[3:]:
        if base[column].astype(str).tolist() != trial[column].astype(str).tolist():
            raise E2DecisionError(f"diagnostic alignment differs: {column}")
    return base, trial


def evaluate_fold(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    fold: tuple[int, int],
    best_iteration: int,
) -> FoldScore:
    if fold not in FOLD_ORDER:
        raise E2DecisionError("fold identity differs")
    if type(best_iteration) is not int or best_iteration < 0:
        raise E2DecisionError("best iteration differs")
    base, trial = _aligned(baseline, candidate)
    target = base["target"].to_numpy(dtype="float64")
    baseline_brier = float(
        np.mean(np.square(base["probability"].to_numpy(dtype="float64") - target))
    )
    candidate_brier = float(
        np.mean(np.square(trial["probability"].to_numpy(dtype="float64") - target))
    )
    return FoldScore(
        fold=fold,
        row_count=len(base),
        baseline_brier=baseline_brier,
        candidate_brier=candidate_brier,
        gain=baseline_brier - candidate_brier,
        best_iteration=best_iteration,
    )


def _ordered_scores(scores: Sequence[FoldScore]) -> tuple[FoldScore, ...]:
    by_fold = {score.fold: score for score in scores}
    if len(by_fold) != len(scores) or set(by_fold) != set(FOLD_ORDER):
        raise E2DecisionError("fold score identities differ")
    ordered = tuple(by_fold[fold] for fold in FOLD_ORDER)
    for score in ordered:
        values = (
            score.baseline_brier,
            score.candidate_brier,
            score.gain,
        )
        if (
            score.row_count <= 0
            or score.best_iteration < 0
            or not all(math.isfinite(value) for value in values)
            or not math.isclose(
                score.baseline_brier - score.candidate_brier,
                score.gain,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise E2DecisionError("fold score values differ")
    return ordered


def _weighted_gain(scores: Sequence[FoldScore]) -> float:
    ordered = _ordered_scores(scores)
    rows = sum(score.row_count for score in ordered)
    return float(sum(score.gain * score.row_count for score in ordered) / rows)


def decide_structure(
    evidence: Sequence[StructureEvidence],
    contract: E2Contract,
) -> StructureDecision:
    if not evidence or len({item.candidate_id for item in evidence}) != len(evidence):
        raise E2DecisionError("structure evidence identities differ")
    if any(item.candidate_id not in contract.structures for item in evidence):
        raise E2DecisionError("structure is not registered")
    ranked: list[tuple[tuple[float, float, float, bool], StructureEvidence]] = []
    for item in evidence:
        folds = _ordered_scores(item.folds)
        ranked.append(
            (
                (
                    min(score.gain for score in folds),
                    _weighted_gain(folds),
                    folds[-1].gain,
                    item.candidate_id == "c1_anchor_residual",
                ),
                item,
            )
        )
    _, selected = max(ranked, key=lambda item: item[0])
    scores = _ordered_scores(selected.folds)
    weighted = _weighted_gain(scores)
    f3_gain = scores[-1].gain
    worst = min(score.gain for score in scores)
    gates = contract.gates
    if weighted < float(gates["structure_weighted_gain"]):
        status, reason = "rejected", "weighted_gain_below_gate"
    elif f3_gain < float(gates["structure_f3_gain"]):
        status, reason = "rejected", "f3_gain_below_gate"
    elif worst < -float(gates["structure_max_fold_regression"]):
        status, reason = "rejected", "fold_regression_above_gate"
    else:
        status, reason = "passed", "structure_gates_passed"
    return StructureDecision(
        status=status,
        selected=selected.candidate_id,
        weighted_gain=weighted,
        f3_gain=f3_gain,
        worst_fold_gain=worst,
        reason=reason,
    )


def decide_seeds(
    candidate_id: str,
    evidence: Sequence[SeedEvidence],
    ensemble_scores: Sequence[FoldScore],
    contract: E2Contract,
) -> SeedDecision:
    if candidate_id not in contract.structures:
        raise E2DecisionError("seed candidate is not registered")
    by_seed = {item.seed: _ordered_scores(item.folds) for item in evidence}
    if len(by_seed) != len(evidence) or set(by_seed) != set(contract.seeds):
        raise E2DecisionError("seed evidence identities differ")
    ensemble = _ordered_scores(ensemble_scores)
    weighted = {seed: _weighted_gain(scores) for seed, scores in by_seed.items()}
    non_worse = {
        fold: sum(by_seed[seed][index].gain >= 0.0 for seed in contract.seeds)
        for index, fold in enumerate(FOLD_ORDER)
    }
    ensemble_weighted = _weighted_gain(ensemble)
    seed_limit = float(contract.gates["seed_max_weighted_regression"])
    if any(gain < -seed_limit for gain in weighted.values()):
        status, reason = "rejected", "seed_weighted_regression"
    elif (weighted[3407] >= 0) != (ensemble_weighted >= 0):
        status, reason = "rejected", "ensemble_direction_differs"
    elif any(count < 2 for count in non_worse.values()):
        status, reason = "rejected", "fewer_than_two_non_worse_seeds"
    else:
        status, reason = "passed", "seed_gates_passed"
    return SeedDecision(
        status=status,
        candidate_id=candidate_id,
        ensemble_weighted_gain=ensemble_weighted,
        seed_weighted_gains=dict(weighted),
        non_worse_seed_counts=dict(non_worse),
        reason=reason,
    )


def blend_probabilities(
    tabm: np.ndarray,
    catboost: np.ndarray,
    method: str,
    catboost_weight: float,
) -> np.ndarray:
    tabm_probability = np.asarray(tabm, dtype="float64")
    cat_probability = np.asarray(catboost, dtype="float64")
    if tabm_probability.shape != cat_probability.shape:
        raise E2DecisionError("blend probability shapes differ")
    if (
        not np.isfinite(tabm_probability).all()
        or not np.isfinite(cat_probability).all()
        or not 0.0 <= catboost_weight <= 1.0
    ):
        raise E2DecisionError("blend probability values differ")
    if method == "catboost":
        if catboost_weight != 1.0:
            raise E2DecisionError("CatBoost blend weight differs")
        return np.clip(cat_probability, 1e-5, 1 - 1e-5)
    if method not in {"probability", "logit"}:
        raise E2DecisionError("blend method differs")
    tabm_clipped = np.clip(tabm_probability, 1e-5, 1 - 1e-5)
    cat_clipped = np.clip(cat_probability, 1e-5, 1 - 1e-5)
    if method == "probability":
        output = (1.0 - catboost_weight) * tabm_clipped + catboost_weight * cat_clipped
    else:
        tabm_logit = np.log(tabm_clipped / (1.0 - tabm_clipped))
        cat_logit = np.log(cat_clipped / (1.0 - cat_clipped))
        combined = (1.0 - catboost_weight) * tabm_logit + catboost_weight * cat_logit
        output = 1.0 / (1.0 + np.exp(-combined))
    return np.clip(output, 1e-5, 1 - 1e-5)


def _blended_frame(
    tabm: pd.DataFrame,
    catboost: pd.DataFrame,
    candidate: BlendCandidate,
) -> pd.DataFrame:
    base, cat = _aligned(tabm, catboost)
    output = base.copy(deep=True)
    output["probability"] = blend_probabilities(
        base["probability"].to_numpy(dtype="float64"),
        cat["probability"].to_numpy(dtype="float64"),
        candidate.method,
        candidate.catboost_weight,
    )
    return output


def _brier(frame: pd.DataFrame) -> float:
    target = frame["target"].to_numpy(dtype="float64")
    probability = frame["probability"].to_numpy(dtype="float64")
    return float(np.mean(np.square(probability - target)))


def _select_blend(
    folds: Sequence[tuple[int, int]],
    tabm: Mapping[tuple[int, int], pd.DataFrame],
    catboost: Mapping[tuple[int, int], pd.DataFrame],
    candidates: Sequence[BlendCandidate],
) -> BlendCandidate:
    rows: list[tuple[float, int, float, int, BlendCandidate]] = []
    for candidate in candidates:
        total_loss = 0.0
        total_rows = 0
        for fold in folds:
            blended = _blended_frame(tabm[fold], catboost[fold], candidate)
            total_loss += _brier(blended) * len(blended)
            total_rows += len(blended)
        method_priority = 0 if candidate.method == "catboost" else 1
        family_priority = 0 if candidate.method == "probability" else 1
        rows.append(
            (
                total_loss / total_rows,
                method_priority,
                -candidate.catboost_weight,
                family_priority,
                candidate,
            )
        )
    return min(rows, key=lambda item: item[:4])[-1]


def causal_blend_decision(
    tabm: Mapping[tuple[int, int], pd.DataFrame],
    catboost: Mapping[tuple[int, int], pd.DataFrame],
    contract: E2Contract,
) -> BlendDecision:
    if set(tabm) != set(FOLD_ORDER) or set(catboost) != set(FOLD_ORDER):
        raise E2DecisionError("blend fold evidence differs")
    f2_candidate = _select_blend(
        (FOLD_ORDER[0],), tabm, catboost, contract.blend_candidates
    )
    f3_candidate = _select_blend(
        FOLD_ORDER[:2], tabm, catboost, contract.blend_candidates
    )
    applied = {
        FOLD_ORDER[1]: f2_candidate,
        FOLD_ORDER[2]: f3_candidate,
    }
    fold_gains: dict[tuple[int, int], float] = {}
    total_gain = 0.0
    total_rows = 0
    for fold, candidate in applied.items():
        blended = _blended_frame(tabm[fold], catboost[fold], candidate)
        cat = _frame(catboost[fold], "CatBoost")
        gain = _brier(cat) - _brier(blended)
        fold_gains[fold] = gain
        total_gain += gain * len(cat)
        total_rows += len(cat)
    sequential_gain = total_gain / total_rows
    f3_blended = _blended_frame(tabm[FOLD_ORDER[2]], catboost[FOLD_ORDER[2]], f3_candidate)
    f3_no_worse = _brier(f3_blended) <= min(
        _brier(_frame(tabm[FOLD_ORDER[2]], "TabM")),
        _brier(_frame(catboost[FOLD_ORDER[2]], "CatBoost")),
    )
    if sequential_gain < float(contract.gates["blend_min_sequential_gain"]):
        status, reason = "rejected", "sequential_gain_below_gate"
    elif min(fold_gains.values()) < -float(contract.gates["blend_max_fold_regression"]):
        status, reason = "rejected", "blend_fold_regression_above_gate"
    elif not f3_no_worse:
        status, reason = "rejected", "blend_f3_is_worse"
    else:
        status, reason = "passed", "causal_blend_gates_passed"
    predictor = "blend" if status == "passed" else "catboost"
    selected = f3_candidate if status == "passed" else BlendCandidate("catboost", 1.0)
    return BlendDecision(
        status=status,
        predictor=predictor,
        method=selected.method,
        catboost_weight=selected.catboost_weight,
        f2_selection_fold=FOLD_ORDER[0],
        f2_applied_method=f2_candidate.method,
        f2_catboost_weight=f2_candidate.catboost_weight,
        f3_selection_folds=FOLD_ORDER[:2],
        f3_applied_method=f3_candidate.method,
        f3_catboost_weight=f3_candidate.catboost_weight,
        sequential_gain=sequential_gain,
        fold_gains_vs_catboost=dict(fold_gains),
        reason=reason,
    )


def _bootstrap(
    gains: np.ndarray,
    groups: Sequence[object],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    group_series = pd.Series(list(groups), dtype=object)
    if len(group_series) != len(gains) or group_series.isna().any():
        raise E2DecisionError("bootstrap group alignment differs")
    codes, unique = pd.factorize(group_series, sort=False)
    if len(unique) == 0:
        raise E2DecisionError("bootstrap groups are empty")
    sums = np.bincount(codes, weights=gains, minlength=len(unique))
    counts = np.bincount(codes, minlength=len(unique))
    generator = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype="float64")
    for index in range(repeats):
        chosen = generator.integers(0, len(unique), size=len(unique))
        samples[index] = sums[chosen].sum() / counts[chosen].sum()
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return float(lower), float(upper)


def _segment_regression(
    baselines: Mapping[tuple[int, int], pd.DataFrame],
    candidates: Mapping[tuple[int, int], pd.DataFrame],
    *,
    minimum_rows: int,
) -> float:
    pooled: list[pd.DataFrame] = []
    maximum = 0.0
    for fold in FOLD_ORDER:
        base, trial = _aligned(baselines[fold], candidates[fold])
        working = base.copy(deep=True)
        target = base["target"].to_numpy(dtype="float64")
        working["baseline_loss"] = np.square(
            base["probability"].to_numpy(dtype="float64") - target
        )
        working["candidate_loss"] = np.square(
            trial["probability"].to_numpy(dtype="float64") - target
        )
        pooled.append(working)
        for column in PREDICTION_COLUMNS[3:]:
            for _, group in working.groupby(column, dropna=False, sort=True):
                if len(group) >= minimum_rows:
                    maximum = max(
                        maximum,
                        float((group["candidate_loss"] - group["baseline_loss"]).mean()),
                    )
    combined = pd.concat(pooled, ignore_index=True)
    for column in PREDICTION_COLUMNS[3:]:
        for _, group in combined.groupby(column, dropna=False, sort=True):
            if len(group) >= minimum_rows:
                maximum = max(
                    maximum,
                    float((group["candidate_loss"] - group["baseline_loss"]).mean()),
                )
    return maximum


def decide_acceptance(
    scores: Sequence[FoldScore],
    baselines: Mapping[tuple[int, int], pd.DataFrame],
    candidates: Mapping[tuple[int, int], pd.DataFrame],
    pitcher_groups: Mapping[tuple[int, int], Sequence[object]],
    contract: E2Contract,
    *,
    blend: BlendDecision | None,
) -> AcceptanceDecision:
    ordered = _ordered_scores(scores)
    if (
        set(baselines) != set(FOLD_ORDER)
        or set(candidates) != set(FOLD_ORDER)
        or set(pitcher_groups) != set(FOLD_ORDER)
    ):
        raise E2DecisionError("acceptance fold evidence differs")
    per_row_gains: list[np.ndarray] = []
    groups: list[object] = []
    for fold in FOLD_ORDER:
        base, trial = _aligned(baselines[fold], candidates[fold])
        target = base["target"].to_numpy(dtype="float64")
        gains = np.square(base["probability"].to_numpy(dtype="float64") - target) - np.square(
            trial["probability"].to_numpy(dtype="float64") - target
        )
        fold_groups = list(pitcher_groups[fold])
        if len(fold_groups) != len(gains):
            raise E2DecisionError("pitcher group alignment differs")
        per_row_gains.append(gains)
        groups.extend(fold_groups)
    lower, upper = _bootstrap(
        np.concatenate(per_row_gains),
        groups,
        repeats=int(contract.gates["bootstrap_repeats"]),
        seed=int(contract.gates["bootstrap_seed"]),
    )
    weighted = _weighted_gain(ordered)
    f3_gain = ordered[-1].gain
    worst = min(score.gain for score in ordered)
    segment_regression = _segment_regression(
        baselines,
        candidates,
        minimum_rows=int(contract.gates["minimum_segment_rows"]),
    )
    if weighted < float(contract.gates["accept_weighted_gain"]):
        status, reason = "rejected", "weighted_gain_below_gate"
    elif f3_gain < float(contract.gates["accept_f3_gain"]):
        status, reason = "rejected", "f3_gain_below_gate"
    elif worst < -float(contract.gates["accept_max_fold_regression"]):
        status, reason = "rejected", "fold_regression_above_gate"
    elif lower <= 0.0:
        status, reason = "rejected", "bootstrap_lower_not_positive"
    elif segment_regression > float(contract.gates["accept_max_segment_regression"]):
        status, reason = "rejected", "segment_regression_above_gate"
    else:
        status, reason = "accepted", "standalone_catboost_gates_passed"
    predictor = (
        "blend"
        if status == "accepted" and blend is not None and blend.status == "passed"
        else "catboost"
    )
    grade = (
        "breakthrough"
        if weighted >= 0.00045
        else "competitive"
        if weighted >= 0.00025
        else "incremental"
        if weighted >= 0.00015
        else "below_incremental"
    )
    return AcceptanceDecision(
        status=status,
        predictor=predictor,
        weighted_gain=weighted,
        f3_gain=f3_gain,
        worst_fold_gain=worst,
        bootstrap_lower=lower,
        bootstrap_upper=upper,
        maximum_segment_regression=segment_regression,
        performance_grade=grade,
        reason=reason,
    )
