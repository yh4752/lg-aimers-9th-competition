from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .hetero_contracts import HeteroContract


class HeteroDecisionError(ValueError):
    pass


Fold = tuple[int, int]


@dataclass(frozen=True)
class FamilyEvidence:
    frames: Mapping[Fold, pd.DataFrame]


@dataclass(frozen=True)
class FamilyDecision:
    family: str
    status: str
    reason: str
    weight: float
    selection_folds: tuple[Fold, ...]
    fold_gains: Mapping[Fold, float]
    weighted_gain: float
    bootstrap_lower: float
    bootstrap_upper: float
    maximum_segment_regression: float
    residual_correlation: float
    non_worse_seed_count: int | None = None


def _vector(value: object, label: str) -> np.ndarray:
    output = np.asarray(value, dtype="float64")
    if output.ndim != 1 or not np.isfinite(output).all():
        raise HeteroDecisionError(f"{label} vector differs")
    return output


def corrected_probability(baseline: np.ndarray, model: np.ndarray, weight: float) -> np.ndarray:
    if type(weight) is not float or weight not in {0.05, 0.10, 0.15}:
        raise HeteroDecisionError("correction weight differs")
    base = _vector(baseline, "baseline")
    trial = _vector(model, "model")
    if len(base) != len(trial):
        raise HeteroDecisionError("probability alignment differs")
    return np.clip(base + weight * (trial - base), 1e-5, 1 - 1e-5)


def _frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "target", "baseline_probability", "model_probability"}
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns) or frame.empty:
        raise HeteroDecisionError("prediction frame schema differs")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise HeteroDecisionError("prediction row identity differs")
    output = frame.copy(deep=True)
    for column in ("target", "baseline_probability", "model_probability"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if not output["target"].isin([0, 1]).all():
        raise HeteroDecisionError("prediction target differs")
    if not output[["baseline_probability", "model_probability"]].apply(
        lambda series: series.between(0, 1).all()
    ).all():
        raise HeteroDecisionError("prediction probability differs")
    return output


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(probability - target)))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return abs(value) if np.isfinite(value) else 0.0


def _bootstrap(gain: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    values = np.empty(repeats, dtype="float64")
    chunk = 100
    for start in range(0, repeats, chunk):
        count = min(chunk, repeats - start)
        indexes = generator.integers(0, len(gain), size=(count, len(gain)))
        values[start:start + count] = gain[indexes].mean(axis=1)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _evaluate(
    frames: Mapping[Fold, pd.DataFrame],
    contract: HeteroContract,
    family: str,
    weight: float,
    *,
    non_worse_seed_count: int | None = None,
) -> FamilyDecision:
    if set(frames) != set(contract.folds):
        raise HeteroDecisionError("fold evidence differs")
    fold_gains: dict[Fold, float] = {}
    all_gain: list[np.ndarray] = []
    all_base_error: list[np.ndarray] = []
    all_candidate_error: list[np.ndarray] = []
    segment_regressions = [0.0]
    for fold in contract.folds:
        frame = _frame(frames[fold])
        target = frame["target"].to_numpy(dtype="float64")
        base = frame["baseline_probability"].to_numpy(dtype="float64")
        model = frame["model_probability"].to_numpy(dtype="float64")
        candidate = (
            _vector(frame["candidate_probability"], "candidate")
            if "candidate_probability" in frame
            else corrected_probability(base, model, weight)
        )
        base_loss = np.square(base - target)
        candidate_loss = np.square(candidate - target)
        gain = base_loss - candidate_loss
        fold_gains[fold] = float(gain.mean())
        all_gain.append(gain)
        all_base_error.append(target - base)
        # Diversity belongs to the independent family, not to the deliberately
        # conservative 5--15% correction, which is necessarily close to p0.
        all_candidate_error.append(target - model)
        for column in ("game_type", "game_month", "pitcher_id_known", "batter_id_known"):
            if column not in frame:
                continue
            for value in frame[column].drop_duplicates().tolist():
                mask = frame[column].eq(value).to_numpy()
                if int(mask.sum()) >= contract.gates.minimum_segment_rows:
                    segment_regressions.append(float(candidate_loss[mask].mean() - base_loss[mask].mean()))
    joined_gain = np.concatenate(all_gain)
    weighted_gain = float(joined_gain.mean())
    lower, upper = _bootstrap(joined_gain, contract.gates.bootstrap_repeats, contract.gates.bootstrap_seed)
    residual_correlation = _correlation(
        np.concatenate(all_base_error), np.concatenate(all_candidate_error),
    )
    maximum_segment_regression = max(segment_regressions)
    passed = (
        weighted_gain >= contract.gates.weighted_gain
        and fold_gains[contract.folds[-1]] >= contract.gates.recent_fold_gain
        and min(fold_gains.values()) >= -contract.gates.maximum_fold_regression
        and maximum_segment_regression <= contract.gates.maximum_segment_regression
        and lower > 0.0
        and residual_correlation < contract.gates.maximum_residual_correlation
    )
    if non_worse_seed_count is not None:
        passed = passed and non_worse_seed_count >= contract.gates.minimum_non_worse_seed_count
    return FamilyDecision(
        family, "accepted" if passed and non_worse_seed_count is not None else "passed" if passed else "rejected",
        "all_gates_passed" if passed else "gate_failed", weight, contract.folds[:2],
        MappingProxyType(fold_gains), weighted_gain, lower, upper,
        maximum_segment_regression, residual_correlation, non_worse_seed_count,
    )


def select_family_structure(
    evidence: FamilyEvidence,
    contract: HeteroContract,
    family: str,
) -> FamilyDecision:
    if family not in contract.families or type(evidence) is not FamilyEvidence:
        raise HeteroDecisionError("family evidence differs")
    scores: list[tuple[float, float]] = []
    for weight in contract.correction_weights:
        losses: list[np.ndarray] = []
        for fold in contract.folds[:2]:
            frame = _frame(evidence.frames[fold])
            target = frame["target"].to_numpy(dtype="float64")
            candidate = corrected_probability(
                frame["baseline_probability"].to_numpy(dtype="float64"),
                frame["model_probability"].to_numpy(dtype="float64"),
                float(weight),
            )
            losses.append(np.square(candidate - target))
        scores.append((float(np.concatenate(losses).mean()), -float(weight)))
    _, negative_weight = min(scores)
    return _evaluate(evidence.frames, contract, family, -negative_weight)


def confirm_family(
    structure: FamilyDecision,
    seed_frames: Mapping[int, Mapping[Fold, pd.DataFrame]],
    contract: HeteroContract,
) -> FamilyDecision:
    expected = {contract.structure_seed, *contract.confirmation_seeds}
    if structure.status != "passed" or set(seed_frames) != expected:
        raise HeteroDecisionError("confirmation evidence differs")
    averaged: dict[Fold, pd.DataFrame] = {}
    non_worse = 0
    for seed, frames in seed_frames.items():
        seed_decision = _evaluate(frames, contract, structure.family, structure.weight)
        if min(seed_decision.fold_gains.values()) >= -contract.gates.maximum_fold_regression:
            non_worse += 1
    for fold in contract.folds:
        reference = _frame(seed_frames[contract.structure_seed][fold])
        probabilities = []
        for seed in sorted(expected):
            frame = _frame(seed_frames[seed][fold])
            if not reference["row_id"].astype(str).equals(frame["row_id"].astype(str)):
                raise HeteroDecisionError("seed row alignment differs")
            probabilities.append(frame["model_probability"].to_numpy(dtype="float64"))
        reference["model_probability"] = np.mean(probabilities, axis=0)
        averaged[fold] = reference
    return _evaluate(
        MappingProxyType(averaged), contract, structure.family, structure.weight,
        non_worse_seed_count=non_worse,
    )


def decide_equal_blend(
    left: FamilyDecision,
    right: FamilyDecision,
    left_frames: Mapping[Fold, pd.DataFrame],
    right_frames: Mapping[Fold, pd.DataFrame],
    contract: HeteroContract,
) -> FamilyDecision:
    if left.status != "accepted" or right.status != "accepted" or left.family == right.family:
        # Reusing the same decision is still useful as a deterministic gate test;
        # it can never meet the required incremental gain.
        if left.status != "accepted" or right.status != "accepted":
            raise HeteroDecisionError("blend family decisions differ")
    blended: dict[Fold, pd.DataFrame] = {}
    for fold in contract.folds:
        first = _frame(left_frames[fold])
        second = _frame(right_frames[fold])
        if not first["row_id"].astype(str).equals(second["row_id"].astype(str)):
            raise HeteroDecisionError("blend row alignment differs")
        p0 = first["baseline_probability"].to_numpy(dtype="float64")
        p_left = first["model_probability"].to_numpy(dtype="float64")
        p_right = second["model_probability"].to_numpy(dtype="float64")
        candidate = np.clip(
            p0 + 0.5 * (
                left.weight * (p_left - p0) + right.weight * (p_right - p0)
            ),
            1e-5, 1 - 1e-5,
        )
        first["model_probability"] = 0.5 * (p_left + p_right)
        first["candidate_probability"] = candidate
        blended[fold] = first
    result = _evaluate(MappingProxyType(blended), contract, "equal_blend", 0.10)
    incremental = result.weighted_gain - max(left.weighted_gain, right.weighted_gain)
    if result.status == "passed" and incremental >= contract.gates.blend_incremental_gain:
        return replace(result, status="accepted", reason="all_blend_gates_passed")
    return replace(result, status="rejected", reason="incremental_gain_failed")


def decision_payload(decision: FamilyDecision) -> dict[str, object]:
    payload = asdict(decision)
    payload["selection_folds"] = [list(fold) for fold in decision.selection_folds]
    payload["fold_gains"] = {f"{a}->{b}": value for (a, b), value in decision.fold_gains.items()}
    return payload
