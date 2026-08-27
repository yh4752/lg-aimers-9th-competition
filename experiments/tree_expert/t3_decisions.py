from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .t3_contracts import T3Contract


class T3DecisionError(ValueError):
    pass


Fold = tuple[int, int]


@dataclass(frozen=True)
class T3StructureEvidence:
    target: Mapping[Fold, np.ndarray]
    baseline: Mapping[Fold, np.ndarray]
    recent: Mapping[Fold, np.ndarray]
    multi: Mapping[float, Mapping[Fold, np.ndarray]]
    maximum_segment_regression: float


@dataclass(frozen=True)
class T3StructureDecision:
    status: str
    reason: str
    decay: float
    recent_weight: float
    selection_folds: tuple[Fold, ...]
    confirmation_fold: Fold
    selection_scores: Mapping[str, float]
    fold_gains: Mapping[Fold, float]
    weighted_gain: float
    maximum_segment_regression: float


@dataclass(frozen=True)
class T3AcceptanceEvidence:
    structure: T3StructureDecision
    seed_fold_gains: Mapping[int, Mapping[Fold, float]]


@dataclass(frozen=True)
class T3AcceptanceDecision:
    status: str
    reason: str
    decay: float
    recent_weight: float
    weighted_gain: float
    fold_gains: Mapping[Fold, float]
    maximum_segment_regression: float
    non_worse_seed_count: int


def _vector(value: object, length: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype="float64")
    if result.ndim != 1 or not np.isfinite(result).all() or (length is not None and len(result) != length):
        raise T3DecisionError("prediction vectors differ")
    return result


def _brier(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target)))


def blended_probability(recent: np.ndarray, multi: np.ndarray, recent_weight: float) -> np.ndarray:
    if type(recent_weight) is not float or recent_weight not in {0.70, 0.80, 0.90}:
        raise T3DecisionError("recent weight differs")
    recent = _vector(recent)
    multi = _vector(multi, len(recent))
    return np.clip(recent_weight * recent + (1.0 - recent_weight) * multi, 1e-5, 1 - 1e-5)


def _score(evidence: T3StructureEvidence, folds: tuple[Fold, ...], decay: float, weight: float) -> float:
    squared: list[np.ndarray] = []
    for fold in folds:
        target = _vector(evidence.target[fold])
        prediction = blended_probability(evidence.recent[fold], evidence.multi[decay][fold], weight)
        squared.append(np.square(prediction - target))
    return float(np.mean(np.concatenate(squared)))


def select_structure(evidence: T3StructureEvidence, contract: T3Contract) -> T3StructureDecision:
    if type(evidence) is not T3StructureEvidence:
        raise T3DecisionError("structure evidence type differs")
    selection_folds = contract.folds[:2]
    scored: list[tuple[float, float, float]] = []
    scores: dict[str, float] = {}
    for decay in contract.decays:
        for weight in contract.recent_weights:
            score = _score(evidence, selection_folds, decay, weight)
            scores[f"d{int(decay * 100):03d}_r{int(weight * 100):03d}"] = score
            scored.append((score, -weight, abs(decay - 0.55)))
    best_score, negative_weight, distance = min(scored)
    weight = -negative_weight
    matching = [decay for decay in contract.decays if abs(decay - 0.55) == distance and _score(evidence, selection_folds, decay, weight) == best_score]
    decay = min(matching)
    fold_gains: dict[Fold, float] = {}
    total_base: list[np.ndarray] = []
    total_candidate: list[np.ndarray] = []
    total_target: list[np.ndarray] = []
    for fold in contract.folds:
        target = _vector(evidence.target[fold])
        baseline = _vector(evidence.baseline[fold], len(target))
        candidate = blended_probability(evidence.recent[fold], evidence.multi[decay][fold], weight)
        fold_gains[fold] = _brier(target, baseline) - _brier(target, candidate)
        total_base.append(baseline)
        total_candidate.append(candidate)
        total_target.append(target)
    joined_target = np.concatenate(total_target)
    weighted_gain = _brier(joined_target, np.concatenate(total_base)) - _brier(joined_target, np.concatenate(total_candidate))
    passed = (
        weighted_gain >= contract.gates.weighted_gain
        and fold_gains[contract.folds[-1]] > contract.gates.recent_fold_gain
        and min(fold_gains.values()) >= -contract.gates.maximum_fold_regression
        and evidence.maximum_segment_regression <= contract.gates.maximum_segment_regression
    )
    return T3StructureDecision(
        status="passed" if passed else "rejected",
        reason="structure_gates_passed" if passed else "structure_gate_failed",
        decay=decay,
        recent_weight=weight,
        selection_folds=selection_folds,
        confirmation_fold=contract.folds[-1],
        selection_scores=MappingProxyType(scores),
        fold_gains=MappingProxyType(fold_gains),
        weighted_gain=weighted_gain,
        maximum_segment_regression=float(evidence.maximum_segment_regression),
    )


def accept_t3(evidence: T3AcceptanceEvidence, contract: T3Contract) -> T3AcceptanceDecision:
    if type(evidence) is not T3AcceptanceEvidence:
        raise T3DecisionError("acceptance evidence type differs")
    expected_seeds = {contract.structure_seed, *contract.confirmation_seeds}
    if set(evidence.seed_fold_gains) != expected_seeds:
        raise T3DecisionError("acceptance seed evidence differs")
    fold_gains = {
        fold: float(np.mean([evidence.seed_fold_gains[seed][fold] for seed in sorted(expected_seeds)]))
        for fold in contract.folds
    }
    weighted_gain = float(np.mean(tuple(fold_gains.values())))
    non_worse = sum(
        min(evidence.seed_fold_gains[seed].values()) >= -contract.gates.maximum_fold_regression
        for seed in expected_seeds
    )
    accepted = (
        evidence.structure.status == "passed"
        and weighted_gain >= contract.gates.weighted_gain
        and fold_gains[contract.folds[-1]] > contract.gates.recent_fold_gain
        and min(fold_gains.values()) >= -contract.gates.maximum_fold_regression
        and evidence.structure.maximum_segment_regression <= contract.gates.maximum_segment_regression
        and non_worse >= contract.gates.minimum_non_worse_seed_count
    )
    return T3AcceptanceDecision(
        status="accepted" if accepted else "rejected",
        reason="all_t3_gates_passed" if accepted else "acceptance_gate_failed",
        decay=evidence.structure.decay,
        recent_weight=evidence.structure.recent_weight,
        weighted_gain=weighted_gain,
        fold_gains=MappingProxyType(fold_gains),
        maximum_segment_regression=evidence.structure.maximum_segment_regression,
        non_worse_seed_count=non_worse,
    )


def structure_payload(decision: T3StructureDecision) -> dict[str, object]:
    return {
        "status": decision.status, "reason": decision.reason,
        "decay": decision.decay, "recent_weight": decision.recent_weight,
        "selection_folds": [list(item) for item in decision.selection_folds],
        "confirmation_fold": list(decision.confirmation_fold),
        "selection_scores": dict(decision.selection_scores),
        "fold_gains": {f"{a}->{b}": value for (a, b), value in decision.fold_gains.items()},
        "weighted_gain": decision.weighted_gain,
        "maximum_segment_regression": decision.maximum_segment_regression,
    }


def acceptance_payload(decision: T3AcceptanceDecision) -> dict[str, object]:
    return {
        "status": decision.status, "reason": decision.reason,
        "decay": decision.decay, "recent_weight": decision.recent_weight,
        "weighted_gain": decision.weighted_gain,
        "fold_gains": {f"{a}->{b}": value for (a, b), value in decision.fold_gains.items()},
        "maximum_segment_regression": decision.maximum_segment_regression,
        "non_worse_seed_count": decision.non_worse_seed_count,
    }
