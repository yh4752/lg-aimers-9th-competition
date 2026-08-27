from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .rf_contracts import RFContract


class RFDecisionError(ValueError):
    pass


Fold = tuple[int, int]


@dataclass(frozen=True)
class RFStructureEvidence:
    target: Mapping[Fold, np.ndarray]
    baseline: Mapping[Fold, np.ndarray]
    game_type: Mapping[Fold, np.ndarray]
    expert: Mapping[str, Mapping[Fold, np.ndarray]]


@dataclass(frozen=True)
class RFStructureDecision:
    status: str
    reason: str
    f_head: str
    include_r: bool
    alpha_r: float
    alpha_f: float
    selection_key: str
    selection_brier: float
    fold_gains: Mapping[Fold, float]
    segment_gains: Mapping[str, float]
    recent_f_gain: float
    weighted_gain: float
    maximum_segment_regression: float


@dataclass(frozen=True)
class RFAcceptanceEvidence:
    structure: RFStructureDecision
    target: Mapping[Fold, np.ndarray]
    baseline: Mapping[Fold, np.ndarray]
    game_type: Mapping[Fold, np.ndarray]
    seed_candidate: Mapping[int, Mapping[Fold, np.ndarray]]


@dataclass(frozen=True)
class RFAcceptanceDecision:
    status: str
    reason: str
    f_head: str
    include_r: bool
    alpha_r: float
    alpha_f: float
    fold_gains: Mapping[Fold, float]
    segment_gains: Mapping[str, float]
    recent_f_gain: float
    weighted_gain: float
    improved_fold_count: int
    maximum_segment_regression: float


def _vector(value: object, length: int | None = None, *, probability: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype="float64")
    if (
        result.ndim != 1
        or not np.isfinite(result).all()
        or (length is not None and len(result) != length)
        or (probability and np.any((result < 0) | (result > 1)))
    ):
        raise RFDecisionError("prediction vectors differ")
    return result


def _segments(value: object, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=str)
    if result.ndim != 1 or len(result) != length or not np.isin(result, ["R", "F"]).all():
        raise RFDecisionError("game_type differs")
    return result


def _alpha(value: object) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise RFDecisionError("alpha differs")
    result = float(value)
    if not np.isfinite(result) or result < 0 or result > 1:
        raise RFDecisionError("alpha differs")
    return result


def _brier(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target)))


def route_probability(
    baseline: np.ndarray,
    r_probability: np.ndarray,
    f_probability: np.ndarray,
    game_type: np.ndarray,
    alpha_r: float,
    alpha_f: float,
) -> np.ndarray:
    base = _vector(baseline, probability=True)
    r_pred = _vector(r_probability, len(base), probability=True)
    f_pred = _vector(f_probability, len(base), probability=True)
    segment = _segments(game_type, len(base))
    r_weight = _alpha(alpha_r)
    f_weight = _alpha(alpha_f)
    weights = np.where(segment == "R", r_weight, f_weight)
    expert = np.where(segment == "R", r_pred, f_pred)
    return np.clip((1.0 - weights) * base + weights * expert, 1e-5, 1 - 1e-5)


def _validate_structure(evidence: RFStructureEvidence, contract: RFContract) -> None:
    if type(evidence) is not RFStructureEvidence:
        raise RFDecisionError("structure evidence type differs")
    if set(evidence.target) != set(contract.folds):
        raise RFDecisionError("structure folds differ")
    if set(evidence.baseline) != set(contract.folds) or set(evidence.game_type) != set(contract.folds):
        raise RFDecisionError("structure folds differ")
    if set(evidence.expert) != {"f_small", "f_wide", "r_expert"}:
        raise RFDecisionError("structure heads differ")
    for head in evidence.expert:
        if set(evidence.expert[head]) != set(contract.folds):
            raise RFDecisionError("structure head folds differ")
    for fold in contract.folds:
        target = _vector(evidence.target[fold])
        if not np.isin(target, [0.0, 1.0]).all():
            raise RFDecisionError("target values differ")
        _vector(evidence.baseline[fold], len(target), probability=True)
        _segments(evidence.game_type[fold], len(target))
        for head in evidence.expert:
            _vector(evidence.expert[head][fold], len(target), probability=True)


def screen_structure_heads(evidence: RFStructureEvidence, contract: RFContract) -> tuple[str, ...]:
    _validate_structure(evidence, contract)
    survivors: list[str] = []
    for head in ("f_small", "f_wide", "r_expert"):
        segment_name = "F" if head.startswith("f_") else "R"
        useful = False
        for fold in contract.folds[:2]:
            target = _vector(evidence.target[fold])
            baseline = _vector(evidence.baseline[fold], len(target), probability=True)
            segment = _segments(evidence.game_type[fold], len(target))
            expert = _vector(evidence.expert[head][fold], len(target), probability=True)
            mask = segment == segment_name
            if not mask.any():
                raise RFDecisionError("selection segment is empty")
            base_score = _brier(target[mask], baseline[mask])
            for alpha in contract.alpha_values:
                blended = (1.0 - alpha) * baseline[mask] + alpha * expert[mask]
                if _brier(target[mask], blended) < base_score:
                    useful = True
                    break
            if useful:
                break
        if useful:
            survivors.append(head)
    return tuple(survivors)


def _candidate(
    evidence: RFStructureEvidence,
    fold: Fold,
    f_head: str,
    alpha_r: float,
    alpha_f: float,
) -> np.ndarray:
    baseline = evidence.baseline[fold]
    return route_probability(
        baseline,
        evidence.expert["r_expert"][fold] if alpha_r > 0 else baseline,
        evidence.expert[f_head][fold],
        evidence.game_type[fold],
        alpha_r,
        alpha_f,
    )


def _joined_brier(evidence: RFStructureEvidence, folds: tuple[Fold, ...], f_head: str, alpha_r: float, alpha_f: float) -> float:
    target = np.concatenate([_vector(evidence.target[fold]) for fold in folds])
    prediction = np.concatenate([
        _candidate(evidence, fold, f_head, alpha_r, alpha_f) for fold in folds
    ])
    return _brier(target, prediction)


def _metrics(
    target: Mapping[Fold, np.ndarray],
    baseline: Mapping[Fold, np.ndarray],
    game_type: Mapping[Fold, np.ndarray],
    candidate: Mapping[Fold, np.ndarray],
    folds: tuple[Fold, ...],
) -> tuple[dict[Fold, float], dict[str, float], float, float]:
    fold_gains: dict[Fold, float] = {}
    for fold in folds:
        y = _vector(target[fold])
        base = _vector(baseline[fold], len(y), probability=True)
        pred = _vector(candidate[fold], len(y), probability=True)
        fold_gains[fold] = _brier(y, base) - _brier(y, pred)
    joined_target = np.concatenate([_vector(target[fold]) for fold in folds])
    joined_base = np.concatenate([_vector(baseline[fold], len(target[fold]), probability=True) for fold in folds])
    joined_pred = np.concatenate([_vector(candidate[fold], len(target[fold]), probability=True) for fold in folds])
    joined_segment = np.concatenate([_segments(game_type[fold], len(target[fold])) for fold in folds])
    weighted_gain = _brier(joined_target, joined_base) - _brier(joined_target, joined_pred)
    segment_gains = {
        name: _brier(joined_target[joined_segment == name], joined_base[joined_segment == name])
        - _brier(joined_target[joined_segment == name], joined_pred[joined_segment == name])
        for name in ("R", "F")
    }
    recent = folds[-1]
    recent_target = _vector(target[recent])
    recent_segment = _segments(game_type[recent], len(recent_target))
    recent_mask = recent_segment == "F"
    if not recent_mask.any():
        raise RFDecisionError("recent F segment is empty")
    recent_base = _vector(baseline[recent], len(recent_target), probability=True)
    recent_pred = _vector(candidate[recent], len(recent_target), probability=True)
    recent_f_gain = _brier(recent_target[recent_mask], recent_base[recent_mask]) - _brier(
        recent_target[recent_mask], recent_pred[recent_mask]
    )
    return fold_gains, segment_gains, weighted_gain, recent_f_gain


def _gate_reason(
    *,
    weighted_gain: float,
    fold_gains: Mapping[Fold, float],
    segment_gains: Mapping[str, float],
    recent_f_gain: float,
    contract: RFContract,
) -> str | None:
    if min(segment_gains.values()) < -contract.gates.maximum_segment_regression:
        return "segment_regression_failed"
    if weighted_gain < contract.gates.weighted_gain:
        return "weighted_gain_failed"
    if sum(value > 0 for value in fold_gains.values()) < contract.gates.minimum_improved_folds:
        return "improved_fold_count_failed"
    if recent_f_gain <= contract.gates.recent_f_gain:
        return "recent_f_gain_failed"
    return None


def select_rf_structure(evidence: RFStructureEvidence, contract: RFContract) -> RFStructureDecision:
    _validate_structure(evidence, contract)
    scored: list[tuple[tuple[float, int, float, int, float, float], str, float, float]] = []
    for f_head in ("f_small", "f_wide"):
        for alpha_f in contract.alpha_values:
            score = _joined_brier(evidence, contract.folds[:2], f_head, 0.0, alpha_f)
            scored.append(((score, 0 if f_head == "f_small" else 1, alpha_f, 0, 0.0, alpha_f), f_head, 0.0, alpha_f))
        for alpha_r in contract.alpha_values:
            for alpha_f in contract.alpha_values:
                score = _joined_brier(evidence, contract.folds[:2], f_head, alpha_r, alpha_f)
                scored.append((
                    (score, 0 if f_head == "f_small" else 1, alpha_r + alpha_f, 1, alpha_r, alpha_f),
                    f_head,
                    alpha_r,
                    alpha_f,
                ))
    key, f_head, alpha_r, alpha_f = min(scored, key=lambda item: item[0])
    candidate = {
        fold: _candidate(evidence, fold, f_head, alpha_r, alpha_f)
        for fold in contract.folds
    }
    fold_gains, segment_gains, weighted_gain, recent_f_gain = _metrics(
        evidence.target, evidence.baseline, evidence.game_type, candidate, contract.folds,
    )
    reason = _gate_reason(
        weighted_gain=weighted_gain,
        fold_gains=fold_gains,
        segment_gains=segment_gains,
        recent_f_gain=recent_f_gain,
        contract=contract,
    )
    return RFStructureDecision(
        status="passed" if reason is None else "rejected",
        reason="structure_gates_passed" if reason is None else reason,
        f_head=f_head,
        include_r=alpha_r > 0,
        alpha_r=alpha_r,
        alpha_f=alpha_f,
        selection_key=f"{f_head}|r={alpha_r:.2f}|f={alpha_f:.2f}",
        selection_brier=float(key[0]),
        fold_gains=MappingProxyType(fold_gains),
        segment_gains=MappingProxyType(segment_gains),
        recent_f_gain=recent_f_gain,
        weighted_gain=weighted_gain,
        maximum_segment_regression=max(0.0, -min(segment_gains.values())),
    )


def accept_rf(evidence: RFAcceptanceEvidence, contract: RFContract) -> RFAcceptanceDecision:
    if type(evidence) is not RFAcceptanceEvidence:
        raise RFDecisionError("acceptance evidence type differs")
    expected_seeds = {contract.structure_seed, *contract.confirmation_seeds}
    if set(evidence.seed_candidate) != expected_seeds:
        raise RFDecisionError("acceptance seeds differ")
    if any(set(value) != set(contract.folds) for value in evidence.seed_candidate.values()):
        raise RFDecisionError("acceptance folds differ")
    candidate: dict[Fold, np.ndarray] = {}
    for fold in contract.folds:
        target = _vector(evidence.target[fold])
        seed_predictions = [
            _vector(evidence.seed_candidate[seed][fold], len(target), probability=True)
            for seed in sorted(expected_seeds)
        ]
        candidate[fold] = np.mean(seed_predictions, axis=0)
    fold_gains, segment_gains, weighted_gain, recent_f_gain = _metrics(
        evidence.target, evidence.baseline, evidence.game_type, candidate, contract.folds,
    )
    reason = _gate_reason(
        weighted_gain=weighted_gain,
        fold_gains=fold_gains,
        segment_gains=segment_gains,
        recent_f_gain=recent_f_gain,
        contract=contract,
    )
    if evidence.structure.status != "passed":
        reason = "structure_not_passed"
    improved = sum(value > 0 for value in fold_gains.values())
    return RFAcceptanceDecision(
        status="accepted" if reason is None else "rejected",
        reason="all_rf_gates_passed" if reason is None else reason,
        f_head=evidence.structure.f_head,
        include_r=evidence.structure.include_r,
        alpha_r=evidence.structure.alpha_r,
        alpha_f=evidence.structure.alpha_f,
        fold_gains=MappingProxyType(fold_gains),
        segment_gains=MappingProxyType(segment_gains),
        recent_f_gain=recent_f_gain,
        weighted_gain=weighted_gain,
        improved_fold_count=improved,
        maximum_segment_regression=max(0.0, -min(segment_gains.values())),
    )


def structure_payload(decision: RFStructureDecision) -> dict[str, object]:
    return {
        "status": decision.status,
        "reason": decision.reason,
        "f_head": decision.f_head,
        "include_r": decision.include_r,
        "alpha_r": decision.alpha_r,
        "alpha_f": decision.alpha_f,
        "selection_key": decision.selection_key,
        "selection_brier": decision.selection_brier,
        "fold_gains": {f"{a}->{b}": value for (a, b), value in decision.fold_gains.items()},
        "segment_gains": dict(decision.segment_gains),
        "recent_f_gain": decision.recent_f_gain,
        "weighted_gain": decision.weighted_gain,
        "maximum_segment_regression": decision.maximum_segment_regression,
    }


def acceptance_payload(decision: RFAcceptanceDecision) -> dict[str, object]:
    return {
        "status": decision.status,
        "reason": decision.reason,
        "f_head": decision.f_head,
        "include_r": decision.include_r,
        "alpha_r": decision.alpha_r,
        "alpha_f": decision.alpha_f,
        "fold_gains": {f"{a}->{b}": value for (a, b), value in decision.fold_gains.items()},
        "segment_gains": dict(decision.segment_gains),
        "recent_f_gain": decision.recent_f_gain,
        "weighted_gain": decision.weighted_gain,
        "improved_fold_count": decision.improved_fold_count,
        "maximum_segment_regression": decision.maximum_segment_regression,
    }
