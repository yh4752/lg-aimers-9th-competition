from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import math
from typing import Mapping

import numpy as np
import pandas as pd

from experiments.catboost_tabm_blend.metrics import (
    FOLD_KEYS,
    PREDICTION_COLUMNS,
    SEGMENT_COLUMNS,
)

from .contracts import DeploymentContract


class DeploymentMetricError(ValueError):
    """Raised when fixed-prefix OOF evidence is invalid or misaligned."""


CATBOOST_PREDICTION_COLUMNS = (
    "row_id",
    "target",
    "p_4",
    "p_32",
    "p_64",
    "p_128",
    "p_192",
    "p_296",
    "p_400",
    *SEGMENT_COLUMNS,
)


@dataclass(frozen=True)
class PrefixCandidate:
    tree_count: int
    fold_brier: Mapping[str, float]
    fold_regression: Mapping[str, float]
    weighted_brier: float
    weighted_gain: float
    passed: bool


@dataclass(frozen=True)
class DeploymentDecision:
    status: str
    selected_tree_count: int | None
    baseline_weighted_brier: float
    candidates: tuple[PrefixCandidate, ...]
    reason: str


@dataclass(frozen=True)
class _AlignedFold:
    target: np.ndarray
    tabm: np.ndarray
    catboost_by_prefix: Mapping[int, np.ndarray]


def _numeric(values: pd.Series, label: str) -> np.ndarray:
    result = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(result).all():
        raise DeploymentMetricError(f"{label} probability is invalid")
    return result


def _validate_tabm(frame: pd.DataFrame, fold: str) -> pd.DataFrame:
    if tuple(frame.columns) != PREDICTION_COLUMNS or frame.empty:
        raise DeploymentMetricError(f"TabM {fold} prediction schema differs")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise DeploymentMetricError(f"TabM {fold} row_id is invalid")
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype="float64")
    probability = _numeric(frame["probability"], f"TabM {fold}")
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise DeploymentMetricError(f"TabM {fold} target is invalid")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise DeploymentMetricError(f"TabM {fold} probability is invalid")
    if any(frame[column].isna().any() for column in SEGMENT_COLUMNS):
        raise DeploymentMetricError(f"TabM {fold} segment is invalid")
    return frame


def _validate_catboost(frame: pd.DataFrame, fold: str) -> pd.DataFrame:
    if tuple(frame.columns) != CATBOOST_PREDICTION_COLUMNS or frame.empty:
        raise DeploymentMetricError(f"CatBoost {fold} prediction schema differs")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise DeploymentMetricError(f"CatBoost {fold} row_id is invalid")
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise DeploymentMetricError(f"CatBoost {fold} target is invalid")
    for column in CATBOOST_PREDICTION_COLUMNS[2:9]:
        probability = _numeric(frame[column], f"CatBoost {fold}")
        if np.any(probability < 0.0) or np.any(probability > 1.0):
            raise DeploymentMetricError(f"CatBoost {fold} probability is invalid")
    if any(frame[column].isna().any() for column in SEGMENT_COLUMNS):
        raise DeploymentMetricError(f"CatBoost {fold} segment is invalid")
    return frame


def _validate_and_align_folds(
    tabm_by_fold: Mapping[str, pd.DataFrame],
    catboost_by_fold: Mapping[str, pd.DataFrame],
    contract: DeploymentContract,
) -> dict[str, _AlignedFold]:
    expected_folds = set(FOLD_KEYS)
    if set(tabm_by_fold) != expected_folds or set(catboost_by_fold) != expected_folds:
        raise DeploymentMetricError("prediction fold set differs")
    output: dict[str, _AlignedFold] = {}
    for fold in FOLD_KEYS:
        tabm = _validate_tabm(tabm_by_fold[fold], fold)
        catboost = _validate_catboost(catboost_by_fold[fold], fold)
        if len(tabm) != len(catboost):
            raise DeploymentMetricError(f"{fold} row count differs")
        for column in ("row_id", "target", *SEGMENT_COLUMNS):
            if not tabm[column].reset_index(drop=True).equals(
                catboost[column].reset_index(drop=True)
            ):
                raise DeploymentMetricError(f"{fold} {column} alignment differs")
        output[fold] = _AlignedFold(
            target=pd.to_numeric(tabm["target"], errors="raise").to_numpy(dtype="float64"),
            tabm=pd.to_numeric(tabm["probability"], errors="raise").to_numpy(dtype="float64"),
            catboost_by_prefix={
                prefix: np.clip(
                    pd.to_numeric(catboost[f"p_{prefix}"], errors="raise").to_numpy(
                        dtype="float64"
                    ),
                    0.0,
                    1.0,
                )
                for prefix in contract.tree_prefixes
            },
        )
    return output


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    value = float(np.mean(np.square(probability - target), dtype=np.float64))
    if not math.isfinite(value):
        raise DeploymentMetricError("Brier is not finite")
    return value


def _gate_passes(
    weighted_gain: float,
    fold_regression: Mapping[str, float],
    contract: DeploymentContract,
) -> bool:
    return Decimal(str(weighted_gain)) >= Decimal(
        str(contract.minimum_weighted_gain)
    ) and all(
        Decimal(str(value)) <= Decimal(str(contract.maximum_fold_regression))
        for value in fold_regression.values()
    )


def _score_prefix(
    aligned: Mapping[str, _AlignedFold],
    prefix: int,
    contract: DeploymentContract,
    baseline_fold: Mapping[str, float],
    baseline_weighted: float,
    row_counts: Mapping[str, int],
) -> PrefixCandidate:
    fold_brier: dict[str, float] = {}
    for fold in FOLD_KEYS:
        evidence = aligned[fold]
        blended = (
            contract.tabm_weight * evidence.tabm
            + (1.0 - contract.tabm_weight) * evidence.catboost_by_prefix[prefix]
        )
        fold_brier[fold] = _brier(evidence.target, blended)
    total = sum(row_counts.values())
    weighted = sum(fold_brier[fold] * row_counts[fold] for fold in FOLD_KEYS) / total
    gain = baseline_weighted - weighted
    regression = {
        fold: fold_brier[fold] - baseline_fold[fold] for fold in FOLD_KEYS
    }
    return PrefixCandidate(
        tree_count=prefix,
        fold_brier=fold_brier,
        fold_regression=regression,
        weighted_brier=weighted,
        weighted_gain=gain,
        passed=_gate_passes(gain, regression, contract),
    )


def _select_decision(
    candidates: tuple[PrefixCandidate, ...],
    passing: tuple[PrefixCandidate, ...],
    baseline_weighted: float,
) -> DeploymentDecision:
    if not passing:
        return DeploymentDecision(
            status="deployment_blocked",
            selected_tree_count=None,
            baseline_weighted_brier=baseline_weighted,
            candidates=candidates,
            reason="no_tree_prefix_passed",
        )
    minimum = min(candidate.weighted_brier for candidate in passing)
    tied = tuple(
        candidate
        for candidate in passing
        if abs(candidate.weighted_brier - minimum) <= 1e-12
    )
    selected = min(tied, key=lambda candidate: candidate.tree_count)
    return DeploymentDecision(
        status="deployment_aligned",
        selected_tree_count=selected.tree_count,
        baseline_weighted_brier=baseline_weighted,
        candidates=candidates,
        reason="fixed_tree_prefix_passed",
    )


def evaluate_prefixes(
    tabm_by_fold: Mapping[str, pd.DataFrame],
    catboost_by_fold: Mapping[str, pd.DataFrame],
    contract: DeploymentContract,
) -> DeploymentDecision:
    aligned = _validate_and_align_folds(tabm_by_fold, catboost_by_fold, contract)
    row_counts = {fold: len(aligned[fold].target) for fold in FOLD_KEYS}
    baseline_fold = {
        fold: _brier(aligned[fold].target, aligned[fold].tabm) for fold in FOLD_KEYS
    }
    total = sum(row_counts.values())
    baseline_weighted = sum(
        baseline_fold[fold] * row_counts[fold] for fold in FOLD_KEYS
    ) / total
    candidates = tuple(
        _score_prefix(
            aligned,
            prefix,
            contract,
            baseline_fold,
            baseline_weighted,
            row_counts,
        )
        for prefix in contract.tree_prefixes
    )
    passing = tuple(candidate for candidate in candidates if candidate.passed)
    return _select_decision(candidates, passing, baseline_weighted)


def _finite_canonical_mapping(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _finite_canonical_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_canonical_mapping(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise DeploymentMetricError("decision contains a non-finite value")
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise DeploymentMetricError("decision contains an unsupported value")


def decision_payload(decision: DeploymentDecision) -> dict[str, object]:
    payload = _finite_canonical_mapping(asdict(decision))
    if type(payload) is not dict:
        raise DeploymentMetricError("decision payload must be an object")
    return payload
