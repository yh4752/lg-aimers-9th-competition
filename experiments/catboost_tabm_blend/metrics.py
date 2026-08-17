from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import math
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import BlendContract


class BlendMetricError(ValueError):
    """Raised when OOF rows cannot be compared exactly."""


FOLD_KEYS = ("2022->2023", "2023->2024")
PREDICTION_COLUMNS = (
    "row_id",
    "target",
    "probability",
    "game_type",
    "game_month",
    "pitcher_id_known",
    "batter_id_known",
)
SEGMENT_COLUMNS = (
    "game_type",
    "game_month",
    "pitcher_id_known",
    "batter_id_known",
)


@dataclass(frozen=True)
class BlendCandidateMetric:
    tabm_weight: float
    fold_brier: Mapping[str, float]
    weighted_brier: float
    weighted_gain: float
    fold_regression: Mapping[str, float]
    passed: bool


@dataclass(frozen=True)
class BlendDecision:
    baseline_fold_brier: Mapping[str, float]
    baseline_weighted_brier: float
    prediction_correlation: Mapping[str, float | None]
    residual_correlation: Mapping[str, float | None]
    segment_diagnostics: Mapping[str, object]
    candidates: tuple[BlendCandidateMetric, ...]
    selected_tabm_weight: float | None
    reason: str


def _validate_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise BlendMetricError(f"{label} prediction schema differs")
    if len(frame) == 0:
        raise BlendMetricError(f"{label} prediction frame is empty")
    row_id = frame["row_id"]
    if row_id.isna().any() or row_id.astype(str).duplicated().any():
        raise BlendMetricError(f"{label} row_id is invalid")
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise BlendMetricError(f"{label} target is invalid")
    probability = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(
        dtype="float64"
    )
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise BlendMetricError(f"{label} probability is invalid")
    if any(frame[column].isna().any() for column in SEGMENT_COLUMNS):
        raise BlendMetricError(f"{label} diagnostic label is missing")
    return frame


def _aligned(tabm: pd.DataFrame, catboost: pd.DataFrame, fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tabm = _validate_frame(tabm, f"TabM {fold}")
    catboost = _validate_frame(catboost, f"CatBoost {fold}")
    if len(tabm) != len(catboost):
        raise BlendMetricError(f"{fold} row count differs")
    for column in ("row_id", "target", *SEGMENT_COLUMNS):
        if not tabm[column].reset_index(drop=True).equals(
            catboost[column].reset_index(drop=True)
        ):
            raise BlendMetricError(f"{fold} {column} alignment differs")
    target = pd.to_numeric(tabm["target"], errors="raise").to_numpy(dtype="float64")
    tabm_probability = pd.to_numeric(tabm["probability"], errors="raise").to_numpy(
        dtype="float64"
    )
    catboost_probability = pd.to_numeric(
        catboost["probability"], errors="raise"
    ).to_numpy(dtype="float64")
    return target, tabm_probability, catboost_probability


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    result = float(np.mean(np.square(probability - target), dtype=np.float64))
    if not math.isfinite(result):
        raise BlendMetricError("Brier is not finite")
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        return None
    if abs(abs(value) - 1.0) <= 1e-15:
        return math.copysign(1.0, value)
    return value


def _gate_passes(
    weighted_gain: float,
    fold_regression: Mapping[str, float],
    minimum_gain: float,
    maximum_regression: float,
) -> bool:
    gain = Decimal(str(weighted_gain))
    minimum = Decimal(str(minimum_gain))
    maximum = Decimal(str(maximum_regression))
    return gain >= minimum and all(
        Decimal(str(value)) <= maximum for value in fold_regression.values()
    )


def _segments(
    frame: pd.DataFrame, target: np.ndarray, probability: np.ndarray
) -> dict[str, dict[str, dict[str, float | int]]]:
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for column in SEGMENT_COLUMNS:
        groups: dict[str, dict[str, float | int]] = {}
        values = frame[column].astype(str).to_numpy()
        for value in sorted(set(values)):
            mask = values == value
            groups[value] = {
                "row_count": int(mask.sum()),
                "brier": _brier(target[mask], probability[mask]),
            }
        output[column] = groups
    return output


def evaluate_blends(
    tabm_by_fold: Mapping[str, pd.DataFrame],
    catboost_by_fold: Mapping[str, pd.DataFrame],
    contract: BlendContract,
) -> BlendDecision:
    if set(tabm_by_fold) != set(FOLD_KEYS) or set(catboost_by_fold) != set(FOLD_KEYS):
        raise BlendMetricError("prediction fold set differs")

    aligned: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    baseline_fold: dict[str, float] = {}
    row_counts: dict[str, int] = {}
    prediction_correlation: dict[str, float | None] = {}
    residual_correlation: dict[str, float | None] = {}
    segment_diagnostics: dict[str, object] = {}
    for fold in FOLD_KEYS:
        target, tabm_probability, catboost_probability = _aligned(
            tabm_by_fold[fold], catboost_by_fold[fold], fold
        )
        aligned[fold] = (target, tabm_probability, catboost_probability)
        baseline_fold[fold] = _brier(target, tabm_probability)
        row_counts[fold] = len(target)
        prediction_correlation[fold] = _correlation(
            tabm_probability, catboost_probability
        )
        residual_correlation[fold] = _correlation(
            target - tabm_probability, target - catboost_probability
        )
        segment_diagnostics[fold] = {
            "tabm": _segments(tabm_by_fold[fold], target, tabm_probability),
            "catboost": _segments(
                catboost_by_fold[fold], target, catboost_probability
            ),
        }

    total_rows = sum(row_counts.values())
    baseline_weighted = sum(
        row_counts[fold] * baseline_fold[fold] for fold in FOLD_KEYS
    ) / total_rows
    candidate_metrics: list[BlendCandidateMetric] = []
    for tabm_weight in (1.0, *contract.tabm_weights):
        fold_brier: dict[str, float] = {}
        for fold in FOLD_KEYS:
            target, tabm_probability, catboost_probability = aligned[fold]
            probability = (
                tabm_weight * tabm_probability
                + (1.0 - tabm_weight) * catboost_probability
            )
            fold_brier[fold] = _brier(target, probability)
            segment_diagnostics[fold][f"tabm_{tabm_weight:.1f}"] = _segments(
                tabm_by_fold[fold], target, probability
            )
        weighted_brier = sum(
            row_counts[fold] * fold_brier[fold] for fold in FOLD_KEYS
        ) / total_rows
        gain = baseline_weighted - weighted_brier
        regression = {
            fold: fold_brier[fold] - baseline_fold[fold] for fold in FOLD_KEYS
        }
        passed = tabm_weight < 1.0 and _gate_passes(
            gain,
            regression,
            contract.minimum_weighted_gain,
            contract.maximum_fold_regression,
        )
        candidate_metrics.append(
            BlendCandidateMetric(
                tabm_weight=tabm_weight,
                fold_brier=fold_brier,
                weighted_brier=weighted_brier,
                weighted_gain=gain,
                fold_regression=regression,
                passed=passed,
            )
        )

    passing = [candidate for candidate in candidate_metrics if candidate.passed]
    selected = min(
        passing, key=lambda candidate: (candidate.weighted_brier, -candidate.tabm_weight)
    ) if passing else None
    return BlendDecision(
        baseline_fold_brier=baseline_fold,
        baseline_weighted_brier=baseline_weighted,
        prediction_correlation=prediction_correlation,
        residual_correlation=residual_correlation,
        segment_diagnostics=segment_diagnostics,
        candidates=tuple(candidate_metrics),
        selected_tabm_weight=None if selected is None else selected.tabm_weight,
        reason="fixed_blend_passed" if selected is not None else "no_fixed_weight_passed",
    )


def decision_payload(decision: BlendDecision) -> dict[str, object]:
    return {
        "baseline_fold_brier": dict(decision.baseline_fold_brier),
        "baseline_weighted_brier": decision.baseline_weighted_brier,
        "prediction_correlation": dict(decision.prediction_correlation),
        "residual_correlation": dict(decision.residual_correlation),
        "segment_diagnostics": decision.segment_diagnostics,
        "candidates": [
            {
                "tabm_weight": item.tabm_weight,
                "fold_brier": dict(item.fold_brier),
                "weighted_brier": item.weighted_brier,
                "weighted_gain": item.weighted_gain,
                "fold_regression": dict(item.fold_regression),
                "passed": item.passed,
            }
            for item in decision.candidates
        ],
        "selected_tabm_weight": decision.selected_tabm_weight,
        "reason": decision.reason,
    }


def canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
