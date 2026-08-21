from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import HierarchicalContract


class HierarchicalMetricError(ValueError):
    """Raised when paired evidence is incomplete or not row-identical."""


SEGMENT_COLUMNS = (
    "game_type",
    "count_state",
    "hand_matchup",
    "base_out_state",
    "pitcher_known",
    "batter_known",
)
FOLDS = ("2022->2023", "2023->2024")
MISSING_CATEGORY = "__MISSING__"
_BOUNDARY_EPSILON = 1e-15


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    status: str
    final_acceptance: bool
    delivery_role: str | None
    fold_brier: Mapping[str, float]
    fold_gain_vs_anchor: Mapping[str, float]
    weighted_gain_vs_anchor: float
    worst_eligible_segment_regression: float
    reason: str


def _at_least(value: float, boundary: float) -> bool:
    return value + _BOUNDARY_EPSILON >= boundary


def _at_most(value: float, boundary: float) -> bool:
    return value <= boundary + _BOUNDARY_EPSILON


def _vector(value: Sequence[float], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype="float64")
    except (TypeError, ValueError) as error:
        raise HierarchicalMetricError(f"{label} is not numeric") from error
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise HierarchicalMetricError(f"{label} must be a finite non-empty vector")
    return result


def brier(target: Sequence[float], probability: Sequence[float]) -> float:
    y = _vector(target, "target")
    p = _vector(probability, "probability")
    if len(y) != len(p):
        raise HierarchicalMetricError("target and probability length differs")
    if not np.isin(y, (0.0, 1.0)).all():
        raise HierarchicalMetricError("target must contain only 0 and 1")
    if ((p < 0) | (p > 1)).any():
        raise HierarchicalMetricError("probability must be in [0, 1]")
    return float(np.mean(np.square(y - p), dtype=np.float64))


def _prediction_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"row_id", "target", "probability"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise HierarchicalMetricError(f"{label} missing columns: {', '.join(missing)}")
    if frame.empty or frame["row_id"].isna().any():
        raise HierarchicalMetricError(f"{label} row_id is missing")
    output = frame.copy()
    output["row_id"] = output["row_id"].astype(str)
    if output["row_id"].duplicated().any():
        raise HierarchicalMetricError(f"{label} row_id must be unique")
    brier(output["target"], output["probability"])
    return output


def align_anchor_and_h1(anchor: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    left = _prediction_frame(anchor, "anchor")
    right = _prediction_frame(candidate, "candidate")
    if set(left["row_id"]) != set(right["row_id"]):
        raise HierarchicalMetricError("row_id set differs")
    indexed = right.set_index("row_id", drop=False).loc[left["row_id"]]
    right_target = indexed["target"].to_numpy(dtype="float64")
    left_target = left["target"].to_numpy(dtype="float64")
    if not np.array_equal(left_target, right_target):
        raise HierarchicalMetricError("target differs on paired rows")
    output = left.copy()
    output["anchor_probability"] = left["probability"].to_numpy(dtype="float64")
    output["candidate_probability"] = indexed["probability"].to_numpy(dtype="float64")
    output = output.drop(columns=["probability"])
    for column in (*SEGMENT_COLUMNS, "game_month"):
        if column not in indexed:
            continue
        candidate_values = indexed[column].reset_index(drop=True)
        if column in output:
            anchor_values = output[column].reset_index(drop=True)
            matches = anchor_values.eq(candidate_values) | (
                anchor_values.isna() & candidate_values.isna()
            )
            if not bool(matches.all()):
                raise HierarchicalMetricError(
                    f"paired evidence differs for segment: {column}"
                )
        output[column] = candidate_values.to_numpy()
    return output


def _category(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna(MISSING_CATEGORY).astype(str)


def _bounded_integer(frame: pd.DataFrame, column: str, upper: int) -> pd.Series:
    if column not in frame:
        raise HierarchicalMetricError(f"missing segment source: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    numeric = values.to_numpy(dtype="float64")
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all() or ((numeric < 0) | (numeric > upper)).any():
        raise HierarchicalMetricError(f"invalid segment source: {column}")
    return values.astype("int64").astype(str)


def build_segment_columns(
    frame: pd.DataFrame,
    *,
    fit_pitcher_ids: set[str],
    fit_batter_ids: set[str],
) -> pd.DataFrame:
    required = {
        "game_type", "pitcher_hand", "batter_hand", "base_state", "pitcher_id",
        "batter_id", "balls_before", "strikes_before", "outs_before",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise HierarchicalMetricError(f"missing segment sources: {', '.join(missing)}")
    if not isinstance(fit_pitcher_ids, set) or not isinstance(fit_batter_ids, set):
        raise HierarchicalMetricError("known ID collections must be sets")
    pitcher_ids = {str(value) for value in fit_pitcher_ids}
    batter_ids = {str(value) for value in fit_batter_ids}
    balls = _bounded_integer(frame, "balls_before", 3)
    strikes = _bounded_integer(frame, "strikes_before", 2)
    outs = _bounded_integer(frame, "outs_before", 2)
    pitcher_hand = _category(frame["pitcher_hand"])
    batter_hand = _category(frame["batter_hand"])
    base = _category(frame["base_state"])
    output = pd.DataFrame(index=frame.index)
    output["game_type"] = _category(frame["game_type"])
    output["count_state"] = balls + "_" + strikes
    output["hand_matchup"] = pitcher_hand + "_" + batter_hand
    output["base_out_state"] = base + "_" + outs
    output["pitcher_known"] = np.where(
        _category(frame["pitcher_id"]).isin(pitcher_ids), "known", "oov"
    )
    output["batter_known"] = np.where(
        _category(frame["batter_id"]).isin(batter_ids), "known", "oov"
    )
    return output


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    indices = np.minimum((probability * bins).astype("int64"), bins - 1)
    total = len(target)
    result = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            result += float(mask.sum()) / total * abs(
                float(target[mask].mean()) - float(probability[mask].mean())
            )
    return result


def paired_fold_metrics(
    anchor_by_fold: Mapping[str, pd.DataFrame],
    candidate_by_fold: Mapping[str, pd.DataFrame],
    *,
    segment_min_rows: int,
) -> Mapping[str, object]:
    if tuple(anchor_by_fold) != FOLDS or tuple(candidate_by_fold) != FOLDS:
        raise HierarchicalMetricError("paired folds must be exactly 2022->2023 and 2023->2024")
    if isinstance(segment_min_rows, bool) or not isinstance(segment_min_rows, int) or segment_min_rows <= 0:
        raise HierarchicalMetricError("segment_min_rows must be positive")
    fold_brier: dict[str, float] = {}
    anchor_brier: dict[str, float] = {}
    fold_gain: dict[str, float] = {}
    fold_rows: dict[str, int] = {}
    segments: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    ece: dict[str, float] = {}
    total_anchor_loss = 0.0
    total_candidate_loss = 0.0
    total_rows = 0
    for fold in FOLDS:
        paired = align_anchor_and_h1(anchor_by_fold[fold], candidate_by_fold[fold])
        y = paired["target"].to_numpy(dtype="float64")
        anchor_probability = paired["anchor_probability"].to_numpy(dtype="float64")
        candidate_probability = paired["candidate_probability"].to_numpy(dtype="float64")
        anchor_score = brier(y, anchor_probability)
        candidate_score = brier(y, candidate_probability)
        anchor_brier[fold] = anchor_score
        fold_brier[fold] = candidate_score
        fold_gain[fold] = anchor_score - candidate_score
        fold_rows[fold] = len(paired)
        total_anchor_loss += anchor_score * len(paired)
        total_candidate_loss += candidate_score * len(paired)
        total_rows += len(paired)
        ece[fold] = _ece(y, candidate_probability)
        for column in SEGMENT_COLUMNS:
            if column not in paired:
                raise HierarchicalMetricError(f"paired evidence missing segment: {column}")
            for level, positions in paired.groupby(column, sort=True, dropna=False).groups.items():
                index = np.asarray(list(positions), dtype="int64")
                count = len(index)
                segment_anchor = brier(y[index], anchor_probability[index])
                segment_candidate = brier(y[index], candidate_probability[index])
                segments.append(
                    {
                        "fold": fold,
                        "column": column,
                        "level": str(level),
                        "rows": count,
                        "anchor_brier": segment_anchor,
                        "candidate_brier": segment_candidate,
                        "regression": segment_candidate - segment_anchor,
                        "eligible": count >= segment_min_rows,
                    }
                )
        if "game_month" in paired:
            for month, positions in paired.groupby("game_month", sort=True).groups.items():
                index = np.asarray(list(positions), dtype="int64")
                monthly.append(
                    {
                        "fold": fold,
                        "game_month": int(month),
                        "rows": len(index),
                        "candidate_brier": brier(y[index], candidate_probability[index]),
                    }
                )
    eligible_regressions = [float(row["regression"]) for row in segments if row["eligible"]]
    return MappingProxyType(
        {
            "status": "completed",
            "fold_brier": MappingProxyType(fold_brier),
            "anchor_fold_brier": MappingProxyType(anchor_brier),
            "fold_gain_vs_anchor": MappingProxyType(fold_gain),
            "fold_rows": MappingProxyType(fold_rows),
            "weighted_gain_vs_anchor": (total_anchor_loss - total_candidate_loss) / total_rows,
            "worst_eligible_segment_regression": max(eligible_regressions, default=0.0),
            "segments": tuple(segments),
            "monthly": tuple(monthly),
            "ece_10": MappingProxyType(ece),
        }
    )


def _mapping_float(metrics: Mapping[str, object], key: str) -> Mapping[str, float]:
    value = metrics.get(key)
    if not isinstance(value, Mapping) or tuple(value) != FOLDS:
        raise HierarchicalMetricError(f"metric {key} is incomplete")
    result: dict[str, float] = {}
    for fold in FOLDS:
        item = value[fold]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise HierarchicalMetricError(f"metric {key} is invalid")
        result[fold] = float(item)
    return MappingProxyType(result)


def _decision(
    candidate_id: str,
    status: str,
    final: bool,
    role: str | None,
    metrics: Mapping[str, object],
    reason: str,
) -> CandidateDecision:
    fold_brier = _mapping_float(metrics, "fold_brier")
    fold_gain = _mapping_float(metrics, "fold_gain_vs_anchor")
    weighted = float(metrics.get("weighted_gain_vs_anchor", 0.0))
    worst = float(metrics.get("worst_eligible_segment_regression", 0.0))
    if not math.isfinite(weighted) or not math.isfinite(worst):
        raise HierarchicalMetricError("decision metrics are non-finite")
    return CandidateDecision(
        candidate_id, status, final, role, fold_brier, fold_gain, weighted, worst, reason
    )


def decide_h1(
    metrics: Mapping[str, object], contract: HierarchicalContract
) -> CandidateDecision:
    if metrics.get("status") != "completed":
        return _decision("H1", "incomplete", False, None, metrics, "H1 evidence is incomplete")
    gain = _mapping_float(metrics, "fold_gain_vs_anchor")
    weighted = float(metrics["weighted_gain_vs_anchor"])
    worst = float(metrics["worst_eligible_segment_regression"])
    old_regression = max(0.0, -gain["2022->2023"])
    latest_gain = gain["2023->2024"]
    strong = contract.gates["H1_strong"]
    if (
        _at_least(weighted, strong["minimum_weighted_gain"])
        and _at_least(latest_gain, strong["minimum_latest_gain"])
        and _at_most(old_regression, strong["maximum_old_fold_regression"])
        and _at_most(worst, strong["maximum_segment_regression"])
    ):
        return _decision("H1", "strong", True, "final_candidate", metrics, "all H1 strong gates passed")
    frontier = contract.gates["H1_frontier"]
    if weighted > 0 and latest_gain > 0 and _at_most(old_regression, frontier["maximum_old_fold_regression"]):
        return _decision(
            "H1", "frontier", False, "public_diagnostic_only", metrics,
            "positive independent evidence passed only the frontier gate",
        )
    return _decision("H1", "rejected", False, None, metrics, "H1 gates failed")


def _segment_regression_vs_h1(
    h1_metrics: Mapping[str, object], calibrated_metrics: Mapping[str, object]
) -> float:
    explicit = calibrated_metrics.get("worst_eligible_segment_regression_vs_h1")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
        value = float(explicit)
        if math.isfinite(value):
            return value
    h1_rows = h1_metrics.get("segments")
    calibrated_rows = calibrated_metrics.get("segments")
    if not isinstance(h1_rows, (tuple, list)) or not isinstance(calibrated_rows, (tuple, list)):
        raise HierarchicalMetricError("segment comparison against H1 is missing")
    def keyed(rows):
        return {
            (row["fold"], row["column"], row["level"]): row
            for row in rows if isinstance(row, Mapping) and row.get("eligible") is True
        }
    left, right = keyed(h1_rows), keyed(calibrated_rows)
    if set(left) != set(right):
        raise HierarchicalMetricError("eligible H1 segment sets differ")
    return max(
        (float(right[key]["candidate_brier"]) - float(left[key]["candidate_brier"]) for key in left),
        default=0.0,
    )


def decide_calibrated(
    candidate_id: str,
    h1_metrics: Mapping[str, object],
    calibrated_metrics: Mapping[str, object],
    contract: HierarchicalContract,
    *,
    h2_metrics: Mapping[str, object] | None = None,
) -> CandidateDecision:
    if candidate_id not in {"H2", "H3"}:
        raise HierarchicalMetricError("calibrated candidate must be H2 or H3")
    if h1_metrics.get("status") != "completed" or calibrated_metrics.get("status") != "completed":
        return _decision(
            candidate_id, "incomplete", False, None, calibrated_metrics,
            f"{candidate_id} evidence is incomplete",
        )
    h1_brier = _mapping_float(h1_metrics, "fold_brier")
    candidate_brier = _mapping_float(calibrated_metrics, "fold_brier")
    candidate_gain = _mapping_float(calibrated_metrics, "fold_gain_vs_anchor")
    latest_vs_h1 = h1_brier["2023->2024"] - candidate_brier["2023->2024"]
    latest_vs_anchor = candidate_gain["2023->2024"]
    worst_vs_h1 = _segment_regression_vs_h1(h1_metrics, calibrated_metrics)
    gates = contract.gates[candidate_id]
    passed = (
        _at_least(latest_vs_h1, gates["minimum_latest_gain_vs_h1"])
        and _at_least(latest_vs_anchor, gates["minimum_latest_gain_vs_anchor"])
        and _at_most(worst_vs_h1, gates["maximum_segment_regression_vs_h1"])
    )
    if candidate_id == "H3":
        if h2_metrics is None or h2_metrics.get("status") != "completed":
            return _decision("H3", "incomplete", False, None, calibrated_metrics, "H2 comparator is incomplete")
        h2_brier = _mapping_float(h2_metrics, "fold_brier")
        latest_vs_h2 = h2_brier["2023->2024"] - candidate_brier["2023->2024"]
        passed = passed and _at_least(latest_vs_h2, gates["minimum_latest_gain_vs_h2"])
    if passed:
        return _decision(
            candidate_id, "accepted", True, "final_candidate", calibrated_metrics,
            f"all independent {candidate_id} gates passed",
        )
    return _decision(candidate_id, "rejected", False, None, calibrated_metrics, f"{candidate_id} gates failed")


def candidate_decision_payload(decision: CandidateDecision) -> dict[str, object]:
    return {
        "candidate_id": decision.candidate_id,
        "status": decision.status,
        "final_acceptance": decision.final_acceptance,
        "delivery_role": decision.delivery_role,
        "fold_brier": dict(decision.fold_brier),
        "fold_gain_vs_anchor": dict(decision.fold_gain_vs_anchor),
        "weighted_gain_vs_anchor": decision.weighted_gain_vs_anchor,
        "worst_eligible_segment_regression": decision.worst_eligible_segment_regression,
        "reason": decision.reason,
    }
