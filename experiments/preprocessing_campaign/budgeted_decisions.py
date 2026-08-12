"""Pure, preregistered decisions for the budgeted preprocessing campaign."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


class DecisionError(ValueError):
    """Raised when evidence cannot be compared safely."""


@dataclass(frozen=True)
class ModelDecision:
    selected_candidate_id: str
    selected_family: str
    reason: str
    retained_candidates: tuple[str, ...]


@dataclass(frozen=True)
class CandidateDecision:
    status: str
    reason: str


def _finite(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DecisionError(f"{key} must be finite")
    return result


def _valid_dl(row: Mapping[str, object]) -> bool:
    return (
        row.get("family") in {"tabm", "ft_transformer", "tabnet"}
        and row.get("hashes_valid") is True
        and int(row.get("completed_epochs", 0)) >= 10
        and int(row.get("validation_points", 0)) >= 3
    )


def choose_dl_representative(
    rows: Sequence[Mapping[str, object]],
    blend_rows: Sequence[Mapping[str, object]],
) -> ModelDecision:
    eligible = [row for row in rows if _valid_dl(row)]
    if not eligible:
        raise DecisionError("no DL candidate has minimum valid evidence")
    for row in eligible:
        _finite(row, "brier")
        _finite(row, "elapsed_seconds")
        if not isinstance(row.get("candidate_id"), str) or not row["candidate_id"]:
            raise DecisionError("candidate_id must be a non-empty string")
    best_brier = min(_finite(row, "brier") for row in eligible)
    tied = [
        row for row in eligible if _finite(row, "brier") <= best_brier + 0.0002
    ]
    selected = min(
        tied,
        key=lambda row: (
            _finite(row, "elapsed_seconds"),
            _finite(row, "brier"),
            str(row["candidate_id"]),
        ),
    )
    reason = (
        "lowest_brier"
        if _finite(selected, "brier") == best_brier
        else "brier_tie_faster"
    )
    retained = {str(selected["candidate_id"])}
    for row in blend_rows:
        candidate_id = row.get("candidate_id")
        if isinstance(candidate_id, str) and _finite(row, "blend_gain") >= 0.0001:
            retained.add(candidate_id)
    return ModelDecision(
        selected_candidate_id=str(selected["candidate_id"]),
        selected_family=str(selected["family"]),
        reason=reason,
        retained_candidates=tuple(sorted(retained)),
    )


def decide_tabnet(
    row: Mapping[str, object],
    *,
    best_dl_brier: float,
    blend_gain: float,
    oov_gain: float,
    overall_delta: float,
) -> CandidateDecision:
    if not _valid_dl(row):
        return CandidateDecision("inconclusive", "minimum_training_not_met")
    brier = _finite(row, "brier")
    if brier <= float(best_dl_brier) + 0.0005:
        return CandidateDecision("promoted", "brier_proximity")
    if float(blend_gain) >= 0.0001:
        return CandidateDecision("promoted", "blend_gain")
    if float(oov_gain) >= 0.0002 and float(overall_delta) <= 0.00005:
        return CandidateDecision("promoted", "oov_gain")
    return CandidateDecision("not_recommended", "no_promotion_rule_met")


def promote_single(*, delta: float, oov_delta: float, blend_gain: float) -> bool:
    values = (delta, oov_delta, blend_gain)
    if not all(math.isfinite(float(value)) for value in values):
        raise DecisionError("single-candidate deltas must be finite")
    if delta <= -0.0001:
        return True
    if delta < 0 and oov_delta <= -0.0002:
        return True
    return delta <= 0.00005 and blend_gain >= 0.0001


def fixed_blend_metrics(
    *,
    row_id: np.ndarray,
    target: np.ndarray,
    catboost_row_id: np.ndarray,
    catboost_probability: np.ndarray,
    candidate_probability: np.ndarray,
    weights: Sequence[float],
) -> tuple[dict[str, float], ...]:
    ids = np.asarray(row_id).astype(str)
    cat_ids = np.asarray(catboost_row_id).astype(str)
    y = np.asarray(target, dtype="float64")
    cat = np.asarray(catboost_probability, dtype="float64")
    candidate = np.asarray(candidate_probability, dtype="float64")
    if not np.array_equal(ids, cat_ids):
        raise DecisionError("row alignment differs between blend candidates")
    if not (y.shape == cat.shape == candidate.shape == ids.shape):
        raise DecisionError("blend arrays must have identical one-dimensional shape")
    if not np.isfinite(np.concatenate([y, cat, candidate])).all():
        raise DecisionError("blend arrays must be finite")
    baseline = float(np.mean(np.square(cat - y)))
    rows = []
    for weight in weights:
        value = float(weight)
        if not math.isfinite(value) or not 0 < value < 1:
            raise DecisionError("blend weight must be between zero and one")
        probability = (1.0 - value) * cat + value * candidate
        brier = float(np.mean(np.square(probability - y)))
        rows.append({"candidate_weight": value, "brier": brier, "gain": baseline - brier})
    return tuple(rows)


def final_preprocessing_status(
    *,
    proxy_deltas: Mapping[int, float],
    proxy_rows: Mapping[int, int],
    full_2024_delta: float,
    segment_deltas: Mapping[str, float],
    hashes_valid: bool,
) -> str:
    if not hashes_valid:
        return "inconclusive"
    if set(proxy_deltas) != {2023, 2024} or set(proxy_rows) != {2023, 2024}:
        raise DecisionError("2023 and 2024 proxy evidence is required")
    if any(proxy_rows[year] <= 0 for year in (2023, 2024)):
        raise DecisionError("proxy row counts must be positive")
    deltas = {year: float(proxy_deltas[year]) for year in (2023, 2024)}
    if not all(math.isfinite(value) for value in deltas.values()):
        raise DecisionError("proxy deltas must be finite")
    full_delta = float(full_2024_delta)
    if not math.isfinite(full_delta):
        raise DecisionError("full delta must be finite")
    required_segments = {"pitcher_oov", "batter_oov", "game_type"}
    if set(segment_deltas) != required_segments:
        raise DecisionError("required segment deltas are missing")
    segments = [float(segment_deltas[name]) for name in sorted(required_segments)]
    if not all(math.isfinite(value) for value in segments):
        raise DecisionError("segment deltas must be finite")
    if deltas[2023] >= 0 or deltas[2024] >= 0:
        return "not_recommended" if all(value >= 0 for value in deltas.values()) else "inconclusive"
    total_rows = proxy_rows[2023] + proxy_rows[2024]
    weighted = sum(deltas[year] * proxy_rows[year] for year in (2023, 2024)) / total_rows
    if weighted > -0.0001 or full_delta >= 0 or max(segments) > 0.0002:
        return "inconclusive"
    return "recommended"
