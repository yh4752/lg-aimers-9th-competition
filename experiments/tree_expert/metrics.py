from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .contracts import E1Contract
from .inputs import PREDICTION_COLUMNS


class TreeMetricError(ValueError):
    """Raised when OOF evidence is misaligned or numerically invalid."""


@dataclass(frozen=True)
class SegmentMetric:
    column: str
    value: str
    row_count: int
    baseline_brier: float
    candidate_brier: float
    gain: float


@dataclass(frozen=True)
class CandidateMetric:
    candidate_id: str
    status: str
    baseline_brier: float | None
    candidate_brier: float | None
    gain: float | None
    bootstrap_lower: float | None
    bootstrap_upper: float | None
    prediction_correlation: float | None
    residual_correlation: float | None
    maximum_segment_regression: float | None
    segments: tuple[SegmentMetric, ...]
    reason: str | None


@dataclass(frozen=True)
class E1Decision:
    status: str
    promoted: tuple[str, ...]
    reason: str


def skipped_metric(candidate_id: str, reason: str) -> CandidateMetric:
    return CandidateMetric(
        candidate_id=candidate_id,
        status="skipped",
        baseline_brier=None,
        candidate_brier=None,
        gain=None,
        bootstrap_lower=None,
        bootstrap_upper=None,
        prediction_correlation=None,
        residual_correlation=None,
        maximum_segment_regression=None,
        segments=(),
        reason=reason,
    )


def _frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or tuple(frame.columns) != PREDICTION_COLUMNS:
        raise TreeMetricError(f"{label} prediction schema differs")
    if frame.empty or frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise TreeMetricError(f"{label} row_id values are invalid")
    if frame.loc[:, PREDICTION_COLUMNS[3:]].isna().any().any():
        raise TreeMetricError(f"{label} diagnostic values are missing")
    output = frame.copy(deep=True)
    output["target"] = pd.to_numeric(output["target"], errors="coerce")
    output["probability"] = pd.to_numeric(output["probability"], errors="coerce")
    if not output["target"].isin([0, 1]).all():
        raise TreeMetricError(f"{label} target values are invalid")
    if output["probability"].isna().any() or not output["probability"].between(0, 1).all():
        raise TreeMetricError(f"{label} probability values are invalid")
    return output


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _bootstrap(
    per_row_gain: np.ndarray,
    groups: Sequence[object],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    group_values = pd.Series(list(groups), dtype=object)
    if len(group_values) != len(per_row_gain) or group_values.isna().any():
        raise TreeMetricError("bootstrap group alignment differs")
    codes, uniques = pd.factorize(group_values, sort=False)
    if len(uniques) == 0:
        raise TreeMetricError("bootstrap groups are empty")
    sums = np.bincount(codes, weights=per_row_gain, minlength=len(uniques))
    counts = np.bincount(codes, minlength=len(uniques))
    generator = np.random.default_rng(seed)
    values = np.empty(repeats, dtype="float64")
    for index in range(repeats):
        sampled = generator.integers(0, len(uniques), size=len(uniques))
        values[index] = sums[sampled].sum() / counts[sampled].sum()
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _segments(
    baseline: pd.DataFrame,
    baseline_loss: np.ndarray,
    candidate_loss: np.ndarray,
    *,
    minimum_rows: int,
) -> tuple[SegmentMetric, ...]:
    output: list[SegmentMetric] = []
    for column in (
        "game_type",
        "game_month",
        "pitcher_id_known",
        "batter_id_known",
    ):
        values = baseline[column]
        for value in sorted(values.unique().tolist(), key=str):
            mask = values.eq(value).to_numpy()
            row_count = int(mask.sum())
            if row_count < minimum_rows:
                continue
            base = float(baseline_loss[mask].mean())
            candidate = float(candidate_loss[mask].mean())
            output.append(
                SegmentMetric(
                    column=column,
                    value=str(value),
                    row_count=row_count,
                    baseline_brier=base,
                    candidate_brier=candidate,
                    gain=base - candidate,
                )
            )
    return tuple(output)


def evaluate_e1_candidate(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    contract: E1Contract,
    *,
    candidate_id: str,
    bootstrap_groups: Sequence[object],
) -> CandidateMetric:
    base = _frame(baseline, "baseline")
    trial = _frame(candidate, "candidate")
    if base["row_id"].astype(str).tolist() != trial["row_id"].astype(str).tolist():
        raise TreeMetricError("row_id alignment differs")
    if not np.array_equal(base["target"].to_numpy(), trial["target"].to_numpy()):
        raise TreeMetricError("target alignment differs")
    for column in PREDICTION_COLUMNS[3:]:
        if base[column].astype(str).tolist() != trial[column].astype(str).tolist():
            raise TreeMetricError(f"diagnostic alignment differs: {column}")

    target = base["target"].to_numpy(dtype="float64")
    baseline_probability = base["probability"].to_numpy(dtype="float64")
    candidate_probability = trial["probability"].to_numpy(dtype="float64")
    baseline_loss = np.square(baseline_probability - target)
    candidate_loss = np.square(candidate_probability - target)
    per_row_gain = baseline_loss - candidate_loss
    baseline_brier = float(baseline_loss.mean())
    candidate_brier = float(candidate_loss.mean())
    segments = _segments(
        base,
        baseline_loss,
        candidate_loss,
        minimum_rows=contract.minimum_segment_rows,
    )
    maximum_regression = max(
        [0.0, *(segment.candidate_brier - segment.baseline_brier for segment in segments)]
    )
    lower, upper = _bootstrap(
        per_row_gain,
        bootstrap_groups,
        repeats=contract.bootstrap_repeats,
        seed=contract.bootstrap_seed,
    )
    return CandidateMetric(
        candidate_id=candidate_id,
        status="completed",
        baseline_brier=baseline_brier,
        candidate_brier=candidate_brier,
        gain=baseline_brier - candidate_brier,
        bootstrap_lower=lower,
        bootstrap_upper=upper,
        prediction_correlation=_correlation(baseline_probability, candidate_probability),
        residual_correlation=_correlation(
            target - baseline_probability,
            target - candidate_probability,
        ),
        maximum_segment_regression=maximum_regression,
        segments=segments,
        reason=None,
    )


def decide_e1(
    metrics: Sequence[CandidateMetric],
    contract: E1Contract,
) -> E1Decision:
    identifiers = [metric.candidate_id for metric in metrics]
    if len(identifiers) != len(set(identifiers)):
        raise TreeMetricError("candidate metric identities are not unique")
    eligible = [metric for metric in metrics if metric.status == "completed"]
    if not eligible:
        return E1Decision("rejected", (), "no_completed_candidates")
    if all(
        metric.gain is not None
        and metric.gain < -contract.stop_if_all_regress_more_than
        for metric in eligible
    ):
        return E1Decision("rejected", (), "all_candidates_regressed_more_than_gate")
    if any(
        metric.gain is None
        or metric.bootstrap_lower is None
        or metric.maximum_segment_regression is None
        for metric in eligible
    ):
        raise TreeMetricError("completed candidate metric is incomplete")
    ranked = sorted(
        eligible,
        key=lambda metric: (
            -float(metric.gain),
            -float(metric.bootstrap_lower),
            float(metric.maximum_segment_regression),
            metric.candidate_id,
        ),
    )
    promoted = tuple(
        metric.candidate_id for metric in ranked[: contract.maximum_promoted]
    )
    return E1Decision("completed", promoted, "top_structures_selected")
