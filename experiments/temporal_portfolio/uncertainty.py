"""Paired uncertainty and fixed-segment diagnostics for temporal OOF."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from .metrics import PortfolioMetricError, brier


SEGMENT_COLUMNS = (
    "game_type",
    "hand_matchup",
    "pitcher_id_known",
    "batter_id_known",
    "trackman_available",
    "history_count_bucket",
    "runner_state",
    "leverage_bucket",
)


class PortfolioUncertaintyError(ValueError):
    """Raised when paired OOF uncertainty evidence is invalid."""


@dataclass(frozen=True)
class BootstrapResult:
    repeats: int
    lower: float
    median: float
    upper: float


@dataclass(frozen=True)
class SegmentRegression:
    segment: str
    value: str
    rows: int
    baseline_brier: float
    candidate_brier: float
    brier_gain: float
    eligible: bool


def pitcher_block_bootstrap(
    frame: pd.DataFrame, *, repeats: int, seed: int
) -> BootstrapResult:
    """Resample complete pitcher blocks and return a paired Brier-gain interval."""

    work = _validated_oof(frame)
    if type(repeats) is not int or not 1 <= repeats <= 100_000:
        raise PortfolioUncertaintyError("repeats must be an exact integer in [1, 100000]")
    if type(seed) is not int or seed < 0:
        raise PortfolioUncertaintyError("seed must be an exact non-negative integer")

    tokens = tuple(_id_token(value) for value in work["pitcher_id"])
    groups: dict[tuple[str, object], list[int]] = {}
    for position, token in enumerate(tokens):
        groups.setdefault(token, []).append(position)
    keys = tuple(sorted(groups, key=lambda item: (item[0], str(item[1]))))
    indices = tuple(np.asarray(groups[key], dtype="int64") for key in keys)
    target = work["target"].to_numpy(dtype="int8", copy=True)
    baseline = work["baseline"].to_numpy(dtype="float64", copy=True)
    candidate = work["candidate"].to_numpy(dtype="float64", copy=True)
    rng = np.random.default_rng(seed)
    gains = np.empty(repeats, dtype="float64")
    for repeat in range(repeats):
        selected = rng.integers(0, len(indices), size=len(indices))
        sampled = np.concatenate([indices[position] for position in selected])
        gains[repeat] = brier(target[sampled], baseline[sampled]) - brier(
            target[sampled], candidate[sampled]
        )
    lower, median, upper = np.quantile(gains, (0.025, 0.5, 0.975))
    return BootstrapResult(repeats, float(lower), float(median), float(upper))


def segment_regressions(
    frame: pd.DataFrame, *, minimum_rows: int = 5_000
) -> tuple[SegmentRegression, ...]:
    """Report paired Brier changes for the preregistered diagnostic segments."""

    work = _validated_oof(frame)
    if type(minimum_rows) is not int or minimum_rows <= 0:
        raise PortfolioUncertaintyError("minimum_rows must be an exact positive integer")
    results: list[SegmentRegression] = []
    for column in SEGMENT_COLUMNS:
        values = work[column].map(_segment_value)
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            rows = int(mask.sum())
            baseline_score = brier(work.loc[mask, "target"], work.loc[mask, "baseline"])
            candidate_score = brier(work.loc[mask, "target"], work.loc[mask, "candidate"])
            results.append(
                SegmentRegression(
                    segment=column,
                    value=value,
                    rows=rows,
                    baseline_brier=baseline_score,
                    candidate_brier=candidate_score,
                    brier_gain=baseline_score - candidate_score,
                    eligible=rows >= minimum_rows,
                )
            )
    return tuple(results)


def _validated_oof(frame: object) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or frame.empty or not frame.columns.is_unique:
        raise PortfolioUncertaintyError("OOF must be a nonempty DataFrame with unique columns")
    required = {
        "row_id",
        "valid_year",
        "pitcher_id",
        "target",
        "baseline",
        "candidate",
        *SEGMENT_COLUMNS,
    }
    if set(frame.columns) != required:
        raise PortfolioUncertaintyError("OOF schema differs from the uncertainty contract")
    work = frame.loc[:, sorted(required)].copy(deep=True).reset_index(drop=True)
    row_tokens = tuple(_id_token(value) for value in work["row_id"])
    if len(set(row_tokens)) != len(row_tokens):
        raise PortfolioUncertaintyError("OOF row_id must be unique")
    pitcher_tokens = tuple(_id_token(value) for value in work["pitcher_id"])
    years = work["valid_year"].tolist()
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 1000 <= int(value) <= 9999
        for value in years
    ):
        raise PortfolioUncertaintyError("valid_year must contain exact four-digit integers")
    try:
        work["target"] = _binary(work["target"])
        work["baseline"] = _probability(work["baseline"], "baseline")
        work["candidate"] = _probability(work["candidate"], "candidate")
    except PortfolioMetricError as error:
        raise PortfolioUncertaintyError(str(error)) from error
    for column in SEGMENT_COLUMNS:
        work[column].map(_segment_value)
    order = sorted(
        range(len(work)),
        key=lambda position: (
            pitcher_tokens[position][0],
            str(pitcher_tokens[position][1]),
            row_tokens[position][0],
            str(row_tokens[position][1]),
        ),
    )
    return work.iloc[order].reset_index(drop=True)


def _binary(values: pd.Series) -> np.ndarray:
    raw = values.tolist()
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) not in (0, 1)
        for value in raw
    ):
        raise PortfolioUncertaintyError("target must contain exact binary integers")
    return np.asarray(raw, dtype="int8")


def _probability(values: pd.Series, label: str) -> np.ndarray:
    raw = values.to_numpy()
    if np.issubdtype(raw.dtype, np.bool_) or not (
        np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)
    ):
        raise PortfolioUncertaintyError(f"{label} must be real numeric")
    result = np.asarray(raw, dtype="float64")
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise PortfolioUncertaintyError(f"{label} must be finite probabilities")
    return result


def _id_token(value: object) -> tuple[str, object]:
    if isinstance(value, (bool, np.bool_)):
        raise PortfolioUncertaintyError("IDs must be canonical strings or integers")
    if isinstance(value, Integral):
        return "int", int(value)
    if isinstance(value, str) and value and value == value.strip():
        return "str", value
    raise PortfolioUncertaintyError("IDs must be canonical strings or integers")


def _segment_value(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PortfolioUncertaintyError("segment values must be canonical strings")
    return value
