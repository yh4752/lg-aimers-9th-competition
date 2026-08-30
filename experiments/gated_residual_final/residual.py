from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


class ResidualError(ValueError):
    pass


@dataclass(frozen=True)
class EntityCounts:
    pitcher: Mapping[str, int]
    batter: Mapping[str, int]
    source_years: tuple[int, ...]


def _logit(probability: np.ndarray) -> np.ndarray:
    return np.log(probability) - np.log1p(-probability)


def _expit(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    output = np.empty_like(value, dtype="float64")
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def gated_residual(
    *, anchor: np.ndarray, direct: np.ndarray, row_reliability: np.ndarray, alpha: float,
) -> np.ndarray:
    p0 = np.asarray(anchor, dtype="float64")
    pdirect = np.asarray(direct, dtype="float64")
    strength = np.asarray(row_reliability, dtype="float64")
    if p0.ndim != 1 or pdirect.shape != p0.shape or strength.shape != p0.shape:
        raise ResidualError("residual shapes differ")
    if (
        not np.isfinite(p0).all() or not np.isfinite(pdirect).all() or not np.isfinite(strength).all()
        or np.any((p0 < 0) | (p0 > 1)) or np.any((pdirect < 0) | (pdirect > 1))
        or np.any((strength < 0) | (strength > 1)) or not np.isfinite(alpha) or alpha < 0
    ):
        raise ResidualError("residual values differ")
    clipped_anchor = np.clip(p0, 1e-6, 1.0 - 1e-6)
    clipped_direct = np.clip(pdirect, 1e-6, 1.0 - 1e-6)
    result = _expit(_logit(clipped_anchor) + float(alpha) * strength * (_logit(clipped_direct) - _logit(clipped_anchor)))
    exact_backoff = (float(alpha) == 0.0) | (strength == 0.0)
    result[exact_backoff] = p0[exact_backoff]
    return result


def reliability(pitcher_count: int, batter_count: int, *, k: int) -> float:
    if type(pitcher_count) is not int or type(batter_count) is not int or pitcher_count < 0 or batter_count < 0:
        raise ResidualError("entity count differs")
    if type(k) is not int or k <= 0:
        raise ResidualError("reliability k differs")
    return float(np.sqrt((pitcher_count / (pitcher_count + k)) * (batter_count / (batter_count + k))))


def temporal_entity_counts(
    frame: pd.DataFrame, *, validation_year: int, year_column: str = "season",
) -> EntityCounts:
    required = {year_column, "pitcher_id", "batter_id"}
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns):
        raise ResidualError("count columns differ")
    years = pd.to_numeric(frame[year_column], errors="coerce")
    if years.isna().any():
        raise ResidualError("count years differ")
    source = frame.loc[years < int(validation_year)].copy()
    source_years = tuple(sorted(int(value) for value in years.loc[source.index].unique()))
    pitcher = source["pitcher_id"].astype(str).value_counts(sort=False).astype(int).to_dict()
    batter = source["batter_id"].astype(str).value_counts(sort=False).astype(int).to_dict()
    return EntityCounts(MappingProxyType(pitcher), MappingProxyType(batter), source_years)


def row_reliability(
    frame: pd.DataFrame, *, pitcher_counts: Mapping[str, int], batter_counts: Mapping[str, int], k: int,
) -> np.ndarray:
    required = {"game_type", "pitcher_id", "batter_id"}
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns):
        raise ResidualError("reliability columns differ")
    output = np.zeros(len(frame), dtype="float64")
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    pitchers = frame["pitcher_id"].astype(str).to_numpy()
    batters = frame["batter_id"].astype(str).to_numpy()
    for index in np.flatnonzero(regular):
        output[index] = reliability(
            int(pitcher_counts.get(pitchers[index], 0)),
            int(batter_counts.get(batters[index], 0)),
            k=k,
        )
    return output
