from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd


class S4TemporalError(ValueError):
    pass


STRUCTURE_FOLDS = ((2021, 2022), (2022, 2023))
CONFIRMATION_FOLD = (2023, 2024)


@dataclass(frozen=True)
class AnchorEvidence:
    candidate_id: str
    fold_gains: Mapping[tuple[int, int], float]
    weighted_gain: float
    worst_fold_gain: float
    residual_correlation: float
    route_by_game_type: bool
    mandatory_role: str | None = None


def _vector(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype="float64")
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise S4TemporalError(f"{label} must be a finite vector")
    return result


def probability_vector(value: object, label: str) -> np.ndarray:
    result = _vector(value, label)
    if np.any((result < 0.0) | (result > 1.0)):
        raise S4TemporalError(f"{label} must contain probabilities")
    return result


def season_decay_weights(years: object, *, cutoff_year: int, decay: float) -> np.ndarray:
    values = np.asarray(years)
    if values.ndim != 1 or values.size == 0:
        raise S4TemporalError("training seasons must be a vector")
    try:
        numeric = values.astype("int64")
    except (TypeError, ValueError) as error:
        raise S4TemporalError("training seasons must be integers") from error
    if not np.array_equal(numeric.astype(str), values.astype(str)) or np.any(numeric > cutoff_year):
        raise S4TemporalError("training seasons exceed cutoff")
    if type(decay) not in {int, float} or not np.isfinite(decay) or not 0.0 < float(decay) <= 1.0:
        raise S4TemporalError("decay must be in (0, 1]")
    return np.power(float(decay), cutoff_year - numeric).astype("float64")


def blend_anchor(recent: object, multi: object, *, recent_weight: float) -> np.ndarray:
    left = probability_vector(recent, "recent")
    right = probability_vector(multi, "multi")
    if left.shape != right.shape:
        raise S4TemporalError("anchor probability rows differ")
    if type(recent_weight) not in {int, float} or not 0.0 <= float(recent_weight) <= 1.0:
        raise S4TemporalError("recent weight differs")
    return np.clip(float(recent_weight) * left + (1.0 - float(recent_weight)) * right, 1e-5, 1.0 - 1e-5)


def align_probability_frames(left: pd.DataFrame, right: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"row_id", "probability"}
    if set(left.columns) < required or set(right.columns) < required:
        raise S4TemporalError("probability frame columns differ")
    if left["row_id"].duplicated().any() or right["row_id"].duplicated().any():
        raise S4TemporalError("probability row identity is duplicated")
    if left["row_id"].tolist() != right["row_id"].tolist():
        raise S4TemporalError("probability row identity differs")
    return probability_vector(left["probability"], "left probability"), probability_vector(right["probability"], "right probability")


def structure_fold_frames(
    frames: Mapping[tuple[int, int], pd.DataFrame],
    *,
    on_read: Callable[[tuple[int, int]], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not set(STRUCTURE_FOLDS).issubset(frames):
        raise S4TemporalError("structure fold frames are absent")
    output: list[pd.DataFrame] = []
    for fold in STRUCTURE_FOLDS:
        if on_read is not None:
            on_read(fold)
        output.append(frames[fold].copy(deep=True))
    return output[0], output[1]
