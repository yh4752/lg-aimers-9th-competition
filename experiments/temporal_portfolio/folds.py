"""Cutoff-safe temporal expert row selection and sample weights."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import math
from numbers import Integral, Real
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import TemporalFold


class FoldError(ValueError):
    """Raised when temporal training rows cannot be selected safely."""


@dataclass(frozen=True)
class WeightedRows:
    frame: pd.DataFrame
    sample_weight: np.ndarray
    row_sha256: str

    def __post_init__(self) -> None:
        frame = _validated_frame_copy(self.frame)
        weights = self.sample_weight
        if type(weights) is not np.ndarray:
            raise FoldError("sample weights must be a NumPy array")
        if weights.ndim != 1:
            raise FoldError("sample weights must be one-dimensional")
        if len(frame) != len(weights):
            raise FoldError("frame and sample weight length differ")
        try:
            copied_weights = np.array(weights, dtype="float32", copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise FoldError("sample weights must be numeric float32 values") from error
        if not np.isfinite(copied_weights).all():
            raise FoldError("sample weights must be finite")
        if not np.greater(copied_weights, 0).all():
            raise FoldError("sample weights must be positive")
        expected_digest = row_id_sha256(frame["row_id"])
        if type(self.row_sha256) is not str or self.row_sha256 != expected_digest:
            raise FoldError("row digest differs from the weighted frame")
        copied_weights.flags.writeable = False
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "sample_weight", copied_weights)


def row_id_sha256(row_ids: Iterable[str]) -> str:
    """Hash an ordered row-ID sequence with collision-safe length prefixes."""

    if type(row_ids) is str:
        raise FoldError("row_id sequence must contain strings")
    digest = sha256()
    try:
        for row_id in row_ids:
            if type(row_id) is not str or not row_id.strip():
                raise FoldError("row_id sequence must contain nonempty strings")
            encoded = row_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    except FoldError:
        raise
    except Exception as error:
        raise FoldError("row_id sequence cannot be hashed") from error
    return digest.hexdigest()


def select_training_rows(
    frame: pd.DataFrame,
    fold: TemporalFold,
    *,
    expert: str,
    decay: Decimal | None,
    prefiltered: bool = False,
    allow_extra: bool = False,
) -> WeightedRows:
    """Select one expert's complete training seasons without crossing its cutoff."""

    _validate_fold(fold)
    if type(prefiltered) is not bool:
        raise FoldError("prefiltered must be an exact bool")
    if type(allow_extra) is not bool:
        raise FoldError("allow_extra must be an exact bool")
    years, approved_decay = _expert_contract(fold, expert, decay)
    prepared = _validated_frame_copy(frame)
    season = prepared["season"]

    if prefiltered and not season.isin(years).all():
        raise FoldError("prefiltered rows differ from the chosen expert seasons")
    if not allow_extra and season.gt(fold.multi_end).any():
        raise FoldError("training rows exceed the fold cutoff")

    selected = prepared.loc[season.isin(years)].copy(deep=True)
    if selected.empty:
        raise FoldError("training rows are empty")
    observed_years = frozenset(int(year) for year in selected["season"])
    missing_years = tuple(year for year in years if year not in observed_years)
    if missing_years:
        raise FoldError(f"required training seasons are missing: {missing_years}")

    if approved_decay is None:
        weights = np.ones(len(selected), dtype="float32")
    else:
        weights = np.asarray(
            [
                approved_decay ** (fold.multi_end - int(season_year))
                for season_year in selected["season"]
            ],
            dtype="float32",
        )
        if not np.isfinite(weights).all() or not np.greater(weights, 0).all():
            raise FoldError("decay produced non-finite or non-positive sample weights")

    return WeightedRows(
        selected,
        weights,
        row_id_sha256(selected["row_id"]),
    )


def _validate_fold(fold: object) -> None:
    if type(fold) is not TemporalFold:
        raise FoldError("fold type must be exactly TemporalFold")
    years = (fold.recent_year, fold.multi_start, fold.multi_end, fold.valid_year)
    if any(type(year) is not int or year <= 0 for year in years):
        raise FoldError("fold years must be positive exact integers")
    if not (
        fold.multi_start <= fold.recent_year == fold.multi_end < fold.valid_year
    ):
        raise FoldError("fold years have invalid temporal ordering")


def _expert_contract(
    fold: TemporalFold, expert: object, decay: object
) -> tuple[tuple[int, ...], Decimal | None]:
    if type(expert) is not str:
        raise FoldError("expert and decay differ from the approved contract")
    if expert == "recent":
        if decay is not None:
            raise FoldError("expert and decay differ from the approved contract")
        return (fold.recent_year,), None
    if expert != "multi" or type(decay) is not Decimal:
        raise FoldError("expert and decay differ from the approved contract")
    if not decay.is_finite() or decay <= Decimal(0) or decay > Decimal(1):
        raise FoldError("multi decay must be finite and satisfy 0 < decay <= 1")
    return tuple(range(fold.multi_start, fold.multi_end + 1)), decay


def _validated_frame_copy(frame: object) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame:
        raise FoldError("training frame must be an actual pandas DataFrame")
    if frame.columns.has_duplicates:
        raise FoldError("training frame has duplicate columns")
    if not {"season", "row_id"}.issubset(frame.columns):
        raise FoldError("training frame is missing exact required columns")

    copied = frame.copy(deep=True)
    copied["season"] = _canonical_seasons(copied["season"])
    copied["row_id"] = _canonical_row_ids(copied["row_id"])
    return copied


def _canonical_seasons(values: pd.Series) -> np.ndarray:
    years: list[int] = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise FoldError("season must contain finite numeric integer years")
        if isinstance(value, Integral):
            year = int(value)
        elif type(value) is Decimal:
            if not value.is_finite() or value != value.to_integral_value():
                raise FoldError("season must contain finite numeric integer years")
            year = int(value)
        elif isinstance(value, Real):
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer() or value != int(numeric):
                raise FoldError("season must contain finite numeric integer years")
            year = int(numeric)
        else:
            raise FoldError("season must contain finite numeric integer years")
        years.append(year)
    try:
        return np.asarray(years, dtype="int64")
    except (OverflowError, TypeError, ValueError) as error:
        raise FoldError("season must fit exact integer years") from error


def _canonical_row_ids(values: pd.Series) -> list[str]:
    try:
        if values.isna().any():
            raise FoldError("row_id must be non-null")
        canonical = [str(value) for value in values.tolist()]
    except FoldError:
        raise
    except Exception as error:
        raise FoldError("row_id values cannot be converted to strings") from error
    if any(not row_id.strip() for row_id in canonical):
        raise FoldError("row_id must be nonempty")
    if len(set(canonical)) != len(canonical):
        raise FoldError("row_id must remain unique after string canonicalization")
    return canonical
