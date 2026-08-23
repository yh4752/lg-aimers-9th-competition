"""Cutoff-safe adapter for the proven independent-DL seasonal features."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import math
from numbers import Integral, Real
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from experiments.independent_dl.feature_sources.seasonal import (
    PITCHER_RATE_COLUMNS,
    PITCHMIX_RATE_COLUMNS,
    TARGET_COLUMN,
    SeasonalSnapshot,
    attach_seasonal_features,
    build_seasonal_snapshot,
)


class SeasonalFeatureError(ValueError):
    """Raised when S1 cannot be fit or transformed without crossing its cutoff."""


_ENTITY_COLUMNS = ("pitcher_id", "batter_id")
_SNAPSHOT_NUMERIC_COLUMNS = (
    "asof_pitcher_n",
    *PITCHER_RATE_COLUMNS,
    "asof_pitcher_pitchmix_n",
    *PITCHMIX_RATE_COLUMNS,
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
)
_RECENT_COLUMNS = tuple(
    f"asof_pitcher_prev{games}_game_success_rate" for games in (1, 3, 5)
)
_FIT_COLUMNS = ("season", *_ENTITY_COLUMNS, *_SNAPSHOT_NUMERIC_COLUMNS, TARGET_COLUMN)
_TRANSFORM_COLUMNS = (
    "season",
    *_ENTITY_COLUMNS,
    *_SNAPSHOT_NUMERIC_COLUMNS,
    *_RECENT_COLUMNS,
)
_POSITION_COLUMN = "__s1_input_position__"


@dataclass(frozen=True, init=False)
class S1State:
    """Immutable fit state whose public snapshot accessor returns defensive copies."""

    valid_year: int
    prior_rate: float
    _cutoff_year: int = field(repr=False)
    _pitcher_columns: tuple[str, ...] = field(repr=False)
    _pitcher_rows: tuple[tuple[object, ...], ...] = field(repr=False)
    _batter_columns: tuple[str, ...] = field(repr=False)
    _batter_rows: tuple[tuple[object, ...], ...] = field(repr=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("S1State instances must be created with fit_s1_state()")

    @classmethod
    def _from_snapshot(
        cls,
        *,
        valid_year: int,
        prior_rate: float,
        snapshot: SeasonalSnapshot,
    ) -> S1State:
        if snapshot.cutoff_year != valid_year - 1:
            raise SeasonalFeatureError("snapshot cutoff differs from valid_year - 1")
        pitcher_columns, pitcher_rows = _freeze_frame(snapshot.pitcher)
        batter_columns, batter_rows = _freeze_frame(snapshot.batter)
        state = object.__new__(cls)
        object.__setattr__(state, "valid_year", valid_year)
        object.__setattr__(state, "prior_rate", prior_rate)
        object.__setattr__(state, "_cutoff_year", snapshot.cutoff_year)
        object.__setattr__(state, "_pitcher_columns", pitcher_columns)
        object.__setattr__(state, "_pitcher_rows", pitcher_rows)
        object.__setattr__(state, "_batter_columns", batter_columns)
        object.__setattr__(state, "_batter_rows", batter_rows)
        return state

    @property
    def snapshot(self) -> SeasonalSnapshot:
        """Return a detached legacy snapshot for audit and compatibility."""

        return SeasonalSnapshot(
            cutoff_year=self._cutoff_year,
            pitcher=pd.DataFrame.from_records(
                self._pitcher_rows, columns=self._pitcher_columns
            ),
            batter=pd.DataFrame.from_records(
                self._batter_rows, columns=self._batter_columns
            ),
        )


def fit_s1_state(train: pd.DataFrame, *, valid_year: int) -> S1State:
    """Fit S1 from labeled rows strictly before one four-digit validation year."""

    _validate_valid_year(valid_year)
    prepared = _validated_frame(train, required=_FIT_COLUMNS, label="S1 fit")
    seasons = _validated_seasons(prepared["season"])
    if len(prepared) == 0:
        raise SeasonalFeatureError("S1 fit rows are empty")
    if any(season >= valid_year for season in seasons):
        raise SeasonalFeatureError("S1 fit rows reach validation season")
    _validate_entities(prepared)
    _validate_numeric_columns(prepared, _SNAPSHOT_NUMERIC_COLUMNS)
    target = _validated_target(prepared[TARGET_COLUMN])
    prior_rate = float(target.mean())
    if not math.isfinite(prior_rate):
        raise SeasonalFeatureError("S1 target prior must be finite")

    try:
        snapshot = build_seasonal_snapshot(prepared, cutoff_year=valid_year - 1)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise SeasonalFeatureError("S1 snapshot construction failed") from error
    return S1State._from_snapshot(
        valid_year=valid_year,
        prior_rate=prior_rate,
        snapshot=snapshot,
    )


def transform_s1(rows: pd.DataFrame, state: S1State) -> pd.DataFrame:
    """Return only S1 columns for unlabeled rows in the state's validation year.

    The target is deliberately rejected.  Training composition should remove it
    before calling this evaluation-facing adapter, making target non-use explicit.
    """

    if type(state) is not S1State:
        raise SeasonalFeatureError("S1 state type must be exactly S1State")
    prepared = _validated_frame(rows, required=_TRANSFORM_COLUMNS, label="S1 transform")
    if TARGET_COLUMN in prepared.columns:
        raise SeasonalFeatureError("S1 transform rows must not contain the target")
    if any(
        column.startswith(("season_", "snapshot_"))
        or column == _POSITION_COLUMN
        for column in prepared.columns
    ):
        raise SeasonalFeatureError("S1 transform rows contain reserved columns")
    seasons = _validated_seasons(prepared["season"])
    if any(season != state.valid_year for season in seasons):
        raise SeasonalFeatureError("S1 transform season differs from valid_year")
    _validate_entities(prepared)
    _validate_numeric_columns(
        prepared,
        (*_SNAPSHOT_NUMERIC_COLUMNS, *_RECENT_COLUMNS),
    )

    working = prepared.copy(deep=True)
    working[_POSITION_COLUMN] = np.arange(len(working), dtype="int64")
    try:
        transformed, categorical = attach_seasonal_features(
            working,
            state.snapshot,
            prior_rate=state.prior_rate,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise SeasonalFeatureError("S1 transform failed") from error
    if len(transformed) != len(working) or _POSITION_COLUMN not in transformed:
        raise SeasonalFeatureError("S1 transform changed the row count")
    transformed = transformed.sort_values(_POSITION_COLUMN, kind="stable")
    expected_positions = np.arange(len(working), dtype="int64")
    if not np.array_equal(
        transformed[_POSITION_COLUMN].to_numpy(dtype="int64"), expected_positions
    ):
        raise SeasonalFeatureError("S1 transform changed input row identity")
    transformed.index = prepared.index

    added = [column for column in transformed.columns if column not in working.columns]
    result = transformed.loc[:, added].copy(deep=True)
    numeric = result.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise SeasonalFeatureError("S1 transform produced non-finite features")
    result.attrs["categorical_columns"] = tuple(categorical)
    return result


def _validated_frame(
    frame: object,
    *,
    required: Iterable[str],
    label: str,
) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame:
        raise SeasonalFeatureError(f"{label} rows must be an actual pandas DataFrame")
    if frame.columns.has_duplicates:
        raise SeasonalFeatureError(f"{label} rows have duplicate columns")
    if any(type(column) is not str for column in frame.columns):
        raise SeasonalFeatureError(f"{label} column names must be strings")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise SeasonalFeatureError(
            f"{label} rows are missing required columns: {missing}"
        )
    return frame.copy(deep=True)


def _validate_valid_year(valid_year: object) -> None:
    if type(valid_year) is not int or not 1000 <= valid_year <= 9999:
        raise SeasonalFeatureError("valid_year must be an exact four-digit int")


def _validated_seasons(values: pd.Series) -> tuple[int, ...]:
    years: list[int] = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise SeasonalFeatureError(
                "season must contain finite numeric integer years"
            )
        if isinstance(value, Integral):
            year = int(value)
        elif type(value) is Decimal:
            if not value.is_finite() or value != value.to_integral_value():
                raise SeasonalFeatureError(
                    "season must contain finite numeric integer years"
                )
            year = int(value)
        elif isinstance(value, Real):
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise SeasonalFeatureError(
                    "season must contain finite numeric integer years"
                )
            year = int(numeric)
        else:
            raise SeasonalFeatureError(
                "season must contain finite numeric integer years"
            )
        if not 1000 <= year <= 9999:
            raise SeasonalFeatureError("season must contain four-digit years")
        years.append(year)
    return tuple(years)


def _validate_entities(frame: pd.DataFrame) -> None:
    for column in _ENTITY_COLUMNS:
        for value in frame[column].tolist():
            if not is_scalar(value):
                raise SeasonalFeatureError(f"{column} must contain scalar entity IDs")
            try:
                missing = bool(pd.isna(value))
                hash(value)
            except (TypeError, ValueError) as error:
                raise SeasonalFeatureError(
                    f"{column} must contain hashable non-null entity IDs"
                ) from error
            if missing:
                raise SeasonalFeatureError(
                    f"{column} must contain hashable non-null entity IDs"
                )


def _validate_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        for value in frame[column].tolist():
            if value is None or value is pd.NA:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (Real, Decimal)
            ):
                raise SeasonalFeatureError(f"{column} must contain numeric values")
            if not math.isfinite(float(value)):
                if bool(pd.isna(value)):
                    continue
                raise SeasonalFeatureError(f"{column} must not contain infinity")


def _validated_target(values: pd.Series) -> np.ndarray:
    target: list[float] = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, Decimal)
        ):
            raise SeasonalFeatureError("S1 target must be finite numeric binary values")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
            raise SeasonalFeatureError("S1 target must be finite numeric binary values")
        target.append(numeric)
    return np.asarray(target, dtype="float64")


def _freeze_frame(
    frame: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...]]:
    if type(frame) is not pd.DataFrame or frame.columns.has_duplicates:
        raise SeasonalFeatureError("S1 snapshot has an invalid frame")
    columns = tuple(str(column) for column in frame.columns)
    rows: list[tuple[object, ...]] = []
    for row in frame.itertuples(index=False, name=None):
        if not all(is_scalar(value) for value in row):
            raise SeasonalFeatureError("S1 snapshot contains a non-scalar value")
        rows.append(tuple(row))
    return columns, tuple(rows)
