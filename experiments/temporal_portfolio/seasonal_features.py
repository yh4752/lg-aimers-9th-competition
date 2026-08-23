"""Cutoff-safe adapter for the proven independent-DL seasonal features."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import math
from numbers import Integral, Real
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_extension_array_dtype,
    is_numeric_dtype,
    is_scalar,
)

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
_TRAIN_POSITION_COLUMN = "__s1_training_position__"
_PITCHER_SNAPSHOT_COLUMNS = (
    "pitcher_id",
    "snapshot_pitcher_success_n",
    "snapshot_pitcher_success_count",
    *(
        item
        for component in ("reverse", "middle", "ball", "strike")
        for item in (
            f"snapshot_pitcher_{component}_n",
            f"snapshot_pitcher_{component}_count",
        )
    ),
    *(
        item
        for component in ("fastball", "breaking", "offspeed")
        for item in (
            f"snapshot_pitchmix_{component}_n",
            f"snapshot_pitchmix_{component}_count",
        )
    ),
)
_BATTER_SNAPSHOT_COLUMNS = (
    "batter_id",
    "snapshot_batter_success_n",
    "snapshot_batter_success_count",
    "snapshot_batter_middle_n",
    "snapshot_batter_middle_count",
)
_GENERATED_COLUMNS = (
    "season_pitcher_n",
    "season_pitcher_log1p_n",
    "season_pitcher_reliability_100",
    *(
        f"season_pitcher_success_smooth_{strength}"
        for strength in (10, 25, 50, 100, 200, 500)
    ),
    "season_pitcher_success_rate",
    "season_vs_career_success",
    *(
        f"season_pitcher_{component}_smooth_50"
        for component in ("reverse", "middle", "ball", "strike")
    ),
    *(f"season_vs_prev{games}_game_success_rate" for games in (1, 3, 5)),
    *(
        f"season_pitchmix_{component}_smooth_50"
        for component in ("fastball", "breaking", "offspeed")
    ),
    "season_pitchmix_n",
    "season_batter_n",
    "season_batter_log1p_n",
    *(f"season_batter_success_smooth_{strength}" for strength in (50, 100, 200, 500)),
    "season_pitcher_batter_success_gap",
    "season_pitcher_n_bucket",
    "season_batter_n_bucket",
)
_RESERVED_TRANSFORM_COLUMNS = frozenset(
    (
        *_PITCHER_SNAPSHOT_COLUMNS[1:],
        *_BATTER_SNAPSHOT_COLUMNS[1:],
        *_GENERATED_COLUMNS,
        _POSITION_COLUMN,
    )
)


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
        _validate_state(state)
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
    if max(seasons) != valid_year - 1:
        raise SeasonalFeatureError(
            "S1 fit rows must include the immediate previous season"
        )
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

    _validate_state(state)
    prepared = _validated_frame(
        rows,
        required=_TRANSFORM_COLUMNS,
        label="S1 transform",
        reject_target=True,
        reserved=_RESERVED_TRANSFORM_COLUMNS,
    )
    seasons = _validated_seasons(prepared["season"])
    if any(season != state.valid_year for season in seasons):
        raise SeasonalFeatureError("S1 transform season differs from valid_year")
    _validate_entities(prepared)
    _validate_numeric_columns(
        prepared,
        (*_SNAPSHOT_NUMERIC_COLUMNS, *_RECENT_COLUMNS),
    )

    working = prepared
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


def build_training_s1(
    train: pd.DataFrame, *, valid_year: int
) -> tuple[S1State, pd.DataFrame]:
    """Build leakage-safe S1 columns for training rows and the validation state.

    Every non-earliest season is transformed against a snapshot ending in the
    immediately preceding season.  The earliest season has no eligible labeled
    history, so it uses an empty snapshot and the neutral binary prior.  The
    returned state alone is fitted through ``valid_year - 1`` and is intended
    for validation/inference rows.
    """

    _validate_valid_year(valid_year)
    prepared = _validated_frame(
        train,
        required=(*_FIT_COLUMNS, *_RECENT_COLUMNS),
        label="S1 training",
    )
    seasons = _validated_seasons(prepared["season"])
    if not seasons:
        raise SeasonalFeatureError("S1 training rows are empty")
    if any(season >= valid_year for season in seasons):
        raise SeasonalFeatureError("S1 training rows reach validation season")
    if max(seasons) != valid_year - 1:
        raise SeasonalFeatureError(
            "S1 training rows must include the immediate previous season"
        )
    _validate_entities(prepared)
    _validate_numeric_columns(prepared, _SNAPSHOT_NUMERIC_COLUMNS)
    _validated_target(prepared[TARGET_COLUMN])

    outputs: list[pd.DataFrame] = []
    unique_years = tuple(sorted(set(seasons)))
    positions = np.arange(len(prepared), dtype="int64")
    for season in unique_years:
        mask = prepared["season"].eq(season).to_numpy()
        selected = prepared.iloc[np.flatnonzero(mask)]
        unlabeled = selected.drop(columns=TARGET_COLUMN)
        prior_rows = prepared.loc[prepared["season"].lt(season)]
        if prior_rows.empty:
            snapshot = SeasonalSnapshot(
                cutoff_year=season - 1,
                pitcher=pd.DataFrame(columns=_PITCHER_SNAPSHOT_COLUMNS),
                batter=pd.DataFrame(columns=_BATTER_SNAPSHOT_COLUMNS),
            )
            added = _attach_with_snapshot(
                unlabeled, snapshot=snapshot, prior_rate=0.5
            )
        else:
            if int(prior_rows["season"].max()) != season - 1:
                raise SeasonalFeatureError(
                    "S1 training seasons must be contiguous for prior snapshots"
                )
            season_state = fit_s1_state(prior_rows, valid_year=season)
            added = transform_s1(unlabeled, season_state)
        added[_TRAIN_POSITION_COLUMN] = positions[mask]
        outputs.append(added)

    combined = pd.concat(outputs, axis=0, ignore_index=True)
    combined = combined.sort_values(_TRAIN_POSITION_COLUMN, kind="stable")
    if not np.array_equal(
        combined[_TRAIN_POSITION_COLUMN].to_numpy(dtype="int64"), positions
    ):
        raise SeasonalFeatureError("S1 training composition changed row identity")
    combined = combined.drop(columns=_TRAIN_POSITION_COLUMN)
    combined.index = train.index
    combined.attrs["categorical_columns"] = (
        "season_pitcher_n_bucket",
        "season_batter_n_bucket",
    )
    return fit_s1_state(prepared, valid_year=valid_year), combined


def _attach_with_snapshot(
    rows: pd.DataFrame, *, snapshot: SeasonalSnapshot, prior_rate: float
) -> pd.DataFrame:
    working = rows.copy(deep=True)
    working[_POSITION_COLUMN] = np.arange(len(working), dtype="int64")
    try:
        transformed, categorical = attach_seasonal_features(
            working, snapshot, prior_rate=prior_rate
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise SeasonalFeatureError("S1 cold-start transform failed") from error
    transformed = transformed.sort_values(_POSITION_COLUMN, kind="stable")
    transformed.index = rows.index
    added = [column for column in transformed.columns if column not in working.columns]
    result = transformed.loc[:, added].copy(deep=True)
    numeric = result.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise SeasonalFeatureError("S1 cold-start produced non-finite features")
    result.attrs["categorical_columns"] = tuple(categorical)
    return result


def _validated_frame(
    frame: object,
    *,
    required: Iterable[str],
    label: str,
    reject_target: bool = False,
    reserved: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame:
        raise SeasonalFeatureError(f"{label} rows must be an actual pandas DataFrame")
    if frame.columns.has_duplicates:
        raise SeasonalFeatureError(f"{label} rows have duplicate columns")
    if any(type(column) is not str for column in frame.columns):
        raise SeasonalFeatureError(f"{label} column names must be strings")
    columns = tuple(frame.columns)
    if reject_target and TARGET_COLUMN in columns:
        raise SeasonalFeatureError("S1 transform rows must not contain the target")
    collisions = sorted(reserved.intersection(columns))
    if collisions:
        raise SeasonalFeatureError(
            f"{label} rows contain reserved columns: {collisions}"
        )
    required_columns = tuple(required)
    missing = sorted(set(required_columns).difference(columns))
    if missing:
        raise SeasonalFeatureError(
            f"{label} rows are missing required columns: {missing}"
        )
    return frame.loc[:, required_columns].copy(deep=True)


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
        values = frame[column]
        if (
            is_numeric_dtype(values.dtype)
            and not is_bool_dtype(values.dtype)
            and not is_complex_dtype(values.dtype)
            and not is_extension_array_dtype(values.dtype)
        ):
            numeric = values.to_numpy(dtype="float64", copy=False)
            if np.isinf(numeric).any():
                raise SeasonalFeatureError(f"{column} must not contain infinity")
            continue
        for value in values.tolist():
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


def _validate_state(state: object) -> None:
    if type(state) is not S1State:
        raise SeasonalFeatureError("S1 state type must be exactly S1State")
    names = tuple(S1State.__dataclass_fields__)
    try:
        values = {name: object.__getattribute__(state, name) for name in names}
    except (AttributeError, KeyError, TypeError) as error:
        raise SeasonalFeatureError("S1 state is missing required fields") from error

    valid_year = values["valid_year"]
    cutoff_year = values["_cutoff_year"]
    prior_rate = values["prior_rate"]
    if type(valid_year) is not int or not 1000 <= valid_year <= 9999:
        raise SeasonalFeatureError("S1 state valid_year is invalid")
    if type(cutoff_year) is not int or cutoff_year != valid_year - 1:
        raise SeasonalFeatureError("S1 state cutoff differs from valid_year - 1")
    if (
        type(prior_rate) is not float
        or not math.isfinite(prior_rate)
        or not 0.0 <= prior_rate <= 1.0
    ):
        raise SeasonalFeatureError("S1 state prior_rate is invalid")

    _validate_snapshot_payload(
        columns=values["_pitcher_columns"],
        rows=values["_pitcher_rows"],
        expected_columns=_PITCHER_SNAPSHOT_COLUMNS,
        entity_column="pitcher_id",
    )
    _validate_snapshot_payload(
        columns=values["_batter_columns"],
        rows=values["_batter_rows"],
        expected_columns=_BATTER_SNAPSHOT_COLUMNS,
        entity_column="batter_id",
    )


def _validate_snapshot_payload(
    *,
    columns: object,
    rows: object,
    expected_columns: tuple[str, ...],
    entity_column: str,
) -> None:
    if (
        type(columns) is not tuple
        or any(type(column) is not str for column in columns)
        or columns != expected_columns
    ):
        raise SeasonalFeatureError(f"S1 state {entity_column} schema is invalid")
    if type(rows) is not tuple or not rows:
        raise SeasonalFeatureError(f"S1 state {entity_column} rows are invalid")
    seen: set[object] = set()
    count_pairs = tuple(
        (columns.index(column.removesuffix("count") + "n"), index)
        for index, column in enumerate(columns)
        if column.endswith("_count")
    )
    for row in rows:
        if type(row) is not tuple or len(row) != len(columns):
            raise SeasonalFeatureError(f"S1 state {entity_column} row shape is invalid")
        entity = row[0]
        if not is_scalar(entity):
            raise SeasonalFeatureError(f"S1 state {entity_column} entity is invalid")
        try:
            missing = bool(pd.isna(entity))
            hash(entity)
            duplicated = entity in seen
            if not missing and not duplicated:
                seen.add(entity)
        except Exception as error:
            raise SeasonalFeatureError(
                f"S1 state {entity_column} entity is invalid"
            ) from error
        if missing or duplicated:
            raise SeasonalFeatureError(f"S1 state {entity_column} entity is invalid")

        numeric: list[float] = []
        for value in row[1:]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (Real, Decimal)
            ):
                raise SeasonalFeatureError(
                    f"S1 state {entity_column} numeric payload is invalid"
                )
            try:
                converted = float(value)
            except Exception as error:
                raise SeasonalFeatureError(
                    f"S1 state {entity_column} numeric payload is invalid"
                ) from error
            if not math.isfinite(converted) or converted < 0.0:
                raise SeasonalFeatureError(
                    f"S1 state {entity_column} numeric payload is invalid"
                )
            numeric.append(converted)
        numeric_by_column = (0.0, *numeric)
        if any(
            numeric_by_column[count_index] > numeric_by_column[n_index]
            for n_index, count_index in count_pairs
        ):
            raise SeasonalFeatureError(
                f"S1 state {entity_column} count exceeds its n"
            )
