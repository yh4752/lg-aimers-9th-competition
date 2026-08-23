"""Immutable, cutoff-bound pitcher TrackMan ablation bundles.

This module is deliberately an adapter over the proven independent-DL
TrackMan matcher.  It validates the adapter boundary, canonicalizes the
validated lookup, and partitions its fixed schema without reimplementing any
matching or aggregation logic.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from experiments.independent_dl.feature_sources.trackman import (
    MAIN_RATE_COLUMNS,
    PHYSICAL_COLUMNS,
    PITCHER_LOOKUP_COLUMNS,
    PITCH_GROUPS,
    build_trackman_lookup,
    validate_trackman_build_result,
)


class PitcherTrackmanError(ValueError):
    """Raised when a pitcher TrackMan state cannot be built or audited safely."""


P0 = (
    "tm_history_n",
    "tm_match_cost",
    "tm_match_margin",
    "tm_match_confidence",
    "tm_match_accepted",
)
P1_PREFIXES = ("tm_career_", "tm_recent_")
P2_PREFIXES = (
    "tm_fastball_",
    "tm_breaking_",
    "tm_offspeed_",
    "tm_other_",
    "tm_history_fastball_",
    "tm_history_breaking_",
    "tm_history_offspeed_",
    "tm_history_other_",
)
P2_EXACT = (
    "tm_fastball_breaking_speed_gap",
    "tm_fastball_offspeed_speed_gap",
)
P3_PREFIXES = ("tm_trend_",)

_BUNDLE_NAMES = ("P0", "P1", "P2", "P3")
_TRAIN_COLUMNS = (
    "season",
    "pitcher_id",
    "pitcher_hand",
    "pitcher_team_id",
    "asof_pitcher_n",
    *MAIN_RATE_COLUMNS,
)
_HISTORY_COLUMNS = (
    "season",
    "pitcher_trackman_id",
    "pitch_type_group",
    "pitcher_hand",
    "pitcher_team",
    *PHYSICAL_COLUMNS,
)


def _selected_names(
    *,
    exact: Iterable[str] = (),
    prefixes: Iterable[str] = (),
    exclude_prefixes: Iterable[str] = (),
) -> tuple[str, ...]:
    exact_set = frozenset(exact)
    included_prefixes = tuple(prefixes)
    excluded_prefixes = tuple(exclude_prefixes)
    return tuple(
        column
        for column in PITCHER_LOOKUP_COLUMNS
        if column != "pitcher_id"
        and (column in exact_set or column.startswith(included_prefixes))
        and not column.startswith(excluded_prefixes)
    )


_EXPECTED_BUNDLE_COLUMNS = {
    "P0": ("pitcher_id", *P0),
    "P1": (
        "pitcher_id",
        *_selected_names(prefixes=P1_PREFIXES, exclude_prefixes=P3_PREFIXES),
    ),
    "P2": (
        "pitcher_id",
        *_selected_names(exact=P2_EXACT, prefixes=P2_PREFIXES),
    ),
    "P3": ("pitcher_id", *_selected_names(prefixes=P3_PREFIXES)),
}


def _validate_static_partition() -> None:
    assigned: list[str] = []
    for name in _BUNDLE_NAMES:
        columns = _EXPECTED_BUNDLE_COLUMNS[name]
        if not columns or columns[0] != "pitcher_id" or len(columns) != len(set(columns)):
            raise RuntimeError(f"invalid pitcher TrackMan {name} selector")
        assigned.extend(columns[1:])
    expected = set(PITCHER_LOOKUP_COLUMNS) - {"pitcher_id"}
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise RuntimeError("pitcher TrackMan selectors do not exactly partition the lookup")


_validate_static_partition()


@dataclass(frozen=True, init=False)
class PitcherTrackmanState:
    """Immutable scalar representation of one canonical pitcher lookup."""

    cutoff_year: int
    _lookup_columns: tuple[str, ...] = field(repr=False)
    _lookup_dtypes: tuple[str, ...] = field(repr=False)
    _lookup_rows: tuple[tuple[object, ...], ...] = field(repr=False)
    _lookup_sha256: str = field(repr=False)
    _bundle_sha256_items: tuple[tuple[str, str], ...] = field(repr=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "PitcherTrackmanState instances must be created with "
            "fit_pitcher_trackman()"
        )

    @classmethod
    def _from_lookup(
        cls, *, cutoff_year: int, lookup: pd.DataFrame
    ) -> PitcherTrackmanState:
        columns, dtypes, rows = _freeze_frame(lookup)
        lookup_digest = _frame_sha256(
            lookup, cutoff_year=cutoff_year, identity="lookup"
        )
        bundle_digests = tuple(
            (
                name,
                _frame_sha256(
                    _select_bundle(lookup, name),
                    cutoff_year=cutoff_year,
                    identity=name,
                ),
            )
            for name in _BUNDLE_NAMES
        )
        state = object.__new__(cls)
        object.__setattr__(state, "cutoff_year", cutoff_year)
        object.__setattr__(state, "_lookup_columns", columns)
        object.__setattr__(state, "_lookup_dtypes", dtypes)
        object.__setattr__(state, "_lookup_rows", rows)
        object.__setattr__(state, "_lookup_sha256", lookup_digest)
        object.__setattr__(state, "_bundle_sha256_items", bundle_digests)
        _validate_state(state)
        return state

    @property
    def lookup(self) -> pd.DataFrame:
        """Return a detached copy of the canonical full lookup."""

        return _validated_state_lookup(self).copy(deep=True)

    @property
    def bundles(self) -> Mapping[str, pd.DataFrame]:
        """Return an immutable mapping whose frame values are defensive copies."""

        _validate_state(self)
        return _BundleMapping(self)

    @property
    def lookup_sha256(self) -> str:
        _validate_state(self)
        return self._lookup_sha256

    @property
    def bundle_sha256(self) -> Mapping[str, str]:
        _validate_state(self)
        return MappingProxyType(dict(self._bundle_sha256_items))


class _BundleMapping(Mapping[str, pd.DataFrame]):
    """Read-only bundle view that never returns state-owned mutable objects."""

    __slots__ = ("_state",)

    def __init__(self, state: PitcherTrackmanState) -> None:
        self._state = state

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in _BUNDLE_NAMES:
            raise KeyError(name)
        lookup = _validated_state_lookup(self._state)
        return _select_bundle(lookup, name).copy(deep=True)

    def __iter__(self) -> Iterator[str]:
        return iter(_BUNDLE_NAMES)

    def __len__(self) -> int:
        return len(_BUNDLE_NAMES)


def fit_pitcher_trackman(
    train: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff_year: int,
) -> PitcherTrackmanState:
    """Fit the proven TrackMan lookup and freeze its exact P0--P3 partition."""

    _validate_cutoff_year(cutoff_year)
    train_seasons = _validate_source_frame(
        train, required=_TRAIN_COLUMNS, label="TrackMan train"
    )
    history_seasons = _validate_source_frame(
        history, required=_HISTORY_COLUMNS, label="TrackMan history"
    )
    train_prefix = train.loc[np.asarray(train_seasons) <= cutoff_year, _TRAIN_COLUMNS]
    history_prefix = history.loc[
        np.asarray(history_seasons) <= cutoff_year, _HISTORY_COLUMNS
    ]
    if train_prefix.empty or history_prefix.empty:
        raise PitcherTrackmanError(
            "TrackMan train and history must both contain data through cutoff"
        )
    _validate_train_prefix(train_prefix)
    _validate_history_prefix(history_prefix)

    try:
        built = build_trackman_lookup(train, history, cutoff_year)
        lookup = validate_trackman_build_result(
            built, expected_cutoff_year=cutoff_year
        )
    except PitcherTrackmanError:
        raise
    except (
        AttributeError,
        FloatingPointError,
        IndexError,
        KeyError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise PitcherTrackmanError("TrackMan lookup construction failed") from error

    canonical = _canonical_lookup(lookup)
    return PitcherTrackmanState._from_lookup(
        cutoff_year=cutoff_year,
        lookup=canonical,
    )


def select_columns(
    lookup: pd.DataFrame,
    *,
    exact: Iterable[str] = (),
    prefixes: Iterable[str] = (),
    exclude_prefixes: Iterable[str] = (),
) -> pd.DataFrame:
    """Select one exact key-first projection in deterministic source order."""

    if type(lookup) is not pd.DataFrame or "pitcher_id" not in lookup.columns:
        raise PitcherTrackmanError("TrackMan column selection requires a valid lookup")
    exact_names = tuple(exact)
    if len(exact_names) != len(set(exact_names)):
        raise PitcherTrackmanError("TrackMan exact selector contains duplicates")
    missing = tuple(column for column in exact_names if column not in lookup.columns)
    if missing:
        raise PitcherTrackmanError(f"TrackMan exact selector is missing columns: {missing}")
    prefix_names = tuple(prefixes)
    if prefix_names:
        selected = _selected_names(
            exact=exact_names,
            prefixes=prefix_names,
            exclude_prefixes=exclude_prefixes,
        )
    else:
        excluded = tuple(exclude_prefixes)
        selected = tuple(
            column
            for column in exact_names
            if column != "pitcher_id" and not column.startswith(excluded)
        )
    return lookup.loc[:, ["pitcher_id", *selected]].copy(deep=True)


def _select_bundle(lookup: pd.DataFrame, name: str) -> pd.DataFrame:
    return lookup.loc[:, list(_EXPECTED_BUNDLE_COLUMNS[name])].copy(deep=True)


def _validate_cutoff_year(cutoff_year: object) -> None:
    if type(cutoff_year) is not int or not 1000 <= cutoff_year <= 9999:
        raise PitcherTrackmanError("cutoff_year must be an exact four-digit int")


def _validate_source_frame(
    frame: object, *, required: tuple[str, ...], label: str
) -> tuple[int, ...]:
    if type(frame) is not pd.DataFrame:
        raise PitcherTrackmanError(f"{label} must be an actual pandas DataFrame")
    if frame.columns.has_duplicates:
        raise PitcherTrackmanError(f"{label} has duplicate columns")
    if any(type(column) is not str for column in frame.columns):
        raise PitcherTrackmanError(f"{label} column names must be strings")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise PitcherTrackmanError(
            f"{label} is missing required columns: {missing}"
        )
    return _validated_seasons(frame["season"], label=label)


def _validated_seasons(values: pd.Series, *, label: str) -> tuple[int, ...]:
    years: list[int] = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise PitcherTrackmanError(
                f"{label} season must contain finite four-digit integer years"
            )
        if isinstance(value, Integral):
            year = int(value)
        elif type(value) is Decimal:
            if not value.is_finite() or value != value.to_integral_value():
                raise PitcherTrackmanError(
                    f"{label} season must contain finite four-digit integer years"
                )
            year = int(value)
        elif isinstance(value, Real):
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise PitcherTrackmanError(
                    f"{label} season must contain finite four-digit integer years"
                )
            year = int(numeric)
        else:
            raise PitcherTrackmanError(
                f"{label} season must contain finite four-digit integer years"
            )
        if not 1000 <= year <= 9999:
            raise PitcherTrackmanError(
                f"{label} season must contain finite four-digit integer years"
            )
        years.append(year)
    return tuple(years)


def _validate_train_prefix(frame: pd.DataFrame) -> None:
    _validate_integral_values(frame["pitcher_id"], "pitcher_id", minimum=0)
    _validate_integral_values(frame["asof_pitcher_n"], "asof_pitcher_n", minimum=0)
    _validate_scalar_values(frame["pitcher_hand"], "pitcher_hand")
    _validate_scalar_values(frame["pitcher_team_id"], "pitcher_team_id")
    for column in MAIN_RATE_COLUMNS:
        _validate_numeric_values(frame[column], column, allow_nan=True)


def _validate_history_prefix(frame: pd.DataFrame) -> None:
    _validate_integral_values(
        frame["pitcher_trackman_id"], "pitcher_trackman_id", minimum=0
    )
    for column in ("pitch_type_group", "pitcher_hand", "pitcher_team"):
        _validate_scalar_values(frame[column], column)
    unexpected_groups = sorted(
        set(frame["pitch_type_group"].tolist()).difference(PITCH_GROUPS), key=str
    )
    if unexpected_groups:
        raise PitcherTrackmanError(
            f"pitch_type_group contains unsupported values: {unexpected_groups}"
        )
    for column in PHYSICAL_COLUMNS:
        _validate_numeric_values(frame[column], column, allow_nan=True)


def _validate_scalar_values(values: pd.Series, label: str) -> None:
    for value in values.tolist():
        if not is_scalar(value):
            raise PitcherTrackmanError(f"{label} must contain scalar non-null values")
        try:
            missing = bool(pd.isna(value))
            hash(value)
        except Exception as error:
            raise PitcherTrackmanError(
                f"{label} must contain hashable non-null values"
            ) from error
        if missing:
            raise PitcherTrackmanError(f"{label} must contain scalar non-null values")


def _validate_integral_values(
    values: pd.Series, label: str, *, minimum: int
) -> None:
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Integral, Real, Decimal)
        ):
            raise PitcherTrackmanError(f"{label} must contain finite integer values")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise PitcherTrackmanError(
                f"{label} must contain finite integer values"
            ) from error
        if not math.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
            raise PitcherTrackmanError(f"{label} must contain finite integer values")


def _validate_numeric_values(
    values: pd.Series, label: str, *, allow_nan: bool
) -> None:
    for value in values.tolist():
        if value is None or value is pd.NA:
            if allow_nan:
                continue
            raise PitcherTrackmanError(f"{label} must contain numeric values")
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Integral, Real, Decimal)
        ):
            raise PitcherTrackmanError(f"{label} must contain numeric values")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise PitcherTrackmanError(f"{label} must contain numeric values") from error
        if math.isnan(numeric):
            if allow_nan:
                continue
            raise PitcherTrackmanError(f"{label} must contain numeric values")
        if math.isinf(numeric):
            raise PitcherTrackmanError(f"{label} must not contain infinity")


def _canonical_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    if type(lookup) is not pd.DataFrame:
        raise PitcherTrackmanError("TrackMan lookup must be an actual DataFrame")
    if lookup.columns.has_duplicates or tuple(lookup.columns) != PITCHER_LOOKUP_COLUMNS:
        raise PitcherTrackmanError("TrackMan lookup schema is invalid")
    _validate_integral_values(lookup["pitcher_id"], "pitcher_id", minimum=0)
    canonical_ids = [int(value) for value in lookup["pitcher_id"].tolist()]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise PitcherTrackmanError("TrackMan lookup pitcher_id must be unique")
    for column in PITCHER_LOOKUP_COLUMNS[1:]:
        _validate_numeric_values(lookup[column], column, allow_nan=True)
    canonical = lookup.copy(deep=True)
    canonical["pitcher_id"] = np.asarray(canonical_ids, dtype="int64")
    # Grouped floating reductions can differ in their last machine bits when
    # equivalent source rows arrive in another order.  Twelve decimal places
    # are well below the precision of the underlying measurements and make the
    # persisted identity independent of that incidental reduction order.
    canonical.loc[:, list(PITCHER_LOOKUP_COLUMNS[1:])] = canonical.loc[
        :, list(PITCHER_LOOKUP_COLUMNS[1:])
    ].round(12)
    canonical = canonical.sort_values("pitcher_id", kind="stable").reset_index(drop=True)
    return canonical.loc[:, list(PITCHER_LOOKUP_COLUMNS)]


def _freeze_frame(
    frame: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[object, ...], ...]]:
    columns = tuple(frame.columns)
    dtypes = tuple(str(dtype) for dtype in frame.dtypes)
    rows: list[tuple[object, ...]] = []
    for row in frame.itertuples(index=False, name=None):
        if not all(is_scalar(value) for value in row):
            raise PitcherTrackmanError("TrackMan lookup contains non-scalar values")
        rows.append(tuple(row))
    return columns, dtypes, tuple(rows)


def _thaw_frame(
    columns: tuple[str, ...],
    dtypes: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> pd.DataFrame:
    try:
        frame = pd.DataFrame.from_records(rows, columns=columns)
        for column, dtype in zip(columns, dtypes, strict=True):
            frame[column] = frame[column].astype(dtype)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise PitcherTrackmanError("Pitcher TrackMan state payload is invalid") from error
    return frame


def _frame_sha256(
    frame: pd.DataFrame, *, cutoff_year: int, identity: str
) -> str:
    digest = sha256()
    digest.update(b"pitcher-trackman-v1\0")
    digest.update(str(cutoff_year).encode("ascii"))
    digest.update(b"\0")
    digest.update(identity.encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(tuple(str(dtype) for dtype in frame.dtypes)).encode("utf-8"))
    digest.update(b"\0")
    try:
        hashed = pd.util.hash_pandas_object(
            frame, index=False, categorize=False
        ).to_numpy(dtype="uint64", copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise PitcherTrackmanError("TrackMan lookup identity cannot be computed") from error
    digest.update(hashed.tobytes(order="C"))
    return digest.hexdigest()


def _validated_state_lookup(state: object) -> pd.DataFrame:
    _validate_state(state)
    return _thaw_frame(
        object.__getattribute__(state, "_lookup_columns"),
        object.__getattribute__(state, "_lookup_dtypes"),
        object.__getattribute__(state, "_lookup_rows"),
    )


def _validate_state(state: object) -> None:
    if type(state) is not PitcherTrackmanState:
        raise PitcherTrackmanError(
            "Pitcher TrackMan state type must be exactly PitcherTrackmanState"
        )
    try:
        cutoff_year = object.__getattribute__(state, "cutoff_year")
        columns = object.__getattribute__(state, "_lookup_columns")
        dtypes = object.__getattribute__(state, "_lookup_dtypes")
        rows = object.__getattribute__(state, "_lookup_rows")
        lookup_digest = object.__getattribute__(state, "_lookup_sha256")
        bundle_items = object.__getattribute__(state, "_bundle_sha256_items")
    except (AttributeError, TypeError) as error:
        raise PitcherTrackmanError("Pitcher TrackMan state is missing fields") from error
    if type(cutoff_year) is not int or not 1000 <= cutoff_year <= 9999:
        raise PitcherTrackmanError("Pitcher TrackMan state cutoff is invalid")
    if columns != PITCHER_LOOKUP_COLUMNS:
        raise PitcherTrackmanError("Pitcher TrackMan state schema is invalid")
    if (
        type(dtypes) is not tuple
        or len(dtypes) != len(columns)
        or any(type(dtype) is not str for dtype in dtypes)
        or type(rows) is not tuple
        or any(type(row) is not tuple or len(row) != len(columns) for row in rows)
    ):
        raise PitcherTrackmanError("Pitcher TrackMan state payload is invalid")
    if not _is_sha256(lookup_digest):
        raise PitcherTrackmanError("Pitcher TrackMan state identity is invalid")
    if (
        type(bundle_items) is not tuple
        or len(bundle_items) != len(_BUNDLE_NAMES)
        or any(type(item) is not tuple or len(item) != 2 for item in bundle_items)
    ):
        raise PitcherTrackmanError("Pitcher TrackMan state identity is invalid")
    if (
        tuple(item[0] for item in bundle_items) != _BUNDLE_NAMES
        or any(not _is_sha256(item[1]) for item in bundle_items)
    ):
        raise PitcherTrackmanError("Pitcher TrackMan state identity is invalid")

    lookup = _thaw_frame(columns, dtypes, rows)
    try:
        canonical = _canonical_lookup(lookup)
    except PitcherTrackmanError as error:
        raise PitcherTrackmanError("Pitcher TrackMan state lookup is invalid") from error
    if not canonical.equals(lookup):
        raise PitcherTrackmanError("Pitcher TrackMan state lookup is not canonical")
    if _frame_sha256(canonical, cutoff_year=cutoff_year, identity="lookup") != lookup_digest:
        raise PitcherTrackmanError("Pitcher TrackMan state lookup identity differs")
    observed_bundle_digests = tuple(
        (
            name,
            _frame_sha256(
                _select_bundle(canonical, name),
                cutoff_year=cutoff_year,
                identity=name,
            ),
        )
        for name in _BUNDLE_NAMES
    )
    if observed_bundle_digests != bundle_items:
        raise PitcherTrackmanError("Pitcher TrackMan state bundle identity differs")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
