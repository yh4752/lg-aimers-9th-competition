"""Cutoff-safe hierarchical target profiles for the privileged tree campaign."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import load_contract


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class ProfileLevel:
    name: str
    keys: tuple[str, ...]
    parent: str
    family: str


LEVELS = (
    ProfileLevel("pitcher", ("pitcher_id",), "global", "identity"),
    ProfileLevel("batter", ("batter_id",), "global", "identity"),
    ProfileLevel("pitcher_count", ("pitcher_id", "balls_before", "strikes_before"), "pitcher", "interaction"),
    ProfileLevel("pitcher_batter_hand", ("pitcher_id", "batter_hand"), "pitcher", "interaction"),
    ProfileLevel("pitcher_base", ("pitcher_id", "base_state"), "pitcher", "interaction"),
    ProfileLevel("pitcher_game", ("pitcher_id", "game_type"), "pitcher", "interaction"),
    ProfileLevel("batter_pitcher_hand", ("batter_id", "pitcher_hand"), "batter", "interaction"),
    ProfileLevel("batter_count", ("batter_id", "balls_before", "strikes_before"), "batter", "interaction"),
    ProfileLevel("team_count", ("pitcher_team_id", "balls_before", "strikes_before"), "global", "interaction"),
    ProfileLevel("direct_matchup", ("pitcher_id", "batter_id"), "global", "matchup"),
)
_KEY_COLUMNS = tuple(dict.fromkeys(key for level in LEVELS for key in level.keys))


@dataclass(frozen=True)
class ProfileStrengths:
    identity: int
    interaction: int
    matchup: int


@dataclass(frozen=True)
class ProfileTable:
    keys: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class ProfileState:
    cutoff_year: int
    strengths: ProfileStrengths
    global_count: int
    global_rate: float
    tables: Mapping[str, ProfileTable]


@dataclass(frozen=True)
class StrengthSelection:
    selected: ProfileStrengths
    scores: Mapping[str, float]


def _strengths(value: ProfileStrengths | tuple[int, int, int]) -> ProfileStrengths:
    result = value if type(value) is ProfileStrengths else ProfileStrengths(*value)
    if type(result) is not ProfileStrengths or any(type(item) is not int or item <= 0 for item in (
        result.identity, result.interaction, result.matchup,
    )):
        raise ProfileError("profile strengths differ")
    return result


def _validate_rows(rows: object, *, targets: bool) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame or not rows.columns.is_unique:
        raise ProfileError("profile rows must be a DataFrame with unique columns")
    required = {*_KEY_COLUMNS, "season"}
    if targets:
        required.add("control_success")
    elif "control_success" in rows.columns:
        raise ProfileError("profile transform rows must not contain targets")
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ProfileError(f"profile rows are missing {missing}")
    result = rows.copy(deep=True)
    seasons = pd.to_numeric(result["season"], errors="coerce")
    if seasons.isna().any() or not np.equal(seasons, np.floor(seasons)).all():
        raise ProfileError("profile seasons differ")
    result["season"] = seasons.astype("int64")
    if result.loc[:, _KEY_COLUMNS].isna().any().any():
        raise ProfileError("profile keys must not be missing")
    if targets:
        target = result["control_success"]
        if not target.isin([0, 1]).all() or target.map(type).eq(bool).any():
            raise ProfileError("profile target must be binary")
        result["control_success"] = target.astype("int8")
    return result


def fit_profiles(
    train: pd.DataFrame,
    *,
    cutoff_year: int,
    strengths: ProfileStrengths | tuple[int, int, int],
) -> ProfileState:
    rows = _validate_rows(train, targets=True)
    if type(cutoff_year) is not int or rows["season"].gt(cutoff_year).any():
        raise ProfileError("profile rows exceed cutoff")
    selected = _strengths(strengths)
    count = len(rows)
    global_rate = float(rows["control_success"].mean()) if count else 0.5
    tables: dict[str, ProfileTable] = {}
    for level in LEVELS:
        grouped = rows.groupby(list(level.keys), sort=True, dropna=False)["control_success"].agg(["count", "sum"]).reset_index()
        table_rows = tuple(
            tuple(values[:-2]) + (int(values[-2]), float(values[-1]))
            for values in grouped.itertuples(index=False, name=None)
        )
        tables[level.name] = ProfileTable(level.keys, table_rows)
    return ProfileState(
        cutoff_year=cutoff_year, strengths=selected, global_count=count, global_rate=global_rate,
        tables=MappingProxyType(tables),
    )


def _table_values(source: pd.DataFrame, table: ProfileTable) -> tuple[np.ndarray, np.ndarray]:
    columns = [*table.keys, "_count", "_sum"]
    lookup = pd.DataFrame(table.rows, columns=columns)
    if len(table.keys) == 1:
        indexed = lookup.set_index(table.keys[0])
        count = source[table.keys[0]].map(indexed["_count"])
        successes = source[table.keys[0]].map(indexed["_sum"])
    else:
        indexed = lookup.set_index(list(table.keys))
        keys = pd.MultiIndex.from_frame(source.loc[:, table.keys])
        count = pd.Series(indexed["_count"].reindex(keys).to_numpy(), index=source.index)
        successes = pd.Series(indexed["_sum"].reindex(keys).to_numpy(), index=source.index)
    return count.fillna(0).to_numpy(dtype="float64"), successes.fillna(0).to_numpy(dtype="float64")


def _clipped_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def transform_profiles(rows: pd.DataFrame, state: ProfileState) -> pd.DataFrame:
    source = _validate_rows(rows, targets=False)
    if type(state) is not ProfileState:
        raise ProfileError("profile state differs")
    output = pd.DataFrame(index=source.index)
    rate_by_level: dict[str, np.ndarray] = {}
    minimum_rows = load_contract().profiles.minimum_rows
    for level in LEVELS:
        table = state.tables[level.name]
        count, successes = _table_values(source, table)
        parent = np.full(len(source), state.global_rate, dtype="float64") if level.parent == "global" else rate_by_level[level.parent]
        raw = np.divide(successes, count, out=parent.copy(), where=count > 0)
        minimum = int(minimum_rows[level.family])
        known = count >= minimum
        strength = float(getattr(state.strengths, level.family))
        shrunk = np.divide(successes + strength * parent, count + strength)
        rate = np.where(known, shrunk, parent)
        prefix = f"profile_{level.name}"
        output[f"{prefix}_count"] = count
        output[f"{prefix}_raw_rate"] = raw
        output[f"{prefix}_parent_rate"] = parent
        output[f"{prefix}_rate"] = rate
        output[f"{prefix}_known"] = known.astype("float64")
        output[f"{prefix}_delta"] = rate - parent
        output[f"{prefix}_logit_delta"] = _clipped_logit(rate) - _clipped_logit(parent)
        rate_by_level[level.name] = rate
    return output.reset_index(drop=True)


def _neutral_profiles(rows: pd.DataFrame) -> pd.DataFrame:
    empty = rows.iloc[:0].copy()
    empty["control_success"] = pd.Series(dtype="int8")
    state = fit_profiles(empty, cutoff_year=int(rows["season"].min()) - 1, strengths=(25, 50, 100))
    return transform_profiles(rows.drop(columns=["control_success"], errors="ignore"), state)


def build_training_profiles(
    rows: pd.DataFrame,
    *,
    valid_year: int,
    strengths: ProfileStrengths | tuple[int, int, int],
) -> pd.DataFrame:
    source = _validate_rows(rows, targets=True)
    if source["season"].ge(valid_year).any():
        raise ProfileError("training profiles include validation or future rows")
    selected = _strengths(strengths)
    parts: list[pd.DataFrame] = []
    for season in sorted(source["season"].unique()):
        current = source.loc[source["season"].eq(season)]
        earlier = source.loc[source["season"].lt(season)]
        if earlier.empty:
            transformed = _neutral_profiles(current)
        else:
            state = fit_profiles(earlier, cutoff_year=int(season) - 1, strengths=selected)
            transformed = transform_profiles(current.drop(columns="control_success"), state)
        transformed.index = current.index
        parts.append(transformed)
    return pd.concat(parts).sort_index().reset_index(drop=True)


def _profile_prediction(frame: pd.DataFrame) -> np.ndarray:
    columns = [name for name in frame if name.endswith("_rate") and not name.endswith("_raw_rate") and not name.endswith("_parent_rate")]
    return frame.loc[:, columns].mean(axis=1).to_numpy(dtype="float64")


def select_strengths(train: pd.DataFrame, folds: tuple[tuple[int, int], ...]) -> StrengthSelection:
    source = _validate_rows(train, targets=True)
    contract = load_contract()
    scores: dict[str, float] = {}
    candidates = [ProfileStrengths(*values) for values in product(
        contract.profiles.identity_strengths,
        contract.profiles.interaction_strengths,
        contract.profiles.matchup_strengths,
    )]
    prepared_folds = []
    for train_end, valid_year in folds[:2]:
        fit_rows = source.loc[source["season"].le(train_end)]
        valid_rows = source.loc[source["season"].eq(valid_year)]
        if fit_rows.empty or valid_rows.empty:
            raise ProfileError("strength selection fold is empty")
        prepared_folds.append((
            fit_profiles(fit_rows, cutoff_year=train_end, strengths=candidates[0]),
            valid_rows,
        ))
    for candidate in candidates:
        fold_scores = []
        for base_state, valid_rows in prepared_folds:
            state = replace(base_state, strengths=candidate)
            transformed = transform_profiles(valid_rows.drop(columns="control_success"), state)
            prediction = _profile_prediction(transformed)
            target = valid_rows["control_success"].to_numpy(dtype="float64")
            fold_scores.append(float(np.mean(np.square(target - prediction))))
        key = f"{candidate.identity}:{candidate.interaction}:{candidate.matchup}"
        scores[key] = float(np.mean(fold_scores))
    selected_key = min(
        scores,
        key=lambda key: (scores[key], tuple(-int(value) for value in key.split(":")), key),
    )
    return StrengthSelection(
        selected=ProfileStrengths(*(int(value) for value in selected_key.split(":"))),
        scores=MappingProxyType(scores),
    )
