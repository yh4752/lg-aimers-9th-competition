from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .hc_contracts import HCProfile


class HCFeatureError(ValueError):
    """Raised when hierarchy features cross a temporal or row-local boundary."""


_MISSING = "__MISSING__"
_REQUIRED = {
    "row_id",
    "season",
    "game_type",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
}
_LEVELS = (
    ("game_type", ("game_type",), "global", "context"),
    ("pitcher", ("pitcher_id",), "global", "identity"),
    ("batter", ("batter_id",), "global", "identity"),
    ("hand_matchup", ("pitcher_hand", "batter_hand"), "global", "context"),
    ("count", ("balls_before", "strikes_before"), "global", "context"),
    ("outs", ("outs_before",), "global", "context"),
    ("base_state", ("base_state",), "global", "context"),
    ("pitcher_game", ("pitcher_id", "game_type"), "pitcher", "interaction"),
    (
        "batter_pitcher_hand",
        ("batter_id", "pitcher_hand"),
        "batter",
        "interaction",
    ),
)


@dataclass(frozen=True)
class HierarchyEntry:
    count: int
    raw_rate: float
    parent_rate: float
    shrunk_rate: float
    parent_level: str
    parent_key: tuple[str, ...]


@dataclass(frozen=True)
class HierarchyLevelState:
    name: str
    keys: tuple[str, ...]
    parent_level: str
    strength: float
    minimum_rows: int
    lookup: Mapping[tuple[str, ...], HierarchyEntry]


@dataclass(frozen=True)
class HierarchyState:
    cutoff_year: int
    profile_name: str
    source_seasons: tuple[int, ...]
    global_count: int
    global_rate: float
    levels: Mapping[str, HierarchyLevelState]


def shrink(*, raw: float, parent: float, count: int, strength: float) -> float:
    values = (raw, parent, strength)
    if (
        any(type(value) not in {int, float} or type(value) is bool for value in values)
        or type(count) is not int
        or type(count) is bool
        or not all(math.isfinite(float(value)) for value in values)
        or count < 0
        or strength <= 0
    ):
        raise HCFeatureError("shrinkage inputs differ")
    return float(parent) + count / (count + float(strength)) * (
        float(raw) - float(parent)
    )


def _normalize(value: object) -> str:
    return _MISSING if pd.isna(value) else str(value)


def _key(values: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(_normalize(value) for value in values)


def _validate_rows(rows: object, *, labeled: bool) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame or rows.columns.has_duplicates:
        raise HCFeatureError("hierarchy rows must be a DataFrame with unique columns")
    required = set(_REQUIRED)
    if labeled:
        required.add("control_success")
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise HCFeatureError(f"hierarchy rows are missing columns: {missing}")
    if rows.empty:
        raise HCFeatureError("hierarchy rows are empty")
    return rows.copy(deep=True)


def _strength(profile: HCProfile, family: str) -> float:
    return {
        "identity": profile.identity_k,
        "context": profile.context_k,
        "interaction": profile.interaction_k,
    }[family]


def _parent_key(level: str, key: tuple[str, ...]) -> tuple[str, ...]:
    if level in {"pitcher_game", "batter_pitcher_hand"}:
        return (key[0],)
    return ()


def fit_hierarchy(
    rows: pd.DataFrame,
    *,
    cutoff_year: int,
    profile_name: str,
    profile: HCProfile,
    minimum_group_rows: Mapping[str, int],
) -> HierarchyState:
    frame = _validate_rows(rows, labeled=True)
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    if seasons.isna().any() or seasons.gt(cutoff_year).any():
        raise HCFeatureError("hierarchy fit contains rows after cutoff")
    target = pd.to_numeric(frame["control_success"], errors="coerce")
    if not target.isin([0, 1]).all():
        raise HCFeatureError("hierarchy target differs")
    if set(minimum_group_rows) != {"identity", "context", "interaction"} or any(
        type(value) is not int or type(value) is bool or value <= 0
        for value in minimum_group_rows.values()
    ):
        raise HCFeatureError("minimum group rows differ")
    global_rate = float(target.mean())
    normalized = frame.copy(deep=True)
    for column in set().union(*(set(keys) for _, keys, _, _ in _LEVELS)):
        normalized[column] = frame[column].map(_normalize)
    levels: dict[str, HierarchyLevelState] = {}
    for name, keys, parent_level, family in _LEVELS:
        grouped = (
            normalized.assign(_target=target.to_numpy(dtype="float64"))
            .groupby(list(keys), sort=True, dropna=False)["_target"]
            .agg(["size", "mean"])
            .reset_index()
        )
        lookup: dict[tuple[str, ...], HierarchyEntry] = {}
        for record in grouped.itertuples(index=False, name=None):
            group_key = _key(tuple(record[: len(keys)]))
            count = int(record[len(keys)])
            raw = float(record[len(keys) + 1])
            parent_key = _parent_key(name, group_key)
            if parent_level == "global":
                parent_rate = global_rate
            else:
                parent_entry = levels[parent_level].lookup.get(parent_key)
                parent_rate = (
                    global_rate
                    if parent_entry is None
                    or parent_entry.count < levels[parent_level].minimum_rows
                    else parent_entry.shrunk_rate
                )
            lookup[group_key] = HierarchyEntry(
                count=count,
                raw_rate=raw,
                parent_rate=parent_rate,
                shrunk_rate=shrink(
                    raw=raw,
                    parent=parent_rate,
                    count=count,
                    strength=_strength(profile, family),
                ),
                parent_level=parent_level,
                parent_key=parent_key,
            )
        levels[name] = HierarchyLevelState(
            name=name,
            keys=keys,
            parent_level=parent_level,
            strength=_strength(profile, family),
            minimum_rows=int(minimum_group_rows[family]),
            lookup=MappingProxyType(lookup),
        )
    return HierarchyState(
        cutoff_year=int(cutoff_year),
        profile_name=str(profile_name),
        source_seasons=tuple(sorted(int(value) for value in seasons.unique())),
        global_count=len(frame),
        global_rate=global_rate,
        levels=MappingProxyType(levels),
    )


def transform_hierarchy(rows: pd.DataFrame, state: HierarchyState) -> pd.DataFrame:
    frame = _validate_rows(rows, labeled=False)
    output = pd.DataFrame(index=frame.index)
    output["hc_global_count"] = float(state.global_count)
    output["hc_global_rate"] = state.global_rate
    output["hc_global_known"] = 1.0
    for name, _, parent_level, _ in _LEVELS:
        level = state.levels[name]
        counts: list[float] = []
        rates: list[float] = []
        known: list[float] = []
        for values in frame.loc[:, list(level.keys)].itertuples(index=False, name=None):
            entry = level.lookup.get(_key(tuple(values)))
            eligible = entry is not None and entry.count >= level.minimum_rows
            counts.append(float(0 if entry is None else entry.count))
            known.append(float(eligible))
            if eligible:
                rates.append(float(entry.shrunk_rate))
            else:
                parent_column = (
                    "hc_global_rate"
                    if parent_level == "global"
                    else f"hc_{parent_level}_rate"
                )
                rates.append(float(output.loc[frame.index[len(rates)], parent_column]))
        output[f"hc_{name}_count"] = counts
        output[f"hc_{name}_rate"] = rates
        output[f"hc_{name}_known"] = known
    return output.astype("float64")


def build_rolling_hierarchy(
    rows: pd.DataFrame,
    *,
    profile_name: str,
    profile: HCProfile,
    minimum_group_rows: Mapping[str, int] | None = None,
) -> Mapping[int, pd.DataFrame]:
    frame = _validate_rows(rows, labeled=True)
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    if seasons.isna().any():
        raise HCFeatureError("hierarchy seasons differ")
    minimums = (
        {"identity": 20, "context": 50, "interaction": 100}
        if minimum_group_rows is None
        else dict(minimum_group_rows)
    )
    output: dict[int, pd.DataFrame] = {}
    for year in sorted(int(value) for value in seasons.unique()):
        fit_rows = frame.loc[seasons.lt(year)]
        valid_rows = frame.loc[seasons.eq(year)].drop(columns="control_success")
        if fit_rows.empty:
            continue
        state = fit_hierarchy(
            fit_rows,
            cutoff_year=year - 1,
            profile_name=profile_name,
            profile=profile,
            minimum_group_rows=minimums,
        )
        output[year] = transform_hierarchy(valid_rows, state)
    return MappingProxyType(output)


def hierarchy_state_payload(state: HierarchyState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "tree_hierarchical_feature_state_v1",
        "cutoff_year": state.cutoff_year,
        "profile_name": state.profile_name,
        "source_seasons": list(state.source_seasons),
        "global_count": state.global_count,
        "global_rate": state.global_rate,
        "levels": {
            name: {
                "keys": list(level.keys),
                "parent_level": level.parent_level,
                "strength": level.strength,
                "minimum_rows": level.minimum_rows,
                "entries": [
                    {
                        "key": list(key),
                        "count": entry.count,
                        "raw_rate": entry.raw_rate,
                        "parent_rate": entry.parent_rate,
                        "shrunk_rate": entry.shrunk_rate,
                        "parent_level": entry.parent_level,
                        "parent_key": list(entry.parent_key),
                    }
                    for key, entry in sorted(level.lookup.items())
                ],
            }
            for name, level in state.levels.items()
        },
    }


def hierarchy_state_from_payload(payload: Mapping[str, object]) -> HierarchyState:
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("artifact_kind") != "tree_hierarchical_feature_state_v1"
        or type(payload.get("levels")) is not dict
        or set(payload["levels"]) != {name for name, _, _, _ in _LEVELS}
    ):
        raise HCFeatureError("hierarchy state payload differs")
    levels: dict[str, HierarchyLevelState] = {}
    for name, keys, parent_level, _ in _LEVELS:
        raw = payload["levels"][name]
        if (
            type(raw) is not dict
            or tuple(raw.get("keys", ())) != keys
            or raw.get("parent_level") != parent_level
            or type(raw.get("entries")) is not list
        ):
            raise HCFeatureError(f"hierarchy level payload differs: {name}")
        lookup: dict[tuple[str, ...], HierarchyEntry] = {}
        for item in raw["entries"]:
            if type(item) is not dict:
                raise HCFeatureError(f"hierarchy entry payload differs: {name}")
            key = tuple(str(value) for value in item["key"])
            if len(key) != len(keys) or key in lookup:
                raise HCFeatureError(f"hierarchy entry key differs: {name}")
            lookup[key] = HierarchyEntry(
                count=int(item["count"]),
                raw_rate=float(item["raw_rate"]),
                parent_rate=float(item["parent_rate"]),
                shrunk_rate=float(item["shrunk_rate"]),
                parent_level=str(item["parent_level"]),
                parent_key=tuple(str(value) for value in item["parent_key"]),
            )
        levels[name] = HierarchyLevelState(
            name=name,
            keys=keys,
            parent_level=parent_level,
            strength=float(raw["strength"]),
            minimum_rows=int(raw["minimum_rows"]),
            lookup=MappingProxyType(lookup),
        )
    return HierarchyState(
        cutoff_year=int(payload["cutoff_year"]),
        profile_name=str(payload["profile_name"]),
        source_seasons=tuple(int(value) for value in payload["source_seasons"]),
        global_count=int(payload["global_count"]),
        global_rate=float(payload["global_rate"]),
        levels=MappingProxyType(levels),
    )
