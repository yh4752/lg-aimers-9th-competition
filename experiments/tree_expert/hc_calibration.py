from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .hc_contracts import HCProfile, load_hc_contract


class HCCalibrationError(ValueError):
    """Raised when rolling hierarchical calibration evidence differs."""


_MISSING = "__MISSING__"
_KEYS = {
    "game_type": ("game_type",),
    "pitcher": ("pitcher_id",),
    "batter": ("batter_id",),
    "hand_matchup": ("pitcher_hand", "batter_hand"),
    "count": ("balls_before", "strikes_before"),
    "outs": ("outs_before",),
    "base_state": ("base_state",),
    "pitcher_game": ("pitcher_id", "game_type"),
    "batter_pitcher_hand": ("batter_id", "pitcher_hand"),
}
_PARENT = {
    "game_type": "global",
    "pitcher": "global",
    "batter": "global",
    "hand_matchup": "global",
    "count": "global",
    "outs": "global",
    "base_state": "global",
    "pitcher_game": "pitcher",
    "batter_pitcher_hand": "batter",
}
_FAMILY = {
    "game_type": "context",
    "pitcher": "identity",
    "batter": "identity",
    "hand_matchup": "context",
    "count": "context",
    "outs": "context",
    "base_state": "context",
    "pitcher_game": "interaction",
    "batter_pitcher_hand": "interaction",
}


@dataclass(frozen=True)
class CalibrationEntry:
    count: int
    raw_delta: float
    parent_cumulative: float
    cumulative_effect: float
    incremental_effect: float


@dataclass(frozen=True)
class CalibrationLevel:
    name: str
    keys: tuple[str, ...]
    parent_level: str
    minimum_rows: int
    strength: float
    lookup: Mapping[tuple[str, ...], CalibrationEntry]


@dataclass(frozen=True)
class CalibrationState:
    prediction_year: int
    profile_name: str
    source: Mapping[int, str]
    global_effect: float
    levels: Mapping[str, CalibrationLevel]
    effect_clip: float
    probability_clip: float


def _normalize(value: object) -> str:
    return _MISSING if pd.isna(value) else str(value)


def _key(values: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(_normalize(value) for value in values)


def _logit(value: float, clip: float) -> float:
    bounded = min(1.0 - clip, max(clip, float(value)))
    return math.log(bounded / (1.0 - bounded))


def _raw_delta(target: pd.Series, probability: pd.Series, clip: float) -> float:
    return _logit(float(target.mean()), clip) - _logit(float(probability.mean()), clip)


def _validate(
    frame: object,
    probability_column: str,
    label: str,
    *,
    require_unique_rows: bool = True,
    require_target: bool = True,
) -> pd.DataFrame:
    required = {
        "row_id",
        probability_column,
        *set().union(*(set(keys) for keys in _KEYS.values())),
    }
    if require_target:
        required.update({"oof_year", "target"})
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns) or frame.empty:
        raise HCCalibrationError(f"{label} schema differs")
    target = (
        pd.to_numeric(frame["target"], errors="coerce")
        if require_target
        else pd.Series(dtype="float64")
    )
    probability = pd.to_numeric(frame[probability_column], errors="coerce")
    if (
        frame["row_id"].isna().any()
        or (require_unique_rows and not frame["row_id"].is_unique)
        or (require_target and not target.isin([0, 1]).all())
        or probability.isna().any()
        or not probability.between(0, 1).all()
    ):
        raise HCCalibrationError(f"{label} values differ")
    return frame.copy(deep=True)


def _source_frame(
    prediction_year: int, source_rows: pd.DataFrame, c1_rows: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, str]]:
    source = _validate(source_rows, "p0", "E2 calibration source")
    c1 = _validate(c1_rows, "p1", "C1 calibration source")
    source_years = pd.to_numeric(source["oof_year"], errors="coerce").astype("int64")
    c1_years = pd.to_numeric(c1["oof_year"], errors="coerce").astype("int64")
    if prediction_year == 2022:
        chosen = source.loc[source_years.eq(2021)].copy(deep=True)
        chosen["_probability"] = chosen["p0"]
        lineage = {2021: "e2"}
    elif prediction_year in {2023, 2024}:
        allowed = tuple(range(2022, prediction_year))
        chosen = c1.loc[c1_years.isin(allowed)].copy(deep=True)
        chosen["_probability"] = chosen["p1"]
        lineage = {year: "c1" for year in allowed}
    elif prediction_year == 2025:
        earliest = source.loc[source_years.eq(2021)].copy(deep=True)
        earliest["_probability"] = earliest["p0"]
        later = c1.loc[c1_years.isin((2022, 2023, 2024))].copy(deep=True)
        later["_probability"] = later["p1"]
        chosen = pd.concat([earliest, later], ignore_index=True)
        lineage = {2021: "e2", 2022: "c1", 2023: "c1", 2024: "c1"}
    else:
        raise HCCalibrationError("calibration prediction year differs")
    if chosen.empty or set(chosen["oof_year"].astype(int)) != set(lineage):
        raise HCCalibrationError("calibration source window differs")
    if chosen["oof_year"].ge(prediction_year).any():
        raise HCCalibrationError("calibration source window differs")
    return chosen, lineage


def _strength(profile: HCProfile, family: str) -> float:
    return {
        "identity": profile.identity_k,
        "context": profile.context_k,
        "interaction": profile.interaction_k,
    }[family]


def fit_rolling_calibrator(
    prediction_year: int,
    source_rows: pd.DataFrame,
    c1_rows: pd.DataFrame,
    *,
    profile_name: str,
    profile: HCProfile,
    minimum_group_rows: Mapping[str, int],
) -> CalibrationState:
    if set(minimum_group_rows) != {"identity", "context", "interaction"}:
        raise HCCalibrationError("calibration minimum rows differ")
    frame, lineage = _source_frame(prediction_year, source_rows, c1_rows)
    contract = load_hc_contract()
    clip = float(contract.calibration["probability_clip"])
    effect_clip = float(contract.calibration["effect_clip"])
    target = pd.to_numeric(frame["target"], errors="raise").astype("float64")
    probability = pd.to_numeric(frame["_probability"], errors="raise").astype("float64")
    global_effect = _raw_delta(target, probability, clip)
    normalized = frame.copy(deep=True)
    for column in set().union(*(set(keys) for keys in _KEYS.values())):
        normalized[column] = frame[column].map(_normalize)
    normalized["_target"] = target.to_numpy()
    normalized["_probability"] = probability.to_numpy()
    levels: dict[str, CalibrationLevel] = {}
    for name, keys in _KEYS.items():
        parent_level = _PARENT[name]
        family = _FAMILY[name]
        strength = _strength(profile, family)
        minimum = int(minimum_group_rows[family])
        grouped = (
            normalized.groupby(list(keys), sort=True, dropna=False)
            .agg(count=("_target", "size"), target_mean=("_target", "mean"), probability_mean=("_probability", "mean"))
            .reset_index()
        )
        lookup: dict[tuple[str, ...], CalibrationEntry] = {}
        for record in grouped.itertuples(index=False, name=None):
            group_key = _key(tuple(record[: len(keys)]))
            count = int(record[len(keys)])
            raw = _logit(float(record[len(keys) + 1]), clip) - _logit(
                float(record[len(keys) + 2]), clip
            )
            if parent_level == "global":
                parent = global_effect
            else:
                parent_entry = levels[parent_level].lookup.get((group_key[0],))
                parent = (
                    global_effect
                    if parent_entry is None
                    or parent_entry.count < levels[parent_level].minimum_rows
                    else parent_entry.cumulative_effect
                )
            cumulative = parent + count / (count + strength) * (raw - parent)
            lookup[group_key] = CalibrationEntry(
                count=count,
                raw_delta=raw,
                parent_cumulative=parent,
                cumulative_effect=cumulative,
                incremental_effect=cumulative - parent,
            )
        levels[name] = CalibrationLevel(
            name=name,
            keys=keys,
            parent_level=parent_level,
            minimum_rows=minimum,
            strength=strength,
            lookup=MappingProxyType(lookup),
        )
    return CalibrationState(
        prediction_year=prediction_year,
        profile_name=profile_name,
        source=MappingProxyType(lineage),
        global_effect=global_effect,
        levels=MappingProxyType(levels),
        effect_clip=effect_clip,
        probability_clip=clip,
    )


def apply_calibrator(
    rows: pd.DataFrame, state: CalibrationState, *, alpha: float
) -> pd.DataFrame:
    frame = _validate(
        rows,
        "p1",
        "calibration inference rows",
        require_unique_rows=False,
        require_target=False,
    )
    if alpha not in load_hc_contract().calibration_alphas:
        raise HCCalibrationError("calibration alpha differs")
    effects: list[float] = []
    for record in frame.to_dict(orient="records"):
        effect = state.global_effect
        for name, level in state.levels.items():
            entry = level.lookup.get(_key(tuple(record[column] for column in level.keys)))
            if entry is not None and entry.count >= level.minimum_rows:
                effect += entry.incremental_effect
        effects.append(float(np.clip(alpha * effect, -state.effect_clip, state.effect_clip)))
    p1 = frame["p1"].to_numpy(dtype="float64")
    bounded = np.clip(p1, state.probability_clip, 1.0 - state.probability_clip)
    logits = np.log(bounded / (1.0 - bounded)) + np.asarray(effects)
    p2 = 1.0 / (1.0 + np.exp(-logits))
    output = frame.copy(deep=True)
    output["hc_calibration_effect"] = effects
    output["p2"] = np.clip(p2, state.probability_clip, 1.0 - state.probability_clip)
    return output


def select_calibration_alpha(predictions: Mapping[float, pd.DataFrame]) -> float:
    registered = load_hc_contract().calibration_alphas
    if set(predictions) != set(registered):
        raise HCCalibrationError("calibration alpha evidence differs")
    scores: list[tuple[float, float]] = []
    for alpha in registered:
        frame = predictions[alpha]
        if type(frame) is not pd.DataFrame or not {"oof_year", "target", "p2"}.issubset(frame):
            raise HCCalibrationError("calibration alpha prediction schema differs")
        structure = frame.loc[frame["oof_year"].isin((2022, 2023))]
        if structure.empty or set(structure["oof_year"].astype(int)) != {2022, 2023}:
            raise HCCalibrationError("calibration structure folds differ")
        target = structure["target"].to_numpy(dtype="float64")
        probability = structure["p2"].to_numpy(dtype="float64")
        scores.append((float(np.mean(np.square(probability - target))), float(alpha)))
    best = min(score for score, _ in scores)
    return min(alpha for score, alpha in scores if score <= best + 1e-12)


def calibration_state_payload(state: CalibrationState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "tree_hierarchical_calibration_state_v1",
        "prediction_year": state.prediction_year,
        "profile_name": state.profile_name,
        "source": {str(year): kind for year, kind in state.source.items()},
        "global_effect": state.global_effect,
        "effect_clip": state.effect_clip,
        "probability_clip": state.probability_clip,
        "levels": {
            name: {
                "keys": list(level.keys),
                "parent_level": level.parent_level,
                "minimum_rows": level.minimum_rows,
                "strength": level.strength,
                "entries": [
                    {
                        "key": list(key),
                        "count": entry.count,
                        "raw_delta": entry.raw_delta,
                        "parent_cumulative": entry.parent_cumulative,
                        "cumulative_effect": entry.cumulative_effect,
                        "incremental_effect": entry.incremental_effect,
                    }
                    for key, entry in sorted(level.lookup.items())
                ],
            }
            for name, level in state.levels.items()
        },
    }


def calibration_state_from_payload(payload: Mapping[str, object]) -> CalibrationState:
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("artifact_kind") != "tree_hierarchical_calibration_state_v1"
        or type(payload.get("levels")) is not dict
        or set(payload["levels"]) != set(_KEYS)
        or type(payload.get("source")) is not dict
    ):
        raise HCCalibrationError("calibration state payload differs")
    levels: dict[str, CalibrationLevel] = {}
    for name, keys in _KEYS.items():
        raw = payload["levels"][name]
        if (
            type(raw) is not dict
            or tuple(raw.get("keys", ())) != keys
            or raw.get("parent_level") != _PARENT[name]
            or type(raw.get("entries")) is not list
        ):
            raise HCCalibrationError(f"calibration level payload differs: {name}")
        lookup: dict[tuple[str, ...], CalibrationEntry] = {}
        for item in raw["entries"]:
            key = tuple(str(value) for value in item["key"])
            if len(key) != len(keys) or key in lookup:
                raise HCCalibrationError(f"calibration entry key differs: {name}")
            lookup[key] = CalibrationEntry(
                count=int(item["count"]),
                raw_delta=float(item["raw_delta"]),
                parent_cumulative=float(item["parent_cumulative"]),
                cumulative_effect=float(item["cumulative_effect"]),
                incremental_effect=float(item["incremental_effect"]),
            )
        levels[name] = CalibrationLevel(
            name=name,
            keys=keys,
            parent_level=_PARENT[name],
            minimum_rows=int(raw["minimum_rows"]),
            strength=float(raw["strength"]),
            lookup=MappingProxyType(lookup),
        )
    return CalibrationState(
        prediction_year=int(payload["prediction_year"]),
        profile_name=str(payload["profile_name"]),
        source=MappingProxyType(
            {int(year): str(kind) for year, kind in payload["source"].items()}
        ),
        global_effect=float(payload["global_effect"]),
        levels=MappingProxyType(levels),
        effect_clip=float(payload["effect_clip"]),
        probability_clip=float(payload["probability_clip"]),
    )
