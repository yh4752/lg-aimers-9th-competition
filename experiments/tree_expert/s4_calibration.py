from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .hc_calibration import (
    CalibrationState,
    calibration_state_from_payload,
    calibration_state_payload,
    fit_rolling_calibrator,
)
from .hc_contracts import load_hc_contract
from .s4_contracts import load_s4_contract


class S4CalibrationError(ValueError):
    pass


_PROFILE_LEVELS = {
    "global_game": ("game_type",),
    "pitcher": ("game_type", "pitcher", "pitcher_game"),
    "batter": ("game_type", "batter", "batter_pitcher_hand"),
    "matchup": ("game_type", "hand_matchup", "count", "outs", "base_state"),
    "rf_matchup": (
        "game_type", "pitcher", "batter", "hand_matchup", "count", "outs",
        "base_state", "pitcher_game", "batter_pitcher_hand",
    ),
}
_MISSING = "__MISSING__"


@dataclass(frozen=True)
class S4CalibrationState:
    prediction_year: int
    profile_name: str
    source: Mapping[int, str]
    active_levels: tuple[str, ...]
    base_state: CalibrationState


def _normalize(value: object) -> str:
    return _MISSING if pd.isna(value) else str(value)


def _prepare_source(frame: object, source: str, destination: str, label: str) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or source not in frame or frame.empty:
        raise S4CalibrationError(f"{label} schema differs")
    result = frame.copy(deep=True)
    if destination in result and destination != source:
        raise S4CalibrationError(f"{label} probability columns differ")
    return result.rename(columns={source: destination})


def fit_s4_calibrator(
    prediction_year: int,
    anchor_rows: pd.DataFrame,
    chain_rows: pd.DataFrame,
    *,
    profile_name: str,
    minimum_group_rows: Mapping[str, int] | None = None,
) -> S4CalibrationState:
    if profile_name not in _PROFILE_LEVELS or profile_name not in load_s4_contract().calibration_profiles:
        raise S4CalibrationError("calibration profile differs")
    hc = load_hc_contract()
    minimum = hc.minimum_group_rows if minimum_group_rows is None else minimum_group_rows
    anchor = _prepare_source(anchor_rows, "p_anchor", "p0", "anchor calibration source")
    chain = _prepare_source(chain_rows, "p_chain", "p1", "chain calibration source")
    try:
        base = fit_rolling_calibrator(
            prediction_year,
            anchor,
            chain,
            profile_name="hc_strong",
            profile=hc.profiles["hc_strong"],
            minimum_group_rows=minimum,
        )
    except ValueError as error:
        raise S4CalibrationError(str(error)) from error
    source = {
        year: "s4_anchor" if kind == "e2" else "s4_chain"
        for year, kind in base.source.items()
    }
    return S4CalibrationState(
        prediction_year=int(prediction_year),
        profile_name=profile_name,
        source=MappingProxyType(source),
        active_levels=_PROFILE_LEVELS[profile_name],
        base_state=base,
    )


def _validate_inference(frame: object, state: S4CalibrationState) -> pd.DataFrame:
    keys = {
        column
        for name in state.active_levels
        for column in state.base_state.levels[name].keys
    }
    required = {"row_id", "p_chain", *keys}
    if type(frame) is not pd.DataFrame or frame.empty or not required.issubset(frame.columns):
        raise S4CalibrationError("calibration inference schema differs")
    probability = pd.to_numeric(frame["p_chain"], errors="coerce")
    if frame["row_id"].isna().any() or probability.isna().any() or not probability.between(0, 1).all():
        raise S4CalibrationError("calibration inference values differ")
    return frame.copy(deep=True)


def apply_s4_calibrator(
    rows: pd.DataFrame, state: S4CalibrationState, *, beta: float
) -> pd.DataFrame:
    registered = load_s4_contract().calibration_betas
    if type(beta) not in {int, float} or float(beta) not in registered:
        raise S4CalibrationError("calibration beta differs")
    frame = _validate_inference(rows, state)
    base = state.base_state
    effects: list[float] = []
    for record in frame.to_dict(orient="records"):
        effect = base.global_effect
        for name in state.active_levels:
            level = base.levels[name]
            key = tuple(_normalize(record[column]) for column in level.keys)
            entry = level.lookup.get(key)
            if entry is not None and entry.count >= level.minimum_rows:
                effect += entry.incremental_effect
        effects.append(float(np.clip(float(beta) * effect, -base.effect_clip, base.effect_clip)))
    probability = frame["p_chain"].to_numpy(dtype="float64")
    bounded = np.clip(probability, base.probability_clip, 1.0 - base.probability_clip)
    logits = np.log(bounded / (1.0 - bounded)) + np.asarray(effects)
    output = frame.copy(deep=True)
    output["s4_calibration_effect"] = effects
    output["p_final"] = np.clip(
        1.0 / (1.0 + np.exp(-logits)),
        base.probability_clip,
        1.0 - base.probability_clip,
    )
    return output


def select_s4_beta(predictions: Mapping[float, pd.DataFrame]) -> float:
    registered = load_s4_contract().calibration_betas
    if set(predictions) != set(registered):
        raise S4CalibrationError("calibration beta evidence differs")
    scored: list[tuple[float, float]] = []
    for beta in registered:
        frame = predictions[beta]
        if type(frame) is not pd.DataFrame or not {"oof_year", "target", "p_final"}.issubset(frame.columns):
            raise S4CalibrationError("calibration beta prediction schema differs")
        structure = frame.loc[frame["oof_year"].isin((2022, 2023))]
        if structure.empty or set(structure["oof_year"].astype(int)) != {2022, 2023}:
            raise S4CalibrationError("calibration structure folds differ")
        target = structure["target"].to_numpy(dtype="float64")
        probability = structure["p_final"].to_numpy(dtype="float64")
        score = float(np.mean(np.square(probability - target)))
        if not math.isfinite(score):
            raise S4CalibrationError("calibration score differs")
        scored.append((score, float(beta)))
    best = min(score for score, _ in scored)
    return min(beta for score, beta in scored if score <= best + 1e-12)


def s4_calibration_state_payload(state: S4CalibrationState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "tree_s4_calibration_state_v1",
        "prediction_year": state.prediction_year,
        "profile_name": state.profile_name,
        "source": {str(year): kind for year, kind in state.source.items()},
        "active_levels": list(state.active_levels),
        "base_state": calibration_state_payload(state.base_state),
    }


def s4_calibration_state_from_payload(payload: Mapping[str, object]) -> S4CalibrationState:
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("artifact_kind") != "tree_s4_calibration_state_v1"
        or payload.get("profile_name") not in _PROFILE_LEVELS
        or type(payload.get("source")) is not dict
        or type(payload.get("base_state")) is not dict
    ):
        raise S4CalibrationError("calibration state payload differs")
    profile = str(payload["profile_name"])
    if tuple(payload.get("active_levels", ())) != _PROFILE_LEVELS[profile]:
        raise S4CalibrationError("calibration state levels differ")
    try:
        base = calibration_state_from_payload(payload["base_state"])
    except ValueError as error:
        raise S4CalibrationError(str(error)) from error
    prediction_year = int(payload["prediction_year"])
    if base.prediction_year != prediction_year:
        raise S4CalibrationError("calibration state year differs")
    return S4CalibrationState(
        prediction_year=prediction_year,
        profile_name=profile,
        source=MappingProxyType({int(year): str(kind) for year, kind in payload["source"].items()}),
        active_levels=_PROFILE_LEVELS[profile],
        base_state=base,
    )
