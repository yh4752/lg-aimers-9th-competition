from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


class CalibrationError(ValueError):
    pass


EffectKey = tuple[str, str]


@dataclass(frozen=True)
class TemporalCalibrator:
    hierarchy: str
    ridge: float
    source_years: tuple[int, ...]
    global_effect: float
    game_effects: Mapping[str, float]
    hand_effects: Mapping[EffectKey, float]
    pitcher_effects: Mapping[EffectKey, float]
    batter_effects: Mapping[EffectKey, float]

    def __reduce__(self):
        return (
            _restore_calibrator,
            (
                self.hierarchy, self.ridge, self.source_years, self.global_effect,
                dict(self.game_effects), dict(self.hand_effects),
                dict(self.pitcher_effects), dict(self.batter_effects),
            ),
        )

    def effect_for(
        self, *, game_type: object, hand_matchup: object, pitcher_id: object, batter_id: object,
    ) -> float:
        game = str(game_type)
        hand = str(hand_matchup)
        pitcher = str(pitcher_id)
        batter = str(batter_id)
        effect = float(self.game_effects.get(game, self.global_effect))
        effect = float(self.hand_effects.get((game, hand), effect))
        effect = float(self.pitcher_effects.get((game, pitcher), effect))
        effect = float(self.batter_effects.get((game, batter), effect))
        return effect


def _restore_calibrator(
    hierarchy: str,
    ridge: float,
    source_years: tuple[int, ...],
    global_effect: float,
    game_effects: Mapping[str, float],
    hand_effects: Mapping[EffectKey, float],
    pitcher_effects: Mapping[EffectKey, float],
    batter_effects: Mapping[EffectKey, float],
) -> TemporalCalibrator:
    return TemporalCalibrator(
        hierarchy, ridge, source_years, global_effect,
        MappingProxyType(dict(game_effects)), MappingProxyType(dict(hand_effects)),
        MappingProxyType(dict(pitcher_effects)), MappingProxyType(dict(batter_effects)),
    )


_REQUIRED = {
    "target", "probability", "oof_year", "game_type", "hand_matchup", "pitcher_id", "batter_id",
}


def _logit(probability: np.ndarray) -> np.ndarray:
    return np.log(probability) - np.log1p(-probability)


def _expit(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    output = np.empty_like(value, dtype="float64")
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or not _REQUIRED.issubset(frame.columns):
        raise CalibrationError("calibration columns differ")
    output = frame.copy(deep=True)
    output["target"] = pd.to_numeric(output["target"], errors="coerce")
    output["probability"] = pd.to_numeric(output["probability"], errors="coerce")
    output["oof_year"] = pd.to_numeric(output["oof_year"], errors="coerce")
    if (
        output[["target", "probability", "oof_year"]].isna().any().any()
        or not output["target"].isin((0, 1)).all()
        or not output["probability"].between(0, 1).all()
    ):
        raise CalibrationError("calibration values differ")
    for column in ("game_type", "hand_matchup", "pitcher_id", "batter_id"):
        output[column] = output[column].astype(str)
    return output


def _ridge_effect(group: pd.DataFrame, *, ridge: float, parent: np.ndarray | float) -> float:
    probability = group["probability"].to_numpy(dtype="float64")
    numerator = float((group["target"].to_numpy(dtype="float64") - probability).sum())
    information = float(np.sum(probability * (1.0 - probability)))
    parent_effect = float(np.mean(parent)) if np.ndim(parent) else float(parent)
    return (numerator + ridge * parent_effect) / (information + ridge)


def _group_effects(
    source: pd.DataFrame, columns: list[str], *, ridge: float, parent_values: np.ndarray,
) -> dict[object, float]:
    work = source.copy(deep=False)
    work = work.assign(_parent=parent_values)
    output: dict[object, float] = {}
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, group in work.groupby(grouper, sort=True, observed=True):
        normalized_key = key if len(columns) == 1 else tuple(str(value) for value in key)
        output[normalized_key] = _ridge_effect(
            group, ridge=ridge, parent=group["_parent"].to_numpy(dtype="float64")
        )
    return output


def fit_temporal_calibrator(
    frame: pd.DataFrame, *, validation_year: int, hierarchy: str, ridge: int | float,
) -> TemporalCalibrator:
    if hierarchy not in {"global_game", "full"}:
        raise CalibrationError("calibration hierarchy differs")
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value <= 0:
        raise CalibrationError("calibration ridge differs")
    validated = _validate(frame)
    source = validated.loc[validated["oof_year"] < int(validation_year)].copy()
    years = tuple(sorted(int(value) for value in source["oof_year"].unique()))
    if source.empty:
        return TemporalCalibrator(
            hierarchy, ridge_value, (), 0.0,
            MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), MappingProxyType({}),
        )
    global_effect = _ridge_effect(source, ridge=ridge_value, parent=0.0)
    global_parent = np.full(len(source), global_effect, dtype="float64")
    game = _group_effects(source, ["game_type"], ridge=ridge_value, parent_values=global_parent)
    game_parent = source["game_type"].map(game).fillna(global_effect).to_numpy(dtype="float64")
    if hierarchy == "global_game":
        return TemporalCalibrator(
            hierarchy, ridge_value, years, global_effect,
            MappingProxyType(game), MappingProxyType({}), MappingProxyType({}), MappingProxyType({}),
        )
    hand = _group_effects(source, ["game_type", "hand_matchup"], ridge=ridge_value, parent_values=game_parent)
    hand_parent = np.array([
        hand.get((game_type, matchup), game.get(game_type, global_effect))
        for game_type, matchup in zip(source["game_type"], source["hand_matchup"])
    ], dtype="float64")
    pitcher = _group_effects(source, ["game_type", "pitcher_id"], ridge=ridge_value, parent_values=hand_parent)
    pitcher_parent = np.array([
        pitcher.get((game_type, pitcher_id), hand_parent[index])
        for index, (game_type, pitcher_id) in enumerate(zip(source["game_type"], source["pitcher_id"]))
    ], dtype="float64")
    batter = _group_effects(source, ["game_type", "batter_id"], ridge=ridge_value, parent_values=pitcher_parent)
    return TemporalCalibrator(
        hierarchy, ridge_value, years, global_effect,
        MappingProxyType(game), MappingProxyType(hand), MappingProxyType(pitcher), MappingProxyType(batter),
    )


def apply_calibration(calibrator: TemporalCalibrator, frame: pd.DataFrame, *, beta: float) -> np.ndarray:
    if type(frame) is not pd.DataFrame or not {
        "probability", "game_type", "hand_matchup", "pitcher_id", "batter_id"
    }.issubset(frame.columns):
        raise CalibrationError("calibration application columns differ")
    probability = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise CalibrationError("calibration application probabilities differ")
    effects = np.array([
        calibrator.effect_for(
            game_type=row.game_type,
            hand_matchup=row.hand_matchup,
            pitcher_id=row.pitcher_id,
            batter_id=row.batter_id,
        )
        for row in frame.itertuples(index=False)
    ], dtype="float64")
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return _expit(_logit(clipped) + float(beta) * effects)
