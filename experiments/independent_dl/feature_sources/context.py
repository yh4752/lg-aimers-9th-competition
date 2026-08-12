"""Frozen contextual target effects.

Source commit: ``9454d68b93971627e3d3f613ce30be690cb5dce2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COLUMN = "control_success"


@dataclass(frozen=True)
class ContextSpec:
    """One smoothed target-effect lookup keyed by game context columns."""

    name: str
    keys: tuple[str, ...]
    strength: float


CONTEXT_SPECS = (
    ContextSpec("count", ("balls_before", "strikes_before"), 500.0),
    ContextSpec("count_out", ("balls_before", "strikes_before", "outs_before"), 1000.0),
    ContextSpec("hand_matchup", ("pitcher_hand", "batter_hand"), 5000.0),
    ContextSpec("inning", ("inning",), 1500.0),
    ContextSpec("batter_team", ("batter_team_id",), 3000.0),
    ContextSpec("game_type", ("game_type",), 1000.0),
)


@dataclass(frozen=True)
class FrozenContextSnapshot:
    """Context lookup tables whose inputs end at ``cutoff_year``."""

    cutoff_year: int
    prior_rate: float
    lookups: dict[str, pd.DataFrame]


def _logit(values: np.ndarray | pd.Series | float) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    array = np.clip(array, 1e-5, 1.0 - 1e-5)
    return np.log(array / (1.0 - array))


def build_frozen_context_snapshot(
    labeled_history: pd.DataFrame,
    cutoff_year: int,
    specs: tuple[ContextSpec, ...] = CONTEXT_SPECS,
) -> FrozenContextSnapshot:
    """Build cutoff-season context effects after explicitly slicing history."""

    history = labeled_history.loc[labeled_history["season"].le(cutoff_year)].copy()
    source = history.loc[history["season"].eq(cutoff_year)]
    if source.empty:
        raise ValueError(f"No labeled rows for cutoff year {cutoff_year}")
    prior_rate = float(source[TARGET_COLUMN].mean())
    prior_logit = float(_logit(prior_rate))
    lookups: dict[str, pd.DataFrame] = {}
    for spec in specs:
        aggregate = (
            source.groupby(list(spec.keys), sort=False, observed=True)[TARGET_COLUMN]
            .agg(["sum", "count"])
            .reset_index()
        )
        probability = (aggregate["sum"] + spec.strength * prior_rate) / (
            aggregate["count"] + spec.strength
        )
        aggregate[f"fc_{spec.name}_logit_effect"] = (
            _logit(probability) - prior_logit
        ).astype("float32")
        aggregate[f"fc_{spec.name}_log1p_n"] = np.log1p(aggregate["count"]).astype(
            "float32"
        )
        lookups[spec.name] = aggregate[
            [
                *spec.keys,
                f"fc_{spec.name}_logit_effect",
                f"fc_{spec.name}_log1p_n",
            ]
        ].copy()
    return FrozenContextSnapshot(cutoff_year, prior_rate, lookups)


def attach_frozen_context_features(
    frame: pd.DataFrame,
    snapshot: FrozenContextSnapshot,
    specs: tuple[ContextSpec, ...] = CONTEXT_SPECS,
) -> pd.DataFrame:
    """Join frozen context effects to destination rows without aggregation."""

    output = frame.copy()
    original_index = output.index
    output["fc_previous_season_global_rate"] = np.float32(snapshot.prior_rate)
    for spec in specs:
        output = output.merge(
            snapshot.lookups[spec.name],
            on=list(spec.keys),
            how="left",
            validate="many_to_one",
            sort=False,
        )
        output.index = original_index
        output[f"fc_{spec.name}_logit_effect"] = output[
            f"fc_{spec.name}_logit_effect"
        ].fillna(0.0).astype("float32")
        output[f"fc_{spec.name}_log1p_n"] = output[
            f"fc_{spec.name}_log1p_n"
        ].fillna(0.0).astype("float32")
    return output


def frozen_context_baseline(features: pd.DataFrame, recipe: str) -> np.ndarray:
    """Apply the sealed frozen-context baseline recipe to a feature matrix."""

    prior = features["fc_previous_season_global_rate"].to_numpy(dtype="float64")
    pitcher = features["hier_pitcher_success_h1000_k50"].to_numpy(dtype="float64")
    batter = features["hier_batter_success_h1000_k200"].to_numpy(dtype="float64")
    score = _logit(pitcher)
    batter_effect = _logit(batter) - _logit(prior)
    if recipe == "stable":
        weights = {
            "batter": 0.50,
            "count": 0.00,
            "count_out": 0.70,
            "hand_matchup": 0.75,
            "inning": 0.30,
            "batter_team": 0.00,
            "game_type": 0.00,
        }
    elif recipe == "recent":
        weights = {
            "batter": 0.55,
            "count": 0.25,
            "count_out": 0.55,
            "hand_matchup": 0.80,
            "inning": 0.30,
            "batter_team": 0.25,
            "game_type": 0.20,
        }
    else:
        raise ValueError(f"Unknown frozen-context recipe: {recipe}")
    score = score + weights["batter"] * batter_effect
    for name in ("count", "count_out", "hand_matchup", "inning", "batter_team", "game_type"):
        score = score + weights[name] * features[
            f"fc_{name}_logit_effect"
        ].to_numpy(dtype="float64")
    return np.clip(1.0 / (1.0 + np.exp(-score)), 1e-5, 1.0 - 1e-5)
