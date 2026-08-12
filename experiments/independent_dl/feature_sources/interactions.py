"""Frozen hierarchical interaction effects.

Source commit: ``9454d68b93971627e3d3f613ce30be690cb5dce2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


TARGET_COLUMN = "control_success"


@dataclass(frozen=True)
class FrozenInteractionSpec:
    """One parent-to-child target-effect lookup with a bounded lookback."""

    name: str
    parent: tuple[str, ...]
    child: tuple[str, ...]
    parent_strength: float
    child_strength: float
    lookback_years: int = 1


INTERACTION_SPECS = (
    FrozenInteractionSpec("pitcher_team_count", ("pitcher_team_id",), ("pitcher_team_id", "balls_before", "strikes_before"), 3000.0, 500.0),
    FrozenInteractionSpec("batter_team_count", ("batter_team_id",), ("batter_team_id", "balls_before", "strikes_before"), 3000.0, 200.0),
    FrozenInteractionSpec("pitcher_game_batter_hand", ("pitcher_id", "game_type"), ("pitcher_id", "game_type", "batter_hand"), 500.0, 200.0),
    FrozenInteractionSpec("batter_game_pitcher_hand", ("batter_id", "game_type"), ("batter_id", "game_type", "pitcher_hand"), 500.0, 100.0),
    FrozenInteractionSpec("pitcher_game_count", ("pitcher_id", "game_type"), ("pitcher_id", "game_type", "balls_before", "strikes_before"), 500.0, 400.0),
    FrozenInteractionSpec("batter_team_game_count", ("batter_team_id", "game_type"), ("batter_team_id", "game_type", "balls_before", "strikes_before"), 1500.0, 200.0),
    FrozenInteractionSpec("team_match", (), ("pitcher_team_id", "batter_team_id"), 0.0, 2000.0),
    FrozenInteractionSpec("hand_base", ("pitcher_hand", "batter_hand"), ("pitcher_hand", "batter_hand", "base_state"), 5000.0, 4000.0),
    FrozenInteractionSpec("hand_count", ("pitcher_hand", "batter_hand"), ("pitcher_hand", "batter_hand", "balls_before", "strikes_before"), 5000.0, 300.0),
    FrozenInteractionSpec("game_hand_count", ("game_type", "pitcher_hand", "batter_hand"), ("game_type", "pitcher_hand", "batter_hand", "balls_before", "strikes_before"), 2000.0, 300.0),
    FrozenInteractionSpec("count_base", ("balls_before", "strikes_before"), ("balls_before", "strikes_before", "base_state"), 3000.0, 500.0),
    FrozenInteractionSpec("count_inning", ("balls_before", "strikes_before"), ("balls_before", "strikes_before", "inning_bucket"), 3000.0, 500.0),
    FrozenInteractionSpec("count_out", ("balls_before", "strikes_before"), ("balls_before", "strikes_before", "outs_before"), 3000.0, 500.0),
    FrozenInteractionSpec("team_match_game", ("game_type",), ("game_type", "pitcher_team_id", "batter_team_id"), 10000.0, 8000.0),
    FrozenInteractionSpec("batter_team_game_pitcher_hand", ("batter_team_id", "game_type"), ("batter_team_id", "game_type", "pitcher_hand"), 2000.0, 3000.0),
    FrozenInteractionSpec("pitcher_team_game_batter_hand", ("pitcher_team_id", "game_type"), ("pitcher_team_id", "game_type", "batter_hand"), 2000.0, 3000.0),
    FrozenInteractionSpec("stable_game_hand_base", ("game_type", "pitcher_hand", "batter_hand"), ("game_type", "pitcher_hand", "batter_hand", "base_state"), 3000.0, 24000.0, lookback_years=2),
    FrozenInteractionSpec("stable_game_hand_count", ("game_type", "pitcher_hand", "batter_hand"), ("game_type", "pitcher_hand", "batter_hand", "balls_before", "strikes_before"), 3000.0, 100.0, lookback_years=2),
)


@dataclass(frozen=True)
class FrozenInteractionSnapshot:
    """Frozen interaction lookup tables built through one cutoff year."""

    cutoff_year: int
    prior_rate: float
    lookups: dict[str, pd.DataFrame]


def _logit(values: np.ndarray | pd.Series | float) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    array = np.clip(array, 1e-5, 1.0 - 1e-5)
    return np.log(array / (1.0 - array))


def _attach_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "inning_bucket" not in output:
        inning = pd.to_numeric(output["inning"], errors="coerce").fillna(0)
        output["inning_bucket"] = np.select(
            [inning <= 3, inning <= 6, inning <= 9],
            ["early", "middle", "late"],
            default="extra",
        )
    return output


def _build_lookup(
    source: pd.DataFrame, spec: FrozenInteractionSpec, global_rate: float
) -> pd.DataFrame:
    child = (
        source.groupby(list(spec.child), sort=False, observed=True)[TARGET_COLUMN]
        .agg(["sum", "count"])
        .reset_index()
    )
    if spec.parent:
        parent = (
            source.groupby(list(spec.parent), sort=False, observed=True)[TARGET_COLUMN]
            .agg(["sum", "count"])
            .reset_index()
        )
        parent["parent_probability"] = (
            parent["sum"] + spec.parent_strength * global_rate
        ) / (parent["count"] + spec.parent_strength)
        child = child.merge(
            parent[[*spec.parent, "parent_probability"]],
            on=list(spec.parent),
            how="left",
            validate="many_to_one",
        )
    else:
        child["parent_probability"] = global_rate
    probability = (
        child["sum"] + spec.child_strength * child["parent_probability"]
    ) / (child["count"] + spec.child_strength)
    effect_column = f"fi_{spec.name}_logit_effect"
    count_column = f"fi_{spec.name}_log1p_n"
    child[effect_column] = (
        _logit(probability) - _logit(child["parent_probability"])
    ).astype("float32")
    child[count_column] = np.log1p(child["count"]).astype("float32")
    return child[[*spec.child, effect_column, count_column]].copy()


def build_frozen_interaction_snapshot(
    labeled_history: pd.DataFrame,
    cutoff_year: int,
    specs: tuple[FrozenInteractionSpec, ...] = INTERACTION_SPECS,
) -> FrozenInteractionSnapshot:
    """Build interaction lookups from rows no later than ``cutoff_year``."""

    history = _attach_derived_columns(
        labeled_history.loc[labeled_history["season"].le(cutoff_year)].copy()
    )
    cutoff_source = history.loc[history["season"].eq(cutoff_year)]
    if cutoff_source.empty:
        raise ValueError(f"No labeled rows for cutoff year {cutoff_year}")
    lookups: dict[str, pd.DataFrame] = {}
    for spec in specs:
        first_year = cutoff_year - int(spec.lookback_years) + 1
        source = history.loc[history["season"].between(first_year, cutoff_year)]
        if source.empty:
            raise ValueError(
                f"No labeled rows for {spec.name} through cutoff year {cutoff_year}"
            )
        lookups[spec.name] = _build_lookup(
            source, spec, float(source[TARGET_COLUMN].mean())
        )
    return FrozenInteractionSnapshot(
        cutoff_year=cutoff_year,
        prior_rate=float(cutoff_source[TARGET_COLUMN].mean()),
        lookups=lookups,
    )


def attach_frozen_interaction_features(
    frame: pd.DataFrame,
    snapshot: FrozenInteractionSnapshot,
    specs: tuple[FrozenInteractionSpec, ...] = INTERACTION_SPECS,
) -> pd.DataFrame:
    """Attach frozen interaction effects independently to each row."""

    output = _attach_derived_columns(frame)
    original_index = output.index
    for spec in specs:
        output = output.merge(
            snapshot.lookups[spec.name],
            on=list(spec.child),
            how="left",
            validate="many_to_one",
            sort=False,
        )
        output.index = original_index
        output[f"fi_{spec.name}_logit_effect"] = output[
            f"fi_{spec.name}_logit_effect"
        ].fillna(0.0).astype("float32")
        output[f"fi_{spec.name}_log1p_n"] = output[
            f"fi_{spec.name}_log1p_n"
        ].fillna(0.0).astype("float32")
    return output


def adjust_with_frozen_interactions(
    prediction: np.ndarray,
    features: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    offset: float = 0.0,
) -> np.ndarray:
    """Apply the sealed interaction-logit correction to probabilities."""

    score = _logit(np.asarray(prediction, dtype="float64")) + float(offset)
    for name, weight in weights.items():
        score = score + float(weight) * features[
            f"fi_{name}_logit_effect"
        ].to_numpy(dtype="float64")
    return np.clip(1.0 / (1.0 + np.exp(-score)), 1e-5, 1.0 - 1e-5)
