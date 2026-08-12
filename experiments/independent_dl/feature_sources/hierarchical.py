"""Frozen prior-season target summaries.

Source commit: ``9454d68b93971627e3d3f613ce30be690cb5dce2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COLUMN = "control_success"


@dataclass(frozen=True)
class PriorSeasonSnapshot:
    """One completed-season global, pitcher, and batter target summary."""

    cutoff_year: int
    prior_rate: float
    pitcher: pd.DataFrame
    batter: pd.DataFrame


def _entity_summary(frame: pd.DataFrame, entity: str, prefix: str) -> pd.DataFrame:
    summary = (
        frame.groupby(entity, sort=False, observed=True)[TARGET_COLUMN]
        .agg(["size", "sum"])
        .reset_index()
    )
    return summary.rename(
        columns={
            "size": f"prev_{prefix}_success_n",
            "sum": f"prev_{prefix}_success_count",
        }
    )


def build_prior_season_snapshot(
    labeled_history: pd.DataFrame, cutoff_year: int
) -> PriorSeasonSnapshot:
    """Summarize the cutoff season after explicitly excluding all future rows."""

    history = labeled_history.loc[labeled_history["season"].le(cutoff_year)].copy()
    source = history.loc[history["season"].eq(cutoff_year)]
    if source.empty:
        raise ValueError(f"No labeled rows for cutoff year {cutoff_year}")
    return PriorSeasonSnapshot(
        cutoff_year=cutoff_year,
        prior_rate=float(source[TARGET_COLUMN].mean()),
        pitcher=_entity_summary(source, "pitcher_id", "pitcher"),
        batter=_entity_summary(source, "batter_id", "batter"),
    )


def _smoothed_prior(
    count: np.ndarray, n: np.ndarray, global_rate: float, strength: float
) -> np.ndarray:
    return (count + strength * global_rate) / (n + strength)


def _posterior(
    current_count: np.ndarray,
    current_n: np.ndarray,
    prior: np.ndarray,
    strength: float,
) -> np.ndarray:
    return (current_count + strength * prior) / (current_n + strength)


def attach_hierarchical_features(
    frame: pd.DataFrame, snapshot: PriorSeasonSnapshot
) -> pd.DataFrame:
    """Attach row-local posteriors backed only by a frozen prior snapshot."""

    output = frame.merge(
        snapshot.pitcher,
        on="pitcher_id",
        how="left",
        validate="many_to_one",
        sort=False,
    ).merge(
        snapshot.batter,
        on="batter_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    output.index = frame.index
    prior_rate = float(snapshot.prior_rate)

    pitcher_prev_n = pd.to_numeric(
        output["prev_pitcher_success_n"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype="float64")
    pitcher_prev_count = pd.to_numeric(
        output["prev_pitcher_success_count"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype="float64")
    pitcher_season_n = pd.to_numeric(
        output["season_pitcher_n"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype="float64")
    pitcher_season_rate = pd.to_numeric(
        output["season_pitcher_success_rate"], errors="coerce"
    ).fillna(prior_rate).to_numpy(dtype="float64")
    pitcher_season_count = pitcher_season_n * pitcher_season_rate

    output["prev_pitcher_log1p_n"] = np.log1p(pitcher_prev_n).astype("float32")
    output["prev_pitcher_seen"] = (pitcher_prev_n > 0).astype("int8")
    for prior_strength in (100.0, 200.0, 500.0, 1000.0):
        prior = _smoothed_prior(
            pitcher_prev_count, pitcher_prev_n, prior_rate, prior_strength
        )
        prior_name = int(prior_strength)
        output[f"prev_pitcher_success_smooth_{prior_name}"] = prior.astype(
            "float32"
        )
        for season_strength in (25.0, 50.0, 100.0, 150.0, 200.0):
            output[
                f"hier_pitcher_success_h{prior_name}_k{int(season_strength)}"
            ] = _posterior(
                pitcher_season_count, pitcher_season_n, prior, season_strength
            ).astype("float32")

    batter_prev_n = pd.to_numeric(
        output["prev_batter_success_n"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype="float64")
    batter_prev_count = pd.to_numeric(
        output["prev_batter_success_count"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype="float64")
    batter_season_n = pd.to_numeric(
        output["season_batter_n"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype="float64")
    batter_season_rate = pd.to_numeric(
        output["season_batter_success_smooth_200"], errors="coerce"
    ).fillna(prior_rate).to_numpy(dtype="float64")
    batter_season_count = np.maximum(
        batter_season_rate * (batter_season_n + 200.0) - 200.0 * prior_rate,
        0.0,
    )
    batter_prior = _smoothed_prior(
        batter_prev_count, batter_prev_n, prior_rate, 1000.0
    )
    output["prev_batter_log1p_n"] = np.log1p(batter_prev_n).astype("float32")
    output["prev_batter_seen"] = (batter_prev_n > 0).astype("int8")
    output["prev_batter_success_smooth_1000"] = batter_prior.astype("float32")
    output["hier_batter_success_h1000_k200"] = _posterior(
        batter_season_count, batter_season_n, batter_prior, 200.0
    ).astype("float32")
    output["hier_pitcher_batter_success_gap"] = (
        output["hier_pitcher_success_h1000_k50"]
        - output["hier_batter_success_h1000_k200"]
    ).astype("float32")
    return output.drop(
        columns=[
            "prev_pitcher_success_n",
            "prev_pitcher_success_count",
            "prev_batter_success_n",
            "prev_batter_success_count",
        ]
    )
