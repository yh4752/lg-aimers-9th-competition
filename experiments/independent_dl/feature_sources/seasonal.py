"""Cutoff-bound current-season deltas.

Source commit: ``9454d68b93971627e3d3f613ce30be690cb5dce2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COLUMN = "control_success"
PITCHER_RATE_COLUMNS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
)
PITCHMIX_RATE_COLUMNS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


@dataclass(frozen=True)
class SeasonalSnapshot:
    """Cumulative pitcher and batter states frozen at a season cutoff."""

    cutoff_year: int
    pitcher: pd.DataFrame
    batter: pd.DataFrame


def _last_rows(frame: pd.DataFrame, entity: str, count: str) -> pd.DataFrame:
    return (
        frame.sort_values([entity, count], kind="stable")
        .groupby(entity, sort=False, observed=True)
        .tail(1)
        .copy()
    )


def build_seasonal_snapshot(train: pd.DataFrame, cutoff_year: int) -> SeasonalSnapshot:
    """Freeze cumulative states using only labeled rows through ``cutoff_year``."""

    history = train.loc[train["season"].le(cutoff_year)].copy()
    if history.empty:
        raise ValueError(f"No history through {cutoff_year}")

    pitcher = _last_rows(history, "pitcher_id", "asof_pitcher_n")
    pitcher_output = pitcher[["pitcher_id"]].copy()
    pitcher_n = pd.to_numeric(pitcher["asof_pitcher_n"]).fillna(0).to_numpy(
        dtype="float64"
    )
    pitcher_output["snapshot_pitcher_success_n"] = pitcher_n + 1.0
    pitcher_output["snapshot_pitcher_success_count"] = (
        pitcher_n
        * pd.to_numeric(pitcher["asof_pitcher_success_rate"])
        .fillna(0)
        .to_numpy(dtype="float64")
        + pitcher[TARGET_COLUMN].to_numpy(dtype="float64")
    )
    for column in PITCHER_RATE_COLUMNS[1:]:
        suffix = column.removeprefix("asof_pitcher_").removesuffix("_rate")
        pitcher_output[f"snapshot_pitcher_{suffix}_n"] = pitcher_n
        pitcher_output[f"snapshot_pitcher_{suffix}_count"] = (
            pitcher_n
            * pd.to_numeric(pitcher[column]).fillna(0).to_numpy(dtype="float64")
        )

    mix_n = pd.to_numeric(pitcher["asof_pitcher_pitchmix_n"]).fillna(0).to_numpy(
        dtype="float64"
    )
    for column in PITCHMIX_RATE_COLUMNS:
        suffix = column.removeprefix("asof_pitcher_").removesuffix("_rate")
        pitcher_output[f"snapshot_pitchmix_{suffix}_n"] = mix_n
        pitcher_output[f"snapshot_pitchmix_{suffix}_count"] = (
            mix_n
            * pd.to_numeric(pitcher[column]).fillna(0).to_numpy(dtype="float64")
        )

    batter = _last_rows(history, "batter_id", "asof_batter_n")
    batter_output = batter[["batter_id"]].copy()
    batter_n = pd.to_numeric(batter["asof_batter_n"]).fillna(0).to_numpy(
        dtype="float64"
    )
    batter_output["snapshot_batter_success_n"] = batter_n + 1.0
    batter_output["snapshot_batter_success_count"] = (
        batter_n
        * pd.to_numeric(batter["asof_batter_success_rate"])
        .fillna(0)
        .to_numpy(dtype="float64")
        + batter[TARGET_COLUMN].to_numpy(dtype="float64")
    )
    batter_output["snapshot_batter_middle_n"] = batter_n
    batter_output["snapshot_batter_middle_count"] = (
        batter_n
        * pd.to_numeric(batter["asof_batter_middle_rate"])
        .fillna(0)
        .to_numpy(dtype="float64")
    )
    return SeasonalSnapshot(
        cutoff_year=cutoff_year,
        pitcher=pitcher_output.reset_index(drop=True),
        batter=batter_output.reset_index(drop=True),
    )


def _season_delta(
    output: pd.DataFrame,
    *,
    current_n: pd.Series,
    current_rate: pd.Series,
    snapshot_n_column: str,
    snapshot_count_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    cumulative_n = pd.to_numeric(current_n).fillna(0).to_numpy(dtype="float64")
    cumulative_count = cumulative_n * pd.to_numeric(current_rate).fillna(0).to_numpy(
        dtype="float64"
    )
    snapshot_n = output[snapshot_n_column].fillna(0).to_numpy(dtype="float64")
    snapshot_count = output[snapshot_count_column].fillna(0).to_numpy(
        dtype="float64"
    )
    delta_n = np.maximum(cumulative_n - snapshot_n, 0.0)
    delta_count = np.clip(cumulative_count - snapshot_count, 0.0, delta_n)
    return delta_n, delta_count


def _smoothed_rate(
    count: np.ndarray, n: np.ndarray, prior: np.ndarray | float, strength: float
) -> np.ndarray:
    return ((count + np.asarray(prior) * strength) / (n + strength)).astype(
        "float32"
    )


def attach_seasonal_features(
    frame: pd.DataFrame, snapshot: SeasonalSnapshot, *, prior_rate: float
) -> tuple[pd.DataFrame, list[str]]:
    """Attach row-local season deltas against one frozen seasonal snapshot."""

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
    if not output.index.equals(frame.index):
        output.index = frame.index

    pitcher_n, pitcher_success = _season_delta(
        output,
        current_n=output["asof_pitcher_n"],
        current_rate=output["asof_pitcher_success_rate"],
        snapshot_n_column="snapshot_pitcher_success_n",
        snapshot_count_column="snapshot_pitcher_success_count",
    )
    output["season_pitcher_n"] = pitcher_n.astype("float32")
    output["season_pitcher_log1p_n"] = np.log1p(pitcher_n).astype("float32")
    output["season_pitcher_reliability_100"] = (
        pitcher_n / (pitcher_n + 100.0)
    ).astype("float32")
    for strength in (10.0, 25.0, 50.0, 100.0, 200.0, 500.0):
        output[f"season_pitcher_success_smooth_{int(strength)}"] = _smoothed_rate(
            pitcher_success, pitcher_n, prior_rate, strength
        )
    season_success = _smoothed_rate(pitcher_success, pitcher_n, prior_rate, 25.0)
    output["season_pitcher_success_rate"] = np.divide(
        pitcher_success,
        pitcher_n,
        out=np.full(len(output), prior_rate, dtype="float64"),
        where=pitcher_n > 0,
    ).astype("float32")
    career_success = pd.to_numeric(output["asof_pitcher_success_rate"]).fillna(
        prior_rate
    ).to_numpy()
    output["season_vs_career_success"] = (
        season_success - career_success
    ).astype("float32")

    for source_column in PITCHER_RATE_COLUMNS[1:]:
        suffix = source_column.removeprefix("asof_pitcher_").removesuffix("_rate")
        component_n, component_count = _season_delta(
            output,
            current_n=output["asof_pitcher_n"],
            current_rate=output[source_column],
            snapshot_n_column=f"snapshot_pitcher_{suffix}_n",
            snapshot_count_column=f"snapshot_pitcher_{suffix}_count",
        )
        career_component = pd.to_numeric(output[source_column]).fillna(0).to_numpy(
            dtype="float64"
        )
        output[f"season_pitcher_{suffix}_smooth_50"] = _smoothed_rate(
            component_count, component_n, career_component, 50.0
        )

    for column in (
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ):
        suffix = column.removeprefix("asof_pitcher_")
        recent_raw = pd.to_numeric(output[column]).to_numpy(dtype="float64")
        recent = np.where(np.isfinite(recent_raw), recent_raw, season_success)
        output[f"season_vs_{suffix}"] = (season_success - recent).astype(
            "float32"
        )

    mix_total = pd.to_numeric(output["asof_pitcher_pitchmix_n"]).fillna(0).to_numpy(
        dtype="float64"
    )
    mix_delta_candidates: list[np.ndarray] = []
    for source_column in PITCHMIX_RATE_COLUMNS:
        suffix = source_column.removeprefix("asof_pitcher_").removesuffix("_rate")
        component_n, component_count = _season_delta(
            output,
            current_n=pd.Series(mix_total, index=output.index),
            current_rate=output[source_column],
            snapshot_n_column=f"snapshot_pitchmix_{suffix}_n",
            snapshot_count_column=f"snapshot_pitchmix_{suffix}_count",
        )
        career_mix = pd.to_numeric(output[source_column]).fillna(0).to_numpy(
            dtype="float64"
        )
        output[f"season_pitchmix_{suffix}_smooth_50"] = _smoothed_rate(
            component_count, component_n, career_mix, 50.0
        )
        mix_delta_candidates.append(component_n)
    output["season_pitchmix_n"] = np.maximum.reduce(mix_delta_candidates).astype(
        "float32"
    )

    batter_n, batter_success = _season_delta(
        output,
        current_n=output["asof_batter_n"],
        current_rate=output["asof_batter_success_rate"],
        snapshot_n_column="snapshot_batter_success_n",
        snapshot_count_column="snapshot_batter_success_count",
    )
    output["season_batter_n"] = batter_n.astype("float32")
    output["season_batter_log1p_n"] = np.log1p(batter_n).astype("float32")
    for strength in (50.0, 100.0, 200.0, 500.0):
        output[f"season_batter_success_smooth_{int(strength)}"] = _smoothed_rate(
            batter_success, batter_n, prior_rate, strength
        )
    output["season_pitcher_batter_success_gap"] = (
        season_success - _smoothed_rate(batter_success, batter_n, prior_rate, 200.0)
    ).astype("float32")

    bucket_labels = ["0", "1_10", "11_50", "51_200", "201_500", "501_1000", "1000p"]
    buckets = [-1, 0, 10, 50, 200, 500, 1000, np.inf]
    output["season_pitcher_n_bucket"] = pd.cut(
        pitcher_n, bins=buckets, labels=bucket_labels
    ).astype(str)
    output["season_batter_n_bucket"] = pd.cut(
        batter_n, bins=buckets, labels=bucket_labels
    ).astype(str)
    output = output.drop(
        columns=[column for column in output if column.startswith("snapshot_")]
    )
    return output, ["season_pitcher_n_bucket", "season_batter_n_bucket"]
