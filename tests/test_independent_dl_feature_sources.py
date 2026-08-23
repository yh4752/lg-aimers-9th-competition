from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.independent_dl.feature_sources.seasonal import (
    build_seasonal_snapshot,
)
from experiments.independent_dl.feature_sources.trackman import (
    build_trackman_lookup,
)


def _tiny_main() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2024],
            "pitcher_id": [11, 11],
            "pitcher_hand": [1, 1],
            "pitcher_team_id": [7, 7],
            "asof_pitcher_n": [99, 100],
            "asof_pitcher_success_rate": [0.55, 0.99],
            "asof_pitcher_reverse_rate": [0.1, 0.9],
            "asof_pitcher_middle_rate": [0.3, 0.9],
            "asof_pitcher_ball_rate": [0.4, 0.9],
            "asof_pitcher_strike_rate": [0.6, 0.1],
            "asof_pitcher_pitchmix_n": [100, 101],
            "asof_pitcher_fastball_rate": [0.5, 0.1],
            "asof_pitcher_breaking_rate": [0.3, 0.1],
            "asof_pitcher_offspeed_rate": [0.2, 0.8],
            "batter_id": [21, 21],
            "asof_batter_n": [49, 50],
            "asof_batter_success_rate": [0.5, 0.9],
            "asof_batter_middle_rate": [0.25, 0.9],
            "control_success": [1, 0],
        }
    )


def _tiny_history() -> pd.DataFrame:
    groups = ["fastball"] * 50 + ["breaking"] * 30 + ["offspeed"] * 20
    frame = pd.DataFrame(
        {
            "season": [2023] * 100,
            "pitcher_trackman_id": [101] * 100,
            "pitch_type_group": groups,
            "pitcher_hand": ["Left"] * 100,
            "pitcher_team": ["A"] * 100,
            "rel_speed": [145.0] * 100,
            "spin_rate": [2200.0] * 100,
            "induced_vert_break": [15.0] * 100,
            "horz_break": [3.0] * 100,
            "extension": [1.8] * 100,
            "rel_height": [1.7] * 100,
            "rel_side": [0.2] * 100,
            "zone_speed": [132.0] * 100,
        }
    )
    future = frame.iloc[[0]].copy()
    future["season"] = 2024
    future["rel_speed"] = 999.0
    return pd.concat([frame, future], ignore_index=True)


def test_trackman_and_seasonal_sources_are_cutoff_bound() -> None:
    train = _tiny_main()
    history = _tiny_history()

    first = build_trackman_lookup(train, history, cutoff_year=2023)
    changed = history.copy()
    changed.loc[changed["season"].eq(2024), "rel_speed"] = -999.0
    second = build_trackman_lookup(train, changed, cutoff_year=2023)
    seasonal = build_seasonal_snapshot(train, cutoff_year=2023)

    pd.testing.assert_frame_equal(first.lookup, second.lookup)
    assert first.cutoff_year == 2023
    assert first.lookup["pitcher_id"].is_unique
    assert any(column.startswith("tm_") for column in first.lookup.columns)
    assert seasonal.cutoff_year == 2023
    assert seasonal.pitcher["snapshot_pitcher_success_n"].tolist() == [100.0]
    assert seasonal.pitcher["snapshot_pitcher_success_count"].tolist() == [55.45]
    assert seasonal.batter["snapshot_batter_success_n"].tolist() == [50.0]
    assert seasonal.batter["snapshot_batter_success_count"].tolist() == [25.5]
    assert np.isfinite(first.lookup.select_dtypes(include="number").to_numpy()).any()
