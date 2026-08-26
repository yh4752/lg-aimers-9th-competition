from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.features import (
    TreeFeatureError,
    fit_tree_features,
    transform_tree_features,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["h0", "h1", "h2", "v0", "v1", "v2"],
            "season": [2022, 2023, 2023, 2024, 2024, 2024],
            "game_month": [3, 4, 5, 3, 4, 5],
            "game_dayofweek": [1, 2, 3, 1, 2, 3],
            "inning": [1, 4, 8, 2, 7, 10],
            "top_bottom": ["T", "B", "T", "B", "T", "B"],
            "game_type": ["R", "R", "F", "R", "F", "R"],
            "balls_before": [0, 1, 2, 3, 0, 1],
            "strikes_before": [0, 1, 2, 1, 2, 0],
            "outs_before": [0, 1, 2, 0, 1, 2],
            "run_top_before": [0, 1, 1, 0, 2, 3],
            "run_bot_before": [0, 0, 2, 1, 1, 2],
            "run_total_before": [0, 1, 3, 1, 3, 5],
            "score_diff_home": [0, -1, 1, 1, -1, -1],
            "score_diff_pitcher_team": [0, 1, -1, -1, 1, 1],
            "runner_on_1b": [0, 1, 0, 1, 0, 1],
            "runner_on_2b": [0, 0, 1, 1, 0, 0],
            "runner_on_3b": [0, 0, 0, 0, 1, 0],
            "num_runners_on": [0, 1, 1, 2, 1, 1],
            "base_state": ["000", "100", "010", "110", "001", "100"],
            "home_win_expectancy": [0.5, 0.45, 0.55, 0.6, 0.4, 0.5],
            "away_win_expectancy": [0.5, 0.55, 0.45, 0.4, 0.6, 0.5],
            "li": [0.5, 1.0, 2.0, 0.8, 2.5, 3.5],
            "pitcher_id": [11, 11, 12, 11, 12, 999],
            "batter_id": [21, 21, 22, 21, 22, 999],
            "pitcher_hand": [1, 1, 2, 1, 2, 1],
            "batter_hand": [1, 1, 2, 2, 1, 2],
            "pitcher_team_id": [7, 7, 8, 7, 8, 99],
            "batter_team_id": [9, 9, 10, 9, 10, 99],
            "asof_pitcher_n": [8, 9, 4, 10, 5, 0],
            "asof_pitcher_success_rate": [0.50, 5 / 9, 0.50, 0.60, 0.60, 0.50],
            "asof_pitcher_reverse_rate": [0.10, 1 / 9, 0.25, 0.20, 0.20, 0.0],
            "asof_pitcher_middle_rate": [0.25, 2 / 9, 0.25, 0.30, 0.20, 0.0],
            "asof_pitcher_ball_rate": [0.40, 4 / 9, 0.50, 0.40, 0.40, 0.0],
            "asof_pitcher_strike_rate": [0.60, 5 / 9, 0.50, 0.60, 0.60, 0.0],
            "asof_pitcher_prev1_game_success_rate": [0.5, 0.5, 0.5, 0.7, np.nan, np.nan],
            "asof_pitcher_prev3_game_success_rate": [0.5, 0.5, 0.5, 0.6, 0.5, np.nan],
            "asof_pitcher_prev5_game_success_rate": [0.5, 0.5, 0.5, 0.5, 0.4, np.nan],
            "asof_pitcher_prev1_game_middle_rate": [0.2, 0.2, 0.3, 0.2, 0.3, np.nan],
            "asof_pitcher_prev3_game_middle_rate": [0.2, 0.2, 0.3, 0.25, 0.25, np.nan],
            "asof_pitcher_prev5_game_middle_rate": [0.2, 0.2, 0.3, 0.3, 0.2, np.nan],
            "asof_batter_n": [6, 7, 3, 8, 4, 0],
            "asof_batter_success_rate": [0.50, 4 / 7, 1 / 3, 0.625, 0.50, 0.50],
            "asof_batter_middle_rate": [1 / 6, 1 / 7, 1 / 3, 0.25, 0.25, 0.0],
            "asof_pitcher_pitchmix_n": [8, 9, 4, 10, 5, 0],
            "asof_pitcher_fastball_rate": [0.50, 5 / 9, 0.50, 0.60, 0.40, 0.0],
            "asof_pitcher_breaking_rate": [0.25, 2 / 9, 0.25, 0.20, 0.40, 0.0],
            "asof_pitcher_offspeed_rate": [0.25, 2 / 9, 0.25, 0.20, 0.20, 0.0],
            "control_success": [1, 0, 1, 1, 0, 1],
        },
        index=[30, 10, 20, 8, 8, 3],
    )


@pytest.fixture
def train_prefix() -> pd.DataFrame:
    rows = _rows()
    return rows.loc[rows["season"].lt(2024)].copy(deep=True)


@pytest.fixture
def valid_rows() -> pd.DataFrame:
    rows = _rows()
    return rows.loc[rows["season"].eq(2024)].drop(columns="control_success")


def test_tree_features_are_invariant_to_validation_neighbors(
    train_prefix: pd.DataFrame,
    valid_rows: pd.DataFrame,
) -> None:
    state, _ = fit_tree_features(
        train_prefix,
        history=None,
        valid_year=2024,
        use_trackman=False,
    )
    whole = transform_tree_features(valid_rows, state)

    for position in (0, len(valid_rows) // 2, len(valid_rows) - 1):
        one = transform_tree_features(valid_rows.iloc[[position]], state)
        pd.testing.assert_frame_equal(
            one.frame.reset_index(drop=True),
            whole.frame.iloc[[position]].reset_index(drop=True),
        )
        np.testing.assert_allclose(one.anchor, whole.anchor[[position]])


def test_tree_features_have_fixed_high_cardinality_crosses(
    train_prefix: pd.DataFrame,
) -> None:
    state, batch = fit_tree_features(
        train_prefix,
        history=None,
        valid_year=2024,
        use_trackman=False,
    )

    assert {
        "pitcher_batter_hand",
        "pitcher_count",
        "pitcher_base",
        "pitcher_game_type",
        "batter_pitcher_hand",
        "team_count",
    } <= set(state.categorical_columns)
    assert tuple(batch.frame.columns) == state.feature_columns
    assert np.isfinite(batch.anchor).all()
    assert np.all((batch.anchor > 0) & (batch.anchor < 1))


def test_training_s1_never_reads_validation_target(
    train_prefix: pd.DataFrame,
    valid_rows: pd.DataFrame,
) -> None:
    state, _ = fit_tree_features(
        train_prefix,
        history=None,
        valid_year=2024,
        use_trackman=False,
    )

    with pytest.raises(TreeFeatureError, match="evaluation rows contain target"):
        transform_tree_features(valid_rows.assign(control_success=1), state)
