from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def preprocessing_frame() -> pd.DataFrame:
    n = 5
    return pd.DataFrame(
        {
            "row_id": [f"r-{index}" for index in range(n)],
            "season": [2023] * n,
            "game_type": ["R", "F", "R", "F", "R"],
            "top_bottom": ["T", "B", "T", "B", "T"],
            "base_state": ["000", "100", "010", "001", "110"],
            "pitcher_id": [11, 12, 11, 13, 14],
            "batter_id": [21, 22, 23, 21, 24],
            "pitcher_hand": [1, 2, 1, 2, 1],
            "batter_hand": [2, 1, 2, 1, 1],
            "pitcher_team_id": [1, 1, 2, 2, 3],
            "batter_team_id": [2, 2, 1, 1, 4],
            "balls_before": [0, 1, 2, 3, 1],
            "strikes_before": [0, 1, 2, 1, 2],
            "home_win_expectancy": [0.5, 0.6, 0.4, 0.7, 0.3],
            "away_win_expectancy": [0.5, 0.4, 0.6, 0.3, 0.7],
            "li": [0.8, 1.0, 1.5, 2.0, 3.0],
            "run_top_before": [0, 1, 2, 3, 4],
            "run_bot_before": [0, 0, 1, 1, 2],
            "run_total_before": [0, 1, 3, 4, 6],
            "score_diff_home": [0, -1, 1, -2, 2],
            "score_diff_pitcher_team": [0, 1, -1, 2, -2],
            "asof_pitcher_n": [100, 50, 25, 10, 5],
            "asof_pitcher_pitchmix_n": [100, 50, 25, 10, 5],
            "asof_pitcher_success_rate": [0.55, 0.45, 0.60, 0.40, 0.50],
            "asof_batter_n": [80, 40, 20, 10, 5],
            "asof_batter_success_rate": [0.52, 0.48, 0.58, 0.42, 0.50],
            "asof_batter_middle_rate": [0.14, 0.13, 0.15, 0.12, 0.16],
            **{
                f"asof_pitcher_prev{k}_game_{kind}_rate": [0.5, 0.4, 0.6, 0.3, 0.5]
                for k in (1, 3, 5)
                for kind in ("success", "middle")
            },
            **{
                f"asof_pitcher_{kind}_rate": [0.2, 0.3, 0.4, 0.5, 0.6]
                for kind in (
                    "ball",
                    "breaking",
                    "fastball",
                    "middle",
                    "offspeed",
                    "reverse",
                    "strike",
                )
            },
            "control_success": [1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def preprocessing_train(preprocessing_frame: pd.DataFrame) -> pd.DataFrame:
    return preprocessing_frame.copy()


@pytest.fixture
def preprocessing_valid(preprocessing_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        preprocessing_frame.iloc[:2]
        .assign(
            row_id=["v-0", "v-1"],
            season=2024,
            pitcher_id=[11, 999],
            batter_id=[999, 22],
        )
        .reset_index(drop=True)
    )


@pytest.fixture
def preprocessing_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "season",
            "pitcher_trackman_id",
            "pitch_type_group",
            "pitcher_hand",
            "pitcher_team",
            "rel_speed",
            "spin_rate",
            "induced_vert_break",
            "horz_break",
            "extension",
            "rel_height",
            "rel_side",
            "zone_speed",
        ]
    )
