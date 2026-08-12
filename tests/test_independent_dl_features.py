from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.independent_dl.features import (
    FeatureContractError,
    fit_feature_view,
    materialize_fold_cache,
    transform_feature_view,
)


def _frame(season: int, row_ids: list[str]) -> pd.DataFrame:
    n = len(row_ids)
    return pd.DataFrame(
        {
            "row_id": row_ids,
            "season": [season] * n,
            "game_month": [3 + index for index in range(n)],
            "game_dayofweek": [index % 7 for index in range(n)],
            "inning": [1 + index for index in range(n)],
            "top_bottom": ["T", "B"][:n],
            "game_type": ["R", "F"][:n],
            "balls_before": [0, 3][:n],
            "strikes_before": [0, 2][:n],
            "outs_before": [0, 2][:n],
            "run_top_before": [0, 2][:n],
            "run_bot_before": [0, 1][:n],
            "score_diff_home": [0, -1][:n],
            "runner_on_1b": [0, 1][:n],
            "runner_on_2b": [0, 0][:n],
            "runner_on_3b": [0, 1][:n],
            "num_runners_on": [0, 2][:n],
            "base_state": ["000", "101"][:n],
            "home_win_expectancy": [0.5, 0.4][:n],
            "away_win_expectancy": [0.5, 0.6][:n],
            "pitcher_id": [11, 12][:n],
            "batter_id": [21, 22][:n],
            "pitcher_hand": [1, 2][:n],
            "batter_hand": [2, 1][:n],
            "pitcher_team_id": [7, 8][:n],
            "batter_team_id": [8, 7][:n],
            "asof_pitcher_n": [99, 49][:n],
            "asof_pitcher_success_rate": [0.55, 0.45][:n],
            "asof_pitcher_fastball_rate": [0.5, 0.4][:n],
            "asof_pitcher_breaking_rate": [0.3, 0.4][:n],
            "asof_pitcher_offspeed_rate": [0.2, 0.2][:n],
            "asof_batter_n": [49, 39][:n],
            "asof_batter_success_rate": [0.5, 0.4][:n],
            "control_success": [1, 0][:n],
        }
    )


@pytest.fixture
def tiny_train() -> pd.DataFrame:
    return _frame(2023, ["tr-1", "tr-2"])


@pytest.fixture
def tiny_valid() -> pd.DataFrame:
    return _frame(2024, ["va-1", "va-2"])


@pytest.fixture
def tiny_history() -> pd.DataFrame:
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


def test_all_views_fit_on_train_and_preserve_validation_rows(
    tiny_train: pd.DataFrame,
    tiny_valid: pd.DataFrame,
    tiny_history: pd.DataFrame,
) -> None:
    for view in (
        "raw_typed",
        "engineered",
        "entity_context",
        "trackman_augmented",
    ):
        state, train = fit_feature_view(
            tiny_train, tiny_history, view=view, cutoff_year=2023
        )
        valid = transform_feature_view(tiny_valid, state)

        assert train.x_num.shape[0] == len(tiny_train)
        assert valid.x_num.shape[0] == len(tiny_valid)
        assert valid.row_id.tolist() == tiny_valid["row_id"].astype(str).tolist()
        assert "control_success" not in state.numeric_columns
        assert "row_id" not in state.numeric_columns
        assert train.x_num.dtype.name == "float32"
        assert train.x_cat.dtype.name == "int64"


def test_unseen_categories_use_reserved_index(
    tiny_train: pd.DataFrame,
    tiny_valid: pd.DataFrame,
    tiny_history: pd.DataFrame,
) -> None:
    state, _ = fit_feature_view(
        tiny_train, tiny_history, view="entity_context", cutoff_year=2023
    )
    changed = tiny_valid.assign(pitcher_id=999999)
    transformed = transform_feature_view(changed, state)
    pitcher_position = state.categorical_columns.index("pitcher_id")

    assert transformed.x_cat[:, pitcher_position].tolist() == [0] * len(changed)


def test_cache_reuse_requires_identity_match(
    tmp_path: Path,
    tiny_train: pd.DataFrame,
    tiny_valid: pd.DataFrame,
    tiny_history: pd.DataFrame,
) -> None:
    first = materialize_fold_cache(
        tmp_path,
        tiny_train,
        tiny_valid,
        tiny_history,
        "raw_typed",
        2023,
        2024,
    )
    second = materialize_fold_cache(
        tmp_path,
        tiny_train,
        tiny_valid,
        tiny_history,
        "raw_typed",
        2023,
        2024,
    )

    assert first.reused is False
    assert second.reused is True
    changed = tiny_valid.iloc[::-1].reset_index(drop=True)
    with pytest.raises(FeatureContractError, match="row identity"):
        materialize_fold_cache(
            tmp_path,
            tiny_train,
            changed,
            tiny_history,
            "raw_typed",
            2023,
            2024,
        )


def test_trackman_cache_reloads_bound_lookup(
    tmp_path: Path,
    tiny_train: pd.DataFrame,
    tiny_valid: pd.DataFrame,
    tiny_history: pd.DataFrame,
) -> None:
    first = materialize_fold_cache(
        tmp_path,
        tiny_train,
        tiny_valid,
        tiny_history,
        "trackman_augmented",
        2023,
        2024,
    )
    second = materialize_fold_cache(
        tmp_path,
        tiny_train,
        tiny_valid,
        tiny_history,
        "trackman_augmented",
        2023,
        2024,
    )

    assert first.reused is False
    assert second.reused is True
    assert second.state.trackman_result is not None
    assert second.state.trackman_lookup_sha256 == first.state.trackman_lookup_sha256
