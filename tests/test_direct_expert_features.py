from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.direct_expert.features import (
    DirectFeatureError,
    feature_profile,
    fit_direct_features,
    transform_direct_features,
)
from tests.direct_expert_fixtures import make_history_rows, make_train_rows


@pytest.fixture
def train_rows() -> pd.DataFrame:
    return make_train_rows()


@pytest.fixture
def history_rows() -> pd.DataFrame:
    return make_history_rows()


def test_direct_features_are_target_blind_and_order_independent(
    train_rows: pd.DataFrame,
    history_rows: pd.DataFrame,
) -> None:
    prefix = train_rows.loc[train_rows["season"].lt(2024)]
    state, fitted = fit_direct_features(prefix, history_rows, valid_year=2024)
    valid = train_rows.loc[train_rows["season"].eq(2024)].drop(columns="control_success")

    forward = transform_direct_features(valid, state)
    reverse = transform_direct_features(valid.iloc[::-1], state)
    aligned = reverse.frame.set_axis(reverse.row_id).loc[forward.row_id]

    assert tuple(fitted.frame.columns) == state.feature_columns
    assert tuple(forward.frame.columns) == state.feature_columns
    pd.testing.assert_frame_equal(
        forward.frame.reset_index(drop=True),
        aligned.reset_index(drop=True),
    )
    assert "control_success" not in forward.frame
    numeric = forward.frame.select_dtypes(exclude="object").to_numpy(dtype="float64")
    assert np.isfinite(numeric).all()


def test_snapshot_rejects_future_fit_rows(
    train_rows: pd.DataFrame,
    history_rows: pd.DataFrame,
) -> None:
    with pytest.raises(DirectFeatureError, match="training rows reach validation season"):
        fit_direct_features(train_rows, history_rows, valid_year=2024)


def test_high_ctr_profile_adds_registered_interactions(
    train_rows: pd.DataFrame,
    history_rows: pd.DataFrame,
) -> None:
    prefix = train_rows.loc[train_rows["season"].lt(2024)]
    state, batch = fit_direct_features(prefix, history_rows, valid_year=2024)
    required = {
        "pitcher_batter",
        "pitcher_batter_count",
        "batter_count",
        "team_matchup_game_type",
    }

    assert required.issubset(batch.frame.columns)
    assert required.issubset(state.high_ctr_columns)
    assert required.issubset(state.categorical_columns)
    standard, _ = feature_profile(batch.frame, state.categorical_columns, "standard")
    high_ctr, _ = feature_profile(batch.frame, state.categorical_columns, "high_ctr")
    assert required.isdisjoint(standard)
    assert required.issubset(high_ctr)


def test_unavailable_batter_trackman_is_explicit(
    train_rows: pd.DataFrame,
    history_rows: pd.DataFrame,
) -> None:
    prefix = train_rows.loc[train_rows["season"].lt(2024)]
    state, batch = fit_direct_features(prefix, history_rows, valid_year=2024)

    assert state.batter_trackman_state is None
    assert batch.frame["tm_batter_unavailable"].eq(1.0).all()


def test_transform_rejects_target_column(
    train_rows: pd.DataFrame,
    history_rows: pd.DataFrame,
) -> None:
    prefix = train_rows.loc[train_rows["season"].lt(2024)]
    state, _ = fit_direct_features(prefix, history_rows, valid_year=2024)
    valid = train_rows.loc[train_rows["season"].eq(2024)]

    with pytest.raises(DirectFeatureError, match="evaluation rows contain target"):
        transform_direct_features(valid, state)
