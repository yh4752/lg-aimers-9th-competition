import pandas as pd
import pytest

from experiments.tree_expert.s4_features import (
    S4FeatureError,
    context_state_from_payload,
    context_state_payload,
    fit_s4_context,
    transform_s4_context,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": ["a", "b", "c", "d"],
        "season": [2021, 2021, 2022, 2022],
        "game_type": ["R", "F", "R", "F"],
        "pitcher_id": ["p1", "p1", "p2", "p2"],
        "batter_id": ["b1", "b2", "b1", "b2"],
        "pitcher_hand": ["R", "R", "L", "L"],
        "batter_hand": ["L", "R", "L", "R"],
        "balls_before": [0, 1, 2, 3],
        "strikes_before": [0, 1, 2, 0],
        "outs_before": [0, 1, 2, 0],
        "base_state": [0, 1, 2, 3],
        "control_success": [0, 1, 1, 0],
    })


def test_transform_is_target_blind_and_order_independent() -> None:
    training = _rows()
    state = fit_s4_context(training, cutoff_year=2022)
    valid = training.drop(columns=["control_success"])
    normal = transform_s4_context(valid, state).sort_values("row_id").reset_index(drop=True)
    shuffled = transform_s4_context(valid.sample(frac=1, random_state=9), state).sort_values("row_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(normal, shuffled)


def test_future_fit_is_rejected() -> None:
    with pytest.raises(S4FeatureError, match="after cutoff"):
        fit_s4_context(_rows(), cutoff_year=2021)


def test_context_state_round_trip() -> None:
    state = fit_s4_context(_rows(), cutoff_year=2022)
    restored = context_state_from_payload(context_state_payload(state))
    assert restored.cutoff_year == 2022
    assert restored.hierarchy.global_count == 4
