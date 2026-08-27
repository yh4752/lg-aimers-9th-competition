from __future__ import annotations

from types import MappingProxyType

import pandas as pd
import pytest

from experiments.temporal_portfolio.seasonal_features import fit_s1_state
from experiments.tree_expert.features import TreeFeatureState
from experiments.tree_expert.rf_state import (
    RFStateError,
    export_frozen_tree_state,
    load_frozen_tree_state,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023], "pitcher_id": [1, 2], "batter_id": [11, 12],
            "asof_pitcher_n": [10, 20], "asof_pitcher_success_rate": [0.5, 0.6],
            "asof_pitcher_reverse_rate": [0.1, 0.2], "asof_pitcher_middle_rate": [0.2, 0.2],
            "asof_pitcher_ball_rate": [0.4, 0.3], "asof_pitcher_strike_rate": [0.6, 0.7],
            "asof_pitcher_pitchmix_n": [10, 20], "asof_pitcher_fastball_rate": [0.5, 0.6],
            "asof_pitcher_breaking_rate": [0.3, 0.2], "asof_pitcher_offspeed_rate": [0.2, 0.2],
            "asof_batter_n": [5, 8], "asof_batter_success_rate": [0.4, 0.5],
            "asof_batter_middle_rate": [0.2, 0.25], "control_success": [0, 1],
        }
    )


def _state() -> TreeFeatureState:
    s1 = fit_s1_state(_rows(), valid_year=2024)
    return TreeFeatureState(
        valid_year=2024,
        prior_rate=s1.prior_rate,
        categorical_columns=("pitcher_id",),
        feature_columns=("pitcher_id", "value"),
        s1_state=s1,
        trackman_state=None,
        source_hashes=MappingProxyType({"S1": "a" * 64}),
    )


def test_rf_frozen_state_round_trip(tmp_path) -> None:
    original = _state()
    export_frozen_tree_state(original, tmp_path / "state")
    restored = load_frozen_tree_state(tmp_path / "state")
    assert restored.valid_year == original.valid_year
    assert restored.feature_columns == original.feature_columns
    pd.testing.assert_frame_equal(restored.s1_state.snapshot.pitcher, original.s1_state.snapshot.pitcher)


def test_rf_frozen_state_rejects_tampering(tmp_path) -> None:
    export_frozen_tree_state(_state(), tmp_path / "state")
    with (tmp_path / "state/s1_pitcher.csv").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RFStateError, match="SHA-256"):
        load_frozen_tree_state(tmp_path / "state")
