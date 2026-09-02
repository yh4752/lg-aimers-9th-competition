from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.failure_regime_e3.labels import E3LabelError, recover_failure_targets


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "pitcher_id": ["p1"] * 4,
            "asof_pitcher_n": [0, 1, 2, 3],
            "asof_pitcher_success_rate": [0.0, 0.0, 0.0, 1.0 / 3.0],
            "asof_pitcher_middle_rate": [0.0, 1.0, 0.5, 1.0 / 3.0],
            "asof_pitcher_ball_rate": [0.0, 1.0, 1.0, 2.0 / 3.0],
            "asof_pitcher_reverse_rate": [0.0, 1.0, 0.5, 1.0 / 3.0],
            "control_success": [0, 0, 1, 0],
        }
    )


def test_overlapping_middle_and_reverse_are_retained() -> None:
    recovered = recover_failure_targets(_rows(), valid_year=2025, tolerance=0.02)

    assert recovered.valid_mask.tolist() == [True, True, True, False]
    assert recovered.frame.loc[0, ["middle", "wild", "reverse"]].tolist() == [1.0, 0.0, 1.0]
    assert recovered.overlap_rate == pytest.approx(0.5)
    assert recovered.positive_counts == {"middle": 1, "wild": 1, "reverse": 1}


def test_ball_delta_is_audit_only_and_wild_is_residual_failure() -> None:
    recovered = recover_failure_targets(_rows(), valid_year=2025, tolerance=0.02)

    assert recovered.frame.loc[0, "ball_result"] == 1.0
    assert recovered.frame.loc[0, "wild"] == 0.0
    assert recovered.frame.loc[1, "ball_result"] == 1.0
    assert recovered.frame.loc[1, "wild"] == 1.0
    assert recovered.frame.loc[2, "success"] == 1.0
    assert recovered.frame.loc[2, "wild"] == 0.0


def test_source_order_is_preserved_after_pitcher_linking() -> None:
    source = pd.concat(
        [
            _rows().assign(pitcher_id="p1"),
            _rows().assign(pitcher_id="p2", season=2023),
        ],
        ignore_index=True,
    ).iloc[[4, 0, 5, 1, 6, 2, 7, 3]].reset_index(drop=True)
    recovered = recover_failure_targets(source, valid_year=2025, tolerance=0.02)

    assert recovered.frame["source_position"].tolist() == list(range(len(source)))
    np.testing.assert_array_equal(recovered.valid_mask, recovered.frame["valid"].to_numpy())


def test_validation_rows_are_rejected() -> None:
    with pytest.raises(E3LabelError, match="reach validation season"):
        recover_failure_targets(_rows(), valid_year=2024, tolerance=0.02)
