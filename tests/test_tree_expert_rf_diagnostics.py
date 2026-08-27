from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.tree_expert.rf_diagnostics import build_rf_diagnostics


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d"],
            "target": [0, 1, 0, 1],
            "game_type": ["R", "F", "R", "F"],
        }
    )


def test_diagnostics_reports_required_segments_and_calibration() -> None:
    report = build_rf_diagnostics(
        _frame(),
        np.array([0.4, 0.6, 0.4, 0.6]),
        np.array([0.3, 0.7, 0.3, 0.7]),
        minimum_rows=2,
    )

    assert {row["segment"] for row in report["game_type"]} == {"R", "F"}
    assert len(report["calibration"]) == 10
    assert "prediction_correlation" in report
    assert report["overall"]["gain"] > 0


def test_diagnostics_ignores_small_reporting_segments_without_mutation() -> None:
    candidate = np.array([0.3, 0.7, 0.3, 0.7])
    before = candidate.copy()

    report = build_rf_diagnostics(
        _frame(),
        np.array([0.4, 0.6, 0.4, 0.6]),
        candidate,
        minimum_rows=10_000,
    )

    assert report["game_type"] == []
    np.testing.assert_array_equal(candidate, before)
