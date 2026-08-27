import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.t3_diagnostics import (
    T3DiagnosticError,
    calibration_diagnostics,
    residual_correlation,
    segment_diagnostics,
)


def labeled_rows():
    return pd.DataFrame({
        "control_success": [0, 1, 0, 1, 1, 0],
        "game_type": ["R", "R", "F", "F", "R", "F"],
        "asof_pitcher_n": [0, 10, 40, 120, 700, 20],
        "pitcher_hand": ["R"] * 6, "batter_hand": ["L"] * 6,
        "balls_before": [0, 1, 2, 3, 0, 1],
        "strikes_before": [0, 1, 2, 1, 2, 0],
    })


def test_diagnostics_use_only_labeled_oof_rows():
    rows = labeled_rows()
    target = rows["control_success"].to_numpy(dtype="float64")
    baseline = np.full(6, 0.5)
    candidate = np.array([0.2, 0.8, 0.3, 0.7, 0.7, 0.2])
    report = segment_diagnostics(rows, target, candidate, baseline, minimum_rows=2)
    assert set(report.columns) == {
        "segment", "value", "rows", "baseline_brier", "candidate_brier", "gain",
    }
    assert (report["rows"] >= 2).all()
    correlation = residual_correlation(target, {"baseline": baseline, "candidate": candidate})
    assert correlation.index.tolist() == ["baseline", "candidate"]
    assert np.allclose(np.diag(correlation), 1.0)
    calibration = calibration_diagnostics(target, candidate)
    assert calibration["rows"].sum() == 6


def test_segment_diagnostics_reject_unlabeled_evaluation_rows():
    rows = labeled_rows().drop(columns="control_success")
    with pytest.raises(T3DiagnosticError, match="labeled OOF"):
        segment_diagnostics(rows, np.zeros(6), np.zeros(6), np.zeros(6), minimum_rows=2)

