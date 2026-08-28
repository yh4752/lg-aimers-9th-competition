import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.hc_metrics import (
    brier_score,
    calibration_gap,
    expected_calibration_error,
    fit_c0_decile_edges,
    paired_cluster_bootstrap,
    segment_diagnostics,
)


def test_basic_probability_metrics_match_hand_calculation():
    target = np.asarray([0.0, 1.0])
    probability = np.asarray([0.2, 0.8])
    assert brier_score(target, probability) == pytest.approx(0.04)
    assert calibration_gap(target, probability) == pytest.approx(0.0)
    assert expected_calibration_error(target, probability, bins=10) == pytest.approx(0.2)


def test_pitcher_cluster_bootstrap_resamples_clusters_not_rows():
    frame = pd.DataFrame(
        {
            "pitcher_id": ["p1", "p1", "p2", "p2"],
            "target": [0.0, 1.0, 0.0, 1.0],
            "p0": [0.3, 0.7, 0.3, 0.7],
            "p1": [0.2, 0.8, 0.2, 0.8],
        }
    )
    result = paired_cluster_bootstrap(frame, "p0", "p1", repeats=1000, seed=3407)
    assert result.cluster_column == "pitcher_id"
    assert result.repeats == 1000
    assert result.lower_95 == pytest.approx(0.05)
    assert result.mean_gain == pytest.approx(0.05)


def test_decile_edges_are_fit_on_supplied_training_evidence_only():
    training = np.linspace(0.0, 1.0, 101)
    edges = fit_c0_decile_edges(training)
    assert len(edges) == 11
    assert edges[0] == 0.0
    assert edges[-1] == 1.0
    assert edges[5] == pytest.approx(0.5)


def test_segment_diagnostics_gate_only_large_segments():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "p0": [0.4, 0.6, 0.4, 0.6, 0.4, 0.6],
            "p1": [0.3, 0.7, 0.5, 0.5, 0.3, 0.7],
            "oof_year": [2024] * 6,
            "game_type": ["R", "R", "F", "F", "R", "R"],
            "pitcher_id_known": ["known"] * 6,
            "batter_id_known": ["known"] * 6,
            "pitcher_hand": ["R"] * 6,
            "batter_hand": ["L"] * 6,
            "balls_before": [0] * 6,
            "strikes_before": [0] * 6,
            "base_state": ["0"] * 6,
            "c0_decile": [0, 0, 0, 0, 0, 0],
        }
    )
    report = segment_diagnostics(frame, "p1", minimum_rows=3)
    game = report.loc[report["family"].eq("game_type")].set_index("value")
    assert bool(game.loc["R", "eligible"])
    assert not bool(game.loc["F", "eligible"])
    assert report.loc[report["eligible"], "regression"].max() == pytest.approx(-1 / 60)


def test_probability_metrics_reject_nonfinite_values():
    with pytest.raises(ValueError, match="probability"):
        brier_score(np.asarray([0.0]), np.asarray([np.nan]))
