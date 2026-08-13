from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.tabm_campaign.decisions import (
    CandidateScore,
    TemporalEvidence,
    choose_refined_champion,
    choose_version_a_survivors,
    ensemble_verdict,
    temporal_verdict,
)
from experiments.tabm_campaign.metrics import MetricError, aligned_brier, ensemble_diagnostics


def _scores() -> list[CandidateScore]:
    return [
        CandidateScore("best", "p3_lite", "periodic", "brier", "one_cycle", 0.2200, 10, 2.0),
        CandidateScore("p2", "p2", "piecewise_linear", "bce", "plateau", 0.2202, 5, 1.0),
        CandidateScore("p3_full", "p3_full", "piecewise_linear", "bce", "plateau", 0.2201, 20, 4.0),
        CandidateScore("diverse", "p2", "piecewise_linear", "bce", "one_cycle", 0.2203, 6, 1.1),
        CandidateScore("duplicate-axis", "p3_lite", "periodic", "brier", "one_cycle", 0.22005, 11, 2.2),
    ]


def test_version_a_keeps_best_and_axis_diversity() -> None:
    survivors = choose_version_a_survivors(_scores())
    assert len(survivors) == 4
    assert survivors[0].reason == "best_overall"
    assert "p2" in {row.score.capacity for row in survivors}
    assert any(row.score.capacity.startswith("p3") for row in survivors)
    assert len({row.score.candidate_id for row in survivors}) == 4


def test_temporal_gate_blocks_primary_regression() -> None:
    evidence = TemporalEvidence("candidate", delta_2024=0.000051, delta_2023=-0.001)
    verdict = temporal_verdict(evidence)
    assert verdict.accepted is False
    assert verdict.reason == "primary_fold_regression"


def test_incomplete_refinement_retains_version_b_champion() -> None:
    evidence = {("refined", 2024): -0.0002, ("version_b", 2024): 0.0, ("version_b", 2023): 0.0}
    verdict = choose_refined_champion(evidence, refined_id="refined", version_b_id="version_b")
    assert verdict.source == "version_b"


def test_ensemble_needs_material_gain_and_older_fold_confirmation() -> None:
    verdict = ensemble_verdict(
        candidate_id="mean2",
        primary_gain=0.000029,
        older_delta=-0.0001,
        weighted_delta=-0.0001,
        segment_passed=True,
    )
    assert verdict.accepted is False
    assert verdict.reason == "primary_gain_below_0_00003"


def test_metrics_require_unique_row_alignment_and_report_equal_mean() -> None:
    target = pd.DataFrame({"row_id": ["a", "b", "c"], "target": [0, 1, 1]})
    left = pd.DataFrame({"row_id": ["c", "a", "b"], "probability": [0.8, 0.1, 0.7]})
    right = pd.DataFrame({"row_id": ["a", "b", "c"], "probability": [0.2, 0.6, 0.9]})
    report = ensemble_diagnostics(target, [left, right])
    expected = np.mean((np.array([0.15, 0.65, 0.85]) - np.array([0, 1, 1])) ** 2)
    assert report.member_count == 2
    assert report.brier == pytest.approx(expected)
    assert aligned_brier(target, left) == pytest.approx(np.mean(np.array([0.1, -0.3, -0.2]) ** 2))

    duplicate = pd.concat([left, left.iloc[:1]], ignore_index=True)
    with pytest.raises(MetricError, match="unique"):
        aligned_brier(target, duplicate)
