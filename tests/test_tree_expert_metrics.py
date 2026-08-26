from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from experiments.tree_expert.contracts import load_e1_contract
from experiments.tree_expert.inputs import PREDICTION_COLUMNS
from experiments.tree_expert.metrics import (
    CandidateMetric,
    TreeMetricError,
    decide_e1,
    evaluate_e1_candidate,
)


def _predictions(probability: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["r0", "r1", "r2", "r3"],
            "target": [0, 1, 0, 1],
            "probability": probability,
            "game_type": ["R", "R", "F", "F"],
            "game_month": [3, 3, 4, 4],
            "pitcher_id_known": ["known", "known", "oov", "known"],
            "batter_id_known": ["known", "oov", "known", "known"],
        }
    ).loc[:, PREDICTION_COLUMNS]


def _metric(candidate_id: str, gain: float, status: str = "completed") -> CandidateMetric:
    return CandidateMetric(
        candidate_id=candidate_id,
        status=status,
        baseline_brier=0.25 if status == "completed" else None,
        candidate_brier=0.25 - gain if status == "completed" else None,
        gain=gain if status == "completed" else None,
        bootstrap_lower=gain - 0.00001 if status == "completed" else None,
        bootstrap_upper=gain + 0.00001 if status == "completed" else None,
        prediction_correlation=0.5 if status == "completed" else None,
        residual_correlation=0.5 if status == "completed" else None,
        maximum_segment_regression=max(0.0, -gain) if status == "completed" else None,
        segments=(),
        reason=None if status == "completed" else "skipped",
    )


def test_e1_metrics_reject_row_reordering() -> None:
    baseline = _predictions([0.4, 0.6, 0.45, 0.55])
    candidate = _predictions([0.3, 0.7, 0.4, 0.6]).iloc[::-1]

    with pytest.raises(TreeMetricError, match="row_id alignment differs"):
        evaluate_e1_candidate(
            baseline,
            candidate,
            load_e1_contract(),
            candidate_id="c0_native_ctr",
            bootstrap_groups=[1, 1, 2, 2],
        )


def test_e1_metrics_compute_positive_gain_and_pitcher_bootstrap() -> None:
    contract = replace(load_e1_contract(), bootstrap_repeats=50, minimum_segment_rows=2)
    metric = evaluate_e1_candidate(
        _predictions([0.4, 0.6, 0.45, 0.55]),
        _predictions([0.3, 0.7, 0.4, 0.6]),
        contract,
        candidate_id="c0_native_ctr",
        bootstrap_groups=[1, 1, 2, 2],
    )

    assert metric.status == "completed"
    assert metric.gain > 0
    assert metric.bootstrap_lower is not None
    assert {segment.column for segment in metric.segments} == {
        "game_type",
        "game_month",
        "pitcher_id_known",
        "batter_id_known",
    }


def test_e1_decision_promotes_best_two_noncatastrophic_candidates() -> None:
    decision = decide_e1(
        [
            _metric("c0_native_ctr", gain=0.00020),
            _metric("c1_anchor_residual", gain=0.00010),
            _metric("c2_trackman_residual", gain=-0.00004),
            _metric("c3_failure_aware", gain=0.0, status="skipped"),
        ],
        load_e1_contract(),
    )

    assert decision.status == "completed"
    assert decision.promoted == ("c0_native_ctr", "c1_anchor_residual")


def test_e1_decision_stops_when_every_candidate_regresses_too_much() -> None:
    decision = decide_e1(
        [
            _metric("c0_native_ctr", gain=-0.00006),
            _metric("c1_anchor_residual", gain=-0.00010),
        ],
        load_e1_contract(),
    )

    assert decision.status == "rejected"
    assert decision.promoted == ()
