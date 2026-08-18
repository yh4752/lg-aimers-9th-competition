from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from experiments.hierarchical_tabm.contracts import load_contract
from experiments.hierarchical_tabm.metrics import (
    HierarchicalMetricError,
    align_anchor_and_h1,
    brier,
    build_segment_columns,
    candidate_decision_payload,
    decide_calibrated,
    decide_h1,
    paired_fold_metrics,
)


def _prediction(ids=("r1", "r2", "r3"), probability=(0.2, 0.8, 0.4)):
    return pd.DataFrame(
        {
            "row_id": ids,
            "target": [0, 1, 0],
            "probability": probability,
            "game_type": ["R", "R", "P"],
            "game_month": [4, 8, 9],
            "count_state": ["0_0", "1_1", "3_2"],
            "hand_matchup": ["R_L", "L_R", "R_R"],
            "base_out_state": ["000_0", "100_1", "111_2"],
            "pitcher_known": ["known", "known", "oov"],
            "batter_known": ["known", "oov", "known"],
        }
    )


def test_aligns_only_exact_row_id_and_target_sets() -> None:
    anchor = _prediction()
    h1 = _prediction(ids=("r3", "r1", "r2"), probability=(0.3, 0.1, 0.7))
    h1["target"] = [0, 0, 1]
    aligned = align_anchor_and_h1(anchor, h1)
    assert aligned["row_id"].tolist() == anchor["row_id"].tolist()
    assert aligned["candidate_probability"].tolist() == [0.1, 0.7, 0.3]
    with pytest.raises(HierarchicalMetricError, match="target differs"):
        align_anchor_and_h1(anchor, h1.assign(target=1 - h1.target))
    with pytest.raises(HierarchicalMetricError, match="row_id set differs"):
        align_anchor_and_h1(anchor, h1.assign(row_id=["x", "r1", "r2"]))


def test_builds_exact_preregistered_segments() -> None:
    frame = pd.DataFrame(
        {
            "row_id": ["r1", "r2"],
            "game_type": ["R", "P"],
            "balls_before": [0, 3],
            "strikes_before": [0, 2],
            "pitcher_hand": ["R", "L"],
            "batter_hand": ["L", "R"],
            "base_state": ["000", "101"],
            "outs_before": [0, 2],
            "pitcher_id": ["p1", "new"],
            "batter_id": ["new", "b1"],
        },
        index=[7, 3],
    )
    labels = build_segment_columns(
        frame, fit_pitcher_ids={"p1"}, fit_batter_ids={"b1"}
    )
    assert tuple(labels) == (
        "game_type", "count_state", "hand_matchup", "base_out_state",
        "pitcher_known", "batter_known",
    )
    assert labels.index.equals(frame.index)
    assert labels.loc[7].tolist() == ["R", "0_0", "R_L", "000_0", "known", "oov"]
    reversed_labels = build_segment_columns(
        frame.iloc[::-1], fit_pitcher_ids={"p1"}, fit_batter_ids={"b1"}
    )
    assert reversed_labels.loc[7].equals(labels.loc[7])


def test_brier_is_strict_float64_mean() -> None:
    assert brier([0, 1], [0.2, 0.7]) == pytest.approx((0.04 + 0.09) / 2)
    with pytest.raises(HierarchicalMetricError):
        brier([0, 2], [0.2, 0.7])


def test_paired_metrics_are_row_weighted_and_exclude_small_hard_segments() -> None:
    anchor = {
        "2022->2023": _prediction(),
        "2023->2024": _prediction(ids=("a", "b", "c")),
    }
    candidate = {
        fold: frame.assign(probability=[0.1, 0.9, 0.3])
        for fold, frame in anchor.items()
    }
    metrics = paired_fold_metrics(anchor, candidate, segment_min_rows=2)
    expected = np.mean([0.01, 0.01, 0.09])
    assert metrics["fold_brier"]["2022->2023"] == pytest.approx(expected)
    assert metrics["weighted_gain_vs_anchor"] == pytest.approx(
        brier(anchor["2022->2023"].target, anchor["2022->2023"].probability)
        - expected
    )
    assert any(not row["eligible"] for row in metrics["segments"])
    assert "ece_10" in metrics
    assert "monthly" in metrics


def _metrics(
    *,
    weighted_gain=0.0001,
    latest_gain=0.00005,
    old_regression=0.00015,
    worst_segment_regression=0.00075,
    status="completed",
):
    anchor = {"2022->2023": 0.25, "2023->2024": 0.25}
    candidate = {
        "2022->2023": 0.25 + old_regression,
        "2023->2024": 0.25 - latest_gain,
    }
    return {
        "status": status,
        "fold_brier": candidate,
        "anchor_fold_brier": anchor,
        "fold_gain_vs_anchor": {
            "2022->2023": -old_regression,
            "2023->2024": latest_gain,
        },
        "fold_rows": {"2022->2023": 1, "2023->2024": 1},
        "weighted_gain_vs_anchor": weighted_gain,
        "worst_eligible_segment_regression": worst_segment_regression,
        "segments": [],
    }


def test_h1_strong_exact_boundaries_pass() -> None:
    decision = decide_h1(_metrics(), load_contract())
    assert decision.status == "strong"
    assert decision.final_acceptance is True
    assert decision.delivery_role == "final_candidate"


@pytest.mark.parametrize(
    "change",
    [
        {"weighted_gain": 0.000099999},
        {"latest_gain": 0.000049999},
        {"old_regression": 0.000150001},
        {"worst_segment_regression": 0.000750001},
    ],
)
def test_h1_strong_fails_just_outside_each_boundary(change) -> None:
    decision = decide_h1(_metrics(**change), load_contract())
    assert decision.status != "strong"


def test_h1_frontier_is_not_final_acceptance() -> None:
    decision = decide_h1(
        _metrics(
            weighted_gain=0.00001,
            latest_gain=0.00001,
            old_regression=0.00020,
            worst_segment_regression=0.00010,
        ),
        load_contract(),
    )
    assert decision.status == "frontier"
    assert decision.final_acceptance is False
    assert decision.delivery_role == "public_diagnostic_only"


def test_h1_nonpositive_independent_gain_is_rejected() -> None:
    decision = decide_h1(
        _metrics(weighted_gain=0.0001, latest_gain=0.0, old_regression=0.0),
        load_contract(),
    )
    assert decision.status == "rejected"


def _calibrated(
    *, latest_vs_h1=0.00003, latest_vs_anchor=0.00008,
    segment_vs_h1=0.0002, status="completed",
):
    h1 = _metrics(
        weighted_gain=0.00001,
        latest_gain=latest_vs_anchor - latest_vs_h1,
        old_regression=0,
    )
    calibrated = deepcopy(h1)
    calibrated["status"] = status
    calibrated["fold_brier"]["2023->2024"] = 0.25 - latest_vs_anchor
    calibrated["fold_gain_vs_anchor"]["2023->2024"] = latest_vs_anchor
    calibrated["worst_eligible_segment_regression_vs_h1"] = segment_vs_h1
    return h1, calibrated


def test_h2_can_pass_when_raw_h1_is_not_strong_at_exact_boundaries() -> None:
    h1, calibrated = _calibrated()
    assert decide_h1(h1, load_contract()).status != "strong"
    decision = decide_calibrated("H2", h1, calibrated, load_contract())
    assert decision.status == "accepted"
    assert decision.final_acceptance is True


@pytest.mark.parametrize(
    "change",
    [
        {"latest_vs_h1": 0.000029999},
        {"latest_vs_anchor": 0.000079999},
        {"segment_vs_h1": 0.000200001},
    ],
)
def test_h2_fails_just_outside_each_boundary(change) -> None:
    h1, calibrated = _calibrated(**change)
    assert decide_calibrated("H2", h1, calibrated, load_contract()).status == "rejected"


def test_h3_requires_material_improvement_over_h2() -> None:
    h1, h3 = _calibrated(latest_vs_h1=0.00005, latest_vs_anchor=0.0001)
    h2 = deepcopy(h3)
    h2["fold_brier"]["2023->2024"] = h3["fold_brier"]["2023->2024"] + 0.00002
    assert decide_calibrated(
        "H3", h1, h3, load_contract(), h2_metrics=h2
    ).status == "accepted"
    h2["fold_brier"]["2023->2024"] = h3["fold_brier"]["2023->2024"] + 0.000019999
    assert decide_calibrated(
        "H3", h1, h3, load_contract(), h2_metrics=h2
    ).status == "rejected"


def test_incomplete_evidence_propagates_without_acceptance() -> None:
    h1, calibrated = _calibrated(status="incomplete")
    decision = decide_calibrated("H2", h1, calibrated, load_contract())
    assert decision.status == "incomplete"
    assert decision.final_acceptance is False
    assert decide_h1(_metrics(status="incomplete"), load_contract()).status == "incomplete"


def test_decision_payload_has_exact_stable_fields() -> None:
    payload = candidate_decision_payload(decide_h1(_metrics(), load_contract()))
    assert tuple(payload) == (
        "candidate_id", "status", "final_acceptance", "delivery_role",
        "fold_brier", "fold_gain_vs_anchor", "weighted_gain_vs_anchor",
        "worst_eligible_segment_regression", "reason",
    )
