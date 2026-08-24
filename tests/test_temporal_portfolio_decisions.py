from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from experiments.temporal_portfolio.decisions import (
    CandidateMetrics,
    CatBoostFoldMetrics,
    decide_candidate,
    decide_catboost_prefix,
    select_t2a_survivors,
)


def _metric(candidate_id: str = "c", family: str = "P1") -> CandidateMetrics:
    return CandidateMetrics(
        candidate_id=candidate_id,
        family=family,
        temporal_gain=Decimal("0.00030"),
        weighted_gain=Decimal("0.00006"),
        latest_gain=Decimal("0.00004"),
        bootstrap_lower=Decimal("0.00001"),
        max_segment_regression=Decimal("0.00040"),
        fold_regressions=(Decimal("0.00001"), Decimal("0.00002"), Decimal("0.00000")),
        improved_fold_count=3,
        worst_fold_regression=Decimal("0.00002"),
        latest_regression=Decimal("0.00001"),
    )


def test_champion_exploratory_and_rejected_are_distinct() -> None:
    champion = _metric()
    exploratory = replace(champion, bootstrap_lower=Decimal("-0.00001"))
    rejected = replace(exploratory, weighted_gain=Decimal("0.00001"), improved_fold_count=1)
    assert decide_candidate(champion).status == "champion"
    assert decide_candidate(exploratory).status == "exploratory"
    assert decide_candidate(rejected).status == "rejected"


def test_incomplete_and_missing_mapping_are_not_mislabeled_rejected() -> None:
    assert decide_candidate(replace(_metric(), evidence_complete=False)).status == "budget_inconclusive"
    assert decide_candidate(replace(_metric(), mapping_evidence="insufficient_mapping")).status == "insufficient_mapping"


def test_t2a_keeps_two_best_and_one_structural_wildcard() -> None:
    candidates = (
        _metric("p1", "P1"),
        replace(_metric("p2", "P2"), weighted_gain=Decimal("0.00005")),
        replace(_metric("b1", "B1"), weighted_gain=Decimal("0.00004")),
        replace(_metric("m1", "M1"), weighted_gain=Decimal("0.00003")),
    )
    promoted = select_t2a_survivors(candidates)
    assert len(promoted) <= 3
    assert any(item.family in {"B1", "M1"} for item in promoted)
    assert [item.candidate_id for item in promoted[:2]] == ["p1", "p2"]


def test_unstable_catboost_prefix_cannot_reach_delivery() -> None:
    decision = decide_catboost_prefix(
        CatBoostFoldMetrics("cat", (16, 384, 16))
    )
    assert decision.status == "unstable_for_deployment"
    assert decision.selected_prefix is None


def test_catboost_prefix_requires_complete_adjacent_fold_evidence() -> None:
    stable = decide_catboost_prefix(CatBoostFoldMetrics("cat", (16, 64, 16)))
    assert stable.status == "stable_for_deployment"
    assert stable.selected_prefix == 16
    incomplete = decide_catboost_prefix(
        CatBoostFoldMetrics("cat", (16,), evidence_complete=False)
    )
    assert incomplete.status == "budget_inconclusive"
    assert incomplete.selected_prefix is None
