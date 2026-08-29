from dataclasses import replace

import numpy as np
import pandas as pd

from experiments.tree_expert.s4_contracts import load_s4_contract
from experiments.tree_expert.s4_decisions import (
    AnchorEvidence,
    S4Evidence,
    decide_submission_eligibility,
    evaluate_full_chain,
    full_chain_archetypes,
    select_anchor_coverage,
)


def _anchors():
    rng = np.random.default_rng(3)
    return tuple(
        AnchorEvidence(
            candidate_id=name,
            mandatory_role=role,
            weighted_gain=0.0001 + index * 0.00001,
            fold_gains={(2021, 2022): 0.0001 - index * 1e-6, (2022, 2023): 0.00008 + index * 1e-6},
            rf_gain=0.00005 + (5 - index) * 1e-6,
            residual_signature=tuple(rng.normal(size=12) + index),
        )
        for index, (name, role) in enumerate((
            ("e2", "e2_control"), ("external", "external_template"),
            ("a", None), ("b", None), ("c", None), ("d", None),
        ))
    )


def test_anchor_coverage_keeps_all_six_roles():
    selected = select_anchor_coverage(_anchors())
    assert {item.role for item in selected} == {
        "e2_control", "external_template", "best_weighted",
        "best_worst_fold", "most_diverse", "best_rf",
    }
    assert len({item.candidate_id for item in selected}) == 6


def test_full_chain_grid_contains_at_least_twelve_archetypes():
    candidates = full_chain_archetypes(load_s4_contract(), select_anchor_coverage(_anchors()))
    assert len(candidates) >= 12
    assert any(item.anchor_role == "external_template" for item in candidates)
    assert any(item.residual_family == "catboost_rf" for item in candidates)
    assert {item.calibration_profile for item in candidates} >= {"pitcher", "batter", "matchup"}


def _passing() -> S4Evidence:
    return S4Evidence(
        candidate_id="c", confirmed=True, weighted_gain=0.0002,
        recent_gain=0.0001, minimum_fold_gain=0.0,
        maximum_segment_regression=0.0, bootstrap_lower=0.00001,
        bootstrap_upper=0.0003, non_worse_seed_count=3,
        residual_correlation=0.5, calibration_gap=0.01, ece=0.02,
    )


def test_acceptance_requires_every_registered_gate():
    assert decide_submission_eligibility(_passing()).status == "accepted"
    failures = {
        "weighted_gain": 0.0,
        "recent_gain": -1e-6,
        "minimum_fold_gain": -0.000031,
        "maximum_segment_regression": 0.000301,
        "bootstrap_lower": -1e-8,
        "non_worse_seed_count": 1,
    }
    for field, value in failures.items():
        assert decide_submission_eligibility(replace(_passing(), **{field: value})).status == "rejected"


def test_unconfirmed_candidate_is_research_only():
    decision = decide_submission_eligibility(replace(_passing(), confirmed=False))
    assert decision.status == "research_only"


def test_evaluation_aligns_rows_and_computes_all_registered_evidence():
    frames = {}
    for fold in load_s4_contract().folds:
        target = np.tile([0, 1], 60)
        base = np.where(target == 1, 0.55, 0.45)
        candidate = np.where(target == 1, 0.60, 0.40)
        frames[fold] = pd.DataFrame({
            "row_id": [f"{fold[1]}_{i}" for i in range(len(target))],
            "target": target, "p_base": base, "p_candidate": candidate,
            "pitcher_id": [f"p{i % 8}" for i in range(len(target))],
            "game_type": np.where(np.arange(len(target)) % 4, "R", "F"),
        })
    evidence = evaluate_full_chain("c", frames, confirmed=True, non_worse_seed_count=3)
    assert evidence.weighted_gain > 0
    assert evidence.recent_gain > 0
    assert evidence.maximum_segment_regression == 0
