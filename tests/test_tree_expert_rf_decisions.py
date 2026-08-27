from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from experiments.tree_expert.rf_contracts import load_rf_contract
from experiments.tree_expert.rf_decisions import (
    RFAcceptanceEvidence,
    RFDecisionError,
    RFStructureEvidence,
    accept_rf,
    route_probability,
    screen_structure_heads,
    select_rf_structure,
)


CONTRACT = load_rf_contract()


def test_route_probability_uses_only_current_row_game_type() -> None:
    base = np.array([0.2, 0.4, 0.6])
    r_probability = np.array([0.3, 0.5, 0.7])
    f_probability = np.array([0.1, 0.2, 0.8])
    game_type = np.array(["R", "F", "R"])

    actual = route_probability(
        base,
        r_probability,
        f_probability,
        game_type,
        alpha_r=0.5,
        alpha_f=0.25,
    )

    np.testing.assert_allclose(actual, [0.25, 0.35, 0.65])


def test_route_probability_is_permutation_invariant() -> None:
    base = np.array([0.2, 0.4, 0.6, 0.8])
    r_probability = np.array([0.3, 0.5, 0.7, 0.9])
    f_probability = np.array([0.1, 0.2, 0.8, 0.6])
    game_type = np.array(["R", "F", "R", "F"])
    order = np.array([2, 0, 3, 1])

    original = route_probability(base, r_probability, f_probability, game_type, 0.5, 0.75)
    shuffled = route_probability(
        base[order],
        r_probability[order],
        f_probability[order],
        game_type[order],
        0.5,
        0.75,
    )

    np.testing.assert_allclose(shuffled, original[order], atol=1e-12)


def test_route_probability_rejects_unknown_game_type() -> None:
    with pytest.raises(RFDecisionError, match="game_type differs"):
        route_probability(
            np.array([0.5]), np.array([0.5]), np.array([0.5]),
            np.array(["X"]), 0.5, 0.5,
        )


def _structure_evidence(confirmation_f_probability: float = 0.9) -> RFStructureEvidence:
    target = {}
    baseline = {}
    game_type = {}
    experts = {head: {} for head in ("f_small", "f_wide", "r_expert")}
    for fold in CONTRACT.folds:
        y = np.array([0.0, 1.0, 0.0, 1.0])
        base = np.array([0.4, 0.6, 0.4, 0.6])
        segment = np.array(["R", "F", "R", "F"])
        f_small = np.array([0.4, 0.9, 0.4, 0.9])
        if fold == CONTRACT.folds[-1]:
            f_small[[1, 3]] = confirmation_f_probability
        target[fold] = y
        baseline[fold] = base
        game_type[fold] = segment
        experts["f_small"][fold] = f_small
        experts["f_wide"][fold] = np.array([0.4, 0.3, 0.4, 0.3])
        experts["r_expert"][fold] = base.copy()
    return RFStructureEvidence(
        target=MappingProxyType(target),
        baseline=MappingProxyType(baseline),
        game_type=MappingProxyType(game_type),
        expert=MappingProxyType({key: MappingProxyType(value) for key, value in experts.items()}),
    )


def test_structure_selection_never_reads_confirmation_fold() -> None:
    first = select_rf_structure(_structure_evidence(confirmation_f_probability=0.0), CONTRACT)
    second = select_rf_structure(_structure_evidence(confirmation_f_probability=1.0), CONTRACT)

    assert first.selection_key == second.selection_key
    assert (first.f_head, first.alpha_r, first.alpha_f) == (
        second.f_head, second.alpha_r, second.alpha_f,
    )


def test_tie_break_prefers_small_f_then_lower_alpha_sum_then_f_only() -> None:
    evidence = _structure_evidence()
    tied = RFStructureEvidence(
        target=evidence.target,
        baseline=evidence.baseline,
        game_type=evidence.game_type,
        expert=MappingProxyType(
            {
                head: MappingProxyType({fold: evidence.baseline[fold].copy() for fold in CONTRACT.folds})
                for head in ("f_small", "f_wide", "r_expert")
            }
        ),
    )

    decision = select_rf_structure(tied, CONTRACT)

    assert decision.f_head == "f_small"
    assert decision.alpha_r == 0.0
    assert decision.alpha_f == 0.25


def test_head_screen_keeps_shrinkage_improvement_and_drops_uniform_failure() -> None:
    evidence = _structure_evidence()

    result = screen_structure_heads(evidence, CONTRACT)

    assert "f_small" in result
    assert "f_wide" not in result


def _acceptance_evidence(
    *,
    recent_f_same: bool = False,
    only_one_improved_fold: bool = False,
    regress_f: bool = False,
) -> RFAcceptanceEvidence:
    structure = select_rf_structure(_structure_evidence(), CONTRACT)
    target = {}
    baseline = {}
    game_type = {}
    seed_candidate = {seed: {} for seed in (42, 2026, 3407)}
    for index, fold in enumerate(CONTRACT.folds):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        base = np.array([0.4, 0.6, 0.4, 0.6])
        candidate = np.array([0.3, 0.7, 0.3, 0.7])
        if only_one_improved_fold and index > 0:
            candidate = base.copy()
        if recent_f_same and fold == CONTRACT.folds[-1]:
            candidate[[1, 3]] = base[[1, 3]]
        if regress_f:
            candidate[[1, 3]] = 0.2
        target[fold] = y
        baseline[fold] = base
        game_type[fold] = np.array(["R", "F", "R", "F"])
        for seed in seed_candidate:
            seed_candidate[seed][fold] = candidate.copy()
    return RFAcceptanceEvidence(
        structure=structure,
        target=MappingProxyType(target),
        baseline=MappingProxyType(baseline),
        game_type=MappingProxyType(game_type),
        seed_candidate=MappingProxyType(
            {seed: MappingProxyType(value) for seed, value in seed_candidate.items()}
        ),
    )


def test_acceptance_passes_all_fixed_gates() -> None:
    assert accept_rf(_acceptance_evidence(), CONTRACT).status == "accepted"


def test_acceptance_requires_recent_f_improvement() -> None:
    decision = accept_rf(_acceptance_evidence(recent_f_same=True), CONTRACT)
    assert decision.status == "rejected"
    assert decision.reason == "recent_f_gain_failed"


def test_acceptance_requires_two_improved_folds() -> None:
    decision = accept_rf(_acceptance_evidence(only_one_improved_fold=True), CONTRACT)
    assert decision.status == "rejected"
    assert decision.reason == "improved_fold_count_failed"


def test_acceptance_rejects_segment_regression() -> None:
    decision = accept_rf(_acceptance_evidence(regress_f=True), CONTRACT)
    assert decision.status == "rejected"
    assert decision.reason == "segment_regression_failed"
