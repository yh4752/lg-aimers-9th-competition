from types import MappingProxyType

import numpy as np
import pytest

from experiments.tree_expert.t3_contracts import load_t3_contract
from experiments.tree_expert.t3_decisions import (
    T3AcceptanceEvidence,
    T3StructureEvidence,
    accept_t3,
    blended_probability,
    select_structure,
)


FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))


def structure_evidence(f3_override=0.75):
    target = {fold: np.array([0.0, 1.0, 0.0, 1.0]) for fold in FOLDS}
    baseline = {fold: np.full(4, 0.5) for fold in FOLDS}
    recent = {fold: np.array([0.15, 0.85, 0.2, 0.8]) for fold in FOLDS}
    multi = {}
    for decay in (0.35, 0.55, 0.75):
        value = 0.25 if decay == 0.55 else 0.45
        multi[decay] = {fold: np.array([value, 1 - value, value, 1 - value]) for fold in FOLDS}
    multi[0.55][FOLDS[-1]] = np.array([f3_override, 1 - f3_override, f3_override, 1 - f3_override])
    return T3StructureEvidence(
        target=MappingProxyType(target), baseline=MappingProxyType(baseline),
        recent=MappingProxyType(recent),
        multi=MappingProxyType({key: MappingProxyType(value) for key, value in multi.items()}),
        maximum_segment_regression=0.0,
    )


def test_structure_selection_ignores_f3_when_choosing_parameters():
    first = select_structure(structure_evidence(0.1), load_t3_contract())
    second = select_structure(structure_evidence(0.9), load_t3_contract())
    assert (first.decay, first.recent_weight) == (second.decay, second.recent_weight)
    assert first.decay == 0.55
    assert first.selection_folds == FOLDS[:2]
    assert first.confirmation_fold == FOLDS[-1]


def test_blended_probability_rejects_unregistered_weight():
    try:
        blended_probability(np.array([0.2]), np.array([0.3]), 0.65)
    except ValueError as error:
        assert "recent weight" in str(error)
    else:
        raise AssertionError("unregistered weight was accepted")


def test_acceptance_requires_two_non_worse_seeds_and_recent_fold_gain():
    structure = select_structure(structure_evidence(0.25), load_t3_contract())
    gains = {
        3407: {fold: 0.0003 for fold in FOLDS},
        42: {fold: 0.0002 for fold in FOLDS},
        2026: {fold: 0.0002 for fold in FOLDS},
    }
    accepted = accept_t3(
        T3AcceptanceEvidence(
            structure,
            MappingProxyType({k: MappingProxyType(v) for k, v in gains.items()}),
            MappingProxyType({fold: 4 for fold in FOLDS}),
        ),
        load_t3_contract(),
    )
    assert accepted.status == "accepted"
    gains[42][FOLDS[-1]] = -0.01
    gains[2026][FOLDS[-1]] = -0.01
    rejected = accept_t3(
        T3AcceptanceEvidence(
            structure,
            MappingProxyType({k: MappingProxyType(v) for k, v in gains.items()}),
            MappingProxyType({fold: 4 for fold in FOLDS}),
        ),
        load_t3_contract(),
    )
    assert rejected.status == "rejected"


def test_acceptance_weighted_gain_uses_fold_row_counts():
    structure = select_structure(structure_evidence(0.25), load_t3_contract())
    per_fold = {FOLDS[0]: 0.0003, FOLDS[1]: 0.0001, FOLDS[2]: 0.0001}
    gains = {seed: MappingProxyType(dict(per_fold)) for seed in (3407, 42, 2026)}
    decision = accept_t3(
        T3AcceptanceEvidence(
            structure,
            MappingProxyType(gains),
            MappingProxyType({FOLDS[0]: 100, FOLDS[1]: 1, FOLDS[2]: 1}),
        ),
        load_t3_contract(),
    )
    assert decision.weighted_gain == pytest.approx((0.0003 * 100 + 0.0001 * 2) / 102)
