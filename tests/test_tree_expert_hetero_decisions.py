from types import MappingProxyType

import numpy as np
import pandas as pd

from experiments.tree_expert.hetero_contracts import load_hetero_contract
from experiments.tree_expert.hetero_decisions import (
    FamilyEvidence,
    confirm_family,
    corrected_probability,
    decide_equal_blend,
    select_family_structure,
)


FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))


def _frame(model_scale=0.9):
    target = np.tile([0, 1], 3000)
    base = np.full(len(target), 0.5)
    variation = 0.18 * np.sin(np.arange(len(target)) * 0.37)
    direction = np.where(target == 1, 1.0, -1.0)
    model = np.clip(np.where(target == 1, model_scale, 1 - model_scale) + variation, 0.01, 0.99)
    return pd.DataFrame({
        "row_id": [f"r{i}" for i in range(len(target))], "target": target,
        "baseline_probability": base, "model_probability": model,
        "game_type": np.where(np.arange(len(target)) % 2, "R", "F"),
    })


def test_correction_is_bounded_and_uses_registered_weight():
    output = corrected_probability(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.1)
    np.testing.assert_allclose(output, [0.1, 0.9])


def test_structure_selects_weight_without_using_confirmation_fold():
    frames = {fold: _frame(0.9) for fold in FOLDS}
    first = select_family_structure(FamilyEvidence(MappingProxyType(frames)), load_hetero_contract(), "xgboost")
    frames[FOLDS[-1]] = _frame(0.1)
    second = select_family_structure(FamilyEvidence(MappingProxyType(frames)), load_hetero_contract(), "xgboost")
    assert first.weight == second.weight == 0.15
    assert first.status == "passed"
    assert second.status == "rejected"


def test_confirmation_averages_three_seeds_and_requires_two_non_worse():
    structure_frames = {fold: _frame(0.9) for fold in FOLDS}
    structure = select_family_structure(
        FamilyEvidence(MappingProxyType(structure_frames)), load_hetero_contract(), "lightgbm",
    )
    seed_frames = {
        seed: MappingProxyType({fold: _frame(0.9) for fold in FOLDS})
        for seed in (3407, 42, 2026)
    }
    decision = confirm_family(structure, MappingProxyType(seed_frames), load_hetero_contract())
    assert decision.status == "accepted"
    assert decision.non_worse_seed_count == 3
    blend = decide_equal_blend(
        decision, decision, seed_frames[3407], seed_frames[3407], load_hetero_contract(),
    )
    assert blend.status == "rejected"
    assert blend.reason == "incremental_gain_failed"
