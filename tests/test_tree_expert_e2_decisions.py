from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.e2_contracts import load_e2_contract
from experiments.tree_expert.e2_decisions import (
    E2DecisionError,
    FoldScore,
    SeedEvidence,
    StructureEvidence,
    blend_probabilities,
    causal_blend_decision,
    decide_acceptance,
    decide_seeds,
    decide_structure,
    evaluate_fold,
)
from experiments.tree_expert.inputs import PREDICTION_COLUMNS


FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))


def _contract(**gates: float | int):
    contract = load_e2_contract()
    changed = dict(contract.gates)
    changed.update(gates)
    return replace(contract, gates=MappingProxyType(changed))


def _score(
    fold: tuple[int, int],
    gain: float,
    *,
    rows: int = 100,
    best_iteration: int = 10,
) -> FoldScore:
    return FoldScore(
        fold=fold,
        row_count=rows,
        baseline_brier=0.25,
        candidate_brier=0.25 - gain,
        gain=gain,
        best_iteration=best_iteration,
    )


def _frame(probability: tuple[float, ...], target: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    size = len(probability)
    return pd.DataFrame(
        {
            "row_id": [f"r{index}" for index in range(size)],
            "target": target,
            "probability": probability,
            "game_type": ["R"] * size,
            "game_month": [4] * size,
            "pitcher_id_known": ["known"] * size,
            "batter_id_known": ["known"] * size,
        }
    ).loc[:, PREDICTION_COLUMNS]


def test_structure_tie_prefers_c1() -> None:
    c1 = StructureEvidence(
        "c1_anchor_residual",
        tuple(_score(fold, 0.0002) for fold in FOLDS),
    )
    c2 = StructureEvidence(
        "c2_trackman_residual",
        tuple(_score(fold, 0.0002) for fold in FOLDS),
    )

    decision = decide_structure((c2, c1), _contract())

    assert decision.status == "passed"
    assert decision.selected == "c1_anchor_residual"


@pytest.mark.parametrize(
    ("gains", "reason"),
    [
        ((0.00009, 0.00009, 0.00009), "weighted_gain_below_gate"),
        ((0.0002, 0.0002, 0.00009), "f3_gain_below_gate"),
        ((-0.000101, 0.0003, 0.0003), "fold_regression_above_gate"),
    ],
)
def test_structure_gate_rejects_each_failed_boundary(
    gains: tuple[float, float, float],
    reason: str,
) -> None:
    evidence = StructureEvidence(
        "c1_anchor_residual",
        tuple(_score(fold, gain) for fold, gain in zip(FOLDS, gains, strict=True)),
    )

    decision = decide_structure((evidence,), _contract())

    assert decision.status == "rejected"
    assert decision.reason == reason


def test_seed_gate_requires_two_non_worse_seeds_per_fold() -> None:
    seeds = (
        SeedEvidence(42, (_score(FOLDS[0], 0.0002), _score(FOLDS[1], -0.00001), _score(FOLDS[2], 0.0002))),
        SeedEvidence(2026, (_score(FOLDS[0], 0.0002), _score(FOLDS[1], -0.00002), _score(FOLDS[2], 0.0002))),
        SeedEvidence(3407, tuple(_score(fold, 0.0002) for fold in FOLDS)),
    )
    ensemble = tuple(_score(fold, 0.0002) for fold in FOLDS)

    decision = decide_seeds("c1_anchor_residual", seeds, ensemble, _contract())

    assert decision.status == "rejected"
    assert decision.reason == "fewer_than_two_non_worse_seeds"


def test_seed_gate_rejects_one_weighted_regression_over_limit() -> None:
    seeds = (
        SeedEvidence(42, tuple(_score(fold, -0.000101) for fold in FOLDS)),
        SeedEvidence(2026, tuple(_score(fold, 0.0002) for fold in FOLDS)),
        SeedEvidence(3407, tuple(_score(fold, 0.0002) for fold in FOLDS)),
    )

    decision = decide_seeds(
        "c1_anchor_residual",
        seeds,
        tuple(_score(fold, 0.0001) for fold in FOLDS),
        _contract(),
    )

    assert decision.status == "rejected"
    assert decision.reason == "seed_weighted_regression"


def test_probability_and_logit_blends_match_fixed_formulas() -> None:
    tabm = np.asarray([0.2, 0.8])
    cat = np.asarray([0.4, 0.6])

    probability = blend_probabilities(tabm, cat, "probability", 0.3)
    logit = blend_probabilities(tabm, cat, "logit", 0.3)

    assert np.allclose(probability, 0.7 * tabm + 0.3 * cat)
    expected_logit = 1 / (
        1
        + np.exp(
            -(0.7 * np.log(tabm / (1 - tabm)) + 0.3 * np.log(cat / (1 - cat)))
        )
    )
    assert np.allclose(logit, expected_logit)


def test_causal_blend_never_uses_f3_to_select_its_f3_candidate() -> None:
    target = (0, 1)
    tabm = {
        FOLDS[0]: _frame((0.45, 0.55), target),
        FOLDS[1]: _frame((0.45, 0.55), target),
        FOLDS[2]: _frame((0.10, 0.90), target),
    }
    cat = {
        FOLDS[0]: _frame((0.10, 0.90), target),
        FOLDS[1]: _frame((0.10, 0.90), target),
        FOLDS[2]: _frame((0.45, 0.55), target),
    }

    decision = causal_blend_decision(tabm, cat, _contract())

    assert decision.f2_selection_fold == FOLDS[0]
    assert decision.f3_selection_folds == FOLDS[:2]
    assert decision.f3_applied_method != "probability" or decision.f3_catboost_weight != 0.0


def test_blend_gate_falls_back_to_catboost_when_sequential_gain_is_too_small() -> None:
    tabm = {fold: _frame((0.2, 0.8)) for fold in FOLDS}
    cat = {fold: _frame((0.2, 0.8)) for fold in FOLDS}

    decision = causal_blend_decision(tabm, cat, _contract())

    assert decision.status == "rejected"
    assert decision.predictor == "catboost"
    assert decision.reason == "sequential_gain_below_gate"


def test_acceptance_uses_standalone_catboost_and_positive_pitcher_bootstrap() -> None:
    baselines = {fold: _frame((0.45, 0.55)) for fold in FOLDS}
    candidates = {fold: _frame((0.35, 0.65)) for fold in FOLDS}
    scores = tuple(
        evaluate_fold(baselines[fold], candidates[fold], fold=fold, best_iteration=10)
        for fold in FOLDS
    )
    groups = {fold: ("p1", "p2") for fold in FOLDS}
    contract = _contract(
        accept_weighted_gain=0.0,
        accept_f3_gain=0.0,
        bootstrap_repeats=100,
        minimum_segment_rows=1,
    )

    decision = decide_acceptance(
        scores,
        baselines,
        candidates,
        groups,
        contract,
        blend=None,
    )

    assert decision.status == "accepted"
    assert decision.predictor == "catboost"
    assert decision.bootstrap_lower > 0
    assert decision.maximum_segment_regression == 0


def test_acceptance_rejects_misaligned_rows() -> None:
    baseline = _frame((0.45, 0.55))
    candidate = _frame((0.35, 0.65)).iloc[::-1].reset_index(drop=True)

    with pytest.raises(E2DecisionError, match="row_id alignment"):
        evaluate_fold(baseline, candidate, fold=FOLDS[0], best_iteration=1)
