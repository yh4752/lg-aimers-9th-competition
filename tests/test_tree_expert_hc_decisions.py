from dataclasses import replace

import pandas as pd
import pytest

from experiments.tree_expert.hc_contracts import load_hc_contract
from experiments.tree_expert.hc_decisions import (
    C1Evidence,
    C2Evidence,
    decide_c1,
    decide_c2,
    choose_winner,
    select_profile,
)


def _c1(**changes):
    evidence = C1Evidence(
        weighted_brier=0.2480,
        weighted_gain=0.00020,
        confirmation_gain=0.00005,
        minimum_fold_gain=0.00001,
        maximum_segment_regression=0.00010,
        bootstrap_lower_95=0.00001,
        non_worse_seed_counts={2022: 3, 2023: 2, 2024: 2},
    )
    return replace(evidence, **changes)


def _c2(**changes):
    evidence = C2Evidence(
        weighted_brier=0.2478,
        weighted_gain=0.00025,
        incremental_gain=0.00005,
        minimum_fold_gain=0.00001,
        c0_calibration_gap=0.01,
        candidate_calibration_gap=0.009,
        c0_ece=0.02,
        candidate_ece=0.019,
        maximum_segment_regression=0.00010,
        bootstrap_lower_95=0.00001,
    )
    return replace(evidence, **changes)


@pytest.mark.parametrize(
    ("change", "gate"),
    [
        ({"weighted_gain": 0.00009}, "weighted_gain"),
        ({"confirmation_gain": 0.0}, "confirmation_gain"),
        ({"minimum_fold_gain": -0.00006}, "minimum_fold_gain"),
        ({"maximum_segment_regression": 0.00036}, "segment_regression"),
        ({"bootstrap_lower_95": 0.0}, "bootstrap_lower"),
        ({"non_worse_seed_counts": {2022: 2, 2023: 1, 2024: 2}}, "seed_consistency"),
    ],
)
def test_each_c1_gate_is_mandatory(change, gate):
    decision = decide_c1(_c1(**change), load_hc_contract())
    assert decision.status == "rejected"
    assert gate in decision.failed_gates


def test_c2_can_pass_independently_when_c1_fails():
    c1 = decide_c1(_c1(weighted_gain=-1.0), load_hc_contract())
    c2 = decide_c2(_c2(), load_hc_contract())
    winner = choose_winner(c1, c2, load_hc_contract())
    assert c1.status == "rejected"
    assert c2.status == "accepted"
    assert winner.candidate == "C2"


@pytest.mark.parametrize(
    ("change", "gate"),
    [
        ({"weighted_gain": 0.00012}, "weighted_gain"),
        ({"incremental_gain": 0.00002}, "incremental_gain"),
        ({"minimum_fold_gain": -0.00004}, "minimum_fold_gain"),
        ({"candidate_calibration_gap": 0.011}, "calibration_gap"),
        ({"candidate_ece": 0.021}, "ece"),
        ({"maximum_segment_regression": 0.00021}, "segment_regression"),
        ({"bootstrap_lower_95": 0.0}, "bootstrap_lower"),
    ],
)
def test_each_c2_gate_is_mandatory(change, gate):
    decision = decide_c2(_c2(**change), load_hc_contract())
    assert decision.status == "rejected"
    assert gate in decision.failed_gates


def test_simpler_c1_wins_within_registered_tie_margin():
    c1 = decide_c1(_c1(weighted_brier=0.24800), load_hc_contract())
    c2 = decide_c2(_c2(weighted_brier=0.24799), load_hc_contract())
    winner = choose_winner(c1, c2, load_hc_contract())
    assert winner.candidate == "C1"
    assert winner.reason == "simpler_within_tie_margin"


def test_lower_brier_wins_outside_tie_margin():
    c1 = decide_c1(_c1(weighted_brier=0.24800), load_hc_contract())
    c2 = decide_c2(_c2(weighted_brier=0.24790), load_hc_contract())
    assert choose_winner(c1, c2, load_hc_contract()).candidate == "C2"


def test_profile_selection_uses_only_structure_folds_and_registered_tie_order():
    predictions = {}
    for profile in ("hc_strong", "hc_balanced", "hc_light"):
        predictions[profile] = {}
        for year in (2022, 2023):
            predictions[profile][year] = pd.DataFrame(
                {"target": [0, 1], "p1": [0.4, 0.6]}
            )
    decision = select_profile(predictions, load_hc_contract())
    assert decision.selected == "hc_strong"
    assert decision.structure_years == (2022, 2023)

    predictions["hc_balanced"][2023].loc[:, "p1"] = [0.1, 0.9]
    decision = select_profile(predictions, load_hc_contract())
    assert decision.selected == "hc_balanced"
