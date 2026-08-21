from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from experiments.catboost_50_50_realign.contracts import load_contract
from experiments.catboost_50_50_realign.metrics import (
    CATBOOST_COLUMNS,
    TABM_COLUMNS,
    PrefixEvidence,
    RealignMetricError,
    decision_from_payload,
    decision_to_payload,
    evaluate_prefixes,
    passes_gates,
)
from experiments.oof_reset_audit.metrics import block_bootstrap_interval


FOLDS = ("2021->2022", "2022->2023", "2023->2024")


def _frames(rows: int = 12_000) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    target = np.arange(rows) % 2
    tabm_probability = np.where(target == 1, 0.60, 0.40)
    months = np.asarray([f"{month:02d}" for month in range(4, 10)])
    tabm: dict[str, pd.DataFrame] = {}
    catboost: dict[str, pd.DataFrame] = {}
    prefix_strength = {
        4: 0.68,
        8: 0.69,
        12: 0.70,
        16: 0.72,
        20: 0.71,
        24: 0.70,
        28: 0.69,
        32: 0.68,
    }
    for fold_index, fold in enumerate(FOLDS):
        row_id = np.asarray([f"{fold}:{index}" for index in range(rows)])
        common = {
            "row_id": row_id,
            "target": target,
            "game_type": np.where(np.arange(rows) % 2, "regular", "special"),
            "game_month": months[np.arange(rows) % len(months)],
            "pitcher_id_known": np.arange(rows) % 3 != 0,
            "batter_id_known": np.arange(rows) % 4 != 0,
        }
        tabm[fold] = pd.DataFrame(
            common | {"probability": tabm_probability}, columns=TABM_COLUMNS
        )
        candidate = dict(common)
        for tree_count, positive_probability in prefix_strength.items():
            adjusted = positive_probability - fold_index * 0.002
            candidate[f"p_{tree_count}"] = np.where(
                target == 1, adjusted, 1.0 - adjusted
            )
        catboost[fold] = pd.DataFrame(candidate, columns=CATBOOST_COLUMNS)
    return tabm, catboost


def _evidence(**changes: object) -> PrefixEvidence:
    base = PrefixEvidence(
        tree_count=16,
        fold_gain={fold: 0.0001 for fold in FOLDS},
        weighted_gain=0.00003,
        latest_bootstrap_lower=0.0,
        maximum_segment_regression=0.00075,
        eligible_segment_count=3,
        bootstrap_status="completed",
        passed=True,
    )
    return replace(base, **changes)


def test_evaluate_prefixes_promotes_best_minimum_fold_gain() -> None:
    tabm, catboost = _frames()

    decision = evaluate_prefixes(tabm, catboost, load_contract())

    assert decision.status == "promoted"
    assert decision.selected_tree_count == 16
    assert decision.selection_order == (
        "maximum_minimum_fold_gain",
        "maximum_weighted_gain",
        "minimum_tree_count",
    )
    assert len(decision.candidates) == 8
    assert all(candidate.passed for candidate in decision.candidates)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"fold_gain": {FOLDS[0]: 0.0, FOLDS[1]: 0.1, FOLDS[2]: 0.1}}, False),
        ({"weighted_gain": 0.000029999999999}, False),
        ({"weighted_gain": 0.00003}, True),
        ({"latest_bootstrap_lower": -1e-15}, False),
        ({"latest_bootstrap_lower": 0.0}, True),
        ({"maximum_segment_regression": 0.00075}, True),
        ({"maximum_segment_regression": 0.000750000000001}, False),
        ({"latest_bootstrap_lower": None}, False),
        ({"eligible_segment_count": 0}, False),
        ({"bootstrap_status": "insufficient_blocks"}, False),
        ({"weighted_gain": float("nan")}, False),
        ({"maximum_segment_regression": float("inf")}, False),
    ],
)
def test_gate_boundaries_are_exact(changes: dict[str, object], expected: bool) -> None:
    assert passes_gates(_evidence(**changes), load_contract()) is expected


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("reverse", None),
        ("missing_row", "row_id set differs"),
        ("duplicate_row", "row_id must be unique"),
        ("target", "target differs"),
        ("segment", "segment differs"),
        ("nan", "probability"),
        ("infinity", "probability"),
        ("out_of_range", "probability"),
        ("missing_fold", "fold set differs"),
        ("extra_fold", "fold set differs"),
    ],
)
def test_alignment_and_input_fail_closed(mutation: str, message: str | None) -> None:
    tabm, catboost = _frames()
    baseline = evaluate_prefixes(tabm, catboost, load_contract())
    if mutation == "reverse":
        catboost[FOLDS[1]] = catboost[FOLDS[1]].iloc[::-1].reset_index(drop=True)
    elif mutation == "missing_row":
        catboost[FOLDS[1]] = catboost[FOLDS[1]].iloc[:-1]
    elif mutation == "duplicate_row":
        catboost[FOLDS[1]].loc[1, "row_id"] = catboost[FOLDS[1]].loc[0, "row_id"]
    elif mutation == "target":
        catboost[FOLDS[1]].loc[0, "target"] = 1 - catboost[FOLDS[1]].loc[0, "target"]
    elif mutation == "segment":
        catboost[FOLDS[1]].loc[0, "game_type"] = "different"
    elif mutation == "nan":
        catboost[FOLDS[1]].loc[0, "p_16"] = np.nan
    elif mutation == "infinity":
        catboost[FOLDS[1]].loc[0, "p_16"] = np.inf
    elif mutation == "out_of_range":
        catboost[FOLDS[1]].loc[0, "p_16"] = 1.1
    elif mutation == "missing_fold":
        del catboost[FOLDS[0]]
    elif mutation == "extra_fold":
        catboost["2020->2021"] = catboost[FOLDS[0]].copy()
    if message is None:
        assert evaluate_prefixes(tabm, catboost, load_contract()) == baseline
    else:
        with pytest.raises(RealignMetricError, match=message):
            evaluate_prefixes(tabm, catboost, load_contract())


def test_exact_column_order_is_required() -> None:
    tabm, catboost = _frames()
    tabm[FOLDS[0]] = tabm[FOLDS[0]][list(reversed(TABM_COLUMNS))]
    with pytest.raises(RealignMetricError, match="TabM columns differ"):
        evaluate_prefixes(tabm, catboost, load_contract())


def test_decision_payload_round_trip_preserves_exact_decision() -> None:
    tabm, catboost = _frames()
    decision = evaluate_prefixes(tabm, catboost, load_contract())

    assert decision_from_payload(decision_to_payload(decision), load_contract()) == decision


def test_tie_within_tolerance_uses_weighted_gain_then_smaller_tree() -> None:
    from experiments.catboost_50_50_realign.metrics import select_candidate

    left = _evidence(tree_count=8, weighted_gain=0.00004)
    right = _evidence(
        tree_count=16,
        fold_gain={fold: 0.0001000000005 for fold in FOLDS},
        weighted_gain=0.00005,
    )
    assert select_candidate((left, right), load_contract()).tree_count == 16
    tied_weight = replace(right, weighted_gain=0.0000400000005)
    assert select_candidate((left, tied_weight), load_contract()).tree_count == 8


def test_gate_uses_decimal_threshold_not_binary_float_rounding() -> None:
    contract = load_contract()
    assert contract.minimum_weighted_gain == Decimal("0.00003")
    assert passes_gates(_evidence(weighted_gain=float("0.00003")), contract)


def test_latest_bootstrap_matches_audit_implementation() -> None:
    tabm, catboost = _frames()
    decision = evaluate_prefixes(tabm, catboost, load_contract())
    selected = next(item for item in decision.candidates if item.tree_count == 16)
    fold = FOLDS[-1]
    target = tabm[fold]["target"].to_numpy("float64")
    tabm_probability = tabm[fold]["probability"].to_numpy("float64")
    catboost_probability = catboost[fold]["p_16"].to_numpy("float64")
    blended = 0.5 * tabm_probability + 0.5 * catboost_probability
    loss_delta = np.square(tabm_probability - target) - np.square(blended - target)
    expected = block_bootstrap_interval(
        pd.DataFrame(
            {
                "block": [
                    f"{fold}:{value}" for value in tabm[fold]["game_month"].astype(str)
                ],
                "loss_delta": loss_delta,
            }
        )
    )
    assert selected.latest_bootstrap_lower == expected["lower"]


def test_insufficient_latest_blocks_rejects_every_prefix() -> None:
    tabm, catboost = _frames()
    replacement = np.asarray([f"0{value}" for value in range(1, 6)])
    for frames in (tabm, catboost):
        frames[FOLDS[-1]]["game_month"] = replacement[
            np.arange(len(frames[FOLDS[-1]])) % len(replacement)
        ]
    decision = evaluate_prefixes(tabm, catboost, load_contract())
    assert decision.status == "rejected"
    assert all(item.bootstrap_status == "insufficient_blocks" for item in decision.candidates)


def test_no_eligible_segment_rejects_every_prefix() -> None:
    tabm, catboost = _frames(rows=4_800)
    decision = evaluate_prefixes(tabm, catboost, load_contract())
    assert decision.status == "rejected"
    assert all(item.eligible_segment_count == 0 for item in decision.candidates)


def test_payload_rejects_nonfinite_metric() -> None:
    tabm, catboost = _frames()
    payload = decision_to_payload(evaluate_prefixes(tabm, catboost, load_contract()))
    payload["candidates"][0]["weighted_gain"] = float("nan")
    with pytest.raises(RealignMetricError, match="finite"):
        decision_from_payload(payload, load_contract())


def test_payload_rejects_inconsistent_rejected_status() -> None:
    tabm, catboost = _frames()
    payload = decision_to_payload(evaluate_prefixes(tabm, catboost, load_contract()))
    payload["status"] = "rejected"
    payload["selected_tree_count"] = None
    with pytest.raises(RealignMetricError, match="status differs"):
        decision_from_payload(payload, load_contract())


def test_payload_requires_each_preregistered_prefix_once() -> None:
    tabm, catboost = _frames()
    payload = decision_to_payload(evaluate_prefixes(tabm, catboost, load_contract()))
    payload["candidates"].pop()
    with pytest.raises(RealignMetricError, match="prefix set differs"):
        decision_from_payload(payload, load_contract())
