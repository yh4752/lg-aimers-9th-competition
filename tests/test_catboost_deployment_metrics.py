from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from experiments.catboost_deployment.contracts import load_contract
from experiments.catboost_deployment.metrics import (
    CATBOOST_PREDICTION_COLUMNS,
    DeploymentMetricError,
    decision_payload,
    evaluate_prefixes,
)


FOLDS = ("2022->2023", "2023->2024")
PREFIXES = (4, 32, 64, 128, 192, 296, 400)


def _tabm(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"{prefix}-{index}" for index in range(4)],
            "target": [0, 1, 0, 1],
            "probability": [0.4, 0.6, 0.45, 0.55],
            "game_type": ["R"] * 4,
            "game_month": [4] * 4,
            "pitcher_id_known": ["known"] * 4,
            "batter_id_known": ["known"] * 4,
        }
    )


def _catboost(tabm: pd.DataFrame, *, inverted: bool = False) -> pd.DataFrame:
    target = tabm["target"].to_numpy(dtype="float64")
    strong = 0.9 * target + 0.1 * (1.0 - target)
    if inverted:
        strong = 1.0 - strong
    data: dict[str, object] = {
        "row_id": tabm["row_id"],
        "target": tabm["target"],
    }
    for prefix in PREFIXES:
        if inverted:
            data[f"p_{prefix}"] = strong
        elif prefix in (128, 192):
            data[f"p_{prefix}"] = strong
        elif prefix == 64:
            data[f"p_{prefix}"] = 0.8 * target + 0.2 * (1.0 - target)
        else:
            data[f"p_{prefix}"] = np.full(len(tabm), 0.5)
    for column in (
        "game_type",
        "game_month",
        "pitcher_id_known",
        "batter_id_known",
    ):
        data[column] = tabm[column]
    frame = pd.DataFrame(data)
    return frame.loc[:, CATBOOST_PREDICTION_COLUMNS]


def _frames(*, inverted: bool = False):
    tabm = {fold: _tabm(fold) for fold in FOLDS}
    catboost = {fold: _catboost(tabm[fold], inverted=inverted) for fold in FOLDS}
    return tabm, catboost


def test_selects_lowest_passing_fixed_prefix_and_smaller_exact_tie() -> None:
    tabm, catboost = _frames()

    decision = evaluate_prefixes(tabm, catboost, load_contract())

    assert decision.status == "deployment_aligned"
    assert decision.selected_tree_count == 128
    assert len(decision.candidates) == 7
    assert {item.tree_count for item in decision.candidates} == set(PREFIXES)
    assert next(item for item in decision.candidates if item.tree_count == 128).passed


def test_blocks_when_no_fixed_prefix_passes() -> None:
    tabm, catboost = _frames(inverted=True)

    decision = evaluate_prefixes(tabm, catboost, load_contract())

    assert decision.status == "deployment_blocked"
    assert decision.selected_tree_count is None
    assert decision.reason == "no_tree_prefix_passed"


def test_fold_mapping_order_does_not_change_decision() -> None:
    tabm, catboost = _frames()
    expected = decision_payload(evaluate_prefixes(tabm, catboost, load_contract()))

    reversed_tabm = dict(reversed(tuple(tabm.items())))
    reversed_catboost = dict(reversed(tuple(catboost.items())))

    assert decision_payload(
        evaluate_prefixes(reversed_tabm, reversed_catboost, load_contract())
    ) == expected


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda frame: frame.drop(index=frame.index[-1]), "row count"),
        (lambda frame: frame.assign(row_id=["x", "x", "y", "z"]), "row_id"),
        (lambda frame: frame.assign(target=[0, 1, 1, 1]), "target"),
        (lambda frame: frame.assign(p_128=[0.1, 0.9, np.nan, 0.9]), "probability"),
    ],
)
def test_rejects_misaligned_or_invalid_predictions(mutate, match: str) -> None:
    tabm, catboost = _frames()
    catboost["2022->2023"] = mutate(catboost["2022->2023"])

    with pytest.raises(DeploymentMetricError, match=match):
        evaluate_prefixes(tabm, catboost, load_contract())


def test_exact_gate_boundary_passes() -> None:
    tabm, catboost = _frames()
    initial = evaluate_prefixes(tabm, catboost, load_contract())
    selected = next(item for item in initial.candidates if item.tree_count == 128)
    contract = replace(
        load_contract(),
        minimum_weighted_gain=selected.weighted_gain,
        maximum_fold_regression=max(selected.fold_regression.values()),
    )

    decision = evaluate_prefixes(tabm, catboost, contract)

    assert next(item for item in decision.candidates if item.tree_count == 128).passed


def test_decision_payload_is_finite_canonical_json_compatible() -> None:
    tabm, catboost = _frames()
    payload = decision_payload(evaluate_prefixes(tabm, catboost, load_contract()))

    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)

    assert '"selected_tree_count": 128' in encoded
