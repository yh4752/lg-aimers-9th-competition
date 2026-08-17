from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments.catboost_tabm_blend.contracts import load_contract
from experiments.catboost_tabm_blend.metrics import (
    BlendMetricError,
    _gate_passes,
    canonical_json,
    decision_payload,
    evaluate_blends,
)


COLUMNS = (
    "row_id",
    "target",
    "probability",
    "game_type",
    "game_month",
    "pitcher_id_known",
    "batter_id_known",
)


def fold_frame(probabilities: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["A", "B", "C", "D"],
            "target": [1, 0, 1, 0],
            "probability": probabilities,
            "game_type": ["R", "R", "P", "P"],
            "game_month": [4, 4, 5, 5],
            "pitcher_id_known": ["known", "known", "oov", "oov"],
            "batter_id_known": ["known", "oov", "known", "oov"],
        },
        columns=COLUMNS,
    )


def two_folds(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"2022->2023": frame.copy(), "2023->2024": frame.copy()}


def test_fixed_weights_and_diagnostics_are_evaluated() -> None:
    tabm = two_folds(fold_frame([0.60, 0.40, 0.60, 0.40]))
    catboost = two_folds(fold_frame([0.80, 0.20, 0.80, 0.20]))

    result = evaluate_blends(tabm, catboost, load_contract())

    assert tuple(item.tabm_weight for item in result.candidates) == (1.0, 0.9, 0.8, 0.7)
    assert result.selected_tabm_weight == 0.7
    assert result.reason == "fixed_blend_passed"
    assert result.prediction_correlation == {"2022->2023": 1.0, "2023->2024": 1.0}
    game_type = result.segment_diagnostics["2022->2023"]["tabm"]["game_type"]
    assert game_type["R"] == {"row_count": 2, "brier": pytest.approx(0.16)}
    assert game_type["P"] == {"row_count": 2, "brier": pytest.approx(0.16)}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.assign(row_id=["A", "A", "C", "D"]),
        lambda frame: frame.iloc[::-1].reset_index(drop=True),
        lambda frame: frame.assign(target=[0, 0, 1, 0]),
        lambda frame: frame.assign(game_type=["P", "R", "P", "P"]),
        lambda frame: frame.assign(probability=[np.nan, 0.2, 0.8, 0.2]),
        lambda frame: frame.assign(probability=[1.1, 0.2, 0.8, 0.2]),
        lambda frame: frame.drop(columns="game_month"),
        lambda frame: frame.assign(extra=1),
    ],
)
def test_alignment_or_schema_drift_is_rejected(mutate) -> None:
    tabm = two_folds(fold_frame([0.60, 0.40, 0.60, 0.40]))
    catboost = two_folds(fold_frame([0.80, 0.20, 0.80, 0.20]))
    catboost["2023->2024"] = mutate(catboost["2023->2024"])

    with pytest.raises(BlendMetricError):
        evaluate_blends(tabm, catboost, load_contract())


def test_fold_set_is_exact() -> None:
    frame = fold_frame([0.60, 0.40, 0.60, 0.40])
    with pytest.raises(BlendMetricError, match="fold"):
        evaluate_blends({"2023->2024": frame}, two_folds(frame), load_contract())


def test_gate_comparison_is_exact_at_float_boundaries() -> None:
    gate = 0.00003
    below = float(np.nextafter(gate, 0.0))
    above = float(np.nextafter(gate, 1.0))

    assert not _gate_passes(below, {"a": 0.0}, gate, gate)
    assert _gate_passes(gate, {"a": gate}, gate, gate)
    assert _gate_passes(above, {"a": gate}, gate, gate)
    assert not _gate_passes(above, {"a": above}, gate, gate)


def test_decision_serialization_is_deterministic_and_finite() -> None:
    tabm = two_folds(fold_frame([0.60, 0.40, 0.60, 0.40]))
    catboost = two_folds(fold_frame([0.80, 0.20, 0.80, 0.20]))
    decision = evaluate_blends(tabm, catboost, load_contract())

    first = canonical_json(decision_payload(decision))
    second = canonical_json(decision_payload(decision))

    assert first == second
    parsed = json.loads(first)
    assert parsed["selected_tabm_weight"] == 0.7
    assert parsed["candidates"][-1]["passed"] is True
