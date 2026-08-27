from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.features import TreeFeatureBatch
from experiments.tree_expert.rf_inference import (
    RFInferenceError,
    RFInferenceRuntime,
    audit_row_independence,
)


class Baseline:
    def predict(self, rows, *, batch_size=4096):
        del batch_size
        return rows["base"].to_numpy(dtype="float64")


class ResidualModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def transformer(rows, _state):
    return TreeFeatureBatch(
        frame=pd.DataFrame({"x": rows["x"].to_numpy()}),
        anchor=rows["anchor"].to_numpy(dtype="float64"),
        row_id=rows["row_id"].astype(str).to_numpy(),
        target=None,
    )


def _runtime() -> RFInferenceRuntime:
    return RFInferenceRuntime(
        baseline_predictor=Baseline(),
        f_state=object(),
        f_models=(ResidualModel(0.1), ResidualModel(0.1), ResidualModel(0.1)),
        alpha_f=0.5,
        r_state=object(),
        r_models=(ResidualModel(-0.1), ResidualModel(-0.1), ResidualModel(-0.1)),
        alpha_r=0.25,
        transformer=transformer,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d"],
            "game_type": ["R", "F", "R", "F"],
            "base": [0.4, 0.6, 0.5, 0.7],
            "anchor": [0.5, 0.5, 0.5, 0.5],
            "x": [1, 2, 3, 4],
        }
    )


def test_inference_routes_fixed_r_and_f_experts() -> None:
    prediction = _runtime().predict(_frame())
    np.testing.assert_allclose(prediction, [0.4, 0.6, 0.475, 0.65])


def test_inference_is_identical_for_singleton_shuffle_reverse_and_batches() -> None:
    report = audit_row_independence(_frame(), _runtime(), tolerance=1e-6)
    assert report["status"] == "passed"
    assert report["maximum_absolute_difference"] <= 1e-6


def test_inference_rejects_unknown_game_type() -> None:
    frame = _frame()
    frame.loc[0, "game_type"] = "X"
    with pytest.raises(RFInferenceError, match="game_type differs"):
        _runtime().predict(frame)
