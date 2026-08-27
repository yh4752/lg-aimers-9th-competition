from types import MappingProxyType

import numpy as np
import pandas as pd

from experiments.tree_expert.features import TreeFeatureBatch
from experiments.tree_expert.t3_inference import T3Predictor


class ConstantResidual:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype="float64")


def transformer(rows, _state):
    return TreeFeatureBatch(
        frame=pd.DataFrame({"x": np.arange(len(rows))}),
        anchor=rows["anchor"].to_numpy(dtype="float64"),
        row_id=rows["row_id"].astype(str).to_numpy(), target=None,
    )


def predictor():
    return T3Predictor(
        state=object(), recent_models=(ConstantResidual(0.02),) * 3,
        multi_models=(ConstantResidual(-0.01),) * 3,
        recent_weight=0.80, transformer=transformer,
    )


def rows():
    return pd.DataFrame({"row_id": ["a", "b", "c"], "anchor": [0.4, 0.5, 0.6]})


def test_inference_blends_recent_and_multi_probabilities():
    prediction = predictor().predict(rows())
    expected = 0.80 * (rows()["anchor"].to_numpy() + 0.02) + 0.20 * (rows()["anchor"].to_numpy() - 0.01)
    np.testing.assert_allclose(prediction, expected)


def test_prediction_is_independent_of_companion_rows_and_order():
    model = predictor()
    source = rows()
    full = model.predict(source, batch_size=4096)
    reverse = model.predict(source.iloc[::-1].reset_index(drop=True), batch_size=1)[::-1]
    np.testing.assert_allclose(full, reverse, atol=1e-6, rtol=0.0)
    for position in range(len(source)):
        singleton = model.predict(source.iloc[[position]].copy(), batch_size=1)
        np.testing.assert_allclose(singleton, full[[position]], atol=1e-6, rtol=0.0)

