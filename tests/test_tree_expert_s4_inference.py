import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.s4_inference import (
    FrozenS4Predictor,
    S4InferenceError,
    audit_s4_independence,
)


class Transformer:
    def transform(self, rows):
        return rows[["value"]].to_numpy(dtype="float64")


class Classifier:
    def __init__(self, offset): self.offset = offset
    def predict_proba(self, x):
        p = np.clip(0.4 + self.offset + 0.01 * x[:, 0], 0.01, 0.99)
        return np.column_stack([1 - p, p])


class Regressor:
    def __init__(self, value): self.value = value
    def predict(self, x): return np.full(len(x), self.value)


def _predictor():
    return FrozenS4Predictor(
        required_columns=("row_id", "game_type", "value"),
        transformer=Transformer(), recent_model=Classifier(0.02), multi_model=Classifier(-0.01),
        recent_weight=0.75, residual_alpha=0.5,
        residual_model=Regressor(0.01), residual_r_model=None, residual_f_model=None,
        calibrator=None,
    )


def _rows():
    return pd.DataFrame({
        "row_id": [f"r{i}" for i in range(41)],
        "game_type": np.where(np.arange(41) % 4, "R", "F"),
        "value": np.arange(41) % 5,
    })


def test_frozen_predictor_is_singleton_shuffle_reverse_and_batch_invariant():
    audit = audit_s4_independence(_predictor(), _rows(), batch_sizes=(1, 17, 256))
    assert audit.maximum_absolute_difference <= 1e-12


def test_predictor_rejects_target_and_unknown_game_type():
    with pytest.raises(S4InferenceError):
        _predictor().predict(_rows().assign(control_success=1))
    with pytest.raises(S4InferenceError):
        _predictor().predict(_rows().assign(game_type="UNKNOWN"))


def test_rf_route_uses_only_current_row_game_type():
    predictor = FrozenS4Predictor(
        required_columns=("row_id", "game_type", "value"), transformer=Transformer(),
        recent_model=Classifier(0), multi_model=Classifier(0), recent_weight=0.5,
        residual_alpha=1.0, residual_model=None,
        residual_r_model=Regressor(0.1), residual_f_model=Regressor(-0.1), calibrator=None,
    )
    result = predictor.predict(_rows().iloc[:2])
    assert result[0] < result[1]
