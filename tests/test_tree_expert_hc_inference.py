from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.features import TreeFeatureBatch
from experiments.tree_expert.hc_inference import (
    HCInferenceError,
    HCInferenceRuntime,
    audit_row_independence,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{index}" for index in range(8)],
            "signal": np.linspace(0.0, 0.7, 8),
            "game_type": ["R", "F"] * 4,
        }
    )


class _Base:
    def predict(self, rows, *, batch_size=4096):
        del batch_size
        return 0.4 + rows["signal"].to_numpy(dtype="float64") * 0.1


class _Residual:
    def __init__(self, offset):
        self.offset = offset

    def predict(self, frame):
        return frame["tree__signal"].to_numpy(dtype="float64") * 0.01 + self.offset


def _tree_transform(rows, state):
    del state
    return TreeFeatureBatch(
        frame=pd.DataFrame({"signal": rows["signal"].to_numpy()}),
        anchor=np.full(len(rows), 0.5),
        row_id=rows["row_id"].astype(str).to_numpy(),
        target=None,
    )


def _hierarchy_transform(rows, state):
    del state
    return pd.DataFrame({"hc_rate": 0.2 + rows["signal"].to_numpy() * 0.0})


def _runtime() -> HCInferenceRuntime:
    return HCInferenceRuntime(
        baseline_predictor=_Base(),
        tree_state=SimpleNamespace(),
        hierarchy_state=SimpleNamespace(),
        c1_models=(_Residual(0.01), _Residual(0.02), _Residual(0.03)),
        feature_columns=("tree__signal", "hc_rate", "p0"),
        categorical_columns=(),
        tree_transformer=_tree_transform,
        hierarchy_transformer=_hierarchy_transform,
    )


def test_hc_runtime_is_row_order_singleton_rebatch_and_twin_independent():
    rows = _rows()
    runtime = _runtime()
    expected = runtime.predict(rows, batch_size=4096)
    reverse_rows = rows.iloc[::-1].reset_index(drop=True)
    reverse = runtime.predict(reverse_rows, batch_size=1)[::-1]
    singleton = np.asarray(
        [runtime.predict(rows.iloc[[index]].copy(), batch_size=1)[0] for index in range(len(rows))]
    )
    np.testing.assert_allclose(expected, reverse, rtol=0, atol=1e-12)
    np.testing.assert_allclose(expected, singleton, rtol=0, atol=1e-12)
    np.testing.assert_allclose(expected, runtime.predict(rows, batch_size=3), rtol=0, atol=1e-12)

    twin = pd.concat([rows.iloc[[0]], rows.iloc[[0]]], ignore_index=True)
    twin.loc[1, "row_id"] = "r0_twin"
    twin_probability = runtime.predict(twin, batch_size=2)
    assert twin_probability[0] == pytest.approx(twin_probability[1], abs=1e-12)

    audit = audit_row_independence(rows, runtime, tolerance=1e-12)
    assert audit["status"] == "passed"
    assert audit["maximum_absolute_difference"] <= 1e-12


def test_hc_runtime_rejects_target_duplicate_ids_and_feature_schema():
    runtime = _runtime()
    rows = _rows()
    with pytest.raises(HCInferenceError, match="target"):
        runtime.predict(rows.assign(control_success=0))
    with pytest.raises(HCInferenceError, match="row identity"):
        runtime.predict(pd.concat([rows, rows.iloc[[0]]], ignore_index=True))

    mismatched = HCInferenceRuntime(
        baseline_predictor=_Base(),
        tree_state=SimpleNamespace(),
        hierarchy_state=SimpleNamespace(),
        c1_models=(_Residual(0.01), _Residual(0.02), _Residual(0.03)),
        feature_columns=("tree__signal", "missing", "p0"),
        categorical_columns=(),
        tree_transformer=_tree_transform,
        hierarchy_transformer=_hierarchy_transform,
    )
    with pytest.raises(HCInferenceError, match="feature schema"):
        mismatched.predict(rows)
