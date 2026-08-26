from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.e2_contracts import load_e2_contract
from experiments.tree_expert.e2_full_fit import AcceptedForFullFit
from experiments.tree_expert.e2_inference import (
    E2InferenceError,
    E2InferenceRuntime,
    audit_inference,
)
from experiments.tree_expert.features import TreeFeatureBatch


def _token(predictor: str = "catboost") -> AcceptedForFullFit:
    return AcceptedForFullFit(
        candidate_id="c1_anchor_residual",
        predictor=predictor,
        seeds=(42, 2026, 3407),
        iterations=MappingProxyType({42: 50, 2026: 50, 3407: 50}),
        decision_sha256="a" * 64,
    )


def _rows(size: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{index}" for index in range(size)],
            "value": np.linspace(0.1, 0.8, size),
        }
    )


class FakeModel:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.fail = False

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.fail:
            raise RuntimeError("model failure")
        return frame["value"].to_numpy(dtype="float64") * 0.1 + self.offset


class FakeTabM:
    def predict_batch(self, rows: pd.DataFrame, *, batch_size: int) -> np.ndarray:
        del batch_size
        return 0.3 + rows["value"].to_numpy(dtype="float64") * 0.05


def _transform(rows: pd.DataFrame, state: object) -> TreeFeatureBatch:
    del state
    values = rows["value"].to_numpy(dtype="float64")
    return TreeFeatureBatch(
        frame=rows.loc[:, ["value"]].copy(),
        anchor=0.4 + values * 0.01,
        row_id=rows["row_id"].astype(str).to_numpy(),
        target=None,
    )


def _runtime(*, predictor: str = "catboost") -> E2InferenceRuntime:
    return E2InferenceRuntime(
        token=_token(predictor),
        state=SimpleNamespace(),
        catboost_models=(FakeModel(0.0), FakeModel(0.01), FakeModel(-0.01)),
        transformer=_transform,
        tabm_predictor=FakeTabM() if predictor == "blend" else None,
        blend_method="probability" if predictor == "blend" else "catboost",
        catboost_weight=0.3 if predictor == "blend" else 1.0,
    )


def test_predictions_are_invariant_to_order_batch_and_companions() -> None:
    runtime = _runtime()
    rows = _rows()
    expected = runtime.predict(rows, batch_size=4096)

    reverse = runtime.predict(rows.iloc[::-1].reset_index(drop=True), batch_size=3)[::-1]
    singleton = np.concatenate(
        [runtime.predict(rows.iloc[[index]], batch_size=1) for index in range(len(rows))]
    )

    np.testing.assert_allclose(reverse, expected, atol=1e-6, rtol=0)
    np.testing.assert_allclose(singleton, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("mutation", ["target", "duplicate", "missing"])
def test_inference_rejects_invalid_row_identity(mutation: str) -> None:
    rows = _rows()
    if mutation == "target":
        rows["control_success"] = 1
    elif mutation == "duplicate":
        rows.loc[1, "row_id"] = rows.loc[0, "row_id"]
    else:
        rows = rows.drop(columns="row_id")

    with pytest.raises(E2InferenceError, match="row_id|target"):
        _runtime().predict(rows)


def test_model_error_is_raised_without_fallback() -> None:
    runtime = _runtime()
    runtime.catboost_models[0].fail = True

    with pytest.raises(RuntimeError, match="model failure"):
        runtime.predict(_rows())


def test_blend_runtime_uses_fixed_method_and_weight() -> None:
    rows = _rows()
    cat = _runtime().predict(rows)
    blended = _runtime(predictor="blend").predict(rows)
    tabm = FakeTabM().predict_batch(rows, batch_size=4096)

    np.testing.assert_allclose(blended, 0.7 * tabm + 0.3 * cat, atol=1e-12)


def test_audit_checks_all_invariance_modes() -> None:
    contract = load_e2_contract()
    runtime_limits = dict(contract.runtime)
    runtime_limits.update(
        inference_rows=8,
        inference_max_seconds=60,
        rss_max_bytes=10**15,
        gpu_max_bytes=10**15,
    )
    contract = replace(contract, runtime=MappingProxyType(runtime_limits))

    audit = audit_inference(_runtime(), _rows(), contract)

    assert audit.status == "passed"
    assert audit.row_count == 8
    assert set(audit.checks) == {
        "reverse",
        "shuffle",
        "batch_1",
        "batch_257",
        "batch_4096",
        "singleton",
        "companion",
    }
    assert audit.maximum_absolute_difference <= 1e-6


def test_inference_source_has_no_cross_row_evaluation_operations() -> None:
    source = Path("experiments/tree_expert/e2_inference.py").read_text()
    for token in (".groupby(", ".rolling(", ".shift(", ".expanding(", ".cumsum("):
        assert token not in source
