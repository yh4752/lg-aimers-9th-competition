from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import experiments.tree_expert.hetero_features as hetero_features
from experiments.tree_expert.features import TreeFeatureBatch
from experiments.tree_expert.hetero_features import (
    HeteroFeatureError,
    fit_hetero_features,
    transform_hetero_features,
)


def _batch(ids: list[str], categories: list[object], numbers: list[float], target=None):
    values = None if target is None else np.asarray(target, dtype="int8")
    return TreeFeatureBatch(
        frame=pd.DataFrame({"category": categories, "number": numbers}),
        anchor=np.full(len(ids), 0.5),
        row_id=np.asarray(ids),
        target=values,
    )


@pytest.fixture
def patched_tree(monkeypatch: pytest.MonkeyPatch):
    state = SimpleNamespace(
        categorical_columns=("category",),
        feature_columns=("category", "number"),
    )
    train_batch = _batch(["a", "b", "c"], ["z", "a", None], [1.0, np.nan, 3.0], [0, 1, 0])

    def fit(*args, **kwargs):
        return state, train_batch

    def transform(rows, active):
        assert active is state
        return _batch(
            rows["row_id"].astype(str).tolist(),
            rows["category"].tolist(),
            rows["number"].tolist(),
        )

    monkeypatch.setattr(hetero_features, "fit_tree_features", fit)
    monkeypatch.setattr(hetero_features, "transform_tree_features", transform)
    return state


def test_fit_uses_sorted_training_only_category_map(patched_tree) -> None:
    state, batch = fit_hetero_features(pd.DataFrame(), valid_year=2024)

    assert dict(state.category_maps["category"]) == {"__MISSING__": 0, "a": 1, "z": 2}
    np.testing.assert_array_equal(batch.matrix[:, 0], [2.0, 1.0, 0.0])
    assert np.isnan(batch.matrix[1, 1])
    assert batch.matrix.flags.writeable is False


def test_transform_is_neighbor_invariant_and_oov_is_minus_one(patched_tree) -> None:
    state, _ = fit_hetero_features(pd.DataFrame(), valid_year=2024)
    rows = pd.DataFrame({
        "row_id": ["v0", "v1", "v2"],
        "category": ["z", "never", "a"],
        "number": [4.0, 5.0, 6.0],
    })
    whole = transform_hetero_features(rows, state)
    one = transform_hetero_features(rows.iloc[[1]], state)

    np.testing.assert_allclose(one.matrix, whole.matrix[[1]])
    assert whole.matrix[1, 0] == -1.0
    assert one.row_id.tolist() == ["v1"]


def test_transform_rejects_infinity(patched_tree) -> None:
    state, _ = fit_hetero_features(pd.DataFrame(), valid_year=2024)
    rows = pd.DataFrame({"row_id": ["v0"], "category": ["a"], "number": [np.inf]})

    with pytest.raises(HeteroFeatureError, match="infinity"):
        transform_hetero_features(rows, state)


def test_transform_rejects_target_from_underlying_builder(
    patched_tree, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ = fit_hetero_features(pd.DataFrame(), valid_year=2024)
    monkeypatch.setattr(
        hetero_features,
        "transform_tree_features",
        lambda rows, active: _batch(["v0"], ["a"], [1.0], [1]),
    )

    with pytest.raises(HeteroFeatureError, match="target"):
        transform_hetero_features(pd.DataFrame(), state)
