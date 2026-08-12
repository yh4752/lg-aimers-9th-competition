from __future__ import annotations

import numpy as np

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.tabr import TabRAdapter


def _batch(prefix: str, *, target: bool) -> FeatureBatch:
    return FeatureBatch(
        row_id=np.array([f"{prefix}-0", f"{prefix}-1"]),
        season=np.array([2023, 2023], dtype="int64"),
        game_type=np.array(["R", "F"]),
        x_num=np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32"),
        x_cat=np.array([[1], [2]], dtype="int64"),
        y=np.array([0.0, 1.0], dtype="float32") if target else None,
    )


class _FakeTabRRuntime:
    def __init__(self) -> None:
        self.context_row_ids: list[str] = []
        self.self_neighbor_masked = False

    def fit_context(self, batch: FeatureBatch) -> None:
        self.context_row_ids = batch.row_id.astype(str).tolist()

    def probabilities(self, model, x_num, x_cat):
        return np.full(len(x_num), 0.5, dtype="float64")

    def loss(self, model, x_num, x_cat, y, row_indices):
        self.self_neighbor_masked = row_indices.tolist() == [0, 1]
        return 0.25


def test_tabr_context_contains_only_training_rows() -> None:
    runtime = _FakeTabRRuntime()
    train = _batch("train", target=True)
    valid = _batch("valid", target=True)
    adapter = TabRAdapter(runtime=runtime)

    adapter.fit_context(train)
    probabilities = adapter.probabilities(object(), valid.x_num, valid.x_cat)

    assert probabilities.tolist() == [0.5, 0.5]
    assert runtime.context_row_ids == train.row_id.tolist()
    assert not set(valid.row_id).intersection(runtime.context_row_ids)


def test_tabr_never_uses_query_as_its_own_neighbor() -> None:
    runtime = _FakeTabRRuntime()
    train = _batch("train", target=True)
    adapter = TabRAdapter(runtime=runtime)
    adapter.fit_context(train)

    adapter.loss(
        object(),
        train.x_num,
        train.x_cat,
        train.y,
        row_indices=np.array([0, 1], dtype="int64"),
    )

    assert runtime.self_neighbor_masked is True
