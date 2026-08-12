from __future__ import annotations

import numpy as np
import torch

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.tabr import TabRAdapter, _TorchTabRRuntime


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


class _CountingKeyModel:
    def __init__(self) -> None:
        self.training = True
        self.full_context_encode_calls = 0

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def encode(self, x_num, x_cat):
        if len(x_num) == 4:
            self.full_context_encode_calls += 1
        return x_num, x_num


def test_tabr_reuses_refreshed_fold_keys_and_matches_exact_topk() -> None:
    context = FeatureBatch(
        row_id=np.array(["r0", "r1", "r2", "r3"]),
        season=np.array([2023, 2023, 2023, 2023], dtype="int64"),
        game_type=np.array(["R", "R", "F", "F"]),
        x_num=np.array([[0.0], [1.0], [3.0], [7.0]], dtype="float32"),
        x_cat=np.zeros((4, 1), dtype="int64"),
        y=np.array([0.0, 1.0, 0.0, 1.0], dtype="float32"),
    )
    runtime = _TorchTabRRuntime(context, retrieval=2)
    model = _CountingKeyModel()
    query_num = torch.tensor([[0.0], [3.0]])
    query_cat = torch.zeros((2, 1), dtype=torch.long)

    runtime.refresh_keys(model, device="cpu")
    first = runtime.search(
        model,
        query_num,
        query_cat,
        row_indices=torch.tensor([0, 2]),
    )
    second = runtime.search(
        model,
        query_num,
        query_cat,
        row_indices=torch.tensor([0, 2]),
    )

    assert model.full_context_encode_calls == 1
    assert first.tolist() == [[1, 2], [1, 0]]
    assert second.tolist() == first.tolist()


def test_tabr_rejects_search_before_key_refresh() -> None:
    context = FeatureBatch(
        row_id=np.array(["r0", "r1", "r2"]),
        season=np.array([2023, 2023, 2023], dtype="int64"),
        game_type=np.array(["R", "R", "F"]),
        x_num=np.array([[0.0], [1.0], [2.0]], dtype="float32"),
        x_cat=np.zeros((3, 1), dtype="int64"),
        y=np.array([0.0, 1.0, 0.0], dtype="float32"),
    )
    runtime = _TorchTabRRuntime(context, retrieval=1)

    try:
        runtime.search(
            _CountingKeyModel(),
            torch.tensor([[0.0]]),
            torch.zeros((1, 1), dtype=torch.long),
            row_indices=None,
        )
    except RuntimeError as error:
        assert "refresh" in str(error).casefold()
    else:
        raise AssertionError("TabR search accepted a missing key cache")


def test_tabr_rejects_keys_from_another_model_object() -> None:
    context = FeatureBatch(
        row_id=np.array(["r0", "r1", "r2"]),
        season=np.array([2023, 2023, 2023], dtype="int64"),
        game_type=np.array(["R", "R", "F"]),
        x_num=np.array([[0.0], [1.0], [2.0]], dtype="float32"),
        x_cat=np.zeros((3, 1), dtype="int64"),
        y=np.array([0.0, 1.0, 0.0], dtype="float32"),
    )
    runtime = _TorchTabRRuntime(context, retrieval=1)
    runtime.refresh_keys(_CountingKeyModel(), device="cpu")

    try:
        runtime.search(
            _CountingKeyModel(),
            torch.tensor([[0.0]]),
            torch.zeros((1, 1), dtype=torch.long),
            row_indices=None,
        )
    except RuntimeError as error:
        assert "model" in str(error).casefold()
    else:
        raise AssertionError("TabR accepted keys from another model")
