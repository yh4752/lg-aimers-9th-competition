from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest
import torch

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.common import DLRuntimeDependencyError
from experiments.independent_dl.models.tabr import TabRAdapter, _TorchTabRRuntime
from experiments.independent_dl.models.tabr_search import (
    FoldTrainFaissIndex,
    TABR_SEARCH_POLICY,
    probe_faiss_gpu,
)


class _FakeFlatIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.keys = np.empty((0, dimension), dtype="float32")
        self.ids = np.empty(0, dtype="int64")

    def add(self, keys: np.ndarray) -> None:
        self.keys = np.asarray(keys, dtype="float32").copy()
        self.ids = np.arange(len(keys), dtype="int64")

    def add_with_ids(self, keys: np.ndarray, ids: np.ndarray) -> None:
        self.keys = np.asarray(keys, dtype="float32").copy()
        self.ids = np.asarray(ids, dtype="int64").copy()

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        self.last_search_k = k
        distances = np.square(
            np.asarray(queries, dtype="float32")[:, None, :] - self.keys[None, :, :]
        ).sum(axis=2)
        order = np.argsort(distances, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(distances, order, axis=1), self.ids[order]


class _FakeIvfIndex(_FakeFlatIndex):
    def __init__(self, quantizer, dimension, nlist, metric) -> None:
        super().__init__(dimension)
        self.nlist = nlist
        self.metric = metric
        self.nprobe = None
        self.trained_on = None

    def train(self, values: np.ndarray) -> None:
        self.trained_on = np.asarray(values).copy()


class _FakeFaiss:
    METRIC_L2 = 1

    def __init__(self) -> None:
        self.last_gpu_index = None
        self.gpu_transfer_calls = 0

    class StandardGpuResources:
        pass

    IndexFlatL2 = _FakeFlatIndex
    IndexIVFFlat = _FakeIvfIndex

    @staticmethod
    def get_num_gpus() -> int:
        return 1

    def index_cpu_to_gpu(self, resources, gpu_id, index):
        assert gpu_id == 0
        self.gpu_transfer_calls += 1
        self.last_gpu_index = index
        return index


class _Reporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def should_emit(self, *, completed_rows: int, stream: str) -> bool:
        return True

    def progress(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def _build_search(
    keys: np.ndarray, *, retrieval: int = 2
) -> tuple[FoldTrainFaissIndex, _FakeFaiss]:
    fake = _FakeFaiss()
    search = FoldTrainFaissIndex.build(
        keys,
        np.arange(len(keys), dtype="int64"),
        retrieval=retrieval,
        faiss_module=fake,
        reporter=_Reporter(),
    )
    return search, fake


def test_faiss_gpu_probe_requires_real_gpu_api() -> None:
    with pytest.raises(DLRuntimeDependencyError, match="FAISS GPU"):
        probe_faiss_gpu(faiss_module=object())

    fake = _FakeFaiss()
    probe_faiss_gpu(faiss_module=fake)
    assert fake.gpu_transfer_calls == 1


def test_colab_requirements_pin_the_t4_faiss_gpu_runtime() -> None:
    requirements = (
        Path(__file__).resolve().parents[1]
        / "experiments/independent_dl/requirements-colab.txt"
    ).read_text(encoding="utf-8")

    assert "faiss-gpu-cu12==1.14.1.post1" in requirements.splitlines()


def test_fold_index_contains_only_supplied_train_keys_and_uses_oversampling() -> None:
    keys = np.arange(24, dtype="float32").reshape(8, 3)
    reporter = _Reporter()
    fake = _FakeFaiss()
    search = FoldTrainFaissIndex.build(
        keys,
        np.arange(8, dtype="int64"),
        retrieval=1,
        faiss_module=fake,
        reporter=reporter,
    )

    result = search.search_and_rerank(
        torch.tensor([[0.0, 1.0, 2.0]]),
        query_absolute_indices=torch.tensor([0]),
    )

    assert fake.last_gpu_index.ids.tolist() == list(range(8))
    assert fake.last_gpu_index.last_search_k == min(
        len(keys), TABR_SEARCH_POLICY["oversample_factor"] + 1
    )
    assert result.tolist() == [[1]]
    assert reporter.events[-1][0] == "TABR_INDEX_READY"


def test_exact_rerank_matches_bruteforce_and_is_query_order_invariant() -> None:
    keys = np.array([[0.0], [1.0], [3.0], [7.0]], dtype="float32")
    search, _ = _build_search(keys, retrieval=2)
    queries = torch.tensor([[0.0], [3.0]])
    self_ids = torch.tensor([0, 2])

    paired = search.search_and_rerank(
        queries, query_absolute_indices=self_ids
    )
    reversed_result = search.search_and_rerank(
        queries.flip(0), query_absolute_indices=self_ids.flip(0)
    ).flip(0)
    singleton = torch.cat(
        [
            search.search_and_rerank(
                queries[index : index + 1],
                query_absolute_indices=self_ids[index : index + 1],
            )
            for index in range(2)
        ]
    )

    assert paired.tolist() == [[1, 2], [1, 0]]
    assert torch.equal(reversed_result, paired)
    assert torch.equal(singleton, paired)


@pytest.mark.parametrize(
    ("keys", "ids", "match"),
    [
        (np.array([[0.0], [np.nan]], dtype="float32"), np.arange(2), "finite"),
        (np.array([[0.0], [1.0]], dtype="float32"), np.array([0, 0]), "unique"),
        (np.array([0.0, 1.0], dtype="float32"), np.arange(2), "two-dimensional"),
    ],
)
def test_fold_index_rejects_malformed_fold_keys(keys, ids, match) -> None:
    with pytest.raises(ValueError, match=match):
        FoldTrainFaissIndex.build(
            keys,
            ids,
            retrieval=1,
            faiss_module=_FakeFaiss(),
            reporter=_Reporter(),
        )


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
    reporter = _Reporter()
    runtime = _TorchTabRRuntime(
        context, retrieval=2, faiss_module=_FakeFaiss(), reporter=reporter
    )
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
    assert "TABR_CONTEXT_ENCODING_PROGRESS" in [event for event, _ in reporter.events]
    assert "TABR_INDEX_READY" in [event for event, _ in reporter.events]
    assert "TABR_SEARCH_PROGRESS" in [event for event, _ in reporter.events]
    search_counts = [
        fields["completed_rows"]
        for event, fields in reporter.events
        if event == "TABR_SEARCH_PROGRESS"
    ]
    assert search_counts == [2, 4]


def test_each_tabr_index_refresh_has_an_independent_progress_stream() -> None:
    context = FeatureBatch(
        row_id=np.array(["r0", "r1", "r2"]),
        season=np.array([2023, 2023, 2023], dtype="int64"),
        game_type=np.array(["R", "R", "F"]),
        x_num=np.array([[0.0], [1.0], [2.0]], dtype="float32"),
        x_cat=np.zeros((3, 1), dtype="int64"),
        y=np.array([0.0, 1.0, 0.0], dtype="float32"),
    )
    reporter = _Reporter()
    runtime = _TorchTabRRuntime(
        context, retrieval=1, faiss_module=_FakeFaiss(), reporter=reporter
    )
    model = _CountingKeyModel()

    runtime.refresh_keys(model, device="cpu")
    runtime.refresh_keys(model, device="cpu")

    streams = [
        fields["stream"]
        for event, fields in reporter.events
        if event == "TABR_CONTEXT_ENCODING_PROGRESS"
    ]
    assert streams == ["tabr_context_encoding_1", "tabr_context_encoding_2"]


def test_tabr_rejects_search_before_key_refresh() -> None:
    context = FeatureBatch(
        row_id=np.array(["r0", "r1", "r2"]),
        season=np.array([2023, 2023, 2023], dtype="int64"),
        game_type=np.array(["R", "R", "F"]),
        x_num=np.array([[0.0], [1.0], [2.0]], dtype="float32"),
        x_cat=np.zeros((3, 1), dtype="int64"),
        y=np.array([0.0, 1.0, 0.0], dtype="float32"),
    )
    runtime = _TorchTabRRuntime(
        context, retrieval=1, faiss_module=_FakeFaiss()
    )

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
    runtime = _TorchTabRRuntime(
        context, retrieval=1, faiss_module=_FakeFaiss()
    )
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


def test_tabr_probes_faiss_gpu_before_encoding_fold_rows() -> None:
    context = FeatureBatch(
        row_id=np.array(["r0", "r1", "r2"]),
        season=np.array([2023, 2023, 2023], dtype="int64"),
        game_type=np.array(["R", "R", "F"]),
        x_num=np.array([[0.0], [1.0], [2.0]], dtype="float32"),
        x_cat=np.zeros((3, 1), dtype="int64"),
        y=np.array([0.0, 1.0, 0.0], dtype="float32"),
    )
    model = _CountingKeyModel()
    runtime = _TorchTabRRuntime(context, retrieval=1, faiss_module=object())

    with pytest.raises(DLRuntimeDependencyError, match="FAISS GPU"):
        runtime.refresh_keys(model, device="cpu")

    assert model.full_context_encode_calls == 0


def test_tabr_freezes_train_neighbors_and_reuses_them_without_search(
    tmp_path: Path,
) -> None:
    context = FeatureBatch(
        row_id=np.array(["r0", "r1", "r2", "r3"]),
        season=np.array([2023, 2023, 2023, 2023], dtype="int64"),
        game_type=np.array(["R", "R", "F", "F"]),
        x_num=np.array([[0.0], [1.0], [3.0], [7.0]], dtype="float32"),
        x_cat=np.zeros((4, 1), dtype="int64"),
        y=np.array([0.0, 1.0, 0.0, 1.0], dtype="float32"),
    )
    reporter = _Reporter()
    runtime = _TorchTabRRuntime(
        context, retrieval=2, faiss_module=_FakeFaiss(), reporter=reporter
    )
    runtime.configure_checkpoint(
        tmp_path,
        {"architecture": "tabr", "retrieval": 2, "width": 256, "blocks": 3},
    )
    model = _CountingKeyModel()
    runtime.refresh_keys(model, device="cpu")

    runtime.freeze_contexts(model, epoch=0, device="cpu")

    frozen = np.load(tmp_path / "frozen_neighbors.npy")
    assert frozen.shape == (4, 2)
    assert all(row not in frozen[row].tolist() for row in range(4))
    state = runtime.checkpoint_state()
    assert state["policy_version"] == TABR_SEARCH_POLICY["version"]
    assert state["frozen_neighbors_shape"] == [4, 2]
    assert len(state["frozen_neighbors_sha256"]) == 64
    frozen_event = [
        fields
        for event, fields in reporter.events
        if event == "TABR_CONTEXTS_FROZEN"
    ][0]
    assert set(frozen_event) == {"epoch", "shape", "sha256"}

    class _NoSearch:
        @staticmethod
        def search_and_rerank(*args, **kwargs):
            raise AssertionError("frozen training rows must not call FAISS")

    runtime._search_index = _NoSearch()
    reused = runtime._search_keys(
        torch.tensor([[0.0], [3.0]]), row_indices=torch.tensor([0, 2])
    )
    assert reused.tolist() == frozen[[0, 2]].tolist()


def test_tabr_restores_only_matching_frozen_neighbor_identity(tmp_path: Path) -> None:
    context = FeatureBatch(
        row_id=np.array(["r0", "r1", "r2", "r3"]),
        season=np.array([2023, 2023, 2023, 2023], dtype="int64"),
        game_type=np.array(["R", "R", "F", "F"]),
        x_num=np.array([[0.0], [1.0], [3.0], [7.0]], dtype="float32"),
        x_cat=np.zeros((4, 1), dtype="int64"),
        y=np.array([0.0, 1.0, 0.0, 1.0], dtype="float32"),
    )
    config = {
        "architecture": "tabr",
        "retrieval": 2,
        "width": 256,
        "blocks": 3,
    }
    source = _TorchTabRRuntime(
        context, retrieval=2, faiss_module=_FakeFaiss()
    )
    source.configure_checkpoint(tmp_path, config)
    model = _CountingKeyModel()
    source.refresh_keys(model, device="cpu")
    source.freeze_contexts(model, epoch=0, device="cpu")
    state = source.checkpoint_state()

    matching = _TorchTabRRuntime(
        context, retrieval=2, faiss_module=_FakeFaiss()
    )
    matching.configure_checkpoint(tmp_path, config)
    assert matching.restore_checkpoint_state(state, model, "cpu") is True
    assert matching.has_frozen_contexts is True

    changed = _TorchTabRRuntime(
        context, retrieval=2, faiss_module=_FakeFaiss()
    )
    changed.configure_checkpoint(tmp_path, {**config, "width": 512})
    assert changed.restore_checkpoint_state(state, model, "cpu") is False
    assert changed.has_frozen_contexts is False

    changed_rows = FeatureBatch(
        row_id=np.array(["other0", "other1", "other2", "other3"]),
        season=context.season,
        game_type=context.game_type,
        x_num=context.x_num,
        x_cat=context.x_cat,
        y=context.y,
    )
    changed_context = _TorchTabRRuntime(
        changed_rows, retrieval=2, faiss_module=_FakeFaiss()
    )
    changed_context.configure_checkpoint(tmp_path, config)
    assert changed_context.restore_checkpoint_state(state, model, "cpu") is False
