"""Fold-local FAISS GPU retrieval with exact PyTorch reranking for TabR."""

from __future__ import annotations

from types import MappingProxyType
import math
import os
import time
from typing import Mapping

import numpy as np

from .common import DLRuntimeDependencyError, import_runtime_module


TABR_SEARCH_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "version": "tabr_t4_ivf_flat_v1",
        "nlist": 4096,
        "nprobe": 64,
        "oversample_factor": 4,
        "training_sample_rows": 262_144,
        "key_batch_rows": 8192,
        "freeze_after_epochs": 1,
    }
)


def _faiss_module(faiss_module: object | None) -> object:
    os.environ["_FAISS_WHEEL_DISABLE_CUDA_PRELOAD"] = "1"
    return import_runtime_module("faiss") if faiss_module is None else faiss_module


def _require_gpu_api(faiss: object) -> None:
    required = (
        "StandardGpuResources",
        "IndexFlatL2",
        "IndexIVFFlat",
        "index_cpu_to_gpu",
        "get_num_gpus",
        "METRIC_L2",
    )
    if any(not hasattr(faiss, name) for name in required):
        raise DLRuntimeDependencyError(
            "FAISS GPU API가 필요합니다. CPU 전용 FAISS로는 TabR을 실행하지 않습니다."
        )
    if int(faiss.get_num_gpus()) < 1:
        raise DLRuntimeDependencyError(
            "FAISS GPU가 보이지 않습니다. GPU 런타임과 faiss-gpu-cu12를 확인하세요."
        )


def probe_faiss_gpu(*, faiss_module: object | None = None) -> None:
    """Fail before full-fold encoding unless a two-vector GPU search works."""

    faiss = _faiss_module(faiss_module)
    _require_gpu_api(faiss)
    try:
        resources = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatL2(2)
        gpu_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
        values = np.array([[0.0, 0.0], [1.0, 1.0]], dtype="float32")
        gpu_index.add(values)
        distances, indices = gpu_index.search(values[:1], 1)
    except Exception as error:
        raise DLRuntimeDependencyError(
            f"FAISS GPU smoke search failed: {type(error).__name__}: {error}"
        ) from error
    if np.asarray(distances).shape != (1, 1) or int(np.asarray(indices)[0, 0]) != 0:
        raise DLRuntimeDependencyError("FAISS GPU smoke search returned invalid output")


def _validated_keys(
    keys: np.ndarray, absolute_indices: np.ndarray, retrieval: int
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(keys, dtype="float32")
    indices = np.asarray(absolute_indices)
    if array.ndim != 2:
        raise ValueError("TabR keys must be a two-dimensional matrix")
    if len(array) < 2 or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError("TabR keys must be a non-empty finite matrix")
    if indices.ndim != 1 or len(indices) != len(array):
        raise ValueError("TabR absolute indices must align with keys")
    if np.issubdtype(indices.dtype, np.bool_) or not np.issubdtype(
        indices.dtype, np.integer
    ):
        raise ValueError("TabR absolute indices must be integers")
    indices = indices.astype("int64", copy=False)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("TabR absolute indices must be unique")
    if not np.array_equal(np.sort(indices), np.arange(len(indices), dtype="int64")):
        raise ValueError("TabR absolute indices must cover the fold-training rows")
    if isinstance(retrieval, bool) or not isinstance(retrieval, int):
        raise ValueError("TabR retrieval must be an integer")
    if retrieval < 1 or retrieval >= len(array):
        raise ValueError("TabR retrieval must be smaller than the fold-training rows")
    return np.ascontiguousarray(array), np.ascontiguousarray(indices)


class FoldTrainFaissIndex:
    """GPU approximate shortlist bound to one immutable fold-training key set."""

    def __init__(
        self,
        *,
        keys: np.ndarray,
        absolute_indices: np.ndarray,
        retrieval: int,
        gpu_index: object,
        gpu_resources: object,
        nlist: int,
        nprobe: int,
    ) -> None:
        self._keys = keys
        self.absolute_indices = absolute_indices
        self.retrieval = retrieval
        self._index = gpu_index
        self._gpu_resources = gpu_resources
        self.nlist = nlist
        self.nprobe = nprobe

    @classmethod
    def build(
        cls,
        keys: np.ndarray,
        absolute_indices: np.ndarray,
        *,
        retrieval: int,
        reporter: object | None,
        faiss_module: object | None = None,
    ) -> "FoldTrainFaissIndex":
        array, indices = _validated_keys(keys, absolute_indices, retrieval)
        faiss = _faiss_module(faiss_module)
        probe_faiss_gpu(faiss_module=faiss)
        started = time.monotonic()
        dimension = int(array.shape[1])
        configured_nlist = int(TABR_SEARCH_POLICY["nlist"])
        nlist = min(configured_nlist, max(1, int(math.sqrt(len(array)))))
        nprobe = min(int(TABR_SEARCH_POLICY["nprobe"]), nlist)
        quantizer = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIVFFlat(
            quantizer, dimension, nlist, faiss.METRIC_L2
        )
        resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
        gpu_index.nprobe = nprobe
        sample_rows = min(
            len(array), int(TABR_SEARCH_POLICY["training_sample_rows"])
        )
        if sample_rows == len(array):
            sample = array
        else:
            rng = np.random.default_rng(42)
            positions = np.sort(
                rng.choice(len(array), size=sample_rows, replace=False)
            )
            sample = np.ascontiguousarray(array[positions])
        gpu_index.train(sample)
        gpu_index.add_with_ids(array, indices)
        result = cls(
            keys=array,
            absolute_indices=indices,
            retrieval=retrieval,
            gpu_index=gpu_index,
            gpu_resources=resources,
            nlist=nlist,
            nprobe=nprobe,
        )
        if reporter is not None:
            reporter.emit(
                "TABR_INDEX_READY",
                index_type="gpu_ivf_flat",
                vector_count=len(array),
                dimension=dimension,
                nlist=nlist,
                nprobe=nprobe,
                build_seconds=time.monotonic() - started,
                gpu_id=0,
            )
        return result

    def search_and_rerank(
        self,
        query_keys: object,
        *,
        query_absolute_indices: object | None,
    ) -> object:
        torch = import_runtime_module("torch")
        if getattr(query_keys, "ndim", None) != 2:
            raise ValueError("TabR query keys must be two-dimensional")
        if int(query_keys.shape[1]) != int(self._keys.shape[1]):
            raise ValueError("TabR query key dimension does not match the index")
        if not bool(torch.isfinite(query_keys).all()):
            raise ValueError("TabR query keys must be finite")
        query_count = int(query_keys.shape[0])
        if query_absolute_indices is None:
            self_ids = None
            self_margin = 0
        else:
            if getattr(query_absolute_indices, "ndim", None) != 1 or len(
                query_absolute_indices
            ) != query_count:
                raise ValueError("TabR query absolute indices are misaligned")
            self_ids = query_absolute_indices.to(
                device=query_keys.device, dtype=torch.long
            )
            self_margin = 1
        shortlist_size = min(
            len(self._keys),
            self.retrieval * int(TABR_SEARCH_POLICY["oversample_factor"])
            + self_margin,
        )
        query_numpy = np.ascontiguousarray(
            query_keys.detach().to(dtype=torch.float32, device="cpu").numpy()
        )
        _, candidate_ids = self._index.search(query_numpy, shortlist_size)
        candidate_ids = np.asarray(candidate_ids, dtype="int64")
        if candidate_ids.shape != (query_count, shortlist_size):
            raise RuntimeError("FAISS returned a malformed TabR shortlist")
        if (candidate_ids < 0).any() or (candidate_ids >= len(self._keys)).any():
            raise RuntimeError("FAISS returned an out-of-fold TabR candidate")
        shortlist = torch.as_tensor(
            candidate_ids, dtype=torch.long, device=query_keys.device
        )
        query_for_distance = query_keys.to(dtype=torch.float32)
        candidate_keys = torch.as_tensor(
            self._keys[candidate_ids],
            dtype=torch.float32,
            device=query_keys.device,
        )
        scores = (
            -query_for_distance.square().sum(-1, keepdim=True)
            + 2.0 * (query_for_distance[:, None, :] * candidate_keys).sum(-1)
            - candidate_keys.square().sum(-1)
        )
        if self_ids is not None:
            scores = scores.masked_fill(
                shortlist.eq(self_ids[:, None]), float("-inf")
            )
        id_order = torch.argsort(shortlist, dim=1, stable=True)
        ordered_ids = shortlist.gather(1, id_order)
        ordered_scores = scores.gather(1, id_order)
        score_order = torch.argsort(
            ordered_scores, dim=1, descending=True, stable=True
        )
        top_positions = score_order[:, : self.retrieval]
        top_scores = ordered_scores.gather(1, top_positions)
        if not bool(torch.isfinite(top_scores).all()):
            raise RuntimeError("TabR shortlist did not contain enough non-self contexts")
        return ordered_ids.gather(1, top_positions)
