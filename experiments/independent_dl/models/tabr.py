"""Fold-safe TabR adapter with FAISS GPU retrieval and exact reranking.

The E/R/P structure and self-neighbor masking follow the official TabR source
at commit 17baa9082506f8e7a0f8d11bb1e08212926a1507. The candidate pool remains
the current fold's training rows and every approximate shortlist is reranked
with the original exact squared-L2 expression.
"""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from ..features import FeatureBatch
from .common import ModelMetadata, import_runtime_module
from .tabr_search import FoldTrainFaissIndex, probe_faiss_gpu


class TabRContractError(ValueError):
    """Raised before retrieval can cross a fold boundary."""


class _TorchTabRRuntime:
    def __init__(
        self,
        context: FeatureBatch,
        retrieval: int,
        *,
        faiss_module: object | None = None,
        reporter: object | None = None,
    ) -> None:
        if context.y is None:
            raise TabRContractError("TabR context requires training targets")
        if len(context.row_id) <= retrieval:
            raise TabRContractError("TabR retrieval must be smaller than the train fold")
        self.context = context
        self.retrieval = retrieval
        self.candidate_chunk_size = 8192
        self._faiss_module = faiss_module
        self._reporter = reporter
        self._search_index: FoldTrainFaissIndex | None = None
        self._cached_model: object | None = None
        self._refresh_generation = 0
        self._searched_queries = 0
        self._search_started = 0.0

    def set_progress_reporter(self, reporter: object) -> None:
        self._reporter = reporter

    @staticmethod
    def _gpu_memory(torch: object) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {"allocated_bytes": 0, "reserved_bytes": 0, "peak_bytes": 0}
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_bytes": int(torch.cuda.max_memory_allocated()),
        }

    def refresh_keys(self, model: object, device: str) -> None:
        probe_faiss_gpu(faiss_module=self._faiss_module)
        torch = import_runtime_module("torch")
        self._refresh_generation += 1
        self._searched_queries = 0
        was_training = bool(model.training)
        model.eval()
        keys: list[object] = []
        encoding_started = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(self.context.row_id), self.candidate_chunk_size):
                stop = min(start + self.candidate_chunk_size, len(self.context.row_id))
                candidate_num = torch.as_tensor(
                    np.asarray(self.context.x_num[start:stop]),
                    dtype=torch.float32,
                    device=device,
                )
                candidate_cat = torch.as_tensor(
                    np.asarray(self.context.x_cat[start:stop]),
                    dtype=torch.long,
                    device=device,
                )
                _, candidate_keys = model.encode(candidate_num, candidate_cat)
                keys.append(candidate_keys.detach().to(dtype=torch.float32, device="cpu"))
                completed_rows = stop
                stream = f"tabr_context_encoding_{self._refresh_generation}"
                if self._reporter is not None and self._reporter.should_emit(
                    completed_rows=completed_rows, stream=stream
                ):
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    self._reporter.progress(
                        "TABR_CONTEXT_ENCODING_PROGRESS",
                        completed_rows=completed_rows,
                        total_rows=len(self.context.row_id),
                        started_at=encoding_started,
                        gpu=self._gpu_memory(torch),
                        stream=stream,
                    )
        model.train(was_training)
        cached_keys = torch.cat(keys, dim=0).contiguous()
        self._search_index = FoldTrainFaissIndex.build(
            cached_keys.numpy(),
            np.arange(len(self.context.row_id), dtype="int64"),
            retrieval=self.retrieval,
            reporter=self._reporter,
            faiss_module=self._faiss_module,
        )
        self._search_started = time.monotonic()
        self._cached_model = model

    def search(
        self,
        model: object,
        x_num: object,
        x_cat: object,
        *,
        row_indices: object | None,
    ) -> object:
        if self._search_index is None:
            raise RuntimeError("TabR retrieval keys require refresh before search")
        if model is not self._cached_model:
            raise RuntimeError("TabR retrieval keys belong to another model")
        _, query_keys = model.encode(x_num, x_cat)
        return self._search_keys(query_keys, row_indices=row_indices)

    def _search_keys(self, query_keys: object, *, row_indices: object | None) -> object:
        torch = import_runtime_module("torch")
        if self._search_index is None:
            raise RuntimeError("TabR retrieval keys require refresh before search")
        with torch.no_grad():
            top_indices = self._search_index.search_and_rerank(
                query_keys,
                query_absolute_indices=row_indices,
            )
        if self._reporter is not None:
            stream = f"tabr_search_{self._refresh_generation}"
            self._searched_queries += len(query_keys)
            completed_rows = self._searched_queries
            if self._reporter.should_emit(
                completed_rows=completed_rows, stream=stream
            ):
                if str(query_keys.device).startswith("cuda"):
                    torch.cuda.synchronize()
                self._reporter.progress(
                    "TABR_SEARCH_PROGRESS",
                    completed_rows=completed_rows,
                    total_rows=max(len(self.context.row_id), completed_rows),
                    started_at=self._search_started,
                    gpu=self._gpu_memory(torch),
                    stream=stream,
                )
        return top_indices

    def probabilities(self, model: object, x_num: object, x_cat: object) -> object:
        return self._forward(model, x_num, x_cat, row_indices=None).sigmoid()

    def loss(
        self,
        model: object,
        x_num: object,
        x_cat: object,
        y: object,
        row_indices: object,
    ) -> object:
        torch = import_runtime_module("torch")
        logits = self._forward(model, x_num, x_cat, row_indices=row_indices)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y.float()
        )

    def _forward(
        self,
        model: object,
        x_num: object,
        x_cat: object,
        *,
        row_indices: object | None,
    ) -> object:
        torch = import_runtime_module("torch")
        device = x_num.device
        query_x, query_k = model.encode(x_num, x_cat)
        top_indices = self._search_keys(query_k, row_indices=row_indices)

        flat_indices = top_indices.detach().cpu().numpy().reshape(-1)
        context_num = torch.as_tensor(
            np.asarray(self.context.x_num[flat_indices]),
            dtype=torch.float32,
            device=device,
        )
        context_cat = torch.as_tensor(
            np.asarray(self.context.x_cat[flat_indices]),
            dtype=torch.long,
            device=device,
        )
        _, context_k = model.encode(context_num, context_cat)
        context_k = context_k.reshape(len(query_k), self.retrieval, -1)
        similarities = (
            -query_k.square().sum(-1, keepdim=True)
            + 2.0 * (query_k[:, None, :] * context_k).sum(-1)
            - context_k.square().sum(-1)
        )
        probabilities = model.context_dropout(similarities.softmax(dim=-1))
        context_y = torch.as_tensor(
            np.asarray(self.context.y[flat_indices]),
            dtype=torch.float32,
            device=device,
        ).reshape(len(query_k), self.retrieval, 1)
        values = model.label_encoder(context_y) + model.context_transform(
            query_k[:, None, :] - context_k
        )
        retrieved = (probabilities.unsqueeze(-1) * values).sum(dim=1)
        return model.predict(query_x + retrieved).squeeze(-1)


def _build_torch_model(
    model_config: Mapping[str, object], metadata: ModelMetadata, device: str
) -> object:
    torch = import_runtime_module("torch")
    checkpoint = import_runtime_module("torch.utils.checkpoint")
    nn = torch.nn
    width = int(model_config["width"])
    blocks = int(model_config["blocks"])
    dropout = float(model_config["dropout"])
    embedding_dim = 16
    d_in = metadata.n_num_features + embedding_dim * len(
        metadata.categorical_cardinalities
    )
    encoder_blocks = max(1, blocks // 2)
    predictor_blocks = max(1, blocks - encoder_blocks)

    def make_block() -> object:
        return nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    class _TabRModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                nn.Embedding(cardinality, embedding_dim)
                for cardinality in metadata.categorical_cardinalities
            )
            self.input = nn.Linear(d_in, width)
            self.encoder = nn.ModuleList(make_block() for _ in range(encoder_blocks))
            self.key = nn.Linear(width, width)
            self.label_encoder = nn.Linear(1, width)
            self.context_transform = nn.Sequential(
                nn.Linear(width, width * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, width, bias=False),
            )
            self.context_dropout = nn.Dropout(dropout)
            self.predictor = nn.ModuleList(
                make_block() for _ in range(predictor_blocks)
            )
            self.head = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))
            self.activation_checkpointing = False

        def enable_activation_checkpointing(self) -> None:
            self.activation_checkpointing = True

        def _embed(self, x_num: object, x_cat: object) -> object:
            pieces = [x_num]
            pieces.extend(
                embedding(x_cat[:, position])
                for position, embedding in enumerate(self.embeddings)
            )
            return self.input(torch.cat(pieces, dim=1))

        def encode(self, x_num: object, x_cat: object) -> tuple[object, object]:
            x = self._embed(x_num, x_cat)
            for block in self.encoder:
                residual = (
                    checkpoint.checkpoint(block, x, use_reentrant=False)
                    if self.activation_checkpointing and self.training
                    else block(x)
                )
                x = x + residual
            return x, self.key(x)

        def predict(self, x: object) -> object:
            for block in self.predictor:
                residual = (
                    checkpoint.checkpoint(block, x, use_reentrant=False)
                    if self.activation_checkpointing and self.training
                    else block(x)
                )
                x = x + residual
            return self.head(x)

    return _TabRModel().to(device)


class TabRAdapter:
    def __init__(self, runtime: object | None = None) -> None:
        self._runtime = runtime
        self._context: FeatureBatch | None = None
        self._reporter: object | None = None

    def set_progress_reporter(self, reporter: object) -> None:
        self._reporter = reporter
        if self._runtime is not None and hasattr(
            self._runtime, "set_progress_reporter"
        ):
            self._runtime.set_progress_reporter(reporter)

    def fit_context(self, batch: FeatureBatch) -> None:
        if batch.y is None:
            raise TabRContractError("TabR context requires training targets")
        if len(set(batch.row_id.astype(str).tolist())) != len(batch.row_id):
            raise TabRContractError("TabR context row IDs must be unique")
        self._context = batch
        if self._runtime is not None and hasattr(self._runtime, "fit_context"):
            self._runtime.fit_context(batch)

    def build(
        self,
        model_config: Mapping[str, object],
        metadata: ModelMetadata,
        device: str,
    ) -> object:
        if self._context is None:
            raise TabRContractError("fit_context must run before TabR build")
        if self._runtime is None:
            self._runtime = _TorchTabRRuntime(
                self._context,
                int(model_config["retrieval"]),
                reporter=self._reporter,
            )
        return _build_torch_model(model_config, metadata, device)

    def loss(
        self,
        model: object,
        x_num: object,
        x_cat: object,
        y: object,
        *,
        row_indices: object,
    ) -> object:
        if self._runtime is None:
            raise TabRContractError("TabR runtime is not prepared")
        return self._runtime.loss(model, x_num, x_cat, y, row_indices)

    def probabilities(self, model: object, x_num: object, x_cat: object) -> object:
        if self._runtime is None:
            raise TabRContractError("TabR runtime is not prepared")
        return self._runtime.probabilities(model, x_num, x_cat)

    def refresh_retrieval_cache(self, model: object, device: str) -> None:
        if self._runtime is None:
            raise TabRContractError("TabR runtime is not prepared")
        self._runtime.refresh_keys(model, device)

    def optimizer(
        self, model: object, training_config: Mapping[str, object]
    ) -> object:
        torch = import_runtime_module("torch")
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
