"""Fold-safe TabR adapter with FAISS GPU retrieval and exact reranking.

The E/R/P structure and self-neighbor masking follow the official TabR source
at commit 17baa9082506f8e7a0f8d11bb1e08212926a1507. The candidate pool remains
the current fold's training rows and every approximate shortlist is reranked
with the original exact squared-L2 expression.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from ..features import FeatureBatch
from .common import ModelMetadata, import_runtime_module
from .tabr_search import (
    FoldTrainFaissIndex,
    TABR_SEARCH_POLICY,
    probe_faiss_gpu,
)


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
        self._context_keys: np.ndarray | None = None
        self._cached_model: object | None = None
        self._refresh_generation = 0
        self._searched_queries = 0
        self._search_started = 0.0
        self._output_dir: Path | None = None
        self._identity: dict[str, object] | None = None
        self._frozen_neighbors: np.ndarray | None = None
        self._frozen_sha256: str | None = None

    def set_progress_reporter(self, reporter: object) -> None:
        self._reporter = reporter

    @property
    def has_frozen_contexts(self) -> bool:
        return self._frozen_neighbors is not None

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def configure_checkpoint(
        self, output_dir: str | Path, model_config: Mapping[str, object]
    ) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        row_digest = sha256()
        for value in self.context.row_id.astype(str):
            encoded = value.encode("utf-8")
            row_digest.update(len(encoded).to_bytes(8, "big"))
            row_digest.update(encoded)
        config_encoded = json.dumps(
            dict(model_config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        identity = {
            "policy_version": TABR_SEARCH_POLICY["version"],
            "context_row_id_sha256": row_digest.hexdigest(),
            "model_config_sha256": sha256(config_encoded).hexdigest(),
            "fold_cutoff_year": int(np.asarray(self.context.season).max()),
        }
        if self._identity is not None and self._identity != identity:
            self._frozen_neighbors = None
            self._frozen_sha256 = None
            self.release_search_index()
        self._output_dir = root
        self._identity = identity

    def checkpoint_state(self) -> dict[str, object] | None:
        if self._identity is None:
            return None
        state = dict(self._identity)
        if self._frozen_neighbors is not None:
            if self._output_dir is None:
                raise RuntimeError("TabR checkpoint output directory is not configured")
            if self._frozen_sha256 is None:
                raise RuntimeError("TabR frozen-neighbor hash is missing")
            path = self._output_dir / "frozen_neighbors.npy"
            state.update(
                {
                    "frozen_neighbors_file": path.name,
                    "frozen_neighbors_shape": list(self._frozen_neighbors.shape),
                    "frozen_neighbors_sha256": self._frozen_sha256,
                }
            )
        return state

    def restore_checkpoint_state(
        self, payload: object, model: object, device: str
    ) -> bool:
        self._frozen_neighbors = None
        self._frozen_sha256 = None
        if self._identity is None or self._output_dir is None:
            return False
        if not isinstance(payload, Mapping):
            return False
        if any(payload.get(key) != value for key, value in self._identity.items()):
            return False
        if payload.get("frozen_neighbors_file") != "frozen_neighbors.npy":
            return False
        shape = payload.get("frozen_neighbors_shape")
        expected_shape = [len(self.context.row_id), self.retrieval]
        expected_sha = payload.get("frozen_neighbors_sha256")
        if shape != expected_shape or not isinstance(expected_sha, str):
            return False
        path = self._output_dir / "frozen_neighbors.npy"
        if not path.is_file() or self._file_sha256(path) != expected_sha:
            return False
        try:
            frozen = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            return False
        if frozen.dtype != np.dtype("int64") or list(frozen.shape) != expected_shape:
            return False
        for start in range(0, len(frozen), self.candidate_chunk_size):
            stop = min(start + self.candidate_chunk_size, len(frozen))
            chunk = np.asarray(frozen[start:stop])
            if (chunk < 0).any() or (chunk >= len(frozen)).any():
                return False
            rows = np.arange(start, stop, dtype="int64")[:, None]
            if (chunk == rows).any():
                return False
        self._frozen_neighbors = frozen
        self._frozen_sha256 = expected_sha
        self._cached_model = model
        return True

    def release_search_index(self) -> None:
        self._search_index = None
        self._context_keys = None

    def prepare_initial_keys(self, model: object, device: str) -> None:
        if self._frozen_neighbors is not None:
            self._cached_model = model
            return
        self.refresh_keys(model, device)

    def freeze_contexts(self, model: object, *, epoch: int, device: str) -> None:
        if self._frozen_neighbors is not None:
            return
        if self._output_dir is None or self._identity is None:
            raise RuntimeError("TabR checkpoint output directory is not configured")
        if self._search_index is None or self._context_keys is None:
            raise RuntimeError("TabR index must be ready before contexts are frozen")
        if model is not self._cached_model:
            raise RuntimeError("TabR index belongs to another model")
        torch = import_runtime_module("torch")
        final_path = self._output_dir / "frozen_neighbors.npy"
        temporary = final_path.with_name(final_path.name + ".tmp")
        frozen = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype="int64",
            shape=(len(self.context.row_id), self.retrieval),
        )
        stream = f"tabr_freeze_{self._refresh_generation}"
        started = time.monotonic()
        try:
            for start in range(
                0, len(self.context.row_id), self.candidate_chunk_size
            ):
                stop = min(
                    start + self.candidate_chunk_size, len(self.context.row_id)
                )
                queries = torch.as_tensor(
                    self._context_keys[start:stop],
                    dtype=torch.float32,
                    device=device,
                )
                row_indices = torch.arange(
                    start, stop, dtype=torch.long, device=device
                )
                neighbors = self._search_index.search_and_rerank(
                    queries, query_absolute_indices=row_indices
                )
                frozen[start:stop] = neighbors.detach().cpu().numpy()
                if self._reporter is not None and self._reporter.should_emit(
                    completed_rows=stop, stream=stream
                ):
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    self._reporter.progress(
                        "TABR_SEARCH_PROGRESS",
                        completed_rows=stop,
                        total_rows=len(self.context.row_id),
                        started_at=started,
                        gpu=self._gpu_memory(torch),
                        stream=stream,
                        purpose="freeze_training_neighbors",
                    )
            frozen.flush()
            del frozen
            os.replace(temporary, final_path)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        self._frozen_neighbors = np.load(
            final_path, mmap_mode="r", allow_pickle=False
        )
        digest = self._file_sha256(final_path)
        self._frozen_sha256 = digest
        if self._reporter is not None:
            self._reporter.emit(
                "TABR_CONTEXTS_FROZEN",
                epoch=epoch,
                shape=list(self._frozen_neighbors.shape),
                sha256=digest,
            )

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
        self._context_keys = cached_keys.numpy()
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
        if self._frozen_neighbors is None and self._search_index is None:
            raise RuntimeError("TabR retrieval keys require refresh before search")
        if model is not self._cached_model:
            raise RuntimeError("TabR retrieval keys belong to another model")
        _, query_keys = model.encode(x_num, x_cat)
        return self._search_keys(query_keys, row_indices=row_indices)

    def _search_keys(self, query_keys: object, *, row_indices: object | None) -> object:
        torch = import_runtime_module("torch")
        if row_indices is not None and self._frozen_neighbors is not None:
            indices = row_indices.detach().to(device="cpu").numpy()
            if (
                indices.ndim != 1
                or len(indices) != len(query_keys)
                or (indices < 0).any()
                or (indices >= len(self._frozen_neighbors)).any()
            ):
                raise RuntimeError("TabR frozen row indices are invalid")
            return torch.as_tensor(
                np.asarray(self._frozen_neighbors[indices]),
                dtype=torch.long,
                device=query_keys.device,
            )
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
        self._output_dir: Path | None = None

    def set_progress_reporter(self, reporter: object) -> None:
        self._reporter = reporter
        if self._runtime is not None and hasattr(
            self._runtime, "set_progress_reporter"
        ):
            self._runtime.set_progress_reporter(reporter)

    def set_output_dir(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)

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
        if self._output_dir is None:
            raise TabRContractError("TabR output directory is not configured")
        configure = getattr(self._runtime, "configure_checkpoint", None)
        if configure is not None:
            configure(self._output_dir, model_config)
        return _build_torch_model(model_config, metadata, device)

    def prepare_initial_retrieval_cache(self, model: object, device: str) -> None:
        if self._runtime is None:
            raise TabRContractError("TabR runtime is not prepared")
        prepare = getattr(self._runtime, "prepare_initial_keys", None)
        if prepare is None:
            self._runtime.refresh_keys(model, device)
        else:
            prepare(model, device)

    def on_epoch_start(self, model: object, epoch: int, device: str) -> None:
        if self._runtime is None:
            raise TabRContractError("TabR runtime is not prepared")
        if epoch < int(TABR_SEARCH_POLICY["freeze_after_epochs"]):
            return
        if bool(getattr(self._runtime, "has_frozen_contexts", False)):
            return
        if getattr(self._runtime, "_search_index", None) is None:
            self._runtime.refresh_keys(model, device)
        self._runtime.freeze_contexts(model, epoch=epoch - 1, device=device)

    def on_epoch_end(self, model: object, epoch: int, device: str) -> None:
        if self._runtime is None:
            raise TabRContractError("TabR runtime is not prepared")
        if epoch + 1 == int(TABR_SEARCH_POLICY["freeze_after_epochs"]):
            self._runtime.freeze_contexts(model, epoch=epoch, device=device)
        if bool(getattr(self._runtime, "has_frozen_contexts", False)):
            release = getattr(self._runtime, "release_search_index", None)
            if release is not None:
                release()

    def checkpoint_state(self) -> object | None:
        if self._runtime is None:
            return None
        getter = getattr(self._runtime, "checkpoint_state", None)
        return None if getter is None else getter()

    def restore_checkpoint_state(
        self, payload: object, model: object, device: str
    ) -> bool:
        if self._runtime is None:
            return False
        restore = getattr(self._runtime, "restore_checkpoint_state", None)
        return False if restore is None else bool(restore(payload, model, device))

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
