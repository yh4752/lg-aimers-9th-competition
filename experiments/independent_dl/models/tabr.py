"""Fold-safe, chunked TabR adapter.

The E/R/P structure and self-neighbor masking follow the official TabR source
at commit 17baa9082506f8e7a0f8d11bb1e08212926a1507. This adapter replaces the
upstream experiment harness and FAISS dependency with exact chunked PyTorch
top-k search so the candidate pool remains the current fold's training rows.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..features import FeatureBatch
from .common import ModelMetadata, import_runtime_module


class TabRContractError(ValueError):
    """Raised before retrieval can cross a fold boundary."""


class _TorchTabRRuntime:
    def __init__(self, context: FeatureBatch, retrieval: int) -> None:
        if context.y is None:
            raise TabRContractError("TabR context requires training targets")
        if len(context.row_id) <= retrieval:
            raise TabRContractError("TabR retrieval must be smaller than the train fold")
        self.context = context
        self.retrieval = retrieval
        self.candidate_chunk_size = 8192

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
        top_scores = None
        top_indices = None
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
                _, candidate_k = model.encode(candidate_num, candidate_cat)
                scores = (
                    -query_k.square().sum(-1, keepdim=True)
                    + 2.0 * (query_k @ candidate_k.T)
                    - candidate_k.square().sum(-1).unsqueeze(0)
                )
                candidate_indices = torch.arange(start, stop, device=device)
                if row_indices is not None:
                    scores = scores.masked_fill(
                        candidate_indices.unsqueeze(0).eq(row_indices.unsqueeze(1)),
                        float("-inf"),
                    )
                expanded_indices = candidate_indices.unsqueeze(0).expand(
                    len(query_k), -1
                )
                if top_scores is not None:
                    scores = torch.cat([top_scores, scores], dim=1)
                    expanded_indices = torch.cat(
                        [top_indices, expanded_indices], dim=1
                    )
                keep = min(self.retrieval, scores.shape[1])
                top_scores, positions = torch.topk(
                    scores, k=keep, dim=1, largest=True, sorted=False
                )
                top_indices = expanded_indices.gather(1, positions)
        if top_indices is None or top_indices.shape[1] != self.retrieval:
            raise RuntimeError("TabR search did not produce the requested contexts")

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
                self._context, int(model_config["retrieval"])
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

    def optimizer(
        self, model: object, training_config: Mapping[str, object]
    ) -> object:
        torch = import_runtime_module("torch")
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
