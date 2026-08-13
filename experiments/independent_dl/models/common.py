"""Dependency-light contracts shared by the DL model adapters."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Mapping, Protocol

import numpy as np

from ..features import FeatureBatch


INSTALL_COMMAND = (
    "python -m pip install -r experiments/independent_dl/requirements-colab.txt"
)


class DLRuntimeDependencyError(RuntimeError):
    """Raised only when a user starts a model without its Colab dependencies."""


@dataclass(frozen=True)
class ModelMetadata:
    n_num_features: int
    categorical_cardinalities: tuple[int, ...]
    train_x_num: np.ndarray | None
    piecewise_bin_edges: tuple[np.ndarray, ...] | None = None


class ModelAdapter(Protocol):
    def build(
        self,
        model_config: Mapping[str, object],
        metadata: ModelMetadata,
        device: str,
    ) -> object: ...

    def loss(
        self,
        model: object,
        x_num: object,
        x_cat: object,
        y: object,
        *,
        row_indices: object,
    ) -> object: ...

    def probabilities(
        self, model: object, x_num: object, x_cat: object
    ) -> object: ...

    def optimizer(
        self,
        model: object,
        training_config: Mapping[str, object],
    ) -> object: ...


def import_runtime_module(name: str) -> object:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise DLRuntimeDependencyError(
            f"필수 DL 패키지 {name!r}가 없습니다. Colab에서 다음 명령을 실행하세요: "
            f"{INSTALL_COMMAND}"
        ) from error


def metadata_from_train(batch: FeatureBatch) -> ModelMetadata:
    x_num = np.asarray(batch.x_num)
    x_cat = np.asarray(batch.x_cat)
    if x_num.ndim != 2 or x_cat.ndim != 2 or len(x_num) != len(x_cat):
        raise ValueError("train feature arrays must be aligned two-dimensional matrices")
    if not np.isfinite(x_num).all():
        raise ValueError("train numerical features must be finite")
    cardinalities: list[int] = []
    for position in range(x_cat.shape[1]):
        values = x_cat[:, position]
        if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
            values.dtype, np.integer
        ):
            raise ValueError("categorical features must contain integer indices")
        if (values < 0).any():
            raise ValueError("categorical feature indices must be non-negative")
        cardinalities.append(int(values.max(initial=0)) + 1)
    return ModelMetadata(
        n_num_features=x_num.shape[1],
        categorical_cardinalities=tuple(cardinalities),
        train_x_num=batch.x_num,
    )


def quantile_bin_edges(values: np.ndarray, *, n_bins: int = 48) -> tuple[np.ndarray, ...]:
    """Compute train-only edges accepted by PiecewiseLinearEmbeddings.

    Constant columns deliberately receive one edge. Version-B piecewise embeddings
    retain their linear component for those columns instead of dropping features.
    """

    array = np.asarray(values, dtype="float64")
    if array.ndim != 2 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("piecewise embeddings require a finite 2-D training matrix")
    quantiles = np.linspace(0.0, 1.0, min(n_bins, len(array) - 1) + 1)
    result: list[np.ndarray] = []
    for position in range(array.shape[1]):
        edges = np.unique(np.quantile(array[:, position], quantiles)).astype(
            "float32"
        )
        if len(edges) == 1:
            value = edges[0]
            edges = np.array(
                [
                    np.nextafter(value, np.float32("-inf")),
                    np.nextafter(value, np.float32("inf")),
                ],
                dtype="float32",
            )
        if not np.isfinite(edges).all():
            raise ValueError("piecewise bin edges must remain finite")
        result.append(edges)
    return tuple(result)


def make_numeric_embeddings(
    mode: str,
    metadata: ModelMetadata,
    torch: object,
    rtdl_num_embeddings: object,
) -> object:
    n_features = metadata.n_num_features
    if mode == "linear_relu":
        return rtdl_num_embeddings.LinearReLUEmbeddings(n_features, 32)
    if mode == "periodic":
        return rtdl_num_embeddings.PeriodicEmbeddings(
            n_features,
            32,
            n_frequencies=48,
            frequency_init_scale=0.01,
            activation=True,
            lite=False,
        )
    if mode == "piecewise_linear":
        fitted_edges = metadata.piecewise_bin_edges
        if fitted_edges is None:
            if metadata.train_x_num is None:
                raise ValueError(
                    "piecewise embeddings require fitted edges or a training matrix"
                )
            fitted_edges = quantile_bin_edges(metadata.train_x_num)
        bins = [
            torch.as_tensor(edge, dtype=torch.float32)
            for edge in fitted_edges
        ]
        return rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins,
            32,
            activation=False,
            version="B",
        )
    raise ValueError(f"unsupported numerical embedding: {mode}")


def make_categorical_backbone_wrapper(
    torch: object,
    backbone: object,
    cardinalities: tuple[int, ...],
    embedding_dim: int,
) -> object:
    """Prepend shared categorical embeddings to an MLP-like backbone."""

    nn = torch.nn

    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.embeddings = nn.ModuleList(
                nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities
            )
            self.activation_checkpointing = False

        def enable_activation_checkpointing(self) -> None:
            self.activation_checkpointing = True

        def _forward(self, x_num: object, x_cat: object) -> object:
            pieces = [x_num]
            pieces.extend(
                embedding(x_cat[:, index])
                for index, embedding in enumerate(self.embeddings)
            )
            return self.backbone(torch.cat(pieces, dim=1))

        def forward(self, x_num: object, x_cat: object) -> object:
            if self.activation_checkpointing and self.training:
                checkpoint = import_runtime_module("torch.utils.checkpoint")
                return checkpoint.checkpoint(
                    self._forward, x_num, x_cat, use_reentrant=False
                )
            return self._forward(x_num, x_cat)

    return _Wrapper()


def make_two_input_checkpoint_wrapper(torch: object, model: object) -> object:
    nn = torch.nn

    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.activation_checkpointing = False

        def enable_activation_checkpointing(self) -> None:
            self.activation_checkpointing = True

        def forward(self, x_num: object, x_cat: object) -> object:
            if self.activation_checkpointing and self.training:
                checkpoint = import_runtime_module("torch.utils.checkpoint")
                return checkpoint.checkpoint(
                    self.model, x_num, x_cat, use_reentrant=False
                )
            return self.model(x_num, x_cat)

    return _Wrapper()
