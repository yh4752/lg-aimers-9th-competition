from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from experiments.independent_dl.models.common import ModelMetadata
from experiments.independent_dl.models.tabnet import TabNetAdapter


class _FakeTabNetNoEmbeddings(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        group_attention_matrix: torch.Tensor | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.group_attention_matrix = group_attention_matrix

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear(values), values.new_tensor(0.25)


def test_tabnet_adapter_builds_two_input_binary_model(monkeypatch) -> None:
    fake_tabnet = SimpleNamespace(TabNetNoEmbeddings=_FakeTabNetNoEmbeddings)

    def fake_import(name: str):
        if name == "torch":
            return torch
        if name == "pytorch_tabnet.tab_network":
            return fake_tabnet
        raise AssertionError(name)

    monkeypatch.setattr(
        "experiments.independent_dl.models.tabnet.import_runtime_module", fake_import
    )
    adapter = TabNetAdapter()
    metadata = ModelMetadata(
        n_num_features=3,
        categorical_cardinalities=(4, 5),
        train_x_num=np.zeros((8, 3), dtype="float32"),
    )
    config = {
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.5,
        "lambda_sparse": 0.0001,
        "momentum": 0.02,
        "mask_type": "sparsemax",
        "embedding_dim": 4,
    }
    model = adapter.build(config, metadata, "cpu")
    x_num = torch.zeros((6, 3), dtype=torch.float32)
    x_cat = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [0, 0], [1, 1]])

    probabilities = adapter.probabilities(model, x_num, x_cat)
    loss = adapter.loss(
        model,
        x_num,
        x_cat,
        torch.tensor([0, 1, 0, 1, 0, 1]),
        row_indices=torch.arange(6),
    )

    assert tuple(probabilities.shape) == (6,)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert loss.ndim == 0


def test_tabnet_adapter_builds_attention_matrix_on_training_device(monkeypatch) -> None:
    fake_tabnet = SimpleNamespace(TabNetNoEmbeddings=_FakeTabNetNoEmbeddings)

    def fake_import(name: str):
        if name == "torch":
            return torch
        if name == "pytorch_tabnet.tab_network":
            return fake_tabnet
        raise AssertionError(name)

    monkeypatch.setattr(
        "experiments.independent_dl.models.tabnet.import_runtime_module", fake_import
    )
    adapter = TabNetAdapter()
    metadata = ModelMetadata(
        n_num_features=3,
        categorical_cardinalities=(4, 5),
        train_x_num=np.zeros((8, 3), dtype="float32"),
    )
    config = {
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.5,
        "lambda_sparse": 0.0001,
        "momentum": 0.02,
        "mask_type": "sparsemax",
        "embedding_dim": 4,
    }

    model = adapter.build(config, metadata, "cpu")

    attention = model.tabnet.group_attention_matrix
    assert attention is not None
    assert tuple(attention.shape) == (11, 11)
    assert attention.device.type == "cpu"
