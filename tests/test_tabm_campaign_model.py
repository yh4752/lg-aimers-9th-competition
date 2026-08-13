from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.independent_dl.models.tabm import TabMAdapter
from experiments.tabm_campaign.model_state import (
    build_inference_metadata,
    fit_numeric_embedding_state,
    load_numeric_embedding_state,
    save_numeric_embedding_state,
)


class _FixedMembers(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        del x_cat
        return self.logits[: len(x_num)].unsqueeze(-1)


def test_tabm_brier_loss_uses_member_mean_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "experiments.independent_dl.models.tabm.import_runtime_module",
        lambda name: torch if name == "torch" else None,
    )
    logits = torch.tensor([[0.0, 1.0], [-1.0, 2.0], [2.0, -2.0]])
    model = _FixedMembers(logits)
    y = torch.tensor([0.0, 1.0, 1.0])
    loss = TabMAdapter(loss_name="brier").loss(
        model,
        torch.zeros((3, 2)),
        torch.zeros((3, 0), dtype=torch.long),
        y,
        row_indices=torch.arange(3),
    )
    expected = ((logits.sigmoid().mean(dim=1) - y) ** 2).mean()
    assert torch.allclose(loss, expected)


def test_tabm_default_loss_remains_bce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "experiments.independent_dl.models.tabm.import_runtime_module",
        lambda name: torch if name == "torch" else None,
    )
    adapter = TabMAdapter()
    assert adapter.loss_name == "bce"
    with pytest.raises(ValueError, match="loss"):
        TabMAdapter(loss_name="hybrid")


@pytest.mark.parametrize("mode", ["piecewise_linear", "periodic"])
def test_numeric_embedding_state_round_trip_without_training_matrix(
    mode: str, tmp_path: Path
) -> None:
    train_x_num = np.asarray(
        [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]], dtype="float32"
    )
    state = fit_numeric_embedding_state(mode, train_x_num)
    path = tmp_path / "numeric.json"
    save_numeric_embedding_state(state, path)
    restored = load_numeric_embedding_state(path)

    assert restored == state
    metadata = build_inference_metadata(restored, categorical_cardinalities=(4, 5))
    assert metadata.train_x_num is None
    assert metadata.n_num_features == 2
    if mode == "piecewise_linear":
        assert metadata.piecewise_bin_edges is not None
