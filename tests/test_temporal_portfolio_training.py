from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.independent_dl.models.tabm import TabMAdapter
from experiments.temporal_portfolio.tabm_training import TemporalTabMAdapter


class _FixedMembers(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        del x_cat
        return self.logits[: len(x_num)].unsqueeze(-1)


def _features(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros((rows, 2)), torch.zeros((rows, 0), dtype=torch.long)


def test_decay_weighted_bce_uses_complete_window_denominator() -> None:
    logits = torch.tensor([[0.0, 1.0], [-1.0, 2.0], [2.0, -2.0]])
    model = _FixedMembers(logits)
    adapter = TemporalTabMAdapter(
        sample_weight=np.array([1.0, 0.5, 0.25]), loss_name="bce"
    )
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)
    target = torch.tensor([1.0, 0.0])

    loss = adapter.loss_for_window(
        model,
        x_num,
        x_cat,
        target,
        row_indices=torch.tensor([0, 1]),
        window_indices=torch.tensor([0, 1, 2]),
    )

    member_targets = target.unsqueeze(1).expand_as(logits[:2])
    per_row = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:2], member_targets, reduction="none"
    ).mean(dim=1)
    expected = (per_row * torch.tensor([1.0, 0.5])).sum() / 1.75
    torch.testing.assert_close(loss, expected)


def test_teacher_changes_only_finite_matched_row_targets() -> None:
    logits = torch.tensor([[0.0, 1.0], [-1.0, 2.0], [2.0, -2.0]])
    adapter = TemporalTabMAdapter(
        sample_weight=np.ones(3),
        loss_name="bce",
        teacher_probability=np.array([0.9, np.nan, 0.1]),
        teacher_lambda=0.25,
    )
    adapter.bind_device("cpu")
    hard_target = torch.tensor([1.0, 0.0, 0.0])

    per_row = adapter.debug_per_row_loss(logits, hard_target)
    hard = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        hard_target.unsqueeze(1).expand_as(logits),
        reduction="none",
    ).mean(dim=1)
    blended_target = torch.tensor([0.975, 0.0, 0.025])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        blended_target.unsqueeze(1).expand_as(logits),
        reduction="none",
    ).mean(dim=1)

    torch.testing.assert_close(per_row[[0, 2]], expected[[0, 2]])
    assert torch.equal(per_row[1], hard[1])
    assert not torch.equal(per_row[0], hard[0])


def test_brier_uses_member_mean_probability_before_square_and_weight() -> None:
    logits = torch.tensor([[0.0, 2.0], [-2.0, 1.0], [1.0, 1.5]])
    model = _FixedMembers(logits)
    adapter = TemporalTabMAdapter(
        sample_weight=np.array([1.0, 0.5, 0.25]), loss_name="brier"
    )
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)
    target = torch.tensor([1.0, 0.0])

    loss = adapter.loss_for_window(
        model,
        x_num,
        x_cat,
        target,
        row_indices=torch.tensor([0, 1]),
        window_indices=torch.tensor([0, 1, 2]),
    )

    per_row = (logits[:2].sigmoid().mean(dim=1) - target).square()
    expected = (per_row * torch.tensor([1.0, 0.5])).sum() / 1.75
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(
    "sample_weight",
    [
        np.array([]),
        np.array([[1.0]]),
        np.array([1.0, 0.0]),
        np.array([1.0, -0.1]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([True, False]),
        np.array(["1.0", "0.5"]),
        np.array([1.0 + 0.0j]),
        np.array([np.finfo(np.float64).max]),
        np.array([np.nextafter(0.0, 1.0)]),
    ],
)
def test_invalid_weights_are_rejected(sample_weight: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="weight"):
        TemporalTabMAdapter(sample_weight=sample_weight, loss_name="bce")


def test_array_inputs_do_not_silently_coerce_python_sequences() -> None:
    with pytest.raises(TypeError, match="weight"):
        TemporalTabMAdapter(sample_weight=[1.0, 0.5], loss_name="bce")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="teacher"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2),
            loss_name="bce",
            teacher_probability=[0.5, np.nan],  # type: ignore[arg-type]
            teacher_lambda=0.25,
        )


@pytest.mark.parametrize(
    "teacher_probability",
    [
        np.array([]),
        np.array([[0.5, 0.5]]),
        np.array([0.5]),
        np.array([0.5, -0.1]),
        np.array([0.5, 1.1]),
        np.array([0.5, np.inf]),
        np.array([True, False]),
        np.array(["0.5", "nan"]),
        np.array([0.5 + 0.0j, 0.2 + 0.0j]),
    ],
)
def test_invalid_teacher_probabilities_are_rejected(
    teacher_probability: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError), match="teacher"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2),
            loss_name="bce",
            teacher_probability=teacher_probability,
            teacher_lambda=0.25,
        )


@pytest.mark.parametrize("teacher_lambda", [True, "0.25", np.nan, np.inf, -0.1, 1.1])
def test_invalid_teacher_lambdas_are_rejected(teacher_lambda: object) -> None:
    with pytest.raises((TypeError, ValueError), match="lambda"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2),
            loss_name="bce",
            teacher_probability=np.array([0.5, np.nan]),
            teacher_lambda=teacher_lambda,
        )


def test_teacher_contract_rejects_ambiguous_combinations() -> None:
    with pytest.raises(ValueError, match="teacher_lambda"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2), loss_name="bce", teacher_lambda=0.25
        )
    with pytest.raises(ValueError, match="Brier"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2),
            loss_name="brier",
            teacher_probability=np.array([0.5, np.nan]),
            teacher_lambda=0.0,
        )


def test_constructor_detaches_and_protects_caller_arrays() -> None:
    sample_weight = np.array([1.0, 0.5])
    teacher_probability = np.array([0.9, np.nan])
    adapter = TemporalTabMAdapter(
        sample_weight=sample_weight,
        loss_name="bce",
        teacher_probability=teacher_probability,
        teacher_lambda=0.25,
    )

    sample_weight[:] = 99.0
    teacher_probability[:] = 0.1

    np.testing.assert_array_equal(adapter.sample_weight_np, [1.0, 0.5])
    np.testing.assert_allclose(adapter.teacher_probability_np, [0.9, np.nan], equal_nan=True)
    assert not adapter.sample_weight_np.flags.writeable
    assert adapter.teacher_probability_np is not None
    assert not adapter.teacher_probability_np.flags.writeable


def test_build_calls_parent_and_binds_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[tuple[object, object, str]] = []

    def fake_build(
        self: TabMAdapter, model_config: object, metadata: object, device: str
    ) -> object:
        calls.append((model_config, metadata, device))
        return sentinel

    monkeypatch.setattr(TabMAdapter, "build", fake_build)
    monkeypatch.setattr(
        "experiments.temporal_portfolio.tabm_training.import_runtime_module",
        lambda name: torch if name == "torch" else None,
    )
    adapter = TemporalTabMAdapter(
        sample_weight=np.array([1.0, 0.5]), loss_name="bce"
    )
    model_config: dict[str, object] = {}
    metadata = object()

    result = adapter.build(model_config, metadata, "cpu")

    assert result is sentinel
    assert calls == [(model_config, metadata, "cpu")]
    assert adapter._weight is not None
    assert adapter._weight.device == torch.device("cpu")


def test_unbound_loss_fails_clearly() -> None:
    model = _FixedMembers(torch.zeros((2, 2)))
    adapter = TemporalTabMAdapter(sample_weight=np.ones(2), loss_name="bce")
    x_num, x_cat = _features(2)

    with pytest.raises(RuntimeError, match="bind_device"):
        adapter.loss_for_window(
            model,
            x_num,
            x_cat,
            torch.tensor([0.0, 1.0]),
            row_indices=torch.tensor([0, 1]),
            window_indices=torch.tensor([0, 1]),
        )


def test_weighted_loss_gradients_remain_finite() -> None:
    model = _FixedMembers(torch.tensor([[0.0, 1.0], [-1.0, 2.0]]))
    adapter = TemporalTabMAdapter(
        sample_weight=np.array([1.0, 0.5]),
        loss_name="bce",
        teacher_probability=np.array([0.9, np.nan]),
        teacher_lambda=0.25,
    )
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)
    loss = adapter.loss_for_window(
        model,
        x_num,
        x_cat,
        torch.tensor([1.0, 0.0]),
        row_indices=torch.tensor([0, 1]),
        window_indices=torch.tensor([0, 1]),
    )

    loss.backward()

    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()


@pytest.mark.parametrize(
    ("row_indices", "window_indices", "message"),
    [
        (torch.tensor([0.0, 1.0]), torch.tensor([0, 1]), "dtype"),
        (torch.tensor([[0, 1]]), torch.tensor([0, 1]), "one-dimensional"),
        (torch.tensor([0, 2]), torch.tensor([0, 1]), "range"),
        (torch.tensor([0, 1]), torch.tensor([-1, 1]), "range"),
    ],
)
def test_loss_rejects_invalid_index_tensors(
    row_indices: torch.Tensor, window_indices: torch.Tensor, message: str
) -> None:
    model = _FixedMembers(torch.zeros((2, 2)))
    adapter = TemporalTabMAdapter(sample_weight=np.ones(2), loss_name="bce")
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)

    with pytest.raises((TypeError, ValueError, IndexError, RuntimeError), match=message):
        adapter.loss_for_window(
            model,
            x_num,
            x_cat,
            torch.tensor([0.0, 1.0]),
            row_indices=row_indices,
            window_indices=window_indices,
        )


def test_loss_rejects_index_device_mismatch() -> None:
    model = _FixedMembers(torch.zeros((2, 2)))
    adapter = TemporalTabMAdapter(sample_weight=np.ones(2), loss_name="bce")
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)

    with pytest.raises(ValueError, match="device"):
        adapter.loss_for_window(
            model,
            x_num,
            x_cat,
            torch.tensor([0.0, 1.0]),
            row_indices=torch.tensor([0, 1], device="meta"),
            window_indices=torch.tensor([0, 1]),
        )
