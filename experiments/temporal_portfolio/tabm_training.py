"""Window-normalized temporal training semantics for TabM experts."""

from __future__ import annotations

from numbers import Real
from typing import Mapping

import numpy as np

from experiments.independent_dl.models.common import (
    ModelMetadata,
    import_runtime_module,
)
from experiments.independent_dl.models.tabm import TabMAdapter


def _float32_array(values: np.ndarray, *, label: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if values.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ) or np.issubdtype(values.dtype, np.bool_):
        raise TypeError(f"{label} must have a real numeric dtype")
    if np.issubdtype(values.dtype, np.floating):
        finite = values[np.isfinite(values)]
        if np.any(np.abs(finite) > np.finfo(np.float32).max):
            raise ValueError(f"{label} entries must be representable as float32")
    result = np.array(values, dtype="float32", copy=True)
    result.setflags(write=False)
    return result


def _validate_loss_inputs(
    sample_weight: np.ndarray,
    teacher_probability: np.ndarray | None,
    teacher_lambda: object,
    *,
    loss_name: str,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    weight = _float32_array(sample_weight, label="sample weight")
    if not np.isfinite(sample_weight).all() or not np.isfinite(weight).all():
        raise ValueError("sample weight entries must be finite")
    if np.any(sample_weight <= 0) or np.any(weight <= 0):
        raise ValueError("sample weight entries must be strictly positive")

    if isinstance(teacher_lambda, bool) or not isinstance(teacher_lambda, Real):
        raise TypeError("teacher lambda must be a finite numeric value")
    teacher_lambda_value = float(teacher_lambda)
    if not np.isfinite(teacher_lambda_value):
        raise ValueError("teacher lambda must be finite")
    if not 0.0 <= teacher_lambda_value <= 1.0:
        raise ValueError("teacher lambda must be in [0, 1]")

    teacher: np.ndarray | None = None
    if teacher_probability is None:
        if teacher_lambda_value != 0.0:
            raise ValueError("teacher_lambda must be zero without teacher probabilities")
    else:
        teacher = _float32_array(
            teacher_probability, label="teacher probability"
        )
        if len(teacher) != len(weight):
            raise ValueError(
                "teacher probability and sample weight lengths must match"
            )
        finite = np.isfinite(teacher_probability)
        invalid_nonfinite = ~finite & ~np.isnan(teacher_probability)
        if invalid_nonfinite.any():
            raise ValueError(
                "teacher probability entries must be finite or NaN"
            )
        if np.any((teacher_probability[finite] < 0) | (teacher_probability[finite] > 1)):
            raise ValueError("finite teacher probability entries must be in [0, 1]")
        if loss_name == "brier":
            raise ValueError("Brier loss does not support teacher probabilities")

    return weight, teacher, teacher_lambda_value


class TemporalTabMAdapter(TabMAdapter):
    """TabM adapter with exact temporal weighting and optional BCE distillation."""

    def __init__(
        self,
        *,
        sample_weight: np.ndarray,
        loss_name: str,
        teacher_probability: np.ndarray | None = None,
        teacher_lambda: float = 0.0,
    ) -> None:
        super().__init__(loss_name)
        weight, teacher, teacher_lambda_value = _validate_loss_inputs(
            sample_weight,
            teacher_probability,
            teacher_lambda,
            loss_name=loss_name,
        )
        self.sample_weight_np = weight
        self.teacher_probability_np = teacher
        self.teacher_np = teacher
        self.teacher_lambda = teacher_lambda_value
        self._weight = None
        self._teacher = None
        self._teacher_mask = None

    def build(
        self,
        model_config: Mapping[str, object],
        metadata: ModelMetadata,
        device: str,
    ) -> object:
        model = super().build(model_config, metadata, device)
        self.bind_device(device)
        return model

    def bind_device(self, device: str) -> None:
        torch = import_runtime_module("torch")
        self._weight = torch.tensor(
            self.sample_weight_np, dtype=torch.float32, device=device
        )
        self._teacher = None
        self._teacher_mask = None
        if self.teacher_probability_np is not None:
            mask = np.isfinite(self.teacher_probability_np)
            filled = np.where(mask, self.teacher_probability_np, np.float32(0.5))
            self._teacher_mask = torch.tensor(mask, dtype=torch.bool, device=device)
            self._teacher = torch.tensor(
                filled, dtype=torch.float32, device=device
            )

    def _bound_weight(self) -> object:
        if self._weight is None:
            raise RuntimeError(
                "TemporalTabMAdapter.bind_device must be called before computing loss"
            )
        return self._weight

    @staticmethod
    def _validate_indices(
        torch: object,
        indices: object,
        *,
        label: str,
        length: int,
        device: object,
    ) -> None:
        if not isinstance(indices, torch.Tensor):
            raise TypeError(f"{label} must be a torch tensor")
        if indices.dtype != torch.long:
            raise TypeError(f"{label} must have torch.long dtype")
        if indices.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        if indices.numel() == 0:
            raise ValueError(f"{label} must not be empty")
        if indices.device != device:
            raise ValueError(f"{label} must be on the bound device")
        in_range = ((indices >= 0) & (indices < length)).all()
        if hasattr(torch, "_assert_async"):
            torch._assert_async(in_range, f"{label} values are out of range")
        else:
            torch._assert(in_range, f"{label} values are out of range")

    def _per_row_loss(
        self,
        torch: object,
        member_logits: object,
        y: object,
        row_indices: object,
    ) -> object:
        if member_logits.ndim != 2 or member_logits.shape[1] == 0:
            raise ValueError("TabM member logits must be a nonempty two-dimensional tensor")
        if not isinstance(y, torch.Tensor) or y.ndim != 1:
            raise ValueError("targets must be a one-dimensional torch tensor")
        if len(y) != member_logits.shape[0] or len(row_indices) != len(y):
            raise ValueError("targets, row indices, and logits must have compatible lengths")
        if y.device != member_logits.device:
            raise ValueError("targets and logits must be on the same device")

        hard_target = y.float().unsqueeze(1).expand_as(member_logits)
        if self.loss_name == "brier":
            probability = member_logits.sigmoid().mean(dim=1)
            return (probability - y.float()).square()

        hard_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            member_logits, hard_target, reduction="none"
        ).mean(dim=1)
        if self._teacher is None:
            return hard_loss
        teacher = self._teacher[row_indices]
        mask = self._teacher_mask[row_indices]
        blended_target = (
            (1.0 - self.teacher_lambda) * y.float()
            + self.teacher_lambda * teacher
        )
        blended_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            member_logits,
            blended_target.unsqueeze(1).expand_as(member_logits),
            reduction="none",
        ).mean(dim=1)
        return torch.where(mask, blended_loss, hard_loss)

    def loss_for_window(
        self,
        model: object,
        x_num: object,
        x_cat: object,
        y: object,
        *,
        row_indices: object,
        window_indices: object,
    ) -> object:
        weight = self._bound_weight()
        torch = import_runtime_module("torch")
        self._validate_indices(
            torch,
            row_indices,
            label="row_indices",
            length=len(weight),
            device=weight.device,
        )
        self._validate_indices(
            torch,
            window_indices,
            label="window_indices",
            length=len(weight),
            device=weight.device,
        )
        member_logits = model(x_num, x_cat).squeeze(-1)
        if member_logits.device != weight.device:
            raise ValueError("model logits must be on the bound device")
        per_row = self._per_row_loss(torch, member_logits, y, row_indices)
        numerator = (per_row * weight[row_indices]).sum()
        denominator = weight[window_indices].sum().clamp_min(1e-12)
        return numerator / denominator

    def debug_per_row_loss(
        self,
        member_logits: object,
        y: object,
        *,
        row_indices: object | None = None,
    ) -> object:
        weight = self._bound_weight()
        torch = import_runtime_module("torch")
        if not isinstance(member_logits, torch.Tensor):
            raise TypeError("member_logits must be a torch tensor")
        if row_indices is None:
            row_indices = torch.arange(
                member_logits.shape[0], dtype=torch.long, device=member_logits.device
            )
        self._validate_indices(
            torch,
            row_indices,
            label="row_indices",
            length=len(weight),
            device=weight.device,
        )
        if member_logits.device != weight.device:
            raise ValueError("member logits must be on the bound device")
        return self._per_row_loss(torch, member_logits, y, row_indices)
