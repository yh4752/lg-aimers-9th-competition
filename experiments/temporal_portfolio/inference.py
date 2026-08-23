"""Frozen, row-local inference for an approved temporal recipe."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .ensembles import Recipe
from .features import transform_portfolio_features
from .metrics import blend_logit


class InferenceRuleError(RuntimeError):
    """Raised when evaluation inference would use mutable or cross-row state."""


@dataclass(frozen=True)
class FrozenManifest:
    stream_order: tuple[str, ...]
    recipe: Recipe
    anchor_rate: float | None = None
    calibration_coefficient: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if (
            type(self.stream_order) is not tuple
            or not self.stream_order
            or len(set(self.stream_order)) != len(self.stream_order)
            or any(type(name) is not str or not name for name in self.stream_order)
        ):
            raise InferenceRuleError("frozen stream order is invalid")
        if type(self.recipe) is not Recipe or not set(self.recipe.components).issubset(
            self.stream_order
        ):
            raise InferenceRuleError("frozen recipe streams differ")
        if self.recipe.anchor is not None:
            if self.anchor_rate is None or not math.isfinite(self.anchor_rate) or not 0 < self.anchor_rate < 1:
                raise InferenceRuleError("frozen anchor rate is invalid")
        elif self.anchor_rate is not None:
            raise InferenceRuleError("unused frozen anchor rate")
        if self.recipe.calibration is not None:
            coefficient = self.calibration_coefficient
            if coefficient is None or len(coefficient) != 2 or not all(
                math.isfinite(value) for value in coefficient
            ):
                raise InferenceRuleError("frozen calibration coefficient is invalid")
        elif self.calibration_coefficient is not None:
            raise InferenceRuleError("unused frozen calibration coefficient")


class FrozenPredictor:
    def __init__(
        self,
        manifest: FrozenManifest,
        models: Mapping[str, object],
        states: Mapping[str, object],
        *,
        transformer: Callable[[pd.DataFrame, object], object] = transform_portfolio_features,
    ) -> None:
        if type(manifest) is not FrozenManifest:
            raise InferenceRuleError("frozen manifest type is invalid")
        model_snapshot = dict(models)
        state_snapshot = dict(states)
        if set(model_snapshot) != set(manifest.stream_order) or set(state_snapshot) != set(
            manifest.stream_order
        ):
            raise InferenceRuleError("frozen model or state streams differ")
        if any(getattr(state, "inference_mode", None) is not True for state in state_snapshot.values()):
            raise InferenceRuleError("frozen inference state is required")
        if not callable(transformer):
            raise InferenceRuleError("frozen transformer is invalid")
        self.manifest = manifest
        self.models = MappingProxyType(model_snapshot)
        self.states = MappingProxyType(state_snapshot)
        self.transformer = transformer

    def predict(self, rows: pd.DataFrame, *, batch_size: int = 512) -> np.ndarray:
        work = _evaluation_rows(rows)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise InferenceRuleError("batch size is invalid")
        streams: dict[str, np.ndarray] = {}
        for name in self.manifest.stream_order:
            pieces: list[np.ndarray] = []
            for start in range(0, len(work), batch_size):
                chunk = work.iloc[start : start + batch_size]
                batch = self.transformer(chunk, self.states[name])
                model = self.models[name]
                method = getattr(model, "predict", None)
                if not callable(method):
                    raise InferenceRuleError("frozen model predict method is missing")
                pieces.append(_probability(method(batch), len(chunk)))
            streams[name] = np.concatenate(pieces)
        return _apply_recipe(self.manifest, streams)


def _apply_recipe(manifest: FrozenManifest, streams: Mapping[str, np.ndarray]) -> np.ndarray:
    recipe = manifest.recipe
    values = [streams[name] for name in recipe.components]
    if recipe.mode == "probability":
        probability = sum(
            float(weight) * value
            for weight, value in zip(recipe.weights, values, strict=True)
        )
    else:
        probability = np.array(values[0], copy=True)
        cumulative = recipe.weights[0]
        for value, weight in zip(values[1:], recipe.weights[1:], strict=True):
            combined = cumulative + weight
            probability = blend_logit(probability, value, cumulative / combined)
            cumulative = combined
    if recipe.anchor is not None:
        beta = recipe.anchor
        probability = blend_logit(
            probability,
            np.full(len(probability), manifest.anchor_rate),
            Decimal("1") - beta,
        )
    if recipe.calibration is not None:
        slope, intercept = manifest.calibration_coefficient
        epsilon = np.finfo("float64").eps
        clipped = np.clip(probability, epsilon, 1 - epsilon)
        logit = np.log(clipped) - np.log1p(-clipped)
        z = slope * logit + intercept
        probability = 1.0 / (1.0 + np.exp(-z))
    return _probability(probability, len(next(iter(streams.values()))))


def _evaluation_rows(rows: object) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame or rows.empty or not rows.columns.is_unique:
        raise InferenceRuleError("evaluation rows are invalid")
    if "control_success" in rows:
        raise InferenceRuleError("evaluation rows contain target")
    if "row_id" not in rows or rows["row_id"].isna().any():
        raise InferenceRuleError("evaluation row identity differs")
    ids = rows["row_id"].astype(str)
    if ids.duplicated().any() or (ids.str.len() == 0).any():
        raise InferenceRuleError("evaluation row identity differs")
    return rows.copy(deep=True)


def _probability(value: object, rows: int) -> np.ndarray:
    output = np.asarray(value, dtype="float64")
    if output.shape != (rows,) or not np.isfinite(output).all() or np.any(
        (output < 0) | (output > 1)
    ):
        raise InferenceRuleError("frozen probabilities are invalid")
    return output
