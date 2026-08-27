from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .features import TreeFeatureBatch, transform_tree_features


class T3InferenceError(ValueError):
    pass


Transformer = Callable[[pd.DataFrame, object], TreeFeatureBatch]


class T3Predictor:
    def __init__(
        self,
        *,
        state: object,
        recent_models: Sequence[object],
        multi_models: Sequence[object],
        recent_weight: float,
        transformer: Transformer = transform_tree_features,
    ) -> None:
        recent = tuple(recent_models)
        multi = tuple(multi_models)
        if len(recent) != 3 or len(multi) != 3:
            raise T3InferenceError("exactly three models per temporal head are required")
        if any(not hasattr(model, "predict") for model in (*recent, *multi)):
            raise T3InferenceError("temporal model interface differs")
        if type(recent_weight) is not float or recent_weight not in {0.70, 0.80, 0.90}:
            raise T3InferenceError("recent weight differs")
        if not callable(transformer):
            raise T3InferenceError("feature transformer differs")
        self.state = state
        self.recent_models = recent
        self.multi_models = multi
        self.recent_weight = recent_weight
        self.transformer = transformer

    @staticmethod
    def _head_probability(models: tuple[object, ...], frame: pd.DataFrame, anchor: np.ndarray) -> np.ndarray:
        members: list[np.ndarray] = []
        for model in models:
            residual = np.asarray(model.predict(frame), dtype="float64")
            if residual.shape != anchor.shape or not np.isfinite(residual).all():
                raise T3InferenceError("temporal residual prediction differs")
            members.append(np.clip(anchor + residual, 1e-5, 1 - 1e-5))
        return np.mean(np.stack(members, axis=0), axis=0)

    def predict(self, rows: pd.DataFrame, *, batch_size: int = 4096) -> np.ndarray:
        if type(rows) is not pd.DataFrame or rows.empty:
            raise T3InferenceError("evaluation rows must be a non-empty DataFrame")
        if "control_success" in rows:
            raise T3InferenceError("evaluation rows contain target")
        if "row_id" not in rows or rows["row_id"].isna().any() or not rows["row_id"].is_unique:
            raise T3InferenceError("row_id values differ")
        if type(batch_size) is not int or batch_size <= 0:
            raise T3InferenceError("batch size differs")
        batch = self.transformer(rows.copy(deep=True), self.state)
        if type(batch) is not TreeFeatureBatch or batch.target is not None:
            raise T3InferenceError("evaluation feature batch differs")
        if not np.array_equal(batch.row_id.astype(str), rows["row_id"].astype(str).to_numpy()):
            raise T3InferenceError("transformed row order differs")
        anchor = np.asarray(batch.anchor, dtype="float64")
        if anchor.shape != (len(rows),) or not np.isfinite(anchor).all():
            raise T3InferenceError("anchor values differ")
        recent = self._head_probability(self.recent_models, batch.frame, anchor)
        multi = self._head_probability(self.multi_models, batch.frame, anchor)
        return np.clip(
            self.recent_weight * recent + (1.0 - self.recent_weight) * multi,
            1e-5, 1 - 1e-5,
        )
