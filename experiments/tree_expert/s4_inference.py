from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class S4InferenceError(ValueError):
    pass


class Transformer(Protocol):
    def transform(self, rows: pd.DataFrame) -> object: ...


class Classifier(Protocol):
    def predict_proba(self, matrix: object) -> object: ...


class Regressor(Protocol):
    def predict(self, matrix: object) -> object: ...


class Calibrator(Protocol):
    def predict(self, rows: pd.DataFrame, probability: np.ndarray) -> object: ...


@dataclass(frozen=True)
class FrozenS4Predictor:
    required_columns: tuple[str, ...]
    transformer: Transformer
    recent_model: Classifier
    multi_model: Classifier
    recent_weight: float
    residual_alpha: float
    residual_model: Regressor | None
    residual_r_model: Regressor | None
    residual_f_model: Regressor | None
    calibrator: Calibrator | None

    def _rows(self, rows: object) -> pd.DataFrame:
        if type(rows) is not pd.DataFrame or rows.empty or not set(self.required_columns).issubset(rows.columns):
            raise S4InferenceError("inference row schema differs")
        if {"target", "control_success"} & set(rows.columns):
            raise S4InferenceError("target column is forbidden at inference")
        if rows["row_id"].isna().any() or not rows["game_type"].astype(str).isin(("R", "F")).all():
            raise S4InferenceError("inference row values differ")
        return rows.loc[:, list(self.required_columns)].copy(deep=True)

    @staticmethod
    def _probability(model: Classifier, matrix: object, rows: int) -> np.ndarray:
        raw = np.asarray(model.predict_proba(matrix), dtype="float64")
        if raw.ndim == 2 and raw.shape == (rows, 2):
            value = raw[:, 1]
        elif raw.ndim == 1 and raw.shape == (rows,):
            value = raw
        else:
            raise S4InferenceError("classifier probability differs")
        if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise S4InferenceError("classifier probability values differ")
        return value

    @staticmethod
    def _residual(model: Regressor, matrix: object, rows: int) -> np.ndarray:
        value = np.asarray(model.predict(matrix), dtype="float64")
        if value.shape != (rows,) or not np.isfinite(value).all():
            raise S4InferenceError("residual prediction differs")
        return value

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        frame = self._rows(rows)
        matrix = self.transformer.transform(frame)
        recent = self._probability(self.recent_model, matrix, len(frame))
        multi = self._probability(self.multi_model, matrix, len(frame))
        if not 0 <= self.recent_weight <= 1 or self.residual_alpha <= 0:
            raise S4InferenceError("frozen mixture differs")
        anchor = self.recent_weight * recent + (1.0 - self.recent_weight) * multi
        if self.residual_model is not None:
            if self.residual_r_model is not None or self.residual_f_model is not None:
                raise S4InferenceError("residual model registry differs")
            correction = self._residual(self.residual_model, matrix, len(frame))
        else:
            if self.residual_r_model is None or self.residual_f_model is None:
                raise S4InferenceError("residual model registry differs")
            r_value = self._residual(self.residual_r_model, matrix, len(frame))
            f_value = self._residual(self.residual_f_model, matrix, len(frame))
            correction = np.where(frame["game_type"].astype(str).to_numpy() == "F", f_value, r_value)
        probability = np.clip(anchor + self.residual_alpha * correction, 1e-5, 1 - 1e-5)
        if self.calibrator is not None:
            probability = np.asarray(self.calibrator.predict(frame, probability), dtype="float64")
        if probability.shape != (len(frame),) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise S4InferenceError("final probability differs")
        return probability


@dataclass(frozen=True)
class S4IndependenceAudit:
    rows: int
    batch_sizes: tuple[int, ...]
    maximum_absolute_difference: float


def audit_s4_independence(
    predictor: FrozenS4Predictor,
    rows: pd.DataFrame,
    *,
    batch_sizes: tuple[int, ...] = (1, 17, 256),
) -> S4IndependenceAudit:
    if type(batch_sizes) is not tuple or not batch_sizes or any(type(value) is not int or value <= 0 for value in batch_sizes):
        raise S4InferenceError("audit batch registry differs")
    frame = predictor._rows(rows)
    if not frame["row_id"].is_unique:
        raise S4InferenceError("audit row IDs must be unique")
    reference = predictor.predict(frame)
    maximum = 0.0
    variants = [frame.iloc[::-1], frame.sample(frac=1, random_state=3407)]
    reference_by_id = dict(zip(frame["row_id"].astype(str), reference, strict=True))
    for variant in variants:
        actual = predictor.predict(variant)
        expected = np.asarray([reference_by_id[value] for value in variant["row_id"].astype(str)])
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    for size in batch_sizes:
        chunks = []
        for start in range(0, len(frame), size):
            chunks.append(predictor.predict(frame.iloc[start:start + size]))
        actual = np.concatenate(chunks)
        maximum = max(maximum, float(np.max(np.abs(actual - reference))))
    return S4IndependenceAudit(len(frame), batch_sizes, maximum)
