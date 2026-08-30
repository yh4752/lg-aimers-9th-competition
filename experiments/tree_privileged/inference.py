"""Row-independent inference for accepted privileged students plus frozen E2."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.temporal_portfolio.lupi_teacher import TEACHER_FEATURES

from .features import CandidateFeatureBatch, CandidateFeatureState, transform_candidate_features
from .matching import MATCH_COLUMNS


class PrivilegedInferenceError(ValueError):
    pass


@dataclass(frozen=True)
class StudentRuntime:
    candidate_id: str
    state: CandidateFeatureState
    models: tuple[object, ...]
    alpha_r: float
    alpha_f: float


@dataclass(frozen=True)
class IndependenceAudit:
    status: str
    checks: Mapping[str, float]
    maximum_absolute_difference: float


class PrivilegedPredictor:
    def __init__(
        self,
        *,
        e2_predictor: object,
        students: Sequence[StudentRuntime],
        transformer: Callable[[pd.DataFrame, CandidateFeatureState], CandidateFeatureBatch] = transform_candidate_features,
    ) -> None:
        if not callable(getattr(e2_predictor, "predict", None)):
            raise PrivilegedInferenceError("E2 predictor differs")
        members = tuple(students)
        if not 1 <= len(members) <= 2:
            raise PrivilegedInferenceError("one or two accepted students are required")
        for member in members:
            if type(member) is not StudentRuntime or len(member.models) != 3 or any(not callable(getattr(model, "predict", None)) for model in member.models):
                raise PrivilegedInferenceError("student runtime differs")
            if not 0 <= member.alpha_r <= 1 or not 0 <= member.alpha_f <= 1:
                raise PrivilegedInferenceError("student R/F alpha differs")
        self.e2_predictor = e2_predictor
        self.students = members
        self.transformer = transformer

    def predict(self, rows: pd.DataFrame, *, batch_size: int = 4096) -> np.ndarray:
        if type(rows) is not pd.DataFrame or rows.empty or "row_id" not in rows:
            raise PrivilegedInferenceError("evaluation rows differ")
        if rows["row_id"].isna().any() or not rows["row_id"].is_unique:
            raise PrivilegedInferenceError("evaluation row IDs differ")
        forbidden = {"control_success", "teacher_probability", *TEACHER_FEATURES, *MATCH_COLUMNS[1:]}
        present = sorted(forbidden & set(rows.columns))
        if present:
            raise PrivilegedInferenceError(f"evaluation contains forbidden columns: {present}")
        if "game_type" not in rows or not rows["game_type"].isin(["R", "F"]).all():
            raise PrivilegedInferenceError("evaluation game_type differs")
        if type(batch_size) is not int or batch_size <= 0:
            raise PrivilegedInferenceError("batch size differs")
        source = rows.copy(deep=True)
        e2 = np.asarray(self.e2_predictor.predict(source.copy(deep=True)), dtype="float64")
        if e2.shape != (len(rows),) or not np.isfinite(e2).all():
            raise PrivilegedInferenceError("E2 probabilities differ")
        predictions = []
        for member in self.students:
            batch = self.transformer(source.copy(deep=True), member.state)
            if type(batch) is not CandidateFeatureBatch or batch.target is not None or batch.soft_target is not None:
                raise PrivilegedInferenceError("student inference batch differs")
            residuals = []
            for model in member.models:
                residual = np.asarray(model.predict(batch.frame), dtype="float64")
                if residual.shape != e2.shape or not np.isfinite(residual).all():
                    raise PrivilegedInferenceError("student residual differs")
                residuals.append(residual)
            student = np.clip(batch.anchor + np.mean(np.stack(residuals), axis=0), 1e-5, 1 - 1e-5)
            alpha = np.where(source["game_type"].to_numpy() == "F", member.alpha_f, member.alpha_r)
            predictions.append(np.clip(e2 + alpha * (student - e2), 1e-5, 1 - 1e-5))
        result = np.mean(np.stack(predictions), axis=0)
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise PrivilegedInferenceError("final probabilities differ")
        return result


def audit_independence(predictor: PrivilegedPredictor, rows: pd.DataFrame) -> IndependenceAudit:
    baseline = predictor.predict(rows)
    ids = rows["row_id"].astype(str).tolist()
    def aligned(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
        mapping = dict(zip(frame["row_id"].astype(str), values, strict=True))
        return np.asarray([mapping[row_id] for row_id in ids])
    checks: dict[str, float] = {}
    reversed_rows = rows.iloc[::-1].reset_index(drop=True)
    checks["reverse"] = float(np.max(np.abs(aligned(reversed_rows, predictor.predict(reversed_rows)) - baseline)))
    shuffled = rows.sample(frac=1, random_state=42).reset_index(drop=True)
    checks["shuffle"] = float(np.max(np.abs(aligned(shuffled, predictor.predict(shuffled)) - baseline)))
    singleton = np.asarray([predictor.predict(rows.iloc[[index]].copy(), batch_size=1)[0] for index in range(len(rows))])
    checks["singleton"] = float(np.max(np.abs(singleton - baseline)))
    for size in (1, 2, 7, 32):
        chunks = [predictor.predict(rows.iloc[start:start + size].copy(), batch_size=size) for start in range(0, len(rows), size)]
        checks[f"batch_{size}"] = float(np.max(np.abs(np.concatenate(chunks) - baseline)))
    maximum = max(checks.values())
    return IndependenceAudit("passed" if maximum <= 1e-6 else "failed", MappingProxyType(checks), maximum)

