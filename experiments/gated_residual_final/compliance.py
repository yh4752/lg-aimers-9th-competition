from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from experiments.direct_expert.compliance import audit_inference


class ComplianceError(ValueError):
    pass


class Predictor(Protocol):
    def predict(self, rows: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class RowIndependenceReport:
    passed: bool
    tolerance: float
    maximum_absolute_difference: float
    checks: Mapping[str, float | int]


class _ValidatedPredictor:
    def __init__(self, source: Predictor):
        self.source = source

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        probability = np.asarray(self.source.predict(rows), dtype="float64")
        if probability.shape != (len(rows),) or not np.isfinite(probability).all():
            raise ComplianceError("probability values differ")
        if np.any((probability < 0) | (probability > 1)):
            raise ComplianceError("probability values differ")
        return probability


def audit_row_independence(
    predictor: Predictor, rows: pd.DataFrame, *, tolerance: float = 1e-6,
) -> RowIndependenceReport:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ComplianceError("audit tolerance differs")
    report = audit_inference(_ValidatedPredictor(predictor), rows)
    checks = {
        "singleton_max_abs": report.singleton_max_abs,
        "reverse_max_abs": report.reverse_max_abs,
        "shuffle_max_abs": report.shuffle_max_abs,
        "rebatch_max_abs": report.rebatch_max_abs,
        "companion_max_abs": report.companion_max_abs,
        "same_feature_audit_row_max_abs": report.same_feature_audit_row_max_abs,
        "rows": report.rows,
    }
    maximum = max(float(value) for key, value in checks.items() if key != "rows")
    return RowIndependenceReport(maximum <= tolerance, float(tolerance), maximum, checks)


def audit_temporal_sources(sources: Mapping[int, tuple[int, ...]]) -> None:
    if not sources:
        raise ComplianceError("temporal sources are absent")
    for validation_year, source_years in sources.items():
        if (
            type(validation_year) is not int
            or type(source_years) is not tuple
            or any(type(year) is not int or year >= validation_year for year in source_years)
            or tuple(sorted(set(source_years))) != source_years
        ):
            raise ComplianceError("temporal source cutoff differs")
