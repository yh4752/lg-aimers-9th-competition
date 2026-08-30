from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.gated_residual_final.compliance import (
    ComplianceError,
    audit_row_independence,
    audit_temporal_sources,
)


class FixedPredictor:
    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["feature"].to_numpy(dtype="float64") / 10.0


def _rows() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": ["a", "b", "c"], "game_type": ["R", "F", "R"], "feature": [2, 4, 7],
    })


def test_batch_context_cannot_change_one_row_prediction() -> None:
    report = audit_row_independence(FixedPredictor(), _rows(), tolerance=1e-12)

    assert report.passed
    assert report.maximum_absolute_difference <= 1e-12


def test_probability_outside_unit_interval_fails_audit() -> None:
    class BadPredictor:
        def predict(self, rows: pd.DataFrame) -> np.ndarray:
            return np.full(len(rows), 1.1)

    with pytest.raises(ComplianceError, match="probability values differ"):
        audit_row_independence(BadPredictor(), _rows())


def test_temporal_source_year_must_precede_validation() -> None:
    audit_temporal_sources({2023: (2022,), 2024: (2022, 2023)})

    with pytest.raises(ComplianceError, match="temporal source cutoff differs"):
        audit_temporal_sources({2024: (2022, 2024)})
