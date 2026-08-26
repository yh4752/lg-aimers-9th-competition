from __future__ import annotations

import pandas as pd
import pytest

from experiments.tree_expert.failure_labels import (
    FailureLabelError,
    audit_failure_labels,
)


def _gate() -> dict[str, float | int]:
    return {
        "minimum_coverage": 0.75,
        "minimum_binary_delta_fraction": 1.0,
        "minimum_success_agreement": 1.0,
        "maximum_middle_reverse_overlap": 0.0,
        "minimum_class_rows": 1,
        "delta_tolerance": 1e-9,
    }


def _failure_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023] * 5,
            "pitcher_id": [7] * 5,
            "asof_pitcher_n": [0, 1, 2, 3, 4],
            "asof_pitcher_success_rate": [0.0, 1.0, 0.5, 1 / 3, 0.25],
            "asof_pitcher_middle_rate": [0.0, 0.0, 0.5, 1 / 3, 0.25],
            "asof_pitcher_reverse_rate": [0.0, 0.0, 0.0, 1 / 3, 0.25],
            "control_success": [1, 0, 0, 0, 1],
        }
    )


def test_failure_recovery_uses_next_train_cumulative_state_only() -> None:
    audit = audit_failure_labels(_failure_rows(), gate=_gate(), valid_year=2024)

    assert audit.status == "passed"
    assert audit.labels.tolist() == ["success", "middle", "reverse", "other_failure"]
    assert audit.source_positions.tolist() == [0, 1, 2, 3]


def test_failure_recovery_rejects_validation_rows() -> None:
    with pytest.raises(FailureLabelError, match="rows reach validation season"):
        audit_failure_labels(
            _failure_rows().assign(season=2024),
            gate=_gate(),
            valid_year=2024,
        )


def test_failed_label_gate_skips_only_c3() -> None:
    sparse = _failure_rows().iloc[[0, 4]].copy(deep=True)

    result = audit_failure_labels(sparse, gate=_gate(), valid_year=2024)

    assert result.status == "skipped_unreliable_labels"
    assert result.labels.size == 0
