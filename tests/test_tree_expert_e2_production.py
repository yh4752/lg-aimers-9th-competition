from pathlib import Path

import pandas as pd
import pytest

from experiments.tree_expert.e2_production import (
    E2ProductionError,
    build_inference_audit_rows,
)


def test_inference_audit_rows_use_fixed_training_rows_without_target(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "row_id": ["old", "a", "b", "c"],
            "season": [2023, 2024, 2024, 2024],
            "control_success": [0, 1, 0, 1],
            "value": [0, 1, 2, 3],
        }
    ).to_csv(path, index=False)
    rows = build_inference_audit_rows(path, row_count=2)
    assert rows["row_id"].tolist() == ["a", "b"]
    assert rows["season"].tolist() == [2025, 2025]
    assert "control_success" not in rows


def test_inference_audit_rows_fail_if_fixed_pool_is_too_small(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {"row_id": ["a"], "season": [2024], "control_success": [1]}
    ).to_csv(path, index=False)
    with pytest.raises(E2ProductionError, match="source rows"):
        build_inference_audit_rows(path, row_count=2)
