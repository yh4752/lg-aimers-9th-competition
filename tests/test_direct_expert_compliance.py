import numpy as np
import pandas as pd
import pytest

from experiments.direct_expert.compliance import (
    DirectExpertComplianceError,
    audit_inference,
    audit_source,
)


class FakeRuntime:
    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.where(rows["game_type"].eq("R"), 0.6, 0.4)


def test_all_row_independence_modes_match() -> None:
    rows = pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(20)],
            "game_type": ["R", "F"] * 10,
            "pitcher_id": np.arange(20),
        }
    )
    report = audit_inference(FakeRuntime(), rows, batch_sizes=(1, 7, 64))
    assert report.singleton_max_abs <= 1e-6
    assert report.reverse_max_abs <= 1e-6
    assert report.shuffle_max_abs <= 1e-6
    assert report.rebatch_max_abs <= 1e-6
    assert report.companion_max_abs <= 1e-6
    assert report.same_feature_audit_row_max_abs <= 1e-6


def test_submission_source_rejects_cross_row_operations() -> None:
    with pytest.raises(DirectExpertComplianceError, match="forbidden evaluation operation"):
        audit_source("def predict(test):\n    return test.groupby('pitcher_id').size()\n")


def test_submission_source_rejects_network_and_processes() -> None:
    with pytest.raises(DirectExpertComplianceError, match="forbidden import"):
        audit_source("import requests\n")
