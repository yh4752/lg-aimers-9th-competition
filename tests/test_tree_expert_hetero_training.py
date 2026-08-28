from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.hetero_contracts import HeteroJob
from experiments.tree_expert.hetero_features import HeteroFeatureBatch
from experiments.tree_expert.hetero_training import HeteroTrainingError, run_hetero_job


class RecordingClassifier:
    def __init__(self, probability=0.7):
        self.probability = probability
        self.target = None
        self.best_iteration = 5

    def fit(self, _x, y, **_kwargs):
        self.target = np.asarray(y)
        return self

    def predict_proba(self, x):
        p = np.full(len(x), self.probability)
        return np.column_stack([1.0 - p, p])

    def save_model(self, path):
        Path(path).write_bytes(b"model")


class Harness:
    def fit(self, rows, *, valid_year):
        del valid_year
        target = rows["control_success"].to_numpy(dtype="int8")
        return object(), HeteroFeatureBatch(
            np.arange(len(rows) * 2, dtype="float32").reshape(len(rows), 2),
            rows["row_id"].astype(str).to_numpy(), target,
        )

    def transform(self, rows, _state):
        return HeteroFeatureBatch(
            np.ones((len(rows), 2), dtype="float32"),
            rows["row_id"].astype(str).to_numpy(), None,
        )


def _train():
    return pd.DataFrame({
        "row_id": ["a", "b", "c", "d"], "season": [2021, 2022, 2023, 2024],
        "control_success": [0, 1, 0, 1],
    })


def _baseline():
    return pd.DataFrame({
        "row_id": ["d"], "target": [1], "probability": [0.55],
        "game_type": ["R"],
    })


def test_job_learns_binary_target_directly_and_preserves_baseline(tmp_path):
    model = RecordingClassifier()
    harness = Harness()
    result = run_hetero_job(
        job=HeteroJob("job", "xgboost", 2023, 2024, 3407),
        train=_train(), baseline=_baseline(), output_dir=tmp_path,
        model_factory=lambda *_args, **_kwargs: model,
        feature_builder=harness.fit, feature_transformer=harness.transform,
    )
    assert result.status == "completed"
    np.testing.assert_array_equal(model.target, [0, 1, 0])
    assert list(result.predictions.columns) == [
        "row_id", "target", "baseline_probability", "model_probability", "game_type",
    ]
    np.testing.assert_allclose(result.predictions["model_probability"], [0.7])
    assert result.model_path.is_file()


def test_baseline_row_order_is_a_hard_boundary(tmp_path):
    bad = _baseline().copy()
    bad.loc[0, "row_id"] = "wrong"
    harness = Harness()
    with pytest.raises(HeteroTrainingError, match="row identity"):
        run_hetero_job(
            job=HeteroJob("job", "lightgbm", 2023, 2024, 3407),
            train=_train(), baseline=bad, output_dir=tmp_path,
            model_factory=lambda *_args, **_kwargs: RecordingClassifier(),
            feature_builder=harness.fit, feature_transformer=harness.transform,
        )
