from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.features import TreeFeatureBatch, TreeFeatureError
from experiments.tree_expert.t3_contracts import T3Job
from experiments.tree_expert.t3_training import T3TrainingError, run_t3_job


class RecordingRegressor:
    def __init__(self, prediction: float = 0.02, best_iteration: int = 7):
        self.prediction = prediction
        self.best_iteration = best_iteration
        self.sample_weight = np.asarray([])

    def fit(self, _x, _y, **kwargs):
        self.sample_weight = np.asarray(kwargs["sample_weight"], dtype="float64")
        return self

    def predict(self, frame):
        return np.full(len(frame), self.prediction, dtype="float64")

    def get_best_iteration(self):
        return self.best_iteration

    def save_model(self, path: str):
        Path(path).write_bytes(b"fake-model")


class RaisingRegressor(RecordingRegressor):
    def fit(self, _x, _y, **_kwargs):
        raise RuntimeError("fit failed")


class FeatureHarness:
    def fit(self, rows, _history, **_kwargs):
        return type("State", (), {"categorical_columns": ("category",)})(), self._batch(rows)

    def transform(self, rows, _state):
        return self._batch(rows)

    @staticmethod
    def _batch(rows):
        target = rows["control_success"].to_numpy(dtype="int8") if "control_success" in rows else None
        return TreeFeatureBatch(
            frame=pd.DataFrame({
                "numeric": np.arange(len(rows), dtype="float32"),
                "category": rows["row_id"].astype(str).to_numpy(),
            }),
            anchor=np.full(len(rows), 0.5, dtype="float64"),
            row_id=rows["row_id"].astype(str).to_numpy(),
            target=target,
        )


def synthetic_train():
    return pd.DataFrame({
        "row_id": ["a", "b", "c", "d"],
        "season": [2021, 2022, 2023, 2023],
        "control_success": [0, 1, 0, 1],
    })


def synthetic_valid():
    return pd.DataFrame({
        "row_id": ["v0", "v1"], "season": [2024, 2024],
        "control_success": [0, 1], "game_type": ["R", "F"],
    })


def synthetic_baseline():
    return pd.DataFrame({
        "row_id": ["v0", "v1"], "target": [0, 1],
        "probability": [0.45, 0.55], "game_type": ["R", "F"],
    })


def multi_job(decay=0.55):
    return T3Job("job", "multi", decay, 2023, 2024, 3407)


def recent_job():
    return T3Job("job", "recent", None, 2023, 2024, 3407)


def run_fixture(tmp_path, *, job=None, valid=None, baseline=None, model=None):
    harness = FeatureHarness()
    return run_t3_job(
        job=job or multi_job(), train=synthetic_train(),
        valid=valid if valid is not None else synthetic_valid(),
        baseline=baseline if baseline is not None else synthetic_baseline(),
        output_dir=tmp_path, model_factory=lambda _: model or RecordingRegressor(),
        feature_builder=harness.fit, feature_transformer=harness.transform,
    )


def test_t3_job_passes_temporal_weights_and_returns_row_aligned_probabilities(tmp_path):
    model = RecordingRegressor()
    result = run_fixture(tmp_path, model=model)
    assert result.status == "completed"
    np.testing.assert_allclose(model.sample_weight, [0.55**2, 0.55, 1.0, 1.0])
    assert result.best_iteration == 7
    assert list(result.predictions.columns) == ["row_id", "target", "probability", "game_type"]
    np.testing.assert_allclose(result.predictions["probability"], [0.52, 0.52])
    assert result.model_path.is_file()


def test_failed_model_writes_no_checkpoint(tmp_path):
    result = run_fixture(tmp_path, job=recent_job(), model=RaisingRegressor())
    assert result.status == "failed"
    assert result.model_path is None
    assert not (tmp_path / "checkpoint.cbm").exists()
    assert "fit failed" in (tmp_path / "worker.log").read_text()


def test_duplicate_validation_row_id_is_rejected(tmp_path):
    valid = synthetic_valid()
    valid.loc[1, "row_id"] = valid.loc[0, "row_id"]
    with pytest.raises(TreeFeatureError, match="row_id"):
        run_fixture(tmp_path, valid=valid)


def test_baseline_row_identity_must_match(tmp_path):
    baseline = synthetic_baseline().iloc[::-1].reset_index(drop=True)
    with pytest.raises(T3TrainingError, match="baseline row identity"):
        run_fixture(tmp_path, baseline=baseline)
