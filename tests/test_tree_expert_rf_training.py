from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.features import TreeFeatureBatch
from experiments.tree_expert.rf_contracts import RFJob, load_rf_contract
from experiments.tree_expert.rf_training import RFTrainingError, run_rf_job


class RecordingRegressor:
    def __init__(self, prediction: float = 0.05):
        self.prediction = prediction
        self.fit_row_ids: tuple[str, ...] = ()
        self.fit_target = np.asarray([])

    def fit(self, x, y, **_kwargs):
        self.fit_row_ids = tuple(x["category"].astype(str))
        self.fit_target = np.asarray(y, dtype="float64")
        return self

    def predict(self, frame):
        return np.full(len(frame), self.prediction, dtype="float64")

    def get_best_iteration(self):
        return 7

    def save_model(self, path: str):
        Path(path).write_bytes(b"fake-model")


class FeatureHarness:
    @staticmethod
    def _batch(rows: pd.DataFrame) -> TreeFeatureBatch:
        target = rows["control_success"].to_numpy(dtype="float64") if "control_success" in rows else None
        return TreeFeatureBatch(
            frame=pd.DataFrame(
                {
                    "numeric": np.arange(len(rows), dtype="float32"),
                    "category": rows["row_id"].astype(str).to_numpy(),
                }
            ),
            anchor=np.full(len(rows), 0.5, dtype="float64"),
            row_id=rows["row_id"].astype(str).to_numpy(),
            target=target,
        )

    def fit(self, rows, _history, **_kwargs):
        state = type("State", (), {"categorical_columns": ("category",)})()
        return state, self._batch(rows)

    def transform(self, rows, _state):
        return self._batch(rows)


def _contract():
    contract = load_rf_contract()
    return replace(contract, gates=replace(contract.gates, minimum_segment_rows=1))


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["r1", "f1", "r2", "f2", "future"],
            "season": [2021, 2021, 2023, 2023, 2024],
            "game_type": ["R", "F", "R", "F", "F"],
            "control_success": [0, 1, 1, 0, 1],
        }
    )


def _valid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["v0", "v1"],
            "season": [2024, 2024],
            "game_type": ["R", "F"],
            "control_success": [0, 1],
        }
    )


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["v0", "v1"],
            "target": [0, 1],
            "probability": [0.45, 0.55],
            "game_type": ["R", "F"],
        }
    )


def _job(segment: str = "F", head: str = "f_small") -> RFJob:
    return RFJob("job", head, segment, 2023, 2024, 3407)


def _run(tmp_path: Path, *, job: RFJob | None = None, train: pd.DataFrame | None = None):
    model = RecordingRegressor()
    harness = FeatureHarness()
    result = run_rf_job(
        job=job or _job(),
        train=_train() if train is None else train,
        valid=_valid(),
        baseline=_baseline(),
        output_dir=tmp_path,
        contract=_contract(),
        model_factory=lambda _: model,
        feature_builder=harness.fit,
        feature_transformer=harness.transform,
    )
    return result, model


def test_f_job_trains_only_prior_f_rows_and_uses_anchor_residual(tmp_path: Path) -> None:
    result, model = _run(tmp_path)

    assert model.fit_row_ids == ("f1", "f2")
    np.testing.assert_allclose(model.fit_target, [0.5, -0.5])
    assert result.status == "completed"
    np.testing.assert_allclose(result.predictions["probability"], [0.55, 0.55])


def test_r_job_uses_r_expert_parameters(tmp_path: Path) -> None:
    parameters = {}
    harness = FeatureHarness()

    def factory(value):
        parameters.update(value)
        return RecordingRegressor()

    run_rf_job(
        job=_job("R", "r_expert"), train=_train(), valid=_valid(), baseline=_baseline(),
        output_dir=tmp_path, contract=_contract(), model_factory=factory,
        feature_builder=harness.fit, feature_transformer=harness.transform,
    )

    assert parameters["depth"] == 8
    assert parameters["l2_leaf_reg"] == 5.0
    assert parameters["devices"] == "0"


def test_job_writes_model_prediction_and_metrics_atomically(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)

    assert result.model_path == tmp_path / "checkpoint.cbm"
    assert result.predictions_path == tmp_path / "predictions.csv"
    assert result.model_path.is_file()
    assert result.predictions_path.is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_job_rejects_missing_segment_in_training_fold(tmp_path: Path) -> None:
    train = _train().loc[lambda frame: frame["game_type"].eq("R")].copy()

    with pytest.raises(RFTrainingError, match="segment training rows are empty"):
        _run(tmp_path, train=train)


def test_job_rejects_baseline_row_reordering(tmp_path: Path) -> None:
    harness = FeatureHarness()
    with pytest.raises(RFTrainingError, match="baseline row identity"):
        run_rf_job(
            job=_job(), train=_train(), valid=_valid(),
            baseline=_baseline().iloc[::-1].reset_index(drop=True),
            output_dir=tmp_path, contract=_contract(), model_factory=lambda _: RecordingRegressor(),
            feature_builder=harness.fit, feature_transformer=harness.transform,
        )
