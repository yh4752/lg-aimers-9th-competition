from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.catboost_deployment.contracts import build_jobs, load_contract
from experiments.catboost_deployment.metrics import CATBOOST_PREDICTION_COLUMNS
from experiments.catboost_deployment.training import (
    run_alignment_job,
    run_full_fit_job,
)


class FakeCatBoost:
    instances: list["FakeCatBoost"] = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.fit_calls: list[tuple[pd.DataFrame, np.ndarray, dict[str, object]]] = []
        self.ntree_ends: list[int] = []
        self.__class__.instances.append(self)

    def fit(self, x, y, **arguments):
        self.fit_calls.append((x.copy(), np.asarray(y).copy(), dict(arguments)))
        Path(arguments["snapshot_file"]).write_bytes(b"snapshot")
        arguments["log_cout"].write("50:\tlearn: 0.5\ttotal: 1s\n")
        return self

    def predict(self, x, *, ntree_end=None):
        if ntree_end is None:
            return np.full(len(x), 0.55)
        self.ntree_ends.append(int(ntree_end))
        return np.linspace(0.2, 0.8, len(x))

    def save_model(self, path):
        Path(path).write_bytes(b"model")


def _data_dir(tmp_path: Path, preprocessing_frame: pd.DataFrame) -> Path:
    frame = preprocessing_frame.copy()
    frame["season"] = [2021, 2022, 2022, 2023, 2023]
    frame["row_id"] = [f"ROW_{index}" for index in range(len(frame))]
    frame["game_month"] = [3, 4, 5, 6, 7]
    root = tmp_path / "data"
    root.mkdir()
    frame.to_csv(root / "train.csv", index=False)
    return root


def _common(tmp_path: Path, preprocessing_frame: pd.DataFrame) -> dict[str, object]:
    return {
        "contract": load_contract(),
        "data_dir": _data_dir(tmp_path, preprocessing_frame),
        "contract_sha256": "a" * 64,
        "input_manifest_sha256": "b" * 64,
        "code_sha256": "c" * 64,
        "absolute_deadline": time.time() + 60,
        "model_factory": FakeCatBoost,
    }


def test_alignment_worker_fits_once_and_predicts_all_prefixes(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    kwargs = _common(tmp_path, preprocessing_frame)
    output = tmp_path / "alignment"

    result = run_alignment_job(
        job=build_jobs(load_contract())[0], output_dir=output, **kwargs
    )

    assert result.status == "completed"
    model = FakeCatBoost.instances[-1]
    assert len(model.fit_calls) == 1
    assert model.parameters["iterations"] == 400
    assert model.ntree_ends == [4, 32, 64, 128, 192, 296, 400]
    _, _, fit_arguments = model.fit_calls[0]
    assert fit_arguments["use_best_model"] is False
    assert "early_stopping_rounds" not in fit_arguments
    assert fit_arguments["snapshot_interval"] == 300
    assert "control_success" not in model.fit_calls[0][0]
    predictions = pd.read_csv(result.predictions_path)
    assert tuple(predictions.columns) == CATBOOST_PREDICTION_COLUMNS
    metrics = json.loads((output / "metrics.json").read_text())
    assert set(metrics["prefix_brier"]) == {str(value) for value in (4, 32, 64, 128, 192, 296, 400)}


def test_full_worker_uses_selected_tree_count_and_freezes_state(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    kwargs = _common(tmp_path, preprocessing_frame)
    output = tmp_path / "full"

    result = run_full_fit_job(
        job=build_jobs(load_contract())[2],
        output_dir=output,
        selected_tree_count=128,
        alignment_decision_sha256="d" * 64,
        **kwargs,
    )

    assert result.status == "completed"
    model = FakeCatBoost.instances[-1]
    assert model.parameters["iterations"] == 128
    assert len(model.fit_calls) == 1
    assert model.ntree_ends == []
    assert result.model_path == output / "model.cbm"
    assert result.preprocessing_path == output / "preprocessing_state.json"
    assert result.preprocessing_path.is_file()
    assert json.loads((output / "job.json").read_text())["alignment_decision_sha256"] == "d" * 64


def test_expired_deadline_returns_inconclusive_without_model(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    kwargs = _common(tmp_path, preprocessing_frame)
    kwargs["absolute_deadline"] = 0

    result = run_alignment_job(
        job=build_jobs(load_contract())[0],
        output_dir=tmp_path / "expired",
        **kwargs,
    )

    assert result.status == "budget_inconclusive"
    assert FakeCatBoost.instances == []


def test_existing_job_identity_drift_is_rejected_before_fit(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    kwargs = _common(tmp_path, preprocessing_frame)
    output = tmp_path / "alignment"
    output.mkdir()
    (output / "job.json").write_text('{"job_id":"changed"}')

    result = run_alignment_job(
        job=build_jobs(load_contract())[0], output_dir=output, **kwargs
    )

    assert result.status == "failed"
    assert "identity differs" in str(result.failure)
    assert FakeCatBoost.instances == []
