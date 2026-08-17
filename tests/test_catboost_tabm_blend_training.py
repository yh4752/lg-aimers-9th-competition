from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.catboost_tabm_blend.contracts import build_jobs, load_contract
from experiments.catboost_tabm_blend.training import run_fold


class FakeCatBoost:
    instances: list["FakeCatBoost"] = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.fit_arguments = None
        self.__class__.instances.append(self)

    def fit(self, x, y, **arguments):
        self.fit_arguments = (x.copy(), np.asarray(y).copy(), arguments)
        snapshot = Path(arguments["snapshot_file"])
        snapshot.write_bytes(b"snapshot")
        arguments["log_cout"].write("50:\tlearn: 0.5\ttest: 0.5\ttotal: 1s\n")
        return self

    def predict(self, x):
        return np.linspace(0.2, 0.8, len(x))

    def save_model(self, path):
        Path(path).write_bytes(b"model")


def _data_dir(tmp_path: Path, preprocessing_frame: pd.DataFrame) -> Path:
    frame = pd.concat(
        [
            preprocessing_frame.iloc[:3].assign(season=[2021, 2022, 2022]),
            preprocessing_frame.iloc[3:].assign(season=2023),
        ],
        ignore_index=True,
    )
    frame["row_id"] = [f"ROW_{index}" for index in range(len(frame))]
    frame["game_month"] = [3, 4, 5, 6, 7]
    root = tmp_path / "data"
    root.mkdir()
    frame.to_csv(root / "train.csv", index=False)
    return root


def test_fold_uses_sealed_features_snapshot_and_artifacts(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    contract = load_contract()
    job = build_jobs(contract)[0]
    output = tmp_path / "job"

    result = run_fold(
        job=job,
        contract=contract,
        data_dir=_data_dir(tmp_path, preprocessing_frame),
        output_dir=output,
        contract_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        code_sha256="c" * 64,
        absolute_deadline=time.time() + 60,
        model_factory=FakeCatBoost,
    )

    assert result.status == "completed"
    assert result.train_rows == 3
    assert result.valid_rows == 2
    model = FakeCatBoost.instances[-1]
    assert model.parameters["allow_writing_files"] is True
    assert model.parameters["train_dir"] == str(output / "catboost_info")
    x_train, y_train, arguments = model.fit_arguments
    assert "hand_matchup" in x_train
    assert "control_success" not in x_train
    assert len(y_train) == 3
    assert arguments["save_snapshot"] is True
    assert arguments["snapshot_interval"] == 300
    assert arguments["snapshot_file"] == str(output / "experiment.cbsnapshot")
    assert arguments["early_stopping_rounds"] == 50
    assert arguments["use_best_model"] is True
    assert set(path.name for path in output.iterdir()) >= {
        "model.cbm",
        "predictions.csv",
        "metrics.json",
        "job.json",
        "worker_result.json",
        "worker.log",
        "experiment.cbsnapshot",
    }
    predictions = pd.read_csv(output / "predictions.csv")
    assert tuple(predictions.columns) == (
        "row_id",
        "target",
        "probability",
        "game_type",
        "game_month",
        "pitcher_id_known",
        "batter_id_known",
    )
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["brier"] == result.brier
    assert metrics["raw_prediction_min"] == 0.2
    assert metrics["raw_prediction_max"] == 0.8
    assert "CATBOOST_PROGRESS" in (output / "worker.log").read_text()


def test_existing_bound_snapshot_is_given_back_to_model(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    contract = load_contract()
    job = build_jobs(contract)[0]
    output = tmp_path / "job"
    data_dir = _data_dir(tmp_path, preprocessing_frame)
    kwargs = dict(
        job=job,
        contract=contract,
        data_dir=data_dir,
        output_dir=output,
        contract_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        code_sha256="c" * 64,
        absolute_deadline=time.time() + 60,
        model_factory=FakeCatBoost,
    )
    run_fold(**kwargs)
    run_fold(**kwargs)

    assert FakeCatBoost.instances[-1].fit_arguments[2]["snapshot_file"] == str(
        output / "experiment.cbsnapshot"
    )


def test_expired_deadline_returns_inconclusive_without_model(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    result = run_fold(
        job=build_jobs(load_contract())[0],
        contract=load_contract(),
        data_dir=_data_dir(tmp_path, preprocessing_frame),
        output_dir=tmp_path / "job",
        contract_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        code_sha256="c" * 64,
        absolute_deadline=0,
        model_factory=FakeCatBoost,
    )

    assert result.status == "budget_inconclusive"
    assert FakeCatBoost.instances == []
    assert not (tmp_path / "job" / "model.cbm").exists()
