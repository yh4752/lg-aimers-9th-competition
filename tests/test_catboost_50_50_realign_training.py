from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from experiments.catboost_50_50_realign.contracts import load_contract
from experiments.catboost_50_50_realign.metrics import (
    CATBOOST_COLUMNS,
    PrefixEvidence,
    RealignDecision,
)
from experiments.catboost_50_50_realign.training import (
    RealignTrainingError,
    run_catboost_f1,
    run_full_fit,
)


class FakeCatBoost:
    instances: list["FakeCatBoost"] = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.fit_calls: list[tuple[pd.DataFrame, np.ndarray, dict[str, object]]] = []
        self.predict_prefixes: list[int] = []
        self.__class__.instances.append(self)

    def fit(self, x, y, **arguments):
        self.fit_calls.append((x.copy(), np.asarray(y).copy(), dict(arguments)))
        Path(arguments["snapshot_file"]).write_bytes(b"snapshot")
        arguments["log_cout"].write("4:\tlearn: 0.5\ttotal: 1s\n")
        return self

    def predict(self, x, *, ntree_end=None):
        if ntree_end is None:
            return np.full(len(x), 0.55)
        self.predict_prefixes.append(int(ntree_end))
        return np.where(np.arange(len(x)) % 2, 0.7, 0.3)

    def save_model(self, path):
        Path(path).write_bytes(b"model")


def _data_dir(tmp_path: Path, preprocessing_frame: pd.DataFrame) -> Path:
    base = preprocessing_frame.iloc[[0, 1, 2, 3, 4, 0]].reset_index(drop=True).copy()
    base["season"] = [2019, 2020, 2021, 2022, 2023, 2024]
    base["row_id"] = [f"ROW_{index}" for index in range(len(base))]
    base["control_success"] = [0, 1, 0, 1, 0, 1]
    base["game_month"] = [4, 5, 6, 7, 8, 9]
    root = tmp_path / "data"
    root.mkdir()
    base.to_csv(root / "train.csv", index=False)
    return root


def _promoted(tree_count: int = 16) -> RealignDecision:
    candidates = tuple(
        PrefixEvidence(
            tree_count=prefix,
            fold_gain={
                "2021->2022": 0.002 if prefix == tree_count else 0.001,
                "2022->2023": 0.002 if prefix == tree_count else 0.001,
                "2023->2024": 0.002 if prefix == tree_count else 0.001,
            },
            weighted_gain=0.002 if prefix == tree_count else 0.001,
            latest_bootstrap_lower=0.0001,
            maximum_segment_regression=0.0,
            eligible_segment_count=3,
            bootstrap_status="completed",
            passed=True,
        )
        for prefix in (4, 8, 12, 16, 20, 24, 28, 32)
    )
    return RealignDecision(
        status="promoted",
        selected_tree_count=tree_count,
        candidates=candidates,
        reason="selected_by_preregistered_order",
    )


def test_f1_fits_once_and_predicts_each_fixed_prefix(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    result = run_catboost_f1(
        data_dir=_data_dir(tmp_path, preprocessing_frame),
        output_dir=tmp_path / "f1",
        absolute_deadline=time.time() + 60,
        model_factory=FakeCatBoost,
    )

    assert result.status == "completed"
    model = FakeCatBoost.instances[-1]
    assert model.parameters["iterations"] == 400
    assert model.predict_prefixes == [4, 8, 12, 16, 20, 24, 28, 32]
    assert len(model.fit_calls) == 1
    x_train, _, fit_arguments = model.fit_calls[0]
    assert len(x_train) == 3
    assert fit_arguments["use_best_model"] is False
    assert fit_arguments["snapshot_interval"] == 300
    assert "early_stopping_rounds" not in fit_arguments
    assert "control_success" not in x_train
    assert tuple(pd.read_csv(result.predictions_path).columns) == CATBOOST_COLUMNS


def test_full_fit_requires_promoted_decision_before_reading_data(tmp_path: Path) -> None:
    blocked = RealignDecision(
        status="rejected",
        selected_tree_count=None,
        candidates=(),
        reason="no_prefix_passed_all_preregistered_gates",
    )
    with pytest.raises(RealignTrainingError, match="promoted decision"):
        run_full_fit(
            data_dir=tmp_path / "missing",
            output_dir=tmp_path / "full",
            decision=blocked,
            absolute_deadline=time.time() + 60,
            model_factory=FakeCatBoost,
        )


def test_full_fit_rejects_promoted_label_without_passing_selected_evidence(
    tmp_path: Path,
) -> None:
    promoted = _promoted(16)
    forged = RealignDecision(
        status="promoted",
        selected_tree_count=16,
        candidates=(
            PrefixEvidence(
                **{
                    **promoted.candidates[3].__dict__,
                    "passed": False,
                }
            ),
        ),
        reason="selected_by_preregistered_order",
    )
    with pytest.raises(RealignTrainingError, match="promoted decision"):
        run_full_fit(
            data_dir=tmp_path / "missing",
            output_dir=tmp_path / "full",
            decision=forged,
            absolute_deadline=time.time() + 60,
            model_factory=FakeCatBoost,
        )


def test_full_fit_uses_all_rows_selected_prefix_and_no_eval_set(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    data_dir = _data_dir(tmp_path, preprocessing_frame)
    result = run_full_fit(
        data_dir=data_dir,
        output_dir=tmp_path / "full",
        decision=_promoted(16),
        absolute_deadline=time.time() + 60,
        model_factory=FakeCatBoost,
    )

    assert result.status == "completed"
    model = FakeCatBoost.instances[-1]
    assert model.parameters["iterations"] == 16
    assert len(model.fit_calls[0][0]) == len(pd.read_csv(data_dir / "train.csv"))
    fit_arguments = model.fit_calls[0][2]
    assert "eval_set" not in fit_arguments
    assert fit_arguments["use_best_model"] is False
    assert result.predictions_path is None
    assert result.model_path.read_bytes() == b"model"


def test_expired_deadline_is_inconclusive_without_starting_model(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    result = run_catboost_f1(
        data_dir=_data_dir(tmp_path, preprocessing_frame),
        output_dir=tmp_path / "f1",
        absolute_deadline=0,
        model_factory=FakeCatBoost,
    )
    assert result.status == "budget_inconclusive"
    assert FakeCatBoost.instances == []


def test_snapshot_identity_drift_is_rejected_before_model_fit(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    FakeCatBoost.instances.clear()
    output = tmp_path / "f1"
    output.mkdir()
    (output / "job.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RealignTrainingError, match="job identity differs"):
        run_catboost_f1(
            data_dir=_data_dir(tmp_path, preprocessing_frame),
            output_dir=output,
            absolute_deadline=time.time() + 60,
            model_factory=FakeCatBoost,
        )
    assert FakeCatBoost.instances == []


def test_training_writes_bound_metrics_and_atomic_outputs(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    output = tmp_path / "f1"
    result = run_catboost_f1(
        data_dir=_data_dir(tmp_path, preprocessing_frame),
        output_dir=output,
        absolute_deadline=time.time() + 60,
        input_manifest_sha256="a" * 64,
        code_sha256="b" * 64,
        model_factory=FakeCatBoost,
    )
    job = json.loads((output / "job.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert job["input_manifest_sha256"] == "a" * 64
    assert job["code_sha256"] == "b" * 64
    assert metrics["predictions_sha256"]
    assert metrics["model_sha256"]
    assert result.preprocessing_path.is_file()
    assert not list(output.glob("*.tmp*"))


def test_retry_after_interrupted_fit_reuses_same_bound_snapshot(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    class InterruptingCatBoost(FakeCatBoost):
        def fit(self, x, y, **arguments):
            Path(arguments["snapshot_file"]).write_bytes(b"snapshot")
            raise RuntimeError("interrupted")

    data_dir = _data_dir(tmp_path, preprocessing_frame)
    output = tmp_path / "f1"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_catboost_f1(
            data_dir=data_dir,
            output_dir=output,
            absolute_deadline=time.time() + 60,
            input_manifest_sha256="a" * 64,
            code_sha256="b" * 64,
            model_factory=InterruptingCatBoost,
        )

    assert (output / "experiment.cbsnapshot").read_bytes() == b"snapshot"
    result = run_catboost_f1(
        data_dir=data_dir,
        output_dir=output,
        absolute_deadline=time.time() + 60,
        input_manifest_sha256="a" * 64,
        code_sha256="b" * 64,
        model_factory=FakeCatBoost,
    )
    assert result.status == "completed"


def test_snapshot_symlink_is_rejected_before_model_fit(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    data_dir = _data_dir(tmp_path, preprocessing_frame)
    output = tmp_path / "f1"
    output.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep")
    (output / "experiment.cbsnapshot").symlink_to(victim)

    with pytest.raises(RealignTrainingError, match="snapshot path is invalid"):
        run_catboost_f1(
            data_dir=data_dir,
            output_dir=output,
            absolute_deadline=time.time() + 60,
            model_factory=FakeCatBoost,
        )
    assert victim.read_bytes() == b"keep"
