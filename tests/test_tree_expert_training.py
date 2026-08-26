from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.contracts import build_e1_jobs, load_e1_contract
from experiments.tree_expert.failure_labels import FailureLabelAudit
from experiments.tree_expert.features import TreeFeatureBatch, TreeFeatureSkip
from experiments.tree_expert.inputs import PREDICTION_COLUMNS, VerifiedOfficialData, file_sha256
from experiments.tree_expert.training import run_e1_job


class FakeModel:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.classes_ = np.asarray([0, 1, 2, 3] if kind == "multiclass" else [0, 1])

    def fit(self, _x, _y, **kwargs) -> "FakeModel":
        snapshot = kwargs.get("snapshot_file")
        if snapshot is not None:
            Path(snapshot).write_bytes(b"snapshot")
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability = np.linspace(0.4, 0.6, len(frame))
        if self.kind == "multiclass":
            return np.column_stack(
                [probability, (1 - probability) / 3, (1 - probability) / 3, (1 - probability) / 3]
            )
        return np.column_stack([1 - probability, probability])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.linspace(-0.1, 0.1, len(frame))

    def save_model(self, path: str) -> None:
        Path(path).write_bytes(f"fake-{self.kind}".encode("ascii"))

    def get_best_iteration(self) -> int:
        return 1


class FakeCatBoostFactory:
    def __call__(self, kind: str, _parameters: dict[str, object]) -> FakeModel:
        return FakeModel(kind)


@dataclass
class FeatureHarness:
    def fit(self, rows: pd.DataFrame, _history, **_kwargs):
        return object(), self._batch(rows)

    def transform(self, rows: pd.DataFrame, _state: object):
        return self._batch(rows)

    @staticmethod
    def _batch(rows: pd.DataFrame) -> TreeFeatureBatch:
        target = (
            rows["control_success"].to_numpy(dtype="int8", copy=True)
            if "control_success" in rows
            else None
        )
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


def _failure_audit(rows: pd.DataFrame, **_kwargs) -> FailureLabelAudit:
    labels = np.asarray(["success", "middle", "reverse", "other_failure"], dtype=object)
    positions = np.arange(4, dtype="int64")
    return FailureLabelAudit(
        status="passed",
        reason="fixture",
        labels=labels,
        source_positions=positions,
        coverage=1.0,
        binary_delta_fraction=1.0,
        success_agreement=1.0,
        middle_reverse_overlap=0.0,
        class_counts={name: 1 for name in labels},
    )


@pytest.fixture
def e1_fixture(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    train = pd.DataFrame(
        {
            "row_id": ["t0", "t1", "t2", "t3", "v0", "v1"],
            "season": [2023, 2023, 2023, 2023, 2024, 2024],
            "control_success": [1, 0, 0, 0, 0, 1],
        }
    )
    train.to_csv(root / "train.csv", index=False)
    pd.DataFrame({"season": [2023]}).to_csv(root / "trackman_history.csv", index=False)
    data = VerifiedOfficialData(
        root=root,
        train=root / "train.csv",
        history=root / "trackman_history.csv",
        train_sha256=file_sha256(root / "train.csv"),
        history_sha256=file_sha256(root / "trackman_history.csv"),
    )
    baseline = pd.DataFrame(
        {
            "row_id": ["v0", "v1"],
            "target": [0, 1],
            "probability": [0.45, 0.55],
            "game_type": ["R", "F"],
            "game_month": [3, 4],
            "pitcher_id_known": ["known", "oov"],
            "batter_id_known": ["known", "known"],
        }
    ).loc[:, PREDICTION_COLUMNS]
    jobs = {job.candidate_id: job for job in build_e1_jobs(load_e1_contract())}
    return type(
        "E1Fixture",
        (),
        {"data": data, "baseline": baseline, "jobs": jobs},
    )()


@pytest.mark.parametrize(
    ("candidate", "expected_mode"),
    [
        ("c0_native_ctr", "binary"),
        ("c1_anchor_residual", "residual"),
        ("c2_trackman_residual", "residual"),
        ("c3_failure_aware", "multiclass"),
    ],
)
def test_fold_worker_writes_valid_probability_for_each_objective(
    tmp_path: Path,
    candidate: str,
    expected_mode: str,
    e1_fixture,
) -> None:
    harness = FeatureHarness()
    result = run_e1_job(
        job=e1_fixture.jobs[candidate],
        data=e1_fixture.data,
        baseline=e1_fixture.baseline,
        output_dir=tmp_path / candidate,
        absolute_deadline=time.time() + 60,
        gpu_id=0,
        model_factory=FakeCatBoostFactory(),
        feature_builder=harness.fit,
        feature_transformer=harness.transform,
        failure_auditor=_failure_audit,
    )

    assert result.status == "completed"
    frame = pd.read_csv(result.predictions_path)
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert frame["probability"].between(0, 1).all()
    metrics = json.loads((tmp_path / candidate / "metrics.json").read_text())
    assert metrics["objective"] == expected_mode
    assert result.model_path.is_file()


def test_skippable_candidate_does_not_fail_worker(tmp_path: Path, e1_fixture) -> None:
    def skipped_builder(*_args, **_kwargs):
        raise TreeFeatureSkip("trackman_coverage=0.100000")

    result = run_e1_job(
        job=e1_fixture.jobs["c2_trackman_residual"],
        data=e1_fixture.data,
        baseline=e1_fixture.baseline,
        output_dir=tmp_path / "c2_trackman_residual",
        absolute_deadline=time.time() + 60,
        gpu_id=0,
        model_factory=FakeCatBoostFactory(),
        feature_builder=skipped_builder,
    )

    assert result.status == "skipped"
    assert result.model_path is None
