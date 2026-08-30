from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import time

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from experiments.tree_expert.inputs import PREDICTION_COLUMNS, VerifiedOfficialData
from experiments.tree_privileged.features import CandidateFeatureBatch, CandidateFeatureSkip
from experiments.tree_privileged.training import PrivilegedJob, run_candidate_job


class RecordingModel:
    def __init__(self, owner): self.owner = owner
    def fit(self, x, y, **kwargs): self.owner.fit_target = np.asarray(y); return self
    def predict(self, x): return np.array([0.02, -0.01])
    def save_model(self, path): Path(path).write_bytes(b"model")
    def get_best_iteration(self): return 7


class RecordingFactory:
    def __init__(self): self.fit_target = None
    def __call__(self, parameters): return RecordingModel(self)


def _data(tmp_path: Path) -> VerifiedOfficialData:
    train = pd.DataFrame({
        "row_id": ["a", "b", "c", "d"], "season": [2022, 2022, 2023, 2023],
        "control_success": [0, 1, 1, 0], "game_type": ["R"] * 4,
        "game_month": [5] * 4, "pitcher_id_known": [1] * 4, "batter_id_known": [1] * 4,
    })
    train_path = tmp_path / "train.csv"; train.to_csv(train_path, index=False)
    history = tmp_path / "trackman_history.csv"; pd.DataFrame({"trackman_id": [1]}).to_csv(history, index=False)
    return VerifiedOfficialData(tmp_path, train_path, history, "1" * 64, "2" * 64)


def _baseline() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": ["c", "d"], "target": [1, 0], "probability": [0.5, 0.5],
        "game_type": ["R", "R"], "game_month": [5, 5],
        "pitcher_id_known": [1, 1], "batter_id_known": [1, 1],
    }).loc[:, PREDICTION_COLUMNS]


def _feature_builder(train, history, *, valid_year, candidate_id, teacher_evidence, strengths):
    state = SimpleNamespace(
        categorical_columns=("cat",), candidate_id=candidate_id, feature_columns=("cat",),
        profile_columns=(), teacher_evidence_hashes=MappingProxyType({}), selected_strengths=None,
    )
    batch = CandidateFeatureBatch(
        pd.DataFrame({"cat": ["x", "y"]}), np.array([0.4, 0.6]), np.array(["a", "b"]),
        np.array([0, 1]), np.array([0.2, 0.8]), None,
    )
    return state, batch


def _transform(rows, state):
    return CandidateFeatureBatch(pd.DataFrame({"cat": ["x", "y"]}), np.array([0.5, 0.5]),
                                 np.array(["c", "d"]), None, None, None)


def test_residual_student_fits_soft_target_minus_anchor(tmp_path: Path) -> None:
    factory = RecordingFactory()
    result = run_candidate_job(
        job=PrivilegedJob("pd35", "PD35", 2022, 2023, 3407), data=_data(tmp_path),
        baseline=_baseline(), output_dir=tmp_path / "job", absolute_deadline=time.time() + 60,
        gpu_id=0, model_factory=factory, feature_builder=_feature_builder, feature_transformer=_transform,
    )
    assert_allclose(factory.fit_target, np.array([-0.2, 0.2]))
    assert result.status == "completed"
    assert result.predictions_path.is_file()
    assert (tmp_path / "job/feature_state.zip").is_file()


def test_distillation_job_skips_before_gpu_fit_when_coverage_fails(tmp_path: Path) -> None:
    called = False
    def skipped(*args, **kwargs):
        raise CandidateFeatureSkip("teacher_coverage_gate")
    def factory(_):
        nonlocal called; called = True; raise AssertionError
    result = run_candidate_job(
        job=PrivilegedJob("d15", "D15", 2022, 2023, 3407), data=_data(tmp_path),
        baseline=_baseline(), output_dir=tmp_path / "job", absolute_deadline=time.time() + 60,
        gpu_id=0, model_factory=factory, feature_builder=skipped, feature_transformer=_transform,
    )
    assert result.status == "skipped" and result.failure == "teacher_coverage_gate"
    assert called is False
