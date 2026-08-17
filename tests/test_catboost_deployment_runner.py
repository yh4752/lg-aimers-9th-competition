from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
import time

import numpy as np
import pandas as pd

from experiments.catboost_deployment.inputs import VerifiedDeploymentInputs
from experiments.catboost_deployment.runner import run_deployment_campaign
from experiments.catboost_deployment.training import (
    run_alignment_job,
    run_full_fit_job,
)
from experiments.catboost_tabm_blend.inputs import VerifiedStageC, VerifiedTrainingInput


class FakeCatBoost:
    prediction = np.array([0.2, 0.8])

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, x, y, **arguments):
        Path(arguments["snapshot_file"]).write_bytes(b"snapshot")
        arguments["log_cout"].write("50:\tlearn: 0.5\ttotal: 1s\n")
        return self

    def predict(self, x, *, ntree_end=None):
        return np.resize(self.prediction, len(x))

    def save_model(self, path):
        Path(path).write_bytes(b"model")


class RegressingCatBoost(FakeCatBoost):
    prediction = np.array([0.9, 0.1])


def _verified(tmp_path: Path, preprocessing_frame: pd.DataFrame) -> VerifiedDeploymentInputs:
    rows = pd.concat(
        [preprocessing_frame, preprocessing_frame.iloc[:2]], ignore_index=True
    )
    rows["season"] = [2021, 2022, 2022, 2023, 2023, 2024, 2024]
    rows["row_id"] = [f"ROW_{index}" for index in range(len(rows))]
    rows["control_success"] = [0, 1, 0, 0, 1, 0, 1]
    rows["game_month"] = [3, 4, 5, 6, 7, 8, 9]
    data_dir = tmp_path / "official"
    data_dir.mkdir()
    rows.to_csv(data_dir / "train.csv", index=False)
    prediction_paths: dict[str, Path] = {}
    fold_rows = {
        "2022->2023": ([3, 4], ["oov", "oov"], ["known", "oov"]),
        "2023->2024": ([5, 6], ["known", "known"], ["known", "known"]),
    }
    for fold, (indices, pitcher_known, batter_known) in fold_rows.items():
        selected = rows.iloc[indices]
        frame = pd.DataFrame(
            {
                "row_id": selected["row_id"].astype(str).to_list(),
                "target": selected["control_success"].to_list(),
                "probability": [0.4, 0.6],
                "game_type": selected["game_type"].astype(str).to_list(),
                "game_month": selected["game_month"].to_list(),
                "pitcher_id_known": pitcher_known,
                "batter_id_known": batter_known,
            }
        )
        path = tmp_path / f"tabm-{fold}.csv"
        frame.to_csv(path, index=False)
        prediction_paths[fold] = path
    training = VerifiedTrainingInput(
        data_dir=data_dir,
        manifest_sha256="1" * 64,
        train_sha256="2" * 64,
        history_sha256="3" * 64,
    )
    stage_c = VerifiedStageC(
        delivery_sha256="4" * 64,
        review_sha256="5" * 64,
        resume_sha256="6" * 64,
        stage_state_sha256="7" * 64,
        prediction_paths=MappingProxyType(prediction_paths),
    )
    source_blend = tmp_path / "source-blend.zip"
    source_blend.write_bytes(b"source")
    return VerifiedDeploymentInputs(
        training=training,
        stage_c=stage_c,
        source_blend_path=source_blend,
        source_blend_sha256="8" * 64,
        source_blend_manifest_sha256="9" * 64,
        source_decision_sha256="a" * 64,
        source_selected_tabm_weight=0.7,
    )


def test_full_fit_runs_only_after_alignment_passes(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    calls: list[str] = []

    def alignment_runtime(**kwargs):
        calls.append(kwargs["job"].job_id)
        return run_alignment_job(**kwargs, model_factory=FakeCatBoost)

    def full_runtime(**kwargs):
        calls.append(kwargs["job"].job_id)
        return run_full_fit_job(**kwargs, model_factory=FakeCatBoost)

    result = run_deployment_campaign(
        verified=_verified(tmp_path, preprocessing_frame),
        output_dir=tmp_path / "output",
        resume_bundle=None,
        absolute_deadline=time.time() + 3000,
        alignment_runtime=alignment_runtime,
        full_runtime=full_runtime,
    )

    assert result.status == "full_training_complete"
    assert calls == ["align_2022_2023", "align_2023_2024", "full_2024"]
    assert result.decision is not None
    assert result.decision.selected_tree_count == 4
    assert result.bundles.delivery is not None


def test_blocked_alignment_never_calls_full_fit(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    calls: list[str] = []

    def alignment_runtime(**kwargs):
        calls.append(kwargs["job"].job_id)
        return run_alignment_job(**kwargs, model_factory=RegressingCatBoost)

    def forbidden_full(**kwargs):
        raise AssertionError("full fit must not run")

    result = run_deployment_campaign(
        verified=_verified(tmp_path, preprocessing_frame),
        output_dir=tmp_path / "output",
        resume_bundle=None,
        absolute_deadline=time.time() + 3000,
        alignment_runtime=alignment_runtime,
        full_runtime=forbidden_full,
    )

    assert result.status == "deployment_blocked"
    assert calls == ["align_2022_2023", "align_2023_2024"]
    assert result.bundles.delivery is None
    assert result.bundles.review is not None


def test_new_job_guard_returns_verified_incomplete_resume(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    callback_paths: list[Path] = []

    result = run_deployment_campaign(
        verified=_verified(tmp_path, preprocessing_frame),
        output_dir=tmp_path / "output",
        resume_bundle=None,
        absolute_deadline=time.time() + 10,
        on_verified_resume=callback_paths.append,
        alignment_runtime=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("guard must prevent worker start")
        ),
    )

    assert result.status == "deployment_incomplete"
    assert callback_paths == [result.bundles.resume]
    assert result.completed_job_ids == ()


def test_completed_resume_reuses_all_jobs_without_training(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    verified = _verified(tmp_path, preprocessing_frame)

    first = run_deployment_campaign(
        verified=verified,
        output_dir=tmp_path / "first",
        resume_bundle=None,
        absolute_deadline=time.time() + 3000,
        alignment_runtime=lambda **kwargs: run_alignment_job(
            **kwargs, model_factory=FakeCatBoost
        ),
        full_runtime=lambda **kwargs: run_full_fit_job(
            **kwargs, model_factory=FakeCatBoost
        ),
    )

    def forbidden(**kwargs):
        raise AssertionError("completed job must be reused")

    second = run_deployment_campaign(
        verified=verified,
        output_dir=tmp_path / "second",
        resume_bundle=first.bundles.resume,
        absolute_deadline=time.time() + 3000,
        alignment_runtime=forbidden,
        full_runtime=forbidden,
    )

    assert second.status == "full_training_complete"
    assert second.completed_job_ids == (
        "align_2022_2023",
        "align_2023_2024",
        "full_2024",
    )
