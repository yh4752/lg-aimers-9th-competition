from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import pandas as pd

from experiments.catboost_tabm_blend.contracts import build_jobs, load_contract
from experiments.catboost_tabm_blend.inputs import VerifiedStageC, VerifiedTrainingInput
from experiments.catboost_tabm_blend.runner import run_campaign
from experiments.catboost_tabm_blend.training import FoldResult


def _frame(probabilities: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["A", "B", "C", "D"],
            "target": [1, 0, 1, 0],
            "probability": probabilities,
            "game_type": ["R", "R", "P", "P"],
            "game_month": [4, 4, 5, 5],
            "pitcher_id_known": ["known", "known", "oov", "oov"],
            "batter_id_known": ["known", "oov", "known", "oov"],
        }
    )


def _verified_inputs(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text("unused")
    tabm_root = tmp_path / "tabm"
    tabm_root.mkdir()
    prediction_paths = {}
    for fold in ("2022->2023", "2023->2024"):
        path = tabm_root / f"{fold}.csv"
        _frame([0.6, 0.4, 0.6, 0.4]).to_csv(path, index=False)
        prediction_paths[fold] = path
    return (
        VerifiedTrainingInput(data, "c" * 64, "d" * 64, "e" * 64),
        VerifiedStageC("f" * 64, "1" * 64, "9" * 64, "2" * 64, prediction_paths),
    )


class FakeRuntime:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, **kwargs) -> FoldResult:
        job = kwargs["job"]
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        self.calls.append(job.job_id)
        _frame([0.8, 0.2, 0.8, 0.2]).to_csv(output / "predictions.csv", index=False)
        brier = 0.04
        (output / "model.cbm").write_bytes(b"model")
        (output / "worker.log").write_text("CATBOOST_JOB_END\n")
        (output / "job.json").write_text(json.dumps({"job_id": job.job_id}))
        (output / "metrics.json").write_text(json.dumps({"job_id": job.job_id, "brier": brier}))
        (output / "worker_result.json").write_text(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "status": "completed",
                    "brier": brier,
                    "model": "model.cbm",
                    "predictions": "predictions.csv",
                    "snapshot": None,
                }
            )
        )
        return FoldResult(
            job_id=job.job_id,
            status="completed",
            train_rows=10,
            valid_rows=4,
            brier=brier,
            model_path=output / "model.cbm",
            predictions_path=output / "predictions.csv",
            snapshot_path=None,
            elapsed_seconds=1.0,
            failure=None,
        )


def test_runner_executes_two_jobs_and_publishes_decision(tmp_path: Path) -> None:
    verified_input, verified_stage_c = _verified_inputs(tmp_path)
    runtime = FakeRuntime()
    resumes: list[Path] = []

    result = run_campaign(
        verified_input=verified_input,
        verified_stage_c=verified_stage_c,
        output_dir=tmp_path / "run",
        resume_bundle=None,
        absolute_deadline=time.time() + 3600,
        on_verified_resume=resumes.append,
        runtime=runtime,
    )

    assert result.stage_complete is True
    assert result.decision is not None
    assert result.decision.selected_tabm_weight == 0.7
    assert runtime.calls == [job.job_id for job in build_jobs(load_contract())]
    assert result.bundles.review is not None
    assert len(resumes) >= 2


def test_completed_resume_reuses_both_jobs(tmp_path: Path) -> None:
    verified_input, verified_stage_c = _verified_inputs(tmp_path)
    first_runtime = FakeRuntime()
    first = run_campaign(
        verified_input=verified_input,
        verified_stage_c=verified_stage_c,
        output_dir=tmp_path / "first",
        resume_bundle=None,
        absolute_deadline=time.time() + 3600,
        runtime=first_runtime,
    )
    second_runtime = FakeRuntime()
    second = run_campaign(
        verified_input=verified_input,
        verified_stage_c=verified_stage_c,
        output_dir=tmp_path / "second",
        resume_bundle=first.bundles.resume,
        absolute_deadline=time.time() + 3600,
        runtime=second_runtime,
    )

    assert second.stage_complete is True
    assert second_runtime.calls == []


def test_new_job_guard_publishes_empty_resume(tmp_path: Path) -> None:
    verified_input, verified_stage_c = _verified_inputs(tmp_path)
    runtime = FakeRuntime()

    result = run_campaign(
        verified_input=verified_input,
        verified_stage_c=verified_stage_c,
        output_dir=tmp_path / "guarded",
        resume_bundle=None,
        absolute_deadline=time.time() + 100,
        runtime=runtime,
    )

    assert result.stage_complete is False
    assert result.completed_job_ids == ()
    assert runtime.calls == []
    assert result.bundles.resume.is_file()
