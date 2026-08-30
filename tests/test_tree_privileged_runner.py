from __future__ import annotations

import json
from pathlib import Path
import time

import pandas as pd

from experiments.tree_expert.inputs import PREDICTION_COLUMNS, VerifiedOfficialData
from experiments.tree_privileged.profiles import ProfileStrengths
from experiments.tree_privileged.runner import _run_jobs
from experiments.tree_privileged.training import CandidateJobResult, PrivilegedJob


def _launcher(job, data, baseline_path, teacher_cache, strengths, output_dir, deadline, gpu_id):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    prediction = pd.read_csv(baseline_path); prediction.to_csv(output / "predictions.csv", index=False)
    (output / "model.cbm").write_bytes(b"model")
    result = {"job_id": job.job_id, "candidate_id": job.candidate_id, "status": "completed",
              "brier": .2, "failure": None, "best_iteration": 3}
    (output / "worker_result.json").write_text(json.dumps(result))
    (output / "gpu.txt").write_text(str(gpu_id))
    time.sleep(.03)
    return CandidateJobResult(job.job_id, job.candidate_id, "completed", .2, output / "model.cbm",
                              output / "predictions.csv", None, 3)


def _inputs(tmp_path: Path):
    train = tmp_path / "train.csv"; history = tmp_path / "history.csv"
    train.write_text("x\n1\n"); history.write_text("x\n1\n")
    data = VerifiedOfficialData(tmp_path, train, history, "1" * 64, "2" * 64)
    baseline_root = tmp_path / "baseline"; baseline_root.mkdir()
    frame = pd.DataFrame({"row_id": ["r"], "target": [1], "probability": [.5], "game_type": ["R"],
                          "game_month": [1], "pitcher_id_known": [1], "batter_id_known": [1]}).loc[:, PREDICTION_COLUMNS]
    frame.to_csv(baseline_root / "2022.csv", index=False)
    return data, baseline_root


def test_scheduler_assigns_only_two_gpus_and_reuses_completed(tmp_path: Path) -> None:
    data, baseline = _inputs(tmp_path)
    jobs = [PrivilegedJob(f"p-{index}", "P", 2021, 2022, 3407) for index in range(4)]
    results, paused = _run_jobs(
        jobs, data=data, baseline_root=baseline, teacher_root=tmp_path / "teacher",
        strengths=ProfileStrengths(25, 50, 100), jobs_root=tmp_path / "jobs",
        deadline=time.time() + 10_000, gpu_ids=(0, 1), log=tmp_path / "log", launcher=_launcher,
    )
    assert paused is False and len(results) == 4
    assert {int((tmp_path / f"jobs/p-{index}/gpu.txt").read_text()) for index in range(4)} == {0, 1}

    replay, replay_paused = _run_jobs(
        jobs, data=data, baseline_root=baseline, teacher_root=tmp_path / "teacher",
        strengths=ProfileStrengths(25, 50, 100), jobs_root=tmp_path / "jobs",
        deadline=time.time() + 10_000, gpu_ids=(0, 1), log=tmp_path / "log",
        launcher=lambda *args: (_ for _ in ()).throw(AssertionError("must reuse")),
    )
    assert replay_paused is False and set(replay) == set(results)


def test_scheduler_starts_nothing_inside_two_hour_guard(tmp_path: Path) -> None:
    data, baseline = _inputs(tmp_path)
    now = time.time()
    results, paused = _run_jobs(
        [PrivilegedJob("p", "P", 2021, 2022, 3407)], data=data, baseline_root=baseline,
        teacher_root=tmp_path / "teacher", strengths=ProfileStrengths(25, 50, 100),
        jobs_root=tmp_path / "jobs", deadline=now + 7000, gpu_ids=(0, 1),
        log=tmp_path / "log", clock=lambda: now, launcher=_launcher,
    )
    assert paused is True and results == {}
