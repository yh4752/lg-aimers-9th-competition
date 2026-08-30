from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time

import pandas as pd

from experiments.direct_expert.artifacts import DirectExpertBindings
from experiments.direct_expert.stage_a import run_stage_a


@dataclass(frozen=True)
class FakeResult:
    status: str
    job_id: str
    predictions: pd.DataFrame
    output_dir: Path


class FakeRuntime:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.gpus: list[int] = []
        self.failures: set[str] = set()
        self.lock = threading.Lock()
        self.active_by_gpu = {0: 0, 1: 0}
        self.maximum_by_gpu = {0: 0, 1: 0}
        self.bindings = DirectExpertBindings(*[str(i) * 64 for i in range(1, 7)])

    def now(self) -> float:
        return 0.0

    def fail(self, job_id: str) -> None:
        self.failures.add(job_id)

    def run_job(self, job, gpu: int, output: Path) -> FakeResult:
        with self.lock:
            self.started.append(job.job_id)
            self.gpus.append(gpu)
            self.active_by_gpu[gpu] += 1
            self.maximum_by_gpu[gpu] = max(self.maximum_by_gpu[gpu], self.active_by_gpu[gpu])
        try:
            time.sleep(0.002)
            if job.job_id in self.failures:
                raise RuntimeError("synthetic failure")
            output.mkdir(parents=True)
            year = job.fold[1]
            frame = pd.DataFrame(
                {
                    "row_id": [f"{year}-0", f"{year}-1"],
                    "target": [0, 1],
                    "probability": [0.4, 0.6],
                    "game_type": ["R", "F"],
                    "pitcher_id": [1, 2],
                    "oof_year": [year, year],
                }
            )
            path = output / "predictions.csv"
            frame.to_csv(path, index=False)
            return FakeResult("completed", job.job_id, frame, output)
        finally:
            with self.lock:
                self.active_by_gpu[gpu] -= 1

    def e2_oof(self, year: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_id": [f"{year}-0", f"{year}-1"],
                "target": [0, 1],
                "p_anchor": [0.45, 0.55],
            }
        )


def test_stage_a_runs_sixteen_unique_jobs_on_two_workers(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    result = run_stage_a(runtime, tmp_path, absolute_deadline=10_000)
    assert result.status == "completed"
    assert len(runtime.started) == 16
    assert len(set(runtime.started)) == 16
    assert set(runtime.gpus) == {0, 1}
    assert runtime.maximum_by_gpu == {0: 1, 1: 1}
    assert result.handoff.is_file()


def test_one_failed_candidate_does_not_stop_siblings(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    runtime.fail("screen__D7__2021_2022__s3407")
    result = run_stage_a(runtime, tmp_path, absolute_deadline=10_000)
    assert result.status == "completed_with_candidate_failure"
    assert len(result.completed_jobs) == 15


def test_deadline_stops_new_jobs_and_publishes_handoff(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    result = run_stage_a(runtime, tmp_path, absolute_deadline=1)
    assert result.status == "incomplete"
    assert not runtime.started
    assert result.handoff.is_file()


def test_stage_a_resume_reuses_all_completed_jobs(tmp_path: Path) -> None:
    first = run_stage_a(FakeRuntime(), tmp_path / "first", absolute_deadline=10_000)
    resumed_runtime = FakeRuntime()
    second = run_stage_a(
        resumed_runtime,
        tmp_path / "second",
        absolute_deadline=1,
        previous_handoff=first.handoff,
    )
    assert second.status == "completed"
    assert resumed_runtime.started == []
