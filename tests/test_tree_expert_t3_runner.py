from __future__ import annotations

from pathlib import Path
import threading
import time

from experiments.tree_expert.t3_contracts import T3Job
import pytest

from experiments.tree_expert.t3_runner import (
    T3CampaignState,
    T3RunnerError,
    require_full_fit_window,
    run_pending_jobs,
)


def jobs():
    return tuple(
        T3Job(f"job_{index}", "recent", None, 2023, 2024, 3407 + index)
        for index in range(4)
    )


class RecordingExecutor:
    def __init__(self):
        self.started = []
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def __call__(self, job, output_dir, gpu_id):
        with self.lock:
            self.started.append((job.job_id, gpu_id))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        time.sleep(0.02)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            '{"status":"completed","job_id":"%s"}' % job.job_id
        )
        with self.lock:
            self.active -= 1
        return "completed"


def state(tmp_path):
    return T3CampaignState(root=tmp_path, completed_jobs=set(), failed_jobs=set())


def test_runner_uses_two_workers_and_records_completed_jobs(tmp_path):
    executor = RecordingExecutor()
    active = state(tmp_path)
    run_pending_jobs(active, jobs(), executor=executor, gpu_ids=(0, 1))
    assert executor.maximum == 2
    assert {gpu for _, gpu in executor.started} == {0, 1}
    assert active.completed_jobs == {job.job_id for job in jobs()}


def test_runner_reuses_completed_jobs(tmp_path):
    executor = RecordingExecutor()
    active = state(tmp_path)
    active.completed_jobs.add("job_0")
    run_pending_jobs(active, jobs(), executor=executor, gpu_ids=(0, 1))
    assert "job_0" not in {job for job, _ in executor.started}


def test_runner_stops_before_new_job_guard(tmp_path):
    executor = RecordingExecutor()
    active = state(tmp_path)
    run_pending_jobs(
        active, jobs(), executor=executor, gpu_ids=(0, 1),
        wall_deadline=10_599.0, clock=lambda: 10_000.0, new_job_guard_seconds=600,
    )
    assert executor.started == []
    assert active.completed_jobs == set()


def test_full_fit_requires_a_dedicated_time_window():
    with pytest.raises(T3RunnerError, match="full fit deferred"):
        require_full_fit_window(
            wall_deadline=13_599.0,
            clock=lambda: 10_000.0,
            guard_seconds=3_600,
        )
    require_full_fit_window(
        wall_deadline=13_600.0,
        clock=lambda: 10_000.0,
        guard_seconds=3_600,
    )
