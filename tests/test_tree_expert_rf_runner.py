from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from experiments.tree_expert.rf_contracts import RFJob
from experiments.tree_expert.rf_runner import (
    RFCampaignState,
    RFRunnerError,
    _prepare_full_fit_output,
    require_full_fit_window,
    run_pending_jobs,
)


def _jobs() -> tuple[RFJob, ...]:
    return tuple(
        RFJob(f"job_{index}", "f_small", "F", 2023, 2024, 3407 + index)
        for index in range(4)
    )


class RecordingExecutor:
    def __init__(self):
        self.started: list[tuple[str, int]] = []
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


class TemporaryFileExecutor:
    def __init__(self):
        self.temporary_ready = threading.Event()

    def __call__(self, job, output_dir, _gpu_id):
        output_dir.mkdir(parents=True, exist_ok=True)
        if job.job_id == "job_0":
            assert self.temporary_ready.wait(timeout=1)
        else:
            temporary = output_dir / ".predictions.csv.tmp"
            temporary.write_text("partial")
            self.temporary_ready.set()
            time.sleep(0.1)
            temporary.unlink()
        return "completed"


def _state(tmp_path: Path) -> RFCampaignState:
    return RFCampaignState(
        root=tmp_path,
        phase="structure",
        status="running",
        completed_jobs=set(),
        failed_jobs=set(),
    )


def test_runner_uses_two_workers_and_records_completed_jobs(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    state = _state(tmp_path)

    run_pending_jobs(state, _jobs(), executor=executor, gpu_ids=(0, 1))

    assert executor.maximum == 2
    assert {gpu for _, gpu in executor.started} == {0, 1}
    assert state.completed_jobs == {job.job_id for job in _jobs()}


def test_runner_reuses_completed_jobs(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    state = _state(tmp_path)
    state.completed_jobs.add("job_0")

    run_pending_jobs(state, _jobs(), executor=executor, gpu_ids=(0, 1))

    assert "job_0" not in {job for job, _ in executor.started}


def test_snapshot_runs_only_after_both_workers_finish(tmp_path: Path) -> None:
    executor = TemporaryFileExecutor()
    state = _state(tmp_path)
    snapshots = []

    def snapshot():
        assert list(tmp_path.rglob("*.tmp")) == []
        snapshots.append(tuple(sorted(state.completed_jobs)))

    run_pending_jobs(
        state,
        _jobs()[:2],
        executor=executor,
        gpu_ids=(0, 1),
        on_progress=snapshot,
    )

    assert snapshots == [("job_0", "job_1")]


def test_runner_stops_before_new_job_guard(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    state = _state(tmp_path)

    run_pending_jobs(
        state,
        _jobs(),
        executor=executor,
        gpu_ids=(0, 1),
        wall_deadline=10_599.0,
        clock=lambda: 10_000.0,
        new_job_guard_seconds=600,
    )

    assert executor.started == []
    assert state.completed_jobs == set()


def test_full_fit_requires_a_dedicated_time_window() -> None:
    with pytest.raises(RFRunnerError, match="full fit deferred"):
        require_full_fit_window(
            wall_deadline=11_799.0,
            clock=lambda: 10_000.0,
            guard_seconds=1_800,
        )
    require_full_fit_window(
        wall_deadline=11_800.0,
        clock=lambda: 10_000.0,
        guard_seconds=1_800,
    )


def test_full_fit_retry_discards_only_partial_full_fit_directory(tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    partial = output / "full_fit"
    partial.mkdir(parents=True)
    (partial / "partial.cbm").write_bytes(b"partial")
    evidence = output / "decisions/acceptance.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"status":"accepted"}')

    prepared = _prepare_full_fit_output(output)

    assert prepared == output / "full_fit"
    assert prepared.is_dir()
    assert list(prepared.iterdir()) == []
    assert evidence.is_file()


def test_full_fit_retry_rejects_symlink_target(tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    output.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (output / "full_fit").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(RFRunnerError, match="full-fit output path differs"):
        _prepare_full_fit_output(output)
