from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from experiments.tree_expert.s4_artifacts import S4Bindings
from experiments.tree_expert.s4_artifacts import S4ArtifactError
from experiments.tree_expert.s4_runner import S4Job, run_s4_campaign
from experiments.tree_expert.s4_state import S4State, record_s4_decision


class FakeRuntime:
    def __init__(self, fail=False):
        self.fail = fail
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def jobs_for_phase(self, phase, state, root):
        counts = {"anchors": 2, "residuals": 2, "full_chains": 12, "confirmation": 2, "full_fit": 0}
        return tuple(S4Job(f"{phase}__{i:02d}", phase, {}) for i in range(counts[phase]))

    def run_job(self, job, job_dir, gpu_id, deadline):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            if self.fail and job.job_id == "anchors__00":
                raise RuntimeError("worker failed")
            job_dir.mkdir(parents=True)
            (job_dir / "result.json").write_text("{}")
            time.sleep(0.01)
            return "completed"
        finally:
            with self.lock: self.active -= 1

    def finalize_phase(self, phase, state, root):
        if phase == "confirmation":
            return record_s4_decision(state, "candidate", "rejected")
        return state


def _bindings():
    return S4Bindings(*("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, "f" * 64))


def test_runner_uses_two_workers_without_duplicate_jobs(tmp_path):
    runtime = FakeRuntime()
    result = run_s4_campaign(_bindings(), tmp_path / "campaign", runtime=runtime, wall_deadline=time.monotonic() + 10000)
    assert result.maximum_concurrent_gpu_jobs == 2
    assert len(result.started_jobs) == len(set(result.started_jobs))
    assert result.completed_full_chain_count == 12


def test_priority_finishes_twelve_full_chains_before_optional_seeds(tmp_path):
    runtime = FakeRuntime()
    result = run_s4_campaign(_bindings(), tmp_path / "campaign", runtime=runtime, wall_deadline=time.monotonic() + 10000)
    first_confirmation = next(i for i, name in enumerate(result.started_jobs) if name.startswith("confirmation"))
    assert sum(name.startswith("full_chains") for name in result.started_jobs[:first_confirmation]) == 12


def test_exception_publishes_one_emergency_handoff(tmp_path):
    with pytest.raises(RuntimeError, match="worker failed"):
        run_s4_campaign(_bindings(), tmp_path / "campaign", runtime=FakeRuntime(True), wall_deadline=time.monotonic() + 10000)
    assert [path.name for path in tmp_path.glob("*handoff.zip")] == ["anchor_residual_hierarchical_handoff.zip"]


class AdvancingClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 2000.0
        return self.value


class FirstLowSpace:
    def __init__(self):
        self.calls = 0

    def __call__(self, _path):
        self.calls += 1
        free = 0 if self.calls == 1 else 10 * 1024**3
        return SimpleNamespace(total=20 * 1024**3, used=0, free=free)


def test_low_space_skips_periodic_snapshot_then_continues(tmp_path):
    runtime = FakeRuntime()
    result = run_s4_campaign(
        _bindings(), tmp_path / "campaign", runtime=runtime,
        wall_deadline=10**12, clock=AdvancingClock(), disk_usage=FirstLowSpace(),
    )
    log = (tmp_path / "campaign/s4_campaign.log").read_text()
    assert "S4_SNAPSHOT_SKIPPED reason=insufficient_space" in log
    assert "S4_SNAPSHOT_READY" in log
    assert result.status == "completed"


def test_low_space_finalization_preserves_previous_handoff(tmp_path):
    previous = tmp_path / "anchor_residual_hierarchical_handoff.zip"
    previous.write_bytes(b"previous-safe-handoff")
    result = run_s4_campaign(
        _bindings(), tmp_path / "campaign", runtime=FakeRuntime(),
        wall_deadline=time.monotonic() + 10000,
        disk_usage=lambda _path: SimpleNamespace(total=1, used=1, free=0),
    )
    assert result.status == "incomplete"
    assert previous.read_bytes() == b"previous-safe-handoff"


def test_low_space_finalization_requires_at_least_one_handoff(tmp_path):
    with pytest.raises(S4ArtifactError, match="final S4 handoff"):
        run_s4_campaign(
            _bindings(), tmp_path / "campaign", runtime=FakeRuntime(),
            wall_deadline=time.monotonic() + 10000,
            disk_usage=lambda _path: SimpleNamespace(total=1, used=1, free=0),
        )
