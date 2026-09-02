from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.failure_regime_e3.artifacts import E3Bindings, verify_bundle
from experiments.failure_regime_e3.runner import run_campaign


def _bindings() -> E3Bindings:
    return E3Bindings(*("abcdef"[index] * 64 for index in range(6)))


@dataclass
class FakeRuntime:
    decision_status: str = "accepted"
    audit_status: bool = True
    clock: float = 0.0
    step: float = 0.0
    fail_once: bool = False

    def __post_init__(self) -> None:
        self.bindings = _bindings()
        self.calls: list[str] = []
        self.compacted: list[str] = []

    def now(self) -> float:
        return self.clock

    def run_oof_job(self, job, gpu: int, output: Path) -> None:
        self.calls.append(f"{job.phase}:{job.job_id}:gpu{gpu}")
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("synthetic worker failure")
        output.mkdir(parents=True)
        (output / "done.txt").write_text(job.job_id, encoding="utf-8")
        self.clock += self.step

    def compact_oof_job(self, job, output: Path) -> None:
        assert (output / "done.txt").is_file()
        self.compacted.append(job.job_id)

    def select_recipe(self, output: Path) -> str:
        self.calls.append("select")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"recipe_id":"recipe-a"}', encoding="utf-8")
        return "recipe-a"

    def decide(self, output: Path, recipe_id: str) -> str:
        self.calls.append(f"decision:{recipe_id}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"status":"%s"}' % self.decision_status, encoding="utf-8")
        return self.decision_status

    def run_full_fit(self, role_id: str, seed: int, gpu: int, output: Path) -> None:
        self.calls.append(f"full_fit:{role_id}:s{seed}:gpu{gpu}")
        output.mkdir(parents=True)
        (output / "model.bin").write_bytes(b"model")

    def audit(self, output: Path) -> bool:
        self.calls.append("audit")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"passed":%s}' % str(self.audit_status).lower(), encoding="utf-8")
        return self.audit_status


def test_runner_orders_all_phases_and_never_creates_delivery(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    result = run_campaign(runtime, tmp_path / "campaign", absolute_deadline=100_000)

    assert result.status == "completed"
    assert result.decision == "accepted"
    assert runtime.calls.count("select") == 1
    assert runtime.calls.count("audit") == 1
    assert sum(call.startswith("screening:") for call in runtime.calls) == 14
    assert sum(call.startswith("confirmation:") for call in runtime.calls) == 7
    assert sum(call.startswith("extra_seeds:") for call in runtime.calls) == 42
    assert len(runtime.compacted) == 63
    assert sum(call.startswith("full_fit:") for call in runtime.calls) == 24
    assert max(runtime.calls.index(call) for call in runtime.calls if call.startswith("screening:")) < runtime.calls.index("select")
    assert result.review.name == "failure_regime_e3_review.zip"
    assert result.handoff.name == "failure_regime_e3_handoff.zip"
    assert not list(tmp_path.rglob("*delivery*"))
    verify_bundle(result.review, "review", runtime.bindings)
    verify_bundle(result.handoff, "handoff", runtime.bindings)


def test_rejection_stops_before_full_fit_but_still_emits_evidence(tmp_path: Path) -> None:
    runtime = FakeRuntime(decision_status="rejected")
    result = run_campaign(runtime, tmp_path / "campaign", absolute_deadline=100_000)
    assert result.status == "rejected" and result.decision == "rejected"
    assert not any(call.startswith("full_fit:") for call in runtime.calls)
    assert "audit" not in runtime.calls
    assert result.review.is_file() and result.handoff.is_file()


def test_deadline_pause_and_resume_reuse_completed_jobs(tmp_path: Path) -> None:
    first_runtime = FakeRuntime(clock=0.0, step=10.0)
    first = run_campaign(
        first_runtime,
        tmp_path / "first",
        absolute_deadline=45.0,
        new_job_guard_seconds=20,
    )
    assert first.status == "paused"
    completed_before_pause = sum(call.startswith("screening:") for call in first_runtime.calls)
    assert completed_before_pause == 4

    second_runtime = FakeRuntime()
    second = run_campaign(
        second_runtime,
        tmp_path / "second",
        absolute_deadline=100_000,
        previous_handoff=first.handoff,
    )
    assert second.status == "completed"
    assert sum(call.startswith("screening:") for call in second_runtime.calls) == 14 - completed_before_pause


def test_audit_failure_blocks_completed_status(tmp_path: Path) -> None:
    runtime = FakeRuntime(audit_status=False)
    result = run_campaign(runtime, tmp_path / "campaign", absolute_deadline=100_000)
    assert result.status == "audit_failed"
    assert result.decision == "accepted"


def test_worker_failure_emits_retryable_handoff(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail_once=True)
    failed = run_campaign(runtime, tmp_path / "failed", absolute_deadline=100_000)
    assert failed.status == "failed"
    assert failed.state.failed_jobs
    assert failed.handoff.is_file()

    resumed_runtime = FakeRuntime()
    resumed = run_campaign(
        resumed_runtime,
        tmp_path / "resumed",
        absolute_deadline=100_000,
        previous_handoff=failed.handoff,
    )
    assert resumed.status == "completed"
