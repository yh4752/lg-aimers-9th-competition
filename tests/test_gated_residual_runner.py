from __future__ import annotations

import json
from pathlib import Path

from experiments.gated_residual_final.artifacts import ArtifactBindings
from experiments.gated_residual_final.runner import (
    AuditOutcome,
    FullFitOutcome,
    SelectionOutcome,
    run_campaign,
)
from experiments.gated_residual_final.selection import CandidateDecision


def _bindings() -> ArtifactBindings:
    return ArtifactBindings("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)


class FakeRuntime:
    bindings = _bindings()

    def __init__(self, root: Path, *, accepted: bool, complete: bool = True):
        self.root = root
        self.accepted = accepted
        self.complete = complete
        self.full_fit_calls = 0
        self.received_completed: tuple[str, ...] = ()

    def now(self) -> float:
        return 100.0

    def select(self, output: Path) -> SelectionOutcome:
        output.mkdir(parents=True, exist_ok=True)
        evidence = output / "evidence.json"
        evidence.write_text(json.dumps({"weighted_gain": 0.001}))
        decision = CandidateDecision("G1_x", "accepted" if self.accepted else "rejected", () if self.accepted else ("latest_gain",))
        return SelectionOutcome(decision, evidence)

    def full_fit(self, decision, output, *, completed_jobs, restored_root, on_job_complete, absolute_deadline):
        self.full_fit_calls += 1
        self.received_completed = completed_jobs
        payloads = {}
        completed = list(completed_jobs)
        for job in ("D0_s42", "D0_s2026"):
            path = output / job / "model.cbm"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(job.encode())
            payloads[f"jobs/{job}/model.cbm"] = path
            if job not in completed:
                completed.append(job)
                on_job_complete(job)
            if not self.complete:
                return FullFitOutcome(False, tuple(completed), payloads)
        return FullFitOutcome(True, tuple(completed), payloads)

    def audit(self, full_fit: FullFitOutcome, output: Path) -> AuditOutcome:
        report = output / "audit.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"status": "passed", "row_independence": "passed"}))
        return AuditOutcome(True, report)


def test_rejected_campaign_stops_before_full_fit(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, accepted=False)

    result = run_campaign(runtime=runtime, output_dir=tmp_path / "campaign")

    assert result.status == "completed_no_candidate"
    assert result.delivery is None
    assert runtime.full_fit_calls == 0
    assert result.review.is_file() and result.handoff.is_file()


def test_interrupted_run_reuses_completed_job(tmp_path: Path) -> None:
    first_runtime = FakeRuntime(tmp_path, accepted=True, complete=False)
    first = run_campaign(runtime=first_runtime, output_dir=tmp_path / "first")
    assert first.status == "paused"

    second_runtime = FakeRuntime(tmp_path, accepted=True, complete=True)
    second = run_campaign(
        runtime=second_runtime,
        output_dir=tmp_path / "second",
        previous_handoff=first.handoff,
    )

    assert "D0_s42" in second_runtime.received_completed
    assert second.status == "completed"
    assert second.delivery is not None


def test_deadline_before_full_fit_emits_handoff_without_training(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, accepted=True)

    result = run_campaign(
        runtime=runtime, output_dir=tmp_path / "campaign", absolute_deadline=101.0,
    )

    assert result.status == "paused"
    assert result.handoff.name == "gated_residual_final_handoff.zip"
    assert runtime.full_fit_calls == 0
