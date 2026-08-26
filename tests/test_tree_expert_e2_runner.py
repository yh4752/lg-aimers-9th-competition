from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from experiments.tree_expert.e2_runner import (
    CampaignState,
    PhaseOutcome,
    load_campaign_state,
    run_e2_campaign,
)


def _bindings() -> dict[str, str]:
    return {
        "contract_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "input_sha256": "3" * 64,
        "train_sha256": "4" * 64,
        "history_sha256": "5" * 64,
    }


@dataclass
class FakeRuntime:
    statuses: dict[str, str]

    def __post_init__(self) -> None:
        self.phase_order: list[str] = []
        self.started_jobs: list[str] = []

    def run_phase(
        self,
        phase: str,
        state: CampaignState,
        *,
        gpu_ids: tuple[int, int],
        wall_deadline: float,
    ) -> PhaseOutcome:
        del wall_deadline
        assert gpu_ids == (0, 1)
        self.phase_order.append(phase)
        job = f"{phase.lower()}-job"
        self.started_jobs.append(job)
        status = self.statuses.get(phase, "passed")
        if status == "failed_sibling":
            return PhaseOutcome(
                status="failed",
                completed=("e2__c1_anchor_residual__tr2021__va2022__s3407",),
                failed=("e2__c2_trackman_residual__tr2021__va2022__s3407",),
                decisions={"phase": phase},
                artifact_paths={},
            )
        return PhaseOutcome(
            status=status,
            completed=(job,),
            failed=(),
            decisions={"phase": phase, "status": status},
            artifact_paths={},
        )

    def publish(self, state: CampaignState, output_dir: Path) -> object:
        output_dir.mkdir(parents=True, exist_ok=True)
        resume = output_dir / "resume.zip"
        resume.write_bytes(b"resume")
        delivery = None
        if state.status == "accepted":
            delivery = output_dir / "delivery.zip"
            delivery.write_bytes(b"delivery")
        return SimpleNamespace(resume=resume, delivery=delivery)


def _run(tmp_path: Path, runtime: FakeRuntime, *, seconds: float = 10_000):
    return run_e2_campaign(
        runtime=runtime,
        output_dir=tmp_path,
        bindings=_bindings(),
        wall_deadline=seconds,
        clock=lambda: 0.0,
    )


def test_pass_path_runs_exact_phase_order(tmp_path: Path) -> None:
    runtime = FakeRuntime({})
    result = _run(tmp_path, runtime)
    assert runtime.phase_order == [
        "B0", "B1", "B2", "B3", "ACCEPTANCE", "FULL_FIT", "AUDIT"
    ]
    assert result.status == "accepted"
    assert result.bundles.delivery is not None


def test_structure_rejection_stops_later_phases(tmp_path: Path) -> None:
    runtime = FakeRuntime({"B1": "rejected"})
    result = _run(tmp_path, runtime)
    assert result.status == "rejected_structure"
    assert runtime.phase_order == ["B0", "B1"]


def test_seed_rejection_stops_before_blend(tmp_path: Path) -> None:
    runtime = FakeRuntime({"B2": "rejected"})
    result = _run(tmp_path, runtime)
    assert result.status == "rejected_seed_instability"
    assert "B3" not in runtime.phase_order


def test_blend_rejection_falls_back_to_catboost(tmp_path: Path) -> None:
    runtime = FakeRuntime({"B3": "rejected"})
    result = _run(tmp_path, runtime)
    assert result.status == "accepted"
    assert runtime.phase_order[-1] == "AUDIT"
    assert result.state.decisions["B3"]["status"] == "rejected"


def test_baseline_failure_is_fail_closed(tmp_path: Path) -> None:
    runtime = FakeRuntime({"B0": "failed"})
    result = _run(tmp_path, runtime)
    assert result.status == "failed"
    assert runtime.phase_order == ["B0"]


def test_resume_starts_at_persisted_phase_and_keeps_completed(tmp_path: Path) -> None:
    runtime = FakeRuntime({"B1": "rejected"})
    state = CampaignState.initial(_bindings())
    state = state.with_update(phase="B1", completed=("exact-completed-job",))
    state_path = tmp_path / "stage_state.json"
    state.write(state_path)

    result = run_e2_campaign(
        runtime=runtime,
        output_dir=tmp_path,
        bindings=_bindings(),
        wall_deadline=10_000,
        state_path=state_path,
        clock=lambda: 0.0,
    )
    assert runtime.phase_order == ["B1"]
    assert "exact-completed-job" in result.state.completed


def test_deadline_publishes_resume_without_new_job(tmp_path: Path) -> None:
    runtime = FakeRuntime({})
    result = _run(tmp_path, runtime, seconds=599)
    assert result.status == "budget_inconclusive"
    assert result.bundles.resume.is_file()
    assert runtime.started_jobs == []


def test_independent_failure_preserves_sibling_evidence(tmp_path: Path) -> None:
    runtime = FakeRuntime({"B1": "failed_sibling"})
    result = _run(tmp_path, runtime)
    assert result.status == "failed"
    assert "e2__c1_anchor_residual__tr2021__va2022__s3407" in result.state.completed
    assert "e2__c2_trackman_residual__tr2021__va2022__s3407" in result.state.failed


def test_saved_state_round_trip_is_exact(tmp_path: Path) -> None:
    state = CampaignState.initial(_bindings()).with_update(
        phase="B2", completed=("a",), decisions={"B1": {"status": "passed"}}
    )
    path = tmp_path / "state.json"
    state.write(path)
    assert load_campaign_state(path, _bindings()) == state
