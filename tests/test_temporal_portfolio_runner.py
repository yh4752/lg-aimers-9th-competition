from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from experiments.preprocessing_campaign.budgeted_scheduler import SchedulerSummary
from experiments.temporal_portfolio.inputs import VerifiedOfficialData
from experiments.temporal_portfolio.planner import PlannerError, PriorReview, plan_stage
from experiments.temporal_portfolio.runner import (
    CompletedJob,
    PortfolioRunnerError,
    run_stage,
)
import pytest


def _verified(tmp_path: Path) -> VerifiedOfficialData:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    hashes = {}
    sizes = {}
    for name in ("train.csv", "test.csv", "trackman_history.csv", "sample_submission.csv"):
        path = tmp_path / name
        path.write_text(name)
        paths.append(path)
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        sizes[name] = path.stat().st_size
    return VerifiedOfficialData(
        tmp_path,
        *paths,
        MappingProxyType(hashes),
        10,
        5,
        MappingProxyType(sizes),
        MappingProxyType({}),
    )


class _Scheduler:
    def __init__(self, now: float, summary: SchedulerSummary | None = None) -> None:
        self.now = now
        self.summary = summary
        self.started_jobs: list[str] = []
        self.deadline: float | None = None

    def clock(self) -> float:
        return self.now

    def run(self, jobs, output_root, *, deadline):
        self.deadline = deadline
        self.started_jobs.extend(job.job_id for job in jobs)
        return self.summary or SchedulerSummary(tuple(self.started_jobs), (), ())


def test_stage_planner_never_requeues_completed_identity(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    from experiments.temporal_portfolio.runner import _input_manifest_sha
    review = PriorReview(_input_manifest_sha(verified), {})
    first = plan_stage("T1", prior_review=review, completed={})
    completed = {first.jobs[0].identity.sha256: "done/job"}
    second = plan_stage("T1", prior_review=review, completed=completed)
    assert len(second.jobs) == len(first.jobs) - 1
    assert not {job.identity.sha256 for job in second.jobs} & set(completed)


def test_t1_job_budget_uses_both_gpu_device_hours(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    from experiments.temporal_portfolio.runner import _input_manifest_sha

    plan = plan_stage(
        "T1",
        prior_review=PriorReview(_input_manifest_sha(verified), {}),
        completed={},
    )
    assert {job.max_seconds for job in plan.jobs} == {2_625}


def test_runner_stops_new_jobs_and_reserves_handoff_time(tmp_path: Path) -> None:
    verified = _verified(tmp_path / "data")
    scheduler = _Scheduler(9_200)
    result = run_stage(
        stage="T1",
        verified=verified,
        output_root=tmp_path / "output",
        deadline=10_000,
        scheduler=scheduler,
    )
    assert result.status == "budget_inconclusive"
    assert result.handoff.path.is_file()
    assert scheduler.started_jobs == []


def test_runner_dispatches_when_budget_is_available(tmp_path: Path) -> None:
    verified = _verified(tmp_path / "data")
    scheduler = _Scheduler(1_000)
    result = run_stage(
        stage="T1",
        verified=verified,
        output_root=tmp_path / "output",
        deadline=10_000,
        scheduler=scheduler,
    )
    assert result.status == "completed"
    assert scheduler.started_jobs
    assert result.handoff.path.is_file()


def test_runner_requires_platform_scheduler(tmp_path: Path) -> None:
    verified = _verified(tmp_path / "data")
    with pytest.raises(PortfolioRunnerError, match="scheduler"):
        run_stage(
            stage="T1",
            verified=verified,
            output_root=tmp_path / "output",
            deadline=10_000,
        )


def test_runner_rejects_completed_catalog_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path / "data")
    from experiments.temporal_portfolio.runner import _input_manifest_sha

    first = plan_stage(
        "T1", prior_review=PriorReview(_input_manifest_sha(verified), {}), completed={}
    )
    claimed = first.jobs[0].identity.sha256
    other = "f" * 64 if claimed != "f" * 64 else "e" * 64
    monkeypatch.setattr(
        "experiments.temporal_portfolio.runner.verify_completed_job",
        lambda path: CompletedJob(Path(path), "completed", other),
    )
    with pytest.raises(PortfolioRunnerError, match="identity"):
        run_stage(
            stage="T1",
            verified=verified,
            output_root=tmp_path / "output",
            deadline=10_000,
            scheduler=_Scheduler(1_000),
            completed={claimed: str(tmp_path / "job")},
        )


def test_later_stage_handoff_advances_prior_lineage(tmp_path: Path) -> None:
    verified = _verified(tmp_path / "data")
    from experiments.temporal_portfolio.runner import _input_manifest_sha

    input_sha = _input_manifest_sha(verified)
    t1_job = plan_stage(
        "T1", prior_review=PriorReview(input_sha, {}), completed={}
    ).jobs[0]
    review = PriorReview(
        input_sha,
        {"T2A": (t1_job,)},
        parent_manifest_sha256="a" * 64,
        sequence=1,
    )
    result = run_stage(
        stage="T2A",
        verified=verified,
        output_root=tmp_path / "output",
        deadline=10_000,
        scheduler=_Scheduler(1_000),
        prior_review=review,
    )
    from experiments.temporal_portfolio.artifacts import verify_handoff

    handoff = verify_handoff(result.handoff.path)
    assert handoff.sequence == 2
    assert handoff.lineage.parent_manifest_sha256 == "a" * 64


def test_t5a_accepts_only_confirmation_seed_jobs(tmp_path: Path) -> None:
    verified = _verified(tmp_path / "data")
    from experiments.temporal_portfolio.runner import _input_manifest_sha

    input_sha = _input_manifest_sha(verified)
    screen_job = plan_stage(
        "T1", prior_review=PriorReview(input_sha, {}), completed={}
    ).jobs[0]
    review = PriorReview(
        input_sha,
        {"T5A": (screen_job,)},
        parent_manifest_sha256="a" * 64,
        sequence=4,
    )
    with pytest.raises(PlannerError, match="confirmation"):
        plan_stage("T5A", prior_review=review, completed={})


def test_t3bt4_uses_sealed_physical_deadline(tmp_path: Path) -> None:
    verified = _verified(tmp_path / "data")
    from experiments.temporal_portfolio.runner import _input_manifest_sha

    input_sha = _input_manifest_sha(verified)
    job = plan_stage(
        "T1", prior_review=PriorReview(input_sha, {}), completed={}
    ).jobs[0]
    review = PriorReview(
        input_sha,
        {"T3BT4": (job,)},
        parent_manifest_sha256="a" * 64,
        sequence=3,
    )
    scheduler = _Scheduler(1_000)
    run_stage(
        stage="T3BT4",
        verified=verified,
        output_root=tmp_path / "output",
        deadline=100_000,
        scheduler=scheduler,
        prior_review=review,
    )
    assert scheduler.deadline == 21_700
