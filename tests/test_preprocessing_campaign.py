from __future__ import annotations

from pathlib import Path
import json
from types import MappingProxyType, SimpleNamespace

import pytest

from experiments.independent_dl.preprocessing_campaign import (
    CampaignInterrupted,
    PreprocessingJobResult,
    estimate_remaining_resources,
    run_preprocessing_campaign,
)
from experiments.independent_dl.training import TrainingTimeBudgetReached


def _tiny_campaign() -> SimpleNamespace:
    jobs = tuple(
        SimpleNamespace(job_id=f"job-{index}", family="tabm", wave="a")
        for index in (1, 2)
    )
    return SimpleNamespace(
        campaign_id="tiny-preprocessing", protocol="test", wave_a_jobs=jobs
    )


class FakeRuntime:
    def __init__(self, *, interrupt_on: str | None = None, fail_on: str | None = None):
        self.interrupt_on = interrupt_on
        self.fail_on = fail_on
        self.started: list[str] = []

    def run_job(self, job: object, output_dir: Path) -> PreprocessingJobResult:
        self.started.append(job.job_id)
        if job.job_id == self.interrupt_on:
            raise CampaignInterrupted("interruption")
        if job.job_id == self.fail_on:
            raise ValueError("candidate-local failure")
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = output_dir / "predictions.csv"
        metrics = output_dir / "metrics.json"
        predictions.write_text("row_id,probability\na,0.5\n", encoding="utf-8")
        metrics.write_text(json.dumps({"brier": 0.25}), encoding="utf-8")
        return PreprocessingJobResult(
            metrics,
            predictions,
            0.25,
            120.0,
            1.0,
            2.0,
        )


class TimeBudgetRuntime:
    def run_job(self, job: object, output_dir: Path) -> PreprocessingJobResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoint.pt").write_bytes(b"checkpoint")
        (output_dir / "checkpoint_meta.json").write_text(
            json.dumps(
                {
                    "candidate_id": job.job_id,
                    "epoch": 2,
                    "checkpoint": "checkpoint.pt",
                }
            ),
            encoding="utf-8",
        )
        raise TrainingTimeBudgetReached("session time budget reached")


def test_campaign_resumes_at_job_granularity_without_repeating_completed(
    tmp_path: Path,
) -> None:
    campaign = _tiny_campaign()
    first = FakeRuntime(interrupt_on="job-2")
    with pytest.raises(CampaignInterrupted, match="interruption"):
        run_preprocessing_campaign(campaign, tmp_path, first)
    second = FakeRuntime()

    summary = run_preprocessing_campaign(campaign, tmp_path, second)

    assert "job-1" not in second.started
    assert summary.completed == ("job-1", "job-2")


def test_one_failed_job_does_not_block_independent_jobs(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail_on="job-1")

    summary = run_preprocessing_campaign(_tiny_campaign(), tmp_path, runtime)

    assert summary.failed == ("job-1",)
    assert "job-2" in summary.completed


def test_campaign_serializes_dataclass_with_mappingproxy_contract(tmp_path: Path) -> None:
    from experiments.independent_dl.preprocessing_contracts import (
        PreprocessingJob,
        PreprocessingSetting,
    )

    job = PreprocessingJob(
        job_id="real-contract-job",
        wave="a",
        anchor_id="tabm_p3",
        family="tabm",
        profile_id="p3",
        epochs=240,
        model=MappingProxyType({"width": 768}),
        training=MappingProxyType({"learning_rate": 0.001}),
        feature_view="raw_typed",
        setting=PreprocessingSetting("dl_standard", "dl_standard", ()),
        train_end_year=2019,
        valid_year=2020,
        seed=42,
    )
    campaign = SimpleNamespace(
        campaign_id="real-contract", protocol="test", wave_a_jobs=(job,)
    )

    summary = run_preprocessing_campaign(campaign, tmp_path, FakeRuntime())

    assert summary.completed == ("real-contract-job",)
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    assert manifest["jobs"]["real-contract-job"]["job"]["model"] == {"width": 768}


def test_max_jobs_limits_new_attempts_without_shrinking_campaign(tmp_path: Path) -> None:
    campaign = _tiny_campaign()
    first = FakeRuntime()

    summary = run_preprocessing_campaign(campaign, tmp_path, first, max_jobs=1)

    assert first.started == ["job-1"]
    assert summary.completed == ("job-1",)
    assert summary.pending == ("job-2",)

    second = FakeRuntime()
    resumed = run_preprocessing_campaign(campaign, tmp_path, second, max_jobs=1)
    assert second.started == ["job-2"]
    assert resumed.completed == ("job-1", "job-2")


def test_time_budget_keeps_current_job_pending_and_checkpointed(tmp_path: Path) -> None:
    with pytest.raises(CampaignInterrupted, match="session time budget"):
        run_preprocessing_campaign(_tiny_campaign(), tmp_path, TimeBudgetRuntime())

    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    assert manifest["jobs"]["job-1"]["state"] == "pending"
    assert (tmp_path / "jobs/job-1/checkpoint_meta.json").is_file()


def test_completed_artifact_hash_change_forces_only_that_job_to_rerun(
    tmp_path: Path,
) -> None:
    campaign = _tiny_campaign()
    run_preprocessing_campaign(campaign, tmp_path, FakeRuntime())
    (tmp_path / "jobs/job-1/predictions.csv").write_text("changed", encoding="utf-8")
    runtime = FakeRuntime()

    run_preprocessing_campaign(campaign, tmp_path, runtime)

    assert runtime.started == ["job-1"]


def test_resource_summary_uses_completed_same_family_median() -> None:
    resource_rows = [
        {"family": "tabm", "elapsed_seconds": 60.0},
        {"family": "tabm", "elapsed_seconds": 120.0},
        {"family": "tabm", "elapsed_seconds": 180.0},
    ]

    estimate = estimate_remaining_resources(
        resource_rows, family="tabm", pending_same_family=10
    )

    assert estimate["estimated_remaining_seconds"] == pytest.approx(10 * 120.0)
