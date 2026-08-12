from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from experiments.independent_dl.preprocessing_campaign import (
    CampaignInterrupted,
    PreprocessingJobResult,
    estimate_remaining_resources,
    run_preprocessing_campaign,
)


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
