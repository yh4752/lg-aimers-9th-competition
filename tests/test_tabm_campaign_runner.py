from __future__ import annotations

from pathlib import Path

from experiments.tabm_campaign.runner import CampaignJobResult, run_one_version


class _Runtime:
    def __init__(self) -> None:
        self.started_versions: list[str] = []
        self.assignments: list[int] = []
        self.shared_model_processes = 0
        self.deadlines: list[float] = []

    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        self.started_versions.append(version)
        self.deadlines.append(job_deadline)
        results = []
        for index, job in enumerate(jobs):
            gpu = index % gpu_count
            self.assignments.append(gpu)
            checkpoint = output_dir / job.candidate_id / "best.bin"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(job.candidate_id.encode())
            results.append(
                CampaignJobResult(
                    candidate_id=job.candidate_id,
                    status="completed",
                    brier=0.20 + index / 10000,
                    best_epoch=2,
                    completed_epochs=3,
                    checkpoint=checkpoint,
                    predictions_path=None,
                    resource_evidence={"gpu": gpu},
                    failure=None,
                )
            )
        return tuple(results)


def test_runner_advances_exactly_one_version(tmp_path: Path) -> None:
    runtime = _Runtime()
    result = run_one_version(tmp_path / "official", tmp_path / "out", runtime=runtime, now=lambda: 1000.0)
    assert result.version == "A"
    assert runtime.started_versions == ["A"]
    assert result.bundles.review.is_file()
    assert result.bundles.resume is not None and result.bundles.resume.is_file()


def test_two_gpus_run_one_independent_job_each(tmp_path: Path) -> None:
    runtime = _Runtime()
    run_one_version(tmp_path / "official", tmp_path / "out", runtime=runtime, now=lambda: 1000.0)
    assert set(runtime.assignments) == {0, 1}
    assert runtime.shared_model_processes == 0


def test_finalization_reserve_is_removed_from_job_deadline(tmp_path: Path) -> None:
    runtime = _Runtime()
    run_one_version(tmp_path / "official", tmp_path / "out", runtime=runtime, now=lambda: 1000.0)
    assert runtime.deadlines == [1000.0 + 7200 - 600]
