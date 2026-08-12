from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from experiments.preprocessing_campaign.budgeted_contracts import BudgetedJob
from experiments.preprocessing_campaign.budgeted_scheduler import (
    ArtifactValidationError,
    BudgetedScheduler,
)


def _job(job_id: str) -> BudgetedJob:
    return BudgetedJob(
        job_id=job_id,
        stage_id=1,
        family="tabm",
        profile_id="p2",
        setting_id="dl_standard",
        preprocessing_profile="dl_standard",
        components=(),
        model=MappingProxyType({"architecture": "tabm"}),
        training=MappingProxyType({"micro_batch_size": 32}),
        train_end_year=2023,
        valid_year=2024,
        seed=42,
        max_seconds=60,
    )


class _FinishedProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def read_available(self) -> tuple[str, ...]:
        return ("fixture worker finished",)

    def terminate(self) -> None:
        self.returncode = -15

    def wait(self, timeout=None) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


class _NeverFinishesProcess(_FinishedProcess):
    def __init__(self) -> None:
        super().__init__(returncode=0)
        self.returncode = None

    def poll(self):
        return self.returncode


class _NeverFinishesFactory:
    def __call__(self, job, temporary_dir, env, deadline):
        del job, env, deadline
        temporary_dir.mkdir(parents=True, exist_ok=True)
        return _NeverFinishesProcess()


class _Factory:
    def __init__(self, *, tamper: bool = False) -> None:
        self.tamper = tamper
        self.calls: list[tuple[str, str, float]] = []

    def __call__(self, job, temporary_dir, env, deadline):
        self.calls.append((job.job_id, env["CUDA_VISIBLE_DEVICES"], deadline))
        temporary_dir.mkdir(parents=True, exist_ok=True)
        metrics = temporary_dir / "metrics.json"
        predictions = temporary_dir / "predictions.csv"
        metrics.write_text('{"brier": 0.24}\n', encoding="utf-8")
        predictions.write_text("row_id,probability\na,0.5\n", encoding="utf-8")
        artifacts = []
        for path in (metrics, predictions):
            digest = sha256(path.read_bytes()).hexdigest()
            artifacts.append({"path": path.name, "sha256": digest})
        if self.tamper:
            artifacts[0]["sha256"] = "0" * 64
        (temporary_dir / "worker_result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": job.job_id,
                    "state": "completed",
                    "artifacts": artifacts,
                }
            ),
            encoding="utf-8",
        )
        return _FinishedProcess()


def _artifact_entry(root: Path, job: BudgetedJob) -> dict[str, object]:
    job_root = root / "jobs" / job.job_id
    job_root.mkdir(parents=True)
    metrics = job_root / "metrics.json"
    predictions = job_root / "predictions.csv"
    metrics.write_text('{"brier": 0.24}\n', encoding="utf-8")
    predictions.write_text("row_id,probability\na,0.5\n", encoding="utf-8")
    return {
        "state": "completed",
        "job": {
            "job_id": job.job_id,
            "stage_id": job.stage_id,
            "family": job.family,
            "profile_id": job.profile_id,
            "setting_id": job.setting_id,
            "preprocessing_profile": job.preprocessing_profile,
            "components": list(job.components),
            "model": dict(job.model),
            "training": dict(job.training),
            "train_end_year": job.train_end_year,
            "valid_year": job.valid_year,
            "seed": job.seed,
            "max_seconds": job.max_seconds,
            "sample_mode": job.sample_mode,
        },
        "artifacts": [
            {
                "path": f"jobs/{job.job_id}/{path.name}",
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in (metrics, predictions)
        ],
    }


def test_scheduler_pins_one_worker_per_gpu_and_parent_merges_only_valid_results(
    tmp_path: Path,
) -> None:
    factory = _Factory()
    scheduler = BudgetedScheduler(
        clock=lambda: 1_000,
        process_factory=factory,
        gpu_probe=lambda: 2,
        poll_seconds=0,
    )

    summary = scheduler.run([_job("a"), _job("b")], tmp_path, deadline=10_000)

    assert [(job_id, gpu) for job_id, gpu, _ in factory.calls] == [
        ("a", "0"),
        ("b", "1"),
    ]
    manifest = json.loads(
        (tmp_path / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["jobs"]) == {"a", "b"}
    assert {entry["state"] for entry in manifest["jobs"].values()} == {"completed"}
    assert summary.completed == ("a", "b")
    assert not (tmp_path / "workers" / "a").exists()


def test_scheduler_does_not_start_job_inside_nine_hundred_second_guard(
    tmp_path: Path,
) -> None:
    factory = _Factory()
    scheduler = BudgetedScheduler(
        clock=lambda: 9_200,
        process_factory=factory,
        gpu_probe=lambda: 2,
        poll_seconds=0,
    )

    summary = scheduler.run([_job("a")], tmp_path, deadline=10_000)

    assert summary.pending == ("a",)
    assert factory.calls == []


def test_worker_cannot_publish_tampered_artifact(tmp_path: Path) -> None:
    scheduler = BudgetedScheduler(
        clock=lambda: 1_000,
        process_factory=_Factory(tamper=True),
        gpu_probe=lambda: 2,
        poll_seconds=0,
    )

    with pytest.raises(ArtifactValidationError, match="sha256"):
        scheduler.run([_job("a")], tmp_path, deadline=10_000)


def test_scheduler_rejects_non_t4x2_shape_before_starting(tmp_path: Path) -> None:
    factory = _Factory()
    scheduler = BudgetedScheduler(
        clock=lambda: 1_000,
        process_factory=factory,
        gpu_probe=lambda: 1,
        poll_seconds=0,
    )

    with pytest.raises(RuntimeError, match="exactly two CUDA devices"):
        scheduler.run([_job("a")], tmp_path, deadline=10_000)
    assert factory.calls == []


def test_scheduler_revalidates_completed_artifacts_before_skipping(tmp_path: Path) -> None:
    job = _job("a")
    entry = _artifact_entry(tmp_path, job)
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps({"schema_version": 1, "jobs": {"a": entry}}),
        encoding="utf-8",
    )
    (tmp_path / "jobs" / "a" / "metrics.json").write_text(
        '{"brier": 0.99}\n', encoding="utf-8"
    )
    scheduler = BudgetedScheduler(
        clock=lambda: 1_000,
        process_factory=_Factory(),
        gpu_probe=lambda: 2,
        poll_seconds=0,
    )

    with pytest.raises(ArtifactValidationError, match="sha256"):
        scheduler.run([job], tmp_path, deadline=10_000)


def test_scheduler_rejects_changed_campaign_identity(tmp_path: Path) -> None:
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": {"config_sha256": "a" * 64},
                "jobs": {},
            }
        ),
        encoding="utf-8",
    )
    scheduler = BudgetedScheduler(
        clock=lambda: 1_000,
        process_factory=_Factory(),
        gpu_probe=lambda: 2,
        campaign_identity={"config_sha256": "b" * 64},
        poll_seconds=0,
    )

    with pytest.raises(ArtifactValidationError, match="identity"):
        scheduler.run([_job("a")], tmp_path, deadline=10_000)


def test_late_stage_catboost_timeout_is_terminal_inconclusive(tmp_path: Path) -> None:
    clock_values = iter([1_000, 1_000, 1_000, 2_000, 2_000, 2_000])
    job = replace(_job("cat"), stage_id=3, family="catboost", max_seconds=60)
    scheduler = BudgetedScheduler(
        clock=lambda: next(clock_values, 2_000),
        process_factory=_NeverFinishesFactory(),
        gpu_probe=lambda: 2,
        poll_seconds=0,
        worker_shutdown_grace_seconds=0,
    )

    summary = scheduler.run([job], tmp_path, deadline=10_000)

    manifest = json.loads(
        (tmp_path / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert summary.completed == ("cat",)
    assert manifest["jobs"]["cat"]["state"] == "inconclusive"
    assert manifest["jobs"]["cat"]["reason"] == "catboost_time_budget_reached"
