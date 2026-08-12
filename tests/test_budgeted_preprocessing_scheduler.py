from __future__ import annotations

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
