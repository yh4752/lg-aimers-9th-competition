from __future__ import annotations

from pathlib import Path

from experiments.catboost_deployment.colab import EmergencyCadence, _SubprocessRuntime


def test_emergency_cadence_requires_time_and_changed_snapshot() -> None:
    cadence = EmergencyCadence(interval_seconds=1200.0, started_at=100.0)

    assert cadence.should_publish(now=100.0, snapshot_sha256="a" * 64) is False
    assert cadence.should_publish(now=1300.0, snapshot_sha256="a" * 64) is True
    cadence.mark_published(now=1300.0, snapshot_sha256="a" * 64)
    assert cadence.should_publish(now=2500.0, snapshot_sha256="a" * 64) is False
    assert cadence.should_publish(now=2499.9, snapshot_sha256="b" * 64) is False
    assert cadence.should_publish(now=2500.0, snapshot_sha256="b" * 64) is True


def test_deadline_kill_without_worker_result_returns_inconclusive(tmp_path: Path) -> None:
    runtime = object.__new__(_SubprocessRuntime)
    job = tmp_path / "job"
    job.mkdir()
    (job / "experiment.cbsnapshot").write_bytes(b"snapshot")

    result = runtime._result(
        job, "align_2022_2023", kind="alignment", killed_for_deadline=True
    )

    assert result.status == "budget_inconclusive"
    assert result.snapshot_path == job / "experiment.cbsnapshot"
