from __future__ import annotations

from pathlib import Path

import pytest

from experiments.tabm_campaign.version_d import RecoverySelection


def test_frozen_recovery_skips_training(tmp_path: Path, monkeypatch) -> None:
    import experiments.tabm_campaign.colab_version_d as runtime

    calls: list[str] = []
    frozen = tmp_path / "frozen.zip"
    frozen.write_bytes(b"frozen")
    review = tmp_path / "review.zip"
    review.write_bytes(b"review")
    delivery = tmp_path / "delivery.zip"
    delivery.write_bytes(b"delivery")
    monkeypatch.setattr(runtime, "verify_inputs", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(
        runtime,
        "restore_recovery",
        lambda *args, **kwargs: RecoverySelection("frozen", frozen, 3),
    )
    monkeypatch.setattr(
        runtime,
        "fit_final",
        lambda *args, **kwargs: calls.append("fit"),
    )
    monkeypatch.setattr(runtime, "_frozen_fit_report", lambda *args: {"status": "completed"})
    monkeypatch.setattr(
        runtime,
        "review_final",
        lambda *args, **kwargs: calls.append("review") or review,
    )
    monkeypatch.setattr(
        runtime,
        "publish_delivery",
        lambda *args, **kwargs: delivery,
    )

    result = runtime.run_version_d(
        data_archive=tmp_path / "data.zip",
        stage_c_delivery=tmp_path / "stage_c.zip",
        work_root=tmp_path / "work",
        recovery_archive=frozen,
        absolute_deadline=4_000_000_000.0,
        runtime_sha256="1" * 64,
        runtime_versions={"python": "3.12"},
        log_path=tmp_path / "run.log",
        on_download=lambda path, kind: calls.append(kind),
    )
    assert result == delivery
    assert "fit" not in calls
    assert calls == ["frozen", "review", "delivery"]


def test_review_failure_never_publishes_delivery(tmp_path: Path, monkeypatch) -> None:
    import experiments.tabm_campaign.colab_version_d as runtime

    published: list[bool] = []
    frozen = tmp_path / "frozen.zip"
    frozen.write_bytes(b"frozen")
    monkeypatch.setattr(runtime, "verify_inputs", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(
        runtime,
        "restore_recovery",
        lambda *args, **kwargs: RecoverySelection("frozen", frozen, 3),
    )
    monkeypatch.setattr(
        runtime,
        "review_final",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("review failed")),
    )
    monkeypatch.setattr(runtime, "_frozen_fit_report", lambda *args: {"status": "completed"})
    monkeypatch.setattr(
        runtime,
        "publish_delivery",
        lambda *args, **kwargs: published.append(True),
    )
    with pytest.raises(RuntimeError, match="review failed"):
        runtime.run_version_d(
            data_archive=tmp_path / "data.zip",
            stage_c_delivery=tmp_path / "stage_c.zip",
            work_root=tmp_path / "work",
            recovery_archive=frozen,
            absolute_deadline=4_000_000_000.0,
            runtime_sha256="1" * 64,
            runtime_versions={"python": "3.12"},
            log_path=tmp_path / "run.log",
            on_download=lambda path, kind: None,
        )
    assert published == []


def test_expired_deadline_stops_before_input_verification(tmp_path: Path, monkeypatch) -> None:
    import experiments.tabm_campaign.colab_version_d as runtime

    verified: list[bool] = []
    monkeypatch.setattr(
        runtime,
        "verify_inputs",
        lambda *args, **kwargs: verified.append(True),
    )
    with pytest.raises(runtime.ColabVersionDError, match="deadline"):
        runtime.run_version_d(
            data_archive=tmp_path / "data.zip",
            stage_c_delivery=tmp_path / "stage_c.zip",
            work_root=tmp_path / "work",
            recovery_archive=None,
            absolute_deadline=0.0,
            runtime_sha256="1" * 64,
            runtime_versions={"python": "3.12"},
            log_path=tmp_path / "run.log",
            on_download=lambda path, kind: None,
        )
    assert verified == []
