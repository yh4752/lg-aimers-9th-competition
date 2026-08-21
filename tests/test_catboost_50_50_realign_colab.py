from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.catboost_50_50_realign.colab import (
    DownloadEvent,
    EmergencySnapshotMonitor,
    RealignColabError,
    classify_and_verify_uploads,
    collect_terminal_downloads,
    run_supervised_campaign,
)
from experiments.catboost_50_50_realign.runner import RealignRun
from experiments.catboost_50_50_realign.inputs import VerifiedRealignInput
from experiments.catboost_50_50_realign.state import CampaignState, serialize_state
from decimal import Decimal


def _result(tmp_path: Path, status: str) -> RealignRun:
    review = tmp_path / "review.zip"
    resume = tmp_path / "resume.zip"
    delivery = tmp_path / "delivery.zip"
    for path in (review, resume, delivery):
        path.write_bytes(path.name.encode())
    return RealignRun(
        status=status,
        selected_tree_count=16 if status == "completed" else None,
        review_bundle=review,
        resume_bundle=resume,
        delivery_bundle=delivery if status == "completed" else None,
    )


def test_completed_download_phases_are_exact(tmp_path: Path) -> None:
    events = [DownloadEvent("f1_complete", tmp_path / "f1.zip")]
    events.extend(collect_terminal_downloads(_result(tmp_path, "completed")))
    assert [event.phase for event in events] == [
        "f1_complete",
        "completed_review",
        "completed_resume",
        "completed_delivery",
    ]


def test_blocked_terminal_has_no_delivery(tmp_path: Path) -> None:
    events = collect_terminal_downloads(_result(tmp_path, "deployment_blocked"))
    assert [event.phase for event in events] == ["blocked_review", "blocked_resume"]


def test_supervisor_downloads_f1_only_after_phase_callback(tmp_path: Path) -> None:
    downloads: list[DownloadEvent] = []

    def campaign(*, on_phase_resume, **kwargs):
        f1 = tmp_path / "f1.zip"
        f1.write_bytes(b"f1")
        on_phase_resume(f1, "f1_complete")
        return _result(tmp_path, "completed")

    result = run_supervised_campaign(
        campaign=campaign,
        campaign_kwargs={},
        download=downloads.append,
        latest_verified_resume=lambda: None,
    )
    assert result.status == "completed"
    assert [event.phase for event in downloads] == [
        "f1_complete",
        "completed_review",
        "completed_resume",
        "completed_delivery",
    ]


def test_error_requests_only_latest_verified_resume(tmp_path: Path) -> None:
    latest = tmp_path / "latest.zip"
    latest.write_bytes(b"resume")
    downloads: list[DownloadEvent] = []

    def campaign(**kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_supervised_campaign(
            campaign=campaign,
            campaign_kwargs={},
            download=downloads.append,
            latest_verified_resume=lambda: latest,
        )
    assert downloads == [DownloadEvent("emergency_resume", latest)]


def test_callback_failure_does_not_replace_previous_resume(tmp_path: Path) -> None:
    prior = tmp_path / "prior.zip"
    prior.write_bytes(b"prior")
    calls = 0

    def download(event):
        nonlocal calls
        calls += 1
        raise RuntimeError("browser rejected download")

    def campaign(*, on_phase_resume, **kwargs):
        current = tmp_path / "current.zip"
        current.write_bytes(b"current")
        on_phase_resume(current, "f1_complete")
        return _result(tmp_path, "completed")

    with pytest.raises(RealignColabError, match="download callback failed"):
        run_supervised_campaign(
            campaign=campaign,
            campaign_kwargs={},
            download=download,
            latest_verified_resume=lambda: prior,
        )
    assert prior.read_bytes() == b"prior"
    assert calls == 2


def test_upload_classifier_accepts_input_with_optional_resume_in_any_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.colab as module

    input_zip = tmp_path / "input.zip"
    resume_zip = tmp_path / "resume.zip"
    for path, kind in (
        (input_zip, "catboost_50_50_realign_input_v1"),
        (resume_zip, "realign_resume_v1"),
    ):
        with ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", f'{{"artifact_kind":"{kind}"}}')
    verified = object()
    monkeypatch.setattr(module, "verify_and_extract_input", lambda *args: verified)
    monkeypatch.setattr(module, "_bindings", lambda value: object())
    checked: list[Path] = []
    monkeypatch.setattr(
        module,
        "verify_resume_bundle",
        lambda path, bindings: checked.append(Path(path)),
    )

    actual, resume = classify_and_verify_uploads(
        [resume_zip, input_zip], run_root=tmp_path / "run"
    )
    assert actual is verified
    assert resume == resume_zip
    assert checked == [resume_zip]


def test_upload_classifier_rejects_wrong_count(tmp_path: Path) -> None:
    with pytest.raises(RealignColabError, match="one or two"):
        classify_and_verify_uploads([], run_root=tmp_path / "run")


def test_active_snapshot_is_stored_without_browser_download(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    data = input_root / "data"
    data.mkdir(parents=True)
    (data / "train.csv").write_text("row_id\nr1\n")
    (data / "trackman_history.csv").write_text("x\n1\n")
    verified = VerifiedRealignInput(
        root=input_root,
        data_dir=data,
        tabm_predictions={},
        catboost_models={},
        catboost_states={},
        catboost_predictions={},
        audit_decision=input_root / "audit.json",
        manifest_sha256="a" * 64,
        fold_keys=("2021->2022", "2022->2023", "2023->2024"),
        audit_weight=Decimal("0.50"),
    )
    campaign = tmp_path / "campaign"
    monitor = EmergencySnapshotMonitor(
        campaign_root=campaign,
        snapshot_root=tmp_path / "snapshots",
        verified=verified,
        interval_seconds=0,
    )
    state = CampaignState(
        "f1_active", (), "tabm_f1_2022", None, None, monitor.bindings
    )
    state_path = campaign / "state/stage_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(serialize_state(state))
    log = campaign / "logs/campaign.log"
    log.parent.mkdir(parents=True)
    log.write_text("active\n")
    checkpoint = campaign / "jobs/tabm_f1_2022/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    latest = monitor.snapshot()
    assert latest is not None and latest.is_file()
    assert tuple((tmp_path / "snapshots").glob("*.zip")) == (latest,)
