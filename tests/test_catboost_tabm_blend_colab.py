from __future__ import annotations

from pathlib import Path
import time
from zipfile import ZipFile

import pytest

from experiments.catboost_tabm_blend.colab import (
    BlendColabError,
    EmergencyCadence,
    build_delivery,
    classify_upload_kind,
    verify_delivery,
)


def _zip(path: Path, names: dict[str, bytes]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, value in names.items():
            archive.writestr(name, value)
    return path


def test_upload_kind_is_content_based(tmp_path: Path) -> None:
    data = _zip(
        tmp_path / "anything.zip",
        {
            "input_manifest.json": b"{}",
            "train.csv": b"x",
            "trackman_history.csv": b"x",
        },
    )
    stage_c = _zip(
        tmp_path / "renamed.zip",
        {
            "delivery_manifest.json": b"{}",
            "colab_stage_C.log": b"x",
            "tabm_search_stage_C_review_bundle.zip": b"x",
            "tabm_search_stage_C_resume_bundle.zip": b"x",
        },
    )
    resume = _zip(
        tmp_path / "third.zip",
        {
            "contract/contract.json": b"{}",
            "state/stage_state.json": b"{}",
            "manifest.json": b'{"artifact_kind":"catboost_tabm_blend_resume"}',
        },
    )

    assert classify_upload_kind(data) == "training_input"
    assert classify_upload_kind(stage_c) == "stage_c_delivery"
    assert classify_upload_kind(resume) == "blend_resume"


def test_unknown_or_unsafe_upload_is_rejected(tmp_path: Path) -> None:
    unknown = _zip(tmp_path / "unknown.zip", {"sample_submission.csv": b"x"})
    unsafe = _zip(tmp_path / "unsafe.zip", {"../train.csv": b"x"})

    with pytest.raises(BlendColabError, match="unknown upload"):
        classify_upload_kind(unknown)
    with pytest.raises(BlendColabError, match="unsafe"):
        classify_upload_kind(unsafe)


def test_emergency_cadence_requires_time_and_changed_hash() -> None:
    cadence = EmergencyCadence(interval_seconds=1200, started_at=100.0)

    assert cadence.should_publish(now=100.0, snapshot_sha256="a" * 64) is False
    assert cadence.should_publish(now=1300.0, snapshot_sha256="a" * 64) is True
    cadence.mark_published(now=1300.0, snapshot_sha256="a" * 64)
    assert cadence.should_publish(now=2500.0, snapshot_sha256="a" * 64) is False
    assert cadence.should_publish(now=2499.0, snapshot_sha256="b" * 64) is False
    assert cadence.should_publish(now=2500.0, snapshot_sha256="b" * 64) is True


def test_delivery_has_exact_members_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "blend_campaign.log"
    review = tmp_path / "review.zip"
    resume = tmp_path / "resume.zip"
    log.write_text("BLEND_DECISION\n")
    review.write_bytes(b"review")
    resume.write_bytes(b"resume")
    monkeypatch.setattr(
        "experiments.catboost_tabm_blend.colab.verify_review_bundle",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.catboost_tabm_blend.colab.verify_resume_bundle",
        lambda *args, **kwargs: None,
    )
    bindings = {"code_sha256": "a" * 64, "contract_sha256": "b" * 64}

    delivery = build_delivery(
        destination=tmp_path / "delivery.zip",
        campaign_log=log,
        review_bundle=review,
        resume_bundle=resume,
        bindings=bindings,
    )
    verify_delivery(delivery, expected_bindings=bindings)
    with ZipFile(delivery) as archive:
        assert set(archive.namelist()) == {
            "blend_campaign.log",
            "catboost_tabm_blend_review.zip",
            "catboost_tabm_blend_resume.zip",
            "delivery_manifest.json",
        }

    _zip(tmp_path / "bad.zip", {"delivery_manifest.json": b"{}"})
    with pytest.raises(BlendColabError):
        verify_delivery(tmp_path / "bad.zip", expected_bindings=bindings)
