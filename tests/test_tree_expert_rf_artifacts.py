from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType
from zipfile import ZipFile

import pytest

from experiments.tree_expert.rf_artifacts import (
    RFArtifactError,
    RFBindings,
    create_rf_delivery,
    create_rf_resume,
    restore_rf_resume,
    verify_rf_delivery,
    verify_rf_resume,
)
from experiments.tree_expert.rf_decisions import RFAcceptanceDecision


BINDINGS = RFBindings(
    contract_sha256="1" * 64,
    code_sha256="2" * 64,
    input_manifest_sha256="3" * 64,
    official_train_sha256="4" * 64,
    official_history_sha256="5" * 64,
    e2_handoff_sha256="6" * 64,
)


def _decision(status: str = "accepted") -> RFAcceptanceDecision:
    return RFAcceptanceDecision(
        status=status,
        reason="all_rf_gates_passed" if status == "accepted" else "weighted_gain_failed",
        f_head="f_small",
        include_r=False,
        alpha_r=0.0,
        alpha_f=0.5,
        fold_gains=MappingProxyType({(2021, 2022): 0.1, (2022, 2023): 0.1, (2023, 2024): 0.1}),
        segment_gains=MappingProxyType({"R": 0.0, "F": 0.1}),
        recent_f_gain=0.1,
        weighted_gain=0.1,
        improved_fold_count=3,
        maximum_segment_regression=0.0,
    )


def _campaign(root: Path, *, accepted: bool = True) -> Path:
    (root / "jobs/rf__f_small__tr2021__va2022__s3407").mkdir(parents=True)
    (root / "jobs/rf__f_small__tr2021__va2022__s3407/metrics.json").write_text(
        '{"status":"completed","job_id":"rf__f_small__tr2021__va2022__s3407"}'
    )
    (root / "jobs/rf__f_small__tr2021__va2022__s3407/checkpoint.cbm").write_bytes(b"model")
    (root / "campaign_state.json").write_text(
        json.dumps({"status": "accepted" if accepted else "rejected"})
    )
    (root / "decisions").mkdir()
    (root / "decisions/acceptance.json").write_text(json.dumps({"status": "accepted" if accepted else "rejected"}))
    (root / "audits").mkdir()
    (root / "audits/independence.json").write_text('{"status":"passed"}')
    (root / "full_fit/f_small/frozen_state").mkdir(parents=True)
    (root / "full_fit/f_small/frozen_state/feature_state.json").write_text("{}")
    (root / "full_fit/f_small/models").mkdir()
    for seed in (42, 2026, 3407):
        (root / f"full_fit/f_small/models/seed_{seed}.cbm").write_bytes(f"model-{seed}".encode())
    (root / "full_fit/full_fit_manifest.json").write_text('{"status":"accepted"}')
    return root


def test_resume_round_trip_preserves_completed_jobs(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path / "campaign")
    (campaign / "jobs/rf__f_small__tr2021__va2022__s3407/.predictions.csv.tmp").write_text("partial")

    bundle = create_rf_resume(campaign, tmp_path / "resume.zip", BINDINGS)
    restored = restore_rf_resume(bundle, tmp_path / "restored", BINDINGS)

    assert restored.completed_job_ids == ("rf__f_small__tr2021__va2022__s3407",)
    assert not tuple(restored.root.rglob("*.tmp"))


def test_resume_rejects_identity_mismatch(tmp_path: Path) -> None:
    bundle = create_rf_resume(_campaign(tmp_path / "campaign"), tmp_path / "resume.zip", BINDINGS)

    with pytest.raises(RFArtifactError, match="artifact bindings differ"):
        verify_rf_resume(bundle, replace(BINDINGS, code_sha256="f" * 64))


def test_resume_rejects_member_tampering(tmp_path: Path) -> None:
    bundle = create_rf_resume(_campaign(tmp_path / "campaign"), tmp_path / "resume.zip", BINDINGS)
    changed = tmp_path / "changed.zip"
    with ZipFile(bundle) as source, ZipFile(changed, "w") as target:
        for info in source.infolist():
            payload = source.read(info)
            if info.filename == "campaign_state.json":
                payload += b"\n"
            target.writestr(info, payload)

    with pytest.raises(RFArtifactError, match="member evidence differs"):
        verify_rf_resume(changed, BINDINGS)


def test_delivery_is_blocked_until_accepted(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path / "campaign", accepted=False)

    with pytest.raises(RFArtifactError, match="accepted decision is required"):
        create_rf_delivery(campaign, tmp_path / "delivery.zip", _decision("rejected"), BINDINGS)


def test_delivery_contains_no_submission_entry_point(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path / "campaign")
    delivery = create_rf_delivery(campaign, tmp_path / "delivery.zip", _decision(), BINDINGS)
    verified = verify_rf_delivery(delivery, BINDINGS)

    assert verified.kind == "tree_expert_rf_delivery_v1"
    with ZipFile(delivery) as archive:
        assert "script.py" not in archive.namelist()
        assert "submission.csv" not in archive.namelist()
