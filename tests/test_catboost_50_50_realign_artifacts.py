from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from experiments.catboost_50_50_realign.artifacts import (
    Bindings,
    CampaignFiles,
    RealignArtifactError,
    TrustedFile,
    restore_resume,
    verify_delivery_bundle,
    verify_resume_bundle,
    write_delivery_bundle,
    write_resume_bundle,
)
from experiments.catboost_50_50_realign.state import (
    CampaignState,
    RealignStateError,
    serialize_state,
    validate_transition,
)


DELIVERY_NAMES = (
    "decision/alignment_decision.json",
    "decision/fold_metrics.json",
    "decision/bootstrap_metrics.json",
    "decision/segment_metrics.json",
    "frozen_catboost/model.cbm",
    "frozen_catboost/preprocessing_state.json",
    "frozen_catboost/inference_manifest.json",
    "logs/campaign.log",
    "policy/policy.json",
)


def _bindings() -> Bindings:
    return Bindings(
        contract_sha256="a" * 64,
        code_sha256="b" * 64,
        input_manifest_sha256="c" * 64,
        train_sha256="d" * 64,
        history_sha256="e" * 64,
    )


def _trusted(root: Path, name: str, payload: bytes) -> TrustedFile:
    path = root.joinpath(*name.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return TrustedFile(path=path, sha256=sha256(payload).hexdigest())


def _completed_state() -> CampaignState:
    return CampaignState(
        status="completed",
        completed_job_ids=("tabm_f1_2022", "catboost_f1_2022", "catboost_full_2024"),
        active_job_id=None,
        decision_sha256="f" * 64,
        selected_tree_count=16,
        bindings=_bindings(),
    )


def _files(tmp_path: Path, *, status: str = "completed") -> CampaignFiles:
    source = tmp_path / "source"
    state = _completed_state()
    state_file = _trusted(source, "state/stage_state.json", serialize_state(state))
    resume = {
        "state/stage_state.json": state_file,
        "jobs/catboost_full_2024/model.cbm": _trusted(
            source, "jobs/catboost_full_2024/model.cbm", b"model"
        ),
        "logs/campaign.log": _trusted(source, "logs/campaign.log", b"done\n"),
    }
    review = {
        "state/stage_state.json": state_file,
        "decision/alignment_decision.json": _trusted(
            source, "decision/alignment_decision.json", b"{}"
        ),
    }
    delivery = {
        name: _trusted(
            source,
            name,
            (
                b'{"selected_tree_count":16,"status":"promoted"}'
                if name == "decision/alignment_decision.json"
                else b"done\n"
                if name == "logs/campaign.log"
                else f"payload:{name}".encode()
            ),
        )
        for name in DELIVERY_NAMES
    }
    delivery["frozen_catboost/inference_manifest.json"] = _trusted(
        source,
        "frozen_catboost/inference_manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "selected_tree_count": 16,
                "model_sha256": delivery["frozen_catboost/model.cbm"].sha256,
                "preprocessing_sha256": delivery[
                    "frozen_catboost/preprocessing_state.json"
                ].sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    )
    delivery["policy/policy.json"] = _trusted(
        source,
        "policy/policy.json",
        b'{"campaign_id":"catboost_50_50_realign_v2","rule_safe":true,'
        b'"submission_package":false,"tabm_weight":"0.50","test_independent":true}',
    )
    return CampaignFiles(status=status, resume=resume, review=review, delivery=delivery)


def test_state_machine_accepts_only_forward_sealed_transitions() -> None:
    bindings = _bindings()
    fresh = CampaignState("fresh", (), None, None, None, bindings)
    active = CampaignState("f1_active", (), "tabm_f1_2022", None, None, bindings)
    f1_complete = CampaignState(
        "f1_complete", ("tabm_f1_2022", "catboost_f1_2022"), None, None, None, bindings
    )
    blocked = CampaignState(
        "deployment_blocked",
        f1_complete.completed_job_ids,
        None,
        "f" * 64,
        None,
        bindings,
    )
    validate_transition(fresh, active)
    validate_transition(active, f1_complete)
    validate_transition(f1_complete, blocked)
    with pytest.raises(RealignStateError, match="state fields differ"):
        CampaignState(
            "f1_active", (), "catboost_f1_2022", None, None, bindings
        ).validate()
    with pytest.raises(RealignStateError, match="transition"):
        validate_transition(blocked, active)
    with pytest.raises(RealignStateError, match="full model"):
        CampaignState(
            "completed", f1_complete.completed_job_ids, None, "f" * 64, 16, bindings
        ).validate()


def test_delivery_is_deterministic_verified_and_not_a_submission(tmp_path: Path) -> None:
    first = write_delivery_bundle(_files(tmp_path / "a"), tmp_path / "first.zip", _bindings())
    second = write_delivery_bundle(_files(tmp_path / "b"), tmp_path / "second.zip", _bindings())
    assert first.read_bytes() == second.read_bytes()
    verified = verify_delivery_bundle(first, expected_bindings=_bindings())
    assert verified.status == "completed"
    assert verified.submission_package is False
    assert verified.selected_tree_count == 16
    with ZipFile(first) as archive:
        assert set(archive.namelist()) == {"manifest.json", *DELIVERY_NAMES}


def test_blocked_candidate_cannot_write_delivery(tmp_path: Path) -> None:
    files = replace(_files(tmp_path), status="deployment_blocked", delivery={})
    with pytest.raises(RealignArtifactError, match="completed delivery evidence"):
        write_delivery_bundle(files, tmp_path / "delivery.zip", _bindings())


def test_trusted_source_mutation_is_detected_without_publishing(tmp_path: Path) -> None:
    files = _files(tmp_path)
    member = files.delivery["frozen_catboost/model.cbm"]
    member.path.write_bytes(b"changed")
    output = tmp_path / "delivery.zip"
    with pytest.raises(RealignArtifactError, match="changed while reading"):
        write_delivery_bundle(files, output, _bindings())
    assert not output.exists()
    assert not output.with_suffix(".zip.tmp").exists()


def test_tampered_member_and_binding_are_rejected(tmp_path: Path) -> None:
    original = write_delivery_bundle(_files(tmp_path), tmp_path / "delivery.zip", _bindings())
    tampered = tmp_path / "tampered.zip"
    with ZipFile(original) as source, ZipFile(tampered, "w", compression=ZIP_DEFLATED) as sink:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "frozen_catboost/model.cbm":
                payload += b"changed"
            sink.writestr(info, payload)
    with pytest.raises(RealignArtifactError, match="content differs"):
        verify_delivery_bundle(tampered, expected_bindings=_bindings())
    with pytest.raises(RealignArtifactError, match="bindings differ"):
        verify_delivery_bundle(
            original,
            expected_bindings=replace(_bindings(), code_sha256="9" * 64),
        )


def test_resume_restores_verified_members_atomically(tmp_path: Path) -> None:
    archive = write_resume_bundle(_files(tmp_path), tmp_path / "resume.zip", _bindings())
    verified = verify_resume_bundle(archive, expected_bindings=_bindings())
    restored = restore_resume(archive, tmp_path / "restored", _bindings())
    assert verified.status == "completed"
    assert restored.status == "completed"
    assert restored.state_path == tmp_path / "restored/state/stage_state.json"
    assert restored.state_path.is_file()


def test_resume_rejects_extra_or_unsafe_member(tmp_path: Path) -> None:
    original = write_resume_bundle(_files(tmp_path), tmp_path / "resume.zip", _bindings())
    altered = tmp_path / "altered.zip"
    with ZipFile(original) as source, ZipFile(altered, "w", compression=ZIP_DEFLATED) as sink:
        for info in source.infolist():
            sink.writestr(info, source.read(info.filename))
        sink.writestr("../escape", b"bad")
    with pytest.raises(RealignArtifactError, match="unsafe ZIP member"):
        verify_resume_bundle(altered, expected_bindings=_bindings())


def test_completed_state_and_model_are_both_required(tmp_path: Path) -> None:
    files = _files(tmp_path)
    resume = dict(files.resume)
    del resume["jobs/catboost_full_2024/model.cbm"]
    with pytest.raises(RealignArtifactError, match="full model"):
        write_resume_bundle(replace(files, resume=resume), tmp_path / "resume.zip", _bindings())


def test_delivery_inference_manifest_must_bind_frozen_files(tmp_path: Path) -> None:
    files = _files(tmp_path)
    members = dict(files.delivery)
    members["frozen_catboost/inference_manifest.json"] = _trusted(
        tmp_path / "changed",
        "frozen_catboost/inference_manifest.json",
        b'{"schema_version":1,"selected_tree_count":16,'
        b'"model_sha256":"0000000000000000000000000000000000000000000000000000000000000000",'
        b'"preprocessing_sha256":"0000000000000000000000000000000000000000000000000000000000000000"}',
    )
    with pytest.raises(RealignArtifactError, match="inference manifest differs"):
        write_delivery_bundle(
            replace(files, delivery=members), tmp_path / "delivery.zip", _bindings()
        )
