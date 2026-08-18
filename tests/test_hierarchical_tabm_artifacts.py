from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from zipfile import ZipFile

import pytest

from experiments.hierarchical_tabm.artifacts import (
    CampaignEvidence,
    DeliveryEvidence,
    HierarchicalArtifactError,
    restore_resume,
    verify_candidate_delivery,
    verify_resume_bundle,
    verify_review_bundle,
    write_campaign_bundles,
    write_candidate_delivery,
)
from experiments.hierarchical_tabm.inputs import EXPECTED_BINDING_KEYS


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return path


def _bindings() -> MappingProxyType:
    return MappingProxyType(
        {key: sha256(key.encode()).hexdigest() for key in sorted(EXPECTED_BINDING_KEYS)}
    )


def _campaign(tmp_path: Path, *, active: bool = True) -> CampaignEvidence:
    root = tmp_path / "source"
    contract = _write(root / "contract.json", {"campaign": "fixture"})
    log = _write(root / "campaign.log", b"CAMPAIGN_START\n")
    job = root / "job"
    _write(job / "worker_result.json", {"job_id": "job1", "status": "completed"})
    _write(job / "predictions.csv", b"row_id,target,probability\nr,0,0.5\n")
    state = _write(
        root / "stage_state.json",
        {
            "schema_version": 1,
            "status": "running",
            "completed_job_ids": ["job1"],
            "active_job_id": "job2" if active else None,
        },
    )
    active_dir = None
    if active:
        active_dir = root / "active"
        _write(active_dir / "checkpoint.pt", b"checkpoint")
        _write(active_dir / "checkpoint.pt.meta.json", {"completed_epochs": 2})
    return CampaignEvidence(
        bindings=_bindings(),
        contract_path=contract,
        log_path=log,
        state_path=state,
        k_selection_path=None,
        completed_job_directories=MappingProxyType({"job1": job}),
        calibration_paths=MappingProxyType({}),
        decision_paths=MappingProxyType({}),
        active_job_directory=active_dir,
    )


def test_review_and_resume_have_exact_verified_members(tmp_path: Path) -> None:
    evidence = _campaign(tmp_path)
    bundles = write_campaign_bundles(evidence, tmp_path / "out")
    review = verify_review_bundle(bundles.review, expected_bindings=evidence.bindings)
    resume = verify_resume_bundle(bundles.resume, expected_bindings=evidence.bindings)
    with ZipFile(bundles.review) as archive:
        review_names = set(archive.namelist())
    with ZipFile(bundles.resume) as archive:
        resume_names = set(archive.namelist())
    assert "active/job2/checkpoint.pt" not in review_names
    assert "active/job2/checkpoint.pt" in resume_names
    assert review["artifact_kind"] == "hierarchical_tabm_review_v1"
    assert resume["artifact_kind"] == "hierarchical_tabm_resume_v1"


def test_restore_resume_recreates_verified_evidence(tmp_path: Path) -> None:
    evidence = _campaign(tmp_path)
    bundle = write_campaign_bundles(evidence, tmp_path / "out").resume
    restored = restore_resume(
        bundle, tmp_path / "restored", expected_bindings=evidence.bindings
    )
    assert tuple(restored.completed_job_directories) == ("job1",)
    assert restored.active_job_directory.name == "job2"
    assert (restored.active_job_directory / "checkpoint.pt").read_bytes() == b"checkpoint"


def test_binding_or_payload_tamper_is_rejected(tmp_path: Path) -> None:
    evidence = _campaign(tmp_path)
    bundle = write_campaign_bundles(evidence, tmp_path / "out").resume
    wrong = dict(evidence.bindings)
    wrong["code_sha256"] = "0" * 64
    with pytest.raises(HierarchicalArtifactError, match="bindings"):
        verify_resume_bundle(bundle, expected_bindings=wrong)

    tampered = tmp_path / "tampered.zip"
    tampered.write_bytes(bundle.read_bytes())
    with ZipFile(tampered, "a") as archive:
        archive.writestr("stage_state.json", b"{}")
    with pytest.raises(HierarchicalArtifactError):
        verify_resume_bundle(tampered, expected_bindings=evidence.bindings)


def test_candidate_delivery_is_review_only_and_contains_no_submission(tmp_path: Path) -> None:
    root = tmp_path / "delivery-source"
    evidence = DeliveryEvidence(
        bindings=_bindings(),
        delivery_roles=MappingProxyType({"H1": "final_candidate", "H2": "final_candidate"}),
        final_checkpoint_path=_write(root / "final_checkpoint.pt", b"model"),
        feature_state_path=_write(root / "feature_state.json", {"state": 1}),
        calibration_paths=MappingProxyType(
            {"H2": _write(root / "calibration_H2.json", {"kind": "H2"})}
        ),
        independence_report_paths=MappingProxyType(
            {
                "H1": _write(root / "independence_H1.json", {"accepted": True}),
                "H2": _write(root / "independence_H2.json", {"accepted": True}),
            }
        ),
    )
    delivery = write_candidate_delivery(evidence, tmp_path / "out")
    manifest = verify_candidate_delivery(delivery, expected_bindings=evidence.bindings)
    with ZipFile(delivery) as archive:
        names = archive.namelist()
    assert manifest["review_only"] is True
    assert manifest["submission_package"] is False
    assert manifest["delivery_candidate_ids"] == ["H1", "H2"]
    assert not any("test" in name.lower() or "submit" in name.lower() for name in names)
    assert "model/final_checkpoint.pt" in names
    assert "state/feature_state.json" in names


def test_frontier_delivery_cannot_claim_final_acceptance(tmp_path: Path) -> None:
    root = tmp_path / "frontier"
    evidence = DeliveryEvidence(
        bindings=_bindings(),
        delivery_roles=MappingProxyType({"H1": "public_diagnostic_only"}),
        final_checkpoint_path=_write(root / "model.pt", b"model"),
        feature_state_path=_write(root / "state.json", {}),
        calibration_paths=MappingProxyType({}),
        independence_report_paths=MappingProxyType(
            {"H1": _write(root / "decision.json", {"final_acceptance": False})}
        ),
    )
    delivery = write_candidate_delivery(evidence, tmp_path / "out")
    manifest = verify_candidate_delivery(delivery, expected_bindings=evidence.bindings)
    assert manifest["delivery_roles"] == {"H1": "public_diagnostic_only"}
