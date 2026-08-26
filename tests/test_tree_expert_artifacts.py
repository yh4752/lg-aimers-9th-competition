from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.tree_expert.artifacts import (
    TreeArtifactError,
    verify_e1_handoff,
    verify_e1_resume,
    write_e1_bundles,
)


def _bindings() -> dict[str, str]:
    return {
        "contract_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "input_manifest_sha256": "3" * 64,
        "train_sha256": "4" * 64,
        "history_sha256": "5" * 64,
        "stage_c_delivery_sha256": "6" * 64,
        "stage_c_review_sha256": "7" * 64,
        "baseline_predictions_sha256": "8" * 64,
    }


def _campaign(tmp_path: Path, *, decision_status: str = "completed") -> dict[str, object]:
    source = tmp_path / "source"
    source.mkdir()
    contract = source / "e1_contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    log = source / "tree_expert_e1.log"
    log.write_text("TREE_E1_JOB_END\n", encoding="utf-8")
    decision = source / "decision.json"
    decision.write_text(
        json.dumps({"status": decision_status, "promoted": []}), encoding="utf-8"
    )
    audit = source / "failure_label_audit.json"
    audit.write_text(json.dumps({"status": "skipped"}), encoding="utf-8")
    completed_id = "e1__c0_native_ctr__tr2023__va2024__s3407"
    skipped_id = "e1__c3_failure_aware__tr2023__va2024__s3407"
    completed = source / completed_id
    skipped = source / skipped_id
    completed.mkdir()
    skipped.mkdir()
    for root, status in ((completed, "completed"), (skipped, "skipped")):
        (root / "job.json").write_text(json.dumps({"job_id": root.name}), encoding="utf-8")
        (root / "worker_result.json").write_text(
            json.dumps({"job_id": root.name, "status": status}), encoding="utf-8"
        )
        (root / "metrics.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        (root / "worker.log").write_text(f"status={status}\n", encoding="utf-8")
    (completed / "predictions.csv").write_text(
        "row_id,target,probability,game_type,game_month,pitcher_id_known,batter_id_known\n"
        "r0,0,0.4,R,3,known,known\n",
        encoding="utf-8",
    )
    (completed / "model.cbm").write_bytes(b"model")
    state = source / "stage_state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "tree_expert_e1_v1",
                "status": decision_status,
                "completed": [completed_id],
                "skipped": [skipped_id],
                "failed": [],
                "active": {},
                "bindings": _bindings(),
            }
        ),
        encoding="utf-8",
    )
    return {
        "contract_path": contract,
        "bindings": _bindings(),
        "state_path": state,
        "log_path": log,
        "job_directories": (completed, skipped),
        "decision_path": decision,
        "audit_paths": (audit,),
    }


def test_e1_resume_round_trip_preserves_completed_and_skipped_jobs(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    bundles = write_e1_bundles(output_dir=tmp_path / "bundles", **campaign)
    verified = verify_e1_resume(bundles.resume, _bindings())

    assert verified.completed == (
        "e1__c0_native_ctr__tr2023__va2024__s3407",
    )
    assert verified.skipped == (
        "e1__c3_failure_aware__tr2023__va2024__s3407",
    )
    assert verified.submission_package is False
    assert bundles.handoff.is_file()
    handoff = verify_e1_handoff(bundles.handoff)
    assert handoff.review_only is True
    assert handoff.submission_package is False


def test_rejected_e1_never_contains_delivery_or_submission(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, decision_status="rejected")

    bundles = write_e1_bundles(output_dir=tmp_path / "bundles", **campaign)

    assert bundles.review.is_file()
    assert bundles.resume.is_file()
    with ZipFile(bundles.review) as archive:
        assert not any(
            "submission" in name or "delivery" in name for name in archive.namelist()
        )


@pytest.mark.parametrize("unsafe", ["../x", "/x", "a/../../x"])
def test_e1_verifier_rejects_unsafe_members(tmp_path: Path, unsafe: str) -> None:
    archive = tmp_path / "malicious.zip"
    with ZipFile(archive, "w") as output:
        output.writestr(unsafe, b"x")

    with pytest.raises(TreeArtifactError, match="unsafe"):
        verify_e1_resume(archive, _bindings())
