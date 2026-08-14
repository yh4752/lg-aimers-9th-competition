from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from competition_rules.contract import load_policy, policy_digest
from experiments.tabm_campaign.artifacts import StageEvidence, write_stage_bundles
from experiments.tabm_campaign.version_d import (
    VersionDError,
    extract_review_inputs,
    extract_training_input,
    load_version_d_contract,
    verify_stage_c_delivery,
)


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)


def _data_contract(path: Path) -> dict[str, object]:
    with ZipFile(path) as archive:
        members = {
            item.filename: {
                "size": item.file_size,
                "sha256": _digest(archive.read(item.filename)),
            }
            for item in archive.infolist()
        }
    return {
        "data_archive": {
            "sha256": _digest(path.read_bytes()),
            "members": members,
        }
    }


def _stage_c_delivery(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    candidate = {
        "family": "tabm",
        "capacity": "p2",
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.1,
        "num_embedding": "piecewise_linear",
        "loss": "bce",
        "scheduler": "plateau",
        "learning_rate": 0.0006,
        "seed": 3407,
    }
    state = {
        "version": "C",
        "stage_complete": True,
        "selected_predictor": "single_s3407",
        "results": [{"candidate_id": "result", "status": "completed"}],
        "final_members": [
            {
                "candidate_id": "final_s3407",
                "status": "completed",
                "seed": 3407,
                "temporal_best_epochs": [3, 0],
                "candidate": candidate,
            }
        ],
    }
    state_bytes = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    bundles = write_stage_bundles(
        tmp_path / "stage_c",
        StageEvidence(
            "C",
            "a" * 64,
            "b" * 64,
            {"stage_state.json": state_bytes},
            {"stage_state.json": state_bytes},
        ),
    )
    assert bundles.resume is not None
    log = b"stage-c-log\n"
    review = bundles.review.read_bytes()
    resume = bundles.resume.read_bytes()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tabm_colab_stage_C_delivery",
        "final_stage_complete": True,
        "members": {
            "colab_stage_C.log": {"size": len(log), "sha256": _digest(log)},
            bundles.review.name: {"size": len(review), "sha256": _digest(review)},
            bundles.resume.name: {"size": len(resume), "sha256": _digest(resume)},
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    delivery = tmp_path / "tabm_colab_stage_C_delivery.zip"
    _write_zip(
        delivery,
        {
            "colab_stage_C.log": log,
            "delivery_manifest.json": manifest_bytes,
            bundles.review.name: review,
            bundles.resume.name: resume,
        },
    )
    contract = {
        "stage_c_delivery": {
            "sha256": _digest(delivery.read_bytes()),
            "review_sha256": _digest(review),
            "resume_sha256": _digest(resume),
            "stage_state_sha256": _digest(state_bytes),
            "completed_jobs": 1,
        },
        "predictor_id": "single_s3407",
        "selected_candidate": candidate,
        "expected_stage_c_manifest_sha256": bundles.manifest_sha256,
    }
    return delivery, contract


def test_contract_seals_single_s3407_and_three_epochs() -> None:
    contract = load_version_d_contract()
    assert contract["predictor_id"] == "single_s3407"
    assert contract["final_fit"]["epochs"] == 3
    assert contract["selected_candidate"]["seed"] == 3407
    assert contract["preprocessing"] == {
        "profile": "dl_standard",
        "components": ["hand_matchup"],
        "fit_scope": "official_train_only",
    }


def test_same_day_policy_review_matches_policy_digest() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = load_policy(root / "competition_rules/policy.json", project_root=root)
    review = json.loads(
        (root / "reports/rules/2026-08-14-policy-review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["verdict"] == "unchanged"
    assert review["policy_sha256"] == policy_digest(policy)


def test_training_extraction_does_not_materialize_test(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    _write_zip(
        archive,
        {
            "train.csv": b"season,target\n2024,1\n",
            "trackman_history.csv": b"x\n1\n",
            "test.csv": b"row_id\na\n",
            "sample_submission.csv": b"row_id,target\na,0.5\n",
        },
    )
    contract = _data_contract(archive)
    train_root = tmp_path / "train_only"
    extract_training_input(archive, train_root, contract)
    assert sorted(path.name for path in train_root.iterdir()) == ["train.csv"]

    review_root = tmp_path / "review"
    extract_review_inputs(archive, review_root, contract)
    assert sorted(path.name for path in review_root.iterdir()) == [
        "sample_submission.csv",
        "test.csv",
    ]


def test_stage_c_delivery_verifies_lineage_and_selection(tmp_path: Path) -> None:
    delivery, contract = _stage_c_delivery(tmp_path)
    evidence = verify_stage_c_delivery(delivery, contract)
    assert evidence["prior_manifest_sha256"] == contract["expected_stage_c_manifest_sha256"]
    assert evidence["selected_predictor"] == "single_s3407"
    assert evidence["final_member"]["seed"] == 3407


def test_stage_c_delivery_rejects_outer_tampering(tmp_path: Path) -> None:
    delivery, contract = _stage_c_delivery(tmp_path)
    contract["stage_c_delivery"]["sha256"] = "0" * 64
    with pytest.raises(VersionDError, match="Stage C delivery SHA-256 differs"):
        verify_stage_c_delivery(delivery, contract)
