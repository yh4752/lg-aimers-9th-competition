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


def _snapshot_identity() -> dict[str, str]:
    return {
        "contract_sha256": "1" * 64,
        "data_archive_sha256": "2" * 64,
        "train_sha256": "3" * 64,
        "runtime_sha256": "4" * 64,
        "environment_sha256": "6" * 64,
        "training_source_sha256": "5" * 64,
    }


def test_emergency_snapshot_is_deterministic_and_bound(tmp_path: Path) -> None:
    from experiments.tabm_campaign.version_d import (
        verify_emergency_snapshot,
        write_emergency_snapshot,
    )

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    state = tmp_path / "preprocessing_state.json"
    state.write_bytes(b"{}")
    numeric = tmp_path / "numeric_embedding_0.json"
    numeric.write_bytes(b"{}")
    identity = _snapshot_identity()
    first = write_emergency_snapshot(
        tmp_path / "one", checkpoint, state, numeric, 1, identity
    )
    second = write_emergency_snapshot(
        tmp_path / "two", checkpoint, state, numeric, 1, identity
    )
    assert first.read_bytes() == second.read_bytes()
    verified = verify_emergency_snapshot(first, identity)
    assert verified.epoch == 1
    assert verified.sha256 == _digest(first.read_bytes())


def test_emergency_snapshot_rejects_identity_change(tmp_path: Path) -> None:
    from experiments.tabm_campaign.version_d import (
        verify_emergency_snapshot,
        write_emergency_snapshot,
    )

    files = []
    for name in ("checkpoint.pt", "preprocessing_state.json", "numeric_embedding_0.json"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files.append(path)
    snapshot = write_emergency_snapshot(
        tmp_path / "snapshots", *files, 2, _snapshot_identity()
    )
    changed = {**_snapshot_identity(), "train_sha256": "9" * 64}
    with pytest.raises(VersionDError, match="emergency snapshot identity differs"):
        verify_emergency_snapshot(snapshot, changed)


def _frozen_artifact(root: Path) -> None:
    members = {
        "preprocessing_state.json": b"{}",
        "numeric_embedding_0.json": b"{}",
        "tabm_member_0_seed_3407.pt": b"weights",
    }
    root.mkdir()
    for name, value in members.items():
        (root / name).write_bytes(value)
    manifest = {
        "schema_version": 1,
        "fit_scope": "official_train_2019_2024_only",
        "epochs": 3,
        "seeds": [3407],
        "scheduler": "constant",
        "preprocessing_state": "preprocessing_state.json",
        "members": [
            {
                "seed": 3407,
                "weights": "tabm_member_0_seed_3407.pt",
                "numeric_state": "numeric_embedding_0.json",
                "model_config": {"architecture": "tabm"},
            }
        ],
        "files": {name: _digest(value) for name, value in members.items()},
        "identity": _snapshot_identity(),
    }
    (root / "inference_manifest.json").write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )


def test_frozen_snapshot_round_trip_has_no_training_state(tmp_path: Path) -> None:
    from experiments.tabm_campaign.version_d import (
        verify_frozen_snapshot,
        write_frozen_snapshot,
    )

    artifact = tmp_path / "frozen"
    _frozen_artifact(artifact)
    snapshot = write_frozen_snapshot(
        tmp_path / "snapshots", artifact, _snapshot_identity()
    )
    verified = verify_frozen_snapshot(snapshot, _snapshot_identity())
    assert verified.sha256 == _digest(snapshot.read_bytes())
    with ZipFile(snapshot) as archive:
        lowered = "\n".join(archive.namelist()).lower()
    assert "optimizer" not in lowered
    assert "scaler" not in lowered
    assert "rng" not in lowered


def test_frozen_restore_replaces_partial_training_state(tmp_path: Path) -> None:
    from experiments.tabm_campaign.version_d import (
        restore_frozen_snapshot,
        write_frozen_snapshot,
    )

    source = tmp_path / "source"
    _frozen_artifact(source)
    snapshot = write_frozen_snapshot(
        tmp_path / "snapshots", source, _snapshot_identity()
    )
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "preprocessing_state.json").write_bytes(b"partial")
    restore_frozen_snapshot(snapshot, destination, _snapshot_identity())
    assert (destination / "inference_manifest.json").is_file()
    assert not (tmp_path / ".destination.previous").exists()


def test_recovery_selects_frozen_before_latest_emergency(tmp_path: Path) -> None:
    from experiments.tabm_campaign.version_d import (
        select_recovery_snapshot,
        write_emergency_snapshot,
        write_frozen_snapshot,
    )

    files = []
    for name in ("checkpoint.pt", "preprocessing_state.json", "numeric_embedding_0.json"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files.append(path)
    emergency = write_emergency_snapshot(
        tmp_path / "emergency", *files, 3, _snapshot_identity()
    )
    artifact = tmp_path / "frozen"
    _frozen_artifact(artifact)
    frozen = write_frozen_snapshot(tmp_path / "snapshots", artifact, _snapshot_identity())
    selected = select_recovery_snapshot([emergency], [frozen], _snapshot_identity())
    assert selected.mode == "frozen"
    assert selected.path == frozen


def test_review_delivery_is_review_only_and_verifiable(tmp_path: Path) -> None:
    from experiments.tabm_campaign.version_d import (
        verify_review_delivery,
        write_review_delivery,
    )

    review = write_stage_bundles(
        tmp_path / "stage_d",
        StageEvidence(
            "D",
            "1" * 64,
            "4" * 64,
            {"final_review.json": b"{}"},
            {},
        ),
    ).review
    log = tmp_path / "version_d.log"
    log.write_bytes(b"VERSION_D_REVIEW_READY\n")
    delivery = write_review_delivery(
        output_dir=tmp_path / "output",
        review_bundle=review,
        log_path=log,
        contract_sha256="1" * 64,
        data_archive_sha256="2" * 64,
        stage_c_delivery_sha256="3" * 64,
        prior_manifest_sha256="4" * 64,
        frozen_sha256="5" * 64,
        runtime_versions={"python": "3.12"},
    )
    manifest = verify_review_delivery(delivery)
    assert manifest["review_only"] is True
    assert manifest["submission_package"] is False
    with ZipFile(delivery) as archive:
        assert set(archive.namelist()) == {
            "delivery_manifest.json",
            "tabm_hand_matchup_final_review_bundle.zip",
            "version_d.log",
        }
