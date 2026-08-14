from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.tabm_campaign.colab_recovery import (
    ColabRecoveryError,
    ProcessReceipt,
    RuntimeIdentity,
    SanitizedResume,
    acquire_process_lock,
    create_emergency_snapshot,
    file_sha256,
    merge_emergency_snapshot,
    process_identity,
    sanitize_stage_c_resume,
    supervise_campaign,
    verify_emergency_snapshots,
    verify_delivery_bundle,
    verify_and_extract_data_archive,
    write_delivery_bundle,
    write_process_receipt,
)
from experiments.tabm_campaign.artifacts import (
    StageEvidence,
    verify_resume_bundle,
    write_stage_bundles,
)
from tools.prepare_tabm_colab_stage_c_handoff import prepare_handoff


_MEMBERS = {
    "sample_submission.csv": b"prediction\n",
    "test.csv": b"row_id\n1\n",
    "trackman_history.csv": b"season\n2023\n",
    "train.csv": b"row_id,target\n1,0\n",
}


def _contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "data_archive": {
            "filename": "lg-aimers-9th-data.zip",
            "max_member_count": 4,
            "max_uncompressed_bytes": sum(len(value) for value in _MEMBERS.values()),
            "members": {
                name: {"size": len(value), "sha256": sha256(value).hexdigest()}
                for name, value in _MEMBERS.items()
            },
        },
    }


def _write_archive(path: Path, members: list[tuple[str | ZipInfo, bytes]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members:
            archive.writestr(name, value)
    return path


def _valid_archive(path: Path) -> Path:
    return _write_archive(path, list(_MEMBERS.items()))


def test_verify_and_extract_data_archive_rejects_traversal(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "data.zip", [("../train.csv", b"escape")])

    with pytest.raises(ColabRecoveryError, match="unsafe"):
        verify_and_extract_data_archive(archive, tmp_path / "data", _contract())


def test_verify_and_extract_data_archive_is_atomic_and_hash_checked(
    tmp_path: Path,
) -> None:
    archive = _valid_archive(tmp_path / "data.zip")

    published = verify_and_extract_data_archive(
        archive, tmp_path / "published", _contract()
    )

    assert {path.name for path in published.iterdir()} == set(_MEMBERS)
    assert not (tmp_path / ".published.extracting").exists()
    for name, value in _MEMBERS.items():
        assert (published / name).read_bytes() == value


def test_existing_verified_data_directory_is_reused(tmp_path: Path) -> None:
    archive = _valid_archive(tmp_path / "data.zip")
    destination = verify_and_extract_data_archive(
        archive, tmp_path / "published", _contract()
    )
    marker = destination / "reuse-marker"
    marker.write_text("must make live verification fail", encoding="utf-8")

    with pytest.raises(ColabRecoveryError, match="member names"):
        verify_and_extract_data_archive(archive, destination, _contract())


@pytest.mark.parametrize("mutation", ["symlink", "duplicate", "extra", "wrong_hash"])
def test_data_archive_rejects_unsafe_or_changed_members(
    tmp_path: Path,
    mutation: str,
) -> None:
    members: list[tuple[str | ZipInfo, bytes]] = list(_MEMBERS.items())
    if mutation == "symlink":
        link = ZipInfo("train.csv")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        members = [
            (link, b"target") if name == "train.csv" else (name, value)
            for name, value in members
        ]
    elif mutation == "duplicate":
        members.append(("train.csv", _MEMBERS["train.csv"]))
    elif mutation == "extra":
        members.append(("extra.csv", b"extra"))
    else:
        members = [
            (name, b"changed" if name == "train.csv" else value)
            for name, value in members
        ]
    if mutation == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive = _write_archive(tmp_path / mutation / "data.zip", members)
    else:
        archive = _write_archive(tmp_path / mutation / "data.zip", members)

    with pytest.raises(ColabRecoveryError):
        verify_and_extract_data_archive(
            archive, tmp_path / f"out-{mutation}", _contract()
        )


def test_file_sha256_streams_expected_digest(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    value = b"abcdef" * 500_000
    path.write_bytes(value)

    assert file_sha256(path, chunk_size=1024) == sha256(value).hexdigest()


def _stage_c_resume(
    root: Path, *, second_incomplete: bool = False
) -> tuple[Path, dict[str, object]]:
    target_id = "target-candidate"
    completed = [
        {
            "candidate_id": f"completed-{index}",
            "status": "completed",
            "brier": 0.24 + index / 100_000,
            "best_epoch": 2,
            "completed_epochs": 3,
            "checkpoint": "best_checkpoint.pt",
            "predictions": "predictions.csv",
            "resource_evidence": {},
            "failure": None,
        }
        for index in range(20)
    ]
    target = {
        "candidate_id": target_id,
        "status": "inconclusive",
        "brier": 0.25,
        "best_epoch": 0,
        "completed_epochs": 3,
        "checkpoint": "best_checkpoint.pt",
        "predictions": "predictions.csv",
        "resource_evidence": {"cache_digest": "c" * 64},
        "failure": None,
    }
    rows = [*completed, target]
    if second_incomplete:
        rows.append({**target, "candidate_id": "unexpected-pending"})
    resume_artifacts: dict[str, object] = {
        row["candidate_id"]: {
            "predictions": f"predictions/{row['candidate_id']}.csv"
        }
        for row in completed
    }
    resume_artifacts[target_id] = {
        "training_files": [
            f"training/{target_id}/checkpoint.pt",
            f"training/{target_id}/checkpoint_meta.json",
            f"training/{target_id}/best_checkpoint.pt",
        ]
    }
    state = {
        "version": "C",
        "stage_complete": False,
        "reason": "older_fold_seed_confirmation_incomplete",
        "results": rows,
        "resume_artifacts": resume_artifacts,
    }
    members: dict[str, bytes] = {
        "stage_state.json": json.dumps(
            state, sort_keys=True, separators=(",", ":")
        ).encode(),
        **{
            f"predictions/{row['candidate_id']}.csv": (
                f"row_id,target,probability\nr-{index},0,0.5\n".encode()
            )
            for index, row in enumerate(completed)
        },
        f"training/{target_id}/checkpoint.pt": b"checkpoint",
        f"training/{target_id}/checkpoint_meta.json": json.dumps(
            {
                "candidate_id": target_id,
                "epoch": 2,
                "checkpoint": "checkpoint.pt",
                "checkpoint_binding": {"cache_sha256": "c" * 64},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        f"training/{target_id}/best_checkpoint.pt": b"best",
    }
    bundle = write_stage_bundles(
        root,
        StageEvidence(
            "C",
            "a" * 64,
            "b" * 64,
            {"stage_state.json": members["stage_state.json"]},
            members,
        ),
    ).resume
    assert bundle is not None
    contract = {
        "base_resume": {
            "filename": bundle.name,
            "sha256": file_sha256(bundle),
            "version": "C",
        },
        "target_candidate_id": target_id,
        "expected_completed_jobs": 20,
    }
    return bundle, contract


def test_sanitize_stage_c_resume_preserves_completed_bytes_and_drops_only_target_training(
    tmp_path: Path,
) -> None:
    source_path, contract = _stage_c_resume(tmp_path / "source")

    sanitized = sanitize_stage_c_resume(
        source_path,
        tmp_path / "sanitized.zip",
        contract,
    )

    assert isinstance(sanitized, SanitizedResume)
    with ZipFile(source_path) as source, ZipFile(sanitized.path) as output:
        target_prefix = f"training/{contract['target_candidate_id']}/"
        for name in source.namelist():
            if name in {"manifest.json", "stage_state.json"} or name.startswith(
                target_prefix
            ):
                continue
            assert output.read(name) == source.read(name)
        state = json.loads(output.read("stage_state.json"))
    target = next(
        row
        for row in state["results"]
        if row["candidate_id"] == contract["target_candidate_id"]
    )
    assert target["status"] == "inconclusive"
    assert target["brier"] is None
    assert target["best_epoch"] is None
    assert target["completed_epochs"] == 0
    assert contract["target_candidate_id"] not in state["resume_artifacts"]
    assert sum(row["status"] == "completed" for row in state["results"]) == 20
    assert verify_resume_bundle(sanitized.path).version == "C"


def test_sanitize_stage_c_resume_rejects_wrong_source_hash(tmp_path: Path) -> None:
    source_path, contract = _stage_c_resume(tmp_path / "source")
    contract["base_resume"]["sha256"] = "0" * 64
    destination = tmp_path / "sanitized.zip"

    with pytest.raises(ColabRecoveryError, match="SHA-256"):
        sanitize_stage_c_resume(source_path, destination, contract)

    assert not destination.exists()


def test_sanitize_stage_c_resume_rejects_a_second_pending_job(
    tmp_path: Path,
) -> None:
    source_path, contract = _stage_c_resume(
        tmp_path / "source", second_incomplete=True
    )

    with pytest.raises(ColabRecoveryError, match="completed/pending"):
        sanitize_stage_c_resume(source_path, tmp_path / "sanitized.zip", contract)


def _sanitized_fixture(
    tmp_path: Path,
) -> tuple[SanitizedResume, dict[str, object]]:
    source, contract = _stage_c_resume(tmp_path / "source")
    return (
        sanitize_stage_c_resume(source, tmp_path / "sanitized.zip", contract),
        contract,
    )


def _runtime_identity(
    sanitized: SanitizedResume,
    contract: dict[str, object],
    *,
    python: str = "3.11.9",
) -> RuntimeIdentity:
    return RuntimeIdentity(
        base_resume_sha256=sanitized.source_sha256,
        base_manifest_sha256=sanitized.source_manifest_sha256,
        sanitized_resume_sha256=sanitized.sanitized_sha256,
        campaign_config_sha256="a" * 64,
        runtime_sha256="e" * 64,
        training_source_sha256="d" * 64,
        cache_sha256="c" * 64,
        python=python,
        torch="2.5.1+cu121",
        cuda_runtime="12.1",
        numpy="2.1.0",
        pandas="2.2.3",
        tabm="0.0.3",
        rtdl_num_embeddings="0.0.12",
        gpu_name="Tesla T4",
        target_candidate_id=str(contract["target_candidate_id"]),
    )


def _stable_training_dir(
    root: Path,
    identity: RuntimeIdentity,
    *,
    epoch: int,
    marker: bytes = b"state",
) -> Path:
    root.mkdir(parents=True)
    (root / "checkpoint.pt").write_bytes(b"checkpoint-" + marker)
    (root / "best_checkpoint.pt").write_bytes(b"best-" + marker)
    (root / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "candidate_id": identity.target_candidate_id,
                "epoch": epoch,
                "checkpoint": "checkpoint.pt",
                "checkpoint_binding": {
                    "config_sha256": identity.campaign_config_sha256,
                    "cache_sha256": identity.cache_sha256,
                    "training_source_sha256": identity.training_source_sha256,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return root


def test_snapshot_round_trip_restores_next_epoch(tmp_path: Path) -> None:
    sanitized, contract = _sanitized_fixture(tmp_path)
    identity = _runtime_identity(sanitized, contract)
    training = _stable_training_dir(
        tmp_path / "training", identity, epoch=4
    )
    log = tmp_path / "colab.log"
    log.write_text("EPOCH_CHECKPOINTED epoch=4\n", encoding="utf-8")

    snapshot = create_emergency_snapshot(
        training_dir=training,
        log_path=log,
        output_dir=tmp_path / "snapshots",
        identity=identity,
    )
    verified = verify_emergency_snapshots(
        [snapshot.path], sanitized, identity
    )
    merged = merge_emergency_snapshot(
        sanitized,
        verified,
        tmp_path / "merged.zip",
    )

    with ZipFile(merged) as archive:
        state = json.loads(archive.read("stage_state.json"))
        target = str(contract["target_candidate_id"])
        binding = state["resume_artifacts"][target]["training_files"]
        assert len(binding) == 3
        meta = json.loads(
            archive.read(f"training/{target}/checkpoint_meta.json")
        )
    assert meta["epoch"] == 4
    row = next(item for item in state["results"] if item["candidate_id"] == target)
    assert row["completed_epochs"] == 5
    assert verify_resume_bundle(merged).version == "C"


def test_snapshot_verifier_selects_highest_compatible_epoch(tmp_path: Path) -> None:
    sanitized, contract = _sanitized_fixture(tmp_path)
    identity = _runtime_identity(sanitized, contract)
    log = tmp_path / "colab.log"
    log.write_text("log\n", encoding="utf-8")
    older = create_emergency_snapshot(
        training_dir=_stable_training_dir(
            tmp_path / "training-2", identity, epoch=2
        ),
        log_path=log,
        output_dir=tmp_path / "snapshots",
        identity=identity,
    )
    newer = create_emergency_snapshot(
        training_dir=_stable_training_dir(
            tmp_path / "training-5", identity, epoch=5
        ),
        log_path=log,
        output_dir=tmp_path / "snapshots",
        identity=identity,
    )

    selected = verify_emergency_snapshots(
        [older.path, newer.path], sanitized, identity
    )

    assert selected.epoch == 5


def test_snapshot_verifier_rejects_environment_mismatch(tmp_path: Path) -> None:
    sanitized, contract = _sanitized_fixture(tmp_path)
    identity = _runtime_identity(sanitized, contract)
    log = tmp_path / "colab.log"
    log.write_text("log\n", encoding="utf-8")
    snapshot = create_emergency_snapshot(
        training_dir=_stable_training_dir(
            tmp_path / "training", identity, epoch=1
        ),
        log_path=log,
        output_dir=tmp_path / "snapshots",
        identity=identity,
    )

    with pytest.raises(ColabRecoveryError, match="environment"):
        verify_emergency_snapshots(
            [snapshot.path],
            sanitized,
            _runtime_identity(sanitized, contract, python="3.12.0"),
        )


def test_snapshot_verifier_rejects_same_epoch_conflict(tmp_path: Path) -> None:
    sanitized, contract = _sanitized_fixture(tmp_path)
    identity = _runtime_identity(sanitized, contract)
    first_log = tmp_path / "first.log"
    second_log = tmp_path / "second.log"
    first_log.write_text("first\n", encoding="utf-8")
    second_log.write_text("second\n", encoding="utf-8")
    first = create_emergency_snapshot(
        training_dir=_stable_training_dir(
            tmp_path / "training-a", identity, epoch=3, marker=b"a"
        ),
        log_path=first_log,
        output_dir=tmp_path / "snapshots",
        identity=identity,
    )
    second = create_emergency_snapshot(
        training_dir=_stable_training_dir(
            tmp_path / "training-b", identity, epoch=3, marker=b"b"
        ),
        log_path=second_log,
        output_dir=tmp_path / "snapshots",
        identity=identity,
    )

    with pytest.raises(ColabRecoveryError, match="same epoch"):
        verify_emergency_snapshots(
            [first.path, second.path], sanitized, identity
        )


def test_live_matching_lock_refuses_duplicate_run(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        receipt = write_process_receipt(
            tmp_path / "run.lock", process.pid, "run-a"
        )

        with pytest.raises(ColabRecoveryError, match="already running"):
            acquire_process_lock(tmp_path / "run.lock", "run-b")

        assert receipt.pid == process.pid
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_stale_or_pid_reused_lock_is_archived(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "process_start_identity": "different-start",
                "command_sha256": "0" * 64,
                "run_uuid": "old",
            }
        ),
        encoding="utf-8",
    )

    acquired = acquire_process_lock(lock, "new")

    assert isinstance(acquired, ProcessReceipt)
    assert acquired.run_uuid == "new"
    assert list(tmp_path.glob("run.lock.stale-*"))


def test_process_identity_disappears_after_process_exit() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    assert process_identity(process.pid) is not None
    process.terminate()
    process.wait(timeout=5)

    assert process_identity(process.pid) is None


def _completed_stage_c_bundles(root: Path) -> tuple[Path, Path]:
    state = json.dumps(
        {"version": "C", "stage_complete": True, "results": []},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    bundles = write_stage_bundles(
        root,
        StageEvidence(
            "C",
            "a" * 64,
            "b" * 64,
            {"stage_state.json": state, "metrics/job_results.json": b"[]"},
            {"stage_state.json": state},
        ),
    )
    assert bundles.resume is not None
    return bundles.review, bundles.resume


def test_delivery_verifies_nested_bundles_and_binds_hashes(tmp_path: Path) -> None:
    review, resume = _completed_stage_c_bundles(tmp_path / "stage-c")
    log = tmp_path / "colab.log"
    log.write_text("BUNDLE_SUCCESS version=C\n", encoding="utf-8")
    sanitized, contract = _sanitized_fixture(tmp_path / "identity")
    identity = _runtime_identity(sanitized, contract)

    delivery = write_delivery_bundle(
        review=review,
        resume=resume,
        log=log,
        identity=identity,
        run_uuid="run-a",
        destination=tmp_path / "tabm_colab_stage_C_delivery.zip",
    )
    verified = verify_delivery_bundle(delivery)

    assert verified.final_stage_complete is True
    assert set(verified.member_sha256) == {
        "tabm_search_stage_C_review_bundle.zip",
        "tabm_search_stage_C_resume_bundle.zip",
        "colab_stage_C.log",
    }


def test_delivery_rejects_incomplete_stage_c(tmp_path: Path) -> None:
    review, resume = _completed_stage_c_bundles(tmp_path / "stage-c")
    with ZipFile(resume) as archive:
        assert json.loads(archive.read("stage_state.json"))["stage_complete"] is True
    incomplete = _stage_c_resume(tmp_path / "incomplete")[0]
    log = tmp_path / "colab.log"
    log.write_text("incomplete\n", encoding="utf-8")
    sanitized, contract = _sanitized_fixture(tmp_path / "identity")

    with pytest.raises(ColabRecoveryError, match="complete"):
        write_delivery_bundle(
            review=review,
            resume=incomplete,
            log=log,
            identity=_runtime_identity(sanitized, contract),
            run_uuid="run-a",
            destination=tmp_path / "delivery.zip",
        )


def test_supervisor_interrupt_requests_epoch_boundary_and_exports_snapshot(
    tmp_path: Path,
) -> None:
    sanitized, contract = _sanitized_fixture(tmp_path / "identity")
    identity = _runtime_identity(sanitized, contract)
    work_root = tmp_path / "work"
    training_dir = (
        work_root
        / "outputs"
        / "stage_C"
        / "jobs"
        / identity.target_candidate_id
    )
    stop_marker = work_root / "stop-after-epoch.request"
    fixture_code = "\n".join(
        [
            "import json, pathlib, time",
            f"root = pathlib.Path({str(training_dir)!r})",
            "root.mkdir(parents=True, exist_ok=True)",
            "(root / 'checkpoint.pt').write_bytes(b'checkpoint')",
            "(root / 'best_checkpoint.pt').write_bytes(b'best')",
            "meta = "
            + repr(
                {
                    "candidate_id": identity.target_candidate_id,
                    "epoch": 0,
                    "checkpoint": "checkpoint.pt",
                    "checkpoint_binding": {
                        "config_sha256": identity.campaign_config_sha256,
                        "cache_sha256": identity.cache_sha256,
                        "training_source_sha256": identity.training_source_sha256,
                    },
                }
            ),
            "(root / 'checkpoint_meta.json').write_text(json.dumps(meta, sort_keys=True, separators=(',', ':')))",
            "print('EPOCH_CHECKPOINTED epoch=0', flush=True)",
            f"marker = pathlib.Path({str(stop_marker)!r})",
            "while not marker.exists(): time.sleep(0.01)",
            "print('TRAINING_STOP_AFTER_EPOCH_REQUESTED completed_epoch=0', flush=True)",
        ]
    )

    result = supervise_campaign(
        command=[sys.executable, "-c", fixture_code],
        work_root=work_root,
        identity=identity,
        snapshot_interval_seconds=0,
        interrupt_after_seconds=0.15,
        interrupt_grace_seconds=3,
        poll_seconds=0.02,
    )

    assert result.interrupted is True
    assert result.latest_snapshot is not None
    assert result.latest_snapshot.epoch == 0
    assert stop_marker.is_file()
    assert process_identity(result.receipt.pid) is None
    assert "EPOCH_CHECKPOINTED" in result.log_path.read_text(encoding="utf-8")


def _handoff_contract(
    data_dir: Path, resume: Path
) -> dict[str, object]:
    members = {
        name: {
            "size": (data_dir / name).stat().st_size,
            "sha256": file_sha256(data_dir / name),
        }
        for name in _MEMBERS
    }
    return {
        "schema_version": 1,
        "base_resume": {
            "filename": "tabm_search_stage_C_resume_bundle.zip",
            "sha256": file_sha256(resume),
            "version": "C",
        },
        "data_archive": {
            "filename": "lg-aimers-9th-data.zip",
            "max_member_count": 4,
            "max_uncompressed_bytes": sum(
                int(evidence["size"]) for evidence in members.values()
            ),
            "members": members,
        },
    }


def _fixture_official_data(root: Path) -> Path:
    root.mkdir(parents=True)
    for name, value in _MEMBERS.items():
        (root / name).write_bytes(value)
    return root


def test_prepare_handoff_is_deterministic(tmp_path: Path) -> None:
    data_dir = _fixture_official_data(tmp_path / "data")
    resume = _stage_c_resume(tmp_path / "resume")[0]
    contract = _handoff_contract(data_dir, resume)

    first = prepare_handoff(
        data_dir, resume, tmp_path / "first", contract=contract
    )
    second = prepare_handoff(
        data_dir, resume, tmp_path / "second", contract=contract
    )

    assert file_sha256(first.data_zip) == file_sha256(second.data_zip)
    assert json.loads(first.manifest.read_text()) == json.loads(
        second.manifest.read_text()
    )
    assert first.resume_zip.read_bytes() == resume.read_bytes()
    assert first.cell.name == "COLAB_STAGE_C_RECOVERY_CELL.py"


def test_prepare_handoff_refuses_changed_source_file(tmp_path: Path) -> None:
    data_dir = _fixture_official_data(tmp_path / "data")
    resume = _stage_c_resume(tmp_path / "resume")[0]
    contract = _handoff_contract(data_dir, resume)
    (data_dir / "train.csv").write_bytes(b"changed")

    with pytest.raises(ColabRecoveryError, match="train.csv"):
        prepare_handoff(
            data_dir, resume, tmp_path / "out", contract=contract
        )
