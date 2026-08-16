from __future__ import annotations

import json
import os
import shutil
import stat
import threading
import time
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.tabm_campaign import row_feature_colab as colab
from experiments.tabm_campaign import row_feature_proxy as proxy
from experiments.tabm_campaign import worker
from experiments.tabm_campaign.artifacts import StageEvidence, write_stage_bundles
from experiments.tabm_campaign.row_feature_contracts import (
    DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
    load_row_feature_proxy_contract,
    row_feature_contract_sha256,
)
from experiments.tabm_campaign.row_feature_runtime import CampaignJobResult


TRAIN = b"row_id,year,control_success\nr1,2023,1\n"
HISTORY = b"pitcher_id,season\np1,2023\n"
REAL_CONTRACT_BYTES = DEFAULT_ROW_FEATURE_PROXY_CONTRACT.read_bytes()
REAL_CONTRACT_SHA = sha256(REAL_CONTRACT_BYTES).hexdigest()


def _write_inputs(root: Path) -> None:
    root.mkdir()
    (root / "train.csv").write_bytes(TRAIN)
    (root / "trackman_history.csv").write_bytes(HISTORY)
    (root / "test.csv").write_text("must,not,ship\n", encoding="utf-8")
    (root / "sample_submission.csv").write_text("must,not,ship\n", encoding="utf-8")


def _zip(path: Path, members: list[tuple[str | ZipInfo, bytes]]) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members:
            archive.writestr(name, value)
    return path


def _manifest() -> bytes:
    return colab.canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "tabm_row_feature_input",
            "stage": "P",
            "campaign_config_sha256": "c" * 64,
            "members": {
                "trackman_history.csv": {
                    "size": len(HISTORY),
                    "sha256": sha256(HISTORY).hexdigest(),
                },
                "train.csv": {
                    "size": len(TRAIN),
                    "sha256": sha256(TRAIN).hexdigest(),
                },
            },
        }
    )


def test_prepare_input_is_exact_deterministic_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_inputs(data)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    colab.prepare_input_archive(
        data,
        first,
        expected_train_sha256=sha256(TRAIN).hexdigest(),
        campaign_config_sha256="c" * 64,
    )
    colab.prepare_input_archive(
        data,
        second,
        expected_train_sha256=sha256(TRAIN).hexdigest(),
        campaign_config_sha256="c" * 64,
    )

    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        assert archive.namelist() == [
            "input_manifest.json",
            "trackman_history.csv",
            "train.csv",
        ]
        manifest = json.loads(archive.read("input_manifest.json"))
    assert set(manifest["members"]) == {"train.csv", "trackman_history.csv"}
    assert manifest["members"]["train.csv"]["sha256"] == sha256(TRAIN).hexdigest()
    assert manifest["members"]["trackman_history.csv"]["size"] == len(HISTORY)
    with pytest.raises(colab.RowFeatureColabError, match="exists"):
        colab.prepare_input_archive(
            data,
            first,
            expected_train_sha256=sha256(TRAIN).hexdigest(),
            campaign_config_sha256="c" * 64,
        )


def test_prepare_input_rejects_wrong_hash_duplicates_nesting_and_symlinks(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_inputs(data)
    with pytest.raises(colab.RowFeatureColabError, match="SHA-256"):
        colab.prepare_input_archive(
            data,
            tmp_path / "wrong.zip",
            expected_train_sha256="0" * 64,
            campaign_config_sha256="c" * 64,
        )
    nested = data / "nested"
    nested.mkdir()
    (nested / "train.csv").write_bytes(TRAIN)
    with pytest.raises(colab.RowFeatureColabError, match="exactly one|top level"):
        colab.prepare_input_archive(
            data,
            tmp_path / "duplicate.zip",
            expected_train_sha256=sha256(TRAIN).hexdigest(),
            campaign_config_sha256="c" * 64,
        )
    (nested / "train.csv").unlink()
    (nested / "trackman_history.csv").write_bytes(HISTORY)
    with pytest.raises(colab.RowFeatureColabError, match="exactly one|top level"):
        colab.prepare_input_archive(
            data,
            tmp_path / "nested.zip",
            expected_train_sha256=sha256(TRAIN).hexdigest(),
            campaign_config_sha256="c" * 64,
        )
    (nested / "trackman_history.csv").unlink()
    os.symlink(data / "train.csv", nested / "alias.csv")
    with pytest.raises(colab.RowFeatureColabError, match="symlink"):
        colab.prepare_input_archive(
            data,
            tmp_path / "symlink.zip",
            expected_train_sha256=sha256(TRAIN).hexdigest(),
            campaign_config_sha256="c" * 64,
        )


def test_prepare_input_rejects_dangling_output_symlink_without_replace(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_inputs(data)
    output = tmp_path / "prepared.zip"
    output.symlink_to(tmp_path / "missing-target.zip")

    with pytest.raises(colab.RowFeatureColabError, match="exists"):
        colab.prepare_input_archive(
            data,
            output,
            expected_train_sha256=sha256(TRAIN).hexdigest(),
            campaign_config_sha256="c" * 64,
        )
    assert output.is_symlink()


def test_prepare_input_never_overwrites_a_concurrently_created_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    _write_inputs(data)
    output = tmp_path / "prepared.zip"
    real_link = os.link

    def concurrent_create(source, destination) -> None:
        Path(destination).write_bytes(b"concurrent owner")
        real_link(source, destination)

    monkeypatch.setattr(colab.os, "link", concurrent_create)
    with pytest.raises(colab.RowFeatureColabError, match="exists"):
        colab.prepare_input_archive(
            data,
            output,
            expected_train_sha256=sha256(TRAIN).hexdigest(),
            campaign_config_sha256="c" * 64,
        )
    assert output.read_bytes() == b"concurrent owner"


def test_prepare_input_replace_explicitly_replaces_a_dangling_symlink(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_inputs(data)
    output = tmp_path / "prepared.zip"
    output.symlink_to(tmp_path / "missing-target.zip")

    colab.prepare_input_archive(
        data,
        output,
        expected_train_sha256=sha256(TRAIN).hexdigest(),
        campaign_config_sha256="c" * 64,
        replace=True,
    )
    assert output.is_file()
    assert not output.is_symlink()


def test_data_archive_verifier_extracts_exact_members_and_binds_manifest(
    tmp_path: Path,
) -> None:
    archive = _zip(
        tmp_path / "data.zip",
        [
            ("input_manifest.json", _manifest()),
            ("trackman_history.csv", HISTORY),
            ("train.csv", TRAIN),
        ],
    )
    verified = colab.verify_and_extract_input_archive(
        archive,
        tmp_path / "published",
        expected_train_sha256=sha256(TRAIN).hexdigest(),
        campaign_config_sha256="c" * 64,
    )

    assert verified.input_manifest_sha256 == sha256(_manifest()).hexdigest()
    assert {path.name for path in verified.data_dir.iterdir()} == {
        "input_manifest.json",
        "train.csv",
        "trackman_history.csv",
    }


@pytest.mark.parametrize(
    "kind",
    ["wrong_hash", "extra", "traversal", "symlink", "duplicate", "bomb", "truncated"],
)
def test_data_archive_verifier_rejects_unsafe_archives(
    tmp_path: Path, kind: str
) -> None:
    members: list[tuple[str | ZipInfo, bytes]] = [
        ("input_manifest.json", _manifest()),
        ("trackman_history.csv", HISTORY),
        ("train.csv", TRAIN),
    ]
    if kind == "wrong_hash":
        manifest = json.loads(_manifest())
        manifest["members"]["train.csv"]["sha256"] = "0" * 64
        members[0] = ("input_manifest.json", colab.canonical_json(manifest))
    elif kind == "extra":
        members.append(("test.csv", b"forbidden"))
    elif kind == "traversal":
        members.append(("../escape", b"bad"))
    elif kind == "symlink":
        link = ZipInfo("train.csv")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        members[-1] = (link, b"target")
    elif kind == "duplicate":
        members.append(("train.csv", TRAIN))
    elif kind == "bomb":
        bomb = b"0" * (1024 * 1024)
        manifest = json.loads(_manifest())
        manifest["members"]["trackman_history.csv"] = {
            "size": len(bomb),
            "sha256": sha256(bomb).hexdigest(),
        }
        members[0] = ("input_manifest.json", colab.canonical_json(manifest))
        members[1] = ("trackman_history.csv", bomb)
    if kind == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive = _zip(tmp_path / f"{kind}.zip", members)
    else:
        archive = _zip(tmp_path / f"{kind}.zip", members)
    if kind == "truncated":
        archive.write_bytes(archive.read_bytes()[:-12])

    with pytest.raises(colab.RowFeatureColabError):
        colab.verify_and_extract_input_archive(
            archive,
            tmp_path / "published",
            expected_train_sha256=sha256(TRAIN).hexdigest(),
            campaign_config_sha256="c" * 64,
        )


def _stage_bundle(
    root: Path,
    *,
    version: str = "P",
    contract_sha256: str = REAL_CONTRACT_SHA,
    train_sha256: str | None = None,
    history_sha256: str | None = None,
    input_manifest_sha256: str = "e" * 64,
    code_sha256: str = "d" * 64,
    drop_state_key: str | None = None,
    corrupt_log: bool = False,
):
    contract = load_row_feature_proxy_contract()
    jobs = proxy.build_proxy_jobs(contract)
    decision = proxy._decision(jobs, {}, contract)
    state = proxy._state_payload(
        contract_sha=contract_sha256,
        train_sha=train_sha256 or sha256(TRAIN).hexdigest(),
        history_sha=history_sha256 or sha256(HISTORY).hexdigest(),
        input_manifest_sha=input_manifest_sha256,
        code_sha=code_sha256,
        jobs=jobs,
        rows={},
        prior_manifest_sha=None,
        decision=decision,
    )
    if drop_state_key is not None:
        state.pop(drop_state_key)
    config = (
        REAL_CONTRACT_BYTES
        if contract_sha256 == REAL_CONTRACT_SHA
        else b"deliberately different contract fixture"
    )
    payload_root = root / "payload"
    payload_root.mkdir(parents=True)
    review, resume = proxy._bundle_members(
        output_dir=payload_root,
        config_bytes=config,
        state=state,
        rows={},
        decision=decision,
    )
    if corrupt_log:
        resume["logs/stage.log"] = b"manifest-valid but semantically false\n"
    return write_stage_bundles(
        root,
        StageEvidence(
            version,
            contract_sha256,
            None,
            review,
            resume,
        ),
        bundle_prefix="tabm_row_feature_stage",
    )


def _verified_input(tmp_path: Path) -> colab.VerifiedInput:
    data = tmp_path / "verified-data"
    data.mkdir()
    return colab.VerifiedInput(
        data,
        "e" * 64,
        sha256(TRAIN).hexdigest(),
        sha256(HISTORY).hexdigest(),
    )


def test_resume_requires_exact_stage_contract_code_and_input_bindings(
    tmp_path: Path,
) -> None:
    verified_input = _verified_input(tmp_path)
    valid = _stage_bundle(tmp_path / "valid").resume
    assert valid is not None
    verified = colab.verify_stage_p_resume(
        valid,
        expected_contract_sha256=REAL_CONTRACT_SHA,
        expected_code_sha256="d" * 64,
        verified_input=verified_input,
    )
    assert verified.version == "P"

    variants = (
        _stage_bundle(tmp_path / "stage-a", version="A").resume,
        _stage_bundle(tmp_path / "contract", contract_sha256="a" * 64).resume,
        _stage_bundle(tmp_path / "code", code_sha256="a" * 64).resume,
        _stage_bundle(tmp_path / "input", train_sha256="a" * 64).resume,
        _stage_bundle(tmp_path / "history", history_sha256="a" * 64).resume,
        _stage_bundle(
            tmp_path / "manifest", input_manifest_sha256="a" * 64
        ).resume,
        _stage_bundle(tmp_path / "schema", drop_state_key="jobs").resume,
        _stage_bundle(tmp_path / "log", corrupt_log=True).resume,
    )
    for resume in variants:
        assert resume is not None
        with pytest.raises(colab.RowFeatureColabError):
            colab.verify_stage_p_resume(
                resume,
                expected_contract_sha256=REAL_CONTRACT_SHA,
                expected_code_sha256="d" * 64,
                verified_input=verified_input,
            )


def test_verified_uploaded_resume_is_registered_before_a_later_setup_error(
    tmp_path: Path,
) -> None:
    verified_input = _verified_input(tmp_path)
    resume = _stage_bundle(tmp_path / "uploaded").resume
    assert resume is not None
    latest: list[Path | None] = [None]
    downloads: list[Path] = []

    def remember(path: Path) -> None:
        latest[0] = path

    colab.register_verified_uploaded_resume(
        resume,
        expected_contract_sha256=REAL_CONTRACT_SHA,
        expected_code_sha256="d" * 64,
        verified_input=verified_input,
        on_verified_resume=remember,
    )
    try:
        raise RuntimeError("simulated dependency failure")
    except RuntimeError:
        if latest[0] is not None:
            downloads.append(latest[0])
    assert downloads == [resume]


def test_untrusted_uploaded_resume_is_not_registered(tmp_path: Path) -> None:
    verified_input = _verified_input(tmp_path)
    resume = _stage_bundle(tmp_path / "uploaded", code_sha256="a" * 64).resume
    assert resume is not None
    latest: list[Path] = []

    with pytest.raises(colab.RowFeatureColabError, match="code_sha256"):
        colab.register_verified_uploaded_resume(
            resume,
            expected_contract_sha256=REAL_CONTRACT_SHA,
            expected_code_sha256="d" * 64,
            verified_input=verified_input,
            on_verified_resume=latest.append,
        )
    assert latest == []


def test_delivery_contains_only_recursively_verified_evidence(tmp_path: Path) -> None:
    bundles = _stage_bundle(tmp_path / "bundles")
    assert bundles.resume is not None
    log = tmp_path / "row_feature_proxy.log"
    log.write_text("JOB_START fixture\n", encoding="utf-8")
    delivery = colab.build_delivery(
        review_bundle=bundles.review,
        resume_bundle=bundles.resume,
        log_path=log,
        output_path=tmp_path / "tabm_row_feature_stage_P_delivery.zip",
        input_manifest_sha256="e" * 64,
        embedded_runtime_sha256="f" * 64,
        campaign_config_sha256=REAL_CONTRACT_SHA,
        code_sha256="d" * 64,
    )

    verified = colab.verify_delivery(
        delivery,
        input_manifest_sha256="e" * 64,
        embedded_runtime_sha256="f" * 64,
        campaign_config_sha256=REAL_CONTRACT_SHA,
        code_sha256="d" * 64,
    )
    assert verified.path == delivery
    with ZipFile(delivery) as archive:
        assert set(archive.namelist()) == {
            "tabm_row_feature_stage_P_review_bundle.zip",
            "tabm_row_feature_stage_P_resume_bundle.zip",
            "row_feature_proxy.log",
            "delivery_manifest.json",
        }


def test_delivery_rejects_tampered_nested_bundle(tmp_path: Path) -> None:
    bundles = _stage_bundle(tmp_path / "bundles")
    assert bundles.resume is not None
    log = tmp_path / "row_feature_proxy.log"
    log.write_bytes(b"ok\n")
    delivery = colab.build_delivery(
        review_bundle=bundles.review,
        resume_bundle=bundles.resume,
        log_path=log,
        output_path=tmp_path / "delivery.zip",
        input_manifest_sha256="e" * 64,
        embedded_runtime_sha256="f" * 64,
        campaign_config_sha256=REAL_CONTRACT_SHA,
        code_sha256="d" * 64,
    )
    with ZipFile(delivery) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["tabm_row_feature_stage_P_resume_bundle.zip"] += b"tamper"
    _zip(delivery, list(members.items()))

    with pytest.raises(colab.RowFeatureColabError):
        colab.verify_delivery(
            delivery,
            input_manifest_sha256="e" * 64,
            embedded_runtime_sha256="f" * 64,
            campaign_config_sha256=REAL_CONTRACT_SHA,
            code_sha256="d" * 64,
        )


def test_delivery_streaming_and_recursive_verification_honor_deadline(
    tmp_path: Path,
) -> None:
    bundles = _stage_bundle(tmp_path / "bundles")
    assert bundles.resume is not None
    log = tmp_path / "row_feature_proxy.log"
    log.write_bytes(b"TRAINING_PROGRESS\n" * 100_000)
    output = tmp_path / "delivery.zip"
    calls = [0]

    def expiring_deadline() -> None:
        calls[0] += 1
        if calls[0] >= 8:
            raise TimeoutError("fixture deadline expired")

    with pytest.raises(TimeoutError, match="deadline"):
        colab.build_delivery(
            review_bundle=bundles.review,
            resume_bundle=bundles.resume,
            log_path=log,
            output_path=output,
            input_manifest_sha256="e" * 64,
            embedded_runtime_sha256="f" * 64,
            campaign_config_sha256=REAL_CONTRACT_SHA,
            code_sha256="d" * 64,
            check_deadline=expiring_deadline,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".delivery.zip-*"))

    delivery = colab.build_delivery(
        review_bundle=bundles.review,
        resume_bundle=bundles.resume,
        log_path=log,
        output_path=output,
        input_manifest_sha256="e" * 64,
        embedded_runtime_sha256="f" * 64,
        campaign_config_sha256=REAL_CONTRACT_SHA,
        code_sha256="d" * 64,
    )
    with pytest.raises(TimeoutError, match="deadline"):
        colab.verify_delivery(
            delivery,
            input_manifest_sha256="e" * 64,
            embedded_runtime_sha256="f" * 64,
            campaign_config_sha256=REAL_CONTRACT_SHA,
            code_sha256="d" * 64,
            check_deadline=lambda: (_ for _ in ()).throw(
                TimeoutError("fixture deadline expired")
            ),
            temporary_dir=tmp_path / "run-root",
        )
    assert not list((tmp_path / "run-root").glob("row-feature-delivery-*"))


def test_active_row_validation_is_deadline_aware_and_isolates_checkpoint_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_row_feature_proxy_contract()
    job = proxy.build_proxy_jobs(contract)[0]
    snapshot_job_dir = tmp_path / job.candidate_id
    snapshot_job_dir.mkdir()
    (snapshot_job_dir / "checkpoint_meta.json").write_bytes(
        colab.canonical_json(
            {
                "candidate_id": job.candidate_id,
                "epoch": 3,
                "checkpoint": "checkpoint.pt",
                "adapter_state": None,
                "checkpoint_binding": {
                    "config_sha256": worker._job_sha(job),
                    "cache_sha256": "a" * 64,
                    "training_source_sha256": worker._training_source_sha256(),
                },
            }
        )
    )
    (snapshot_job_dir / "progress.jsonl").write_text(
        json.dumps(
            {
                "event": "EPOCH_CHECKPOINTED",
                "candidate_id": job.candidate_id,
                "epoch": 3,
                "best_epoch": 2,
                "best_brier": 0.04,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot_job_dir / "checkpoint.pt").write_bytes(b"not-a-torch-payload")
    (snapshot_job_dir / "best_checkpoint.pt").write_bytes(b"not-a-torch-payload")
    observed: dict[str, object] = {}

    def isolated(snapshot_dir, isolated_job, result, *, best_epoch, check_deadline):
        observed["isolated_path"] = snapshot_dir
        observed["isolated_job"] = isolated_job
        observed["isolated_best_epoch"] = best_epoch
        check_deadline()

    def validate(*args, **kwargs):
        observed.update(kwargs)
        return {"accepted": True}

    monkeypatch.setattr(proxy, "_validate_runtime_result", validate)
    monkeypatch.setattr(
        colab, "_validate_active_checkpoint_payload_isolated", isolated
    )
    checks = [0]

    result = colab._validated_active_row(
        snapshot_job_dir,
        job,
        check_deadline=lambda: checks.__setitem__(0, checks[0] + 1),
    )

    assert result == {"accepted": True}
    assert observed["validate_checkpoint_payload"] is False
    assert callable(observed["check_deadline"])
    assert observed["isolated_job"] == job
    assert observed["isolated_best_epoch"] == 2
    assert checks[0] > 0


def test_isolated_checkpoint_validator_is_killed_at_deadline(tmp_path: Path) -> None:
    job = proxy.build_proxy_jobs(load_row_feature_proxy_contract())[0]
    result = CampaignJobResult(
        candidate_id=job.candidate_id,
        status="inconclusive",
        brier=0.04,
        best_epoch=0,
        completed_epochs=1,
        checkpoint=tmp_path / "checkpoint.pt",
        predictions_path=None,
        resource_evidence={"cache_digest": "a" * 64},
        failure="active_checkpoint_snapshot",
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="deadline"):
        colab._validate_active_checkpoint_payload_isolated(
            tmp_path,
            job,
            result,
            best_epoch=0,
            check_deadline=lambda: (_ for _ in ()).throw(
                TimeoutError("fixture deadline expired")
            ),
        )

    assert time.monotonic() - started < 2.0


def test_active_checkpoint_snapshot_restores_the_same_completed_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_row_feature_proxy_contract()
    contract_sha = row_feature_contract_sha256()
    code_sha = proxy._code_sha256()
    jobs = proxy.build_proxy_jobs(contract)
    active_job = jobs[0]
    live = tmp_path / "live"
    job_dir = live / "jobs" / active_job.candidate_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_bytes(colab.canonical_json(worker._job_payload(active_job)))
    (job_dir / "checkpoint.pt").write_bytes(b"atomic-checkpoint-epoch-3")
    (job_dir / "best_checkpoint.pt").write_bytes(b"atomic-best-epoch-2")
    (job_dir / "worker.log").write_text("TRAINING_PROGRESS fixture\n", encoding="utf-8")
    (job_dir / "progress.jsonl").write_text(
        json.dumps({"event": "EPOCH_CHECKPOINTED", "epoch": 3}) + "\n",
        encoding="utf-8",
    )
    (job_dir / "checkpoint_meta.json").write_bytes(
        colab.canonical_json(
            {
                "candidate_id": active_job.candidate_id,
                "epoch": 3,
                "checkpoint": "checkpoint.pt",
                "adapter_state": None,
                "checkpoint_binding": {
                    "config_sha256": worker._job_sha(active_job),
                    "cache_sha256": "a" * 64,
                    "training_source_sha256": worker._training_source_sha256(),
                },
            }
        )
    )
    decision = proxy._decision(jobs, {}, contract)
    state = proxy._state_payload(
        contract_sha=contract_sha,
        train_sha=contract.official_train_sha256,
        history_sha="b" * 64,
        input_manifest_sha="e" * 64,
        code_sha=code_sha,
        jobs=jobs,
        rows={},
        prior_manifest_sha=None,
        decision=decision,
    )
    proxy._write_local_state(live / "stage_state.json", state)

    def accept_fixture(
        snapshot_job_dir: Path, job, check_deadline=None
    ) -> dict[str, object]:
        artifacts = {
            path.name: {
                "path": f"jobs/{job.candidate_id}/{path.name}",
                "sha256": colab.file_sha256(path),
            }
            for path in snapshot_job_dir.iterdir()
        }
        return {
            "candidate_id": job.candidate_id,
            "feature_bundle": job.feature_bundle,
            "seed": job.seed,
            "job_sha256": worker._job_sha(job),
            "status": "inconclusive",
            "disposition": "inconclusive",
            "brier": 0.04,
            "best_epoch": 2,
            "completed_epochs": 4,
            "checkpoint_name": "checkpoint.pt",
            "predictions_name": None,
            "resource_evidence": {"cache_digest": "a" * 64},
            "failure": "active_checkpoint_snapshot",
            "artifacts": artifacts,
        }

    monkeypatch.setattr(colab, "_validated_active_row", accept_fixture)
    snapshot = colab.publish_active_checkpoint_snapshot(
        live_output_dir=live,
        snapshot_dir=tmp_path / "snapshots",
        active_job=active_job,
        contract=contract,
        contract_path=DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
        contract_sha256=contract_sha,
        train_sha256=contract.official_train_sha256,
        history_sha256="b" * 64,
        input_manifest_sha256="e" * 64,
        code_sha256=code_sha,
        sequence=7,
    )
    assert snapshot.epoch == 3

    restored = tmp_path / "restored"
    restored_state, restored_rows, _ = proxy._restore_resume(
        snapshot.path,
        output_dir=restored,
        config_bytes=DEFAULT_ROW_FEATURE_PROXY_CONTRACT.read_bytes(),
        contract_sha=contract_sha,
        train_sha=contract.official_train_sha256,
        history_sha="b" * 64,
        input_manifest_sha="e" * 64,
        code_sha=code_sha,
        contract=contract,
        jobs=jobs,
    )
    assert restored_state["stage_complete"] is False
    assert restored_rows[active_job.candidate_id]["completed_epochs"] == 4
    assert (
        restored / "jobs" / active_job.candidate_id / "checkpoint.pt"
    ).read_bytes() == b"atomic-checkpoint-epoch-3"

    expired_dir = tmp_path / "expired-snapshots"
    deadline_ticks = [0]

    def advancing_deadline() -> None:
        deadline_ticks[0] += 1
        if deadline_ticks[0] >= 4:
            raise TimeoutError("fixture deadline expired")

    with pytest.raises(TimeoutError, match="deadline"):
        colab.publish_active_checkpoint_snapshot(
            live_output_dir=live,
            snapshot_dir=expired_dir,
            active_job=active_job,
            contract=contract,
            contract_path=DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
            contract_sha256=contract_sha,
            train_sha256=contract.official_train_sha256,
            history_sha256="b" * 64,
            input_manifest_sha256="e" * 64,
            code_sha256=code_sha,
            sequence=8,
            check_deadline=advancing_deadline,
        )
    assert not list(expired_dir.glob("*.zip"))
    assert not list(expired_dir.glob(".*"))


def test_active_snapshot_rejects_checkpoint_binding_before_publication(
    tmp_path: Path,
) -> None:
    contract = load_row_feature_proxy_contract()
    job = proxy.build_proxy_jobs(contract)[0]
    source = tmp_path / "job"
    source.mkdir()
    (source / "checkpoint.pt").write_bytes(b"checkpoint")
    (source / "best_checkpoint.pt").write_bytes(b"best")
    (source / "job.json").write_bytes(colab.canonical_json(worker._job_payload(job)))
    (source / "checkpoint_meta.json").write_bytes(
        colab.canonical_json(
            {
                "candidate_id": job.candidate_id,
                "epoch": 0,
                "checkpoint": "checkpoint.pt",
                "adapter_state": None,
                "checkpoint_binding": {
                    "config_sha256": "0" * 64,
                    "cache_sha256": "a" * 64,
                    "training_source_sha256": worker._training_source_sha256(),
                },
            }
        )
    )
    with pytest.raises(colab.RowFeatureColabError, match="binding"):
        colab._copy_active_candidate(source, tmp_path / "copy", job)


def test_slow_snapshot_shutdown_preserves_a_successful_delegate_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = proxy.build_proxy_jobs(load_row_feature_proxy_contract())[0]
    active_started = threading.Event()

    class Store:
        latest = object()
        snapshot_dir = tmp_path / "snapshots"

        def accept(self, snapshot):
            self.latest = snapshot
            return snapshot

    class Delegate:
        def run_jobs(self, version, jobs, output_dir, **kwargs):
            job_dir = output_dir / job.candidate_id
            job_dir.mkdir(parents=True)
            (job_dir / "checkpoint_meta.json").write_text(
                json.dumps({"epoch": 0}), encoding="utf-8"
            )
            assert active_started.wait(2)
            return ("delegate-success",)

    runtime = colab.SnapshottingCampaignRuntime(
        tmp_path / "data",
        live_output_dir=tmp_path / "live",
        store=Store(),
        contract=load_row_feature_proxy_contract(),
        contract_path=DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
        contract_sha256=REAL_CONTRACT_SHA,
        train_sha256="a" * 64,
        history_sha256="b" * 64,
        input_manifest_sha256="c" * 64,
        code_sha256="d" * 64,
        snapshot_interval_seconds=600,
        poll_seconds=0.01,
        session_deadline=time.time() + 15,
        delegate=Delegate(),
    )

    def slow_snapshot(_job):
        active_started.set()
        time.sleep(6)
        return colab.EmergencySnapshot(
            tmp_path / "slow.zip", "e" * 64, "f" * 64, job.candidate_id, 0
        )

    monkeypatch.setattr(runtime, "_active_snapshot", slow_snapshot)
    started = time.monotonic()
    result = runtime.run_jobs(
        "P", (job,), tmp_path / "jobs", gpu_count=1, job_deadline=time.time() + 5
    )
    assert result == ("delegate-success",)
    assert time.monotonic() - started >= 6


def test_active_snapshot_is_first_checkpoint_then_bounded_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = proxy.build_proxy_jobs(load_row_feature_proxy_contract())[0]
    accepted: list[int] = []
    first = threading.Event()
    second = threading.Event()

    class Store:
        latest = object()
        snapshot_dir = tmp_path / "snapshots"

        def accept(self, snapshot):
            accepted.append(snapshot.epoch)
            self.latest = snapshot
            (first if len(accepted) == 1 else second).set()
            return snapshot

    class Delegate:
        def run_jobs(self, version, jobs, output_dir, **kwargs):
            job_dir = output_dir / job.candidate_id
            job_dir.mkdir(parents=True)
            meta = job_dir / "checkpoint_meta.json"
            meta.write_text(json.dumps({"epoch": 0}), encoding="utf-8")
            assert first.wait(1)
            time.sleep(0.31)
            meta.write_text(json.dumps({"epoch": 1}), encoding="utf-8")
            assert second.wait(0.06)
            return ("done",)

    runtime = colab.SnapshottingCampaignRuntime(
        tmp_path / "data",
        live_output_dir=tmp_path / "live",
        store=Store(),
        contract=load_row_feature_proxy_contract(),
        contract_path=DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
        contract_sha256=REAL_CONTRACT_SHA,
        train_sha256="a" * 64,
        history_sha256="b" * 64,
        input_manifest_sha256="c" * 64,
        code_sha256="d" * 64,
        snapshot_interval_seconds=0.2,
        poll_seconds=0.01,
        session_deadline=time.time() + 5,
        delegate=Delegate(),
    )

    def snapshot(_job):
        epoch = json.loads(
            (tmp_path / "jobs" / job.candidate_id / "checkpoint_meta.json").read_text()
        )["epoch"]
        return colab.EmergencySnapshot(
            tmp_path / f"{epoch}.zip", "e" * 64, "f" * 64, job.candidate_id, epoch
        )

    monkeypatch.setattr(runtime, "_active_snapshot", snapshot)
    assert runtime.run_jobs(
        "P", (job,), tmp_path / "jobs", gpu_count=1, job_deadline=time.time() + 4
    ) == ("done",)
    assert accepted == [0, 1]


def test_snapshot_store_swaps_after_callback_and_removes_only_owned_old_zip(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    uploaded = _stage_bundle(tmp_path / "uploaded").resume
    assert uploaded is not None
    current = colab.EmergencySnapshot(
        uploaded,
        colab.file_sha256(uploaded),
        colab.verify_resume_bundle(uploaded).manifest_sha256,
        "uploaded",
        0,
    )
    callbacks: list[Path] = []
    fail = [True]

    def callback(snapshot) -> None:
        callbacks.append(snapshot.path)
        if fail[0]:
            raise RuntimeError("download failed")

    store = colab._VerifiedSnapshotStore(
        snapshot_dir=snapshot_dir, on_verified_snapshot=callback
    )
    store.latest = current

    def copied_snapshot(name: str) -> colab.EmergencySnapshot:
        path = snapshot_dir / name
        shutil.copyfile(uploaded, path)
        verified = colab.verify_resume_bundle(path)
        return colab.EmergencySnapshot(
            path, colab.file_sha256(path), verified.manifest_sha256, name, 1
        )

    rejected = copied_snapshot("rejected.zip")
    with pytest.raises(RuntimeError, match="download failed"):
        store.accept(rejected)
    assert store.latest is current
    assert uploaded.exists()
    assert not rejected.path.exists()

    fail[0] = False
    first_snapshot = copied_snapshot("first.zip")
    store.accept(first_snapshot)
    assert uploaded.exists()
    second_snapshot = copied_snapshot("second.zip")
    store.accept(second_snapshot)
    assert store.latest is second_snapshot
    assert not first_snapshot.path.exists()
    assert list(snapshot_dir.glob("*.zip")) == [second_snapshot.path]
    assert callbacks == [rejected.path, first_snapshot.path, second_snapshot.path]
