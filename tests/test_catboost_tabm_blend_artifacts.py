from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from experiments.catboost_tabm_blend.artifacts import (
    BlendArtifactError,
    verify_resume_bundle,
    verify_review_bundle,
    write_bundles,
)
from experiments.catboost_tabm_blend.contracts import DEFAULT_CONTRACT, build_jobs, load_contract


def _bindings() -> dict[str, str]:
    return {
        "contract_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "train_sha256": "d" * 64,
        "history_sha256": "e" * 64,
        "stage_c_delivery_sha256": "f" * 64,
        "stage_c_review_sha256": "1" * 64,
        "stage_c_state_sha256": "2" * 64,
    }


def _job_dir(root: Path, job_id: str) -> Path:
    directory = root / job_id
    directory.mkdir(parents=True)
    prediction = pd.DataFrame(
        {
            "row_id": ["A", "B"],
            "target": [0, 1],
            "probability": [0.25, 0.75],
            "game_type": ["R", "R"],
            "game_month": [4, 4],
            "pitcher_id_known": ["known", "oov"],
            "batter_id_known": ["known", "known"],
        }
    )
    prediction.to_csv(directory / "predictions.csv", index=False)
    brier = 0.0625
    (directory / "job.json").write_text(json.dumps({"job_id": job_id}))
    (directory / "metrics.json").write_text(json.dumps({"job_id": job_id, "brier": brier}))
    (directory / "model.cbm").write_bytes(b"model-" + job_id.encode())
    (directory / "worker.log").write_text("CATBOOST_JOB_END\n")
    (directory / "worker_result.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "completed",
                "brier": brier,
                "model": "model.cbm",
                "predictions": "predictions.csv",
                "snapshot": None,
            }
        )
    )
    return directory


def _state(path: Path, job_ids: list[str], *, complete: bool) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "catboost_tabm_blend_v1",
                "stage_complete": complete,
                "completed_job_ids": job_ids,
                "active_job_id": None,
                "bindings": _bindings(),
            },
            sort_keys=True,
        )
    )
    return path


def test_completed_campaign_writes_and_verifies_exact_bundles(tmp_path: Path) -> None:
    job_ids = [job.job_id for job in build_jobs(load_contract())]
    jobs = {job_id: _job_dir(tmp_path / "jobs", job_id) for job_id in job_ids}
    state = _state(tmp_path / "stage_state.json", job_ids, complete=True)
    log = tmp_path / "campaign.log"
    log.write_text("BLEND_DECISION selected_tabm_weight=0.8\n")
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"selected_tabm_weight": 0.8}))

    bundles = write_bundles(
        output_dir=tmp_path / "bundles",
        contract_path=DEFAULT_CONTRACT,
        bindings=_bindings(),
        stage_state_path=state,
        campaign_log_path=log,
        job_directories=jobs,
        decision_path=decision,
    )

    assert bundles.review is not None
    verify_review_bundle(bundles.review, expected_bindings=_bindings())
    verified = verify_resume_bundle(bundles.resume, expected_bindings=_bindings())
    assert verified.stage_complete is True
    assert verified.completed_job_ids == tuple(job_ids)
    with ZipFile(bundles.review) as archive:
        assert set(archive.namelist()) == {
            "contract/contract.json",
            "decision/blend_decision.json",
            "logs/campaign.log",
            "metrics/fold_results.json",
            f"predictions/{job_ids[0]}.csv",
            f"predictions/{job_ids[1]}.csv",
            "state/stage_state.json",
            "manifest.json",
        }


def test_incomplete_campaign_has_resume_only(tmp_path: Path) -> None:
    job_id = build_jobs(load_contract())[0].job_id
    job = _job_dir(tmp_path / "jobs", job_id)
    state = _state(tmp_path / "stage_state.json", [job_id], complete=False)
    log = tmp_path / "campaign.log"
    log.write_text("CATBOOST_FOLD_RESULT\n")

    bundles = write_bundles(
        output_dir=tmp_path / "bundles",
        contract_path=DEFAULT_CONTRACT,
        bindings=_bindings(),
        stage_state_path=state,
        campaign_log_path=log,
        job_directories={job_id: job},
        decision_path=None,
    )

    assert bundles.review is None
    verified = verify_resume_bundle(bundles.resume, expected_bindings=_bindings())
    assert verified.stage_complete is False
    assert verified.completed_job_ids == (job_id,)


def test_active_snapshot_is_resumable_without_completed_prediction(tmp_path: Path) -> None:
    job_id = build_jobs(load_contract())[0].job_id
    active = tmp_path / "jobs" / job_id
    active.mkdir(parents=True)
    (active / "job.json").write_text(json.dumps({"job_id": job_id}))
    (active / "worker.log").write_text("CATBOOST_PROGRESS iteration=50\n")
    (active / "experiment.cbsnapshot").write_bytes(b"active-snapshot")
    state = tmp_path / "stage_state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "catboost_tabm_blend_v1",
                "stage_complete": False,
                "completed_job_ids": [],
                "active_job_id": job_id,
                "bindings": _bindings(),
            },
            sort_keys=True,
        )
    )
    log = tmp_path / "campaign.log"
    log.write_text("CATBOOST_PROGRESS\n")

    bundles = write_bundles(
        output_dir=tmp_path / "bundles",
        contract_path=DEFAULT_CONTRACT,
        bindings=_bindings(),
        stage_state_path=state,
        campaign_log_path=log,
        job_directories={job_id: active},
        decision_path=None,
    )
    verified = verify_resume_bundle(bundles.resume, expected_bindings=_bindings())

    assert verified.completed_job_ids == ()
    assert verified.active_job_id == job_id
    with ZipFile(bundles.resume) as archive:
        names = set(archive.namelist())
    assert f"jobs/{job_id}/experiment.cbsnapshot" in names
    assert f"jobs/{job_id}/predictions.csv" not in names


def test_tampered_resume_is_rejected(tmp_path: Path) -> None:
    job_id = build_jobs(load_contract())[0].job_id
    job = _job_dir(tmp_path / "jobs", job_id)
    state = _state(tmp_path / "stage_state.json", [job_id], complete=False)
    log = tmp_path / "campaign.log"
    log.write_text("log")
    bundle = write_bundles(
        output_dir=tmp_path / "bundles",
        contract_path=DEFAULT_CONTRACT,
        bindings=_bindings(),
        stage_state_path=state,
        campaign_log_path=log,
        job_directories={job_id: job},
        decision_path=None,
    ).resume
    with ZipFile(bundle) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[f"jobs/{job_id}/worker.log"] = b"tampered"
    bundle.write_bytes(_deterministic_zip(members))

    with pytest.raises(BlendArtifactError):
        verify_resume_bundle(bundle, expected_bindings=_bindings())


def test_failed_reverification_preserves_prior_bundle(tmp_path: Path, monkeypatch) -> None:
    import experiments.catboost_tabm_blend.artifacts as artifacts

    job_id = build_jobs(load_contract())[0].job_id
    job = _job_dir(tmp_path / "jobs", job_id)
    state = _state(tmp_path / "stage_state.json", [job_id], complete=False)
    log = tmp_path / "campaign.log"
    log.write_text("log")
    arguments = dict(
        output_dir=tmp_path / "bundles",
        contract_path=DEFAULT_CONTRACT,
        bindings=_bindings(),
        stage_state_path=state,
        campaign_log_path=log,
        job_directories={job_id: job},
        decision_path=None,
    )
    prior = write_bundles(**arguments).resume
    prior_bytes = prior.read_bytes()
    (job / "worker.log").write_text("changed")
    monkeypatch.setattr(
        artifacts,
        "verify_resume_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(BlendArtifactError("reject")),
    )

    with pytest.raises(BlendArtifactError, match="reject"):
        artifacts.write_bundles(**arguments)

    assert prior.read_bytes() == prior_bytes


def _deterministic_zip(members: dict[str, bytes]) -> bytes:
    from io import BytesIO
    from zipfile import ZIP_DEFLATED, ZipInfo

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(members.items()):
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return buffer.getvalue()
