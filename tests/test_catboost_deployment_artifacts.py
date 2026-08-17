from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from experiments.catboost_deployment.artifacts import (
    DeploymentArtifactError,
    verify_deployment_review,
    verify_deployment_resume,
    verify_full_training_delivery,
    write_deployment_bundles,
)
from experiments.catboost_deployment.contracts import DEFAULT_CONTRACT
from experiments.catboost_deployment.metrics import CATBOOST_PREDICTION_COLUMNS
from experiments.catboost_deployment.state import serialize_feature_state
from experiments.catboost_preprocessing.features import fit_catboost_features


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bindings() -> dict[str, str]:
    names = (
        "contract_sha256",
        "code_sha256",
        "input_manifest_sha256",
        "train_sha256",
        "history_sha256",
        "stage_c_delivery_sha256",
        "stage_c_review_sha256",
        "stage_c_state_sha256",
        "source_blend_delivery_sha256",
        "source_blend_manifest_sha256",
        "source_decision_sha256",
    )
    bindings = {name: f"{index:x}" * 64 for index, name in enumerate(names, 1)}
    bindings["contract_sha256"] = sha256(DEFAULT_CONTRACT.read_bytes()).hexdigest()
    return bindings


def _preprocessing_bytes(preprocessing_frame: pd.DataFrame) -> bytes:
    state, _ = fit_catboost_features(
        preprocessing_frame, components=("hand_matchup",)
    )
    return serialize_feature_state(state)


def _alignment_job(
    root: Path, job_id: str, preprocessing_frame: pd.DataFrame
) -> Path:
    root.mkdir(parents=True)
    target = np.array([0.0, 1.0])
    data: dict[str, object] = {"row_id": [f"{job_id}-0", f"{job_id}-1"], "target": target}
    briers: dict[str, float] = {}
    for prefix in (4, 32, 64, 128, 192, 296, 400):
        probability = np.array([0.3, 0.7])
        data[f"p_{prefix}"] = probability
        briers[str(prefix)] = float(np.mean(np.square(probability - target)))
    data.update(
        game_type=["R", "R"],
        game_month=[4, 4],
        pitcher_id_known=["known", "known"],
        batter_id_known=["known", "known"],
    )
    pd.DataFrame(data).loc[:, CATBOOST_PREDICTION_COLUMNS].to_csv(
        root / "predictions.csv", index=False
    )
    (root / "model.cbm").write_bytes(b"alignment-model")
    (root / "preprocessing_state.json").write_bytes(
        _preprocessing_bytes(preprocessing_frame)
    )
    (root / "worker.log").write_text("CATBOOST_DEPLOY_PROGRESS\n")
    (root / "job.json").write_bytes(_canonical({"job_id": job_id, "kind": "alignment"}))
    metrics = {
        "job_id": job_id,
        "kind": "alignment",
        "train_rows": 2,
        "valid_rows": 2,
        "selected_tree_count": 400,
        "model_sha256": _sha(root / "model.cbm"),
        "preprocessing_sha256": _sha(root / "preprocessing_state.json"),
        "predictions_sha256": _sha(root / "predictions.csv"),
        "snapshot_sha256": None,
        "prefix_brier": briers,
    }
    (root / "metrics.json").write_bytes(_canonical(metrics))
    result = {
        "job_id": job_id,
        "kind": "alignment",
        "status": "completed",
        "train_rows": 2,
        "valid_rows": 2,
        "predictions": "predictions.csv",
        "model": "model.cbm",
        "preprocessing": "preprocessing_state.json",
        "snapshot": None,
        "elapsed_seconds": 1.0,
        "failure": None,
    }
    (root / "worker_result.json").write_bytes(_canonical(result))
    return root


def _full_job(
    root: Path, preprocessing_frame: pd.DataFrame, decision_sha: str
) -> Path:
    root.mkdir(parents=True)
    (root / "model.cbm").write_bytes(b"full-model")
    (root / "preprocessing_state.json").write_bytes(
        _preprocessing_bytes(preprocessing_frame)
    )
    (root / "worker.log").write_text("CATBOOST_DEPLOY_PROGRESS\n")
    (root / "job.json").write_bytes(
        _canonical(
            {
                "job_id": "full_2024",
                "kind": "full_fit",
                "selected_tree_count": 128,
                "alignment_decision_sha256": decision_sha,
            }
        )
    )
    metrics = {
        "job_id": "full_2024",
        "kind": "full_fit",
        "train_rows": 1475092,
        "valid_rows": None,
        "selected_tree_count": 128,
        "model_sha256": _sha(root / "model.cbm"),
        "preprocessing_sha256": _sha(root / "preprocessing_state.json"),
        "snapshot_sha256": None,
        "alignment_decision_sha256": decision_sha,
    }
    (root / "metrics.json").write_bytes(_canonical(metrics))
    result = {
        "job_id": "full_2024",
        "kind": "full_fit",
        "status": "completed",
        "train_rows": 1475092,
        "valid_rows": None,
        "predictions": None,
        "model": "model.cbm",
        "preprocessing": "preprocessing_state.json",
        "snapshot": None,
        "elapsed_seconds": 1.0,
        "failure": None,
    }
    (root / "worker_result.json").write_bytes(_canonical(result))
    return root


def _decision(path: Path, *, blocked: bool = False) -> Path:
    payload = {
        "status": "deployment_blocked" if blocked else "deployment_aligned",
        "selected_tree_count": None if blocked else 128,
        "baseline_weighted_brier": 0.25,
        "candidates": [],
        "reason": "no_tree_prefix_passed" if blocked else "fixed_tree_prefix_passed",
    }
    path.write_bytes(_canonical(payload))
    return path


def _state(
    path: Path,
    bindings: dict[str, str],
    completed: list[str],
    *,
    status: str,
    decision_sha: str | None = None,
    selected_tree_count: int | None = None,
) -> Path:
    path.write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "campaign_id": "catboost_deployment_v1",
                "status": status,
                "completed_job_ids": completed,
                "active_job_id": None,
                "decision_sha256": decision_sha,
                "selected_tree_count": selected_tree_count,
                "bindings": bindings,
            }
        )
    )
    return path


def _write(
    tmp_path: Path,
    preprocessing_frame: pd.DataFrame,
    *,
    status: str,
    include_full: bool = False,
    blocked: bool = False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bindings = _bindings()
    log = tmp_path / "campaign.log"
    log.write_text("DEPLOY_ALIGNMENT_DECISION\n")
    jobs: dict[str, Path] = {}
    decision = None
    decision_sha = None
    completed: list[str] = []
    if status != "fresh":
        for job_id in ("align_2022_2023", "align_2023_2024"):
            jobs[job_id] = _alignment_job(tmp_path / "jobs" / job_id, job_id, preprocessing_frame)
            completed.append(job_id)
        decision = _decision(tmp_path / "decision.json", blocked=blocked)
        decision_sha = _sha(decision)
    if include_full:
        jobs["full_2024"] = _full_job(
            tmp_path / "jobs" / "full_2024", preprocessing_frame, str(decision_sha)
        )
        completed.append("full_2024")
    state = _state(
        tmp_path / "state.json",
        bindings,
        completed,
        status=status,
        decision_sha=decision_sha,
        selected_tree_count=None if blocked or decision is None else 128,
    )
    bundles = write_deployment_bundles(
        output_dir=tmp_path / "bundles",
        bindings=bindings,
        contract_path=DEFAULT_CONTRACT,
        stage_state_path=state,
        campaign_log_path=log,
        job_directories=jobs,
        decision_path=decision,
    )
    return bundles, bindings, jobs


def test_fresh_state_emits_resume_only(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    bundles, bindings, _ = _write(tmp_path, preprocessing_frame, status="fresh")

    assert bundles.review is None
    assert bundles.delivery is None
    verify_deployment_resume(bundles.resume, expected_bindings=bindings)


def test_blocked_alignment_has_review_but_no_delivery(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    bundles, _, _ = _write(
        tmp_path, preprocessing_frame, status="deployment_blocked", blocked=True
    )

    assert bundles.review is not None
    assert bundles.delivery is None


def test_full_training_delivery_is_recursive_and_exact(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    bundles, bindings, _ = _write(
        tmp_path,
        preprocessing_frame,
        status="full_training_complete",
        include_full=True,
    )

    assert bundles.delivery is not None
    verify_full_training_delivery(bundles.delivery, expected_bindings=bindings)
    with ZipFile(bundles.delivery) as archive:
        assert set(archive.namelist()) == {
            "catboost_deployment.log",
            "catboost_deployment_review.zip",
            "catboost_deployment_resume.zip",
            "delivery_manifest.json",
        }


def test_rejects_model_hash_mismatch(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    bindings = _bindings()
    decision = _decision(tmp_path / "decision.json")
    decision_sha = _sha(decision)
    jobs = {
        "align_2022_2023": _alignment_job(
            tmp_path / "jobs/a", "align_2022_2023", preprocessing_frame
        ),
        "align_2023_2024": _alignment_job(
            tmp_path / "jobs/b", "align_2023_2024", preprocessing_frame
        ),
        "full_2024": _full_job(tmp_path / "jobs/c", preprocessing_frame, decision_sha),
    }
    (jobs["full_2024"] / "model.cbm").write_bytes(b"tampered")
    state = _state(
        tmp_path / "state.json",
        bindings,
        list(jobs),
        status="full_training_complete",
        decision_sha=decision_sha,
        selected_tree_count=128,
    )
    log = tmp_path / "log"
    log.write_text("x")

    with pytest.raises(DeploymentArtifactError, match="model SHA"):
        write_deployment_bundles(
            output_dir=tmp_path / "bundles",
            bindings=bindings,
            contract_path=DEFAULT_CONTRACT,
            stage_state_path=state,
            campaign_log_path=log,
            job_directories=jobs,
            decision_path=decision,
        )


def test_resume_rejects_consistently_rehashed_model_tamper(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    bundles, bindings, _ = _write(
        tmp_path / "source",
        preprocessing_frame,
        status="deployment_aligned",
    )
    with ZipFile(bundles.resume) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    target = "jobs/align_2022_2023/model.cbm"
    members[target] = b"forged-model"
    manifest = json.loads(members["manifest.json"])
    manifest["members"][target] = {
        "size": len(members[target]),
        "sha256": sha256(members[target]).hexdigest(),
    }
    members["manifest.json"] = _canonical(manifest)
    forged = tmp_path / "forged.zip"
    with ZipFile(forged, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)

    with pytest.raises(DeploymentArtifactError, match="model SHA"):
        verify_deployment_resume(forged, expected_bindings=bindings)


def test_review_rejects_consistently_rehashed_frozen_model_tamper(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    bundles, bindings, _ = _write(
        tmp_path / "source",
        preprocessing_frame,
        status="full_training_complete",
        include_full=True,
    )
    assert bundles.review is not None
    with ZipFile(bundles.review) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    target = "model/model.cbm"
    members[target] = b"forged-model"
    manifest = json.loads(members["manifest.json"])
    manifest["members"][target] = {
        "size": len(members[target]),
        "sha256": sha256(members[target]).hexdigest(),
    }
    members["manifest.json"] = _canonical(manifest)
    forged = tmp_path / "forged-review.zip"
    with ZipFile(forged, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)

    with pytest.raises(DeploymentArtifactError, match="frozen model SHA"):
        verify_deployment_review(forged, expected_bindings=bindings)
