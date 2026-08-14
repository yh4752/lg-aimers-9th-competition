from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import pytest

from submission.tabm_candidate import CANDIDATE_ID, ImportedTabMCandidate
from submission.tabm_existing_evidence import (
    ExistingEvidenceError,
    verify_existing_gpu_evidence,
)


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _zip(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return stream.getvalue()


def _manifest_members(members: dict[str, bytes]) -> dict[str, dict[str, object]]:
    return {
        name: {"sha256": sha256(value).hexdigest(), "size": len(value)}
        for name, value in members.items()
    }


def _candidate(tmp_path: Path) -> ImportedTabMCandidate:
    model = tmp_path / "candidate/model"
    model.mkdir(parents=True)
    values = {
        "inference_manifest.json": b"inference\n",
        "numeric_embedding_0.json": b"embedding\n",
        "preprocessing_state.json": b"preprocessing\n",
        "tabm_member_0_seed_3407.pt": b"weights\n",
    }
    combined = sha256()
    hashes: dict[str, str] = {}
    for name, value in sorted(values.items()):
        (model / name).write_bytes(value)
        digest = sha256(value).hexdigest()
        hashes[name] = digest
        combined.update(name.encode() + b"\0" + bytes.fromhex(digest))
    return ImportedTabMCandidate(
        candidate_id=CANDIDATE_ID,
        root=model.parent,
        model_dir=model,
        delivery_sha256="1" * 64,
        review_bundle_sha256="2" * 64,
        model_sha256=combined.hexdigest(),
        member_sha256=hashes,
    )


def _stage_c(tmp_path: Path, *, accepted: bool = True) -> Path:
    state = {
        "final_members": [{"seed": 3407, "status": "completed"}],
        "predictor_evidence": [
            {
                "accepted": accepted,
                "predictor_id": "single_s3407",
                "primary_brier": 0.24811080225115084,
                "older_brier": 0.25086573594721084,
                "members": [
                    {
                        "candidate_id": "a__p2__piecewise_linear__bce__plateau__s42",
                        "seed": 3407,
                    }
                ],
            }
        ],
    }
    resume_members = {"stage_state.json": _json(state)}
    resume_members["manifest.json"] = _json(
        {
            "schema_version": 1,
            "artifact_kind": "resume",
            "version": "C",
            "review_only": True,
            "members": {
                "stage_state.json": sha256(resume_members["stage_state.json"]).hexdigest()
            },
        }
    )
    outer_members = {
        "colab_stage_C.log": b"COLAB_STAGE_C_DELIVERY_READY\n",
        "tabm_search_stage_C_resume_bundle.zip": _zip(resume_members),
        "tabm_search_stage_C_review_bundle.zip": b"review",
    }
    outer_members["delivery_manifest.json"] = _json(
        {
            "schema_version": 1,
            "artifact_kind": "tabm_colab_stage_C_delivery",
            "final_stage_complete": True,
            "run_uuid": "fixture",
            "runtime_identity": {"gpu_name": "Tesla T4"},
            "members": _manifest_members(outer_members),
        }
    )
    path = tmp_path / "stage_c.zip"
    path.write_bytes(_zip(outer_members))
    return path


def _stage_d(
    tmp_path: Path,
    stage_c: Path,
    candidate: ImportedTabMCandidate,
    *,
    gpu_name: str = "Tesla T4",
    rows: int = 245_789,
    independence_delta: float = 2.9802322387695312e-08,
) -> Path:
    review = {
        "acceptance_status": "review_ready",
        "artifact_sha256": dict(candidate.member_sha256),
        "dependency_probe": {"elapsed_seconds": 1.7805408000003808, "status": "passed"},
        "fit_report": {"status": "completed", "epochs": 3, "rows": 1_475_092},
        "fixed_epochs": 3,
        "frozen_sample_probe": {
            "status": "passed",
            "stdout_tail": "0.41,0.42,0.43,0.44,0.45\n",
        },
        "independence": {
            "max_abs_probability_delta": independence_delta,
            "row_count": 5,
        },
        "review_only": True,
        "scale": {
            "elapsed_seconds": 11.938824568999735,
            "peak_gpu_allocated_bytes": 973_573_120,
            "peak_rss_bytes": 4_092_870_656,
            "rows": rows,
        },
        "version": "D",
    }
    inner_members = {
        "final_review.json": _json(review),
        **{
            f"frozen_inference/{name}": (candidate.model_dir / name).read_bytes()
            for name in candidate.member_sha256
        },
        "policy/policy.json": b"{}\n",
        "policy/policy_review.json": b"{}\n",
    }
    inner_members["manifest.json"] = _json(
        {
            "schema_version": 1,
            "artifact_kind": "review",
            "version": "D",
            "review_only": True,
            "members": {
                name: sha256(value).hexdigest() for name, value in inner_members.items()
            },
        }
    )
    inner = _zip(inner_members)
    log = (
        f"VERSION_D_GPU_READY device_count=1 name={gpu_name}\n"
        "VERSION_D_INDEPENDENCE_PASSED\n"
        f"VERSION_D_SCALE_GATE_PASSED rows={rows} seconds=11.939\n"
    ).encode()
    declared = {
        "tabm_hand_matchup_final_review_bundle.zip": inner,
        "version_d.log": log,
    }
    outer_members = dict(declared)
    outer_members["delivery_manifest.json"] = _json(
        {
            "schema_version": 1,
            "artifact_kind": "tabm_version_D_review_delivery",
            "contract_sha256": "3" * 64,
            "data_archive_sha256": "4" * 64,
            "frozen_sha256": "5" * 64,
            "members": _manifest_members(declared),
            "prior_manifest_sha256": "6" * 64,
            "review_only": True,
            "runtime_versions": {"tabm": "0.0.3"},
            "stage_c_delivery_sha256": sha256(stage_c.read_bytes()).hexdigest(),
            "submission_package": False,
        }
    )
    path = tmp_path / "stage_d.zip"
    path.write_bytes(_zip(outer_members))
    return path


def test_verified_evidence_binds_stage_c_d_model_and_t4(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    stage_c = _stage_c(tmp_path)
    stage_d = _stage_d(tmp_path, stage_c, candidate)

    evidence = verify_existing_gpu_evidence(
        stage_c_delivery=stage_c,
        stage_d_delivery=stage_d,
        candidate=candidate,
    )

    assert evidence.candidate_id == CANDIDATE_ID
    assert evidence.model_sha256 == candidate.model_sha256
    assert evidence.gpu_name == "Tesla T4"
    assert evidence.capacity_rows == 245_789
    assert evidence.inference_seconds == pytest.approx(11.938824568999735)
    assert evidence.primary_brier == pytest.approx(0.24811080225115084)
    assert evidence.older_brier == pytest.approx(0.25086573594721084)
    assert evidence.gpu_sample_probabilities == (0.41, 0.42, 0.43, 0.44, 0.45)


def test_existing_evidence_rejects_stage_c_or_model_mismatch(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    stage_c = _stage_c(tmp_path)
    stage_d = _stage_d(tmp_path, stage_c, candidate)
    stage_c.write_bytes(stage_c.read_bytes() + b"changed")
    with pytest.raises(ExistingEvidenceError, match="Stage C delivery SHA-256"):
        verify_existing_gpu_evidence(
            stage_c_delivery=stage_c, stage_d_delivery=stage_d, candidate=candidate
        )

    stage_c = _stage_c(tmp_path)
    stage_d = _stage_d(tmp_path, stage_c, candidate)
    changed = replace(candidate, member_sha256={**candidate.member_sha256, "tabm_member_0_seed_3407.pt": "f" * 64})
    with pytest.raises(ExistingEvidenceError, match="model SHA-256"):
        verify_existing_gpu_evidence(
            stage_c_delivery=stage_c, stage_d_delivery=stage_d, candidate=changed
        )


@pytest.mark.parametrize(
    "gpu_name,rows,delta,message",
    [
        ("NVIDIA L4", 245_789, 0.0, "Tesla T4"),
        ("Tesla T4", 10, 0.0, "245789"),
        ("Tesla T4", 245_789, 1e-3, "row independence"),
    ],
)
def test_existing_evidence_rejects_invalid_gpu_gates(
    tmp_path: Path, gpu_name: str, rows: int, delta: float, message: str
) -> None:
    candidate = _candidate(tmp_path)
    stage_c = _stage_c(tmp_path)
    stage_d = _stage_d(
        tmp_path,
        stage_c,
        candidate,
        gpu_name=gpu_name,
        rows=rows,
        independence_delta=delta,
    )
    with pytest.raises(ExistingEvidenceError, match=message):
        verify_existing_gpu_evidence(
            stage_c_delivery=stage_c, stage_d_delivery=stage_d, candidate=candidate
        )


def test_existing_evidence_rejects_unaccepted_predictor(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    stage_c = _stage_c(tmp_path, accepted=False)
    stage_d = _stage_d(tmp_path, stage_c, candidate)
    with pytest.raises(ExistingEvidenceError, match="single_s3407"):
        verify_existing_gpu_evidence(
            stage_c_delivery=stage_c, stage_d_delivery=stage_d, candidate=candidate
        )
