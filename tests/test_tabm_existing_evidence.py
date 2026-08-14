from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import pytest
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from competition_rules.contract import load_policy, policy_digest
from competition_rules.evidence_gate import validate_full_audit
from submission.tabm_candidate import CANDIDATE_ID, ImportedTabMCandidate
from submission.tabm_existing_evidence import (
    ExistingEvidenceError,
    build_reused_gpu_acceptance,
    verify_existing_gpu_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


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


class _Predictor:
    def state_digest(self) -> str:
        return "d" * 64

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = frame[["x"]].to_numpy(dtype="float32")
        categorical = np.zeros((len(frame), 1), dtype="int64")
        return numeric, categorical

    def predict_batch(
        self, frame: pd.DataFrame, *, batch_size: int = 2048
    ) -> np.ndarray:
        del batch_size
        values = {f"r{index}": 0.41 + index * 0.01 for index in range(5)}
        return frame["row_id"].map(values).to_numpy(dtype="float64")


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.DataFrame(
        {"row_id": [f"r{index}" for index in range(5)], "x": range(5)}
    )
    sample = pd.DataFrame(
        {"row_id": test["row_id"], "control_success": [0.0] * len(test)}
    )
    return test, sample


def _policy_files(project: Path) -> tuple[Path, Path, datetime]:
    rules = project / "competition_rules"
    rules.mkdir(parents=True)
    policy_path = rules / "policy.json"
    policy_path.write_bytes((ROOT / "competition_rules/policy.json").read_bytes())
    policy = load_policy(policy_path, project_root=project)
    package_time = datetime(2026, 8, 15, 18, tzinfo=ZoneInfo("Asia/Seoul"))
    review_path = project / "policy_review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_version": policy["policy_version"],
                "policy_sha256": policy_digest(policy),
                "reviewed_at": package_time.isoformat(),
                "sources": policy["official_sources"],
                "verdict": "unchanged",
            }
        ),
        encoding="utf-8",
    )
    return policy_path, review_path, package_time


def _acceptance_args(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    candidate = _candidate(project)
    stage_c = _stage_c(project)
    stage_d = _stage_d(project, stage_c, candidate)
    gpu = verify_existing_gpu_evidence(
        stage_c_delivery=stage_c,
        stage_d_delivery=stage_d,
        candidate=candidate,
    )
    test, sample = _frames()
    policy, review, package_time = _policy_files(project)
    return {
        "project_root": project,
        "candidate": candidate,
        "gpu_evidence": gpu,
        "test_frame": test,
        "sample_frame": sample,
        "runtime_bytes": b"def predict(frame):\n    return frame['x']\n",
        "output_dir": project / "evidence",
        "load_predictor": _Predictor,
        "python_probe": {
            "status": "passed",
            "python": "3.11.14",
            "tabm": "0.0.3",
            "rtdl_num_embeddings": "0.0.12",
            "probabilities": list(gpu.gpu_sample_probabilities),
        },
        "package_bytes": 11_000_000,
        "extracted_bytes": 11_500_000,
        "policy_path": policy,
        "policy_review_path": review,
        "package_time": package_time,
    }


def test_build_acceptance_combines_current_sample_and_existing_t4(
    tmp_path: Path,
) -> None:
    result = build_reused_gpu_acceptance(**_acceptance_args(tmp_path))

    assert result.acceptance["status"] == "passed"
    assert all(result.acceptance["gates"].values())
    assert result.benchmark["inference_seconds"] == pytest.approx(
        11.938824568999735
    )
    assert result.benchmark["peak_vram_bytes"] == 973_573_120
    validate_full_audit(result.audit_manifest, expected_identity=result.identity)
    assert json.loads(result.acceptance_path.read_text()) == result.acceptance
    gpu = json.loads((result.acceptance_path.parent / "gpu_evidence.json").read_text())
    assert gpu["model_sha256"] == result.identity.model_sha256


def test_build_acceptance_rejects_gpu_cpu_probability_drift(
    tmp_path: Path,
) -> None:
    arguments = _acceptance_args(tmp_path)
    probe = dict(arguments["python_probe"])
    probe["probabilities"] = [0.9] * 5
    arguments["python_probe"] = probe

    with pytest.raises(ExistingEvidenceError, match="sample prediction parity"):
        build_reused_gpu_acceptance(**arguments)

    assert not Path(arguments["output_dir"]).exists()


def test_build_acceptance_rejects_wrong_python_or_stale_policy(
    tmp_path: Path,
) -> None:
    arguments = _acceptance_args(tmp_path)
    probe = dict(arguments["python_probe"])
    probe["python"] = "3.13.12"
    arguments["python_probe"] = probe
    with pytest.raises(ExistingEvidenceError, match="Python 3.11"):
        build_reused_gpu_acceptance(**arguments)
    assert not Path(arguments["output_dir"]).exists()

    arguments = _acceptance_args(tmp_path / "stale")
    arguments["package_time"] = datetime(
        2026, 8, 16, 1, tzinfo=ZoneInfo("Asia/Seoul")
    )
    with pytest.raises(ExistingEvidenceError, match="same KST date"):
        build_reused_gpu_acceptance(**arguments)
    assert not Path(arguments["output_dir"]).exists()


def test_build_acceptance_rejects_non_row_local_runtime(tmp_path: Path) -> None:
    arguments = _acceptance_args(tmp_path)
    arguments["runtime_bytes"] = (
        b"def predict(frame):\n    return frame.groupby('x').size()\n"
    )

    with pytest.raises(ExistingEvidenceError, match="source gate"):
        build_reused_gpu_acceptance(**arguments)

    assert not Path(arguments["output_dir"]).exists()
