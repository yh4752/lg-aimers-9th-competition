from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import pytest


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _zip(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)
    return stream.getvalue()


def _declared(members: dict[str, bytes]) -> dict[str, dict[str, object]]:
    return {
        name: {"sha256": sha256(value).hexdigest(), "size": len(value)}
        for name, value in members.items()
    }


def _handoff(path: Path, *, status: str = "accepted") -> str:
    models = {
        42: b"model-42",
        2026: b"model-2026",
        3407: b"model-3407",
    }
    payloads = {
        "evidence/acceptance_decision.json": _json(
            {
                "bootstrap_lower": 0.0006,
                "bootstrap_upper": 0.0010,
                "f3_gain": 0.00034,
                "maximum_segment_regression": 0.0003,
                "performance_grade": "breakthrough",
                "predictor": "catboost",
                "reason": "standalone_catboost_gates_passed",
                "status": "accepted",
                "weighted_gain": 0.00083,
                "worst_fold_gain": 0.00034,
            }
        ),
        "evidence/full_fit_manifest.json": _json(
            {
                "candidate_id": "c1_anchor_residual",
                "model_sha256": {
                    str(seed): sha256(value).hexdigest()
                    for seed, value in models.items()
                },
                "predictor": "catboost",
            }
        ),
        "evidence/inference_audit.json": _json(
            {
                "checks": {
                    "batch_1": 0.0,
                    "batch_257": 0.0,
                    "batch_4096": 0.0,
                    "companion": 0.0,
                    "reverse": 0.0,
                    "shuffle": 0.0,
                    "singleton": 0.0,
                },
                "elapsed_seconds": 64.939,
                "maximum_absolute_difference": 0.0,
                "peak_gpu_bytes": 0,
                "peak_rss_bytes": 3_308_498_944,
                "reason": "inference_audit_passed",
                "row_count": 245_789,
                "status": "passed",
            }
        ),
        "frozen_state/feature_state.json": _json(
            {
                "artifact_kind": "tree_expert_e2_frozen_state_v1",
                "candidate_id": "c1_anchor_residual",
                "members": {
                    "s1_batter.csv": {
                        "sha256": sha256(b"batter\n").hexdigest(),
                        "size": 7,
                    },
                    "s1_pitcher.csv": {
                        "sha256": sha256(b"pitcher\n").hexdigest(),
                        "size": 8,
                    },
                },
                "schema_version": 1,
                "trackman": None,
                "valid_year": 2025,
            }
        ),
        "frozen_state/s1_batter.csv": b"batter\n",
        "frozen_state/s1_pitcher.csv": b"pitcher\n",
        **{f"models/catboost_seed_{seed}.cbm": value for seed, value in models.items()},
    }
    delivery_manifest = {
        "artifact_kind": "tree_expert_e2_model_delivery_v1",
        "campaign_id": "tree_expert_e2_v1",
        "candidate_id": "c1_anchor_residual",
        "decision_sha256": sha256(
            payloads["evidence/acceptance_decision.json"]
        ).hexdigest(),
        "iterations": {"42": 78, "2026": 86, "3407": 149},
        "members": _declared(payloads),
        "predictor": "catboost",
        "review_only": False,
        "schema_version": 1,
        "seeds": [42, 2026, 3407],
        "submission_package": False,
    }
    delivery = _zip({**payloads, "manifest.json": _json(delivery_manifest)})
    outer_members = {
        "tree_expert_e2.log": b"TREE_E2_CAMPAIGN_SUCCESS status=accepted\n",
        "tree_expert_e2_model_delivery.zip": delivery,
        "tree_expert_e2_resume.zip": b"resume",
        "tree_expert_e2_review.zip": b"review",
    }
    handoff_manifest = {
        "artifact_kind": "tree_expert_e2_handoff_v1",
        "campaign_id": "tree_expert_e2_v1",
        "delivery": True,
        "members": _declared(outer_members),
        "review_only": False,
        "schema_version": 1,
        "status": status,
        "submission_package": False,
    }
    value = _zip({**outer_members, "handoff_manifest.json": _json(handoff_manifest)})
    path.write_bytes(value)
    return sha256(value).hexdigest()


def test_imports_only_registered_accepted_delivery(tmp_path: Path, monkeypatch) -> None:
    from submission import tree_expert_e2_candidate as module

    source = tmp_path / "handoff.zip"
    monkeypatch.setattr(module, "TREE_E2_HANDOFF_SHA256", _handoff(source))

    candidate = module.import_tree_expert_e2_candidate(source, tmp_path / "candidate")

    assert candidate.candidate_id == "c1_anchor_residual"
    assert candidate.seeds == (42, 2026, 3407)
    assert set(candidate.member_sha256) == {
        "frozen_state/feature_state.json",
        "frozen_state/s1_batter.csv",
        "frozen_state/s1_pitcher.csv",
        "models/catboost_seed_42.cbm",
        "models/catboost_seed_2026.cbm",
        "models/catboost_seed_3407.cbm",
    }
    assert candidate.model_dir.joinpath("models/catboost_seed_42.cbm").read_bytes() == b"model-42"
    manifest = json.loads(candidate.root.joinpath("candidate_manifest.json").read_text())
    assert manifest["handoff_sha256"] == candidate.handoff_sha256
    assert manifest["members"] == dict(candidate.member_sha256)


def test_rejects_unregistered_or_rejected_handoff_before_output(
    tmp_path: Path, monkeypatch
) -> None:
    from submission import tree_expert_e2_candidate as module

    source = tmp_path / "handoff.zip"
    registered = _handoff(source, status="rejected")
    monkeypatch.setattr(module, "TREE_E2_HANDOFF_SHA256", registered)
    output = tmp_path / "candidate"

    with pytest.raises(module.TreeE2CandidateError, match="not accepted"):
        module.import_tree_expert_e2_candidate(source, output)
    assert not output.exists()

    monkeypatch.setattr(module, "TREE_E2_HANDOFF_SHA256", "0" * 64)
    with pytest.raises(module.TreeE2CandidateError, match="SHA-256"):
        module.import_tree_expert_e2_candidate(source, output)
    assert not output.exists()
