from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import pytest

from submission.tabm_candidate import (
    CANDIDATE_ID,
    TabMCandidateError,
    import_review_delivery,
    load_imported_candidate,
)


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _zip(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return stream.getvalue()


def _delivery(
    tmp_path: Path,
    *,
    status: str = "review_ready",
    extra_frozen_member: str | None = None,
) -> Path:
    model_files = {
        "numeric_embedding_0.json": _json({"mode": "piecewise_linear"}),
        "preprocessing_state.json": _json({"view": "raw_typed"}),
        "tabm_member_0_seed_3407.pt": b"weights",
    }
    inference_manifest = {
        "schema_version": 1,
        "epochs": 3,
        "seeds": [3407],
        "scheduler": "constant",
        "fit_scope": "official_train_2019_2024_only",
        "row_count": 1_475_092,
        "preprocessing_state": "preprocessing_state.json",
        "members": [
            {
                "seed": 3407,
                "weights": "tabm_member_0_seed_3407.pt",
                "numeric_state": "numeric_embedding_0.json",
                "model_config": {
                    "architecture": "tabm",
                    "blocks": 4,
                    "dropout": 0.1,
                    "k": 32,
                    "num_embedding": "piecewise_linear",
                    "width": 512,
                },
            }
        ],
        "files": {name: sha256(value).hexdigest() for name, value in model_files.items()},
        "identity": {"runtime_sha256": "a" * 64},
    }
    frozen = {
        "frozen_inference/inference_manifest.json": _json(inference_manifest),
        **{f"frozen_inference/{name}": value for name, value in model_files.items()},
    }
    if extra_frozen_member is not None:
        frozen[f"frozen_inference/{extra_frozen_member}"] = b"forbidden"
    final_review = {
        "acceptance_status": status,
        "version": "D",
        "review_only": True,
        "fixed_epochs": 3,
        "fit_report": {
            "epochs": 3,
            "rows": 1_475_092,
            "scheduler": "constant",
            "seeds": [3407],
            "status": "completed",
        },
        "artifact_sha256": {
            name.removeprefix("frozen_inference/"): sha256(value).hexdigest()
            for name, value in frozen.items()
        },
    }
    inner_members = {
        "final_review.json": _json(final_review),
        **frozen,
        "policy/policy.json": b"{}\n",
        "policy/policy_review.json": b"{}\n",
    }
    inner_manifest = {
        "schema_version": 1,
        "artifact_kind": "review",
        "version": "D",
        "review_only": True,
        "members": {
            name: sha256(value).hexdigest() for name, value in inner_members.items()
        },
    }
    inner_members["manifest.json"] = _json(inner_manifest)
    inner = _zip(inner_members)
    log = b"VERSION_D_REVIEW_READY\n"
    outer_manifest = {
        "schema_version": 1,
        "artifact_kind": "tabm_version_D_review_delivery",
        "review_only": True,
        "submission_package": False,
        "members": {
            "tabm_hand_matchup_final_review_bundle.zip": {
                "sha256": sha256(inner).hexdigest(),
                "size": len(inner),
            },
            "version_d.log": {"sha256": sha256(log).hexdigest(), "size": len(log)},
        },
    }
    path = tmp_path / "delivery.zip"
    path.write_bytes(
        _zip(
            {
                "delivery_manifest.json": _json(outer_manifest),
                "tabm_hand_matchup_final_review_bundle.zip": inner,
                "version_d.log": log,
            }
        )
    )
    return path


def test_import_review_delivery_is_hash_bound_and_exclusive(tmp_path: Path) -> None:
    result = import_review_delivery(_delivery(tmp_path), tmp_path / "candidate")

    assert result.candidate_id == CANDIDATE_ID
    assert sorted(result.member_sha256) == [
        "inference_manifest.json",
        "numeric_embedding_0.json",
        "preprocessing_state.json",
        "tabm_member_0_seed_3407.pt",
    ]
    assert len(result.delivery_sha256) == 64
    assert len(result.review_bundle_sha256) == 64
    assert len(result.model_sha256) == 64
    assert result.model_dir.is_dir()
    assert load_imported_candidate(result.root) == result
    with pytest.raises(TabMCandidateError, match="destination already exists"):
        import_review_delivery(_delivery(tmp_path), tmp_path / "candidate")


def test_import_rejects_outer_hash_difference(tmp_path: Path) -> None:
    path = _delivery(tmp_path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["version_d.log"] = b"changed"
    path.write_bytes(_zip(members))

    with pytest.raises(TabMCandidateError, match="member SHA-256"):
        import_review_delivery(path, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_import_rejects_non_review_ready_delivery(tmp_path: Path) -> None:
    with pytest.raises(TabMCandidateError, match="review_ready"):
        import_review_delivery(
            _delivery(tmp_path, status="rejected"), tmp_path / "candidate"
        )


def test_import_rejects_training_state(tmp_path: Path) -> None:
    with pytest.raises(TabMCandidateError, match="training state"):
        import_review_delivery(
            _delivery(tmp_path, extra_frozen_member="optimizer.pt"),
            tmp_path / "candidate",
        )
