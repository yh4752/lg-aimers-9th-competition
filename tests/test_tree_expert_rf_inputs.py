from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
import pytest

from experiments.tree_expert.rf_inputs import (
    RFInputError,
    file_sha256,
    prepare_rf_input,
    verify_and_extract_rf_input,
)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _manifest(kind: str, members: dict[str, bytes], **extra: object) -> bytes:
    value = {
        "schema_version": 1,
        "artifact_kind": kind,
        "members": {
            name: {"sha256": sha256(payload).hexdigest(), "size": len(payload)}
            for name, payload in sorted(members.items())
        },
        **extra,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _nested_zip(members: dict[str, bytes], manifest: bytes) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(_zip_info(name), payload)
        archive.writestr(_zip_info("manifest.json"), manifest)
    return buffer.getvalue()


def make_e2_handoff(path: Path) -> Path:
    folds: dict[str, bytes] = {}
    for train_end, valid_year in ((2021, 2022), (2022, 2023), (2023, 2024)):
        folds[f"ensembles/{train_end}_{valid_year}.csv"] = pd.DataFrame(
            {
                "row_id": [f"r{valid_year}a", f"r{valid_year}b"],
                "target": [0, 1],
                "probability": [0.45, 0.55],
                "game_type": ["R", "F"],
            }
        ).to_csv(index=False).encode()
    acceptance = b'{"status":"accepted","predictor":"catboost"}'
    review_members = {**folds, "decisions/acceptance.json": acceptance}
    review = _nested_zip(
        review_members,
        _manifest("tree_expert_e2_review_v1", review_members),
    )

    delivery_members = {
        "evidence/acceptance_decision.json": acceptance,
        "evidence/full_fit_manifest.json": b'{"candidate_id":"c1_anchor_residual","predictor":"catboost"}',
        "evidence/inference_audit.json": b'{"status":"passed"}',
        "frozen_state/feature_state.json": b'{}',
        "models/catboost_seed_42.cbm": b"model-42",
        "models/catboost_seed_2026.cbm": b"model-2026",
        "models/catboost_seed_3407.cbm": b"model-3407",
    }
    delivery = _nested_zip(
        delivery_members,
        _manifest(
            "tree_expert_e2_model_delivery_v1",
            delivery_members,
            campaign_id="tree_expert_e2_v1",
            review_only=False,
            submission_package=False,
            candidate_id="c1_anchor_residual",
            predictor="catboost",
            seeds=[42, 2026, 3407],
            iterations={"42": 10, "2026": 10, "3407": 10},
            decision_sha256=sha256(acceptance).hexdigest(),
        ),
    )
    resume = b"resume"
    log = b"accepted\n"
    handoff_members = {
        "tree_expert_e2_model_delivery.zip": delivery,
        "tree_expert_e2_resume.zip": resume,
        "tree_expert_e2_review.zip": review,
        "tree_expert_e2.log": log,
    }
    handoff_manifest = _manifest(
        "tree_expert_e2_handoff_v1",
        handoff_members,
        campaign_id="tree_expert_e2_v1",
        review_only=False,
        submission_package=False,
        status="accepted",
        delivery=True,
    )
    with ZipFile(path, "w") as archive:
        for name, payload in handoff_members.items():
            archive.writestr(_zip_info(name), payload)
        archive.writestr(_zip_info("handoff_manifest.json"), handoff_manifest)
    return path


def _rewrite_outer(path: Path, output: Path, transform) -> Path:
    with ZipFile(path) as source, ZipFile(output, "w") as target:
        for info in source.infolist():
            target.writestr(_zip_info(info.filename), transform(info.filename, source.read(info)))
    return output


def test_prepare_and_verify_rf_input_round_trip(tmp_path: Path) -> None:
    handoff = make_e2_handoff(tmp_path / "e2.zip")
    digest = file_sha256(handoff)

    archive = prepare_rf_input(
        e2_handoff=handoff,
        output=tmp_path / "rf_input.zip",
        expected_e2_sha256=digest,
    )
    verified = verify_and_extract_rf_input(
        archive,
        tmp_path / "verified",
        expected_e2_sha256=digest,
    )

    assert verified.e2_candidate_id == "c1_anchor_residual"
    assert set(verified.fold_predictions) == {(2021, 2022), (2022, 2023), (2023, 2024)}
    assert verified.full_fit_root.is_dir()
    assert (verified.full_fit_root / "models/catboost_seed_3407.cbm").is_file()


def test_rf_input_rejects_modified_member(tmp_path: Path) -> None:
    handoff = make_e2_handoff(tmp_path / "e2.zip")
    digest = file_sha256(handoff)
    archive = prepare_rf_input(
        e2_handoff=handoff,
        output=tmp_path / "rf_input.zip",
        expected_e2_sha256=digest,
    )
    tampered = _rewrite_outer(
        archive,
        tmp_path / "tampered.zip",
        lambda name, payload: payload + b"\n" if name == "e2/fold_2022_2023.csv" else payload,
    )

    with pytest.raises(RFInputError, match="member SHA-256 differs"):
        verify_and_extract_rf_input(tampered, tmp_path / "verified", expected_e2_sha256=digest)


def test_rf_input_rejects_wrong_e2_candidate(tmp_path: Path) -> None:
    handoff = make_e2_handoff(tmp_path / "e2.zip")
    digest = file_sha256(handoff)
    archive = prepare_rf_input(
        e2_handoff=handoff,
        output=tmp_path / "rf_input.zip",
        expected_e2_sha256=digest,
    )

    def change(name: str, payload: bytes) -> bytes:
        if name != "manifest.json":
            return payload
        manifest = json.loads(payload)
        manifest["e2_candidate_id"] = "c2_trackman_residual"
        return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    altered = _rewrite_outer(archive, tmp_path / "altered.zip", change)
    with pytest.raises(RFInputError, match="RF input identity differs"):
        verify_and_extract_rf_input(altered, tmp_path / "verified", expected_e2_sha256=digest)


def test_rf_input_accepts_expanded_directory(tmp_path: Path) -> None:
    handoff = make_e2_handoff(tmp_path / "e2.zip")
    digest = file_sha256(handoff)
    archive = prepare_rf_input(
        e2_handoff=handoff,
        output=tmp_path / "rf_input.zip",
        expected_e2_sha256=digest,
    )
    expanded = tmp_path / "expanded"
    with ZipFile(archive) as source:
        source.extractall(expanded)

    verified = verify_and_extract_rf_input(
        expanded,
        tmp_path / "verified",
        expected_e2_sha256=digest,
    )
    assert verified.manifest_sha256
