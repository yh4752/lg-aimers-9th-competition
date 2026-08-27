from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
import pytest

import experiments.tree_expert.t3_inputs as t3_inputs

from experiments.tree_expert.t3_inputs import (
    T3InputError,
    file_sha256,
    prepare_t3_input,
    verify_and_extract_t3_input,
)


def zip_info(name):
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def make_e2_handoff(path: Path, *, extra_payload: bytes = b""):
    folds = {}
    for train_end, valid in ((2021, 2022), (2022, 2023), (2023, 2024)):
        folds[f"ensembles/{train_end}_{valid}.csv"] = pd.DataFrame({
            "row_id": [f"r{valid}a", f"r{valid}b"], "target": [0, 1],
            "probability": [0.45, 0.55], "game_type": ["R", "F"],
            "game_month": [3, 4], "pitcher_id_known": ["known", "oov"],
            "batter_id_known": ["known", "known"],
        }).to_csv(index=False).encode()
    review_buffer = io.BytesIO()
    with ZipFile(review_buffer, "w") as archive:
        for name, payload in folds.items():
            archive.writestr(zip_info(name), payload)
        archive.writestr(zip_info("decisions/acceptance.json"), b'{"status":"accepted","predictor":"catboost"}')
    review = review_buffer.getvalue()
    handoff = {
        "artifact_kind": "tree_expert_e2_handoff_v1", "campaign_id": "tree_expert_e2_v1",
        "delivery": True, "review_only": False, "schema_version": 1,
        "status": "accepted", "submission_package": False,
        "members": {
            "tree_expert_e2_review.zip": {"sha256": sha256(review).hexdigest(), "size": len(review)}
        },
    }
    with ZipFile(path, "w") as archive:
        archive.writestr(zip_info("handoff_manifest.json"), json.dumps(handoff, sort_keys=True).encode())
        archive.writestr(zip_info("tree_expert_e2_review.zip"), review)
        if extra_payload:
            archive.writestr(zip_info("tree_expert_e2_resume.zip"), extra_payload)
    return path


def test_prepare_and_verify_t3_input_round_trip(tmp_path):
    handoff = make_e2_handoff(tmp_path / "e2.zip")
    digest = file_sha256(handoff)
    archive = prepare_t3_input(
        e2_handoff=handoff, output=tmp_path / "t3_input.zip", expected_e2_sha256=digest,
    )
    verified = verify_and_extract_t3_input(
        archive, tmp_path / "verified", expected_e2_sha256=digest,
    )
    assert verified.e2_candidate_id == "c1_anchor_residual"
    assert verified.e2_handoff_sha256 == digest
    assert set(verified.fold_predictions) == {(2021, 2022), (2022, 2023), (2023, 2024)}


def test_t3_input_rejects_changed_prediction_bytes(tmp_path):
    handoff = make_e2_handoff(tmp_path / "e2.zip")
    digest = file_sha256(handoff)
    archive = prepare_t3_input(
        e2_handoff=handoff, output=tmp_path / "t3_input.zip", expected_e2_sha256=digest,
    )
    changed = tmp_path / "changed.zip"
    with ZipFile(archive) as source, ZipFile(changed, "w") as target:
        for info in source.infolist():
            payload = source.read(info)
            if info.filename == "e2/fold_2022_2023.csv":
                payload += b"\n"
            target.writestr(zip_info(info.filename), payload)
    with pytest.raises(T3InputError, match="member SHA-256 differs"):
        verify_and_extract_t3_input(changed, tmp_path / "verified", expected_e2_sha256=digest)


def test_e2_handoff_uses_a_separate_larger_expansion_limit(tmp_path, monkeypatch):
    handoff = make_e2_handoff(tmp_path / "e2.zip", extra_payload=b"x" * 200)
    digest = file_sha256(handoff)
    monkeypatch.setattr(t3_inputs, "_MAX_EXPANDED", 100)
    monkeypatch.setattr(t3_inputs, "_MAX_E2_HANDOFF_EXPANDED", 1024 * 1024, raising=False)
    result = prepare_t3_input(
        e2_handoff=handoff, output=tmp_path / "t3_input.zip", expected_e2_sha256=digest,
    )
    assert result.is_file()
