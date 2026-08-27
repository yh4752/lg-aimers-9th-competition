from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
import pytest

from experiments.tree_expert.hc_inputs import (
    HCInputError,
    choose_unique_hc_input,
    prepare_hc_input,
    verify_and_extract_hc_input,
)


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(_info(name), payload)
    return output.getvalue()


def _manifest(kind: str, members: dict[str, bytes], **extra: object) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "artifact_kind": kind,
            **extra,
            "members": {
                name: {"size": len(payload), "sha256": sha256(payload).hexdigest()}
                for name, payload in sorted(members.items())
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _prediction(valid_year: int) -> bytes:
    return pd.DataFrame(
        {
            "row_id": [f"r{valid_year}a", f"r{valid_year}b"],
            "target": [0, 1],
            "probability": [0.4, 0.6],
            "game_type": ["R", "F"],
            "game_month": [3, 4],
            "pitcher_id_known": ["known", "oov"],
            "batter_id_known": ["known", "known"],
        }
    ).to_csv(index=False).encode()


def _delivery() -> bytes:
    payloads = {
        "evidence/acceptance_decision.json": b"{}",
        "evidence/full_fit_manifest.json": b"{}",
        "evidence/inference_audit.json": b"{}",
        "frozen_state/feature_state.json": b"{}",
        "models/catboost_seed_42.cbm": b"42",
        "models/catboost_seed_2026.cbm": b"2026",
        "models/catboost_seed_3407.cbm": b"3407",
    }
    manifest = _manifest(
        "tree_expert_e2_model_delivery_v1",
        payloads,
        campaign_id="tree_expert_e2_v1",
        review_only=False,
        submission_package=False,
        candidate_id="c1_anchor_residual",
        predictor="catboost",
        seeds=[42, 2026, 3407],
        iterations={"42": 50, "2026": 50, "3407": 50},
        decision_sha256="1" * 64,
    )
    return _zip_bytes({**payloads, "manifest.json": manifest})


def _accepted_handoff(path: Path) -> Path:
    review_members = {
        "decisions/acceptance.json": b'{"predictor":"catboost","status":"accepted"}',
        "ensembles/2021_2022.csv": _prediction(2022),
        "ensembles/2022_2023.csv": _prediction(2023),
        "ensembles/2023_2024.csv": _prediction(2024),
    }
    review = _zip_bytes(review_members)
    payloads = {
        "tree_expert_e2_review.zip": review,
        "tree_expert_e2_resume.zip": b"resume",
        "tree_expert_e2.log": b"ok\n",
        "tree_expert_e2_model_delivery.zip": _delivery(),
    }
    manifest = _manifest(
        "tree_expert_e2_handoff_v1",
        payloads,
        campaign_id="tree_expert_e2_v1",
        review_only=False,
        submission_package=False,
        status="accepted",
        delivery=True,
    )
    path.write_bytes(_zip_bytes({**payloads, "handoff_manifest.json": manifest}))
    return path


def test_prepare_and_verify_hc_input_round_trip(tmp_path):
    handoff = _accepted_handoff(tmp_path / "e2.zip")
    expected = sha256(handoff.read_bytes()).hexdigest()
    archive = prepare_hc_input(
        e2_handoff=handoff,
        output=tmp_path / "hc_input.zip",
        expected_e2_sha256=expected,
    )
    verified = verify_and_extract_hc_input(
        archive,
        tmp_path / "verified",
        expected_e2_sha256=expected,
    )
    assert verified.e2_handoff_sha256 == expected
    assert verified.candidate_id == "c1_anchor_residual"
    assert set(verified.fold_predictions) == {
        (2021, 2022),
        (2022, 2023),
        (2023, 2024),
    }
    assert verified.e2_delivery.is_file()


def test_prepare_rejects_wrong_e2_hash(tmp_path):
    handoff = _accepted_handoff(tmp_path / "e2.zip")
    with pytest.raises(HCInputError, match="E2 handoff SHA-256 differs"):
        prepare_hc_input(
            e2_handoff=handoff,
            output=tmp_path / "hc_input.zip",
            expected_e2_sha256="0" * 64,
        )


def test_changed_member_is_rejected(tmp_path):
    handoff = _accepted_handoff(tmp_path / "e2.zip")
    expected = sha256(handoff.read_bytes()).hexdigest()
    archive = prepare_hc_input(
        e2_handoff=handoff,
        output=tmp_path / "hc_input.zip",
        expected_e2_sha256=expected,
    )
    changed = tmp_path / "changed.zip"
    with ZipFile(archive) as source, ZipFile(changed, "w") as target:
        for info in source.infolist():
            payload = source.read(info)
            if info.filename == "e2/fold_2022_2023.csv":
                payload += b"\n"
            target.writestr(_info(info.filename), payload)
    with pytest.raises(HCInputError, match="member SHA-256 differs"):
        verify_and_extract_hc_input(
            changed,
            tmp_path / "changed_out",
            expected_e2_sha256=expected,
        )


def test_zip_and_expanded_copy_with_same_identity_are_deduplicated(tmp_path):
    handoff = _accepted_handoff(tmp_path / "e2.zip")
    expected = sha256(handoff.read_bytes()).hexdigest()
    archive = prepare_hc_input(
        e2_handoff=handoff,
        output=tmp_path / "hc_input.zip",
        expected_e2_sha256=expected,
    )
    expanded = tmp_path / "expanded"
    with ZipFile(archive) as source:
        source.extractall(expanded)
    assert choose_unique_hc_input([archive, expanded]) == archive


def test_expanded_kaggle_input_ignores_unregistered_wrapper_files(tmp_path):
    handoff = _accepted_handoff(tmp_path / "e2.zip")
    expected = sha256(handoff.read_bytes()).hexdigest()
    archive = prepare_hc_input(
        e2_handoff=handoff,
        output=tmp_path / "hc_input.zip",
        expected_e2_sha256=expected,
    )
    expanded = tmp_path / "kaggle_dataset"
    with ZipFile(archive) as source:
        source.extractall(expanded)
    (expanded / "dataset-metadata.json").write_text('{"title":"wrapper"}')
    (expanded / "hc_input_original.zip").write_bytes(archive.read_bytes())

    verified = verify_and_extract_hc_input(
        expanded,
        tmp_path / "verified_expanded",
        expected_e2_sha256=expected,
    )
    assert verified.e2_handoff_sha256 == expected


def test_two_distinct_input_identities_are_rejected(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "manifest.json").write_text('{"artifact_kind":"tree_hierarchical_input_v1","identity":"a"}')
    (second / "manifest.json").write_text('{"artifact_kind":"tree_hierarchical_input_v1","identity":"b"}')
    with pytest.raises(HCInputError, match="distinct HC input identities"):
        choose_unique_hc_input([first, second])


def test_unsafe_parent_member_is_rejected(tmp_path):
    unsafe = tmp_path / "unsafe.zip"
    with ZipFile(unsafe, "w") as archive:
        archive.writestr("../manifest.json", b"{}")
    with pytest.raises(HCInputError, match="unsafe"):
        verify_and_extract_hc_input(unsafe, tmp_path / "out", expected_e2_sha256="0" * 64)
