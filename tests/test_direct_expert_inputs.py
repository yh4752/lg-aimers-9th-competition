from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.direct_expert.inputs import (
    DirectExpertInputError,
    prepare_direct_expert_input,
    verify_and_extract_input,
)
from tests.direct_expert_fixtures import (
    make_e2_submission,
    make_s4_handoff,
    rewrite_member,
)


def _prepared(tmp_path: Path) -> tuple[Path, str, str]:
    s4, logical_e2 = make_s4_handoff(tmp_path)
    e2, receipt = make_e2_submission(tmp_path)
    output = prepare_direct_expert_input(
        s4,
        e2,
        receipt,
        tmp_path / "input.zip",
        expected_s4_sha256=sha256(s4.read_bytes()).hexdigest(),
        expected_e2_sha256=sha256(e2.read_bytes()).hexdigest(),
    )
    return output, logical_e2, sha256(e2.read_bytes()).hexdigest()


def test_input_contains_s4_e2_oof_and_accepted_e2_runtime(tmp_path: Path) -> None:
    output, logical_e2, e2_digest = _prepared(tmp_path)
    verified = verify_and_extract_input(output, tmp_path / "verified")

    assert verified.e2_oof_years == (2021, 2022, 2023, 2024)
    assert verified.e2_submission_sha256 == e2_digest
    assert verified.logical_e2_handoff_sha256 == logical_e2


def test_expanded_kaggle_directory_matches_zip(tmp_path: Path) -> None:
    prepared, _, _ = _prepared(tmp_path)
    expanded = tmp_path / "expanded"
    with ZipFile(prepared) as archive:
        archive.extractall(expanded)

    left = verify_and_extract_input(prepared, tmp_path / "left")
    right = verify_and_extract_input(expanded, tmp_path / "right")
    assert left.manifest_sha256 == right.manifest_sha256


def test_expanded_kaggle_directory_ignores_unreferenced_dataset_metadata(tmp_path: Path) -> None:
    prepared, _, _ = _prepared(tmp_path)
    expanded = tmp_path / "expanded"
    with ZipFile(prepared) as archive:
        archive.extractall(expanded)
    (expanded / "dataset-metadata.json").write_text("{}")

    verified = verify_and_extract_input(expanded, tmp_path / "verified")

    assert verified.e2_oof_years == (2021, 2022, 2023, 2024)


def test_expanded_kaggle_directory_rebuilds_nested_submission_zip(tmp_path: Path) -> None:
    prepared, _, e2_digest = _prepared(tmp_path)
    expanded = tmp_path / "expanded"
    with ZipFile(prepared) as archive:
        archive.extractall(expanded)
    nested_zip = expanded / "e2_submission/catboost_3seed_v1.zip"
    nested_directory = nested_zip.with_suffix("")
    with ZipFile(nested_zip) as archive:
        archive.extractall(nested_directory)
    nested_zip.unlink()

    verified = verify_and_extract_input(expanded, tmp_path / "verified")

    assert verified.e2_submission_sha256 == e2_digest


def test_tampered_e2_oof_is_rejected(tmp_path: Path) -> None:
    prepared, _, _ = _prepared(tmp_path)
    changed = rewrite_member(
        prepared,
        "e2_oof/2024.csv",
        b"row_id,target,p_anchor\nX,0,1\n",
    )

    with pytest.raises(DirectExpertInputError, match="member digest differs"):
        verify_and_extract_input(changed, tmp_path / "bad")


def test_existing_or_symlink_destination_is_rejected(tmp_path: Path) -> None:
    prepared, _, _ = _prepared(tmp_path)
    destination = tmp_path / "exists"
    destination.mkdir()

    with pytest.raises(DirectExpertInputError, match="destination already exists"):
        verify_and_extract_input(prepared, destination)
