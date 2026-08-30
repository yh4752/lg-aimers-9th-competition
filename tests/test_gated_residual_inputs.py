from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.gated_residual_final.inputs import (
    FinalInputError,
    prepare_final_input,
    verify_and_extract_final_input,
)


def _sources(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    paths = []
    for name, payload in (("e2.zip", b"e2"), ("stage_a.zip", b"a"), ("stage_b.zip", b"b")):
        path = tmp_path / name
        path.write_bytes(payload)
        paths.append(path)
    hashes = {name: sha256(path.read_bytes()).hexdigest() for name, path in zip(("e2_input", "stage_a", "stage_b"), paths)}
    return paths[0], paths[1], paths[2], hashes


def _prepared(tmp_path: Path) -> Path:
    e2, stage_a, stage_b, hashes = _sources(tmp_path)
    return prepare_final_input(
        e2_input=e2,
        stage_a=stage_a,
        stage_b=stage_b,
        output=tmp_path / "final.zip",
        expected_hashes=hashes,
    )


def test_round_trip_renames_every_nested_archive(tmp_path: Path) -> None:
    archive = _prepared(tmp_path)

    with ZipFile(archive) as handle:
        assert set(handle.namelist()) == {
            "e2/input.bin", "direct/stage_a.bin", "direct/stage_b.bin",
            "sources.json", "manifest.json",
        }
        assert not any(name.endswith(".zip") for name in handle.namelist())

    verified = verify_and_extract_final_input(archive, tmp_path / "verified")
    assert verified.e2_input.name == "direct_expert_input.zip"
    assert verified.stage_a.name == "direct_expert_stage_A_handoff.zip"
    assert verified.stage_b.name == "direct_expert_stage_B_handoff.zip"


def test_expanded_kaggle_dataset_has_same_identity(tmp_path: Path) -> None:
    archive = _prepared(tmp_path)
    expanded = tmp_path / "expanded"
    with ZipFile(archive) as handle:
        handle.extractall(expanded)
    (expanded / "dataset-metadata.json").write_text("{}")

    left = verify_and_extract_final_input(archive, tmp_path / "left")
    right = verify_and_extract_final_input(expanded, tmp_path / "right")

    assert left.manifest_sha256 == right.manifest_sha256
    assert left.source_hashes == right.source_hashes


def test_changed_member_is_rejected(tmp_path: Path) -> None:
    archive = _prepared(tmp_path)
    expanded = tmp_path / "expanded"
    with ZipFile(archive) as handle:
        handle.extractall(expanded)
    (expanded / "direct/stage_b.bin").write_bytes(b"changed")

    with pytest.raises(FinalInputError, match="member digest differs"):
        verify_and_extract_final_input(expanded, tmp_path / "bad")


def test_existing_destination_is_rejected(tmp_path: Path) -> None:
    archive = _prepared(tmp_path)
    destination = tmp_path / "exists"
    destination.mkdir()

    with pytest.raises(FinalInputError, match="destination already exists"):
        verify_and_extract_final_input(archive, destination)
