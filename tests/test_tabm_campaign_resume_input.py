from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from experiments.tabm_campaign.artifacts import ArtifactError, verify_resume_bundle
from experiments.tabm_campaign.resume_input import normalize_resume_input


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _extracted_resume(root: Path, *, version: str = "A", marker: bytes = b"state") -> Path:
    source = root / f"tabm_search_stage_{version}_resume_bundle"
    members = {
        "stage_state.json": _canonical(
            {"version": version, "stage_complete": True, "marker": marker.decode()}
        ),
        "training/candidate/checkpoint.pt": b"checkpoint-" + marker,
    }
    for name, value in members.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "resume",
        "review_only": True,
        "version": version,
        "campaign_config_sha256": "a" * 64,
        "prior_manifest_sha256": None if version == "A" else "b" * 64,
        "members": {name: sha256(value).hexdigest() for name, value in members.items()},
    }
    (source / "manifest.json").write_bytes(_canonical(manifest))
    return source


def _zip_extracted(source: Path, destination: Path) -> Path:
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    return destination


def test_extracted_resume_is_verified_and_rebuilt_for_existing_runner(
    tmp_path: Path,
) -> None:
    source = _extracted_resume(tmp_path / "input")

    result = normalize_resume_input(tmp_path / "input", tmp_path / "working")

    assert result.source == "extracted"
    assert result.original_path == source
    assert result.path is not None and result.path.is_file()
    assert result.version == "A"
    verified = verify_resume_bundle(result.path)
    assert verified.manifest_sha256 == result.manifest_sha256


def test_intact_zip_and_its_extracted_copy_are_one_logical_resume(
    tmp_path: Path,
) -> None:
    source = _extracted_resume(tmp_path / "input")
    intact = _zip_extracted(
        source, tmp_path / "input" / "tabm_search_stage_A_resume_bundle.zip"
    )

    result = normalize_resume_input(tmp_path / "input", tmp_path / "working")

    assert result.source == "zip"
    assert result.path == intact
    assert result.version == "A"


def test_extracted_resume_rejects_member_hash_mismatch(tmp_path: Path) -> None:
    source = _extracted_resume(tmp_path / "input")
    (source / "stage_state.json").write_bytes(b"modified")

    with pytest.raises(ArtifactError, match="SHA-256 differs"):
        normalize_resume_input(tmp_path / "input", tmp_path / "working")


def test_different_resume_inputs_are_rejected_as_ambiguous(tmp_path: Path) -> None:
    _extracted_resume(tmp_path / "input" / "first", marker=b"first")
    _extracted_resume(tmp_path / "input" / "second", marker=b"second")

    with pytest.raises(ArtifactError, match="multiple different resume inputs"):
        normalize_resume_input(tmp_path / "input", tmp_path / "working")


def test_no_resume_input_returns_explicit_none(tmp_path: Path) -> None:
    result = normalize_resume_input(tmp_path / "input", tmp_path / "working")

    assert result.source == "none"
    assert result.path is None
    assert result.version is None
