from __future__ import annotations

import io
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from experiments.tabm_campaign.artifacts import (
    ArtifactError,
    StageEvidence,
    verify_resume_bundle,
    write_stage_bundles,
)


def _evidence() -> StageEvidence:
    return StageEvidence(
        version="A",
        campaign_config_sha256="1" * 64,
        prior_manifest_sha256=None,
        review_members={
            "metrics/candidates.json": b'[{"brier":0.2}]',
            "logs/stage.log": b"done\n",
        },
        resume_members={"stage_state.json": b'{"completed":["a"]}', "checkpoints/a.bin": b"weights"},
    )


def test_bundle_is_reproducible_and_hash_validated(tmp_path: Path) -> None:
    first = write_stage_bundles(tmp_path / "a", _evidence())
    second = write_stage_bundles(tmp_path / "b", _evidence())
    assert sha256(first.review.read_bytes()).hexdigest() == sha256(second.review.read_bytes()).hexdigest()
    assert sha256(first.resume.read_bytes()).hexdigest() == sha256(second.resume.read_bytes()).hexdigest()
    verified = verify_resume_bundle(first.resume)
    assert verified.version == "A"


def test_resume_rejects_modified_member(tmp_path: Path) -> None:
    bundle = write_stage_bundles(tmp_path, _evidence()).resume
    with ZipFile(bundle, "r") as source:
        members = {name: source.read(name) for name in source.namelist()}
    members["stage_state.json"] = b"tampered"
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as target:
        for name, value in members.items():
            target.writestr(name, value)
    bundle.write_bytes(buffer.getvalue())
    with pytest.raises(ArtifactError, match="SHA-256"):
        verify_resume_bundle(bundle)


def test_stage_transition_and_member_paths_are_strict(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="prior manifest"):
        write_stage_bundles(
            tmp_path,
            StageEvidence("B", "1" * 64, None, {"metrics.json": b"{}"}, {"state.json": b"{}"}),
        )
    with pytest.raises(ArtifactError, match="unsafe"):
        write_stage_bundles(
            tmp_path,
            StageEvidence("A", "1" * 64, None, {"../escape": b"x"}, {"state.json": b"{}"}),
        )
