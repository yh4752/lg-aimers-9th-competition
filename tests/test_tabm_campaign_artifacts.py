from __future__ import annotations

import io
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from experiments.tabm_campaign.artifacts import (
    ArtifactError,
    StageEvidence,
    verify_review_bundle,
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
    review = verify_review_bundle(first.review)
    assert review.version == "A"
    assert set(review.member_sha256) == {"metrics/candidates.json", "logs/stage.log"}


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


def test_stage_p_uses_independent_prefix_and_optional_prior_chain(tmp_path: Path) -> None:
    evidence = StageEvidence(
        "P",
        "2" * 64,
        None,
        {"state/stage_state.json": b"{}"},
        {"state/stage_state.json": b"{}"},
    )

    first = write_stage_bundles(
        tmp_path / "first",
        evidence,
        bundle_prefix="tabm_row_feature_stage",
    )
    assert first.review.name == "tabm_row_feature_stage_P_review_bundle.zip"
    assert first.resume is not None
    assert first.resume.name == "tabm_row_feature_stage_P_resume_bundle.zip"
    assert verify_review_bundle(first.review).prior_manifest_sha256 is None
    assert verify_resume_bundle(first.resume).version == "P"

    resumed = write_stage_bundles(
        tmp_path / "resumed",
        StageEvidence(
            "P",
            "2" * 64,
            first.manifest_sha256,
            evidence.review_members,
            evidence.resume_members,
        ),
        bundle_prefix="tabm_row_feature_stage",
    )
    assert verify_resume_bundle(resumed.resume).prior_manifest_sha256 == first.manifest_sha256


@pytest.mark.parametrize(
    "prefix",
    ["", ".", "..", "../stage", "stage/name", r"stage\name"],
)
def test_bundle_prefix_must_be_one_safe_filename_component(
    tmp_path: Path, prefix: str
) -> None:
    with pytest.raises(ArtifactError, match="prefix"):
        write_stage_bundles(tmp_path, _evidence(), bundle_prefix=prefix)


def test_default_stage_a_bundle_bytes_and_names_remain_pinned(tmp_path: Path) -> None:
    bundle = write_stage_bundles(tmp_path, _evidence())

    assert bundle.review.name == "tabm_search_stage_A_review_bundle.zip"
    assert bundle.resume is not None
    assert bundle.resume.name == "tabm_search_stage_A_resume_bundle.zip"
    assert bundle.review_sha256 == "7b7a8783ff6d365e0b32abf5496be9ec6686068656bdbcc72aa9d27443f6592a"
    assert bundle.resume_sha256 == "366efe0d8733cd110a8f1e6aaf246beadf8118c613c382556d374f347186e598"
    assert bundle.manifest_sha256 == "115b675768a4d10792e76df88b45cc5c32d514496b18331b8ddc70333e5ec687"


def test_verifier_rejects_unknown_manifest_schema(tmp_path: Path) -> None:
    bundle = write_stage_bundles(tmp_path, _evidence()).resume
    assert bundle is not None
    with ZipFile(bundle, "r") as source:
        members = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["schema_version"] = 2
    members["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as target:
        for name, value in members.items():
            target.writestr(name, value)

    with pytest.raises(ArtifactError, match="schema"):
        verify_resume_bundle(bundle)
