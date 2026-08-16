from __future__ import annotations

import io
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.tabm_campaign import artifacts as artifact_module
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


def test_verifier_rejects_extreme_compression_ratio_before_read(tmp_path: Path) -> None:
    evidence = StageEvidence(
        "P",
        "3" * 64,
        None,
        {"state/stage_state.json": b"x"},
        {"jobs/rfp__baseline__s42/checkpoint.pt": b"0" * (1024 * 1024)},
    )

    with pytest.raises(ArtifactError, match="compression ratio"):
        write_stage_bundles(tmp_path, evidence)


def test_verifier_rejects_duplicate_member_before_read(tmp_path: Path) -> None:
    source = write_stage_bundles(tmp_path / "source", _evidence()).resume
    assert source is not None
    with ZipFile(source) as archive:
        members = [(name, archive.read(name)) for name in archive.namelist()]
    forged = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(forged, "w", compression=ZIP_DEFLATED) as archive:
            for name, value in members:
                archive.writestr(name, value)
            archive.writestr("stage_state.json", b"duplicate")

    with pytest.raises(ArtifactError, match="duplicate"):
        verify_resume_bundle(forged)


def test_verifier_rejects_traversal_and_symlink_members_before_read(
    tmp_path: Path,
) -> None:
    source = write_stage_bundles(tmp_path / "source", _evidence()).resume
    assert source is not None
    with ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}

    traversal = tmp_path / "traversal.zip"
    manifest = json.loads(members["manifest.json"])
    manifest["members"]["../escape"] = sha256(b"escape").hexdigest()
    with ZipFile(traversal, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            if name == "manifest.json":
                value = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            archive.writestr(name, value)
        archive.writestr("../escape", b"escape")
    with pytest.raises(ArtifactError, match="unsafe"):
        verify_resume_bundle(traversal)

    symlink = tmp_path / "symlink.zip"
    with ZipFile(symlink, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            if name == "stage_state.json":
                info = ZipInfo(name)
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                archive.writestr(info, value)
            else:
                archive.writestr(name, value)
    with pytest.raises(ArtifactError, match="regular file"):
        verify_resume_bundle(symlink)


def test_verifier_enforces_member_and_total_uncompressed_size_limits(
    tmp_path: Path, monkeypatch
) -> None:
    evidence = StageEvidence(
        "P",
        "4" * 64,
        None,
        {"state/stage_state.json": b"review"},
        {"one.bin": b"12345678", "two.bin": b"abcdefgh"},
    )
    monkeypatch.setattr(artifact_module, "_MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES", 7)
    with pytest.raises(ArtifactError, match="member exceeds"):
        write_stage_bundles(tmp_path / "member", evidence)

    monkeypatch.setattr(artifact_module, "_MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES", 1000)
    monkeypatch.setattr(artifact_module, "_MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES", 100)
    with pytest.raises(ArtifactError, match="total uncompressed"):
        write_stage_bundles(tmp_path / "total", evidence)


@pytest.mark.parametrize("kind", ["review", "resume"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 1),
        ("version", []),
        ("campaign_config_sha256", 1),
        ("campaign_config_sha256", []),
        ("campaign_config_sha256", None),
        ("prior_manifest_sha256", 1),
        ("prior_manifest_sha256", []),
        ("member_hash", 1),
        ("member_hash", []),
        ("member_hash", None),
    ],
)
def test_verifiers_reject_non_string_manifest_bindings_as_artifact_error(
    tmp_path: Path, kind: str, field: str, value: object
) -> None:
    paths = write_stage_bundles(tmp_path / "source", _evidence())
    source = paths.review if kind == "review" else paths.resume
    assert source is not None
    target = tmp_path / f"{kind}-{field}.zip"
    with ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    if field == "member_hash":
        member = next(iter(manifest["members"]))
        manifest["members"][member] = value
    else:
        manifest[field] = value
    members["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name, member_value in members.items():
            archive.writestr(name, member_value)

    verifier = verify_review_bundle if kind == "review" else verify_resume_bundle
    with pytest.raises(ArtifactError):
        verifier(target)


def test_stage_p_stream_writer_rejects_symlink_swap_after_manifest_hash(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "checkpoint.pt"
    replacement = tmp_path / "replacement.pt"
    source.write_bytes(b"same bytes")
    replacement.write_bytes(b"same bytes")
    expected_sha256 = sha256(b"same bytes").hexdigest()
    original_open = artifact_module._open_regular_descriptor
    open_count = 0

    def open_then_swap(path: Path):
        nonlocal open_count
        if path == source:
            open_count += 1
        if path == source and open_count == 2:
            source.unlink()
            source.symlink_to(replacement)
        return original_open(path)

    monkeypatch.setattr(artifact_module, "_open_regular_descriptor", open_then_swap)
    evidence = StageEvidence(
        "P",
        "5" * 64,
        None,
        {"state/stage_state.json": b"{}"},
        {
            "jobs/rfp__baseline__s42/checkpoint.pt": artifact_module._StagePFile(
                source, expected_sha256
            )
        },
    )

    with pytest.raises(ArtifactError, match="safe regular file"):
        write_stage_bundles(tmp_path / "out", evidence)
