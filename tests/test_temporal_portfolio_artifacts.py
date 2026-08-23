from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from hashlib import sha256
import io
import itertools
import json
from pathlib import Path
import struct
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.temporal_portfolio import artifacts as artifact_module
from experiments.temporal_portfolio.artifacts import (
    BoundFile,
    HandoffPath,
    PortfolioArtifactError,
    StageEvidence,
    bind_file,
    discover_handoffs,
    verify_handoff,
    write_handoff,
)
from experiments.temporal_portfolio.state import Bindings, Lineage


def _bindings(digit: str = "1") -> Bindings:
    return Bindings("temporal_portfolio_v1", digit * 64, "2" * 64)


def _lineage(
    *,
    sequence: int = 1,
    parent: str = "2" * 64,
    training: str = "3" * 64,
    runtime: str = "4" * 64,
    stage: str = "T1",
) -> Lineage:
    return Lineage(
        "temporal_portfolio_v1",
        stage,
        sequence,
        parent,
        training,
        1,
        runtime,
    )


def _evidence(
    *,
    bindings: Bindings | None = None,
    lineage: Lineage | None = None,
    marker: str = "base",
) -> StageEvidence:
    return StageEvidence(
        stage=(lineage or _lineage()).stage,
        bindings=bindings or _bindings(),
        lineage=lineage or _lineage(),
        review_members={"metrics/review.json": b'{"accepted":true}'},
        resume_members={
            "state/campaign.json": b'{"status":"active"}',
            "checkpoints/model.bin": marker.encode("utf-8"),
        },
        run_log=b"STAGE_COMPLETE stage=T1\n",
        summary={"stage": "T1", "metrics": {"brier": 0.19}},
    )


def _zip_members(path: Path) -> dict[str, bytes]:
    with ZipFile(path) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def _zip_bytes(members: list[tuple[str | ZipInfo, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members:
            archive.writestr(name, value)
    return buffer.getvalue()


def _rewrite_outer(
    source: Path,
    destination: Path,
    *,
    replace: dict[str, bytes] | None = None,
    manifest_transform=None,
    extras: list[tuple[str | ZipInfo, bytes]] | None = None,
    canonical_manifest: bool = True,
) -> Path:
    members = _zip_members(source)
    members.update(replace or {})
    manifest = json.loads(members["handoff_manifest.json"])
    for name, value in (replace or {}).items():
        if name != "handoff_manifest.json" and name in manifest["members"]:
            manifest["members"][name] = {
                "sha256": sha256(value).hexdigest(),
                "size_bytes": len(value),
            }
    if manifest_transform is not None:
        manifest_transform(manifest)
    if canonical_manifest:
        members["handoff_manifest.json"] = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    else:
        members["handoff_manifest.json"] = json.dumps(manifest, indent=2).encode()
    destination.write_bytes(
        _zip_bytes([(name, value) for name, value in members.items()] + (extras or []))
    )
    return destination


def _patch_encrypted_flags(value: bytes) -> bytes:
    patched = bytearray(value)
    offset = 0
    while True:
        offset = patched.find(b"PK\x03\x04", offset)
        if offset < 0:
            break
        flags = struct.unpack_from("<H", patched, offset + 6)[0]
        struct.pack_into("<H", patched, offset + 6, flags | 1)
        offset += 4
    offset = 0
    while True:
        offset = patched.find(b"PK\x01\x02", offset)
        if offset < 0:
            break
        flags = struct.unpack_from("<H", patched, offset + 8)[0]
        struct.pack_into("<H", patched, offset + 8, flags | 1)
        offset += 4
    return bytes(patched)


def _patch_eocd(value: bytes, offset: int, fmt: str, replacement: int) -> bytes:
    patched = bytearray(value)
    eocd = patched.rfind(b"PK\x05\x06")
    assert eocd >= 0
    struct.pack_into(fmt, patched, eocd + offset, replacement)
    return bytes(patched)


class _ItemsMapping(Mapping[str, object]):
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        self.items_calls += 1
        return iter(self._items)


def test_handoff_round_trip_is_deterministic_across_roots(tmp_path: Path) -> None:
    evidence = _evidence()
    first = write_handoff(tmp_path / "a", evidence)
    second = write_handoff(tmp_path / "b", evidence)

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    verified = verify_handoff(first.path)
    assert verified.stage == "T1"
    assert verified.sequence == 1
    assert verified.sha256 == first.sha256
    assert set(verified.members) == {
        "review.zip",
        "resume.zip",
        "run.log",
        "stage_summary.json",
    }
    with ZipFile(first.path) as archive:
        manifest_bytes = archive.read("handoff_manifest.json")
        manifest = json.loads(manifest_bytes)
    assert verified.manifest_sha256 == sha256(manifest_bytes).hexdigest()
    assert manifest["artifact_kind"] == "temporal_handoff_v1"
    assert manifest["submission_package"] is False
    assert manifest["bindings"] == {
        "campaign_id": "temporal_portfolio_v1",
        "contract_sha256": "1" * 64,
        "input_manifest_sha256": "2" * 64,
    }
    assert manifest["lineage"] == {
        "campaign_id": "temporal_portfolio_v1",
        "parent_manifest_sha256": "2" * 64,
        "runtime_sha256": "4" * 64,
        "sequence": 1,
        "stage": "T1",
        "state_schema_version": 1,
        "training_sha256": "3" * 64,
    }
    assert all(set(record) == {"sha256", "size_bytes"} for record in manifest["members"].values())


@pytest.mark.parametrize("mode", ["zip", "directory"])
def test_discovery_reads_explicit_zip_or_kaggle_extracted_directory(
    tmp_path: Path, mode: str
) -> None:
    handoff = write_handoff(tmp_path / "source", _evidence()).path
    if mode == "zip":
        root = handoff
    else:
        root = tmp_path / "input" / "dataset-slug" / "published-artifact"
        root.mkdir(parents=True)
        with ZipFile(handoff) as archive:
            archive.extractall(root)

    found = discover_handoffs([root])
    assert len(found) == 1
    assert found[0].manifest_sha256 == verify_handoff(handoff).manifest_sha256


def test_stage_evidence_and_return_values_are_immutable_snapshots(tmp_path: Path) -> None:
    reviews = {"metrics.json": b"original"}
    resumes = {"state.json": b"state"}
    nested = {"score": 1}
    summary = {"nested": nested}
    evidence = StageEvidence(
        "T1", _bindings(), _lineage(), reviews, resumes, b"log", summary
    )
    reviews["metrics.json"] = b"changed"
    resumes["new.bin"] = b"new"
    nested["score"] = 99
    summary["other"] = True

    output = write_handoff(tmp_path, evidence)
    verified = verify_handoff(output.path)
    with ZipFile(output.path) as outer:
        assert json.loads(outer.read("stage_summary.json")) == {"nested": {"score": 1}}
        review = outer.read("review.zip")
    with ZipFile(io.BytesIO(review)) as nested_zip:
        assert nested_zip.read("metrics.json") == b"original"
    assert "new.bin" not in verified.resume_members
    with pytest.raises(TypeError):
        evidence.review_members["x"] = b"x"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        output.sha256 = "0" * 64  # type: ignore[misc]


def test_stage_evidence_requires_authoritative_state_schema_version() -> None:
    unsupported = Lineage(
        "temporal_portfolio_v1",
        "T1",
        1,
        "2" * 64,
        "3" * 64,
        2,
        "4" * 64,
    )

    with pytest.raises(PortfolioArtifactError, match="schema"):
        _evidence(lineage=unsupported)


@pytest.mark.parametrize(
    "items",
    [
        [("duplicate", 1), ("duplicate", 2)],
        [["not", "a tuple"]],
        [("too", "many", "values")],
        ["not an item pair"],
    ],
)
def test_stage_summary_snapshot_rejects_duplicate_and_malformed_mapping_items(
    items: list[object],
) -> None:
    summary = _ItemsMapping(items)

    with pytest.raises(PortfolioArtifactError, match="duplicate|malformed"):
        StageEvidence(
            "T1",
            _bindings(),
            _lineage(),
            {"metrics.json": b"{}"},
            {"state.json": b"{}"},
            b"log",
            summary,
        )
    assert summary.items_calls == 1


def test_stage_summary_mapping_is_snapshotted_exactly_once(tmp_path: Path) -> None:
    summary = _ItemsMapping([("score", 1)])
    evidence = StageEvidence(
        "T1",
        _bindings(),
        _lineage(),
        {"metrics.json": b"{}"},
        {"state.json": b"{}"},
        b"log",
        summary,
    )
    assert summary.items_calls == 1

    output = write_handoff(tmp_path, evidence)

    assert summary.items_calls == 1
    with ZipFile(output.path) as archive:
        assert archive.read("stage_summary.json") == b'{"score":1}'


def test_bound_file_is_read_safely_and_rejects_changed_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"first")
    bound = bind_file(checkpoint)
    assert isinstance(bound, BoundFile)
    checkpoint.write_bytes(b"other")
    evidence = StageEvidence(
        "T1",
        _bindings(),
        _lineage(),
        {"metrics.json": b"{}"},
        {"checkpoint.bin": bound},
        b"log",
        {},
    )

    with pytest.raises(PortfolioArtifactError, match="changed|differs"):
        write_handoff(tmp_path / "output", evidence)


def test_nested_review_and_resume_manifests_are_exact_and_verified(tmp_path: Path) -> None:
    path = write_handoff(tmp_path, _evidence()).path
    outer = _zip_members(path)
    for name, kind in (
        ("review.zip", "temporal_review_v1"),
        ("resume.zip", "temporal_resume_v1"),
    ):
        with ZipFile(io.BytesIO(outer[name])) as archive:
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            assert manifest_bytes == artifact_module.canonical_json(manifest)
            assert manifest["artifact_kind"] == kind
            assert manifest["submission_package"] is False
            assert set(archive.namelist()) == {*manifest["members"], "manifest.json"}


def test_corrupted_nested_zip_is_rejected_even_when_outer_hash_is_updated(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    forged = _rewrite_outer(
        source,
        tmp_path / "forged.zip",
        replace={"review.zip": b"not a zip"},
    )
    with pytest.raises(PortfolioArtifactError):
        verify_handoff(forged)


def test_modified_outer_member_is_rejected(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    forged = tmp_path / "forged.zip"
    members = _zip_members(source)
    members["run.log"] = b"tampered"
    forged.write_bytes(_zip_bytes(list(members.items())))

    with pytest.raises(PortfolioArtifactError, match="SHA-256|size"):
        verify_handoff(forged)


@pytest.mark.parametrize(
    "summary_bytes",
    [
        b"[]",
        b'{\n  "score": 1\n}',
        b"\xff",
    ],
)
def test_stage_summary_must_be_a_canonical_utf8_json_object(
    tmp_path: Path, summary_bytes: bytes
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    forged = _rewrite_outer(
        source,
        tmp_path / "forged.zip",
        replace={"stage_summary.json": summary_bytes},
    )

    with pytest.raises(PortfolioArtifactError, match="summary|canonical|UTF-8|object"):
        verify_handoff(forged)


def test_manifest_lineage_rejects_unsupported_state_schema_version(
    tmp_path: Path,
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path

    def unsupported_schema(manifest: dict[str, object]) -> None:
        manifest["lineage"]["state_schema_version"] = 2  # type: ignore[index]

    forged = _rewrite_outer(
        source,
        tmp_path / "schema-two.zip",
        manifest_transform=unsupported_schema,
    )

    with pytest.raises(PortfolioArtifactError, match="schema"):
        verify_handoff(forged)


def test_noncanonical_outer_and_nested_manifests_are_rejected(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    outer = _rewrite_outer(
        source,
        tmp_path / "outer.zip",
        canonical_manifest=False,
    )
    with pytest.raises(PortfolioArtifactError, match="canonical"):
        verify_handoff(outer)

    members = _zip_members(source)
    with ZipFile(io.BytesIO(members["review.zip"])) as archive:
        nested = {name: archive.read(name) for name in archive.namelist()}
    nested_manifest = json.loads(nested["manifest.json"])
    nested["manifest.json"] = json.dumps(nested_manifest, indent=2).encode()
    forged_nested = _zip_bytes(list(nested.items()))
    forged = _rewrite_outer(
        source,
        tmp_path / "nested.zip",
        replace={"review.zip": forged_nested},
    )
    with pytest.raises(PortfolioArtifactError, match="canonical"):
        verify_handoff(forged)


@pytest.mark.parametrize(
    ("unsafe_name", "match"),
    [
        ("../escape", "unsafe"),
        ("/absolute", "unsafe"),
        (r"folder\escape", "unsafe"),
        ("folder/./escape", "unsafe"),
        ("control\x01name", "unsafe"),
    ],
)
def test_outer_metadata_rejects_unsafe_paths_before_read(
    tmp_path: Path, unsafe_name: str, match: str
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    forged = _rewrite_outer(
        source,
        tmp_path / "forged.zip",
        extras=[(unsafe_name, b"escape")],
    )

    with pytest.raises(PortfolioArtifactError, match=match):
        verify_handoff(forged)


def test_outer_metadata_rejects_duplicate_symlink_directory_and_special_file(
    tmp_path: Path,
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    members = _zip_members(source)

    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate"):
        duplicate.write_bytes(
            _zip_bytes(list(members.items()) + [("run.log", b"duplicate")])
        )
    with pytest.raises(PortfolioArtifactError, match="duplicate"):
        verify_handoff(duplicate)

    cases: list[tuple[str, ZipInfo]] = []
    symlink = ZipInfo("review.zip")
    symlink.create_system = 3
    symlink.external_attr = 0o120777 << 16
    cases.append(("regular file", symlink))
    directory = ZipInfo("extra/")
    directory.create_system = 3
    directory.external_attr = 0o040755 << 16
    cases.append(("regular file", directory))
    special = ZipInfo("review.zip")
    special.create_system = 3
    special.external_attr = 0o060600 << 16
    cases.append(("regular file", special))

    for index, (match, info) in enumerate(cases):
        forged_members: list[tuple[str | ZipInfo, bytes]] = []
        for name, value in members.items():
            forged_members.append((info if name == info.filename else name, value))
        if info.filename == "extra/":
            forged_members.append((info, b""))
        target = tmp_path / f"metadata-{index}.zip"
        target.write_bytes(_zip_bytes(forged_members))
        with pytest.raises(PortfolioArtifactError, match=match):
            verify_handoff(target)


def test_outer_metadata_rejects_encryption_before_read(tmp_path: Path) -> None:
    source = write_handoff(tmp_path, _evidence()).path
    encrypted = tmp_path / "encrypted.zip"
    encrypted.write_bytes(_patch_encrypted_flags(source.read_bytes()))

    with pytest.raises(PortfolioArtifactError, match="encrypted"):
        verify_handoff(encrypted)


@pytest.mark.parametrize(
    ("offset", "fmt", "replacement", "match"),
    [
        (4, "<H", 1, "multi-disk"),
        (10, "<H", 0xFFFF, "ZIP64"),
    ],
)
def test_eocd_preflight_rejects_multidisk_and_zip64_before_zipfile(
    tmp_path: Path,
    offset: int,
    fmt: str,
    replacement: int,
    match: str,
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    forged = tmp_path / "forged.zip"
    forged.write_bytes(_patch_eocd(source.read_bytes(), offset, fmt, replacement))

    with pytest.raises(PortfolioArtifactError, match=match):
        verify_handoff(forged)


def test_eocd_preflight_bounds_entry_count_central_bytes_and_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    monkeypatch.setattr(artifact_module, "_MAX_ZIP_ENTRIES", 4, raising=False)
    with pytest.raises(PortfolioArtifactError, match="entry count"):
        verify_handoff(source)

    monkeypatch.setattr(artifact_module, "_MAX_ZIP_ENTRIES", 100)
    monkeypatch.setattr(
        artifact_module, "_MAX_CENTRAL_DIRECTORY_BYTES", 1, raising=False
    )
    with pytest.raises(PortfolioArtifactError, match="central directory"):
        verify_handoff(source)

    monkeypatch.setattr(artifact_module, "_MAX_CENTRAL_DIRECTORY_BYTES", 1024 * 1024)
    with ZipFile(source) as archive:
        central_offset = archive.start_dir
    forged = tmp_path / "outside.zip"
    forged.write_bytes(
        _patch_eocd(source.read_bytes(), 16, "<L", central_offset + 1)
    )
    with pytest.raises(PortfolioArtifactError, match="central directory"):
        verify_handoff(forged)


def test_nested_zip_is_subject_to_eocd_preflight(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    members = _zip_members(source)
    malformed_review = _patch_eocd(members["review.zip"], 4, "<H", 1)
    forged = _rewrite_outer(
        source,
        tmp_path / "nested-eocd.zip",
        replace={"review.zip": malformed_review},
    )

    with pytest.raises(PortfolioArtifactError, match="multi-disk"):
        verify_handoff(forged)


def test_metadata_size_compressed_size_ratio_and_zero_corner_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    monkeypatch.setattr(artifact_module, "_MAX_MEMBER_UNCOMPRESSED_BYTES", 10)
    with pytest.raises(PortfolioArtifactError, match="uncompressed"):
        verify_handoff(source)

    monkeypatch.setattr(artifact_module, "_MAX_MEMBER_UNCOMPRESSED_BYTES", 10**9)
    monkeypatch.setattr(artifact_module, "_MAX_TOTAL_UNCOMPRESSED_BYTES", 20)
    with pytest.raises(PortfolioArtifactError, match="total uncompressed"):
        verify_handoff(source)

    monkeypatch.setattr(artifact_module, "_MAX_TOTAL_UNCOMPRESSED_BYTES", 10**9)
    monkeypatch.setattr(artifact_module, "_MAX_MEMBER_COMPRESSED_BYTES", 2)
    with pytest.raises(PortfolioArtifactError, match="compressed"):
        verify_handoff(source)

    monkeypatch.setattr(artifact_module, "_MAX_MEMBER_COMPRESSED_BYTES", 10**9)
    monkeypatch.setattr(artifact_module, "_MAX_TOTAL_COMPRESSED_BYTES", 4)
    with pytest.raises(PortfolioArtifactError, match="total compressed"):
        verify_handoff(source)

    monkeypatch.setattr(artifact_module, "_MAX_TOTAL_COMPRESSED_BYTES", 10**9)
    monkeypatch.setattr(artifact_module, "_MAX_COMPRESSION_RATIO", 1.0)
    with pytest.raises(PortfolioArtifactError, match="compression ratio"):
        verify_handoff(source)

    monkeypatch.setattr(artifact_module, "_MAX_COMPRESSION_RATIO", 200.0)
    empty_log = StageEvidence(
        "T1",
        _bindings(),
        _lineage(),
        {"metrics.json": b"{}"},
        {"state.json": b"{}"},
        b"",
        {},
    )
    verify_handoff(write_handoff(tmp_path / "empty", empty_log).path)


def test_nested_zip_bomb_metadata_is_rejected_before_member_read(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    payload = b"0" * (256 * 1024)
    manifest = artifact_module.canonical_json(
        {
            "artifact_kind": "temporal_review_v1",
            "bindings": {
                "campaign_id": "temporal_portfolio_v1",
                "contract_sha256": "1" * 64,
                "input_manifest_sha256": "2" * 64,
            },
            "members": {
                "bomb.bin": {
                    "sha256": sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            },
            "schema_version": 1,
            "submission_package": False,
        }
    )
    bomb = _zip_bytes([("bomb.bin", payload), ("manifest.json", manifest)])
    forged = _rewrite_outer(
        source,
        tmp_path / "bomb.zip",
        replace={"review.zip": bomb},
    )

    with pytest.raises(PortfolioArtifactError, match="compression ratio"):
        verify_handoff(forged)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", "1"),
        ("artifact_kind", 1),
        ("submission_package", 0),
        ("bindings", []),
        ("lineage", []),
        ("members", []),
        ("extra", "unexpected"),
    ],
)
def test_manifest_rejects_malformed_types_and_unknown_keys(
    tmp_path: Path, field: str, value: object
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path

    def transform(manifest: dict[str, object]) -> None:
        manifest[field] = value

    forged = _rewrite_outer(source, tmp_path / "forged.zip", manifest_transform=transform)
    with pytest.raises(PortfolioArtifactError):
        verify_handoff(forged)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("bindings", "campaign_id", 1),
        ("bindings", "contract_sha256", "A" * 64),
        ("bindings", "input_manifest_sha256", None),
        ("lineage", "stage", []),
        ("lineage", "sequence", True),
        ("lineage", "parent_manifest_sha256", "A" * 64),
        ("lineage", "training_sha256", None),
        ("lineage", "state_schema_version", 1.0),
        ("lineage", "runtime_sha256", "x" * 64),
        ("member", "sha256", "A" * 64),
        ("member", "size_bytes", True),
    ],
)
def test_manifest_rejects_malformed_binding_lineage_and_member_fields(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path

    def transform(manifest: dict[str, object]) -> None:
        if section == "member":
            record = next(iter(manifest["members"].values()))  # type: ignore[union-attr]
            record[field] = value
        else:
            manifest[section][field] = value  # type: ignore[index]

    forged = _rewrite_outer(source, tmp_path / "forged.zip", manifest_transform=transform)
    with pytest.raises(PortfolioArtifactError):
        verify_handoff(forged)


def test_manifest_requires_exact_keys_at_every_level(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path

    def transform(manifest: dict[str, object]) -> None:
        manifest["bindings"]["extra"] = True  # type: ignore[index]

    forged = _rewrite_outer(source, tmp_path / "forged.zip", manifest_transform=transform)
    with pytest.raises(PortfolioArtifactError, match="keys"):
        verify_handoff(forged)


def test_manifest_member_names_must_exactly_match_archive(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path

    def transform(manifest: dict[str, object]) -> None:
        del manifest["members"]["run.log"]  # type: ignore[index]

    forged = _rewrite_outer(source, tmp_path / "forged.zip", manifest_transform=transform)
    with pytest.raises(PortfolioArtifactError, match="members"):
        verify_handoff(forged)


def test_malformed_zip_json_unicode_and_io_are_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed_zip = tmp_path / "malformed.zip"
    malformed_zip.write_bytes(b"bad")
    with pytest.raises(PortfolioArtifactError):
        verify_handoff(malformed_zip)

    source = write_handoff(tmp_path / "source", _evidence()).path
    members = _zip_members(source)
    members["handoff_manifest.json"] = b"{bad json"
    malformed_json = tmp_path / "json.zip"
    malformed_json.write_bytes(_zip_bytes(list(members.items())))
    with pytest.raises(PortfolioArtifactError):
        verify_handoff(malformed_json)

    members["handoff_manifest.json"] = b"\xff"
    malformed_unicode = tmp_path / "unicode.zip"
    malformed_unicode.write_bytes(_zip_bytes(list(members.items())))
    with pytest.raises(PortfolioArtifactError):
        verify_handoff(malformed_unicode)

    def fail_open(*args, **kwargs):
        raise OSError("fixture I/O failure")

    monkeypatch.setattr(artifact_module.os, "open", fail_open)
    with pytest.raises(PortfolioArtifactError):
        verify_handoff(source)


def test_discovery_selects_highest_sequence_of_one_linear_chain(tmp_path: Path) -> None:
    first = write_handoff(tmp_path / "one", _evidence()).path
    first_verified = verify_handoff(first)
    second_evidence = _evidence(
        lineage=_lineage(sequence=2, parent=first_verified.manifest_sha256),
        marker="second",
    )
    second = write_handoff(tmp_path / "two", second_evidence).path

    found = discover_handoffs([tmp_path])
    assert len(found) == 1
    assert found[0].sequence == 2
    assert found[0].manifest_sha256 == verify_handoff(second).manifest_sha256


def test_discovery_rejects_equal_sequence_different_manifests(tmp_path: Path) -> None:
    write_handoff(tmp_path / "a", _evidence(marker="a"))
    write_handoff(tmp_path / "b", _evidence(marker="b"))

    with pytest.raises(PortfolioArtifactError, match="equal sequence"):
        discover_handoffs([tmp_path])


def test_discovery_rejects_forks_and_disconnected_parents(tmp_path: Path) -> None:
    first = write_handoff(tmp_path / "one", _evidence()).path
    root_hash = verify_handoff(first).manifest_sha256
    write_handoff(
        tmp_path / "two",
        _evidence(lineage=_lineage(sequence=2, parent=root_hash), marker="two"),
    )
    write_handoff(
        tmp_path / "fork",
        _evidence(lineage=_lineage(sequence=3, parent=root_hash), marker="fork"),
    )
    with pytest.raises(PortfolioArtifactError, match="disconnected|fork"):
        discover_handoffs([tmp_path])


@pytest.mark.parametrize("conflict", ["bindings", "training", "runtime"])
def test_discovery_rejects_binding_and_lineage_identity_conflicts(
    tmp_path: Path, conflict: str
) -> None:
    first = write_handoff(tmp_path / "one", _evidence()).path
    root_hash = verify_handoff(first).manifest_sha256
    bindings = _bindings("5") if conflict == "bindings" else _bindings()
    lineage = _lineage(
        sequence=2,
        parent=root_hash,
        training="6" * 64 if conflict == "training" else "3" * 64,
        runtime="7" * 64 if conflict == "runtime" else "4" * 64,
    )
    write_handoff(tmp_path / "two", _evidence(bindings=bindings, lineage=lineage))

    with pytest.raises(PortfolioArtifactError, match="bindings|lineage"):
        discover_handoffs([tmp_path])


@pytest.mark.parametrize("opaque_name", ["opaque-upload.zip", "opaque-upload"])
def test_directory_discovery_identifies_handoff_by_internal_manifest(
    tmp_path: Path, opaque_name: str
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    expected = verify_handoff(source).manifest_sha256
    scan = tmp_path / "scan"
    scan.mkdir()
    renamed = scan / opaque_name
    source.replace(renamed)

    found = discover_handoffs([scan])

    assert found[0].path == renamed
    assert found[0].manifest_sha256 == expected


def test_directory_discovery_ignores_unrelated_files_and_zips(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    scan = tmp_path / "scan"
    scan.mkdir()
    renamed = scan / "actual-upload.bin"
    source.replace(renamed)
    (scan / "notes.txt").write_bytes(b"not a zip")
    (scan / "unrelated.zip").write_bytes(_zip_bytes([("data.txt", b"data")]))

    found = discover_handoffs([scan])

    assert found[0].path == renamed


def test_internal_manifest_candidate_that_fails_verification_poisons_discovery(
    tmp_path: Path,
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    scan = tmp_path / "scan"
    scan.mkdir()
    source.replace(scan / source.name)
    (scan / "malformed-opaque").write_bytes(
        _zip_bytes([("handoff_manifest.json", b"{malformed")])
    )

    with pytest.raises(PortfolioArtifactError):
        discover_handoffs([scan])


def test_discovery_ignores_structurally_malformed_unrelated_zip(
    tmp_path: Path,
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    scan = tmp_path / "scan"
    scan.mkdir()
    source.replace(scan / "valid-opaque")
    unrelated = _zip_bytes([("unrelated.txt", b"data")])
    (scan / "broken-unrelated").write_bytes(
        _patch_eocd(unrelated, 4, "<H", 1)
    )

    found = discover_handoffs([scan])

    assert found[0].path.name == "valid-opaque"


def test_discovery_fails_closed_for_structurally_malformed_manifest_zip(
    tmp_path: Path,
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    scan = tmp_path / "scan"
    scan.mkdir()
    source.replace(scan / "valid-opaque")
    manifest_zip = _zip_bytes([("handoff_manifest.json", b"{}")])
    (scan / "broken-manifest").write_bytes(
        _patch_eocd(manifest_zip, 4, "<H", 1)
    )

    with pytest.raises(PortfolioArtifactError, match="multi-disk"):
        discover_handoffs([scan])


def test_discovery_fails_closed_for_manifest_zip_missing_eocd(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    scan = tmp_path / "scan"
    scan.mkdir()
    source.replace(scan / "valid-opaque")
    manifest_zip = _zip_bytes([("handoff_manifest.json", b"{}")])
    (scan / "missing-eocd").write_bytes(manifest_zip[:-22])

    with pytest.raises(PortfolioArtifactError, match="manifest-bearing"):
        discover_handoffs([scan])


def test_discovery_bounds_total_archive_bytes_inspected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    scan = tmp_path / "scan"
    scan.mkdir()
    renamed = scan / "opaque"
    source.replace(renamed)
    monkeypatch.setattr(
        artifact_module,
        "_MAX_DISCOVERY_INSPECTED_BYTES",
        renamed.stat().st_size - 1,
    )

    with pytest.raises(PortfolioArtifactError, match="inspection byte limit"):
        discover_handoffs([scan])


def test_discovery_bounds_manifest_bearing_candidate_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "one").write_bytes(source.read_bytes())
    (scan / "two").write_bytes(source.read_bytes())
    monkeypatch.setattr(artifact_module, "_MAX_DISCOVERY_CANDIDATES", 1)

    with pytest.raises(PortfolioArtifactError, match="candidate limit"):
        discover_handoffs([scan])


def test_discovery_snapshots_only_root_limit_plus_one(tmp_path: Path) -> None:
    def too_many_roots():
        for _ in range(artifact_module._MAX_DISCOVERY_ROOTS + 1):
            yield tmp_path
        raise AssertionError("discovery exhausted an arbitrary roots iterable")

    with pytest.raises(PortfolioArtifactError, match="root limit"):
        discover_handoffs(too_many_roots())


def test_discovery_charges_direct_file_root_to_shared_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    monkeypatch.setattr(
        artifact_module,
        "_MAX_DISCOVERY_INSPECTED_BYTES",
        source.stat().st_size - 1,
    )

    with pytest.raises(PortfolioArtifactError, match="inspection byte limit"):
        discover_handoffs([source])


def test_discovery_charges_extracted_members_to_shared_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with ZipFile(source) as archive:
        archive.extractall(extracted)
    total = sum(path.stat().st_size for path in extracted.iterdir())
    monkeypatch.setattr(
        artifact_module, "_MAX_DISCOVERY_INSPECTED_BYTES", total - 1
    )

    with pytest.raises(PortfolioArtifactError, match="inspection byte limit"):
        discover_handoffs([extracted])


def test_discovery_shares_directory_inode_identities_across_roots(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(PortfolioArtifactError, match="directory alias"):
        discover_handoffs([empty, empty])


def test_discovery_preflight_rejects_file_swap_after_directory_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_handoff(tmp_path / "source", _evidence(marker="first")).path
    replacement = write_handoff(
        tmp_path / "replacement", _evidence(marker="replacement")
    ).path
    scan = tmp_path / "scan"
    scan.mkdir()
    opaque = scan / "opaque"
    source.replace(opaque)
    original_open = artifact_module._open_safe_file
    swapped = False

    def swap_then_open(path: Path, label: str):
        nonlocal swapped
        if path == opaque and label == "discovery archive" and not swapped:
            swapped = True
            replacement.replace(opaque)
        return original_open(path, label)

    monkeypatch.setattr(artifact_module, "_open_safe_file", swap_then_open)

    with pytest.raises(PortfolioArtifactError, match="changed"):
        discover_handoffs([scan])


def test_discovery_is_bounded_and_rejects_symlink_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = write_handoff(tmp_path / "target", _evidence()).path
    alias = tmp_path / "alias.zip"
    alias.symlink_to(target)
    with pytest.raises(PortfolioArtifactError, match="symlink"):
        discover_handoffs([alias])

    scan = tmp_path / "scan"
    scan.mkdir()
    for index in range(3):
        (scan / f"untrusted-{index}.txt").write_text("x")
    monkeypatch.setattr(artifact_module, "_MAX_DISCOVERY_ENTRIES", 2)
    with pytest.raises(PortfolioArtifactError, match="limit"):
        discover_handoffs([scan])


def test_extracted_directory_rejects_extra_and_symlink_members(tmp_path: Path) -> None:
    source = write_handoff(tmp_path / "source", _evidence()).path
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with ZipFile(source) as archive:
        archive.extractall(extracted)
    (extracted / "extra.txt").write_text("untrusted")
    with pytest.raises(PortfolioArtifactError, match="members"):
        discover_handoffs([extracted])
    (extracted / "extra.txt").unlink()
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim")
    (extracted / "run.log").unlink()
    (extracted / "run.log").symlink_to(victim)
    with pytest.raises(PortfolioArtifactError, match="symlink|regular"):
        discover_handoffs([extracted])


def test_atomic_publish_rejects_output_symlink_and_preserves_target(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    victim = tmp_path / "victim.zip"
    victim.write_bytes(b"keep")
    destination = output / "temporal_portfolio_stage_T1_handoff.zip"
    destination.symlink_to(victim)

    with pytest.raises(PortfolioArtifactError, match="symlink"):
        write_handoff(output, _evidence())
    assert victim.read_bytes() == b"keep"
    assert not list(output.glob(".*.tmp"))


def test_publication_is_idempotent_for_identical_handoff_bytes(tmp_path: Path) -> None:
    first = write_handoff(tmp_path, _evidence())
    before = first.path.stat()

    second = write_handoff(tmp_path, _evidence())
    after = second.path.stat()

    assert second.sha256 == first.sha256
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
    )


def test_publication_accepts_only_next_connected_same_identity_sequence(
    tmp_path: Path,
) -> None:
    first = write_handoff(tmp_path, _evidence()).path
    first_verified = verify_handoff(first)
    successor = _evidence(
        lineage=_lineage(
            sequence=2,
            parent=first_verified.manifest_sha256,
        ),
        marker="successor",
    )

    second = write_handoff(tmp_path, successor)

    assert verify_handoff(second.path).sequence == 2


def test_publication_holds_and_releases_interprocess_directory_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_flock = artifact_module.fcntl.flock
    operations: list[int] = []

    def recording_flock(descriptor: int, operation: int) -> None:
        operations.append(operation)
        original_flock(descriptor, operation)

    monkeypatch.setattr(artifact_module.fcntl, "flock", recording_flock)

    write_handoff(tmp_path, _evidence())

    assert operations == [artifact_module.fcntl.LOCK_EX, artifact_module.fcntl.LOCK_UN]


@pytest.mark.parametrize("conflict", ["equal", "rollback", "parent", "bindings"])
def test_publication_rejects_conflicting_rollback_and_disconnected_overwrite(
    tmp_path: Path, conflict: str
) -> None:
    first = write_handoff(tmp_path, _evidence()).path
    first_verified = verify_handoff(first)
    if conflict == "rollback":
        successor = _evidence(
            lineage=_lineage(
                sequence=2, parent=first_verified.manifest_sha256
            ),
            marker="successor",
        )
        write_handoff(tmp_path, successor)
        candidate = _evidence(marker="rollback")
    elif conflict == "equal":
        candidate = _evidence(marker="conflicting-equal")
    elif conflict == "parent":
        candidate = _evidence(
            lineage=_lineage(sequence=2, parent="9" * 64), marker="disconnected"
        )
    else:
        candidate = _evidence(
            bindings=_bindings("5"),
            lineage=_lineage(
                sequence=2, parent=first_verified.manifest_sha256
            ),
            marker="different-bindings",
        )
    before = first.read_bytes() if conflict != "rollback" else (tmp_path / first.name).read_bytes()

    with pytest.raises(PortfolioArtifactError, match="rollback|conflict|parent|bindings|sequence|identity"):
        write_handoff(tmp_path, candidate)

    assert (tmp_path / first.name).read_bytes() == before


def test_safe_file_open_does_not_follow_ancestor_swapped_after_old_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "safe"
    parent.mkdir()
    source = parent / "checkpoint.bin"
    source.write_bytes(b"trusted")
    moved = tmp_path / "moved-safe"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    (attacker / source.name).write_bytes(b"attacker")
    original_check = artifact_module._reject_symlink_components
    swapped = False

    def check_then_swap(path: Path, *, include_final: bool = True) -> None:
        nonlocal swapped
        original_check(path, include_final=include_final)
        if Path(path) == source and not swapped:
            swapped = True
            parent.rename(moved)
            parent.symlink_to(attacker, target_is_directory=True)

    monkeypatch.setattr(
        artifact_module, "_reject_symlink_components", check_then_swap
    )

    bound = bind_file(source)

    assert bound.sha256 == sha256(b"trusted").hexdigest()


def test_publication_rejects_output_ancestor_swap_before_atomic_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    moved = tmp_path / "moved-output"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    original_prepare = artifact_module._prepare_output_root
    swapped = False

    def prepare_then_swap(root: str | Path) -> Path:
        nonlocal swapped
        prepared = original_prepare(root)
        if not swapped:
            swapped = True
            prepared.rename(moved)
            prepared.symlink_to(attacker, target_is_directory=True)
        return prepared

    monkeypatch.setattr(artifact_module, "_prepare_output_root", prepare_then_swap)

    with pytest.raises(PortfolioArtifactError, match="symlink|directory"):
        write_handoff(output, _evidence())
    assert not list(attacker.iterdir())


def test_atomic_publish_leaves_no_partial_file_when_self_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = artifact_module._verify_handoff_at

    def fail_verification(directory: int, name: str, path: Path):
        if name.startswith(".temporal_portfolio"):
            raise PortfolioArtifactError("fixture verification failure")
        return original(directory, name, path)

    monkeypatch.setattr(artifact_module, "_verify_handoff_at", fail_verification)
    output = tmp_path / "output"
    with pytest.raises(PortfolioArtifactError, match="fixture"):
        write_handoff(output, _evidence())
    assert not list(output.glob("*.zip"))
    assert not list(output.glob(".*"))


def test_bound_file_and_handoff_descriptor_validate_types(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"x")
    with pytest.raises(PortfolioArtifactError):
        BoundFile(source, "A" * 64, 1)
    with pytest.raises(PortfolioArtifactError):
        BoundFile(source, "0" * 64, True)
    with pytest.raises(PortfolioArtifactError):
        HandoffPath(source, "A" * 64)


def test_no_submission_package_is_created(tmp_path: Path) -> None:
    output = write_handoff(tmp_path, _evidence())
    assert output.path.name == "temporal_portfolio_stage_T1_handoff.zip"
    assert sorted(path.name for path in tmp_path.iterdir()) == [output.path.name]
    assert "submit" not in output.path.name.lower()
