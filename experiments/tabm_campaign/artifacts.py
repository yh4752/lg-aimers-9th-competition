from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


class ArtifactError(ValueError):
    """Raised when a campaign evidence bundle cannot be trusted."""


_VERSIONS = ("A", "B", "C", "D", "P")
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
_MAX_ZIP_COMPRESSION_RATIO = 200.0
_MIN_RATIO_CHECK_BYTES = 64 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class _StagePFile:
    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class StageEvidence:
    version: str
    campaign_config_sha256: str
    prior_manifest_sha256: str | None
    review_members: Mapping[str, bytes | _StagePFile]
    resume_members: Mapping[str, bytes | _StagePFile]


@dataclass(frozen=True)
class BundlePaths:
    review: Path
    resume: Path | None
    review_sha256: str
    resume_sha256: str | None
    manifest_sha256: str


@dataclass(frozen=True)
class VerifiedResume:
    path: Path
    version: str
    campaign_config_sha256: str
    prior_manifest_sha256: str | None
    manifest_sha256: str
    member_sha256: Mapping[str, str]


@dataclass(frozen=True)
class VerifiedReview:
    path: Path
    version: str
    campaign_config_sha256: str
    prior_manifest_sha256: str | None
    manifest_sha256: str
    member_sha256: Mapping[str, str]


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def _open_regular_descriptor(path: Path) -> tuple[int, os.stat_result]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    directory = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:-1]:
            child = os.open(
                component,
                directory_flags,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | nofollow,
            dir_fd=directory,
        )
    except OSError as error:
        raise ArtifactError(f"bundle source is not a safe regular file: {path}") from error
    finally:
        os.close(directory)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactError(f"bundle source is not a regular file: {path}")
        return descriptor, metadata
    except Exception:
        os.close(descriptor)
        raise


def _file_digest(path: Path) -> str:
    digest = sha256()
    descriptor, _ = _open_regular_descriptor(path)
    with os.fdopen(descriptor, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _member_digest(value: bytes | _StagePFile) -> str:
    return _digest(value) if type(value) is bytes else value.expected_sha256


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or name.endswith("/")
        or name == "manifest.json"
    ):
        raise ArtifactError(f"unsafe or reserved ZIP member path: {name}")


def _validate_archive_entries(archive: ZipFile, *, label: str) -> list[str]:
    """Reject unsafe ZIP metadata before any member is decompressed."""

    names = archive.namelist()
    if len(names) != len(set(names)) or "manifest.json" not in names:
        raise ArtifactError(f"{label} bundle has duplicate members or no manifest")
    total_size = 0
    for info in archive.infolist():
        if info.filename != "manifest.json":
            _validate_member_name(info.filename)
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if (
            info.is_dir()
            or stat.S_ISLNK(mode)
            or file_type not in {0, stat.S_IFREG}
        ):
            raise ArtifactError(
                f"{label} bundle member is not a regular file: {info.filename}"
            )
        if info.file_size > _MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES:
            raise ArtifactError(
                f"{label} bundle member exceeds the uncompressed size limit: {info.filename}"
            )
        if info.filename == "manifest.json" and info.file_size > _MAX_MANIFEST_BYTES:
            raise ArtifactError(f"{label} bundle manifest exceeds the size limit")
        total_size += info.file_size
        if total_size > _MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
            raise ArtifactError(f"{label} bundle exceeds the total uncompressed size limit")
        if info.file_size >= _MIN_RATIO_CHECK_BYTES:
            if info.compress_size == 0:
                raise ArtifactError(
                    f"{label} bundle member has an invalid compression ratio: {info.filename}"
                )
            ratio = info.file_size / info.compress_size
            if ratio > _MAX_ZIP_COMPRESSION_RATIO:
                raise ArtifactError(
                    f"{label} bundle member exceeds the compression ratio limit: {info.filename}"
                )
    return names


def _validate_evidence(evidence: StageEvidence) -> None:
    if evidence.version not in _VERSIONS:
        raise ArtifactError(f"unknown campaign version: {evidence.version}")
    if not _valid_sha(evidence.campaign_config_sha256):
        raise ArtifactError("campaign config SHA-256 is invalid")
    if evidence.version in {"A", "P"}:
        if evidence.prior_manifest_sha256 is not None and not _valid_sha(evidence.prior_manifest_sha256):
            raise ArtifactError(f"Version {evidence.version} restart prior manifest SHA-256 is invalid")
    elif not _valid_sha(evidence.prior_manifest_sha256):
        raise ArtifactError(f"Version {evidence.version} requires the prior manifest SHA-256")
    if not evidence.review_members:
        raise ArtifactError("review bundle must not be empty")
    if evidence.version != "D" and not evidence.resume_members:
        raise ArtifactError("Versions A-C and P require resume evidence")
    for name, value in [*evidence.review_members.items(), *evidence.resume_members.items()]:
        _validate_member_name(name)
        if type(value) is bytes:
            continue
        if (
            evidence.version == "P"
            and type(value) is _StagePFile
        ):
            if not _valid_sha(value.expected_sha256):
                raise ArtifactError(f"Stage P bundle member SHA-256 is invalid: {name}")
            descriptor, _ = _open_regular_descriptor(value.path)
            os.close(descriptor)
            continue
        if evidence.version == "P":
            raise ArtifactError(f"Stage P bundle member must be bytes or a regular file: {name}")
        else:
            raise ArtifactError(f"bundle member must be bytes: {name}")


def _manifest(
    evidence: StageEvidence,
    kind: str,
    members: Mapping[str, bytes | _StagePFile],
) -> bytes:
    return _canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": kind,
            "review_only": True,
            "version": evidence.version,
            "campaign_config_sha256": evidence.campaign_config_sha256,
            "prior_manifest_sha256": evidence.prior_manifest_sha256,
            "members": {
                name: _member_digest(value) for name, value in sorted(members.items())
            },
        }
    )


def _zip_bytes(evidence: StageEvidence, kind: str, members: Mapping[str, bytes]) -> tuple[bytes, str]:
    manifest = _manifest(evidence, kind, members)
    all_members = {**members, "manifest.json": manifest}
    import io

    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(all_members):
            info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, all_members[name])
    return buffer.getvalue(), _digest(manifest)


def _atomic_zip_publish(
    path: Path,
    evidence: StageEvidence,
    kind: str,
    members: Mapping[str, bytes | _StagePFile],
) -> tuple[str, str]:
    """Publish Stage P without materializing file members or the ZIP in memory."""

    manifest = _manifest(evidence, kind, members)
    expected = json.loads(manifest)["members"]
    all_members: dict[str, bytes | _StagePFile] = {**members, "manifest.json": manifest}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    os.close(descriptor)
    try:
        with ZipFile(
            temporary_name, "w", compression=ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(all_members):
                value = all_members[name]
                info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                observed = sha256()
                if type(value) is bytes:
                    info.file_size = len(value)
                    with archive.open(info, "w") as destination:
                        destination.write(value)
                        observed.update(value)
                else:
                    descriptor, metadata = _open_regular_descriptor(value.path)
                    info.file_size = metadata.st_size
                    with os.fdopen(descriptor, "rb") as source:
                        with archive.open(info, "w") as destination:
                            while chunk := source.read(1024 * 1024):
                                destination.write(chunk)
                                observed.update(chunk)
                if name != "manifest.json" and observed.hexdigest() != expected[name]:
                    raise ArtifactError(f"bundle member changed during publication: {name}")
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return _file_digest(path), _digest(manifest)


def _atomic_publish(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validate_bundle_prefix(bundle_prefix: str) -> None:
    if (
        not isinstance(bundle_prefix, str)
        or not bundle_prefix
        or bundle_prefix in {".", ".."}
        or "/" in bundle_prefix
        or "\\" in bundle_prefix
        or "\0" in bundle_prefix
    ):
        raise ArtifactError("bundle prefix must be one safe filename component")


def write_stage_bundles(
    output_dir: str | Path,
    evidence: StageEvidence,
    *,
    bundle_prefix: str = "tabm_search_stage",
) -> BundlePaths:
    _validate_evidence(evidence)
    _validate_bundle_prefix(bundle_prefix)
    root = Path(output_dir)
    review = root / f"{bundle_prefix}_{evidence.version}_review_bundle.zip"
    if evidence.version == "P":
        review_sha, review_manifest_sha = _atomic_zip_publish(
            review, evidence, "review", evidence.review_members
        )
    else:
        review_bytes, review_manifest_sha = _zip_bytes(
            evidence, "review", evidence.review_members  # type: ignore[arg-type]
        )
        _atomic_publish(review, review_bytes)
        review_sha = _digest(review_bytes)

    resume: Path | None = None
    resume_sha: str | None = None
    manifest_sha = review_manifest_sha
    if evidence.version != "D":
        resume = root / f"{bundle_prefix}_{evidence.version}_resume_bundle.zip"
        if evidence.version == "P":
            resume_sha, manifest_sha = _atomic_zip_publish(
                resume, evidence, "resume", evidence.resume_members
            )
        else:
            resume_bytes, manifest_sha = _zip_bytes(
                evidence, "resume", evidence.resume_members  # type: ignore[arg-type]
            )
            _atomic_publish(resume, resume_bytes)
            resume_sha = _digest(resume_bytes)
        verify_resume_bundle(resume)
    return BundlePaths(review, resume, review_sha, resume_sha, manifest_sha)


def verify_review_bundle(path: str | Path) -> VerifiedReview:
    """Verify every declared member of a review-only evidence bundle."""

    bundle = Path(path)
    try:
        with ZipFile(bundle, "r") as archive:
            names = _validate_archive_entries(archive, label="review")
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            if manifest.get("schema_version") != 1:
                raise ArtifactError("review manifest schema version is invalid")
            if manifest.get("artifact_kind") != "review" or manifest.get("review_only") is not True:
                raise ArtifactError("bundle is not review-only review evidence")
            expected = manifest.get("members")
            if not isinstance(expected, dict) or set(expected) != set(names) - {"manifest.json"}:
                raise ArtifactError("review member manifest differs")
            for name, expected_hash in expected.items():
                if not _valid_sha(expected_hash):
                    raise ArtifactError(f"review member SHA-256 is invalid: {name}")
                digest = sha256()
                with archive.open(name, "r") as member:
                    while chunk := member.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    raise ArtifactError(f"review member SHA-256 differs: {name}")
            version = manifest.get("version")
            config_sha = manifest.get("campaign_config_sha256")
            prior_sha = manifest.get("prior_manifest_sha256")
            if type(version) is not str or version not in _VERSIONS:
                raise ArtifactError("review version must be A, B, C, D, or P")
            if not _valid_sha(config_sha) or (
                version not in {"A", "P"} and not _valid_sha(prior_sha)
            ) or (prior_sha is not None and not _valid_sha(prior_sha)):
                raise ArtifactError("review manifest hash binding is invalid")
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"cannot verify review bundle: {exc}") from exc
    return VerifiedReview(
        bundle,
        version,
        config_sha,
        prior_sha,
        _digest(manifest_bytes),
        dict(expected),
    )


def verify_resume_bundle(path: str | Path) -> VerifiedResume:
    bundle = Path(path)
    try:
        with ZipFile(bundle, "r") as archive:
            names = _validate_archive_entries(archive, label="resume")
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            if manifest.get("schema_version") != 1:
                raise ArtifactError("resume manifest schema version is invalid")
            if manifest.get("artifact_kind") != "resume" or manifest.get("review_only") is not True:
                raise ArtifactError("bundle is not review-only resume evidence")
            expected = manifest.get("members")
            if not isinstance(expected, dict) or set(expected) != set(names) - {"manifest.json"}:
                raise ArtifactError("resume member manifest differs")
            for name, expected_hash in expected.items():
                if not _valid_sha(expected_hash):
                    raise ArtifactError(f"resume member SHA-256 is invalid: {name}")
                digest = sha256()
                with archive.open(name, "r") as member:
                    while chunk := member.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    raise ArtifactError(f"resume member SHA-256 differs: {name}")
            version = manifest.get("version")
            if type(version) is not str or version not in {"A", "B", "C", "P"}:
                raise ArtifactError("resume version must be A, B, C, or P")
            config_sha = manifest.get("campaign_config_sha256")
            prior_sha = manifest.get("prior_manifest_sha256")
            if not _valid_sha(config_sha) or (
                version not in {"A", "P"} and not _valid_sha(prior_sha)
            ) or (prior_sha is not None and not _valid_sha(prior_sha)):
                raise ArtifactError("resume manifest hash binding is invalid")
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"cannot verify resume bundle: {exc}") from exc
    return VerifiedResume(
        bundle,
        version,
        config_sha,
        prior_sha,
        _digest(manifest_bytes),
        dict(expected),
    )
