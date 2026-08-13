from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


class ArtifactError(ValueError):
    """Raised when a campaign evidence bundle cannot be trusted."""


_VERSIONS = ("A", "B", "C", "D")
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class StageEvidence:
    version: str
    campaign_config_sha256: str
    prior_manifest_sha256: str | None
    review_members: Mapping[str, bytes]
    resume_members: Mapping[str, bytes]


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


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _valid_sha(value: str | None) -> bool:
    return value is not None and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


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


def _validate_evidence(evidence: StageEvidence) -> None:
    if evidence.version not in _VERSIONS:
        raise ArtifactError(f"unknown campaign version: {evidence.version}")
    if not _valid_sha(evidence.campaign_config_sha256):
        raise ArtifactError("campaign config SHA-256 is invalid")
    if evidence.version == "A":
        if evidence.prior_manifest_sha256 is not None:
            raise ArtifactError("Version A must not have a prior manifest")
    elif not _valid_sha(evidence.prior_manifest_sha256):
        raise ArtifactError(f"Version {evidence.version} requires the prior manifest SHA-256")
    if not evidence.review_members:
        raise ArtifactError("review bundle must not be empty")
    if evidence.version != "D" and not evidence.resume_members:
        raise ArtifactError("Versions A-C require resume evidence")
    for name, value in [*evidence.review_members.items(), *evidence.resume_members.items()]:
        _validate_member_name(name)
        if not isinstance(value, bytes):
            raise ArtifactError(f"bundle member must be bytes: {name}")


def _manifest(evidence: StageEvidence, kind: str, members: Mapping[str, bytes]) -> bytes:
    return _canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": kind,
            "review_only": True,
            "version": evidence.version,
            "campaign_config_sha256": evidence.campaign_config_sha256,
            "prior_manifest_sha256": evidence.prior_manifest_sha256,
            "members": {name: _digest(value) for name, value in sorted(members.items())},
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


def write_stage_bundles(output_dir: str | Path, evidence: StageEvidence) -> BundlePaths:
    _validate_evidence(evidence)
    root = Path(output_dir)
    review_bytes, review_manifest_sha = _zip_bytes(evidence, "review", evidence.review_members)
    review = root / f"tabm_search_stage_{evidence.version}_review_bundle.zip"
    _atomic_publish(review, review_bytes)

    resume: Path | None = None
    resume_sha: str | None = None
    manifest_sha = review_manifest_sha
    if evidence.version != "D":
        resume_bytes, manifest_sha = _zip_bytes(evidence, "resume", evidence.resume_members)
        resume = root / f"tabm_search_stage_{evidence.version}_resume_bundle.zip"
        _atomic_publish(resume, resume_bytes)
        resume_sha = _digest(resume_bytes)
        verify_resume_bundle(resume)
    return BundlePaths(review, resume, _digest(review_bytes), resume_sha, manifest_sha)


def verify_resume_bundle(path: str | Path) -> VerifiedResume:
    bundle = Path(path)
    try:
        with ZipFile(bundle, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise ArtifactError("resume bundle has duplicate members or no manifest")
            for name in names:
                if name != "manifest.json":
                    _validate_member_name(name)
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            if manifest.get("artifact_kind") != "resume" or manifest.get("review_only") is not True:
                raise ArtifactError("bundle is not review-only resume evidence")
            expected = manifest.get("members")
            if not isinstance(expected, dict) or set(expected) != set(names) - {"manifest.json"}:
                raise ArtifactError("resume member manifest differs")
            for name, expected_hash in expected.items():
                if _digest(archive.read(name)) != expected_hash:
                    raise ArtifactError(f"resume member SHA-256 differs: {name}")
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"cannot verify resume bundle: {exc}") from exc
    version = str(manifest.get("version"))
    if version not in {"A", "B", "C"}:
        raise ArtifactError("resume version must be A, B, or C")
    config_sha = str(manifest.get("campaign_config_sha256"))
    prior_sha = manifest.get("prior_manifest_sha256")
    if not _valid_sha(config_sha) or (version != "A" and not _valid_sha(prior_sha)):
        raise ArtifactError("resume manifest hash binding is invalid")
    return VerifiedResume(
        bundle,
        version,
        config_sha,
        None if prior_sha is None else str(prior_sha),
        _digest(manifest_bytes),
        {str(name): str(value) for name, value in expected.items()},
    )
