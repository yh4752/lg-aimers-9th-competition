"""Normalize intact and Kaggle-extracted campaign resume inputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .artifacts import ArtifactError, VerifiedResume, verify_resume_bundle


_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class NormalizedResume:
    path: Path | None
    source: str
    original_path: Path | None
    version: str | None
    manifest_sha256: str | None


@dataclass(frozen=True)
class _Candidate:
    path: Path
    source: str
    normalized_path: Path
    verified: VerifiedResume


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        name
        and "\\" not in name
        and not path.is_absolute()
        and ".." not in path.parts
        and not name.endswith("/")
        and name != "manifest.json"
    )


def _read_resume_manifest(path: Path) -> tuple[bytes, dict[str, object]] | None:
    try:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("artifact_kind") != "resume":
        return None
    return manifest_bytes, manifest


def _rebuild_extracted_resume(
    source: Path,
    manifest_bytes: bytes,
    manifest: dict[str, object],
    destination: Path,
) -> VerifiedResume:
    if manifest.get("review_only") is not True:
        raise ArtifactError(f"extracted resume is not review-only: {source}")
    expected = manifest.get("members")
    if not isinstance(expected, dict) or not expected:
        raise ArtifactError(f"extracted resume member manifest is invalid: {source}")
    expected_names = {str(name) for name in expected}
    if any(not _safe_member_name(name) for name in expected_names):
        raise ArtifactError(f"extracted resume has an unsafe member path: {source}")

    actual_names: set[str] = set()
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(f"extracted resume contains a symlink: {path}")
        if path.is_file() and path.name != "manifest.json":
            actual_names.add(path.relative_to(source).as_posix())
    if actual_names != expected_names:
        raise ArtifactError(f"extracted resume member manifest differs: {source}")

    members: dict[str, bytes] = {}
    for name in sorted(expected_names):
        value = (source / PurePosixPath(name)).read_bytes()
        if sha256(value).hexdigest() != expected[name]:
            raise ArtifactError(f"resume member SHA-256 differs: {name}")
        members[name] = value

    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in [*members.items(), ("manifest.json", manifest_bytes)]:
            info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return verify_resume_bundle(destination)


def normalize_resume_input(input_root: Path, working_root: Path) -> NormalizedResume:
    """Return one verified logical resume, rebuilding Kaggle-extracted input."""

    candidates: list[_Candidate] = []
    for path in sorted(input_root.rglob("tabm_search_stage_*_resume_bundle.zip")):
        if not path.is_file() or path.is_symlink():
            continue
        verified = verify_resume_bundle(path)
        candidates.append(_Candidate(path, "zip", path, verified))

    rebuilt_root = working_root / "rebuilt"
    for index, manifest_path in enumerate(sorted(input_root.rglob("manifest.json"))):
        found = _read_resume_manifest(manifest_path)
        if found is None:
            continue
        manifest_bytes, manifest = found
        source = manifest_path.parent
        destination = rebuilt_root / f"resume_{index}.zip"
        verified = _rebuild_extracted_resume(
            source, manifest_bytes, manifest, destination
        )
        candidates.append(_Candidate(source, "extracted", destination, verified))

    if not candidates:
        return NormalizedResume(None, "none", None, None, None)
    by_manifest: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_manifest.setdefault(candidate.verified.manifest_sha256, []).append(candidate)
    if len(by_manifest) != 1:
        raise ArtifactError(
            "multiple different resume inputs found: "
            + ", ".join(str(candidate.path) for candidate in candidates)
        )
    logical = next(iter(by_manifest.values()))
    selected = next(
        (candidate for candidate in logical if candidate.source == "zip"), logical[0]
    )
    return NormalizedResume(
        selected.normalized_path,
        selected.source,
        selected.path,
        selected.verified.version,
        selected.verified.manifest_sha256,
    )
