"""Deterministic review/resume/delivery/handoff artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo


class PrivilegedArtifactError(ValueError):
    pass


_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class CampaignArtifactState:
    status: str
    bindings: Mapping[str, str]
    review_sources: Mapping[str, Path]
    resume_sources: Mapping[str, Path]
    delivery_root: Path | None
    log_path: Path


@dataclass(frozen=True)
class CampaignBundles:
    review: Path
    resume: Path
    handoff: Path
    delivery: Path | None


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_TIMESTAMP); info.compress_type = ZIP_DEFLATED; info.external_attr = 0o100644 << 16
    return info


def _regular(path: Path) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PrivilegedArtifactError(f"artifact source differs: {source}")
    return source


def _members(sources: Mapping[str, Path]) -> dict[str, Path]:
    output = {}
    for name, path in sources.items():
        posix = PurePosixPath(name)
        if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
            raise PrivilegedArtifactError(f"unsafe artifact member: {name}")
        output[name] = _regular(path)
    return output


def _tree(root: Path) -> dict[str, Path]:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise PrivilegedArtifactError("delivery root differs")
    output = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink(): raise PrivilegedArtifactError("delivery contains a symlink")
        if path.is_file(): output[path.relative_to(source).as_posix()] = path
    return output


def _write(path: Path, kind: str, status: str, bindings: Mapping[str, str], sources: Mapping[str, Path]) -> Path:
    source_members = _members(sources)
    manifest = _canonical({
        "schema_version": 1, "artifact_kind": kind, "campaign_id": "tree_privileged_profile_v1",
        "status": status, "submission_package": False, "bindings": dict(bindings),
        "members": {name: {"size": value.stat().st_size, "sha256": file_sha256(value)} for name, value in sorted(source_members.items())},
    })
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent); os.close(descriptor)
    temporary = Path(name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for member, source in sorted(source_members.items()):
                with source.open("rb") as reader, archive.open(_info(member), "w") as writer:
                    shutil.copyfileobj(reader, writer, 1024 * 1024)
            archive.writestr(_info("manifest.json"), manifest)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def verify_bundle(path: Path, *, kind: str, expected_bindings: Mapping[str, str]) -> Mapping[str, object]:
    try:
        with ZipFile(_regular(path)) as archive:
            infos = archive.infolist(); names = [info.filename for info in infos]
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise PrivilegedArtifactError("artifact member set differs")
            for info in infos:
                posix = PurePosixPath(info.filename); mode = info.external_attr >> 16
                if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts) or info.is_dir() or stat.S_ISLNK(mode):
                    raise PrivilegedArtifactError("unsafe artifact member")
            manifest = json.loads(archive.read("manifest.json"))
            if (type(manifest) is not dict or manifest.get("artifact_kind") != kind
                    or manifest.get("campaign_id") != "tree_privileged_profile_v1"
                    or manifest.get("submission_package") is not False
                    or manifest.get("bindings") != dict(expected_bindings)):
                raise PrivilegedArtifactError("artifact bindings or identity differ")
            declared = manifest.get("members")
            if type(declared) is not dict or set(declared) != set(names) - {"manifest.json"}:
                raise PrivilegedArtifactError("artifact member declaration differs")
            for name, evidence in declared.items():
                payload = archive.read(name)
                if evidence != {"size": len(payload), "sha256": sha256(payload).hexdigest()}:
                    raise PrivilegedArtifactError(f"artifact member differs: {name}")
            return MappingProxyType(manifest)
    except PrivilegedArtifactError: raise
    except (BadZipFile, OSError, KeyError, json.JSONDecodeError) as error:
        raise PrivilegedArtifactError(f"artifact verification failed: {error}") from error


def verify_delivery(path: Path, expected_bindings: Mapping[str, str]) -> Mapping[str, object]:
    return verify_bundle(path, kind="tree_privileged_delivery_v1", expected_bindings=expected_bindings)


def write_campaign_bundles(state: CampaignArtifactState, output_dir: Path) -> CampaignBundles:
    if type(state) is not CampaignArtifactState:
        raise PrivilegedArtifactError("campaign state type differs")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    review = _write(output / "tree_privileged_review.zip", "tree_privileged_review_v1", state.status,
                    state.bindings, state.review_sources)
    resume_sources = dict(state.resume_sources); resume_sources["review.zip"] = review
    resume = _write(output / "tree_privileged_resume.zip", "tree_privileged_resume_v1", state.status,
                    state.bindings, resume_sources)
    delivery = None
    if state.status == "accepted":
        if state.delivery_root is None:
            raise PrivilegedArtifactError("accepted campaign has no full-fit delivery")
        delivery = _write(output / "tree_privileged_delivery.zip", "tree_privileged_delivery_v1", state.status,
                          state.bindings, _tree(state.delivery_root))
        verify_delivery(delivery, state.bindings)
    elif state.delivery_root is not None:
        raise PrivilegedArtifactError("non-accepted campaign cannot contain delivery")
    handoff_sources = {"review.zip": review, "resume.zip": resume, "campaign.log": state.log_path}
    if delivery is not None: handoff_sources["delivery.zip"] = delivery
    handoff = _write(output / "tree_privileged_handoff.zip", "tree_privileged_handoff_v1", state.status,
                     state.bindings, handoff_sources)
    return CampaignBundles(review, resume, handoff, delivery)

