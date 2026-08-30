from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .inputs import canonical_json, file_sha256


class DirectExpertArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class DirectExpertBindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str
    e2_submission_sha256: str


_KINDS = {
    "stage_a": "direct_expert_stage_a_handoff_v1",
    "resume": "direct_expert_resume_v1",
    "review": "direct_expert_review_v1",
    "handoff": "direct_expert_handoff_v1",
    "delivery": "direct_expert_delivery_v1",
}
_TIME = (2026, 1, 1, 0, 0, 0)
_BLOCK = 1024 * 1024


def _bindings(bindings: DirectExpertBindings) -> dict[str, str]:
    if type(bindings) is not DirectExpertBindings:
        raise DirectExpertArtifactError("artifact bindings differ")
    values = asdict(bindings)
    if any(type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in values.values()):
        raise DirectExpertArtifactError("artifact bindings differ")
    return values


def _safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or name == "manifest.json":
        raise DirectExpertArtifactError("unsafe artifact member")


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, _TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def create_bundle(
    kind: str,
    payloads: Mapping[str, Path],
    destination: Path,
    bindings: DirectExpertBindings,
) -> Path:
    if kind not in _KINDS or not payloads:
        raise DirectExpertArtifactError("artifact kind differs")
    if kind == "delivery" and "accepted/token.json" not in payloads:
        raise DirectExpertArtifactError("accepted token is required")
    evidence = {}
    for name, path in sorted(payloads.items()):
        _safe_name(name)
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise DirectExpertArtifactError("artifact source differs")
        evidence[name] = {"sha256": file_sha256(source), "size": source.stat().st_size}
    manifest = canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": _KINDS[kind],
            "bindings": _bindings(bindings),
            "members": evidence,
            "delivery": kind == "delivery",
            "submission_package": False,
        }
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w") as archive:
            for name, path in sorted(payloads.items()):
                with Path(path).open("rb") as source, archive.open(_info(name), "w") as target:
                    shutil.copyfileobj(source, target, _BLOCK)
            archive.writestr(_info("manifest.json"), manifest)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    verify_bundle(output, kind, bindings)
    return output


def verify_bundle(path: Path, kind: str, bindings: DirectExpertBindings) -> dict[str, object]:
    if kind not in _KINDS:
        raise DirectExpertArtifactError("artifact kind differs")
    source = Path(path)
    if source.is_symlink():
        raise DirectExpertArtifactError("artifact is not a regular source")
    if source.is_dir():
        files: dict[str, Path] = {}
        for member in sorted(source.rglob("*")):
            if member.is_symlink():
                raise DirectExpertArtifactError("unsafe artifact member")
            if member.is_file():
                files[member.relative_to(source).as_posix()] = member
            elif not member.is_dir():
                raise DirectExpertArtifactError("unsafe artifact member")
        if "manifest.json" not in files:
            raise DirectExpertArtifactError("artifact manifest is absent")
        try:
            manifest = json.loads(files["manifest.json"].read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DirectExpertArtifactError("artifact manifest is unreadable") from error
        if manifest.get("artifact_kind") != _KINDS[kind]:
            raise DirectExpertArtifactError("artifact identity differs")
        if manifest.get("bindings") != _bindings(bindings):
            raise DirectExpertArtifactError("artifact bindings differ")
        members = manifest.get("members")
        if type(members) is not dict or set(members) != set(files) - {"manifest.json"}:
            raise DirectExpertArtifactError("artifact member set differs")
        for name, declared in members.items():
            _safe_name(name)
            member = files[name]
            if declared != {"sha256": file_sha256(member), "size": member.stat().st_size}:
                raise DirectExpertArtifactError(f"artifact member digest differs: {name}")
        if kind == "delivery" and (manifest.get("delivery") is not True or manifest.get("submission_package") is not False):
            raise DirectExpertArtifactError("delivery metadata differs")
        return manifest
    try:
        with ZipFile(source) as archive:
            infos = {}
            for info in archive.infolist():
                pure = PurePosixPath(info.filename)
                if info.filename != "manifest.json":
                    _safe_name(info.filename)
                if info.filename in infos or info.is_dir() or info.flag_bits & 1 or pure.is_absolute() or ".." in pure.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise DirectExpertArtifactError("unsafe artifact member")
                infos[info.filename] = info
            if "manifest.json" not in infos:
                raise DirectExpertArtifactError("artifact manifest is absent")
            manifest = json.loads(archive.read(infos["manifest.json"]))
            if manifest.get("artifact_kind") != _KINDS[kind]:
                raise DirectExpertArtifactError("artifact identity differs")
            if manifest.get("bindings") != _bindings(bindings):
                raise DirectExpertArtifactError("artifact bindings differ")
            members = manifest.get("members")
            if type(members) is not dict or set(members) != set(infos) - {"manifest.json"}:
                raise DirectExpertArtifactError("artifact member set differs")
            for name, declared in members.items():
                digest = sha256()
                size = 0
                with archive.open(infos[name]) as source:
                    for block in iter(lambda: source.read(_BLOCK), b""):
                        digest.update(block)
                        size += len(block)
                if declared != {"sha256": digest.hexdigest(), "size": size}:
                    raise DirectExpertArtifactError(f"artifact member digest differs: {name}")
            if kind == "delivery" and (manifest.get("delivery") is not True or manifest.get("submission_package") is not False):
                raise DirectExpertArtifactError("delivery metadata differs")
            return manifest
    except (OSError, BadZipFile, json.JSONDecodeError) as error:
        if isinstance(error, DirectExpertArtifactError):
            raise
        raise DirectExpertArtifactError("artifact is not a valid ZIP") from error


def extract_bundle(
    path: Path,
    destination: Path,
    kind: str,
    bindings: DirectExpertBindings,
) -> Path:
    source = Path(path)
    target = Path(destination)
    manifest = verify_bundle(source, kind, bindings)
    if target.exists() or target.is_symlink():
        raise DirectExpertArtifactError("artifact extraction destination already exists")
    target.mkdir(parents=True)
    names = (*sorted(manifest["members"]), "manifest.json")
    if source.is_dir():
        for name in names:
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, output)
    else:
        with ZipFile(source) as archive:
            for name in names:
                output = target / name
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as reader, output.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, _BLOCK)
    verify_bundle(target, kind, bindings)
    return target
