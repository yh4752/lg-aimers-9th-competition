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


class E3ArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class E3Bindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str
    e2_submission_sha256: str


_KINDS = {
    "review": "failure_regime_e3_review_v1",
    "handoff": "failure_regime_e3_handoff_v1",
}
_TIME = (2026, 1, 1, 0, 0, 0)
_BLOCK = 1024 * 1024


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise E3ArtifactError("artifact evidence is not canonicalizable") from error


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def bindings_payload(bindings: E3Bindings) -> dict[str, str]:
    if type(bindings) is not E3Bindings:
        raise E3ArtifactError("artifact bindings differ")
    values = asdict(bindings)
    if any(
        type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in values.values()
    ):
        raise E3ArtifactError("artifact bindings differ")
    return values


def _safe_name(name: str) -> None:
    pure = PurePosixPath(name)
    if not name or name == "manifest.json" or pure.is_absolute() or ".." in pure.parts or "\\" in name:
        raise E3ArtifactError("unsafe artifact member")


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, _TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def create_bundle(kind: str, payloads: Mapping[str, Path], destination: Path, bindings: E3Bindings) -> Path:
    if kind not in _KINDS or not payloads:
        raise E3ArtifactError("artifact kind differs")
    members: dict[str, dict[str, object]] = {}
    sources: dict[str, Path] = {}
    for name, raw_path in sorted(payloads.items()):
        _safe_name(name)
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise E3ArtifactError("artifact source differs")
        sources[name] = path
        members[name] = {"sha256": file_sha256(path), "size": path.stat().st_size}
    manifest = canonical_json({
        "schema_version": 1,
        "artifact_kind": _KINDS[kind],
        "bindings": bindings_payload(bindings),
        "members": members,
        "submission_package": False,
    })
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w") as archive:
            for name, path in sources.items():
                with path.open("rb") as reader, archive.open(_info(name), "w") as writer:
                    shutil.copyfileobj(reader, writer, _BLOCK)
            archive.writestr(_info("manifest.json"), manifest)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    verify_bundle(output, kind, bindings)
    return output


def _verify_manifest(manifest: object, kind: str, bindings: E3Bindings, names: set[str]) -> dict[str, object]:
    if type(manifest) is not dict or manifest.get("artifact_kind") != _KINDS[kind]:
        raise E3ArtifactError("artifact identity differs")
    if manifest.get("bindings") != bindings_payload(bindings):
        raise E3ArtifactError("artifact bindings differ")
    members = manifest.get("members")
    if type(members) is not dict or set(members) != names - {"manifest.json"}:
        raise E3ArtifactError("artifact member set differs")
    if manifest.get("submission_package") is not False:
        raise E3ArtifactError("artifact submission metadata differs")
    return manifest


def verify_bundle(path: Path, kind: str, bindings: E3Bindings) -> dict[str, object]:
    if kind not in _KINDS:
        raise E3ArtifactError("artifact kind differs")
    source = Path(path)
    if source.is_symlink():
        raise E3ArtifactError("artifact is not a regular source")
    if source.is_dir():
        files: dict[str, Path] = {}
        for item in sorted(source.rglob("*")):
            if item.is_symlink() or (not item.is_file() and not item.is_dir()):
                raise E3ArtifactError("unsafe artifact member")
            if item.is_file():
                files[item.relative_to(source).as_posix()] = item
        if "manifest.json" not in files:
            raise E3ArtifactError("artifact manifest is absent")
        try:
            manifest = json.loads(files["manifest.json"].read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise E3ArtifactError("artifact manifest is unreadable") from error
        checked = _verify_manifest(manifest, kind, bindings, set(files))
        for name, declared in checked["members"].items():
            _safe_name(name)
            item = files[name]
            if declared != {"sha256": file_sha256(item), "size": item.stat().st_size}:
                raise E3ArtifactError(f"artifact member digest differs: {name}")
        return checked
    try:
        with ZipFile(source) as archive:
            infos: dict[str, ZipInfo] = {}
            for info in archive.infolist():
                pure = PurePosixPath(info.filename)
                if info.filename != "manifest.json":
                    _safe_name(info.filename)
                if info.filename in infos or info.is_dir() or info.flag_bits & 1 or pure.is_absolute() or ".." in pure.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise E3ArtifactError("unsafe artifact member")
                infos[info.filename] = info
            if "manifest.json" not in infos:
                raise E3ArtifactError("artifact manifest is absent")
            checked = _verify_manifest(json.loads(archive.read("manifest.json")), kind, bindings, set(infos))
            for name, declared in checked["members"].items():
                digest = sha256()
                size = 0
                with archive.open(name) as reader:
                    for block in iter(lambda: reader.read(_BLOCK), b""):
                        digest.update(block)
                        size += len(block)
                if declared != {"sha256": digest.hexdigest(), "size": size}:
                    raise E3ArtifactError(f"artifact member digest differs: {name}")
            return checked
    except E3ArtifactError:
        raise
    except (OSError, BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E3ArtifactError("artifact is not a valid ZIP") from error


def extract_bundle(path: Path, destination: Path, kind: str, bindings: E3Bindings) -> Path:
    source = Path(path)
    manifest = verify_bundle(source, kind, bindings)
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise E3ArtifactError("artifact extraction destination already exists")
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
