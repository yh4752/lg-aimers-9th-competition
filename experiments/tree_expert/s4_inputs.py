from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Iterable
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .e2_artifacts import E2ArtifactError, file_sha256, verify_e2_handoff
from .s4_contracts import load_s4_contract


class S4InputError(ValueError):
    pass


INPUT_KIND = "tree_s4_input_v1"
_TIME = (2026, 1, 1, 0, 0, 0)
_MEMBERS = {"e2/handoff.zip", "manifest.json"}
_MAX_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class VerifiedS4Input:
    artifact_kind: str
    root: Path
    manifest_sha256: str
    e2_handoff_sha256: str
    e2_handoff: Path


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, _TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _payloads(source: Path) -> dict[str, bytes]:
    path = Path(source)
    if path.is_symlink():
        raise S4InputError("unsafe S4 input source")
    if path.is_dir():
        output: dict[str, bytes] = {}
        for item in sorted(path.rglob("*")):
            if item.is_symlink():
                raise S4InputError("unsafe S4 input member")
            if item.is_file():
                output[item.relative_to(path).as_posix()] = item.read_bytes()
        return output
    try:
        with ZipFile(path) as archive:
            output = {}
            total = 0
            for info in archive.infolist():
                name = info.filename
                pure = PurePosixPath(name)
                mode = info.external_attr >> 16
                if (
                    name in output or info.is_dir() or info.flag_bits & 1 or pure.is_absolute()
                    or ".." in pure.parts or "\\" in name or stat.S_ISLNK(mode)
                ):
                    raise S4InputError("unsafe or duplicate S4 input member")
                total += info.file_size
                if total > _MAX_BYTES:
                    raise S4InputError("S4 input expanded size exceeds limit")
                output[name] = archive.read(info)
            return output
    except (OSError, BadZipFile) as error:
        raise S4InputError("S4 input is not a valid archive") from error


def _manifest(payload: bytes, e2_sha256: str) -> bytes:
    return _canonical({
        "schema_version": 1,
        "artifact_kind": INPUT_KIND,
        "e2_handoff_sha256": e2_sha256,
        "members": {"e2/handoff.zip": {"size": len(payload), "sha256": sha256(payload).hexdigest()}},
    })


def prepare_s4_input(
    e2_handoff: Path,
    output: Path,
    *,
    expected_e2_sha256: str | None = None,
) -> Path:
    source = Path(e2_handoff)
    expected = expected_e2_sha256 or load_s4_contract().inputs["e2_handoff_sha256"]
    try:
        verified = verify_e2_handoff(source)
    except E2ArtifactError as error:
        raise S4InputError(f"E2 handoff is invalid: {error}") from error
    actual = file_sha256(source)
    if not verified.delivery or actual != expected:
        raise S4InputError("E2 handoff identity differs")
    payload = source.read_bytes()
    manifest = _manifest(payload, actual)
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(_info("e2/handoff.zip"), payload)
            archive.writestr(_info("manifest.json"), manifest)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def verify_and_extract_s4_input(
    source: Path,
    destination: Path,
    *,
    expected_e2_sha256: str | None = None,
) -> VerifiedS4Input:
    payloads = _payloads(Path(source))
    if set(payloads) != _MEMBERS:
        raise S4InputError("S4 input member set differs")
    try:
        manifest = json.loads(payloads["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S4InputError("S4 input manifest is invalid") from error
    handoff = payloads["e2/handoff.zip"]
    expected_manifest = _manifest(handoff, str(manifest.get("e2_handoff_sha256")))
    if type(manifest) is not dict or payloads["manifest.json"] != expected_manifest:
        raise S4InputError("S4 input manifest differs")
    expected = expected_e2_sha256 or load_s4_contract().inputs["e2_handoff_sha256"]
    actual = sha256(handoff).hexdigest()
    if actual != expected or manifest.get("e2_handoff_sha256") != actual:
        raise S4InputError("E2 handoff identity differs")
    root = Path(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    handoff_path = root / "e2_handoff.zip"
    handoff_path.write_bytes(handoff)
    try:
        verified = verify_e2_handoff(handoff_path)
    except E2ArtifactError as error:
        raise S4InputError(f"embedded E2 handoff is invalid: {error}") from error
    if not verified.delivery:
        raise S4InputError("embedded E2 delivery is absent")
    return VerifiedS4Input(
        artifact_kind=INPUT_KIND,
        root=root,
        manifest_sha256=sha256(payloads["manifest.json"]).hexdigest(),
        e2_handoff_sha256=actual,
        e2_handoff=handoff_path,
    )


def discover_s4_input_candidates(paths: Iterable[Path]) -> tuple[Path, ...]:
    identities: dict[str, Path] = {}
    for path in paths:
        try:
            payloads = _payloads(Path(path))
            manifest = json.loads(payloads.get("manifest.json", b"{}"))
        except (S4InputError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if type(manifest) is dict and manifest.get("artifact_kind") == INPUT_KIND:
            identity = sha256(payloads["manifest.json"]).hexdigest()
            identities.setdefault(identity, Path(path))
    if len(identities) > 1:
        raise S4InputError("distinct S4 input identities")
    return tuple(identities.values())
