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

from .s4_contracts import load_s4_contract


class S4InputError(ValueError):
    pass


INPUT_KIND = "tree_s4_input_v1"
_TIME = (2026, 1, 1, 0, 0, 0)
_MEMBERS = {"e2/handoff.zip", "manifest.json"}
_MAX_BYTES = 8 * 1024 * 1024 * 1024
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_RATIO = 250.0


@dataclass(frozen=True)
class VerifiedS4Input:
    artifact_kind: str
    root: Path
    manifest_sha256: str
    e2_handoff_sha256: str
    e2_handoff: Path


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise S4InputError("artifact source is not a regular file")
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_e2_handoff(path: Path) -> bool:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise S4InputError("E2 handoff source is not a regular file")
    try:
        with ZipFile(source) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise S4InputError("E2 handoff has duplicate members")
            total = 0
            for info in infos:
                pure = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                ratio = info.file_size / max(1, info.compress_size)
                if (
                    info.flag_bits & 1 or info.is_dir() or pure.is_absolute()
                    or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts)
                    or "\\" in info.filename or stat.S_ISLNK(mode)
                    or info.file_size > _MAX_MEMBER_BYTES
                    or (info.file_size >= 64 * 1024 and ratio > _MAX_RATIO)
                ):
                    raise S4InputError(f"unsafe E2 handoff member: {info.filename}")
                total += info.file_size
            if total > _MAX_BYTES or "handoff_manifest.json" not in names:
                raise S4InputError("E2 handoff members differ")
            manifest_bytes = archive.read("handoff_manifest.json")
            manifest = json.loads(manifest_bytes)
            expected_keys = {
                "schema_version", "artifact_kind", "campaign_id", "review_only",
                "submission_package", "status", "delivery", "members",
            }
            if (
                type(manifest) is not dict or set(manifest) != expected_keys
                or manifest["schema_version"] != 1
                or manifest["artifact_kind"] != "tree_expert_e2_handoff_v1"
                or manifest["campaign_id"] != "tree_expert_e2_v1"
                or manifest["submission_package"] is not False
                or type(manifest["delivery"]) is not bool
                or manifest["review_only"] is not (not manifest["delivery"])
            ):
                raise S4InputError("E2 handoff identity differs")
            expected_names = {
                "handoff_manifest.json", "tree_expert_e2_review.zip",
                "tree_expert_e2_resume.zip", "tree_expert_e2.log",
            }
            if manifest["delivery"]:
                expected_names.add("tree_expert_e2_model_delivery.zip")
            if set(names) != expected_names:
                raise S4InputError("E2 handoff members differ")
            declared = manifest.get("members")
            if type(declared) is not dict or set(declared) != expected_names - {"handoff_manifest.json"}:
                raise S4InputError("E2 handoff manifest members differ")
            for name, evidence in declared.items():
                if type(evidence) is not dict or set(evidence) != {"size", "sha256"}:
                    raise S4InputError(f"E2 handoff member evidence differs: {name}")
                digest = sha256()
                with archive.open(name) as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                if evidence["size"] != archive.getinfo(name).file_size or evidence["sha256"] != digest.hexdigest():
                    raise S4InputError(f"E2 handoff member differs: {name}")
            return bool(manifest["delivery"])
    except S4InputError:
        raise
    except (OSError, BadZipFile, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise S4InputError(f"E2 handoff is invalid: {error}") from error


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


def _expanded_zip_payload(source: Path, label: str) -> bytes:
    root = Path(source)
    if root.is_symlink() or not root.is_dir():
        raise S4InputError(f"unsafe expanded {label}")
    for item in root.rglob("*"):
        if item.is_symlink():
            raise S4InputError(f"unsafe expanded {label} member")
    manifest_names = [
        name for name in ("manifest.json", "handoff_manifest.json")
        if (root / name).is_file()
    ]
    if len(manifest_names) != 1:
        raise S4InputError(f"expanded {label} manifest differs")
    manifest_name = manifest_names[0]
    manifest_bytes = (root / manifest_name).read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S4InputError(f"expanded {label} manifest is invalid") from error
    declared = manifest.get("members") if type(manifest) is dict else None
    if type(declared) is not dict:
        raise S4InputError(f"expanded {label} member manifest differs")
    payloads: dict[str, bytes] = {manifest_name: manifest_bytes}
    total = len(manifest_bytes)
    for name, evidence in sorted(declared.items()):
        pure = PurePosixPath(name)
        if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts) or "\\" in name:
            raise S4InputError(f"unsafe expanded {label} member: {name}")
        target = root.joinpath(*pure.parts)
        if target.is_file() and not target.is_symlink():
            payload = target.read_bytes()
        elif name.endswith(".zip") and target.with_suffix("").is_dir():
            payload = _expanded_zip_payload(target.with_suffix(""), f"{label} member {name}")
        else:
            raise S4InputError(f"expanded {label} member is absent: {name}")
        total += len(payload)
        if len(payload) > _MAX_MEMBER_BYTES or total > _MAX_BYTES:
            raise S4InputError(f"expanded {label} size exceeds limit")
        if (
            type(evidence) is not dict or set(evidence) != {"size", "sha256"}
            or evidence["size"] != len(payload)
            or evidence["sha256"] != sha256(payload).hexdigest()
        ):
            raise S4InputError(f"expanded {label} member differs: {name}")
        payloads[name] = payload
    output = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
    try:
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in sorted(payloads.items()):
                archive.writestr(_info(name), payload)
        output.seek(0)
        return output.read()
    finally:
        output.close()


def _s4_input_payloads(source: Path) -> dict[str, bytes]:
    root = Path(source)
    if not root.is_dir():
        return _payloads(root)
    if root.is_symlink():
        raise S4InputError("unsafe S4 input source")
    manifest = root / "manifest.json"
    handoff = root / "e2/handoff.zip"
    expanded_handoff = root / "e2/handoff"
    if manifest.is_symlink() or not manifest.is_file():
        raise S4InputError("S4 input manifest is absent")
    if handoff.is_file() and not handoff.is_symlink():
        handoff_payload = handoff.read_bytes()
    elif expanded_handoff.is_dir() and not expanded_handoff.is_symlink():
        handoff_payload = _expanded_zip_payload(expanded_handoff, "E2 handoff")
    else:
        raise S4InputError("S4 input E2 handoff is absent")
    return {"manifest.json": manifest.read_bytes(), "e2/handoff.zip": handoff_payload}


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
    delivery = _verify_e2_handoff(source)
    actual = file_sha256(source)
    if not delivery or actual != expected:
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
    payloads = _s4_input_payloads(Path(source))
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
    if not _verify_e2_handoff(handoff_path):
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
