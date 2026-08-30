from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from .contracts import load_contract


class FinalInputError(ValueError):
    pass


_PAYLOAD_NAMES = ("e2/input.bin", "direct/stage_a.bin", "direct/stage_b.bin", "sources.json")
_MEMBERS = frozenset({*_PAYLOAD_NAMES, "manifest.json"})
_ALLOWED_DATASET_METADATA = frozenset({"dataset-metadata.json"})
_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_BLOCK = 1024 * 1024


@dataclass(frozen=True)
class VerifiedFinalInput:
    root: Path
    manifest_sha256: str
    source_hashes: Mapping[str, str]
    e2_input: Path
    stage_a: Path
    stage_b: Path


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FinalInputError("JSON value is not canonicalizable") from error


def file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FinalInputError(f"source is not a regular file: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _info(name: str, *, stored: bool) -> ZipInfo:
    info = ZipInfo(name, _ZIP_TIME)
    info.compress_type = ZIP_STORED if stored else ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _metadata(payload: bytes) -> dict[str, object]:
    return {"sha256": sha256(payload).hexdigest(), "size": len(payload)}


def prepare_final_input(
    *,
    e2_input: Path,
    stage_a: Path,
    stage_b: Path,
    output: Path,
    expected_hashes: Mapping[str, str] | None = None,
) -> Path:
    sources = {"e2_input": Path(e2_input), "stage_a": Path(stage_a), "stage_b": Path(stage_b)}
    expected = dict(expected_hashes or {
        "e2_input": load_contract().expected_hashes["direct_expert_input"],
        "stage_a": load_contract().expected_hashes["stage_a_handoff"],
        "stage_b": load_contract().expected_hashes["stage_b_handoff"],
    })
    if set(expected) != set(sources):
        raise FinalInputError("expected source hash keys differ")
    actual = {name: file_sha256(path) for name, path in sources.items()}
    if actual != expected:
        raise FinalInputError("source SHA-256 differs")
    payloads = {
        "e2/input.bin": sources["e2_input"].read_bytes(),
        "direct/stage_a.bin": sources["stage_a"].read_bytes(),
        "direct/stage_b.bin": sources["stage_b"].read_bytes(),
        "sources.json": canonical_json({"hashes": actual}),
    }
    manifest = canonical_json({
        "artifact_kind": "gated_residual_final_input_v1",
        "members": {name: _metadata(payload) for name, payload in sorted(payloads.items())},
        "source_hashes": actual,
    })
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FinalInputError("output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", allowZip64=True) as archive:
        for name, payload in sorted(payloads.items()):
            archive.writestr(_info(name, stored=name.endswith(".bin")), payload)
        archive.writestr(_info("manifest.json", stored=False), manifest)
    return target


def _safe_zip_payloads(source: Path) -> dict[str, bytes]:
    try:
        with ZipFile(source) as archive:
            output: dict[str, bytes] = {}
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    info.filename in output or info.is_dir() or info.flag_bits & 1
                    or path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts)
                    or "\\" in info.filename or stat.S_ISLNK(mode)
                ):
                    raise FinalInputError(f"unsafe final input member: {info.filename}")
                output[info.filename] = archive.read(info)
            return output
    except (OSError, BadZipFile) as error:
        if isinstance(error, FinalInputError):
            raise
        raise FinalInputError("final input is not a valid ZIP") from error


def _directory_payloads(source: Path) -> dict[str, bytes]:
    output: dict[str, bytes] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise FinalInputError("final input contains a symlink")
        if not path.is_file():
            continue
        name = path.relative_to(source).as_posix()
        if name in _ALLOWED_DATASET_METADATA:
            continue
        output[name] = path.read_bytes()
    return output


def _payloads(source: Path) -> dict[str, bytes]:
    path = Path(source)
    if path.is_file() and not path.is_symlink():
        return _safe_zip_payloads(path)
    if path.is_dir() and not path.is_symlink():
        return _directory_payloads(path)
    raise FinalInputError("final input source differs")


def verify_and_extract_final_input(source: Path, destination: Path) -> VerifiedFinalInput:
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FinalInputError("destination already exists")
    payloads = _payloads(Path(source))
    if set(payloads) != _MEMBERS:
        raise FinalInputError("final input member set differs")
    try:
        manifest = json.loads(payloads["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalInputError("manifest is unreadable") from error
    if type(manifest) is not dict or set(manifest) != {"artifact_kind", "members", "source_hashes"}:
        raise FinalInputError("manifest keys differ")
    if manifest["artifact_kind"] != "gated_residual_final_input_v1":
        raise FinalInputError("final input identity differs")
    declared = manifest["members"]
    if type(declared) is not dict or set(declared) != set(_PAYLOAD_NAMES):
        raise FinalInputError("declared member set differs")
    for name in _PAYLOAD_NAMES:
        if declared[name] != _metadata(payloads[name]):
            raise FinalInputError(f"member digest differs: {name}")
    source_hashes = manifest["source_hashes"]
    if type(source_hashes) is not dict or set(source_hashes) != {"e2_input", "stage_a", "stage_b"}:
        raise FinalInputError("source hashes differ")
    if json.loads(payloads["sources.json"]) != {"hashes": source_hashes}:
        raise FinalInputError("source evidence differs")
    binary_map = {
        "e2/input.bin": "direct_expert_input.zip",
        "direct/stage_a.bin": "direct_expert_stage_A_handoff.zip",
        "direct/stage_b.bin": "direct_expert_stage_B_handoff.zip",
    }
    target.mkdir(parents=True)
    for source_name, output_name in binary_map.items():
        (target / output_name).write_bytes(payloads[source_name])
    return VerifiedFinalInput(
        root=target,
        manifest_sha256=sha256(payloads["manifest.json"]).hexdigest(),
        source_hashes=MappingProxyType(dict(source_hashes)),
        e2_input=target / binary_map["e2/input.bin"],
        stage_a=target / binary_map["direct/stage_a.bin"],
        stage_b=target / binary_map["direct/stage_b.bin"],
    )
