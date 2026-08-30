from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .inputs import canonical_json, file_sha256


class FinalArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactBindings:
    code_sha256: str
    contract_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str


_KINDS = {
    "review": "gated_residual_final_review_v1",
    "handoff": "gated_residual_final_handoff_v1",
    "delivery": "gated_residual_final_delivery_v1",
}
_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_BLOCK = 1024 * 1024


def _binding_payload(bindings: ArtifactBindings) -> dict[str, str]:
    payload = asdict(bindings)
    if any(type(value) is not str or len(value) != 64 for value in payload.values()):
        raise FinalArtifactError("artifact bindings differ")
    return payload


def _member_metadata(path: Path) -> dict[str, object]:
    return {"sha256": file_sha256(path), "size": path.stat().st_size}


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, _ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_file(archive: ZipFile, name: str, path: Path) -> None:
    with Path(path).open("rb") as source, archive.open(_zip_info(name), "w", force_zip64=True) as target:
        for block in iter(lambda: source.read(_BLOCK), b""):
            target.write(block)


def _create_bundle(
    path: Path,
    *,
    kind: str,
    bindings: ArtifactBindings,
    payloads: Mapping[str, Path],
    acceptance_evidence: Mapping[str, object] | None = None,
) -> Path:
    if kind not in _KINDS or not payloads:
        raise FinalArtifactError("artifact configuration differs")
    if any(
        PurePosixPath(name).is_absolute() or not PurePosixPath(name).parts
        or any(part in {"", ".", ".."} for part in PurePosixPath(name).parts)
        or name == "manifest.json" or not Path(source).is_file() or Path(source).is_symlink()
        for name, source in payloads.items()
    ):
        raise FinalArtifactError("artifact payload differs")
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FinalArtifactError("artifact output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "artifact_kind": _KINDS[kind],
        "bindings": _binding_payload(bindings),
        "members": {name: _member_metadata(Path(source)) for name, source in sorted(payloads.items())},
    }
    if acceptance_evidence is not None:
        manifest["acceptance_evidence"] = dict(acceptance_evidence)
    with ZipFile(target, "w", allowZip64=True) as archive:
        for name, source in sorted(payloads.items()):
            _write_file(archive, name, Path(source))
        archive.writestr(_zip_info("manifest.json"), canonical_json(manifest))
    return target


def create_handoff(path: Path, *, bindings: ArtifactBindings, payloads: Mapping[str, Path]) -> Path:
    return _create_bundle(path, kind="handoff", bindings=bindings, payloads=payloads)


def create_review(path: Path, *, bindings: ArtifactBindings, payloads: Mapping[str, Path]) -> Path:
    return _create_bundle(path, kind="review", bindings=bindings, payloads=payloads)


def create_delivery(
    path: Path,
    *,
    bindings: ArtifactBindings,
    payloads: Mapping[str, Path],
    acceptance_evidence: Mapping[str, object],
) -> Path:
    if acceptance_evidence.get("status") != "accepted" or type(acceptance_evidence.get("candidate_id")) is not str:
        raise FinalArtifactError("accepted evidence is required")
    return _create_bundle(
        path, kind="delivery", bindings=bindings, payloads=payloads,
        acceptance_evidence=acceptance_evidence,
    )


def verify_bundle(
    path: Path, *, kind: str, expected_bindings: ArtifactBindings,
) -> Mapping[str, object]:
    if kind not in _KINDS:
        raise FinalArtifactError("artifact kind differs")
    try:
        with ZipFile(path) as archive:
            infos = {}
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    info.filename in infos or info.is_dir() or info.flag_bits & 1
                    or member.is_absolute() or not member.parts
                    or any(part in {"", ".", ".."} for part in member.parts)
                    or "\\" in info.filename or stat.S_ISLNK(mode)
                ):
                    raise FinalArtifactError(f"unsafe artifact member: {info.filename}")
                infos[info.filename] = info
            if "manifest.json" not in infos:
                raise FinalArtifactError("artifact manifest is absent")
            manifest = json.loads(archive.read(infos["manifest.json"]))
            expected_keys = {"artifact_kind", "bindings", "members"}
            if kind == "delivery":
                expected_keys.add("acceptance_evidence")
            if type(manifest) is not dict or set(manifest) != expected_keys:
                raise FinalArtifactError("artifact manifest differs")
            if manifest["artifact_kind"] != _KINDS[kind]:
                raise FinalArtifactError("artifact identity differs")
            if manifest["bindings"] != _binding_payload(expected_bindings):
                raise FinalArtifactError("artifact bindings differ")
            members = manifest["members"]
            if type(members) is not dict or set(members) != set(infos) - {"manifest.json"}:
                raise FinalArtifactError("artifact member set differs")
            for name, declared in members.items():
                payload = archive.read(infos[name])
                actual = {"sha256": sha256(payload).hexdigest(), "size": len(payload)}
                if declared != actual:
                    raise FinalArtifactError(f"artifact member digest differs: {name}")
            if kind == "delivery" and manifest["acceptance_evidence"].get("status") != "accepted":
                raise FinalArtifactError("accepted evidence is required")
            return MappingProxyType(manifest)
    except (OSError, BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, FinalArtifactError):
            raise
        raise FinalArtifactError("artifact is unreadable") from error
