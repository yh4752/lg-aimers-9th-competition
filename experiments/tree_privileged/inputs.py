from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from experiments.tree_expert.e2_artifacts import E2ArtifactError, verify_e2_handoff

from .contracts import contract_sha256, load_contract


class PrivilegedInputError(ValueError):
    pass


INPUT_KIND = "tree_privileged_input_v1"
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_PAYLOAD_NAMES = {
    "e2/model_delivery.zip",
    "e2/oof/2022.csv",
    "e2/oof/2023.csv",
    "e2/oof/2024.csv",
}
_REVIEW_NAMES = {
    "e2/oof/2022.csv": "ensembles/2021_2022.csv",
    "e2/oof/2023.csv": "ensembles/2022_2023.csv",
    "e2/oof/2024.csv": "ensembles/2023_2024.csv",
}
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_RATIO = 250.0


@dataclass(frozen=True)
class VerifiedPrivilegedInput:
    root: Path
    manifest_sha256: str
    e2_handoff_sha256: str
    e2_delivery: Path
    e2_oof_root: Path


def file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PrivilegedInputError(f"input source is not a regular file: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PrivilegedInputError("input manifest cannot be serialized") from error


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _safe_infos(archive: ZipFile, label: str) -> dict[str, ZipInfo]:
    output: dict[str, ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        mode = info.external_attr >> 16
        ratio = info.file_size / max(1, info.compress_size)
        if name in output:
            raise PrivilegedInputError(f"{label} has duplicate members")
        if (
            path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts)
            or info.is_dir() or stat.S_ISLNK(mode) or info.file_size > _MAX_MEMBER_BYTES
            or (info.file_size >= 64 * 1024 and ratio > _MAX_RATIO)
        ):
            raise PrivilegedInputError(f"unsafe {label} member: {name}")
        total += info.file_size
        output[name] = info
    if total > _MAX_TOTAL_BYTES:
        raise PrivilegedInputError(f"{label} is too large")
    return output


def _evidence(payload: bytes) -> dict[str, object]:
    return {"size": len(payload), "sha256": sha256(payload).hexdigest()}


def _write_zip(path: Path, members: Mapping[str, bytes]) -> Path:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in sorted(members.items()):
                archive.writestr(_info(name), payload)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _read_e2_payloads(source: Path) -> dict[str, bytes]:
    try:
        with ZipFile(source) as outer:
            outer_infos = _safe_infos(outer, "E2 handoff")
            review_payload = outer.read(outer_infos["tree_expert_e2_review.zip"])
            delivery = outer.read(outer_infos["tree_expert_e2_model_delivery.zip"])
        with ZipFile(io.BytesIO(review_payload)) as review:
            review_infos = _safe_infos(review, "E2 review")
            payloads = {output: review.read(review_infos[source_name])
                        for output, source_name in _REVIEW_NAMES.items()}
    except (BadZipFile, KeyError) as error:
        raise PrivilegedInputError("accepted E2 handoff nested members differ") from error
    payloads["e2/model_delivery.zip"] = delivery
    return payloads


def prepare_input(
    e2_handoff: Path,
    output: Path,
    *,
    expected_e2_sha256: str | None = None,
) -> Path:
    source = Path(e2_handoff)
    expected = expected_e2_sha256 or load_contract().inputs["e2_handoff_sha256"]
    actual = file_sha256(source)
    if actual != expected:
        raise PrivilegedInputError("E2 handoff SHA-256 differs")
    try:
        verified = verify_e2_handoff(source)
    except E2ArtifactError as error:
        raise PrivilegedInputError(f"E2 handoff verification failed: {error}") from error
    if verified.status != "accepted" or verified.delivery is not True:
        raise PrivilegedInputError("E2 handoff is not accepted with delivery")
    payloads = _read_e2_payloads(source)
    manifest = _canonical({
        "schema_version": 1,
        "artifact_kind": INPUT_KIND,
        "campaign_id": "tree_privileged_profile_v1",
        "contract_sha256": contract_sha256(),
        "e2_handoff_sha256": actual,
        "members": {name: _evidence(payload) for name, payload in sorted(payloads.items())},
    })
    return _write_zip(Path(output), {**payloads, "manifest.json": manifest})


def _source_payloads(path: Path) -> dict[str, bytes]:
    source = Path(path)
    try:
        with ZipFile(source) as archive:
            infos = _safe_infos(archive, "privileged input")
            return {name: archive.read(info) for name, info in infos.items()}
    except BadZipFile as error:
        raise PrivilegedInputError("privileged input is not a ZIP") from error


def verify_and_extract_input(
    source: Path,
    destination: Path,
    *,
    expected_e2_sha256: str | None = None,
) -> VerifiedPrivilegedInput:
    payloads = _source_payloads(Path(source))
    if set(payloads) != {*_PAYLOAD_NAMES, "manifest.json"}:
        raise PrivilegedInputError("input member set differs")
    try:
        manifest = json.loads(payloads["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrivilegedInputError("input manifest is invalid") from error
    expected_sha = expected_e2_sha256 or load_contract().inputs["e2_handoff_sha256"]
    expected_manifest = _canonical({
        "schema_version": 1,
        "artifact_kind": INPUT_KIND,
        "campaign_id": "tree_privileged_profile_v1",
        "contract_sha256": contract_sha256(),
        "e2_handoff_sha256": expected_sha,
        "members": {name: _evidence(payloads[name]) for name in sorted(_PAYLOAD_NAMES)},
    })
    if payloads["manifest.json"] != expected_manifest:
        raise PrivilegedInputError("input manifest or binding differs")
    root = Path(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in sorted(_PAYLOAD_NAMES):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[name])
    return VerifiedPrivilegedInput(
        root=root,
        manifest_sha256=sha256(payloads["manifest.json"]).hexdigest(),
        e2_handoff_sha256=expected_sha,
        e2_delivery=root / "e2/model_delivery.zip",
        e2_oof_root=root / "e2/oof",
    )

