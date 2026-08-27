from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .rf_decisions import RFAcceptanceDecision, acceptance_payload


class RFArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class RFBindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    official_train_sha256: str
    official_history_sha256: str
    e2_handoff_sha256: str


@dataclass(frozen=True)
class VerifiedRFBundle:
    path: Path
    kind: str
    manifest_sha256: str
    root: Path | None = None
    completed_job_ids: tuple[str, ...] = ()


_BINDING_KEYS = {
    "contract_sha256",
    "code_sha256",
    "input_manifest_sha256",
    "official_train_sha256",
    "official_history_sha256",
    "e2_handoff_sha256",
}
_MAX_EXPANDED = 4 * 1024 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _binding_payload(bindings: RFBindings) -> dict[str, str]:
    if type(bindings) is not RFBindings:
        raise RFArtifactError("artifact bindings type differs")
    result = {key: str(getattr(bindings, key)) for key in sorted(_BINDING_KEYS)}
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in result.values()):
        raise RFArtifactError("artifact bindings differ")
    return result


def _stable_sources(root: Path, *, include_models: bool) -> dict[str, Path]:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise RFArtifactError("campaign root differs")
    result: dict[str, Path] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source)
        if (
            path.name.startswith(".")
            or path.name.endswith(".tmp")
            or "catboost_info" in relative.parts
            or any(part in {"bundles", "snapshots", "restored_resume"} for part in relative.parts)
            or (not include_models and path.suffix in {".cbm", ".cbsnapshot", ".pt"})
            or (not include_models and relative.parts and relative.parts[0] == "full_fit")
        ):
            continue
        result[relative.as_posix()] = path
    if "campaign_state.json" not in result:
        raise RFArtifactError("campaign state is absent")
    return result


def _write_bundle(
    output: Path,
    *,
    kind: str,
    bindings: RFBindings,
    members: Mapping[str, Path | bytes],
) -> Path:
    payloads: dict[str, bytes] = {}
    for name, value in members.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or name == "manifest.json":
            raise RFArtifactError("artifact member path differs")
        if isinstance(value, bytes):
            payload = value
        else:
            source = Path(value)
            if source.is_symlink() or not source.is_file():
                raise RFArtifactError(f"artifact source differs: {name}")
            payload = source.read_bytes()
        payloads[name] = payload
    manifest = {
        "schema_version": 1,
        "artifact_kind": kind,
        "campaign_id": "tree_expert_rf_v1",
        "review_only": kind == "tree_expert_rf_review_v1",
        "submission_package": False,
        "bindings": _binding_payload(bindings),
        "members": {
            name: {"sha256": sha256(payload).hexdigest(), "size": len(payload)}
            for name, payload in sorted(payloads.items())
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with ZipFile(temporary, "w") as archive:
            for name, payload in sorted(payloads.items()):
                archive.writestr(_zip_info(name), payload)
            archive.writestr(_zip_info("manifest.json"), _canonical(manifest))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    _verify_bundle(destination, kind, bindings)
    return destination


def _safe_infos(archive: ZipFile) -> dict[str, ZipInfo]:
    result: dict[str, ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            info.filename in result
            or info.flag_bits & 0x1
            or info.is_dir()
            or mode & 0o170000 == 0o120000
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in info.filename
        ):
            raise RFArtifactError("unsafe or duplicate artifact member")
        total += info.file_size
        if total > _MAX_EXPANDED:
            raise RFArtifactError("artifact expanded size exceeds limit")
        result[info.filename] = info
    return result


def _verify_bundle(path: Path, kind: str, bindings: RFBindings) -> VerifiedRFBundle:
    try:
        with ZipFile(Path(path)) as archive:
            infos = _safe_infos(archive)
            if "manifest.json" not in infos:
                raise RFArtifactError("artifact manifest is absent")
            manifest_bytes = archive.read(infos["manifest.json"])
            manifest = json.loads(manifest_bytes)
            if (
                type(manifest) is not dict
                or manifest.get("schema_version") != 1
                or manifest.get("artifact_kind") != kind
                or manifest.get("campaign_id") != "tree_expert_rf_v1"
                or manifest.get("submission_package") is not False
            ):
                raise RFArtifactError("artifact identity differs")
            if manifest.get("bindings") != _binding_payload(bindings):
                raise RFArtifactError("artifact bindings differ")
            names = set(infos) - {"manifest.json"}
            declared = manifest.get("members")
            if type(declared) is not dict or set(declared) != names:
                raise RFArtifactError("artifact manifest members differ")
            for name in sorted(names):
                payload = archive.read(infos[name])
                evidence = declared[name]
                if (
                    type(evidence) is not dict
                    or evidence.get("size") != len(payload)
                    or evidence.get("sha256") != sha256(payload).hexdigest()
                ):
                    raise RFArtifactError(f"member evidence differs: {name}")
            return VerifiedRFBundle(Path(path), kind, sha256(manifest_bytes).hexdigest())
    except RFArtifactError:
        raise
    except (OSError, BadZipFile, json.JSONDecodeError) as error:
        raise RFArtifactError(f"cannot verify artifact: {error}") from error


def create_rf_resume(campaign_root: Path, output: Path, bindings: RFBindings) -> Path:
    return _write_bundle(
        output,
        kind="tree_expert_rf_resume_v1",
        bindings=bindings,
        members=_stable_sources(campaign_root, include_models=True),
    )


def verify_rf_resume(path: Path, expected_bindings: RFBindings) -> VerifiedRFBundle:
    return _verify_bundle(path, "tree_expert_rf_resume_v1", expected_bindings)


def restore_rf_resume(path: Path, destination: Path, bindings: RFBindings) -> VerifiedRFBundle:
    verified = verify_rf_resume(path, bindings)
    root = Path(destination)
    if root.exists():
        raise RFArtifactError("resume destination already exists")
    root.mkdir(parents=True)
    with ZipFile(path) as archive:
        infos = _safe_infos(archive)
        for name, info in infos.items():
            if name == "manifest.json":
                continue
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise RFArtifactError("unsafe resume destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    completed: list[str] = []
    for metrics_path in sorted((root / "jobs").glob("*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RFArtifactError("restored job metrics differ") from error
        if metrics.get("status") == "completed" and metrics.get("job_id") == metrics_path.parent.name:
            completed.append(metrics_path.parent.name)
    return VerifiedRFBundle(
        verified.path,
        verified.kind,
        verified.manifest_sha256,
        root=root,
        completed_job_ids=tuple(completed),
    )


def create_rf_review(campaign_root: Path, output: Path, bindings: RFBindings) -> Path:
    return _write_bundle(
        output,
        kind="tree_expert_rf_review_v1",
        bindings=bindings,
        members=_stable_sources(campaign_root, include_models=False),
    )


def verify_rf_review(path: Path, expected_bindings: RFBindings) -> VerifiedRFBundle:
    return _verify_bundle(path, "tree_expert_rf_review_v1", expected_bindings)


def create_rf_delivery(
    campaign_root: Path,
    output: Path,
    decision: RFAcceptanceDecision,
    bindings: RFBindings,
) -> Path:
    if type(decision) is not RFAcceptanceDecision or decision.status != "accepted":
        raise RFArtifactError("accepted decision is required")
    root = Path(campaign_root)
    audit_path = root / "audits/independence.json"
    manifest_path = root / "full_fit/full_fit_manifest.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        full_fit = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RFArtifactError("delivery evidence is absent") from error
    if audit.get("status") != "passed" or full_fit.get("status") != "accepted":
        raise RFArtifactError("delivery evidence is not accepted")
    members = {
        name: path
        for name, path in _stable_sources(root, include_models=True).items()
        if name.startswith("full_fit/") or name in {"audits/independence.json", "decisions/acceptance.json"}
    }
    members["evidence/accepted_decision.json"] = _canonical(acceptance_payload(decision))
    delivery = _write_bundle(
        output,
        kind="tree_expert_rf_delivery_v1",
        bindings=bindings,
        members=members,
    )
    with ZipFile(delivery) as archive:
        if {"script.py", "submission.csv"}.intersection(archive.namelist()):
            raise RFArtifactError("delivery contains a submission entry point")
    return delivery


def verify_rf_delivery(path: Path, expected_bindings: RFBindings) -> VerifiedRFBundle:
    return _verify_bundle(path, "tree_expert_rf_delivery_v1", expected_bindings)
