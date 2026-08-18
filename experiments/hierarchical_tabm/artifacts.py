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
from typing import Callable, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .inputs import EXPECTED_BINDING_KEYS


class HierarchicalArtifactError(ValueError):
    """Raised when campaign evidence is incomplete, unsafe, or inconsistent."""


@dataclass(frozen=True)
class CampaignBundles:
    review: Path
    resume: Path


@dataclass(frozen=True)
class CampaignEvidence:
    bindings: Mapping[str, str]
    contract_path: Path
    log_path: Path
    state_path: Path
    k_selection_path: Path | None
    completed_job_directories: Mapping[str, Path]
    calibration_paths: Mapping[str, Path]
    decision_paths: Mapping[str, Path]
    active_job_directory: Path | None


@dataclass(frozen=True)
class DeliveryEvidence:
    bindings: Mapping[str, str]
    delivery_roles: Mapping[str, str]
    final_checkpoint_path: Path
    feature_state_path: Path
    calibration_paths: Mapping[str, Path]
    independence_report_paths: Mapping[str, Path]


@dataclass(frozen=True)
class RestoredCampaign:
    root: Path
    state: Mapping[str, object]
    completed_job_directories: Mapping[str, Path]
    active_job_directory: Path | None


_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_MAX_MEMBER = 2 * 1024 * 1024 * 1024
_MAX_TOTAL = 4 * 1024 * 1024 * 1024
_MAX_RATIO = 200.0
_MAX_MANIFEST = 2 * 1024 * 1024
_ROLES = {"final_candidate", "public_diagnostic_only"}
_CANDIDATE_ORDER = ("H1", "H2", "H3")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _file_sha256(path: Path, check_deadline: Callable[[], None] | None = None) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if check_deadline is not None:
                check_deadline()
            digest.update(chunk)
    return digest.hexdigest()


def _bindings(value: Mapping[str, str]) -> dict[str, str]:
    if set(value) != EXPECTED_BINDING_KEYS:
        raise HierarchicalArtifactError("artifact bindings keys differ")
    output: dict[str, str] = {}
    for key in sorted(value):
        digest = value[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise HierarchicalArtifactError("artifact bindings contain an invalid SHA-256")
        output[key] = digest
    return output


def _safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or any("test" in part.lower() or "submit" in part.lower() for part in path.parts)
    ):
        raise HierarchicalArtifactError(f"unsafe artifact member: {name}")


def _source(path: Path) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file() or not stat.S_ISREG(value.stat().st_mode):
        raise HierarchicalArtifactError(f"artifact source is not a regular file: {value}")
    return value


def _directory_members(root: Path, prefix: str) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise HierarchicalArtifactError(f"artifact directory is invalid: {root}")
    output: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HierarchicalArtifactError("artifact directory contains a symlink")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        name = f"{prefix}/{relative}"
        _safe_name(name)
        output[name] = _source(path)
    if not output:
        raise HierarchicalArtifactError("artifact directory is empty")
    return output


def _campaign_members(evidence: CampaignEvidence, *, resume: bool) -> dict[str, Path]:
    members = {
        "contract.json": _source(evidence.contract_path),
        "campaign.log": _source(evidence.log_path),
        "stage_state.json": _source(evidence.state_path),
    }
    if evidence.k_selection_path is not None:
        members["k_selection.json"] = _source(evidence.k_selection_path)
    for job_id, directory in evidence.completed_job_directories.items():
        _safe_name(f"jobs/{job_id}")
        members.update(_directory_members(directory, f"jobs/{job_id}"))
    for candidate_id, path in evidence.calibration_paths.items():
        members[f"calibration/{candidate_id}.json"] = _source(path)
    for candidate_id, path in evidence.decision_paths.items():
        members[f"decisions/{candidate_id}.json"] = _source(path)
    if resume and evidence.active_job_directory is not None:
        state = _read_json(evidence.state_path, "stage state")
        active_id = state.get("active_job_id")
        if not isinstance(active_id, str) or not active_id:
            raise HierarchicalArtifactError("active job directory lacks state identity")
        members.update(
            _directory_members(evidence.active_job_directory, f"active/{active_id}")
        )
    if len(members) != len(set(members)):
        raise HierarchicalArtifactError("artifact member collision")
    return members


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _publish(
    target: Path,
    *,
    kind: str,
    bindings: Mapping[str, str],
    members: Mapping[str, Path],
    extra_manifest: Mapping[str, object] | None = None,
    check_deadline: Callable[[], None] | None = None,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise HierarchicalArtifactError(f"artifact already exists: {target}")
    trusted: dict[str, dict[str, object]] = {}
    for name, path in sorted(members.items()):
        _safe_name(name)
        source = _source(path)
        trusted[name] = {"size": source.stat().st_size, "sha256": _file_sha256(source, check_deadline)}
    manifest = {
        "schema_version": 1,
        "artifact_kind": kind,
        "review_only": True,
        "submission_package": False,
        "bindings": _bindings(bindings),
        "members": trusted,
        **dict(extra_manifest or {}),
    }
    manifest_bytes = _canonical_json(manifest)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(_zip_info("manifest.json"), manifest_bytes)
            for name, path in sorted(members.items()):
                digest = sha256()
                size = 0
                with path.open("rb") as source_stream, archive.open(_zip_info(name), "w") as destination:
                    while chunk := source_stream.read(1024 * 1024):
                        if check_deadline is not None:
                            check_deadline()
                        destination.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != trusted[name]["size"] or digest.hexdigest() != trusted[name]["sha256"]:
                    raise HierarchicalArtifactError("artifact source changed during publication")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    return target


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HierarchicalArtifactError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise HierarchicalArtifactError(f"{label} must be an object")
    return value


def _validate_info(info: ZipInfo) -> None:
    _safe_name(info.filename)
    mode = info.external_attr >> 16
    if (
        info.is_dir()
        or stat.S_IFMT(mode) == stat.S_IFLNK
        or info.file_size > _MAX_MEMBER
        or (info.file_size >= 64 * 1024 and info.file_size / max(1, info.compress_size) > _MAX_RATIO)
    ):
        raise HierarchicalArtifactError(f"unsafe artifact member: {info.filename}")


def _verify(
    path: Path, *, kind: str, expected_bindings: Mapping[str, str]
) -> dict[str, object]:
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise HierarchicalArtifactError("artifact member inventory differs")
            if sum(info.file_size for info in infos) > _MAX_TOTAL:
                raise HierarchicalArtifactError("artifact total size exceeds limit")
            for info in infos:
                _validate_info(info)
            manifest_info = archive.getinfo("manifest.json")
            if manifest_info.file_size > _MAX_MANIFEST:
                raise HierarchicalArtifactError("artifact manifest exceeds limit")
            try:
                manifest = json.loads(archive.read(manifest_info))
            except Exception as error:
                raise HierarchicalArtifactError("artifact manifest is unreadable") from error
            if type(manifest) is not dict:
                raise HierarchicalArtifactError("artifact manifest must be an object")
            if (
                manifest.get("schema_version") != 1
                or manifest.get("artifact_kind") != kind
                or manifest.get("review_only") is not True
                or manifest.get("submission_package") is not False
                or manifest.get("bindings") != _bindings(expected_bindings)
            ):
                raise HierarchicalArtifactError("artifact bindings or identity differ")
            declared = manifest.get("members")
            if type(declared) is not dict or set(declared) != set(names) - {"manifest.json"}:
                raise HierarchicalArtifactError("artifact member manifest differs")
            for name in sorted(declared):
                evidence = declared[name]
                if type(evidence) is not dict or set(evidence) != {"size", "sha256"}:
                    raise HierarchicalArtifactError("artifact member evidence differs")
                digest = sha256()
                size = 0
                with archive.open(name) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                if size != evidence["size"] or digest.hexdigest() != evidence["sha256"]:
                    raise HierarchicalArtifactError("artifact member hash differs")
            state = json.loads(archive.read("stage_state.json")) if "stage_state.json" in names else None
            if kind in {"hierarchical_tabm_review_v1", "hierarchical_tabm_resume_v1"}:
                if type(state) is not dict or type(state.get("completed_job_ids")) is not list:
                    raise HierarchicalArtifactError("stage state is invalid")
                completed = tuple(state["completed_job_ids"])
                observed = tuple(
                    sorted({name.split("/")[1] for name in names if name.startswith("jobs/")})
                )
                if tuple(sorted(completed)) != observed:
                    raise HierarchicalArtifactError("completed job evidence differs")
                active = state.get("active_job_id")
                active_members = [name for name in names if name.startswith("active/")]
                if kind.endswith("resume_v1"):
                    if (active is None) != (not active_members):
                        raise HierarchicalArtifactError("active job evidence differs")
                elif active_members:
                    raise HierarchicalArtifactError("review contains active checkpoint")
            return manifest
    except HierarchicalArtifactError:
        raise
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise HierarchicalArtifactError(f"cannot verify artifact: {error}") from error


def write_campaign_bundles(
    evidence: CampaignEvidence,
    output_dir: Path,
    *,
    check_deadline: Callable[[], None] | None = None,
) -> CampaignBundles:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    review = _publish(
        output / "hierarchical_tabm_review.zip",
        kind="hierarchical_tabm_review_v1",
        bindings=evidence.bindings,
        members=_campaign_members(evidence, resume=False),
        check_deadline=check_deadline,
    )
    resume = _publish(
        output / "hierarchical_tabm_resume.zip",
        kind="hierarchical_tabm_resume_v1",
        bindings=evidence.bindings,
        members=_campaign_members(evidence, resume=True),
        check_deadline=check_deadline,
    )
    verify_review_bundle(review, expected_bindings=evidence.bindings)
    verify_resume_bundle(resume, expected_bindings=evidence.bindings)
    return CampaignBundles(review, resume)


def verify_review_bundle(
    path: Path, *, expected_bindings: Mapping[str, str]
) -> Mapping[str, object]:
    return MappingProxyType(
        _verify(Path(path), kind="hierarchical_tabm_review_v1", expected_bindings=expected_bindings)
    )


def verify_resume_bundle(
    path: Path, *, expected_bindings: Mapping[str, str]
) -> Mapping[str, object]:
    return MappingProxyType(
        _verify(Path(path), kind="hierarchical_tabm_resume_v1", expected_bindings=expected_bindings)
    )


def restore_resume(
    path: Path,
    destination: Path,
    *,
    expected_bindings: Mapping[str, str],
    check_deadline: Callable[[], None] | None = None,
) -> RestoredCampaign:
    verify_resume_bundle(path, expected_bindings=expected_bindings)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise HierarchicalArtifactError("resume destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        with ZipFile(path) as archive:
            for info in archive.infolist():
                if info.filename == "manifest.json":
                    continue
                target = temporary / PurePosixPath(info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        if check_deadline is not None:
                            check_deadline()
                        output.write(chunk)
        state = _read_json(temporary / "stage_state.json", "stage state")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    completed = {
        job_id: destination / "jobs" / job_id
        for job_id in state["completed_job_ids"]
    }
    active_id = state.get("active_job_id")
    return RestoredCampaign(
        destination,
        MappingProxyType(state),
        MappingProxyType(completed),
        None if active_id is None else destination / "active" / str(active_id),
    )


def write_candidate_delivery(
    evidence: DeliveryEvidence,
    output_dir: Path,
    *,
    check_deadline: Callable[[], None] | None = None,
) -> Path:
    roles = dict(evidence.delivery_roles)
    ordered_ids = [candidate for candidate in _CANDIDATE_ORDER if candidate in roles]
    if set(roles) != set(ordered_ids) or not roles or any(role not in _ROLES for role in roles.values()):
        raise HierarchicalArtifactError("candidate delivery roles differ")
    if set(evidence.independence_report_paths) != set(roles):
        raise HierarchicalArtifactError("candidate independence reports differ")
    if any(candidate != "H1" and roles[candidate] == "public_diagnostic_only" for candidate in roles):
        raise HierarchicalArtifactError("only H1 may be diagnostic-only")
    members = {
        "model/final_checkpoint.pt": _source(evidence.final_checkpoint_path),
        "state/feature_state.json": _source(evidence.feature_state_path),
    }
    for candidate, path in evidence.calibration_paths.items():
        if candidate not in {"H2", "H3"} or candidate not in roles:
            raise HierarchicalArtifactError("candidate calibration set differs")
        members[f"state/calibration_{candidate}.json"] = _source(path)
    for candidate, path in evidence.independence_report_paths.items():
        members[f"evidence/decision_{candidate}.json"] = _source(path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = _publish(
        output / "hierarchical_tabm_candidate_delivery.zip",
        kind="hierarchical_tabm_candidate_delivery_v1",
        bindings=evidence.bindings,
        members=members,
        extra_manifest={
            "delivery_candidate_ids": ordered_ids,
            "delivery_roles": {candidate: roles[candidate] for candidate in ordered_ids},
        },
        check_deadline=check_deadline,
    )
    verify_candidate_delivery(target, expected_bindings=evidence.bindings)
    return target


def verify_candidate_delivery(
    path: Path, *, expected_bindings: Mapping[str, str]
) -> Mapping[str, object]:
    manifest = _verify(
        Path(path),
        kind="hierarchical_tabm_candidate_delivery_v1",
        expected_bindings=expected_bindings,
    )
    ids = manifest.get("delivery_candidate_ids")
    roles = manifest.get("delivery_roles")
    if (
        type(ids) is not list
        or ids != [candidate for candidate in _CANDIDATE_ORDER if candidate in ids]
        or type(roles) is not dict
        or list(roles) != ids
        or any(role not in _ROLES for role in roles.values())
    ):
        raise HierarchicalArtifactError("candidate delivery policy differs")
    if "model/final_checkpoint.pt" not in manifest["members"] or "state/feature_state.json" not in manifest["members"]:
        raise HierarchicalArtifactError("candidate delivery model state is missing")
    return MappingProxyType(manifest)
