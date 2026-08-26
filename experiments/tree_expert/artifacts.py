from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


class TreeArtifactError(ValueError):
    """Raised when E1 evidence cannot be safely written or verified."""


_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_BINDING_KEYS = {
    "contract_sha256",
    "code_sha256",
    "input_manifest_sha256",
    "train_sha256",
    "history_sha256",
    "stage_c_delivery_sha256",
    "stage_c_review_sha256",
    "baseline_predictions_sha256",
}
_REVIEW_FILES = {
    "job.json",
    "worker.log",
    "worker_result.json",
    "metrics.json",
    "predictions.csv",
    "failure_label_audit.json",
}
_RESUME_FILES = _REVIEW_FILES | {"model.cbm", "experiment.cbsnapshot"}
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_RATIO = 250.0


@dataclass(frozen=True)
class E1BundlePaths:
    review: Path
    resume: Path
    handoff: Path
    review_sha256: str
    resume_sha256: str
    handoff_sha256: str


@dataclass(frozen=True)
class VerifiedE1Resume:
    path: Path
    manifest_sha256: str
    bindings: Mapping[str, str]
    completed: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[str, ...]
    submission_package: bool


@dataclass(frozen=True)
class VerifiedE1Handoff:
    path: Path
    manifest_sha256: str
    review_only: bool
    submission_package: bool


def file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TreeArtifactError(f"artifact source is not a regular file: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _bindings(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _BINDING_KEYS:
        raise TreeArtifactError("artifact binding keys differ")
    output = dict(value)
    if any(not _valid_sha(item) for item in output.values()):
        raise TreeArtifactError("artifact binding SHA-256 is invalid")
    return output


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TreeArtifactError(f"artifact JSON is invalid: {error}") from error


def _source(path: Path) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise TreeArtifactError(f"artifact source is not a regular file: {value}")
    return value


def _evidence(value: bytes | Path) -> dict[str, object]:
    if isinstance(value, bytes):
        return {"size": len(value), "sha256": sha256(value).hexdigest()}
    path = _source(value)
    return {"size": path.stat().st_size, "sha256": file_sha256(path)}


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_member(archive: ZipFile, name: str, value: bytes | Path) -> None:
    info = _zip_info(name)
    if isinstance(value, bytes):
        archive.writestr(info, value)
        return
    with _source(value).open("rb") as source, archive.open(info, "w") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _safe_entries(archive: ZipFile) -> list[str]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise TreeArtifactError("artifact has duplicate members")
    total = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        ratio = info.file_size / max(1, info.compress_size)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or info.is_dir()
            or stat.S_ISLNK(mode)
            or info.file_size > _MAX_MEMBER_BYTES
            or (info.file_size >= 64 * 1024 and ratio > _MAX_RATIO)
        ):
            raise TreeArtifactError(f"unsafe artifact member: {info.filename}")
        total += info.file_size
    if total > _MAX_TOTAL_BYTES:
        raise TreeArtifactError("artifact uncompressed size is too large")
    return names


def _load_state(path: Path, bindings: Mapping[str, str]) -> dict[str, object]:
    try:
        state = json.loads(_source(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise TreeArtifactError(f"cannot read E1 state: {error}") from error
    expected = {
        "schema_version",
        "campaign_id",
        "status",
        "completed",
        "skipped",
        "failed",
        "active",
        "bindings",
    }
    if type(state) is not dict or set(state) != expected:
        raise TreeArtifactError("E1 state keys differ")
    if state["schema_version"] != 1 or state["campaign_id"] != "tree_expert_e1_v1":
        raise TreeArtifactError("E1 state identity differs")
    if state["bindings"] != dict(bindings):
        raise TreeArtifactError("E1 state bindings differ")
    for name in ("completed", "skipped", "failed"):
        values = state[name]
        if type(values) is not list or any(type(item) is not str for item in values):
            raise TreeArtifactError(f"E1 state {name} differs")
    if type(state["active"]) is not dict:
        raise TreeArtifactError("E1 state active jobs differ")
    return state


def _manifest(
    *,
    kind: str,
    bindings: Mapping[str, str],
    state: Mapping[str, object],
    members: Mapping[str, bytes | Path],
) -> bytes:
    return _canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": kind,
            "review_only": True,
            "submission_package": False,
            "campaign_id": "tree_expert_e1_v1",
            "status": state["status"],
            "completed": state["completed"],
            "skipped": state["skipped"],
            "failed": state["failed"],
            "bindings": dict(bindings),
            "members": {
                name: _evidence(value) for name, value in sorted(members.items())
            },
        }
    )


def _write_bundle(
    destination: Path,
    *,
    kind: str,
    bindings: Mapping[str, str],
    state: Mapping[str, object],
    members: Mapping[str, bytes | Path],
) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    payloads = dict(members)
    payloads["manifest.json"] = _manifest(
        kind=kind, bindings=bindings, state=state, members=members
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, value in sorted(payloads.items()):
                _write_member(archive, name, value)
        _verify_bundle(temporary, kind=kind, expected_bindings=bindings)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _verify_bundle(
    path: Path,
    *,
    kind: str,
    expected_bindings: Mapping[str, str],
) -> dict[str, object]:
    source = Path(path)
    try:
        with ZipFile(source, "r") as archive:
            names = _safe_entries(archive)
            if "manifest.json" not in names:
                raise TreeArtifactError("artifact manifest is absent")
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            expected_keys = {
                "schema_version",
                "artifact_kind",
                "review_only",
                "submission_package",
                "campaign_id",
                "status",
                "completed",
                "skipped",
                "failed",
                "bindings",
                "members",
            }
            if type(manifest) is not dict or set(manifest) != expected_keys:
                raise TreeArtifactError("artifact manifest keys differ")
            if (
                manifest["schema_version"] != 1
                or manifest["artifact_kind"] != kind
                or manifest["review_only"] is not True
                or manifest["submission_package"] is not False
                or manifest["campaign_id"] != "tree_expert_e1_v1"
                or manifest["bindings"] != _bindings(expected_bindings)
            ):
                raise TreeArtifactError("artifact identity or bindings differ")
            members = manifest["members"]
            if type(members) is not dict or set(members) != set(names) - {"manifest.json"}:
                raise TreeArtifactError("artifact member manifest differs")
            for name, evidence in members.items():
                if type(evidence) is not dict or set(evidence) != {"size", "sha256"}:
                    raise TreeArtifactError(f"artifact member evidence is invalid: {name}")
                info = archive.getinfo(name)
                digest = sha256()
                with archive.open(name, "r") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                if evidence["size"] != info.file_size or evidence["sha256"] != digest.hexdigest():
                    raise TreeArtifactError(f"artifact member differs: {name}")
            manifest["__manifest_sha256__"] = sha256(manifest_bytes).hexdigest()
            return manifest
    except TreeArtifactError:
        raise
    except Exception as error:
        raise TreeArtifactError(f"cannot verify E1 artifact: {error}") from error


def _job_members(
    job_directories: Sequence[Path],
    *,
    allowed: set[str],
) -> dict[str, Path]:
    members: dict[str, Path] = {}
    for directory in sorted((Path(path) for path in job_directories), key=lambda path: path.name):
        if directory.is_symlink() or not directory.is_dir() or "/" in directory.name:
            raise TreeArtifactError(f"job directory is invalid: {directory}")
        for name in sorted(allowed):
            source = directory / name
            if source.exists():
                members[f"jobs/{directory.name}/{name}"] = _source(source)
    return members


def _write_handoff(
    output: Path,
    *,
    review: Path,
    resume: Path,
    log: Path,
) -> Path:
    payloads: dict[str, Path] = {
        "tree_expert_e1_review.zip": review,
        "tree_expert_e1_resume.zip": resume,
        "tree_expert_e1.log": log,
    }
    manifest = _canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "tree_expert_e1_handoff_v1",
            "review_only": True,
            "submission_package": False,
            "members": {
                name: _evidence(path) for name, path in sorted(payloads.items())
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, value in sorted({**payloads, "handoff_manifest.json": manifest}.items()):
                _write_member(archive, name, value)
        verify_e1_handoff(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_e1_bundles(
    *,
    output_dir: Path,
    contract_path: Path,
    bindings: Mapping[str, str],
    state_path: Path,
    log_path: Path,
    job_directories: Sequence[Path],
    decision_path: Path | None,
    audit_paths: Sequence[Path],
) -> E1BundlePaths:
    sealed_bindings = _bindings(bindings)
    state = _load_state(state_path, sealed_bindings)
    common: dict[str, bytes | Path] = {
        "contract/e1_contract.json": _source(contract_path),
        "campaign/stage_state.json": _source(state_path),
        "campaign/tree_expert_e1.log": _source(log_path),
    }
    if decision_path is not None:
        common["campaign/decision.json"] = _source(decision_path)
    for audit in audit_paths:
        source = _source(audit)
        name = f"audits/{source.name}"
        if name in common:
            raise TreeArtifactError(f"duplicate audit name: {source.name}")
        common[name] = source

    root = Path(output_dir)
    review = _write_bundle(
        root / "tree_expert_e1_review.zip",
        kind="tree_expert_e1_review_v1",
        bindings=sealed_bindings,
        state=state,
        members={**common, **_job_members(job_directories, allowed=_REVIEW_FILES)},
    )
    resume = _write_bundle(
        root / "tree_expert_e1_resume.zip",
        kind="tree_expert_e1_resume_v1",
        bindings=sealed_bindings,
        state=state,
        members={**common, **_job_members(job_directories, allowed=_RESUME_FILES)},
    )
    handoff = _write_handoff(
        root / "tree_expert_e1_handoff.zip",
        review=review,
        resume=resume,
        log=_source(log_path),
    )
    return E1BundlePaths(
        review=review,
        resume=resume,
        handoff=handoff,
        review_sha256=file_sha256(review),
        resume_sha256=file_sha256(resume),
        handoff_sha256=file_sha256(handoff),
    )


def verify_e1_resume(
    path: Path,
    expected_bindings: Mapping[str, str],
) -> VerifiedE1Resume:
    manifest = _verify_bundle(
        Path(path),
        kind="tree_expert_e1_resume_v1",
        expected_bindings=expected_bindings,
    )
    for name in ("completed", "skipped", "failed"):
        values = manifest[name]
        if type(values) is not list or any(type(item) is not str for item in values):
            raise TreeArtifactError(f"resume {name} values differ")
    return VerifiedE1Resume(
        path=Path(path),
        manifest_sha256=str(manifest["__manifest_sha256__"]),
        bindings=_bindings(expected_bindings),
        completed=tuple(manifest["completed"]),
        skipped=tuple(manifest["skipped"]),
        failed=tuple(manifest["failed"]),
        submission_package=False,
    )


def verify_e1_handoff(path: Path) -> VerifiedE1Handoff:
    source = Path(path)
    expected_names = {
        "handoff_manifest.json",
        "tree_expert_e1_review.zip",
        "tree_expert_e1_resume.zip",
        "tree_expert_e1.log",
    }
    try:
        with ZipFile(source, "r") as archive:
            names = _safe_entries(archive)
            if set(names) != expected_names:
                raise TreeArtifactError("handoff member names differ")
            manifest_bytes = archive.read("handoff_manifest.json")
            manifest = json.loads(manifest_bytes)
            if (
                type(manifest) is not dict
                or set(manifest)
                != {
                    "schema_version",
                    "artifact_kind",
                    "review_only",
                    "submission_package",
                    "members",
                }
                or manifest["schema_version"] != 1
                or manifest["artifact_kind"] != "tree_expert_e1_handoff_v1"
                or manifest["review_only"] is not True
                or manifest["submission_package"] is not False
            ):
                raise TreeArtifactError("handoff manifest identity differs")
            members = manifest["members"]
            if type(members) is not dict or set(members) != expected_names - {
                "handoff_manifest.json"
            }:
                raise TreeArtifactError("handoff member manifest differs")
            for name, evidence in members.items():
                if type(evidence) is not dict or set(evidence) != {"size", "sha256"}:
                    raise TreeArtifactError(f"handoff member evidence is invalid: {name}")
                payload = archive.read(name)
                if (
                    evidence["size"] != len(payload)
                    or evidence["sha256"] != sha256(payload).hexdigest()
                ):
                    raise TreeArtifactError(f"handoff member differs: {name}")
    except TreeArtifactError:
        raise
    except Exception as error:
        raise TreeArtifactError(f"cannot verify E1 handoff: {error}") from error
    return VerifiedE1Handoff(
        path=source,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        review_only=True,
        submission_package=False,
    )


def extract_e1_resume(
    path: Path,
    destination: Path,
    expected_bindings: Mapping[str, str],
) -> Path:
    verify_e1_resume(path, expected_bindings)
    root = Path(destination)
    if root.exists():
        if root.is_symlink() or any(root.iterdir()):
            raise TreeArtifactError("resume destination is not empty")
    else:
        root.mkdir(parents=True)
    with ZipFile(path, "r") as archive:
        for info in archive.infolist():
            output = root.joinpath(*PurePosixPath(info.filename).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, output.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    return root
