"""Compact review and restart bundles for the T1 physical stage."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Callable, Mapping
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from .worker import verify_worker_result


class T1ArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class T1Bundles:
    review: Path
    resume: Path


_COMPLETED_FILES = ("predictions.csv", "metrics.json", "checkpoint_meta.json")
_RESTART_FILES = (
    "checkpoint.pt",
    "best_checkpoint.pt",
    "checkpoint_meta.json",
    "progress.jsonl",
)
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def write_t1_bundles(
    output_dir: str | Path,
    *,
    jobs_root: str | Path,
    completed: tuple[str, ...],
    pending: tuple[str, ...],
    failed: tuple[str, ...],
    verifier: Callable[[str | Path], Mapping[str, object]] = verify_worker_result,
) -> T1Bundles:
    _validate_job_lists(completed, pending, failed)
    source = Path(jobs_root).resolve()
    review_members: dict[str, bytes] = {}
    resume_members: dict[str, bytes] = {}
    identities: dict[str, str] = {}
    for job_id in completed:
        root = source / job_id
        if (root / "compact_result.json").is_file() and not (
            root / "worker_result.json"
        ).exists():
            payload = verify_compact_result(root)
        else:
            payload = verifier(root)
        if payload.get("status") != "completed" or payload.get("job_id") != job_id:
            raise T1ArtifactError(f"completed worker identity differs: {job_id}")
        identity = payload.get("training_identity_sha256")
        if not _is_sha256(identity):
            raise T1ArtifactError(f"completed worker identity is invalid: {job_id}")
        identities[job_id] = identity
        compact_members: dict[str, dict[str, object]] = {}
        for name in _COMPLETED_FILES:
            data = _regular_bytes(root / name)
            archive_name = f"jobs/{job_id}/{name}"
            review_members[archive_name] = data
            resume_members[archive_name] = data
            compact_members[name] = {
                "size_bytes": len(data),
                "sha256": sha256(data).hexdigest(),
            }
        compact = _json_bytes(
            {
                "schema_version": 1,
                "job_id": job_id,
                "status": "completed",
                "training_identity_sha256": identity,
                "members": compact_members,
            }
        )
        resume_members[f"jobs/{job_id}/compact_result.json"] = compact

    for job_id in (*pending, *failed):
        root = source / job_id
        if not root.is_dir():
            continue
        if not (root / "checkpoint.pt").is_file() or not (
            root / "checkpoint_meta.json"
        ).is_file():
            continue
        for name in _RESTART_FILES:
            path = root / name
            if path.exists() or path.is_symlink():
                resume_members[f"jobs/{job_id}/{name}"] = _regular_bytes(path)

    common = {
        "schema_version": 1,
        "stage": "T1",
        "completed": list(completed),
        "pending": list(pending),
        "failed": list(failed),
        "training_identities": identities,
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    review = output / "temporal_t1_review.zip"
    resume = output / "temporal_t1_resume.zip"
    _write_bundle(review, "temporal_t1_review_v1", common, review_members)
    _write_bundle(resume, "temporal_t1_resume_v1", common, resume_members)
    return T1Bundles(review, resume)


def restore_t1_resume(bundle: str | Path, output_root: str | Path) -> Path:
    source = Path(bundle)
    manifest, members = _read_bundle(source, expected_kind="temporal_t1_resume_v1")
    del manifest
    return _restore_members(members, output_root)


def restore_t1_resume_source(source: str | Path, output_root: str | Path) -> Path:
    path = Path(source)
    if path.is_dir() and not path.is_symlink():
        manifest, members = _read_directory_bundle(
            path, expected_kind="temporal_t1_resume_v1"
        )
        del manifest
        return _restore_members(members, output_root)
    return restore_t1_resume(path, output_root)


def _restore_members(
    members: Mapping[str, bytes], output_root: str | Path
) -> Path:
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in members.items():
        target = destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
                raise T1ArtifactError(f"resume target differs: {name}")
            continue
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
    return destination


def _read_directory_bundle(
    root: Path, *, expected_kind: str
) -> tuple[Mapping[str, object], dict[str, bytes]]:
    manifest_raw = _regular_bytes(root / "manifest.json")
    try:
        manifest = json.loads(manifest_raw)
    except json.JSONDecodeError as error:
        raise T1ArtifactError("T1 directory manifest is invalid") from error
    if (
        manifest_raw != _json_bytes(manifest)
        or type(manifest) is not dict
        or manifest.get("artifact_kind") != expected_kind
        or type(manifest.get("members")) is not dict
    ):
        raise T1ArtifactError("T1 directory manifest identity differs")
    declared = set(manifest["members"])
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if declared != actual:
        raise T1ArtifactError("T1 directory members differ")
    members: dict[str, bytes] = {}
    for name, record in manifest["members"].items():
        _safe_member_name(name)
        data = _regular_bytes(root.joinpath(*PurePosixPath(name).parts))
        if (
            type(record) is not dict
            or set(record) != {"size_bytes", "sha256"}
            or record["size_bytes"] != len(data)
            or record["sha256"] != sha256(data).hexdigest()
        ):
            raise T1ArtifactError(f"T1 directory member differs: {name}")
        members[name] = data
    return manifest, members


def verify_compact_result(root: str | Path) -> Mapping[str, object]:
    directory = Path(root)
    raw = _regular_bytes(directory / "compact_result.json")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise T1ArtifactError("compact worker result is invalid") from error
    if raw != _json_bytes(payload) or type(payload) is not dict or set(payload) != {
        "schema_version",
        "job_id",
        "status",
        "training_identity_sha256",
        "members",
    }:
        raise T1ArtifactError("compact worker result schema differs")
    if (
        payload["schema_version"] != 1
        or payload["status"] != "completed"
        or type(payload["job_id"]) is not str
        or not _is_sha256(payload["training_identity_sha256"])
        or type(payload["members"]) is not dict
        or set(payload["members"]) != set(_COMPLETED_FILES)
    ):
        raise T1ArtifactError("compact worker result identity differs")
    for name, record in payload["members"].items():
        data = _regular_bytes(directory / name)
        if (
            type(record) is not dict
            or set(record) != {"size_bytes", "sha256"}
            or record["size_bytes"] != len(data)
            or record["sha256"] != sha256(data).hexdigest()
        ):
            raise T1ArtifactError(f"compact worker member differs: {name}")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != {"compact_result.json", *_COMPLETED_FILES}:
        raise T1ArtifactError("compact worker files differ")
    return payload


def _write_bundle(
    path: Path,
    kind: str,
    common: Mapping[str, object],
    members: Mapping[str, bytes],
) -> None:
    records = {
        name: {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
        for name, data in sorted(members.items())
    }
    manifest = _json_bytes({"artifact_kind": kind, **common, "members": records})
    temporary = path.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w") as archive:
        archive.writestr(_zip_info("manifest.json"), manifest)
        for name, data in sorted(members.items()):
            archive.writestr(_zip_info(name), data)
    os.replace(temporary, path)
    _read_bundle(path, expected_kind=kind)


def _read_bundle(
    path: Path, *, expected_kind: str
) -> tuple[Mapping[str, object], dict[str, bytes]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000_000:
        raise T1ArtifactError("T1 bundle file is invalid")
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > 512:
                raise T1ArtifactError("T1 bundle member count is invalid")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise T1ArtifactError("T1 bundle members differ")
            for info in infos:
                _safe_member_name(info.filename)
                mode = info.external_attr >> 16
                if mode and not stat.S_ISREG(mode):
                    raise T1ArtifactError("T1 bundle contains a non-regular member")
            manifest_raw = archive.read("manifest.json")
            manifest = json.loads(manifest_raw)
            if manifest_raw != _json_bytes(manifest):
                raise T1ArtifactError("T1 manifest is not canonical")
            if (
                type(manifest) is not dict
                or manifest.get("artifact_kind") != expected_kind
                or type(manifest.get("members")) is not dict
                or set(manifest["members"]) != set(names) - {"manifest.json"}
            ):
                raise T1ArtifactError("T1 manifest identity differs")
            members: dict[str, bytes] = {}
            for name, record in manifest["members"].items():
                data = archive.read(name)
                if (
                    type(record) is not dict
                    or set(record) != {"size_bytes", "sha256"}
                    or record["size_bytes"] != len(data)
                    or record["sha256"] != sha256(data).hexdigest()
                ):
                    raise T1ArtifactError(f"T1 bundle member differs: {name}")
                members[name] = data
            return manifest, members
    except T1ArtifactError:
        raise
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise T1ArtifactError("cannot verify T1 bundle") from error


def _safe_member_name(name: str) -> None:
    pure = PurePosixPath(name)
    if (
        type(name) is not str
        or not name
        or pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != name
    ):
        raise T1ArtifactError("T1 bundle member path is unsafe")


def _regular_bytes(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise T1ArtifactError(f"required T1 file is missing: {path.name}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise T1ArtifactError(f"T1 file is not regular: {path.name}")
    return path.read_bytes()


def _validate_job_lists(*values: tuple[str, ...]) -> None:
    if any(type(value) is not tuple for value in values):
        raise T1ArtifactError("T1 job lists must be tuples")
    names = [name for value in values for name in value]
    if (
        any(type(name) is not str or not name or "/" in name or "\\" in name for name in names)
        or len(names) != len(set(names))
    ):
        raise T1ArtifactError("T1 job lists differ")


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, _ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info
