from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Callable, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd


class BlendArtifactError(ValueError):
    """Raised when blend evidence is incomplete or corrupted."""


@dataclass(frozen=True)
class BundlePaths:
    review: Path | None
    resume: Path
    review_sha256: str | None
    resume_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class VerifiedBlendResume:
    path: Path
    manifest_sha256: str
    bindings: Mapping[str, str]
    stage_complete: bool
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None


_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_SHA_CHARS = set("0123456789abcdef")
_JOB_REQUIRED = {
    "job.json",
    "metrics.json",
    "model.cbm",
    "predictions.csv",
    "worker.log",
    "worker_result.json",
}
_ACTIVE_REQUIRED = {"job.json", "worker.log", "experiment.cbsnapshot"}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA_CHARS


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BlendArtifactError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise BlendArtifactError(f"{label} must be an object")
    return value


def _validate_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    value = dict(bindings)
    required = {
        "contract_sha256",
        "code_sha256",
        "input_manifest_sha256",
        "train_sha256",
        "history_sha256",
        "stage_c_delivery_sha256",
        "stage_c_review_sha256",
        "stage_c_state_sha256",
    }
    if set(value) != required or any(not _valid_sha(item) for item in value.values()):
        raise BlendArtifactError("artifact bindings differ")
    return value


def _validate_job(directory: Path, job_id: str) -> dict[str, object]:
    if directory.is_symlink() or not directory.is_dir():
        raise BlendArtifactError(f"job directory is unsafe: {job_id}")
    names = {path.name for path in directory.iterdir() if path.is_file()}
    if not _JOB_REQUIRED <= names:
        raise BlendArtifactError(f"completed job artifacts are missing: {job_id}")
    result = _json_file(directory / "worker_result.json", "worker result")
    metrics = _json_file(directory / "metrics.json", "job metrics")
    if (
        result.get("job_id") != job_id
        or result.get("status") != "completed"
        or metrics.get("job_id") != job_id
    ):
        raise BlendArtifactError(f"completed job identity differs: {job_id}")
    frame = pd.read_csv(directory / "predictions.csv")
    if not {"target", "probability"} <= set(frame) or frame.empty:
        raise BlendArtifactError(f"completed prediction schema differs: {job_id}")
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype="float64")
    probability = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(
        dtype="float64"
    )
    brier = float(np.mean(np.square(probability - target), dtype=np.float64))
    if (
        not math.isfinite(brier)
        or not math.isclose(float(metrics.get("brier", math.nan)), brier, abs_tol=1e-12)
        or not math.isclose(float(result.get("brier", math.nan)), brier, abs_tol=1e-12)
    ):
        raise BlendArtifactError(f"completed job Brier differs: {job_id}")
    return result


def _zip_info(name: str, size: int) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.file_size = size
    return info


def _write_archive(
    destination: Path,
    *,
    kind: str,
    bindings: Mapping[str, str],
    sources: Mapping[str, bytes | Path],
    check_deadline: Callable[[], None] | None,
    verify_temporary: Callable[[Path], None],
) -> str:
    evidence: dict[str, dict[str, object]] = {}
    for name, source in sorted(sources.items()):
        if type(source) is bytes:
            size = len(source)
            digest = sha256(source).hexdigest()
        else:
            if source.is_symlink() or not source.is_file():
                raise BlendArtifactError(f"bundle source is unsafe: {name}")
            size = source.stat().st_size
            digest = _file_sha(source)
        evidence[name] = {"size": size, "sha256": digest}
    manifest = _canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": kind,
            "campaign_id": "catboost_tabm_blend_v1",
            "review_only": True,
            "submission_package": False,
            "bindings": dict(bindings),
            "members": evidence,
        }
    )
    all_sources: dict[str, bytes | Path] = {**sources, "manifest.json": manifest}
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with ZipFile(temporary_name, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name, source in sorted(all_sources.items()):
                if check_deadline is not None:
                    check_deadline()
                size = len(source) if type(source) is bytes else source.stat().st_size
                with archive.open(_zip_info(name, size), "w") as output:
                    if type(source) is bytes:
                        output.write(source)
                    else:
                        with source.open("rb") as input_stream:
                            while chunk := input_stream.read(1024 * 1024):
                                if check_deadline is not None:
                                    check_deadline()
                                output.write(chunk)
        verify_temporary(Path(temporary_name))
        if check_deadline is not None:
            check_deadline()
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return sha256(manifest).hexdigest()


def write_bundles(
    *,
    output_dir: Path,
    contract_path: Path,
    bindings: Mapping[str, str],
    stage_state_path: Path,
    campaign_log_path: Path,
    job_directories: Mapping[str, Path],
    decision_path: Path | None,
    check_deadline: Callable[[], None] | None = None,
) -> BundlePaths:
    bindings = _validate_bindings(bindings)
    state = _json_file(Path(stage_state_path), "stage state")
    if (
        state.get("schema_version") != 1
        or state.get("campaign_id") != "catboost_tabm_blend_v1"
        or state.get("bindings") != bindings
        or type(state.get("completed_job_ids")) is not list
    ):
        raise BlendArtifactError("stage state differs")
    completed = tuple(state["completed_job_ids"])
    active = state.get("active_job_id")
    if active is not None and type(active) is not str:
        raise BlendArtifactError("active job identity differs")
    expected_directories = set(completed) | ({active} if active is not None else set())
    if (
        len(completed) != len(set(completed))
        or active in completed
        or expected_directories != set(job_directories)
    ):
        raise BlendArtifactError("completed job set differs")
    results = [_validate_job(Path(job_directories[job_id]), job_id) for job_id in completed]
    output_dir = Path(output_dir)
    resume_sources: dict[str, bytes | Path] = {
        "contract/contract.json": Path(contract_path),
        "state/stage_state.json": Path(stage_state_path),
    }
    for job_id in completed:
        directory = Path(job_directories[job_id])
        for name in sorted(_JOB_REQUIRED):
            resume_sources[f"jobs/{job_id}/{name}"] = directory / name
        snapshot = directory / "experiment.cbsnapshot"
        if snapshot.is_file():
            resume_sources[f"jobs/{job_id}/experiment.cbsnapshot"] = snapshot
    if active is not None:
        directory = Path(job_directories[active])
        if directory.is_symlink() or not directory.is_dir():
            raise BlendArtifactError("active job directory is unsafe")
        if not all((directory / name).is_file() for name in _ACTIVE_REQUIRED):
            raise BlendArtifactError("active job snapshot artifacts are missing")
        job_payload = _json_file(directory / "job.json", "active job")
        if job_payload.get("job_id") != active:
            raise BlendArtifactError("active job identity differs")
        for name in sorted(_ACTIVE_REQUIRED):
            resume_sources[f"jobs/{active}/{name}"] = directory / name
    resume_path = output_dir / "catboost_tabm_blend_resume.zip"
    manifest_sha = _write_archive(
        resume_path,
        kind="resume",
        bindings=bindings,
        sources=resume_sources,
        check_deadline=check_deadline,
        verify_temporary=lambda path: verify_resume_bundle(
            path, expected_bindings=bindings
        ),
    )

    review_path: Path | None = None
    review_sha: str | None = None
    if state.get("stage_complete") is True:
        if decision_path is None or len(completed) != 2:
            raise BlendArtifactError("complete stage requires two jobs and a decision")
        review_sources: dict[str, bytes | Path] = {
            "contract/contract.json": Path(contract_path),
            "decision/blend_decision.json": Path(decision_path),
            "logs/campaign.log": Path(campaign_log_path),
            "metrics/fold_results.json": _canonical_json(results),
            "state/stage_state.json": Path(stage_state_path),
        }
        for job_id in completed:
            review_sources[f"predictions/{job_id}.csv"] = (
                Path(job_directories[job_id]) / "predictions.csv"
            )
        review_path = output_dir / "catboost_tabm_blend_review.zip"
        _write_archive(
            review_path,
            kind="review",
            bindings=bindings,
            sources=review_sources,
            check_deadline=check_deadline,
            verify_temporary=lambda path: verify_review_bundle(
                path, expected_bindings=bindings
            ),
        )
        review_sha = _file_sha(review_path)
    return BundlePaths(
        review=review_path,
        resume=resume_path,
        review_sha256=review_sha,
        resume_sha256=_file_sha(resume_path),
        manifest_sha256=manifest_sha,
    )


def _safe_archive(archive: ZipFile) -> tuple[dict[str, ZipInfo], dict[str, object], bytes]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or "manifest.json" not in names:
        raise BlendArtifactError("bundle member set is invalid")
    if sum(info.file_size for info in infos) > 8 * 1024 * 1024 * 1024:
        raise BlendArtifactError("bundle exceeds total size limit")
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            not info.filename
            or "\\" in info.filename
            or path.is_absolute()
            or ".." in path.parts
            or info.is_dir()
            or stat.S_IFMT(mode) == stat.S_IFLNK
            or info.file_size > 2 * 1024 * 1024 * 1024
            or (
                info.file_size >= 64 * 1024
                and info.file_size / max(1, info.compress_size) > 200.0
            )
        ):
            raise BlendArtifactError(f"unsafe bundle member: {info.filename}")
    manifest_bytes = archive.read("manifest.json")
    try:
        manifest = json.loads(manifest_bytes)
    except Exception as error:
        raise BlendArtifactError("bundle manifest is unreadable") from error
    if type(manifest) is not dict:
        raise BlendArtifactError("bundle manifest must be an object")
    return {info.filename: info for info in infos}, manifest, manifest_bytes


def _verify_bundle(
    path: Path, *, kind: str, expected_bindings: Mapping[str, str]
) -> tuple[dict[str, object], bytes, dict[str, ZipInfo]]:
    expected_bindings = _validate_bindings(expected_bindings)
    try:
        with ZipFile(path) as archive:
            infos, manifest, manifest_bytes = _safe_archive(archive)
            if set(manifest) != {
                "schema_version",
                "artifact_kind",
                "campaign_id",
                "review_only",
                "submission_package",
                "bindings",
                "members",
            }:
                raise BlendArtifactError("bundle manifest keys differ")
            if (
                manifest["schema_version"] != 1
                or manifest["artifact_kind"] != kind
                or manifest["campaign_id"] != "catboost_tabm_blend_v1"
                or manifest["review_only"] is not True
                or manifest["submission_package"] is not False
                or manifest["bindings"] != expected_bindings
                or type(manifest["members"]) is not dict
                or set(manifest["members"]) != set(infos) - {"manifest.json"}
            ):
                raise BlendArtifactError("bundle identity differs")
            for name, evidence in manifest["members"].items():
                if (
                    type(evidence) is not dict
                    or set(evidence) != {"size", "sha256"}
                    or type(evidence["size"]) is not int
                    or evidence["size"] != infos[name].file_size
                    or not _valid_sha(evidence["sha256"])
                ):
                    raise BlendArtifactError(f"bundle evidence differs: {name}")
                digest = sha256()
                with archive.open(name) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != evidence["sha256"]:
                    raise BlendArtifactError(f"bundle member differs: {name}")
            state = json.loads(archive.read("state/stage_state.json"))
    except BlendArtifactError:
        raise
    except (BadZipFile, OSError, KeyError, json.JSONDecodeError) as error:
        raise BlendArtifactError(f"cannot verify bundle: {error}") from error
    if (
        type(state) is not dict
        or state.get("schema_version") != 1
        or state.get("campaign_id") != "catboost_tabm_blend_v1"
        or state.get("bindings") != expected_bindings
        or type(state.get("completed_job_ids")) is not list
    ):
        raise BlendArtifactError("bundle stage state differs")
    return state, manifest_bytes, infos


def verify_resume_bundle(
    path: Path, *, expected_bindings: Mapping[str, str]
) -> VerifiedBlendResume:
    path = Path(path)
    state, manifest_bytes, infos = _verify_bundle(
        path, kind="resume", expected_bindings=expected_bindings
    )
    completed = tuple(state["completed_job_ids"])
    expected = {"manifest.json", "contract/contract.json", "state/stage_state.json"}
    for job_id in completed:
        expected.update(f"jobs/{job_id}/{name}" for name in _JOB_REQUIRED)
        snapshot = f"jobs/{job_id}/experiment.cbsnapshot"
        if snapshot in infos:
            expected.add(snapshot)
    active = state.get("active_job_id")
    if active is not None:
        expected.update(f"jobs/{active}/{name}" for name in _ACTIVE_REQUIRED)
    if set(infos) != expected:
        raise BlendArtifactError("resume member set differs from state")
    return VerifiedBlendResume(
        path=path,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        bindings=dict(expected_bindings),
        stage_complete=state.get("stage_complete") is True,
        completed_job_ids=completed,
        active_job_id=state.get("active_job_id"),
    )


def verify_review_bundle(
    path: Path, *, expected_bindings: Mapping[str, str]
) -> None:
    state, _, infos = _verify_bundle(
        Path(path), kind="review", expected_bindings=expected_bindings
    )
    completed = tuple(state["completed_job_ids"])
    expected = {
        "manifest.json",
        "contract/contract.json",
        "decision/blend_decision.json",
        "logs/campaign.log",
        "metrics/fold_results.json",
        "state/stage_state.json",
        *(f"predictions/{job_id}.csv" for job_id in completed),
    }
    if state.get("stage_complete") is not True or len(completed) != 2 or set(infos) != expected:
        raise BlendArtifactError("review member set differs from complete state")
