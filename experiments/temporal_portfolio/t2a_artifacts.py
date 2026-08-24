"""Compact review and restart bundles for the dynamic T2-A stage."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .t1_artifacts import (
    _COMPLETED_FILES,
    _RESTART_FILES,
    _read_bundle,
    _read_directory_bundle,
    _regular_bytes,
    _restore_members,
    _validate_job_lists,
    _write_bundle,
    verify_compact_result,
)
from .worker import verify_worker_result


class T2AArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class T2ABundles:
    review: Path
    resume: Path


def write_t2a_bundles(
    output_dir: str | Path,
    *,
    jobs_root: str | Path,
    completed: tuple[str, ...],
    pending: tuple[str, ...],
    failed: tuple[str, ...],
    skipped: Mapping[str, str],
    evidence: Mapping[str, object],
    t1_decision_sha256: str,
) -> T2ABundles:
    _validate_job_lists(completed, pending, failed)
    if not isinstance(skipped, Mapping) or not isinstance(evidence, Mapping):
        raise T2AArtifactError("T2-A state is invalid")
    source = Path(jobs_root).resolve()
    review_members: dict[str, bytes] = {}
    resume_members: dict[str, bytes] = {}
    identities: dict[str, str] = {}
    for job_id in completed:
        root = source / job_id
        payload = (
            verify_compact_result(root)
            if (root / "compact_result.json").is_file()
            and not (root / "worker_result.json").exists()
            else verify_worker_result(root)
        )
        if payload.get("status") != "completed" or payload.get("job_id") != job_id:
            raise T2AArtifactError(f"completed worker identity differs: {job_id}")
        identity = payload.get("training_identity_sha256")
        if (
            type(identity) is not str
            or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
        ):
            raise T2AArtifactError(f"completed worker identity is invalid: {job_id}")
        identities[job_id] = identity
        compact_members = {}
        for name in _COMPLETED_FILES:
            data = _regular_bytes(root / name)
            archive_name = f"jobs/{job_id}/{name}"
            review_members[archive_name] = data
            resume_members[archive_name] = data
            compact_members[name] = {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
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
        if not (root / "checkpoint.pt").is_file() or not (root / "checkpoint_meta.json").is_file():
            continue
        for name in _RESTART_FILES:
            path = root / name
            if path.exists() or path.is_symlink():
                resume_members[f"jobs/{job_id}/{name}"] = _regular_bytes(path)
    state = _json_bytes({"skipped": dict(sorted(skipped.items())), "evidence": evidence})
    review_members["t2a_evidence.json"] = state
    resume_members["t2a_evidence.json"] = state
    common = {
        "schema_version": 1,
        "stage": "T2A",
        "t1_decision_sha256": t1_decision_sha256,
        "completed": list(completed),
        "pending": list(pending),
        "failed": list(failed),
        "skipped": dict(sorted(skipped.items())),
        "training_identities": identities,
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    review = output / "temporal_t2a_review.zip"
    resume = output / "temporal_t2a_resume.zip"
    _write_bundle(review, "temporal_t2a_review_v1", common, review_members)
    _write_bundle(resume, "temporal_t2a_resume_v1", common, resume_members)
    return T2ABundles(review, resume)


def restore_t2a_resume_source(source: str | Path, output_root: str | Path) -> Path:
    path = Path(source)
    if path.is_dir() and not path.is_symlink():
        _, members = _read_directory_bundle(path, expected_kind="temporal_t2a_resume_v1")
    else:
        _, members = _read_bundle(path, expected_kind="temporal_t2a_resume_v1")
    members.pop("t2a_evidence.json", None)
    return _restore_members(members, output_root)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
