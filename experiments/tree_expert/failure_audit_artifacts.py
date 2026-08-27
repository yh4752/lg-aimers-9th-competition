from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .failure_audit import CutoffAudit, FailureAuditResult, TypeDecision


class FailureAuditArtifactError(ValueError):
    pass


_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MEMBERS = {
    "audit_summary.json",
    "class_counts.csv",
    "exclusion_counts.csv",
    "cutoffs/A1.json",
    "cutoffs/A2.json",
    "cutoffs/A3.json",
    "cutoffs/A4.json",
    "audit.log",
}


@dataclass(frozen=True)
class FailureAuditBindings:
    contract_sha256: str
    code_sha256: str
    official_train_sha256: str
    official_history_sha256: str


@dataclass(frozen=True)
class VerifiedFailureAuditReview:
    path: Path
    archive_sha256: str
    manifest_sha256: str
    member_sha256: Mapping[str, str]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise FailureAuditArtifactError(f"evidence is not serializable: {error}") from error


def _bindings_payload(bindings: FailureAuditBindings) -> dict[str, str]:
    return {
        "contract_sha256": bindings.contract_sha256,
        "code_sha256": bindings.code_sha256,
        "official_train_sha256": bindings.official_train_sha256,
        "official_history_sha256": bindings.official_history_sha256,
    }


def _cutoff_payload(item: CutoffAudit) -> dict[str, object]:
    return {
        "audit_id": item.audit_id,
        "cutoff_year": item.cutoff_year,
        "status": item.status,
        "failed_gates": list(item.failed_gates),
        "maximum_source_season": item.maximum_source_season,
        "row_count": item.row_count,
        "linked_count": item.linked_count,
        "labeled_count": item.labeled_count,
        "coverage": item.coverage,
        "binary_delta_fraction": item.binary_delta_fraction,
        "success_agreement": item.success_agreement,
        "middle_reverse_overlap": item.middle_reverse_overlap,
        "class_counts": dict(item.class_counts),
        "exclusion_counts": dict(item.exclusion_counts),
    }


def _decision_payload(item: TypeDecision) -> dict[str, object]:
    return {
        "failure_type": item.failure_type,
        "status": item.status,
        "reason": item.reason,
        "positive_rows": dict(item.positive_rows),
        "negative_rows": dict(item.negative_rows),
    }


def _members(result: FailureAuditResult) -> dict[str, bytes]:
    summary = {
        "schema_version": 1,
        "campaign_id": "failure_expert_label_audit_v1",
        "review_only": True,
        "submission_package": False,
        "cutoffs": {
            name: _cutoff_payload(item) for name, item in result.cutoffs.items()
        },
        "type_decisions": {
            name: _decision_payload(item)
            for name, item in result.type_decisions.items()
        },
    }
    members = {
        "audit_summary.json": _canonical_json(summary),
        "class_counts.csv": result.class_counts.sort_values(
            list(result.class_counts.columns), kind="stable"
        ).to_csv(index=False, lineterminator="\n").encode("utf-8"),
        "exclusion_counts.csv": result.exclusion_counts.sort_values(
            list(result.exclusion_counts.columns), kind="stable"
        ).to_csv(index=False, lineterminator="\n").encode("utf-8"),
    }
    for name, item in result.cutoffs.items():
        members[f"cutoffs/{name}.json"] = _canonical_json(_cutoff_payload(item))
    lines = [
        f"FAIL_AUDIT_CUTOFF_RESULT cutoff={name} status={item.status}"
        for name, item in result.cutoffs.items()
    ]
    lines.extend(
        f"FAIL_AUDIT_DECISION type={name} status={item.status} reason={item.reason}"
        for name, item in result.type_decisions.items()
    )
    members["audit.log"] = ("\n".join(lines) + "\n").encode("utf-8")
    if set(members) != _MEMBERS:
        raise FailureAuditArtifactError("review member set differs")
    return members


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write(path: Path, members: Mapping[str, bytes]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in sorted(members.items()):
                archive.writestr(_zip_info(name), payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def create_failure_audit_review(
    result: FailureAuditResult,
    path: Path,
    bindings: FailureAuditBindings,
) -> Path:
    members = _members(result)
    evidence = {
        name: {"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
        for name, payload in sorted(members.items())
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "failure_expert_label_audit_review_v1",
        "review_only": True,
        "submission_package": False,
        "bindings": _bindings_payload(bindings),
        "members": evidence,
    }
    _write(Path(path), {**members, "manifest.json": _canonical_json(manifest)})
    verify_failure_audit_review(Path(path), bindings)
    return Path(path)


def _safe_infos(archive: ZipFile) -> dict[str, ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or set(names) != {*_MEMBERS, "manifest.json"}:
        raise FailureAuditArtifactError("review member names differ")
    for info in infos:
        pure = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or info.flag_bits & 0x1
            or info.is_dir()
            or stat.S_ISLNK(mode)
            or info.file_size > _MAX_MEMBER_BYTES
        ):
            raise FailureAuditArtifactError(f"unsafe review member: {info.filename}")
    return {info.filename: info for info in infos}


def verify_failure_audit_review(
    path: Path,
    bindings: FailureAuditBindings,
) -> VerifiedFailureAuditReview:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FailureAuditArtifactError("review archive is not a regular file")
    try:
        with ZipFile(source, "r") as archive:
            infos = _safe_infos(archive)
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            if (
                type(manifest) is not dict
                or manifest.get("schema_version") != 1
                or manifest.get("artifact_kind") != "failure_expert_label_audit_review_v1"
                or manifest.get("review_only") is not True
                or manifest.get("submission_package") is not False
                or manifest.get("bindings") != _bindings_payload(bindings)
                or set(manifest) != {
                    "schema_version",
                    "artifact_kind",
                    "review_only",
                    "submission_package",
                    "bindings",
                    "members",
                }
            ):
                raise FailureAuditArtifactError("review manifest differs")
            evidence = manifest.get("members")
            if type(evidence) is not dict or set(evidence) != _MEMBERS:
                raise FailureAuditArtifactError("review member evidence differs")
            hashes: dict[str, str] = {}
            for name in sorted(_MEMBERS):
                payload = archive.read(infos[name])
                actual = {"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
                if evidence.get(name) != actual:
                    raise FailureAuditArtifactError("member evidence differs")
                hashes[name] = actual["sha256"]
    except FailureAuditArtifactError:
        raise
    except Exception as error:
        raise FailureAuditArtifactError(f"review archive cannot be verified: {error}") from error
    return VerifiedFailureAuditReview(
        path=source,
        archive_sha256=file_sha256(source),
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        member_sha256=MappingProxyType(hashes),
    )
