from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .hc_contracts import load_hc_contract
from .hc_state import CampaignState, HCBindings, state_from_payload, state_payload


class HCArtifactError(ValueError):
    """Raised when HC evidence is unsafe, incomplete, or changes identity."""


@dataclass(frozen=True)
class VerifiedResume:
    path: Path
    manifest_sha256: str
    status: str
    stage: str


@dataclass(frozen=True)
class VerifiedHandoff:
    path: Path
    manifest_sha256: str
    status: str
    delivery: bool


_TIME = (2026, 1, 1, 0, 0, 0)
_MAX_MEMBER = 2 * 1024 * 1024 * 1024
_MAX_TOTAL = 8 * 1024 * 1024 * 1024


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _evidence(payload: bytes) -> dict[str, object]:
    return {"size": len(payload), "sha256": sha256(payload).hexdigest()}


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, _TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_zip(path: Path, members: Mapping[str, bytes]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in sorted(members.items()):
                archive.writestr(_info(name), payload)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _safe_payloads(path: Path) -> dict[str, bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise HCArtifactError("artifact source differs")
    try:
        with ZipFile(source) as archive:
            payloads: dict[str, bytes] = {}
            total = 0
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    info.filename in payloads
                    or info.flag_bits & 0x1
                    or info.is_dir()
                    or member.is_absolute()
                    or any(part in {"", ".", ".."} for part in member.parts)
                    or "\\" in info.filename
                    or stat.S_ISLNK(mode)
                    or info.file_size > _MAX_MEMBER
                ):
                    raise HCArtifactError("unsafe or duplicate artifact member")
                total += info.file_size
                if total > _MAX_TOTAL:
                    raise HCArtifactError("artifact expanded size exceeds limit")
                payloads[info.filename] = archive.read(info)
            return payloads
    except BadZipFile as error:
        raise HCArtifactError("artifact is not a valid ZIP") from error


def _verify_members(
    payloads: Mapping[str, bytes], manifest: Mapping[str, object], manifest_name: str
) -> None:
    declared = manifest.get("members")
    if type(declared) is not dict or set(declared) != set(payloads) - {manifest_name}:
        raise HCArtifactError("artifact member set differs")
    for name, evidence in declared.items():
        payload = payloads[name]
        if (
            type(evidence) is not dict
            or set(evidence) != {"size", "sha256"}
            or evidence["size"] != len(payload)
            or evidence["sha256"] != sha256(payload).hexdigest()
        ):
            raise HCArtifactError(f"artifact member SHA-256 differs: {name}")


def _json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HCArtifactError(f"{label} differs") from error
    if type(value) is not dict:
        raise HCArtifactError(f"{label} differs")
    return value


def _binding_payload(bindings: HCBindings) -> dict[str, str]:
    return dict(bindings.__dict__)


def create_resume_bundle(
    campaign_root: Path,
    output: Path,
    bindings: HCBindings,
    state: CampaignState,
) -> Path:
    if state.bindings != bindings:
        raise HCArtifactError("resume bindings differ")
    root = Path(campaign_root)
    members: dict[str, bytes] = {"state/state.json": _canonical(state_payload(state))}
    for job_id in state.completed_jobs:
        job_root = root / "jobs" / job_id
        if job_root.is_symlink() or not job_root.is_dir():
            raise HCArtifactError(f"completed job is absent: {job_id}")
        files = [path for path in sorted(job_root.rglob("*")) if path.is_file()]
        if not files:
            raise HCArtifactError(f"completed job is empty: {job_id}")
        for path in files:
            if path.is_symlink():
                raise HCArtifactError(f"completed job has a symlink: {job_id}")
            members[f"jobs/{job_id}/{path.relative_to(job_root).as_posix()}"] = path.read_bytes()
    manifest = _canonical(
        {
            "schema_version": 1,
            "artifact_kind": "tree_hierarchical_resume_v1",
            "campaign_id": state.campaign_id,
            "stage": state.stage,
            "status": state.status,
            "submission_package": False,
            "bindings": _binding_payload(bindings),
            "members": {name: _evidence(payload) for name, payload in sorted(members.items())},
        }
    )
    result = _write_zip(Path(output), {**members, "manifest.json": manifest})
    verify_resume_bundle(result, bindings)
    return result


def verify_resume_bundle(path: Path, bindings: HCBindings) -> VerifiedResume:
    payloads = _safe_payloads(Path(path))
    if "manifest.json" not in payloads or "state/state.json" not in payloads:
        raise HCArtifactError("resume members differ")
    manifest_bytes = payloads["manifest.json"]
    manifest = _json(manifest_bytes, "resume manifest")
    if (
        manifest.get("artifact_kind") != "tree_hierarchical_resume_v1"
        or manifest.get("campaign_id") != "tree_hierarchical_residual_v1"
        or manifest.get("submission_package") is not False
        or manifest.get("bindings") != _binding_payload(bindings)
    ):
        raise HCArtifactError("resume bindings differ")
    _verify_members(payloads, manifest, "manifest.json")
    state = state_from_payload(_json(payloads["state/state.json"], "resume state"))
    if state.bindings != bindings or state.stage != manifest.get("stage") or state.status != manifest.get("status"):
        raise HCArtifactError("resume state differs")
    expected_prefixes = {f"jobs/{job}/" for job in state.completed_jobs}
    actual_prefixes = {
        f"jobs/{name.split('/')[1]}/"
        for name in payloads
        if name.startswith("jobs/") and len(name.split("/")) >= 3
    }
    if actual_prefixes != expected_prefixes:
        raise HCArtifactError("resume completed jobs differ")
    return VerifiedResume(Path(path), sha256(manifest_bytes).hexdigest(), state.status, state.stage)


def create_review_bundle(
    *,
    sources: Mapping[str, Path],
    state: CampaignState,
    output: Path,
) -> Path:
    members = {name: Path(path).read_bytes() for name, path in sources.items()}
    members["state/state.json"] = _canonical(state_payload(state))
    manifest = _canonical(
        {
            "schema_version": 1,
            "artifact_kind": "tree_hierarchical_review_v1",
            "campaign_id": state.campaign_id,
            "stage": state.stage,
            "status": state.status,
            "submission_package": False,
            "bindings": _binding_payload(state.bindings),
            "members": {name: _evidence(payload) for name, payload in sorted(members.items())},
        }
    )
    return _write_zip(Path(output), {**members, "manifest.json": manifest})


def create_model_delivery(
    *,
    state: CampaignState,
    sources: Mapping[str, Path],
    inference_audit: Mapping[str, object],
    output: Path,
) -> Path:
    winner = state.decisions.get("winner")
    if winner is None or winner.get("status") != "accepted" or winner.get("candidate") not in {"C1", "C2"}:
        raise HCArtifactError("model delivery requires accepted evidence")
    contract = load_hc_contract()
    if (
        inference_audit.get("status") != "passed"
        or float(inference_audit.get("max_probability_error", float("inf")))
        > contract.runtime.probability_tolerance
    ):
        raise HCArtifactError("model delivery inference audit differs")
    if not sources:
        raise HCArtifactError("model delivery sources are empty")
    members = {name: Path(path).read_bytes() for name, path in sources.items()}
    members["evidence/acceptance.json"] = _canonical(dict(winner))
    members["evidence/inference_audit.json"] = _canonical(dict(inference_audit))
    if sum(len(payload) for payload in members.values()) > contract.runtime.model_state_max_bytes:
        raise HCArtifactError("model delivery exceeds state size")
    if any(name in {"script.py", "submission.csv"} for name in members):
        raise HCArtifactError("model delivery contains submission content")
    manifest = _canonical(
        {
            "schema_version": 1,
            "artifact_kind": "tree_hierarchical_model_delivery_v1",
            "campaign_id": state.campaign_id,
            "candidate": winner["candidate"],
            "review_only": False,
            "submission_package": False,
            "bindings": _binding_payload(state.bindings),
            "members": {name: _evidence(payload) for name, payload in sorted(members.items())},
        }
    )
    return _write_zip(Path(output), {**members, "manifest.json": manifest})


def create_handoff(
    *,
    review: Path,
    resume: Path,
    log: Path,
    output: Path,
    status: str,
    acceptance: Path | None = None,
    delivery: Path | None = None,
) -> Path:
    members = {
        "tree_hierarchical_review.zip": Path(review).read_bytes(),
        "tree_hierarchical_resume.zip": Path(resume).read_bytes(),
        "tree_hierarchical.log": Path(log).read_bytes(),
    }
    if acceptance is not None:
        members["acceptance.json"] = Path(acceptance).read_bytes()
    if delivery is not None:
        members["tree_hierarchical_model_delivery.zip"] = Path(delivery).read_bytes()
    manifest = _canonical(
        {
            "schema_version": 1,
            "artifact_kind": "tree_hierarchical_handoff_v1",
            "campaign_id": "tree_hierarchical_residual_v1",
            "status": status,
            "delivery": delivery is not None,
            "review_only": delivery is None,
            "submission_package": False,
            "members": {name: _evidence(payload) for name, payload in sorted(members.items())},
        }
    )
    result = _write_zip(Path(output), {**members, "handoff_manifest.json": manifest})
    verify_handoff(result)
    return result


def verify_handoff(path: Path) -> VerifiedHandoff:
    payloads = _safe_payloads(Path(path))
    if "handoff_manifest.json" not in payloads:
        raise HCArtifactError("handoff manifest is absent")
    manifest_bytes = payloads["handoff_manifest.json"]
    manifest = _json(manifest_bytes, "handoff manifest")
    delivery = manifest.get("delivery")
    if (
        manifest.get("artifact_kind") != "tree_hierarchical_handoff_v1"
        or manifest.get("campaign_id") != "tree_hierarchical_residual_v1"
        or manifest.get("submission_package") is not False
        or type(delivery) is not bool
        or manifest.get("review_only") is not (not delivery)
    ):
        raise HCArtifactError("handoff identity differs")
    expected = {
        "handoff_manifest.json",
        "tree_hierarchical_review.zip",
        "tree_hierarchical_resume.zip",
        "tree_hierarchical.log",
    }
    if "acceptance.json" in payloads:
        expected.add("acceptance.json")
    if delivery:
        expected.add("tree_hierarchical_model_delivery.zip")
    if set(payloads) != expected:
        raise HCArtifactError("handoff members differ")
    _verify_members(payloads, manifest, "handoff_manifest.json")
    return VerifiedHandoff(
        Path(path), sha256(manifest_bytes).hexdigest(), str(manifest.get("status")), delivery
    )
