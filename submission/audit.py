"""Read-only, fail-closed final audit for DACON package inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import stat

from competition_rules.contract import (
    RulesContractError,
    load_policy,
    load_policy_review,
)
from competition_rules.evidence_gate import (
    AuditIdentity,
    RulesEvidenceError,
    validate_full_audit,
)
from .adapters import resolve_adapter_factory
from .contract import PackageRequest


class SubmissionAuditError(ValueError):
    """Raised before packaging when one required proof is absent or stale."""


_GATES = {
    "temporal_validation",
    "performance",
    "provenance",
    "row_independence",
    "evaluator_runtime",
    "pretrained_license",
    "current_rules",
}
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class AuditSnapshot:
    request: PackageRequest
    policy: dict[str, object]
    acceptance: dict[str, object]
    benchmark: dict[str, object]
    identity: AuditIdentity
    requirements_bytes: bytes
    model_files: tuple[tuple[str, bytes], ...]
    model_sha256: str
    artifact_metadata: dict[str, object]


def _safe_existing(path: Path, root: Path, *, directory: bool = False) -> Path:
    try:
        relative = path.expanduser().absolute().relative_to(root)
    except ValueError as error:
        raise SubmissionAuditError(f"path is outside project root: {path}") from error
    current = root
    try:
        for part in relative.parts:
            current = current / part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise SubmissionAuditError(f"symlink is not allowed: {current}")
    except SubmissionAuditError:
        raise
    except OSError as error:
        raise SubmissionAuditError(f"cannot inspect path: {path}") from error
    valid = current.is_dir() if directory else current.is_file()
    if not valid:
        raise SubmissionAuditError(f"required path has wrong type: {path}")
    return current


def _safe_output(path: Path, root: Path) -> Path:
    candidate = path.expanduser().absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise SubmissionAuditError(f"output is outside project root: {path}") from error
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
            raise SubmissionAuditError(f"symlink output parent is not allowed: {current}")
    if candidate.exists():
        raise SubmissionAuditError(f"output already exists: {candidate}")
    return candidate


def _load(path: Path) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in items:
            if key in output:
                raise SubmissionAuditError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    def constant(value: str) -> object:
        raise SubmissionAuditError(f"non-finite JSON: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except SubmissionAuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SubmissionAuditError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise SubmissionAuditError(f"JSON must be an object: {path}")
    return value


def _identity(value: object) -> AuditIdentity:
    if not isinstance(value, dict) or set(value) != set(AuditIdentity.__dataclass_fields__):
        raise SubmissionAuditError("identity fields are invalid")
    try:
        return AuditIdentity(**value)
    except (TypeError, RulesEvidenceError) as error:
        raise SubmissionAuditError("identity is invalid") from error


def _model_snapshot(model_dir: Path) -> tuple[tuple[tuple[str, bytes], ...], str]:
    paths = sorted(model_dir.rglob("*"))
    files: list[tuple[str, bytes]] = []
    for path in paths:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise SubmissionAuditError(f"symlink model member is not allowed: {path}")
        if path.is_file():
            relative = path.relative_to(model_dir).as_posix()
            if relative.startswith("/") or ".." in Path(relative).parts:
                raise SubmissionAuditError("unsafe model member")
            files.append((relative, path.read_bytes()))
    if not files:
        raise SubmissionAuditError("model directory is empty")
    if len(files) == 1:
        digest = sha256(files[0][1]).hexdigest()
    else:
        combined = sha256()
        for name, data in files:
            combined.update(name.encode("utf-8") + b"\0" + sha256(data).digest())
        digest = combined.hexdigest()
    return tuple(files), digest


def _require_number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubmissionAuditError(f"benchmark {key} is invalid")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise SubmissionAuditError(f"benchmark {key} is invalid")
    return number


def _artifact_metadata(
    *,
    request: PackageRequest,
    root: Path,
    model_dir: Path,
    model_files: tuple[tuple[str, bytes], ...],
    model_sha256: str,
    identity: AuditIdentity,
) -> dict[str, object]:
    from .tabm_candidate import CANDIDATE_ID

    if request.adapter_id != CANDIDATE_ID:
        return {
            "candidate_id": identity.candidate_id,
            "model_sha256": model_sha256,
            "adapter_sha256": identity.adapter_sha256,
            "runtime_sha256": identity.runtime_sha256,
        }
    manifest_path = _safe_existing(model_dir.parent / "candidate_manifest.json", root)
    manifest = _load(manifest_path)
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "candidate_id",
        "delivery_sha256",
        "review_bundle_sha256",
        "model_sha256",
        "members",
    }
    members = {name: sha256(data).hexdigest() for name, data in model_files}
    if (
        set(manifest) != expected_keys
        or manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "tabm_submission_validation_candidate"
        or manifest["candidate_id"] != identity.candidate_id
        or manifest["candidate_id"] != CANDIDATE_ID
        or manifest["model_sha256"] != model_sha256
        or manifest["members"] != members
    ):
        raise SubmissionAuditError("TabM candidate manifest differs from audited model")
    return {
        "candidate_id": manifest["candidate_id"],
        "delivery_sha256": manifest["delivery_sha256"],
        "review_bundle_sha256": manifest["review_bundle_sha256"],
        "model_sha256": manifest["model_sha256"],
        "members": manifest["members"],
    }


def audit_package_request(request: PackageRequest) -> AuditSnapshot:
    """Check all evidence before any archive member is written."""

    if not isinstance(request, PackageRequest):
        raise SubmissionAuditError("package request has invalid type")
    root = request.project_root.expanduser().resolve(strict=True)
    policy_path = _safe_existing(request.policy_path, root)
    review_path = _safe_existing(request.policy_review_path, root)
    acceptance_path = _safe_existing(request.acceptance_path, root)
    manifest_path = _safe_existing(request.full_audit_manifest_path, root)
    benchmark_path = _safe_existing(request.runtime_benchmark_path, root)
    model_dir = _safe_existing(request.model_dir, root, directory=True)
    requirements_path = _safe_existing(request.requirements_path, root)
    _safe_output(request.archive_path, root)
    _safe_output(request.receipt_path, root)
    if request.archive_path.absolute() == request.receipt_path.absolute():
        raise SubmissionAuditError("archive and receipt paths must differ")
    try:
        policy = load_policy(policy_path, project_root=root)
        load_policy_review(review_path, policy=policy, package_time=request.package_time)
    except RulesContractError as error:
        raise SubmissionAuditError(str(error)) from error

    acceptance = _load(acceptance_path)
    if set(acceptance) != {
        "schema_version", "status", "candidate_id", "policy_version",
        "adapter_id", "identity", "gates",
    }:
        raise SubmissionAuditError("acceptance fields are invalid")
    if acceptance["schema_version"] != 1 or acceptance["status"] != "passed":
        raise SubmissionAuditError("acceptance status is not passed")
    if acceptance["policy_version"] != policy["policy_version"]:
        raise SubmissionAuditError("acceptance policy is stale")
    if acceptance["adapter_id"] != request.adapter_id:
        raise SubmissionAuditError("acceptance adapter differs")
    gates = acceptance["gates"]
    if not isinstance(gates, dict) or set(gates) != _GATES or any(
        value is not True for value in gates.values()
    ):
        raise SubmissionAuditError("acceptance gate is missing or false")
    identity = _identity(acceptance["identity"])
    if identity.policy_version != policy["policy_version"] or identity.candidate_id != acceptance["candidate_id"]:
        raise SubmissionAuditError("acceptance identity differs")
    try:
        resolve_adapter_factory(request.adapter_id)
    except Exception as error:
        raise SubmissionAuditError("adapter is not registered") from error
    try:
        validate_full_audit(manifest_path, expected_identity=identity)
    except RulesEvidenceError as error:
        raise SubmissionAuditError("full audit does not match acceptance") from error

    model_files, model_digest = _model_snapshot(model_dir)
    if model_digest != identity.model_sha256:
        raise SubmissionAuditError("live model SHA does not match acceptance")
    artifact_metadata = _artifact_metadata(
        request=request,
        root=root,
        model_dir=model_dir,
        model_files=model_files,
        model_sha256=model_digest,
        identity=identity,
    )
    benchmark = _load(benchmark_path)
    expected_benchmark_keys = {
        "schema_version", "status", "candidate_id", "adapter_id", "identity",
        "install_seconds", "inference_seconds", "peak_ram_bytes", "peak_vram_bytes",
        "extracted_bytes", "package_bytes",
    }
    if set(benchmark) != expected_benchmark_keys or benchmark["schema_version"] != 1 or benchmark["status"] != "passed":
        raise SubmissionAuditError("runtime benchmark is invalid")
    if benchmark["candidate_id"] != identity.candidate_id or benchmark["adapter_id"] != request.adapter_id:
        raise SubmissionAuditError("runtime benchmark candidate differs")
    if benchmark["identity"] != asdict(identity):
        raise SubmissionAuditError("runtime benchmark identity differs")
    install = _require_number(benchmark, "install_seconds")
    inference = _require_number(benchmark, "inference_seconds")
    package_bytes = _require_number(benchmark, "package_bytes")
    extracted = _require_number(benchmark, "extracted_bytes")
    ram = _require_number(benchmark, "peak_ram_bytes")
    vram = _require_number(benchmark, "peak_vram_bytes")
    if install > policy["limits"]["install_seconds"]:
        raise SubmissionAuditError("install time exceeds limit")
    if inference > policy["runtime_safety_seconds"]:
        raise SubmissionAuditError("runtime safety threshold exceeded")
    if package_bytes > policy["limits"]["package_bytes"]:
        raise SubmissionAuditError("package size exceeds limit")
    if extracted > policy["limits"]["extracted_bytes"]:
        raise SubmissionAuditError("extracted size exceeds limit")
    if ram > 28 * 1024**3 or vram > 22.4 * 1024**3:
        raise SubmissionAuditError("memory or VRAM exceeds evaluator capacity")
    return AuditSnapshot(
        request=request,
        policy=policy,
        acceptance=acceptance,
        benchmark=benchmark,
        identity=identity,
        requirements_bytes=requirements_path.read_bytes(),
        model_files=model_files,
        model_sha256=model_digest,
        artifact_metadata=artifact_metadata,
    )
