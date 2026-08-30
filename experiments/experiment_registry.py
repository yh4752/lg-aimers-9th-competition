from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Mapping


class RegistryError(ValueError):
    pass


STATUSES = frozenset({"failed", "rejected", "accepted", "public_scored", "diagnostic"})
EVIDENCE_GRADES = frozenset({"A", "B", "C"})
FAILURE_CLASSES = frozenset({
    "none",
    "performance",
    "instability",
    "diversity",
    "deployment_alignment",
    "data_eligibility",
    "runtime",
    "rule_quarantine",
    "evidence_missing",
})
REPEAT_POLICIES = frozenset({"closed", "redefine", "retain", "blocked", "reference_only"})
RULE_STATUSES = frozenset({"passed", "failed", "not_run", "quarantined", "unknown"})
REQUIRED_FIELDS = frozenset({
    "experiment_id",
    "family",
    "variant",
    "completed_at",
    "status",
    "status_reason",
    "evidence_grade",
    "comparison_group",
    "folds",
    "validation_rows",
    "baseline_id",
    "baseline_brier",
    "candidate_brier",
    "weighted_gain",
    "worst_fold_gain",
    "latest_fold_gain",
    "max_segment_regression",
    "residual_correlation",
    "seed_stability",
    "public_score",
    "submission_sha256",
    "artifact_paths",
    "artifact_sha256",
    "evidence_paths",
    "rule_audit_status",
    "row_independence_status",
    "failure_class",
    "lesson",
    "repeat_policy",
})
GAP_FIELDS = frozenset({"experiment_id", "reason", "required_evidence"})
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _number_or_none(
    value: object,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    if value is None:
        return
    if type(value) not in (int, float):
        raise RegistryError(f"{name} must be numeric or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RegistryError(f"{name} must be finite")
    if lower is not None and numeric < lower:
        raise RegistryError(f"{name} is below {lower}")
    if upper is not None and numeric > upper:
        raise RegistryError(f"{name} is above {upper}")


def _string(value: object, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not str or not value.strip():
        raise RegistryError(f"{name} must be a non-empty string")


def _string_list(value: object, name: str) -> None:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise RegistryError(f"{name} must be a list of non-empty strings")


def validate_registry(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != 1:
        raise RegistryError("schema_version must be 1")
    if set(payload) != {"schema_version", "experiments", "evidence_gaps"}:
        raise RegistryError("registry member set differs")
    experiments = payload.get("experiments")
    gaps = payload.get("evidence_gaps")
    if type(experiments) is not list or type(gaps) is not list:
        raise RegistryError("experiments and evidence_gaps must be lists")

    seen: set[str] = set()
    for raw in experiments:
        if type(raw) is not dict or set(raw) != REQUIRED_FIELDS:
            raise RegistryError("experiment member set differs")
        experiment_id = raw["experiment_id"]
        _string(experiment_id, "experiment_id")
        assert isinstance(experiment_id, str)
        if experiment_id in seen:
            raise RegistryError(f"duplicate experiment_id: {experiment_id}")
        seen.add(experiment_id)

        for name in (
            "family",
            "variant",
            "completed_at",
            "status_reason",
            "comparison_group",
            "seed_stability",
            "lesson",
        ):
            _string(raw[name], name)
        _string(raw["baseline_id"], "baseline_id", nullable=True)
        _string(raw["submission_sha256"], "submission_sha256", nullable=True)
        _string_list(raw["folds"], "folds")
        _string_list(raw["artifact_paths"], "artifact_paths")
        _string_list(raw["artifact_sha256"], "artifact_sha256")
        _string_list(raw["evidence_paths"], "evidence_paths")

        if raw["status"] not in STATUSES:
            raise RegistryError("status is invalid")
        if raw["evidence_grade"] not in EVIDENCE_GRADES:
            raise RegistryError("evidence_grade is invalid")
        if raw["failure_class"] not in FAILURE_CLASSES:
            raise RegistryError("failure_class is invalid")
        if raw["repeat_policy"] not in REPEAT_POLICIES:
            raise RegistryError("repeat_policy is invalid")
        if raw["rule_audit_status"] not in RULE_STATUSES:
            raise RegistryError("rule_audit_status is invalid")
        if raw["row_independence_status"] not in RULE_STATUSES:
            raise RegistryError("row_independence_status is invalid")

        rows = raw["validation_rows"]
        if rows is not None and (type(rows) is not int or rows < 0):
            raise RegistryError("validation_rows must be a non-negative integer or null")
        _number_or_none(raw["baseline_brier"], "baseline_brier", 0.0, 1.0)
        _number_or_none(raw["candidate_brier"], "candidate_brier", 0.0, 1.0)
        for name in (
            "weighted_gain",
            "worst_fold_gain",
            "latest_fold_gain",
            "max_segment_regression",
        ):
            _number_or_none(raw[name], name)
        _number_or_none(raw["residual_correlation"], "residual_correlation", -1.0, 1.0)
        _number_or_none(raw["public_score"], "public_score")

        declared_hashes = raw["artifact_sha256"]
        assert isinstance(declared_hashes, list)
        if any(_HASH.fullmatch(value) is None for value in declared_hashes):
            raise RegistryError("artifact_sha256 contains an invalid digest")
        submission_hash = raw["submission_sha256"]
        if submission_hash is not None and _HASH.fullmatch(str(submission_hash)) is None:
            raise RegistryError("submission_sha256 is invalid")
        if len(raw["artifact_paths"]) != len(declared_hashes):
            raise RegistryError("artifact path and sha256 counts differ")

        if raw["status"] == "public_scored" and raw["public_score"] is None:
            raise RegistryError("public_score is required for public_scored")
        if (
            raw["status"] == "public_scored"
            and raw["evidence_grade"] in {"A", "B"}
            and submission_hash is None
        ):
            raise RegistryError("submission_sha256 is required for verified public_scored")
        if raw["status"] == "accepted" and raw["rule_audit_status"] == "failed":
            raise RegistryError("accepted record cannot have failed rule_audit_status")

    gap_ids: set[str] = set()
    for gap in gaps:
        if type(gap) is not dict or set(gap) != GAP_FIELDS:
            raise RegistryError("evidence gap member set differs")
        for name in GAP_FIELDS:
            _string(gap[name], f"evidence gap {name}")
        gap_id = str(gap["experiment_id"])
        if gap_id in gap_ids or gap_id in seen:
            raise RegistryError(f"duplicate evidence gap: {gap_id}")
        gap_ids.add(gap_id)


def load_registry(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot load registry: {error}") from error
    if type(payload) is not dict:
        raise RegistryError("registry root must be an object")
    validate_registry(payload)
    return payload


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
