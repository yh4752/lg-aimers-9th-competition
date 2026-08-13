"""Strict, dependency-free contracts for current DACON rules evidence."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import stat
from typing import Mapping
from zoneinfo import ZoneInfo


class RulesContractError(ValueError):
    """Raised when rule evidence is missing, unsafe, or ambiguous."""


_POLICY_KEYS = {
    "schema_version",
    "policy_version",
    "competition_id",
    "official_sources",
    "allowed_data_sources",
    "prohibited_external_apis",
    "decimal_places",
    "runtime_safety_seconds",
    "limits",
    "manual_rules",
    "environment",
    "archive_members",
    "output_path",
}
_LIMITS = {
    "install_seconds": 600,
    "inference_seconds": 600,
    "package_bytes": 10_000_000_000,
    "extracted_bytes": 32_000_000_000,
    "evaluation_rows": 245789,
}
_MANUAL_RULES = {
    "team_max_members": 5,
    "duplicate_registration_allowed": False,
    "daily_submission_limit": 5,
}
_ENVIRONMENT = {
    "python": "3.11.15",
    "os": "Ubuntu 22.04",
    "gpu": "NVIDIA L4 22.4 GiB",
    "cpu_count": 6,
    "ram_gib": 28,
    "offline_after_install": True,
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise RulesContractError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> object:
    raise RulesContractError(f"non-finite JSON number: {value}")


def _validate_finite(value: object, path: str = "root") -> None:
    if isinstance(value, float) and not isfinite(value):
        raise RulesContractError(f"non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def _safe_regular_file(path: str | Path, project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve(strict=True)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise RulesContractError(f"path is outside project root: {candidate}") from error
    current = root
    try:
        for part in relative.parts:
            current = current / part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RulesContractError(f"symlink is not allowed: {current}")
    except (OSError, ValueError) as error:
        raise RulesContractError(f"cannot inspect rules file: {candidate}") from error
    if not stat.S_ISREG(current.lstat().st_mode):
        raise RulesContractError(f"rules path is not a regular file: {current}")
    return current


def _load_json(path: str | Path, *, project_root: str | Path) -> object:
    safe = _safe_regular_file(path, project_root)
    try:
        payload = json.loads(
            safe.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RulesContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RulesContractError(f"invalid JSON file: {safe}") from error
    _validate_finite(payload)
    return payload


def _exact_mapping(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RulesContractError(f"{label} has invalid keys")
    return value


def load_policy(path: str | Path, *, project_root: str | Path) -> dict[str, object]:
    """Load schema-1 policy from one project-contained regular file."""

    policy = _exact_mapping(_load_json(path, project_root=project_root), _POLICY_KEYS, "policy")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise RulesContractError("policy schema_version must be 1")
    if policy["policy_version"] != "dacon-236743-2026-08-13":
        raise RulesContractError("unknown policy_version")
    if policy["competition_id"] != "236743":
        raise RulesContractError("competition_id must be 236743")
    sources = policy["official_sources"]
    if not isinstance(sources, list) or len(sources) != 6:
        raise RulesContractError("official_sources must contain six entries")
    for source in sources:
        item = _exact_mapping(source, {"title", "url"}, "official source")
        if not isinstance(item["title"], str) or not item["title"]:
            raise RulesContractError("official source title is invalid")
        if not isinstance(item["url"], str) or not item["url"].startswith("https://dacon.io/"):
            raise RulesContractError("official source URL is invalid")
    if policy["allowed_data_sources"] != ["official_train", "official_trackman"]:
        raise RulesContractError("allowed_data_sources are invalid")
    if policy["prohibited_external_apis"] is not True:
        raise RulesContractError("external APIs must be prohibited")
    if type(policy["decimal_places"]) is not int or policy["decimal_places"] != 8:
        raise RulesContractError("decimal_places must be 8")
    if type(policy["runtime_safety_seconds"]) is not int or policy["runtime_safety_seconds"] != 480:
        raise RulesContractError("runtime_safety_seconds must be 480")
    if policy["limits"] != _LIMITS:
        raise RulesContractError("limits do not match the current policy")
    if policy["manual_rules"] != _MANUAL_RULES:
        raise RulesContractError("manual_rules do not match the current policy")
    if policy["environment"] != _ENVIRONMENT:
        raise RulesContractError("environment does not match the current policy")
    if policy["archive_members"] != ["script.py", "requirements.txt", "model/"]:
        raise RulesContractError("archive_members are invalid")
    if policy["output_path"] != "output/submission.csv":
        raise RulesContractError("output_path is invalid")
    return policy


def policy_digest(policy: Mapping[str, object]) -> str:
    """Hash UTF-8 canonical JSON with sorted keys and compact separators."""

    try:
        encoded = json.dumps(
            policy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RulesContractError("policy cannot be canonically encoded") from error
    return sha256(encoded).hexdigest()


def load_policy_review(
    path: str | Path,
    *,
    policy: Mapping[str, object],
    package_time: datetime,
) -> dict[str, object]:
    """Require exact policy identity and the same Asia/Seoul calendar date."""

    safe = Path(path).expanduser()
    root = safe.parent if safe.is_absolute() else Path.cwd()
    review = _exact_mapping(
        _load_json(path, project_root=root),
        {"schema_version", "policy_version", "policy_sha256", "reviewed_at", "sources", "verdict"},
        "policy review",
    )
    if type(review["schema_version"]) is not int or review["schema_version"] != 1:
        raise RulesContractError("review schema_version must be 1")
    if review["policy_version"] != policy.get("policy_version"):
        raise RulesContractError("review policy_version does not match")
    if review["policy_sha256"] != policy_digest(policy):
        raise RulesContractError("review policy_sha256 does not match")
    if review["sources"] != policy.get("official_sources"):
        raise RulesContractError("review sources do not match")
    if review["verdict"] != "unchanged":
        raise RulesContractError("review verdict must be unchanged")
    if package_time.tzinfo is None or package_time.utcoffset() is None:
        raise RulesContractError("package_time must be timezone-aware")
    try:
        reviewed_at = datetime.fromisoformat(str(review["reviewed_at"]))
    except ValueError as error:
        raise RulesContractError("reviewed_at is invalid") from error
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise RulesContractError("reviewed_at must be timezone-aware")
    kst = ZoneInfo("Asia/Seoul")
    if reviewed_at.astimezone(kst).date() != package_time.astimezone(kst).date():
        raise RulesContractError("review must use the same KST date as packaging")
    return review
