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
_EXPERIMENT_KEYS = {
    "schema_version",
    "contract_id",
    "rules_version",
    "scope",
    "candidate_source",
    "candidate_count",
    "candidate_ids_sha256",
    "allowed_derivations",
    "config_sha256",
    "inference_source_paths",
    "data_sources",
    "fit_scope",
    "evaluation_scope",
    "time_scope",
    "external_api",
    "pretrained_models",
    "retrieval_corpora",
}
_LOWER_HEX = frozenset("0123456789abcdef")


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
    except RulesContractError:
        raise
    except (OSError, ValueError) as error:
        raise RulesContractError(f"cannot inspect rules file: {candidate}") from error
    if not stat.S_ISREG(current.lstat().st_mode):
        raise RulesContractError(f"rules path is not a regular file: {current}")
    return current


def _safe_project_path(path: str | Path, project_root: str | Path) -> Path:
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
            if stat.S_ISLNK(current.lstat().st_mode):
                raise RulesContractError(f"symlink is not allowed: {current}")
    except RulesContractError:
        raise
    except (OSError, ValueError) as error:
        raise RulesContractError(f"cannot inspect project path: {candidate}") from error
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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise RulesContractError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise RulesContractError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_candidate_digest(candidate_ids: list[str]) -> str:
    encoded = json.dumps(
        candidate_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _derived_candidate_ids(
    source: str,
    *,
    root: Path,
    config_paths: list[Path],
    explicit: object,
) -> list[str]:
    if source == "explicit":
        if not isinstance(explicit, list) or not explicit:
            raise RulesContractError("candidate_ids must be a non-empty list")
        if any(not isinstance(item, str) or not item for item in explicit):
            raise RulesContractError("candidate_ids contain an invalid ID")
        if len(set(explicit)) != len(explicit):
            raise RulesContractError("candidate_ids must be unique")
        return list(explicit)
    if source == "component_only":
        return []
    if source == "independent_dl_non_tabicl":
        from experiments.independent_dl.contracts import load_campaign

        config = next(
            (path for path in config_paths if path.name == "campaign_v1.json"), None
        )
        if config is None:
            raise RulesContractError("independent DL campaign config is missing")
        return [
            candidate.candidate_id
            for candidate in load_campaign(config).candidates
            if candidate.family != "tabicl_v2"
        ]
    if source == "preprocessing_wave_a":
        from experiments.independent_dl.preprocessing_contracts import (
            load_preprocessing_campaign,
        )

        config = next(
            (
                path
                for path in config_paths
                if path.name == "preprocessing_ablation_v1.json"
            ),
            None,
        )
        if config is None:
            raise RulesContractError("preprocessing campaign config is missing")
        return [
            job.job_id for job in load_preprocessing_campaign(config).wave_a_jobs
        ]
    if source == "budgeted_jobs":
        from experiments.preprocessing_campaign.budgeted_contracts import (
            load_budgeted_campaign,
        )

        config = next(
            (
                path
                for path in config_paths
                if path.name == "budgeted_campaign_v1.json"
            ),
            None,
        )
        if config is None:
            raise RulesContractError("budgeted campaign config is missing")
        campaign = load_budgeted_campaign(config)
        return [
            job.job_id
            for stage_id in sorted(campaign.stages)
            for job in campaign.stages[stage_id]
        ]
    if source == "preprocessing_and_budgeted_jobs":
        preprocessing = _derived_candidate_ids(
            "preprocessing_wave_a",
            root=root,
            config_paths=config_paths,
            explicit=None,
        )
        budgeted = _derived_candidate_ids(
            "budgeted_jobs",
            root=root,
            config_paths=config_paths,
            explicit=None,
        )
        return [*preprocessing, *budgeted]
    raise RulesContractError("candidate_source is invalid")


def validate_experiment_contract(
    path: str | Path,
    *,
    project_root: str | Path,
    candidate_id: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """Validate one current-policy, config-bound experiment contract."""

    root = Path(project_root).expanduser().resolve(strict=True)
    payload = _load_json(path, project_root=root)
    if not isinstance(payload, dict):
        raise RulesContractError("experiment contract must be an object")
    allowed_keys = _EXPERIMENT_KEYS | {"candidate_ids"}
    if not _EXPERIMENT_KEYS.issubset(payload) or not set(payload).issubset(allowed_keys):
        raise RulesContractError("experiment contract has invalid keys")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise RulesContractError("schema_version must be 1")
    if not isinstance(payload["contract_id"], str) or not payload["contract_id"]:
        raise RulesContractError("contract_id is invalid")
    if payload["rules_version"] != "dacon-236743-2026-08-13":
        raise RulesContractError("rules_version is not current")
    if payload["scope"] not in {"config_bound_campaign", "component"}:
        raise RulesContractError("scope is invalid")
    if payload["data_sources"] != ["official_train", "official_trackman"]:
        raise RulesContractError("data_sources are invalid")
    if payload["fit_scope"] != "training_rows_only":
        raise RulesContractError("fit_scope must be training_rows_only")
    if payload["evaluation_scope"] != "current_row_only":
        raise RulesContractError("evaluation_scope must be current_row_only")
    if payload["time_scope"] != "pre_pitch_only":
        raise RulesContractError("time_scope must be pre_pitch_only")
    if payload["external_api"] is not False:
        raise RulesContractError("external_api must be false")

    config_bindings = payload["config_sha256"]
    if not isinstance(config_bindings, dict) or not config_bindings:
        raise RulesContractError("config_sha256 must be a non-empty object")
    config_paths: list[Path] = []
    for relative, expected in config_bindings.items():
        if not isinstance(relative, str) or not relative:
            raise RulesContractError("config_sha256 path is invalid")
        expected_sha = _lower_sha256(expected, "config SHA-256")
        config = _safe_regular_file(relative, root)
        if _sha256_file(config) != expected_sha:
            raise RulesContractError(f"config SHA-256 mismatch: {relative}")
        config_paths.append(config)
    if config_path is not None:
        requested = _safe_regular_file(config_path, root)
        if requested not in config_paths:
            raise RulesContractError("config_path is not covered by the contract")

    source_paths = payload["inference_source_paths"]
    if not isinstance(source_paths, list) or not source_paths:
        raise RulesContractError("inference_source_paths must be non-empty")
    for relative in source_paths:
        if not isinstance(relative, str) or not relative:
            raise RulesContractError("inference_source_paths contain an invalid path")
        source_path = _safe_project_path(relative, root)
        if not (source_path.is_file() or source_path.is_dir()):
            raise RulesContractError("inference source path is not a file or directory")

    allowed_derivations = payload["allowed_derivations"]
    if not isinstance(allowed_derivations, list) or any(
        item not in {"confirmation_seed", "boundary_expansion", "wave_promotion"}
        for item in allowed_derivations
    ) or len(set(allowed_derivations)) != len(allowed_derivations):
        raise RulesContractError("allowed_derivations are invalid")
    for collection_name in ("pretrained_models", "retrieval_corpora"):
        if not isinstance(payload[collection_name], list):
            raise RulesContractError(f"{collection_name} must be a list")
    for model in payload["pretrained_models"]:
        if not isinstance(model, dict) or set(model) != {"source", "version", "license", "sha256"}:
            raise RulesContractError("pretrained_models entry is invalid")
        if any(not isinstance(model[key], str) or not model[key] for key in ("source", "version", "license")):
            raise RulesContractError("pretrained model provenance is incomplete")
        _lower_sha256(model["sha256"], "pretrained model SHA-256")

    candidate_ids = _derived_candidate_ids(
        str(payload["candidate_source"]),
        root=root,
        config_paths=config_paths,
        explicit=payload.get("candidate_ids"),
    )
    if type(payload["candidate_count"]) is not int or payload["candidate_count"] != len(candidate_ids):
        raise RulesContractError("candidate_count does not match derived candidates")
    if _lower_sha256(payload["candidate_ids_sha256"], "candidate_ids_sha256") != _canonical_candidate_digest(candidate_ids):
        raise RulesContractError("candidate_ids_sha256 does not match derived candidates")
    if candidate_id is not None and candidate_id not in candidate_ids:
        raise RulesContractError(f"candidate is not covered: {candidate_id}")
    report = dict(payload)
    report["covered_candidate_ids"] = candidate_ids
    return report


def assert_experiment_runnable(
    *,
    project_root: str | Path,
    contract_path: str | Path,
    config_path: str | Path,
    candidate_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Validate current policy, experiment contract, candidates, and source."""

    root = Path(project_root).expanduser().resolve(strict=True)
    policy = load_policy(root / "competition_rules/policy.json", project_root=root)
    contract = validate_experiment_contract(
        contract_path,
        project_root=root,
        config_path=config_path,
    )
    if contract["rules_version"] != policy["policy_version"]:
        raise RulesContractError("experiment contract does not match current policy")
    covered = contract["covered_candidate_ids"]
    if candidate_ids is not None:
        if not isinstance(candidate_ids, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in candidate_ids
        ):
            raise RulesContractError("candidate_ids are invalid")
        missing = sorted(set(candidate_ids) - set(covered))
        if missing:
            raise RulesContractError(f"candidate is not covered: {missing[0]}")
    from .code_gate import inspect_inference_source

    source_gate = inspect_inference_source(
        contract["inference_source_paths"], project_root=root
    )
    return {
        **contract,
        "status": "passed",
        "policy_sha256": policy_digest(policy),
        "source_gate": source_gate,
    }
