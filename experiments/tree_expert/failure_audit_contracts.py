from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re


class FailureAuditContractError(ValueError):
    pass


DEFAULT_CONTRACT = Path(__file__).with_name("failure_audit_contract.json")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = {
    "schema_version",
    "campaign_id",
    "review_only",
    "submission_package",
    "official_train_sha256",
    "official_history_sha256",
    "cutoffs",
    "types",
    "delta_tolerance",
    "quality_gates",
    "sample_gates",
    "runtime",
}
_CUTOFFS = (("A1", 2021), ("A2", 2022), ("A3", 2023), ("A4", 2024))
_TYPES = ("middle", "reverse", "other_failure")
_HASHES = {
    "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
}
_QUALITY_GATES = {
    "minimum_coverage": 0.98,
    "minimum_binary_delta_fraction": 0.999,
    "minimum_success_agreement": 0.999,
    "maximum_middle_reverse_overlap": 0.001,
}
_SAMPLE_GATES = {"minimum_positive_rows": 5000, "minimum_negative_rows": 5000}
_RUNTIME = {"wall_seconds": 1800, "maximum_rss_bytes": 12884901888}


@dataclass(frozen=True)
class FailureAuditContract:
    campaign_id: str
    official_train_sha256: str
    official_history_sha256: str
    cutoffs: tuple[tuple[str, int], ...]
    types: tuple[str, ...]
    delta_tolerance: float
    minimum_coverage: float
    minimum_binary_delta_fraction: float
    minimum_success_agreement: float
    maximum_middle_reverse_overlap: float
    minimum_positive_rows: int
    minimum_negative_rows: int
    wall_seconds: int
    maximum_rss_bytes: int


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise FailureAuditContractError(f"{label} keys differ")
    return dict(value)


def _finite(value: object, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise FailureAuditContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FailureAuditContractError(f"{label} must be finite")
    return result


def load_failure_audit_contract(path: Path = DEFAULT_CONTRACT) -> FailureAuditContract:
    try:
        root = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FailureAuditContractError("failure audit contract cannot be loaded") from error
    root = _object(root, _TOP_KEYS, "failure audit contract")

    cutoffs = tuple(
        (item.get("audit_id"), item.get("year"))
        for item in root["cutoffs"]
        if type(item) is dict and set(item) == {"audit_id", "year"}
    ) if type(root["cutoffs"]) is list else ()
    types = tuple(root["types"]) if type(root["types"]) is list else ()
    quality = _object(root["quality_gates"], set(_QUALITY_GATES), "quality gates")
    samples = _object(root["sample_gates"], set(_SAMPLE_GATES), "sample gates")
    runtime = _object(root["runtime"], set(_RUNTIME), "runtime")
    hashes = {name: root[name] for name in _HASHES}

    fixed = (
        root["schema_version"] == 1
        and root["campaign_id"] == "failure_expert_label_audit_v1"
        and root["review_only"] is True
        and root["submission_package"] is False
        and hashes == _HASHES
        and all(type(value) is str and _SHA_RE.fullmatch(value) for value in hashes.values())
        and cutoffs == _CUTOFFS
        and types == _TYPES
        and _finite(root["delta_tolerance"], "delta tolerance") == 0.02
        and quality == _QUALITY_GATES
        and samples == _SAMPLE_GATES
        and runtime == _RUNTIME
    )
    if not fixed:
        raise FailureAuditContractError("fixed audit contract differs")

    return FailureAuditContract(
        campaign_id="failure_expert_label_audit_v1",
        official_train_sha256=str(hashes["official_train_sha256"]),
        official_history_sha256=str(hashes["official_history_sha256"]),
        cutoffs=_CUTOFFS,
        types=_TYPES,
        delta_tolerance=0.02,
        minimum_coverage=_finite(quality["minimum_coverage"], "minimum coverage"),
        minimum_binary_delta_fraction=_finite(
            quality["minimum_binary_delta_fraction"], "minimum binary delta fraction"
        ),
        minimum_success_agreement=_finite(
            quality["minimum_success_agreement"], "minimum success agreement"
        ),
        maximum_middle_reverse_overlap=_finite(
            quality["maximum_middle_reverse_overlap"], "maximum overlap"
        ),
        minimum_positive_rows=int(samples["minimum_positive_rows"]),
        minimum_negative_rows=int(samples["minimum_negative_rows"]),
        wall_seconds=int(runtime["wall_seconds"]),
        maximum_rss_bytes=int(runtime["maximum_rss_bytes"]),
    )


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()
