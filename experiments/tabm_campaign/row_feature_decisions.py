"""Pure paired decisions for the sealed Stage P row-feature proxy."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from experiments.independent_dl.row_features import ROW_FEATURE_BUNDLES

from .row_feature_contracts import RowFeatureProxyContract


class RowFeatureDecisionError(ValueError):
    """Raised when Stage P proxy evidence is malformed."""


@dataclass(frozen=True)
class ProxyMetric:
    bundle: str | None
    seed: int
    status: str
    brier: float | None


@dataclass(frozen=True)
class BundleDecision:
    bundle: str
    seed_delta: Mapping[int, float]
    mean_delta: float
    worst_seed_delta: float
    classification: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seed_delta",
            MappingProxyType(dict(self.seed_delta)),
        )


@dataclass(frozen=True)
class ProxyDecision:
    status: str
    reason: str | None
    strong_survivors: tuple[str, ...]
    safety_survivors: tuple[str, ...]
    failed_bundles: tuple[str, ...]
    rows: tuple[BundleDecision, ...]


_ALLOWED_STATUSES = {"completed", "inconclusive", "failed"}
_EXPECTED_SEEDS = (42, 3407)
_DECISION_STATUSES = {"complete", "incomplete", "blocked"}
_CLASSIFICATIONS = {"strong", "safety", "rejected"}
_DECIMAL_ZERO = Decimal("0")
_DECIMAL_ONE = Decimal("1")
_MISSING_LABELS = (
    *(f"baseline@{seed}" for seed in _EXPECTED_SEEDS),
    *(
        f"{bundle}@{seed}"
        for bundle in ROW_FEATURE_BUNDLES
        for seed in _EXPECTED_SEEDS
    ),
)


def _validate_contract(contract: RowFeatureProxyContract) -> None:
    if not isinstance(contract, RowFeatureProxyContract):
        raise RowFeatureDecisionError("contract must be a RowFeatureProxyContract")
    if (
        type(contract.seeds) is not tuple
        or any(type(seed) is not int for seed in contract.seeds)
        or contract.seeds != _EXPECTED_SEEDS
    ):
        raise RowFeatureDecisionError("contract seeds must equal (42, 3407)")
    if (
        type(contract.feature_bundles) is not tuple
        or any(type(bundle) is not str for bundle in contract.feature_bundles)
        or contract.feature_bundles != ROW_FEATURE_BUNDLES
    ):
        raise RowFeatureDecisionError(
            "contract feature bundles must equal ROW_FEATURE_BUNDLES in order"
        )
    for label, value in (
        ("mean_delta_max", contract.proxy_gate.mean_delta_max),
        ("worst_seed_delta_max", contract.proxy_gate.worst_seed_delta_max),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise RowFeatureDecisionError(f"contract proxy gate {label} must be finite")


def _validated_brier(metric: ProxyMetric) -> Decimal | None:
    value = metric.brier
    if value is None:
        if metric.status == "completed":
            raise RowFeatureDecisionError("completed metric brier must be present")
        return None
    if type(value) is not float:
        raise RowFeatureDecisionError("metric brier must be a Python float or None")
    result = Decimal(str(value))
    if (
        not result.is_finite()
        or not _DECIMAL_ZERO <= result <= _DECIMAL_ONE
    ):
        raise RowFeatureDecisionError("metric brier must be finite and in [0, 1]")
    return result


def _empty_decision(status: str, reason: str) -> ProxyDecision:
    return ProxyDecision(status, reason, (), (), (), ())


def _bundle_decision(
    bundle: str,
    seed_delta: Mapping[int, Decimal],
    mean_delta: Decimal,
    worst_seed_delta: Decimal,
    classification: str,
) -> BundleDecision:
    float_deltas = {seed: float(value) for seed, value in seed_delta.items()}
    return BundleDecision(
        bundle=bundle,
        seed_delta=float_deltas,
        mean_delta=float(mean_delta),
        worst_seed_delta=float(worst_seed_delta),
        classification=classification,
    )


def decide_proxy_survivors(
    metrics: Iterable[ProxyMetric],
    contract: RowFeatureProxyContract,
) -> ProxyDecision:
    """Classify complete feature bundles against their same-seed baselines."""

    _validate_contract(contract)
    evidence: dict[tuple[str | None, int], tuple[str, Decimal | None]] = {}
    for metric in metrics:
        if type(metric) is not ProxyMetric:
            raise RowFeatureDecisionError("every metric must be a ProxyMetric")
        if metric.bundle is not None and type(metric.bundle) is not str:
            raise RowFeatureDecisionError("metric bundle must be a string or None")
        if metric.bundle is not None and metric.bundle not in contract.feature_bundles:
            raise RowFeatureDecisionError(f"unknown metric bundle: {metric.bundle}")
        if type(metric.seed) is not int or metric.seed not in contract.seeds:
            raise RowFeatureDecisionError(f"unknown metric seed: {metric.seed}")
        if type(metric.status) is not str or metric.status not in _ALLOWED_STATUSES:
            raise RowFeatureDecisionError(f"invalid metric status: {metric.status}")
        brier = _validated_brier(metric)
        pair = (metric.bundle, metric.seed)
        if pair in evidence:
            raise RowFeatureDecisionError(
                f"duplicate metric pair: {metric.bundle}@{metric.seed}"
            )
        evidence[pair] = (metric.status, brier)

    if any(
        evidence.get((None, seed), (None, None))[0] == "failed"
        for seed in contract.seeds
    ):
        return _empty_decision("blocked", "baseline_failed")

    expected = [
        *((None, seed) for seed in contract.seeds),
        *(
            (bundle, seed)
            for bundle in contract.feature_bundles
            for seed in contract.seeds
        ),
    ]
    missing = [pair for pair in expected if pair not in evidence]
    if missing:
        labels = [
            f"{'baseline' if bundle is None else bundle}@{seed}"
            for bundle, seed in missing
        ]
        return _empty_decision("incomplete", f"missing_evidence:{','.join(labels)}")

    baseline_statuses = [evidence[(None, seed)][0] for seed in contract.seeds]
    if "inconclusive" in baseline_statuses:
        return _empty_decision("incomplete", "baseline_inconclusive")

    inconclusive_bundles = [
        bundle
        for bundle in contract.feature_bundles
        if any(
            evidence[(bundle, seed)][0] == "inconclusive"
            for seed in contract.seeds
        )
    ]
    if inconclusive_bundles:
        return _empty_decision(
            "incomplete",
            f"feature_inconclusive:{','.join(inconclusive_bundles)}",
        )

    failed_bundles = tuple(
        bundle
        for bundle in contract.feature_bundles
        if any(evidence[(bundle, seed)][0] == "failed" for seed in contract.seeds)
    )
    mean_limit = Decimal(str(contract.proxy_gate.mean_delta_max))
    worst_limit = Decimal(str(contract.proxy_gate.worst_seed_delta_max))
    computed: list[
        tuple[str, Mapping[int, Decimal], Decimal, Decimal, bool]
    ] = []
    for bundle in contract.feature_bundles:
        if bundle in failed_bundles:
            continue
        seed_delta = {
            seed: evidence[(bundle, seed)][1] - evidence[(None, seed)][1]  # type: ignore[operator]
            for seed in contract.seeds
        }
        mean_delta = sum(seed_delta.values(), _DECIMAL_ZERO) / Decimal(
            len(seed_delta)
        )
        worst_seed_delta = max(seed_delta.values())
        strong = (
            mean_delta <= mean_limit
            and worst_seed_delta <= worst_limit
        )
        computed.append(
            (bundle, seed_delta, mean_delta, worst_seed_delta, strong)
        )

    strong_survivors = tuple(row[0] for row in computed if row[4])
    safety_candidates = [
        row for row in computed if not row[4] and row[2] < _DECIMAL_ZERO
    ]
    safety_bundle = (
        min(safety_candidates, key=lambda row: row[2])[0]
        if safety_candidates
        else None
    )
    safety_survivors = (safety_bundle,) if safety_bundle is not None else ()
    rows = tuple(
        _bundle_decision(
            bundle,
            seed_delta,
            mean_delta,
            worst_seed_delta,
            (
                "strong"
                if strong
                else "safety"
                if bundle == safety_bundle
                else "rejected"
            ),
        )
        for bundle, seed_delta, mean_delta, worst_seed_delta, strong in computed
    )
    return ProxyDecision(
        status="complete",
        reason=None,
        strong_survivors=strong_survivors,
        safety_survivors=safety_survivors,
        failed_bundles=failed_bundles,
        rows=rows,
    )


def _ordered_bundle_tuple(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise RowFeatureDecisionError(f"decision {label} must be a tuple")
    if any(
        type(bundle) is not str or bundle not in ROW_FEATURE_BUNDLES
        for bundle in value
    ):
        raise RowFeatureDecisionError(f"decision {label} contains an unknown bundle")
    positions = [ROW_FEATURE_BUNDLES.index(bundle) for bundle in value]
    if len(set(value)) != len(value) or positions != sorted(positions):
        raise RowFeatureDecisionError(
            f"decision {label} must be unique and in sealed bundle order"
        )
    return value


def _is_ordered_subset(values: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    if not values or len(set(values)) != len(values):
        return False
    try:
        positions = [expected.index(value) for value in values]
    except ValueError:
        return False
    return positions == sorted(positions)


def _valid_incomplete_reason(reason: str | None) -> bool:
    if reason == "baseline_inconclusive":
        return True
    if type(reason) is not str:
        return False
    if reason.startswith("feature_inconclusive:"):
        values = tuple(reason.removeprefix("feature_inconclusive:").split(","))
        return _is_ordered_subset(values, ROW_FEATURE_BUNDLES)
    if reason.startswith("missing_evidence:"):
        values = tuple(reason.removeprefix("missing_evidence:").split(","))
        return _is_ordered_subset(values, _MISSING_LABELS)
    return False


def _validate_proxy_decision(decision: ProxyDecision) -> None:
    if type(decision) is not ProxyDecision:
        raise RowFeatureDecisionError("decision must be exactly a ProxyDecision")
    if type(decision.status) is not str or decision.status not in _DECISION_STATUSES:
        raise RowFeatureDecisionError("decision status is invalid")
    if decision.reason is not None and type(decision.reason) is not str:
        raise RowFeatureDecisionError("decision reason must be a string or None")

    strong = _ordered_bundle_tuple(decision.strong_survivors, "strong_survivors")
    safety = _ordered_bundle_tuple(decision.safety_survivors, "safety_survivors")
    failed = _ordered_bundle_tuple(decision.failed_bundles, "failed_bundles")
    if type(decision.rows) is not tuple:
        raise RowFeatureDecisionError("decision rows must be a tuple")
    if set(strong) & set(safety) or set(strong) & set(failed) or set(safety) & set(failed):
        raise RowFeatureDecisionError("decision bundle groups must be disjoint")
    if len(safety) > 1:
        raise RowFeatureDecisionError("decision may have at most one safety survivor")

    if decision.status != "complete":
        if decision.status == "blocked":
            reason_valid = decision.reason == "baseline_failed"
        else:
            reason_valid = _valid_incomplete_reason(decision.reason)
        if not reason_valid:
            raise RowFeatureDecisionError("decision reason contradicts its status")
        if strong or safety or failed or decision.rows:
            raise RowFeatureDecisionError(
                "non-complete decision must not contain bundles or rows"
            )
        return

    if decision.reason is not None:
        raise RowFeatureDecisionError("complete decision reason must be None")
    expected_rows = tuple(
        bundle for bundle in ROW_FEATURE_BUNDLES if bundle not in failed
    )
    actual_rows: list[str] = []
    for row in decision.rows:
        if type(row) is not BundleDecision:
            raise RowFeatureDecisionError("every decision row must be exactly BundleDecision")
        if type(row.bundle) is not str or row.bundle not in ROW_FEATURE_BUNDLES:
            raise RowFeatureDecisionError("decision row contains an unknown bundle")
        if not isinstance(row.seed_delta, Mapping):
            raise RowFeatureDecisionError("decision row seed_delta must be a mapping")
        seeds = tuple(row.seed_delta)
        if (
            any(type(seed) is not int for seed in seeds)
            or seeds != _EXPECTED_SEEDS
        ):
            raise RowFeatureDecisionError(
                "decision row seed_delta keys must equal (42, 3407) in order"
            )
        numbers = (
            *(row.seed_delta[seed] for seed in _EXPECTED_SEEDS),
            row.mean_delta,
            row.worst_seed_delta,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in numbers):
            raise RowFeatureDecisionError("decision row numbers must be finite floats")
        decimal_deltas = tuple(
            Decimal(str(row.seed_delta[seed])) for seed in _EXPECTED_SEEDS
        )
        derived_mean = sum(decimal_deltas, _DECIMAL_ZERO) / Decimal(
            len(decimal_deltas)
        )
        derived_worst = max(decimal_deltas)
        if not math.isclose(
            row.mean_delta,
            float(derived_mean),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RowFeatureDecisionError(
                "decision row mean_delta contradicts seed_delta"
            )
        if not math.isclose(
            row.worst_seed_delta,
            float(derived_worst),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RowFeatureDecisionError(
                "decision row worst_seed_delta contradicts seed_delta"
            )
        if (
            type(row.classification) is not str
            or row.classification not in _CLASSIFICATIONS
        ):
            raise RowFeatureDecisionError("decision row classification is invalid")
        expected_classification = (
            "strong"
            if row.bundle in strong
            else "safety"
            if row.bundle in safety
            else "rejected"
        )
        if row.classification != expected_classification:
            raise RowFeatureDecisionError(
                "decision row classification contradicts survivor tuples"
            )
        actual_rows.append(row.bundle)
    if tuple(actual_rows) != expected_rows:
        raise RowFeatureDecisionError(
            "decision rows must cover nonfailed bundles in sealed order"
        )


def proxy_decision_payload(decision: ProxyDecision) -> dict[str, Any]:
    """Return a fresh JSON-ready representation of a proxy decision."""

    _validate_proxy_decision(decision)
    return {
        "status": decision.status,
        "reason": decision.reason,
        "strong_survivors": list(decision.strong_survivors),
        "safety_survivors": list(decision.safety_survivors),
        "failed_bundles": list(decision.failed_bundles),
        "rows": [
            {
                "bundle": row.bundle,
                "seed_delta": {
                    str(seed): row.seed_delta[seed]
                    for seed in sorted(row.seed_delta)
                },
                "mean_delta": row.mean_delta,
                "worst_seed_delta": row.worst_seed_delta,
                "classification": row.classification,
            }
            for row in decision.rows
        ],
    }


def proxy_decision_json(decision: ProxyDecision) -> bytes:
    """Serialize a proxy decision as deterministic canonical UTF-8 JSON."""

    return json.dumps(
        proxy_decision_payload(decision),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
