"""Pure paired decisions for the sealed Stage P row-feature proxy."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
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


def _validated_brier(metric: ProxyMetric) -> float | None:
    value = metric.brier
    if value is None:
        if metric.status == "completed":
            raise RowFeatureDecisionError("completed metric brier must be present")
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RowFeatureDecisionError("metric brier must be a real number or None")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RowFeatureDecisionError("metric brier must be finite and in [0, 1]")
    return result


def _empty_decision(status: str, reason: str) -> ProxyDecision:
    return ProxyDecision(status, reason, (), (), (), ())


def decide_proxy_survivors(
    metrics: Iterable[ProxyMetric],
    contract: RowFeatureProxyContract,
) -> ProxyDecision:
    """Classify complete feature bundles against their same-seed baselines."""

    _validate_contract(contract)
    evidence: dict[tuple[str | None, int], tuple[str, float | None]] = {}
    for metric in metrics:
        if not isinstance(metric, ProxyMetric):
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
    if "failed" in baseline_statuses:
        return _empty_decision("blocked", "baseline_failed")
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
    computed: list[tuple[str, Mapping[int, float], float, float, bool]] = []
    for bundle in contract.feature_bundles:
        if bundle in failed_bundles:
            continue
        seed_delta = {
            seed: evidence[(bundle, seed)][1] - evidence[(None, seed)][1]  # type: ignore[operator]
            for seed in contract.seeds
        }
        mean_delta = sum(seed_delta.values()) / len(seed_delta)
        worst_seed_delta = max(seed_delta.values())
        strong = (
            mean_delta <= contract.proxy_gate.mean_delta_max
            and worst_seed_delta <= contract.proxy_gate.worst_seed_delta_max
        )
        computed.append(
            (bundle, seed_delta, mean_delta, worst_seed_delta, strong)
        )

    strong_survivors = tuple(row[0] for row in computed if row[4])
    safety_candidates = [row for row in computed if not row[4] and row[2] < 0.0]
    safety_bundle = (
        min(safety_candidates, key=lambda row: row[2])[0]
        if safety_candidates
        else None
    )
    safety_survivors = (safety_bundle,) if safety_bundle is not None else ()
    rows = tuple(
        BundleDecision(
            bundle=bundle,
            seed_delta=seed_delta,
            mean_delta=mean_delta,
            worst_seed_delta=worst_seed_delta,
            classification=(
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


def proxy_decision_payload(decision: ProxyDecision) -> dict[str, Any]:
    """Return a fresh JSON-ready representation of a proxy decision."""

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
