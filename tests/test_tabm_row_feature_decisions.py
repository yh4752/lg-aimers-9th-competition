from __future__ import annotations

import json
import math
import random
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_UP, getcontext, setcontext
from typing import Any

import numpy as np
import pytest

from experiments.tabm_campaign.row_feature_contracts import (
    load_row_feature_proxy_contract,
)
from experiments.tabm_campaign.row_feature_decisions import (
    BundleDecision,
    ProxyDecision,
    ProxyMetric,
    RowFeatureDecisionError,
    decide_proxy_survivors,
    proxy_decision_json,
    proxy_decision_payload,
)


CONTRACT = load_row_feature_proxy_contract()
SEEDS = CONTRACT.seeds
BUNDLES = CONTRACT.feature_bundles


class _BundleName(str):
    pass


class _FloatValue(float):
    pass


class _ProxyMetricSubclass(ProxyMetric):
    pass


class _BundleDecisionSubclass(BundleDecision):
    pass


class _ProxyDecisionSubclass(ProxyDecision):
    pass


def _evidence(
    *,
    baseline: dict[int, float] | None = None,
    deltas: dict[str, tuple[float, float]] | None = None,
    statuses: dict[tuple[str | None, int], str] | None = None,
    briers: dict[tuple[str | None, int], float | None] | None = None,
) -> list[ProxyMetric]:
    baseline = baseline or {seed: 0.25 for seed in SEEDS}
    deltas = deltas or {}
    statuses = statuses or {}
    briers = briers or {}
    metrics: list[ProxyMetric] = []
    for seed in SEEDS:
        key = (None, seed)
        status = statuses.get(key, "completed")
        score = baseline[seed] if status == "completed" else None
        metrics.append(ProxyMetric(None, seed, status, briers.get(key, score)))
    for bundle in BUNDLES:
        bundle_deltas = deltas.get(bundle, (0.0, 0.0))
        for index, seed in enumerate(SEEDS):
            key = (bundle, seed)
            status = statuses.get(key, "completed")
            score = baseline[seed] + bundle_deltas[index] if status == "completed" else None
            metrics.append(ProxyMetric(bundle, seed, status, briers.get(key, score)))
    return metrics


def _row(decision: ProxyDecision, bundle: str) -> BundleDecision:
    return next(row for row in decision.rows if row.bundle == bundle)


def test_strong_gate_boundary_equality_passes() -> None:
    gate = CONTRACT.proxy_gate
    other_delta = 2 * gate.mean_delta_max - gate.worst_seed_delta_max
    baseline = {SEEDS[0]: -other_delta, SEEDS[1]: 0.0}
    decision = decide_proxy_survivors(
        _evidence(
            baseline=baseline,
            deltas={BUNDLES[0]: (other_delta, gate.worst_seed_delta_max)},
        ),
        CONTRACT,
    )

    row = _row(decision, BUNDLES[0])
    assert row.mean_delta == gate.mean_delta_max
    assert row.worst_seed_delta == gate.worst_seed_delta_max
    assert row.classification == "strong"
    assert BUNDLES[0] in decision.strong_survivors


def test_common_baseline_decimal_mean_boundary_passes() -> None:
    decision = decide_proxy_survivors(
        _evidence(
            baseline={seed: 0.3 for seed in SEEDS},
            deltas={BUNDLES[0]: (-0.00003, -0.00003)},
        ),
        CONTRACT,
    )

    row = _row(decision, BUNDLES[0])
    assert row.mean_delta == CONTRACT.proxy_gate.mean_delta_max
    assert row.classification == "strong"


def test_decimal_worst_seed_boundary_passes() -> None:
    decision = decide_proxy_survivors(
        _evidence(
            baseline={SEEDS[0]: 0.00003, SEEDS[1]: 0.3},
            deltas={BUNDLES[0]: (0.00005, -0.00012)},
        ),
        CONTRACT,
    )

    row = _row(decision, BUNDLES[0])
    assert row.worst_seed_delta == CONTRACT.proxy_gate.worst_seed_delta_max
    assert row.classification == "strong"


def test_any_representable_positive_overshoot_above_zero_gate_is_rejected() -> None:
    baseline = 0.3
    candidate = math.nextafter(baseline, math.inf)
    overshoot = candidate - baseline
    custom_contract = replace(
        CONTRACT,
        proxy_gate=replace(
            CONTRACT.proxy_gate,
            mean_delta_max=0.0,
            worst_seed_delta_max=0.0,
        ),
    )

    decision = decide_proxy_survivors(
        _evidence(
            baseline={seed: baseline for seed in SEEDS},
            deltas={BUNDLES[0]: (overshoot, overshoot)},
        ),
        custom_contract,
    )

    row = _row(decision, BUNDLES[0])
    assert row.mean_delta > 0.0
    assert row.classification == "rejected"
    assert BUNDLES[0] not in decision.strong_survivors


def test_just_outside_mean_gate_is_not_strong() -> None:
    delta = CONTRACT.proxy_gate.mean_delta_max + 1e-12
    decision = decide_proxy_survivors(
        _evidence(deltas={BUNDLES[0]: (delta, delta)}), CONTRACT
    )

    assert _row(decision, BUNDLES[0]).classification == "safety"
    assert BUNDLES[0] not in decision.strong_survivors


def test_just_outside_worst_seed_gate_is_not_strong() -> None:
    gate = CONTRACT.proxy_gate
    worst = gate.worst_seed_delta_max + 1e-12
    other = 2 * gate.mean_delta_max - worst
    decision = decide_proxy_survivors(
        _evidence(deltas={BUNDLES[0]: (other, worst)}), CONTRACT
    )

    assert _row(decision, BUNDLES[0]).classification == "safety"
    assert BUNDLES[0] not in decision.strong_survivors


def test_all_strong_bundles_are_retained_without_a_cap() -> None:
    decision = decide_proxy_survivors(
        _evidence(deltas={bundle: (-0.001, -0.001) for bundle in BUNDLES}),
        CONTRACT,
    )

    assert decision.status == "complete"
    assert decision.strong_survivors == BUNDLES
    assert decision.safety_survivors == ()
    assert [row.classification for row in decision.rows] == ["strong"] * len(BUNDLES)


def test_only_best_negative_nonstrong_bundle_is_safety_survivor() -> None:
    decision = decide_proxy_survivors(
        _evidence(
            deltas={
                BUNDLES[0]: (-0.0001, -0.0001),
                BUNDLES[1]: (-0.0002, 0.0001),
                BUNDLES[2]: (-0.00011, 0.00007),
            }
        ),
        CONTRACT,
    )

    assert decision.strong_survivors == (BUNDLES[0],)
    assert decision.safety_survivors == (BUNDLES[1],)
    assert _row(decision, BUNDLES[0]).classification == "strong"
    assert _row(decision, BUNDLES[1]).classification == "safety"
    assert _row(decision, BUNDLES[2]).classification == "rejected"
    assert set(decision.strong_survivors).isdisjoint(decision.safety_survivors)


def test_no_safety_survivor_when_no_nonstrong_mean_is_negative() -> None:
    decision = decide_proxy_survivors(_evidence(), CONTRACT)

    assert decision.status == "complete"
    assert decision.strong_survivors == ()
    assert decision.safety_survivors == ()
    assert all(row.classification == "rejected" for row in decision.rows)


def test_equal_safety_means_tie_break_in_contract_order() -> None:
    tied = (-0.0002, 0.0001)
    decision = decide_proxy_survivors(
        _evidence(deltas={BUNDLES[0]: tied, BUNDLES[1]: tied}), CONTRACT
    )

    assert decision.safety_survivors == (BUNDLES[0],)
    assert _row(decision, BUNDLES[0]).classification == "safety"
    assert _row(decision, BUNDLES[1]).classification == "rejected"


def test_safety_ranking_uses_unrounded_mean_regardless_of_input_order() -> None:
    displayed_tie = -0.00004
    lower_raw_mean = displayed_tie - 1e-9
    worst = 0.0001
    metrics = _evidence(
        deltas={
            BUNDLES[0]: (2 * displayed_tie - worst, worst),
            BUNDLES[1]: (2 * lower_raw_mean - worst, worst),
        }
    )

    ordered = decide_proxy_survivors(metrics, CONTRACT)
    reversed_input = decide_proxy_survivors(reversed(metrics), CONTRACT)
    first = _row(ordered, BUNDLES[0])
    second = _row(ordered, BUNDLES[1])

    assert round(first.mean_delta, 8) == round(second.mean_delta, 8)
    assert second.mean_delta < first.mean_delta
    assert ordered.safety_survivors == (BUNDLES[1],)
    assert reversed_input == ordered


def test_each_seed_delta_uses_its_own_baseline() -> None:
    baseline = {SEEDS[0]: 0.2, SEEDS[1]: 0.8}
    paired = (-0.001, 0.00002)
    decision = decide_proxy_survivors(
        _evidence(baseline=baseline, deltas={BUNDLES[0]: paired}), CONTRACT
    )

    row = _row(decision, BUNDLES[0])
    assert row.seed_delta[SEEDS[0]] == pytest.approx(paired[0])
    assert row.seed_delta[SEEDS[1]] == pytest.approx(paired[1])
    assert row.mean_delta == pytest.approx(sum(paired) / 2)
    assert row.classification == "strong"


def test_failed_feature_is_listed_and_excludes_only_itself() -> None:
    failed = BUNDLES[1]
    decision = decide_proxy_survivors(
        _evidence(statuses={(failed, SEEDS[0]): "failed"}), CONTRACT
    )

    assert decision.status == "complete"
    assert decision.failed_bundles == (failed,)
    assert [row.bundle for row in decision.rows] == [
        bundle for bundle in BUNDLES if bundle != failed
    ]


@pytest.mark.parametrize("seed", SEEDS)
def test_baseline_failure_on_either_seed_blocks(seed: int) -> None:
    decision = decide_proxy_survivors(
        _evidence(statuses={(None, seed): "failed"}), CONTRACT
    )

    assert decision == ProxyDecision(
        status="blocked",
        reason="baseline_failed",
        strong_survivors=(),
        safety_survivors=(),
        failed_bundles=(),
        rows=(),
    )


def test_observed_baseline_failure_blocks_even_when_grid_is_missing() -> None:
    decision = decide_proxy_survivors(
        [ProxyMetric(None, SEEDS[0], "failed", None)],
        CONTRACT,
    )

    assert decision == ProxyDecision(
        status="blocked",
        reason="baseline_failed",
        strong_survivors=(),
        safety_survivors=(),
        failed_bundles=(),
        rows=(),
    )


def test_failed_bundle_with_inconclusive_counterpart_is_incomplete() -> None:
    failed = BUNDLES[0]
    independent = BUNDLES[1]
    decision = decide_proxy_survivors(
        _evidence(
            statuses={
                (failed, SEEDS[0]): "failed",
                (failed, SEEDS[1]): "inconclusive",
            },
            deltas={independent: (-0.0001, -0.0001)},
        ),
        CONTRACT,
    )

    assert decision.status == "incomplete"
    assert decision.reason == f"feature_inconclusive:{failed}"
    assert decision.strong_survivors == decision.safety_survivors == ()
    assert decision.failed_bundles == ()
    assert decision.rows == ()


def test_failed_bundle_with_missing_counterpart_is_incomplete() -> None:
    failed = BUNDLES[0]
    metrics = [
        metric
        for metric in _evidence(statuses={(failed, SEEDS[0]): "failed"})
        if (metric.bundle, metric.seed) != (failed, SEEDS[1])
    ]

    decision = decide_proxy_survivors(metrics, CONTRACT)

    assert decision.status == "incomplete"
    assert decision.reason == f"missing_evidence:{failed}@{SEEDS[1]}"
    assert decision.strong_survivors == decision.safety_survivors == ()
    assert decision.failed_bundles == ()
    assert decision.rows == ()


def test_failed_bundle_does_not_hide_unrelated_missing_evidence() -> None:
    failed = BUNDLES[0]
    independent = BUNDLES[1]
    missing_seed = SEEDS[1]
    metrics = [
        metric
        for metric in _evidence(statuses={(failed, SEEDS[0]): "failed"})
        if (metric.bundle, metric.seed) != (independent, missing_seed)
    ]

    decision = decide_proxy_survivors(metrics, CONTRACT)

    assert decision.status == "incomplete"
    assert decision.reason == f"missing_evidence:{independent}@{missing_seed}"
    assert decision.strong_survivors == decision.safety_survivors == ()
    assert decision.rows == ()


def test_failed_bundle_does_not_hide_unrelated_inconclusive_evidence() -> None:
    failed = BUNDLES[0]
    independent = BUNDLES[1]
    decision = decide_proxy_survivors(
        _evidence(
            statuses={
                (failed, SEEDS[0]): "failed",
                (independent, SEEDS[1]): "inconclusive",
            }
        ),
        CONTRACT,
    )

    assert decision.status == "incomplete"
    assert decision.reason == f"feature_inconclusive:{independent}"
    assert decision.strong_survivors == decision.safety_survivors == ()
    assert decision.rows == ()


def test_missing_pair_returns_incomplete_and_names_the_evidence() -> None:
    metrics = [
        metric
        for metric in _evidence()
        if (metric.bundle, metric.seed) != (BUNDLES[2], SEEDS[1])
    ]
    decision = decide_proxy_survivors(metrics, CONTRACT)

    assert decision.status == "incomplete"
    assert decision.reason is not None
    assert BUNDLES[2] in decision.reason
    assert str(SEEDS[1]) in decision.reason
    assert decision.strong_survivors == decision.safety_survivors == ()
    assert decision.rows == ()


def test_missing_baseline_pair_returns_incomplete_and_names_baseline_seed() -> None:
    missing_seed = SEEDS[1]
    metrics = [
        metric
        for metric in _evidence()
        if (metric.bundle, metric.seed) != (None, missing_seed)
    ]

    decision = decide_proxy_survivors(metrics, CONTRACT)

    assert decision.status == "incomplete"
    assert decision.reason == f"missing_evidence:baseline@{missing_seed}"
    assert decision.strong_survivors == ()
    assert decision.safety_survivors == ()
    assert decision.rows == ()


@pytest.mark.parametrize("seed", SEEDS)
def test_inconclusive_baseline_returns_incomplete(seed: int) -> None:
    decision = decide_proxy_survivors(
        _evidence(statuses={(None, seed): "inconclusive"}), CONTRACT
    )

    assert decision.status == "incomplete"
    assert decision.reason == "baseline_inconclusive"
    assert decision.rows == ()


def test_inconclusive_candidate_returns_incomplete() -> None:
    bundle = BUNDLES[3]
    decision = decide_proxy_survivors(
        _evidence(statuses={(bundle, SEEDS[0]): "inconclusive"}), CONTRACT
    )

    assert decision.status == "incomplete"
    assert decision.reason == f"feature_inconclusive:{bundle}"
    assert decision.strong_survivors == decision.safety_survivors == ()
    assert decision.rows == ()


@pytest.mark.parametrize("status", ["failed", "inconclusive"])
def test_noncompleted_metric_may_carry_finite_brier_but_it_is_not_selected(
    status: str,
) -> None:
    key = (BUNDLES[0], SEEDS[0])
    decision = decide_proxy_survivors(
        _evidence(statuses={key: status}, briers={key: 0.01}), CONTRACT
    )

    assert BUNDLES[0] not in decision.strong_survivors
    assert BUNDLES[0] not in decision.safety_survivors


def test_duplicate_metric_pair_is_malformed() -> None:
    metrics = _evidence()
    metrics.append(metrics[0])

    with pytest.raises(RowFeatureDecisionError, match="duplicate"):
        decide_proxy_survivors(metrics, CONTRACT)


@pytest.mark.parametrize(
    "bad_metric",
    [
        ProxyMetric("unknown", SEEDS[0], "completed", 0.25),
        ProxyMetric(BUNDLES[0], 99, "completed", 0.25),
        ProxyMetric(BUNDLES[0], SEEDS[0], "done", 0.25),
        ProxyMetric(7, SEEDS[0], "completed", 0.25),  # type: ignore[arg-type]
        ProxyMetric(BUNDLES[0], True, "completed", 0.25),
        ProxyMetric(BUNDLES[0], SEEDS[0], 1, 0.25),  # type: ignore[arg-type]
        _ProxyMetricSubclass(BUNDLES[0], SEEDS[0], "completed", 0.25),
        object(),
    ],
)
def test_unknown_or_wrongly_typed_metric_fields_are_malformed(
    bad_metric: Any,
) -> None:
    with pytest.raises(RowFeatureDecisionError):
        decide_proxy_survivors([bad_metric], CONTRACT)


@pytest.mark.parametrize(
    ("status", "brier"),
    [
        ("completed", None),
        ("completed", 0),
        ("completed", True),
        ("completed", "0.25"),
        ("completed", _FloatValue(0.25)),
        ("completed", Decimal("0.25")),
        ("completed", np.float64(0.25)),
        ("completed", float("nan")),
        ("completed", float("inf")),
        ("completed", float("-inf")),
        ("completed", -0.0001),
        ("completed", 1.0001),
        ("failed", float("nan")),
        ("inconclusive", 1.1),
    ],
)
def test_invalid_brier_is_malformed_even_when_status_does_not_use_it(
    status: str, brier: Any
) -> None:
    bad = ProxyMetric(BUNDLES[0], SEEDS[0], status, brier)

    with pytest.raises(RowFeatureDecisionError, match="brier"):
        decide_proxy_survivors([bad], CONTRACT)


@pytest.mark.parametrize(
    "bad_contract",
    [
        replace(CONTRACT, seeds=(SEEDS[0],)),
        replace(CONTRACT, seeds=tuple(reversed(SEEDS))),
        replace(CONTRACT, seeds=(42.0, 3407.0)),  # type: ignore[arg-type]
        replace(CONTRACT, feature_bundles=BUNDLES[:-1]),
        replace(CONTRACT, feature_bundles=tuple(reversed(BUNDLES))),
        replace(
            CONTRACT,
            feature_bundles=(_BundleName(BUNDLES[0]), *BUNDLES[1:]),
        ),
    ],
)
def test_decision_rejects_contract_with_changed_seed_or_bundle_grid(
    bad_contract: Any,
) -> None:
    with pytest.raises(RowFeatureDecisionError, match="contract"):
        decide_proxy_survivors(_evidence(), bad_contract)


def test_selection_uses_thresholds_from_passed_contract() -> None:
    custom_gate = replace(
        CONTRACT.proxy_gate,
        mean_delta_max=0.0,
        worst_seed_delta_max=0.0,
    )
    custom_contract = replace(CONTRACT, proxy_gate=custom_gate)

    decision = decide_proxy_survivors(_evidence(), custom_contract)

    assert decision.strong_survivors == BUNDLES
    proxy_decision_payload(decision)


def test_shuffled_input_produces_identical_decision_and_json() -> None:
    metrics = _evidence(
        deltas={
            BUNDLES[0]: (-0.0001, -0.0001),
            BUNDLES[1]: (-0.0002, 0.0001),
        },
        statuses={(BUNDLES[-1], SEEDS[1]): "failed"},
    )
    shuffled = list(metrics)
    random.Random(20260817).shuffle(shuffled)

    ordered_decision = decide_proxy_survivors(metrics, CONTRACT)
    shuffled_decision = decide_proxy_survivors(shuffled, CONTRACT)
    assert shuffled_decision == ordered_decision
    assert proxy_decision_json(shuffled_decision) == proxy_decision_json(ordered_decision)


def test_decimal_arithmetic_is_independent_of_global_context() -> None:
    baseline = 0.123456789
    candidate = 0.12342679
    metrics = _evidence(
        baseline={seed: baseline for seed in SEEDS},
        briers={(BUNDLES[0], seed): candidate for seed in SEEDS},
    )
    original_context = getcontext().copy()
    results: list[tuple[ProxyDecision, bytes]] = []
    try:
        for precision, rounding in (
            (28, ROUND_HALF_EVEN),
            (4, ROUND_HALF_EVEN),
            (4, ROUND_UP),
        ):
            getcontext().prec = precision
            getcontext().rounding = rounding
            decision = decide_proxy_survivors(metrics, CONTRACT)
            results.append((decision, proxy_decision_json(decision)))
            assert getcontext().prec == precision
            assert getcontext().rounding == rounding
    finally:
        setcontext(original_context)

    assert results[0] == results[1] == results[2]
    assert results[0][0].safety_survivors == (BUNDLES[0],)
    assert _row(results[0][0], BUNDLES[0]).classification == "safety"


def test_dataclasses_and_seed_mapping_are_immutable() -> None:
    source = {SEEDS[0]: -0.1, SEEDS[1]: -0.2}
    row = BundleDecision(BUNDLES[0], source, -0.15, -0.1, "strong")
    source[SEEDS[0]] = 99.0

    assert row.seed_delta[SEEDS[0]] == -0.1
    with pytest.raises(TypeError):
        row.seed_delta[SEEDS[0]] = 0.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        row.classification = "rejected"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        ProxyMetric(None, SEEDS[0], "completed", 0.2).status = "failed"  # type: ignore[misc]


def test_payload_is_plain_json_ready_shape_and_json_is_canonical() -> None:
    decision = decide_proxy_survivors(
        _evidence(deltas={BUNDLES[0]: (-0.0001, -0.0001)}), CONTRACT
    )
    payload = proxy_decision_payload(decision)

    assert list(payload) == [
        "status",
        "reason",
        "strong_survivors",
        "safety_survivors",
        "failed_bundles",
        "rows",
    ]
    assert isinstance(payload["strong_survivors"], list)
    assert isinstance(payload["rows"], list)
    assert list(payload["rows"][0]) == [
        "bundle",
        "seed_delta",
        "mean_delta",
        "worst_seed_delta",
        "classification",
    ]
    assert payload["rows"][0]["seed_delta"].keys() == {str(seed) for seed in SEEDS}
    assert isinstance(payload["rows"][0]["seed_delta"], dict)
    assert proxy_decision_json(decision) == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    payload["rows"][0]["seed_delta"][str(SEEDS[0])] = 123.0
    assert _row(decision, BUNDLES[0]).seed_delta[SEEDS[0]] != 123.0


def test_json_facing_stats_allow_bounded_delta_float_representation_loss() -> None:
    baseline = {
        SEEDS[0]: 0.5271933079440781,
        SEEDS[1]: 0.00035259763517903053,
    }
    candidates = {
        (BUNDLES[0], SEEDS[0]): 0.5265440012906939,
        (BUNDLES[0], SEEDS[1]): 0.0012426075040075874,
    }
    decision = decide_proxy_survivors(
        _evidence(baseline=baseline, briers=candidates), CONTRACT
    )
    row = _row(decision, BUNDLES[0])
    decimal_deltas = [Decimal(str(row.seed_delta[seed])) for seed in SEEDS]
    expected_mean = float(sum(decimal_deltas, Decimal("0")) / Decimal(len(SEEDS)))

    assert row.mean_delta != expected_mean
    assert math.isclose(row.mean_delta, expected_mean, rel_tol=0.0, abs_tol=1e-15)
    proxy_decision_payload(decision)


def test_exact_nonstrong_boundary_survives_delta_float_persistence() -> None:
    baseline = 3.373543958473668e-05
    candidate = 3.7354395847366805e-06
    briers = {
        (BUNDLES[0], seed): candidate
        for seed in SEEDS
    }
    decision = decide_proxy_survivors(
        _evidence(
            baseline={seed: baseline for seed in SEEDS},
            briers=briers,
        ),
        CONTRACT,
    )

    assert decision.safety_survivors == (BUNDLES[0],)
    assert _row(decision, BUNDLES[0]).mean_delta == -0.00003
    proxy_decision_payload(decision)


def test_serializers_accept_normal_decision_states_deterministically() -> None:
    complete = decide_proxy_survivors(_evidence(), CONTRACT)
    missing = decide_proxy_survivors(_evidence()[:-1], CONTRACT)
    blocked = decide_proxy_survivors(
        [ProxyMetric(None, SEEDS[0], "failed", None)], CONTRACT
    )

    for decision in (complete, missing, blocked):
        assert proxy_decision_payload(decision) == proxy_decision_payload(decision)
        assert proxy_decision_json(decision) == proxy_decision_json(decision)


def _invalid_serialization_decision(case: str) -> ProxyDecision:
    valid = decide_proxy_survivors(
        _evidence(deltas={BUNDLES[0]: (-0.0001, -0.0001)}), CONTRACT
    )
    first = valid.rows[0]
    if case == "decision_subclass":
        return _ProxyDecisionSubclass(
            valid.status,
            valid.reason,
            valid.strong_survivors,
            valid.safety_survivors,
            valid.failed_bundles,
            valid.rows,
        )
    if case == "row_subclass":
        row = _BundleDecisionSubclass(
            first.bundle,
            first.seed_delta,
            first.mean_delta,
            first.worst_seed_delta,
            first.classification,
        )
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "unknown_status":
        return replace(valid, status="unknown")
    if case == "unknown_incomplete_reason":
        return ProxyDecision(
            "incomplete",
            "missing_evidence:not-a-bundle@999",
            (),
            (),
            (),
            (),
        )
    if case == "unordered_incomplete_reason":
        return ProxyDecision(
            "incomplete",
            f"feature_inconclusive:{BUNDLES[1]},{BUNDLES[0]}",
            (),
            (),
            (),
            (),
        )
    if case == "unknown_bundle":
        return replace(valid, rows=(replace(first, bundle="unknown"), *valid.rows[1:]))
    if case == "unknown_classification":
        row = replace(first, classification="maybe")
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "missing_seed":
        row = replace(first, seed_delta={SEEDS[0]: first.seed_delta[SEEDS[0]]})
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "reversed_seeds":
        row = replace(
            first,
            seed_delta={seed: first.seed_delta[seed] for seed in reversed(SEEDS)},
        )
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "extra_seed":
        row = replace(first, seed_delta={**first.seed_delta, 999: 0.0})
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "nonfinite_seed_delta":
        row = replace(
            first,
            seed_delta={SEEDS[0]: float("nan"), SEEDS[1]: 0.0},
        )
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "nonfinite_mean":
        row = replace(first, mean_delta=float("inf"))
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "nonfinite_worst":
        row = replace(first, worst_seed_delta=float("-inf"))
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "inconsistent_mean":
        row = replace(first, mean_delta=first.mean_delta + 1e-9)
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "inconsistent_worst":
        row = replace(first, worst_seed_delta=first.worst_seed_delta + 1e-9)
        return replace(valid, rows=(row, *valid.rows[1:]))
    if case == "contradictory_survivors":
        return replace(valid, strong_survivors=())
    if case == "duplicate_survivors":
        return replace(valid, strong_survivors=(BUNDLES[0], BUNDLES[0]))
    if case == "unordered_survivors":
        return replace(valid, strong_survivors=(BUNDLES[1], BUNDLES[0]))
    if case == "overlapping_survivors":
        return replace(valid, safety_survivors=(BUNDLES[0],))
    if case == "contradictory_rows":
        return replace(valid, rows=valid.rows[:-1])
    if case == "nontuple_rows":
        return replace(valid, rows=list(valid.rows))  # type: ignore[arg-type]
    if case == "contradictory_reason":
        return replace(valid, reason="complete_but_has_reason")
    raise AssertionError(f"unknown test case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "decision_subclass",
        "row_subclass",
        "unknown_status",
        "unknown_incomplete_reason",
        "unordered_incomplete_reason",
        "unknown_bundle",
        "unknown_classification",
        "missing_seed",
        "reversed_seeds",
        "extra_seed",
        "nonfinite_seed_delta",
        "nonfinite_mean",
        "nonfinite_worst",
        "inconsistent_mean",
        "inconsistent_worst",
        "contradictory_survivors",
        "duplicate_survivors",
        "unordered_survivors",
        "overlapping_survivors",
        "contradictory_rows",
        "nontuple_rows",
        "contradictory_reason",
    ],
)
@pytest.mark.parametrize("serializer", [proxy_decision_payload, proxy_decision_json])
def test_serializers_reject_invalid_manual_decisions(
    case: str,
    serializer: Any,
) -> None:
    with pytest.raises(RowFeatureDecisionError):
        serializer(_invalid_serialization_decision(case))
