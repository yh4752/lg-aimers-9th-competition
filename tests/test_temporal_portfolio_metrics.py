from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from experiments.temporal_portfolio.contracts import load_contract
from experiments.temporal_portfolio.metrics import (
    PortfolioMetricError,
    T1Recipe,
    align_oof,
    apply_anchor,
    blend_logit,
    blend_probability,
    brier,
    build_t1_recipes,
    score_gain_from_brier_gain,
    score_tier,
)
from experiments.temporal_portfolio.uncertainty import (
    SEGMENT_COLUMNS,
    PortfolioUncertaintyError,
    pitcher_block_bootstrap,
    segment_regressions,
)


SEGMENTS = ("game_type", "pitcher_id_known")


def _uncertainty_frame() -> pd.DataFrame:
    rows = 12
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(rows)],
            "valid_year": [2022] * 4 + [2023] * 4 + [2024] * 4,
            "pitcher_id": [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3,
            "target": [0, 1] * 6,
            "baseline": np.linspace(0.25, 0.75, rows),
            "candidate": np.linspace(0.23, 0.73, rows),
            "game_type": ["regular"] * 10 + ["final"] * 2,
            "hand_matchup": ["same", "opposite"] * 6,
            "pitcher_id_known": ["known"] * 9 + ["oov"] * 3,
            "batter_id_known": ["known", "oov"] * 6,
            "trackman_available": ["yes"] * 8 + ["no"] * 4,
            "history_count_bucket": ["high", "low", "medium"] * 4,
            "runner_state": ["empty", "occupied"] * 6,
            "leverage_bucket": ["low", "medium", "high"] * 4,
        }
    )


def test_pitcher_block_bootstrap_is_deterministic_and_paired() -> None:
    frame = _uncertainty_frame()
    first = pitcher_block_bootstrap(frame, repeats=1000, seed=3407)
    second = pitcher_block_bootstrap(frame, repeats=1000, seed=3407)
    assert first == second
    assert first.repeats == 1000
    assert first.lower <= first.median <= first.upper


def test_small_segments_are_diagnostic_not_eligible() -> None:
    result = segment_regressions(_uncertainty_frame(), minimum_rows=5)
    assert {item.segment for item in result} == set(SEGMENT_COLUMNS)
    small = next(item for item in result if item.rows < 5)
    assert small.eligible is False
    assert all(item.brier_gain == pytest.approx(item.baseline_brier - item.candidate_brier) for item in result)


def test_uncertainty_is_row_order_invariant_and_detached() -> None:
    frame = _uncertainty_frame()
    shuffled = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    first = pitcher_block_bootstrap(frame, repeats=100, seed=7)
    second = pitcher_block_bootstrap(shuffled, repeats=100, seed=7)
    assert first == second


@pytest.mark.parametrize("column", ["row_id", "pitcher_id"])
def test_uncertainty_rejects_invalid_ids(column: str) -> None:
    frame = _uncertainty_frame()
    frame.loc[0, column] = None
    with pytest.raises(PortfolioUncertaintyError, match="IDs|row_id"):
        pitcher_block_bootstrap(frame, repeats=10, seed=1)


def test_uncertainty_rejects_schema_probability_and_control_errors() -> None:
    with pytest.raises(PortfolioUncertaintyError, match="schema"):
        pitcher_block_bootstrap(
            _uncertainty_frame().drop(columns="runner_state"), repeats=10, seed=1
        )
    invalid = _uncertainty_frame()
    invalid.loc[0, "candidate"] = np.nan
    with pytest.raises(PortfolioUncertaintyError, match="candidate"):
        pitcher_block_bootstrap(invalid, repeats=10, seed=1)
    with pytest.raises(PortfolioUncertaintyError, match="repeats"):
        pitcher_block_bootstrap(_uncertainty_frame(), repeats=True, seed=1)
    with pytest.raises(PortfolioUncertaintyError, match="minimum_rows"):
        segment_regressions(_uncertainty_frame(), minimum_rows=0)


def _prediction_frame(
    probabilities: tuple[float, ...] = (0.2, 0.8, 0.4),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["b", "a", "c"],
            "valid_year": [2023, 2022, 2024],
            "target": [0, 1, 0],
            "probability": probabilities,
            "game_type": ["R", "F", "R"],
            "pitcher_id_known": ["known", "oov", "known"],
        }
    )


def _align(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return align_oof(
        (("recent", left), ("multi", right)), segment_columns=SEGMENTS
    )


def test_align_oof_canonically_aligns_reordered_candidate_frames() -> None:
    left = _prediction_frame()
    right = _prediction_frame((0.3, 0.7, 0.5)).iloc[[2, 0, 1]]

    aligned = _align(left, right)

    assert tuple(aligned.columns) == (
        "row_id", "valid_year", "target", *SEGMENTS, "recent", "multi"
    )
    assert list(zip(aligned.row_id, aligned.valid_year, strict=True)) == [
        ("a", 2022), ("b", 2023), ("c", 2024)
    ]
    np.testing.assert_array_equal(aligned["multi"], [0.7, 0.3, 0.5])


@pytest.mark.parametrize("mutation", ("missing", "extra", "changed_key"))
def test_align_oof_rejects_any_key_set_difference(mutation: str) -> None:
    right = _prediction_frame()
    if mutation == "missing":
        right = right.iloc[:-1]
    elif mutation == "extra":
        right = pd.concat(
            [right, right.iloc[[0]].assign(row_id="extra", valid_year=2025)],
            ignore_index=True,
        )
    else:
        right.loc[0, "valid_year"] = 2024

    with pytest.raises(PortfolioMetricError, match="key set differs"):
        _align(_prediction_frame(), right)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (("target", 1, "target differs"), ("game_type", "X", "segment differs")),
)
def test_align_oof_rejects_target_or_registered_segment_mutation(
    column: str, value: object, message: str
) -> None:
    right = _prediction_frame()
    right.loc[0, column] = value

    with pytest.raises(PortfolioMetricError, match=message):
        _align(_prediction_frame(), right)


def test_align_oof_requires_nonnull_unique_ids_and_keys() -> None:
    for invalid in (
        _prediction_frame().assign(row_id=["a", "a", "c"]),
        _prediction_frame().assign(row_id=["a", None, "c"]),
    ):
        with pytest.raises(PortfolioMetricError, match="row_id"):
            _align(invalid, _prediction_frame())


@pytest.mark.parametrize("probability", (np.nan, np.inf, -0.01, 1.01))
def test_align_oof_rejects_invalid_probabilities(probability: float) -> None:
    invalid = _prediction_frame()
    invalid.loc[0, "probability"] = probability

    with pytest.raises(PortfolioMetricError, match="probability"):
        _align(invalid, _prediction_frame())


def test_align_oof_rejects_duplicate_names_columns_and_schema_ambiguity() -> None:
    frame = _prediction_frame()
    duplicate_columns = frame.copy()
    duplicate_columns.columns = [*frame.columns[:-1], "probability"]

    with pytest.raises(PortfolioMetricError, match="candidate names"):
        align_oof((("same", frame), ("same", frame)), segment_columns=SEGMENTS)
    with pytest.raises(PortfolioMetricError, match="columns must be unique"):
        align_oof((("a", duplicate_columns),), segment_columns=SEGMENTS)
    with pytest.raises(PortfolioMetricError, match="schema differs"):
        align_oof((("a", frame.assign(unexpected=1)),), segment_columns=SEGMENTS)
    for reserved_name in ("target", "probability", "game_type"):
        with pytest.raises(PortfolioMetricError, match="candidate name"):
            align_oof(((reserved_name, frame),), segment_columns=SEGMENTS)


def test_align_oof_result_is_detached_from_callers() -> None:
    left = _prediction_frame()
    right = _prediction_frame((0.3, 0.7, 0.5))
    aligned = _align(left, right)
    original = aligned.copy(deep=True)

    left.loc[:, "probability"] = 0.0
    right.loc[:, "game_type"] = "mutated"

    pd.testing.assert_frame_equal(aligned, original)


def test_brier_and_probability_blend_have_expected_values_and_endpoints() -> None:
    target = np.array([0, 1], dtype="int64")
    left = np.array([0.2, 0.8])
    right = np.array([0.4, 0.6])

    assert brier(target, left) == pytest.approx(0.04)
    np.testing.assert_array_equal(blend_probability(left, right, Decimal("1")), left)
    np.testing.assert_array_equal(blend_probability(left, right, Decimal("0")), right)
    np.testing.assert_allclose(
        blend_probability(left, right, Decimal("0.50")), [0.3, 0.7]
    )


def test_logit_blend_has_exact_endpoints_and_finite_extreme_probabilities() -> None:
    left = np.array([0.0, 1.0, 0.2])
    right = np.array([1.0, 0.0, 0.8])

    np.testing.assert_array_equal(blend_logit(left, right, Decimal("1")), left)
    np.testing.assert_array_equal(blend_logit(left, right, Decimal("0")), right)
    middle = blend_logit(left, right, Decimal("0.50"))
    assert np.isfinite(middle).all()
    assert np.all((middle >= 0.0) & (middle <= 1.0))


def test_apply_anchor_and_blends_do_not_mutate_inputs() -> None:
    left = np.array([0.2, 0.8])
    right = np.array([0.4, 0.6])
    left_before = left.copy()
    right_before = right.copy()

    anchored = apply_anchor(left, anchor_rate=0.55, beta=Decimal("0.05"))
    blend_probability(left, right, Decimal("0.50"))
    blend_logit(left, right, Decimal("0.50"))

    assert np.all((anchored > 0.0) & (anchored < 1.0))
    np.testing.assert_array_equal(left, left_before)
    np.testing.assert_array_equal(right, right_before)


def test_anchor_beta_endpoints_preserve_prediction_or_return_anchor() -> None:
    probability = np.array([0.0, 0.8, 1.0])

    np.testing.assert_array_equal(
        apply_anchor(probability, anchor_rate=0.55, beta=Decimal("0")),
        probability,
    )
    np.testing.assert_array_equal(
        apply_anchor(probability, anchor_rate=0.55, beta=Decimal("1")),
        np.full(3, 0.55),
    )


@pytest.mark.parametrize(
    "function,args",
    (
        (brier, (np.array([]), np.array([]))),
        (brier, (np.array([[0, 1]]), np.array([0.2, 0.8]))),
        (brier, (np.array([0, 1]), np.array([0.2]))),
        (brier, (np.array([0, 2]), np.array([0.2, 0.8]))),
        (blend_probability, (np.array([0.2]), np.array([0.3, 0.4]), Decimal("0.5"))),
    ),
)
def test_metrics_reject_invalid_vector_shapes_and_values(function: object, args: tuple[object, ...]) -> None:
    with pytest.raises(PortfolioMetricError):
        function(*args)


@pytest.mark.parametrize("invalid", (True, 1, 0.5, "0.5", Decimal("NaN"), Decimal("-0.1"), Decimal("1.1")))
def test_blend_weights_and_anchor_betas_require_bounded_exact_decimals(invalid: object) -> None:
    left = np.array([0.2])
    right = np.array([0.8])

    with pytest.raises(PortfolioMetricError, match="Decimal"):
        blend_probability(left, right, invalid)
    with pytest.raises(PortfolioMetricError, match="Decimal"):
        apply_anchor(left, anchor_rate=0.5, beta=invalid)


def test_score_gain_uses_exact_decimal_contract_formula() -> None:
    assert score_gain_from_brier_gain(
        Decimal("0.00025"), Decimal("0.25")
    ) == Decimal("100")
    assert score_gain_from_brier_gain(
        Decimal("-0.00025"), Decimal("0.25")
    ) == Decimal("-100")


@pytest.mark.parametrize("baseline", (Decimal("0"), Decimal("-1"), Decimal("Infinity"), 0.25, "0.25", True))
def test_score_gain_rejects_nonpositive_or_non_decimal_baseline(baseline: object) -> None:
    with pytest.raises(PortfolioMetricError, match="baseline"):
        score_gain_from_brier_gain(Decimal("0.1"), baseline)


@pytest.mark.parametrize(
    ("gain", "expected"),
    (
        ("0.00045", "breakthrough"),
        ("0.000449999", "competitive"),
        ("0.00025", "competitive"),
        ("0.000249999", "incremental"),
        ("0.00005", "incremental"),
        ("0.000049999", "below_incremental"),
        ("-1", "below_incremental"),
    ),
)
def test_score_tier_boundaries(gain: str, expected: str) -> None:
    assert score_tier(Decimal(gain)) == expected


def test_t1_recipe_grid_is_complete_unique_deterministic_and_exact() -> None:
    contract = load_contract()
    first = build_t1_recipes(contract)
    second = build_t1_recipes(contract)

    assert len(first) == 160
    assert first == second
    assert len({recipe.recipe_id for recipe in first}) == 160
    assert first[0] == T1Recipe(
        Decimal("0.40"), Decimal("0.50"), "probability", Decimal("0")
    )
    assert first[-1] == T1Recipe(
        Decimal("1.00"), Decimal("1.00"), "logit", Decimal("0.10")
    )
    assert first[0].recipe_id == "t1__d0p40__rw0p50__probability__b0"


@pytest.mark.parametrize(
    "contract",
    (
        replace(load_contract(), decays=(Decimal("0.40"),)),
        replace(load_contract(), recent_weights=load_contract().recent_weights[::-1]),
        replace(load_contract(), anchor_betas=(Decimal("0.0"), *load_contract().anchor_betas[1:])),
    ),
)
def test_t1_recipe_grid_rejects_any_contract_dimension_drift(contract: object) -> None:
    with pytest.raises(PortfolioMetricError, match="contract .* differ"):
        build_t1_recipes(contract)
