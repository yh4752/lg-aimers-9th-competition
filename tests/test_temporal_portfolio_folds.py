from dataclasses import FrozenInstanceError
from decimal import Decimal
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from experiments.temporal_portfolio.contracts import TemporalFold
from experiments.temporal_portfolio.folds import (
    FoldError,
    WeightedRows,
    row_id_sha256,
    select_training_rows,
)


FOLD = TemporalFold(2021, 2019, 2021, 2022)


def _frame(
    seasons: list[object], row_ids: list[object] | None = None
) -> pd.DataFrame:
    ids = row_ids if row_ids is not None else [f"r{index}" for index in range(len(seasons))]
    return pd.DataFrame({"season": seasons, "row_id": ids})


def _expected_digest(row_ids: list[str]) -> str:
    digest = sha256()
    for row_id in row_ids:
        encoded = row_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def test_recent_and_multi_rows_follow_the_contract() -> None:
    frame = _frame([2019, 2020, 2021, 2022], list("abcd"))

    recent = select_training_rows(
        frame, FOLD, expert="recent", decay=None, allow_extra=True
    )
    multi = select_training_rows(
        frame, FOLD, expert="multi", decay=Decimal("0.55"), allow_extra=True
    )

    assert recent.frame["row_id"].tolist() == ["c"]
    assert recent.sample_weight.tolist() == [1.0]
    assert recent.sample_weight.dtype == np.dtype("float32")
    assert multi.frame["row_id"].tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(
        multi.sample_weight,
        np.asarray([Decimal("0.55") ** 2, Decimal("0.55"), 1], dtype="float32"),
    )


def test_validation_and_future_rows_are_rejected_by_default() -> None:
    frame = _frame([2019, 2020, 2021, 2022, 2023])

    with pytest.raises(FoldError, match="training rows.*cutoff"):
        select_training_rows(frame, FOLD, expert="multi", decay=Decimal("0.55"))


def test_allow_extra_filters_validation_and_future_rows_safely() -> None:
    frame = _frame([2019, 2022, 2020, 2023, 2021])
    frame.index = [10, 20, 30, 40, 50]

    selected = select_training_rows(
        frame, FOLD, expert="multi", decay=Decimal("1.00"), allow_extra=True
    )

    assert selected.frame.index.tolist() == [10, 30, 50]
    assert selected.frame["season"].tolist() == [2019, 2020, 2021]
    assert selected.frame["row_id"].tolist() == ["r0", "r2", "r4"]


class _DerivedFold(TemporalFold):
    pass


@pytest.mark.parametrize("fold", (object(), _DerivedFold(2021, 2019, 2021, 2022)))
def test_fold_must_have_the_exact_temporal_fold_type(fold: object) -> None:
    with pytest.raises(FoldError, match="fold type"):
        select_training_rows(_frame([2021]), fold, expert="recent", decay=None)


@pytest.mark.parametrize(
    "fold",
    (
        TemporalFold(True, 2019, 2021, 2022),
        TemporalFold(2021, 2022, 2021, 2023),
        TemporalFold(2020, 2019, 2021, 2022),
        TemporalFold(2021, 2019, 2021, 2021),
    ),
)
def test_fold_years_must_have_exact_valid_ordering(fold: TemporalFold) -> None:
    with pytest.raises(FoldError, match="fold years"):
        select_training_rows(_frame([2021]), fold, expert="recent", decay=None)


@pytest.mark.parametrize(
    ("expert", "decay"),
    (
        ("recent", Decimal("1")),
        ("multi", None),
        ("multi", 0.55),
        ("multi", True),
        ("other", None),
        (True, None),
    ),
)
def test_expert_and_decay_types_must_match_exactly(
    expert: object, decay: object
) -> None:
    with pytest.raises(FoldError, match="expert and decay"):
        select_training_rows(_frame([2019, 2020, 2021]), FOLD, expert=expert, decay=decay)


@pytest.mark.parametrize(
    "decay",
    (
        Decimal("0"),
        Decimal("-0.1"),
        Decimal("1.0001"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ),
)
def test_multi_decay_must_be_finite_positive_and_at_most_one(
    decay: Decimal,
) -> None:
    with pytest.raises(FoldError, match="decay"):
        select_training_rows(
            _frame([2019, 2020, 2021]), FOLD, expert="multi", decay=decay
        )


@pytest.mark.parametrize("flag", (0, 1, np.bool_(True), None, "false"))
def test_prefiltered_must_be_an_exact_bool(flag: object) -> None:
    with pytest.raises(FoldError, match="prefiltered"):
        select_training_rows(
            _frame([2021]), FOLD, expert="recent", decay=None, prefiltered=flag
        )


@pytest.mark.parametrize("flag", (0, 1, np.bool_(True), None, "false"))
def test_allow_extra_must_be_an_exact_bool(flag: object) -> None:
    with pytest.raises(FoldError, match="allow_extra"):
        select_training_rows(
            _frame([2021]), FOLD, expert="recent", decay=None, allow_extra=flag
        )


def test_prefiltered_is_an_auditable_assertion_for_the_chosen_expert() -> None:
    with pytest.raises(FoldError, match="prefiltered"):
        select_training_rows(
            _frame([2020, 2021]),
            FOLD,
            expert="recent",
            decay=None,
            prefiltered=True,
        )
    with pytest.raises(FoldError, match="prefiltered"):
        select_training_rows(
            _frame([2018, 2019, 2020, 2021]),
            FOLD,
            expert="multi",
            decay=Decimal("0.55"),
            prefiltered=True,
            allow_extra=True,
        )


def test_prefiltered_accepts_only_the_complete_chosen_season_set() -> None:
    selected = select_training_rows(
        _frame([2020, 2019, 2021]),
        FOLD,
        expert="multi",
        decay=Decimal("1"),
        prefiltered=True,
    )

    assert selected.frame["season"].tolist() == [2020, 2019, 2021]


def test_frame_must_be_an_actual_dataframe_with_exact_required_columns() -> None:
    with pytest.raises(FoldError, match="DataFrame"):
        select_training_rows(
            {"season": [2021], "row_id": ["a"]},
            FOLD,
            expert="recent",
            decay=None,
        )
    with pytest.raises(FoldError, match="required columns"):
        select_training_rows(
            pd.DataFrame({"Season": [2021], "row_id": ["a"]}),
            FOLD,
            expert="recent",
            decay=None,
        )


def test_duplicate_columns_are_rejected() -> None:
    frame = pd.DataFrame([[2021, "a", "shadow"]], columns=["season", "row_id", "row_id"])

    with pytest.raises(FoldError, match="duplicate columns"):
        select_training_rows(frame, FOLD, expert="recent", decay=None)


@pytest.mark.parametrize(
    "season",
    (None, np.nan, np.inf, -np.inf, 2021.5, True, np.bool_(False), "2021"),
)
def test_season_values_must_be_finite_numeric_integer_years(season: object) -> None:
    with pytest.raises(FoldError, match="season"):
        select_training_rows(_frame([season]), FOLD, expert="recent", decay=None)


def test_integral_numeric_seasons_are_canonicalized_without_mutating_input() -> None:
    frame = _frame([2021.0], [7])
    original = frame.copy(deep=True)

    selected = select_training_rows(frame, FOLD, expert="recent", decay=None)

    pd.testing.assert_frame_equal(frame, original)
    assert selected.frame["season"].tolist() == [2021]
    assert selected.frame["row_id"].tolist() == ["7"]


@pytest.mark.parametrize("row_id", (None, np.nan, "", "   "))
def test_row_ids_must_be_non_null_and_nonempty(row_id: object) -> None:
    with pytest.raises(FoldError, match="row_id"):
        select_training_rows(_frame([2021], [row_id]), FOLD, expert="recent", decay=None)


@pytest.mark.parametrize("row_ids", (["a", "a"], [1, "1"]))
def test_row_ids_must_remain_unique_after_string_canonicalization(
    row_ids: list[object],
) -> None:
    with pytest.raises(FoldError, match="row_id.*unique"):
        select_training_rows(
            _frame([2021, 2021], row_ids), FOLD, expert="recent", decay=None
        )


def test_missing_required_years_and_empty_training_rows_are_rejected() -> None:
    with pytest.raises(FoldError, match="required training seasons"):
        select_training_rows(
            _frame([2019, 2021]), FOLD, expert="multi", decay=Decimal("0.55")
        )
    with pytest.raises(FoldError, match="training rows are empty"):
        select_training_rows(
            _frame([], []), FOLD, expert="recent", decay=None
        )


def test_selection_preserves_input_order_and_index_without_aliasing_caller() -> None:
    frame = pd.DataFrame(
        {
            "season": [2020, 2018, 2021, 2019],
            "row_id": [20, 18, 21, 19],
            "value": [[1], [2], [3], [4]],
        },
        index=[8, 6, 4, 2],
    )
    original = frame.copy(deep=True)

    selected = select_training_rows(
        frame, FOLD, expert="multi", decay=Decimal("0.55")
    )

    pd.testing.assert_frame_equal(frame, original)
    assert selected.frame is not frame
    assert selected.frame.index.tolist() == [8, 4, 2]
    assert selected.frame["row_id"].tolist() == ["20", "21", "19"]


def test_row_digest_is_length_prefixed_collision_safe_and_order_sensitive() -> None:
    assert row_id_sha256(["a", "bc"]) == _expected_digest(["a", "bc"])
    assert row_id_sha256(["a", "bc"]) != row_id_sha256(["ab", "c"])
    assert row_id_sha256(["a", "bc"]) != row_id_sha256(["bc", "a"])


def test_weighted_rows_defensively_copies_frame_and_freezes_weights() -> None:
    frame = _frame([2021], ["a"])
    weights = np.asarray([1.0], dtype="float32")
    rows = WeightedRows(frame, weights, row_id_sha256(["a"]))

    frame.loc[0, "row_id"] = "changed"
    weights[0] = 0.5

    assert rows.frame["row_id"].tolist() == ["a"]
    assert rows.sample_weight.tolist() == [1.0]
    assert not rows.sample_weight.flags.writeable
    with pytest.raises(ValueError):
        rows.sample_weight[0] = 0.5
    with pytest.raises(FrozenInstanceError):
        rows.row_sha256 = "0" * 64


@pytest.mark.parametrize(
    ("weights", "digest", "message"),
    (
        (np.ones(2, dtype="float32"), None, "length"),
        (np.asarray([0.0], dtype="float32"), None, "positive"),
        (np.asarray([np.inf], dtype="float32"), None, "finite"),
        (np.ones((1, 1), dtype="float32"), None, "one-dimensional"),
        (np.asarray(["bad"], dtype=object), None, "numeric"),
        (np.ones(1, dtype="float32"), "0" * 64, "digest"),
    ),
)
def test_weighted_rows_validates_weight_shape_values_length_and_digest(
    weights: np.ndarray, digest: str | None, message: str
) -> None:
    frame = _frame([2021], ["a"])

    with pytest.raises(FoldError, match=message):
        WeightedRows(frame, weights, digest or row_id_sha256(["a"]))
