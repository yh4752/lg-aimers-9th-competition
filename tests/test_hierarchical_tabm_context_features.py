from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from experiments.hierarchical_tabm.context_features import (
    ContextFeatureError,
    HIERARCHY_LEVELS,
    context_state_from_payload,
    context_state_payload,
    context_state_sha256,
    fit_context_state,
    transform_frozen,
    transform_training_loo,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(8)],
            "control_success": [1, 0, 1, 0, 1, 0, 1, 0],
            "balls_before": [0, 0, 0, 0, 1, 1, 1, 2],
            "strikes_before": [0, 0, 0, 0, 0, 0, 0, 1],
            "pitcher_hand": ["R", "R", "R", "L", "R", "R", "L", "R"],
            "batter_hand": ["R", "R", "R", "R", "L", "L", "L", "R"],
            "base_state": ["000", "000", "100", "000", "000", "000", "000", "010"],
            "outs_before": [0, 0, 0, 0, 1, 1, 1, 2],
            "game_type": ["A", "A", "A", "A", "B", "C", "B", "A"],
        }
    )


def test_hierarchy_levels_are_exact_and_ordered() -> None:
    assert HIERARCHY_LEVELS == (
        ("balls_before", "strikes_before"),
        ("balls_before", "strikes_before", "pitcher_hand", "batter_hand"),
        (
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "batter_hand",
            "base_state",
            "outs_before",
        ),
        (
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "batter_hand",
            "base_state",
            "outs_before",
            "game_type",
        ),
    )


def test_fitted_rates_match_parent_smoothed_hand_calculation() -> None:
    state = fit_context_state(_rows(), smoothing_k=2.0)
    assert state.global_rate == pytest.approx(0.5)
    count_key = ("0", "0")
    hand_key = ("0", "0", "R", "R")
    base_key = ("0", "0", "R", "R", "000", "0")
    game_key = (*base_key, "A")
    count_rate = (2.0 + 2.0 * 0.5) / 6.0
    hand_rate = (2.0 + 2.0 * count_rate) / 5.0
    base_rate = (1.0 + 2.0 * hand_rate) / 4.0
    game_rate = (1.0 + 2.0 * base_rate) / 4.0
    assert state.levels[0].rates[count_key] == pytest.approx(count_rate)
    assert state.levels[1].rates[hand_key] == pytest.approx(hand_rate)
    assert state.levels[2].rates[base_key] == pytest.approx(base_rate)
    assert state.levels[3].rates[game_key] == pytest.approx(game_rate)


def test_leave_one_out_excludes_own_target_at_every_level() -> None:
    rows = _rows()
    state = fit_context_state(rows, smoothing_k=2.0)
    transformed = transform_training_loo(rows, state)

    global_loo = 3.0 / 7.0
    count_loo = (1.0 + 2.0 * global_loo) / 5.0
    hand_loo = (1.0 + 2.0 * count_loo) / 4.0
    base_loo = (0.0 + 2.0 * hand_loo) / 3.0
    game_loo = (0.0 + 2.0 * base_loo) / 3.0
    assert transformed.loc[0, "hier_context_rate"] == pytest.approx(game_loo)

    flipped = rows.copy()
    flipped.loc[0, "control_success"] = 0
    changed = transform_training_loo(
        flipped, fit_context_state(flipped, smoothing_k=2.0)
    )
    assert changed.loc[0, "hier_context_rate"] == pytest.approx(
        transformed.loc[0, "hier_context_rate"]
    )


def test_single_row_leave_one_out_uses_neutral_prior() -> None:
    rows = _rows().iloc[[0]].copy()
    state = fit_context_state(rows, smoothing_k=32.0)
    output = transform_training_loo(rows, state)
    assert output["hier_context_rate"].tolist() == pytest.approx([0.5])


def test_frozen_transform_backs_off_one_level_at_a_time() -> None:
    state = fit_context_state(_rows(), smoothing_k=2.0)
    rows = pd.DataFrame(
        {
            "row_id": ["game", "base", "hand", "count"],
            "balls_before": [0, 0, 0, 3],
            "strikes_before": [0, 0, 0, 2],
            "pitcher_hand": ["R", "R", "X", "R"],
            "batter_hand": ["R", "R", "R", "R"],
            "base_state": ["000", "111", "000", "000"],
            "outs_before": [0, 0, 0, 0],
            "game_type": ["NEW", "A", "A", "A"],
        }
    )
    output = transform_frozen(rows, state)
    count_key = ("0", "0")
    hand_key = (*count_key, "R", "R")
    base_key = (*hand_key, "000", "0")
    assert output["hier_context_rate"].tolist() == pytest.approx(
        [
            state.levels[2].rates[base_key],
            state.levels[1].rates[hand_key],
            state.levels[0].rates[count_key],
            state.global_rate,
        ]
    )


def test_frozen_transform_is_target_blind_and_order_independent() -> None:
    train = _rows()
    state = fit_context_state(train, smoothing_k=32.0)
    valid = train.drop(columns="control_success").copy()
    baseline = transform_frozen(valid, state).set_index(valid["row_id"])
    with_changed_target = transform_frozen(
        valid.assign(control_success=np.arange(len(valid)) % 2), state
    ).set_index(valid["row_id"])
    reversed_output = transform_frozen(valid.iloc[::-1], state)
    reversed_output.index = valid.iloc[::-1]["row_id"]
    pd.testing.assert_frame_equal(baseline.sort_index(), with_changed_target.sort_index())
    pd.testing.assert_frame_equal(baseline.sort_index(), reversed_output.sort_index())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("balls_before", 4, "balls_before"),
        ("strikes_before", 3, "strikes_before"),
        ("outs_before", -1, "outs_before"),
        ("base_state", "12", "base_state"),
        ("control_success", 0.5, "control_success"),
    ],
)
def test_rejects_invalid_training_values(
    column: str, value: object, message: str
) -> None:
    rows = _rows()
    rows.loc[0, column] = value
    with pytest.raises(ContextFeatureError, match=message):
        fit_context_state(rows, smoothing_k=32.0)


@pytest.mark.parametrize("column", ["row_id", "balls_before", "game_type"])
def test_rejects_missing_required_columns(column: str) -> None:
    with pytest.raises(ContextFeatureError, match=column):
        fit_context_state(_rows().drop(columns=column), smoothing_k=32.0)


def test_rejects_empty_duplicate_ids_and_invalid_k() -> None:
    with pytest.raises(ContextFeatureError, match="empty"):
        fit_context_state(_rows().iloc[0:0], smoothing_k=32.0)
    duplicate = _rows()
    duplicate.loc[1, "row_id"] = duplicate.loc[0, "row_id"]
    with pytest.raises(ContextFeatureError, match="row_id"):
        fit_context_state(duplicate, smoothing_k=32.0)
    with pytest.raises(ContextFeatureError, match="smoothing_k"):
        fit_context_state(_rows(), smoothing_k=0.0)


def test_missing_categories_use_one_explicit_marker() -> None:
    rows = _rows()
    rows.loc[0, ["pitcher_hand", "batter_hand", "base_state", "game_type"]] = None
    state = fit_context_state(rows, smoothing_k=32.0)
    assert "__MISSING__" in state.levels[-1].counts.keys().__iter__().__next__() or any(
        "__MISSING__" in key for key in state.levels[-1].counts
    )
    assert np.isfinite(transform_training_loo(rows, state).to_numpy()).all()


def test_state_round_trip_is_canonical_and_immutable() -> None:
    state = fit_context_state(_rows(), smoothing_k=32.0)
    payload = context_state_payload(state)
    restored = context_state_from_payload(payload)
    assert context_state_payload(restored) == payload
    assert context_state_sha256(restored) == context_state_sha256(state)
    with pytest.raises(TypeError):
        restored.levels[0].rates[("0", "0")] = 0.0  # type: ignore[index]


@pytest.mark.parametrize("mutation", ["extra", "rate", "duplicate", "levels"])
def test_rejects_corrupt_serialized_state(mutation: str) -> None:
    payload = deepcopy(context_state_payload(fit_context_state(_rows(), smoothing_k=2.0)))
    if mutation == "extra":
        payload["extra"] = 1
    elif mutation == "rate":
        payload["levels"][0]["entries"][0]["rate"] += 0.01
    elif mutation == "duplicate":
        payload["levels"][0]["entries"].append(
            deepcopy(payload["levels"][0]["entries"][0])
        )
    else:
        payload["levels"] = payload["levels"][::-1]
    with pytest.raises(ContextFeatureError):
        context_state_from_payload(payload)

