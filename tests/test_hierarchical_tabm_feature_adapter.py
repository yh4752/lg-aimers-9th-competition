from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from experiments.hierarchical_tabm.context_features import fit_context_state
from experiments.hierarchical_tabm.feature_adapter import (
    EXPECTED_HIERARCHY_COLUMNS,
    HierarchicalFeatureError,
    attach_hierarchical_numeric,
    feature_state_from_payload,
    feature_state_payload,
    feature_state_sha256,
    prepare_fold,
    transform_with_state,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["r0", "r1", "r2", "r3", "r4", "r5"],
            "season": [2021, 2021, 2021, 2021, 2022, 2022],
            "game_type": ["R", "R", "R", "P", "R", "X"],
            "balls_before": [0, 0, 1, 1, 0, 2],
            "strikes_before": [0, 0, 1, 1, 0, 2],
            "pitcher_hand": ["R", "R", "L", "L", "R", "S"],
            "batter_hand": ["L", "L", "R", "R", "L", "R"],
            "base_state": ["000", "000", "100", "100", "000", "111"],
            "outs_before": [0, 0, 1, 1, 0, 2],
            "pitcher_id": ["p1", "p1", "p2", "p2", "p1", "new-p"],
            "batter_id": ["b1", "b2", "b1", "b2", "b1", "new-b"],
            "asof_pitcher_n": [100, 0, 20, 40, 120, 3],
            "asof_pitcher_success_rate": [0.75, np.nan, 0.40, 0.55, 0.70, 0.2],
            "asof_batter_n": [80, 30, 0, 50, 100, 2],
            "asof_batter_success_rate": [0.25, 0.60, np.nan, 0.45, 0.30, 0.8],
            "li": [0.2, 0.4, 0.8, 1.0, 0.3, 9.0],
            "control_success": [1, 0, 1, 0, 1, 0],
        }
    )


def _prepared():
    rows = _rows()
    train = rows.iloc[:4].copy()
    valid = rows.iloc[4:].copy()
    context = fit_context_state(train, smoothing_k=32.0)
    return train, valid, prepare_fold(
        train,
        valid,
        context_state=context,
        pitcher_k=100.0,
        batter_k=250.0,
    )


def _batch_by_id(batch) -> dict[str, tuple[tuple[float, ...], tuple[int, ...]]]:
    return {
        str(row_id): (
            tuple(float(value) for value in batch.x_num[position]),
            tuple(int(value) for value in batch.x_cat[position]),
        )
        for position, row_id in enumerate(batch.row_id)
    }


def test_adds_exact_hierarchy_columns_in_order_and_arithmetic() -> None:
    rows = _rows().iloc[:2].copy()
    context = np.array([0.5, 0.25])
    output = attach_hierarchical_numeric(
        rows, context, pitcher_k=100.0, batter_k=250.0
    )

    assert tuple(output.columns[-8:]) == EXPECTED_HIERARCHY_COLUMNS
    assert output.index.equals(rows.index)
    assert output.loc[0, "hier_pitcher_reliability"] == pytest.approx(100 / 200)
    assert output.loc[0, "hier_batter_reliability"] == pytest.approx(80 / 330)
    assert output.loc[0, "hier_pitcher_context_gap"] == pytest.approx(0.25)
    assert output.loc[0, "hier_pitcher_weighted_gap"] == pytest.approx(0.25 * 0.5)
    assert output.loc[1, "hier_pitcher_context_gap"] == pytest.approx(0.0)
    assert output.loc[1, "hier_context_logit"] == pytest.approx(np.log(0.25 / 0.75))
    assert np.isfinite(output.loc[:, EXPECTED_HIERARCHY_COLUMNS]).all().all()


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_context_logit_is_clipped(rate: float) -> None:
    output = attach_hierarchical_numeric(
        _rows().iloc[:1], [rate], pitcher_k=100.0, batter_k=250.0
    )
    clipped = np.clip(rate, 1e-6, 1 - 1e-6)
    assert output.loc[0, "hier_context_logit"] == pytest.approx(
        np.log(clipped / (1 - clipped))
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("asof_pitcher_n", -1),
        ("asof_batter_n", np.inf),
        ("asof_pitcher_success_rate", 1.01),
        ("asof_batter_success_rate", -0.01),
    ],
)
def test_invalid_entity_history_is_rejected(column: str, value: float) -> None:
    frame = _rows().iloc[:1].copy()
    frame.loc[frame.index[0], column] = value
    with pytest.raises(HierarchicalFeatureError):
        attach_hierarchical_numeric(
            frame, [0.5], pitcher_k=100.0, batter_k=250.0
        )


@pytest.mark.parametrize("bad_k", [0, -1, np.inf, True])
def test_invalid_reliability_k_is_rejected(bad_k: object) -> None:
    with pytest.raises(HierarchicalFeatureError):
        attach_hierarchical_numeric(
            _rows().iloc[:1], [0.5], pitcher_k=bad_k, batter_k=250.0
        )


def test_output_collision_and_context_alignment_are_rejected() -> None:
    frame = _rows().iloc[:1].copy()
    frame["hier_context_rate"] = 0.5
    with pytest.raises(HierarchicalFeatureError, match="collision"):
        attach_hierarchical_numeric(frame, [0.5], pitcher_k=100, batter_k=250)
    with pytest.raises(HierarchicalFeatureError, match="length"):
        attach_hierarchical_numeric(_rows().iloc[:1], [], pitcher_k=100, batter_k=250)
    with pytest.raises(HierarchicalFeatureError, match="context"):
        attach_hierarchical_numeric(
            _rows().iloc[:1], [np.nan], pitcher_k=100, batter_k=250
        )


def test_prepare_fold_uses_loo_for_train_and_frozen_for_valid() -> None:
    train, valid, prepared = _prepared()

    assert prepared.train.row_id.tolist() == train["row_id"].tolist()
    assert prepared.valid.row_id.tolist() == valid["row_id"].tolist()
    assert prepared.train.y.tolist() == train["control_success"].astype(float).tolist()
    assert prepared.metadata.n_num_features == prepared.train.x_num.shape[1]
    assert prepared.metadata.train_x_num is prepared.train.x_num
    assert tuple(prepared.state.numeric_columns[-8:]) == EXPECTED_HIERARCHY_COLUMNS
    assert prepared.state.categorical_columns.count("hand_matchup") == 1
    assert np.isfinite(prepared.train.x_num).all()
    assert np.isfinite(prepared.valid.x_num).all()


def test_frozen_feature_batch_is_order_and_batch_independent() -> None:
    _, valid, prepared = _prepared()
    baseline = _batch_by_id(transform_with_state(valid, prepared.state))
    reversed_rows = _batch_by_id(
        transform_with_state(valid.iloc[::-1].copy(), prepared.state)
    )
    singles = {}
    for position in range(len(valid)):
        singles.update(
            _batch_by_id(transform_with_state(valid.iloc[[position]], prepared.state))
        )

    assert reversed_rows == baseline
    assert singles == baseline


def test_validation_rows_and_targets_cannot_change_a_fixed_row() -> None:
    _, valid, prepared = _prepared()
    baseline_sha = feature_state_sha256(prepared.state)
    baseline = _batch_by_id(transform_with_state(valid.iloc[[0]], prepared.state))["r4"]
    changed = valid.copy()
    changed.loc[changed.index[1], "li"] = -99999
    changed.loc[changed.index[1], "control_success"] = 1
    changed = pd.concat([changed, changed.iloc[[1]].assign(row_id="extra")])

    current = _batch_by_id(transform_with_state(changed, prepared.state))["r4"]
    assert current == baseline
    assert feature_state_sha256(prepared.state) == baseline_sha


def test_unseen_categories_map_to_zero_without_changing_state() -> None:
    _, valid, prepared = _prepared()
    batch = transform_with_state(valid.iloc[[1]], prepared.state)
    pitcher_position = prepared.state.categorical_columns.index("pitcher_id")
    batter_position = prepared.state.categorical_columns.index("batter_id")
    assert batch.x_cat[0, pitcher_position] == 0
    assert batch.x_cat[0, batter_position] == 0


def test_feature_state_round_trip_is_canonical_and_exact() -> None:
    _, valid, prepared = _prepared()
    payload = feature_state_payload(prepared.state)
    restored = feature_state_from_payload(payload)

    assert feature_state_payload(restored) == payload
    assert feature_state_sha256(restored) == feature_state_sha256(prepared.state)
    assert _batch_by_id(transform_with_state(valid, restored)) == _batch_by_id(
        transform_with_state(valid, prepared.state)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["preprocessing"]["numeric_mean"].update({"li": 999.0}),
        lambda p: p["category_maps"]["pitcher_id"].update({"p1": 9}),
        lambda p: p["numeric_columns"].reverse(),
        lambda p: p.update({"context_sha256": "0" * 64}),
        lambda p: p.update({"extra": 1}),
    ],
)
def test_corrupt_feature_state_is_rejected(mutation) -> None:
    _, _, prepared = _prepared()
    payload = deepcopy(feature_state_payload(prepared.state))
    mutation(payload)
    with pytest.raises(HierarchicalFeatureError):
        feature_state_from_payload(payload)
