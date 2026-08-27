import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.hc_contracts import load_hc_contract
from experiments.tree_expert.hc_features import (
    HCFeatureError,
    build_rolling_hierarchy,
    fit_hierarchy,
    shrink,
    transform_hierarchy,
)


def _rows() -> pd.DataFrame:
    records = []
    row = 0
    for season in (2019, 2020, 2021, 2022, 2023, 2024):
        for repeat in range(24):
            records.append(
                {
                    "row_id": f"r{row}",
                    "season": season,
                    "control_success": int((repeat + season) % 3 != 0),
                    "game_type": "R" if repeat % 3 else "F",
                    "pitcher_id": f"p{repeat % 3}",
                    "batter_id": f"b{repeat % 4}",
                    "pitcher_hand": "R" if repeat % 2 else "L",
                    "batter_hand": "L" if repeat % 3 else "R",
                    "balls_before": repeat % 4,
                    "strikes_before": repeat % 3,
                    "outs_before": repeat % 3,
                    "base_state": str(repeat % 4),
                }
            )
            row += 1
    return pd.DataFrame(records)


def test_future_targets_cannot_change_earlier_hierarchy():
    rows = _rows()
    profile = load_hc_contract().profiles["hc_balanced"]
    original = build_rolling_hierarchy(rows, profile_name="hc_balanced", profile=profile)
    changed = rows.copy(deep=True)
    changed.loc[changed["season"].ge(2023), "control_success"] ^= 1
    mutated = build_rolling_hierarchy(changed, profile_name="hc_balanced", profile=profile)
    pd.testing.assert_frame_equal(original[2022], mutated[2022])


def test_shrinkage_uses_parent_rate():
    assert shrink(raw=0.8, parent=0.5, count=20, strength=80) == pytest.approx(0.56)


def test_sparse_interaction_falls_back_to_registered_parent():
    rows = _rows()
    contract = load_hc_contract()
    state = fit_hierarchy(
        rows.loc[rows["season"].le(2021)],
        cutoff_year=2021,
        profile_name="hc_strong",
        profile=contract.profiles["hc_strong"],
        minimum_group_rows=contract.minimum_group_rows,
    )
    query = rows.loc[rows["season"].eq(2022)].head(1).drop(columns="control_success")
    transformed = transform_hierarchy(query, state)
    assert transformed.loc[query.index[0], "hc_pitcher_game_known"] == 0.0
    assert transformed.loc[query.index[0], "hc_pitcher_game_rate"] == pytest.approx(
        transformed.loc[query.index[0], "hc_pitcher_rate"]
    )


def test_transform_is_singleton_shuffle_rebatch_and_duplicate_independent():
    rows = _rows()
    contract = load_hc_contract()
    state = fit_hierarchy(
        rows.loc[rows["season"].le(2022)],
        cutoff_year=2022,
        profile_name="hc_balanced",
        profile=contract.profiles["hc_balanced"],
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    query = rows.loc[rows["season"].eq(2023)].drop(columns="control_success").head(8)
    batch = transform_hierarchy(query, state)
    shuffled_query = query.sample(frac=1, random_state=7)
    shuffled = transform_hierarchy(shuffled_query, state).loc[query.index]
    singleton = pd.concat(
        [transform_hierarchy(query.loc[[index]], state) for index in query.index]
    ).loc[query.index]
    rebatch = pd.concat(
        [transform_hierarchy(part, state) for part in np.array_split(query, 3)]
    ).loc[query.index]
    duplicate_query = pd.concat([query, query.iloc[[0]]], ignore_index=True)
    duplicate = transform_hierarchy(duplicate_query, state)
    np.testing.assert_allclose(batch, shuffled, rtol=0, atol=1e-12)
    np.testing.assert_allclose(batch, singleton, rtol=0, atol=1e-12)
    np.testing.assert_allclose(batch, rebatch, rtol=0, atol=1e-12)
    np.testing.assert_allclose(duplicate.iloc[0], duplicate.iloc[-1], rtol=0, atol=1e-12)


def test_fit_rejects_rows_after_cutoff():
    rows = _rows()
    with pytest.raises(HCFeatureError, match="after cutoff"):
        fit_hierarchy(
            rows,
            cutoff_year=2022,
            profile_name="hc_balanced",
            profile=load_hc_contract().profiles["hc_balanced"],
            minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
        )
