from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from experiments.tree_privileged.profiles import (
    ProfileStrengths, build_training_profiles, fit_profiles, select_strengths, transform_profiles,
)


def _rows(seasons=(2021, 2022, 2023), repeats: int = 360) -> pd.DataFrame:
    output = []
    for season in seasons:
        for index in range(repeats):
            output.append({
                "season": season, "pitcher_id": 1 if index < repeats - 5 else 2,
                "batter_id": 10 + index % 2, "balls_before": index % 4,
                "strikes_before": index % 3, "batter_hand": "L" if index % 2 else "R",
                "pitcher_hand": "R", "base_state": str(index % 4), "game_type": "R",
                "pitcher_team_id": 100, "control_success": int(index % 4 == 0),
            })
    return pd.DataFrame(output)


def test_profile_rate_shrinks_to_parent_and_unknown_falls_back() -> None:
    train = _rows(seasons=(2021, 2022))
    state = fit_profiles(train, cutoff_year=2022, strengths=ProfileStrengths(25, 50, 100))
    known_row = train.iloc[[0]].drop(columns="control_success")
    unknown_row = known_row.copy(); unknown_row["pitcher_id"] = 999
    known = transform_profiles(known_row, state)
    unknown = transform_profiles(unknown_row, state)
    assert known.loc[0, "profile_pitcher_count_known"] == 1.0
    assert known.loc[0, "profile_pitcher_count_rate"] != known.loc[0, "profile_pitcher_rate"]
    assert unknown.loc[0, "profile_pitcher_count_known"] == 0.0
    assert unknown.loc[0, "profile_pitcher_count_rate"] == unknown.loc[0, "profile_pitcher_rate"]


def test_training_profiles_never_use_same_or_future_season_targets() -> None:
    rows = _rows()
    expected = build_training_profiles(rows, valid_year=2024, strengths=(25, 50, 100))
    changed = rows.copy()
    changed.loc[changed["season"].eq(2023), "control_success"] ^= 1
    replay = build_training_profiles(changed, valid_year=2024, strengths=(25, 50, 100))
    mask = rows["season"].eq(2023).to_numpy()
    assert_frame_equal(expected.loc[mask].reset_index(drop=True), replay.loc[mask].reset_index(drop=True))


def test_strength_selection_ignores_latest_fold_targets() -> None:
    rows = _rows(seasons=(2021, 2022, 2023, 2024))
    first = select_strengths(rows, ((2021, 2022), (2022, 2023), (2023, 2024)))
    changed = rows.copy()
    changed.loc[changed["season"].eq(2024), "control_success"] ^= 1
    second = select_strengths(changed, ((2021, 2022), (2022, 2023), (2023, 2024)))
    assert first.selected == second.selected
    assert dict(first.scores) == dict(second.scores)
