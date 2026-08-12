"""Focused local feature-profile rules from the verified feature views.

Source commit: ``9454d68b93971627e3d3f613ce30be690cb5dce2``.
"""

from __future__ import annotations

import pandas as pd


_PAIR_COLUMNS = {
    "team_matchup",
    "pitcher_batter",
    "pitcher_count_state",
}
_TEAMMATE_ADDED_COLUMNS = {
    "we_pitcher_team",
    "pitcher_success_smooth_200",
}
_SMOOTH_PROBE_COLUMNS = {
    "pitcher_success_smooth_125",
    "pitcher_success_smooth_150",
    "pitcher_success_smooth_175",
    "pitcher_success_smooth_250",
    "pitcher_success_smooth_300",
    "pitcher_success_smooth_400",
}
_ROUND8_HAND_COLUMNS = {
    "r8_hand_base",
    "r8_hand_count",
    "r8_game_hand",
    "r8_game_hand_count",
    "r8_count_base",
    "r8_hand_inning",
}
_EXPERIMENTAL_ADDED_COLUMNS = (
    _TEAMMATE_ADDED_COLUMNS | _SMOOTH_PROBE_COLUMNS | _ROUND8_HAND_COLUMNS
)


def apply_r9_feature_profile(
    features: pd.DataFrame,
    categorical: list[str],
    profile: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the exact source drop rules for the two final-R9 profiles."""

    if profile == "no_pairs":
        dropped = _PAIR_COLUMNS | _EXPERIMENTAL_ADDED_COLUMNS
    elif profile == "smooth_k150":
        dropped = _PAIR_COLUMNS | (
            _EXPERIMENTAL_ADDED_COLUMNS - {"pitcher_success_smooth_150"}
        )
    else:
        raise ValueError(f"Unsupported local R9 feature profile: {profile}")

    present = [column for column in dropped if column in features.columns]
    return features.drop(columns=present), [
        column for column in categorical if column not in dropped
    ]
