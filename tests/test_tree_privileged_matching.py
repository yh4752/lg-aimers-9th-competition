from __future__ import annotations

import pandas as pd

from experiments.temporal_portfolio.lupi_matching import EntityMaps
from experiments.tree_privileged.matching import MATCH_COLUMNS, match_training_pitches


def _main() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": ["regular", "futures"], "season": [2024, 2024], "game_month": [5, 5],
        "game_dayofweek": [2, 2], "game_type": ["R", "F"], "pitcher_team_id": [1, 1],
        "batter_team_id": [2, 2], "inning": [1, 1], "top_bottom": ["T", "T"],
        "balls_before": [0, 0], "strikes_before": [0, 0], "outs_before": [0, 0],
        "pitcher_id": [11, 11], "batter_id": [21, 21], "control_success": [1, 0],
    })


def _history(duplicate_regular: bool = False) -> pd.DataFrame:
    rows = [
        [101, "r1", 1, 2024, 5, 2, "KIW_HER", "DOO_BEA", 1, "Top", 0, 0, 0, 111, 211],
        [102, "f1", 1, 2024, 5, 2, "MIN_KIW", "MIN_DOO", 1, "Top", 0, 0, 0, 111, 211],
    ]
    if duplicate_regular:
        rows.append([103, "r2", 1, 2024, 5, 2, "KIW_HER", "DOO_BEA", 1, "Top", 0, 0, 0, 111, 211])
    return pd.DataFrame(rows, columns=(
        "trackman_id", "trackman_game_id", "pitch_no", "season", "game_month", "game_dayofweek",
        "pitcher_team", "batter_team", "inning", "top_bottom", "balls_before", "strikes_before",
        "outs_before", "pitcher_trackman_id", "batter_trackman_id",
    ))


def _maps() -> EntityMaps:
    return EntityMaps.from_mappings(pitchers={11: 111}, batters={21: 211})


def test_same_franchise_regular_and_futures_games_do_not_share_team_code() -> None:
    main = _main()
    result = match_training_pitches(main, _history(), cutoff_year=2024, entity_maps=_maps())
    accepted = result.loc[result["lupi_match_accepted"].eq(1)]
    assert tuple(result.columns) == MATCH_COLUMNS
    assert accepted["row_id"].tolist() == main["row_id"].tolist()
    assert accepted["trackman_id"].is_unique
    assert result.loc[main["game_type"].eq("R"), "matched_game_type"].eq("R").all()
    assert result.loc[main["game_type"].eq("F"), "matched_game_type"].eq("F").all()


def test_ambiguous_candidate_games_are_rejected_not_first_selected() -> None:
    main = _main().iloc[[0]].copy()
    result = match_training_pitches(main, _history(duplicate_regular=True), cutoff_year=2024, entity_maps=_maps())
    assert result["lupi_match_accepted"].eq(0).all()
    assert result["trackman_id"].isna().all()
