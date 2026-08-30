from __future__ import annotations

import pandas as pd
from types import SimpleNamespace

from experiments.temporal_portfolio.lupi_matching import EntityMaps
import experiments.tree_privileged.matching as matching_module
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


def test_entity_maps_use_the_mapping_contract_that_retains_trackman_ids(monkeypatch) -> None:
    pitcher_mapping = pd.DataFrame({
        "pitcher_id": [11, 12], "pitcher_trackman_id": [111, 112],
        "tm_match_accepted": [1, 0],
    })
    batter_mapping = pd.DataFrame({
        "batter_id": [21, 22], "batter_trackman_id": [211, 212],
        "tm_batter_match_accepted": [1, 0],
    })
    monkeypatch.setattr(
        matching_module, "build_pitcher_mapping",
        lambda *_args, **_kwargs: (pitcher_mapping, pd.DataFrame()), raising=False,
    )
    monkeypatch.setattr(
        matching_module, "fit_batter_trackman",
        lambda *_args, **_kwargs: SimpleNamespace(mapping=batter_mapping),
    )
    maps = matching_module._fit_maps(pd.DataFrame(), pd.DataFrame(), 2021)
    assert dict(maps.pitchers) == {11: 111}
    assert dict(maps.batters) == {21: 211}


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


def test_impossible_trackman_states_are_ignored_without_losing_valid_matches() -> None:
    history = _history()
    invalid = pd.DataFrame([
        [201, "bad-inning", 1, 2024, 5, 2, "KIW_HER", "DOO_BEA", 0, "Top", 0, 0, 0, 111, 211],
        [202, "bad-balls", 1, 2024, 5, 2, "KIW_HER", "DOO_BEA", 1, "Top", 4, 0, 0, 111, 211],
        [203, "bad-strikes", 1, 2024, 5, 2, "KIW_HER", "DOO_BEA", 1, "Top", 0, 3, 0, 111, 211],
        [204, "bad-outs", 1, 2024, 5, 2, "KIW_HER", "DOO_BEA", 1, "Top", 0, 0, 3, 111, 211],
    ], columns=history.columns)
    result = match_training_pitches(
        _main(), pd.concat([history, invalid], ignore_index=True),
        cutoff_year=2024, entity_maps=_maps(),
    )
    assert result["lupi_match_accepted"].eq(1).all()
    assert result["trackman_id"].tolist() == [101, 102]


def test_ambiguous_trackman_games_are_ignored_without_losing_valid_matches() -> None:
    history = _history()
    invalid_games = pd.DataFrame([
        [301, "duplicate-pitch", 1, 2024, 5, 2, "KIW_HER", "DOO_BEA", 1, "Top", 0, 0, 0, 111, 211],
        [302, "duplicate-pitch", 1, 2024, 5, 2, "KIW_HER", "DOO_BEA", 1, "Top", 0, 0, 0, 111, 211],
        [303, "mixed-metadata", 1, 2024, 5, 2, "KIW_HER", "DOO_BEA", 1, "Top", 0, 0, 0, 111, 211],
        [304, "mixed-metadata", 2, 2024, 5, 3, "KIW_HER", "DOO_BEA", 1, "Top", 0, 1, 0, 111, 211],
    ], columns=history.columns)
    result = match_training_pitches(
        _main(), pd.concat([history, invalid_games], ignore_index=True),
        cutoff_year=2024, entity_maps=_maps(),
    )
    assert result["lupi_match_accepted"].eq(1).all()
    assert result["trackman_id"].tolist() == [101, 102]
