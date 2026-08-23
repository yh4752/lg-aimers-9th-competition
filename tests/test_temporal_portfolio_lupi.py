from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

from experiments.temporal_portfolio.lupi_matching import (
    EntityMaps,
    LUPI_MATCH_COLUMNS,
    LupiMatchingError,
    fit_lupi_matches,
    match_current_pitch_rows,
)


def _main_game(*, rows: int = 6) -> pd.DataFrame:
    states = [
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 1, 1, 0),
        (1, 2, 1, 0),
        (1, 2, 2, 0),
        (1, 3, 2, 0),
        (1, 0, 0, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 1),
        (1, 2, 1, 1),
    ][:rows]
    return pd.DataFrame(
        {
            "row_id": [f"main-{index}" for index in range(rows)],
            "season": [2023] * rows,
            "game_month": [5] * rows,
            "game_dayofweek": [2] * rows,
            "pitcher_team_id": [20] * rows,
            "batter_team_id": [10] * rows,
            "inning": [state[0] for state in states],
            "top_bottom": ["T"] * rows,
            "balls_before": [state[1] for state in states],
            "strikes_before": [state[2] for state in states],
            "outs_before": [state[3] for state in states],
            "pitcher_id": [11] * rows,
            "batter_id": [21] * rows,
            "control_success": [index % 2 for index in range(rows)],
        },
        index=[90, 12, 44, 3, 71, 8, 63, 35, 19, 5][:rows],
    )


def _trackman_game(*, rows: int = 6, game_id: str = "tm-game-1") -> pd.DataFrame:
    main = _main_game(rows=rows)
    return pd.DataFrame(
        {
            "trackman_id": [f"pitch-{index}" for index in range(rows)],
            "trackman_game_id": [game_id] * rows,
            "season": main["season"].to_list(),
            "game_month": main["game_month"].to_list(),
            "game_dayofweek": main["game_dayofweek"].to_list(),
            "pitcher_team": ["B"] * rows,
            "batter_team": ["A"] * rows,
            "inning": main["inning"].to_list(),
            "top_bottom": main["top_bottom"].to_list(),
            "balls_before": main["balls_before"].to_list(),
            "strikes_before": main["strikes_before"].to_list(),
            "outs_before": main["outs_before"].to_list(),
            "pitcher_trackman_id": [101] * rows,
            "batter_trackman_id": [201] * rows,
            "rel_speed": [145.0 + index for index in range(rows)],
        }
    )


def _id_maps() -> EntityMaps:
    return EntityMaps.from_mappings(
        pitchers={11: 101},
        batters={21: 201},
        teams={10: "A", 20: "B"},
    )


def test_exact_unique_monotonic_alignment_accepts_and_preserves_source_order() -> None:
    main = _main_game()

    matched = match_current_pitch_rows(main, _trackman_game(), id_maps=_id_maps())

    assert tuple(matched.columns) == LUPI_MATCH_COLUMNS
    assert matched["row_id"].tolist() == main["row_id"].tolist()
    assert matched["trackman_id"].tolist() == [f"pitch-{index}" for index in range(6)]
    assert matched["trackman_id"].is_unique
    assert matched["lupi_match_accepted"].eq(1).all()
    assert matched["lupi_match_coverage"].eq(1.0).all()
    assert matched["lupi_match_mean_cost"].eq(0.0).all()
    assert matched["lupi_match_exact_token_agreement"].eq(1.0).all()
    assert matched["lupi_match_exact_token_evidence"].eq(6).all()


def test_duplicate_or_near_tied_candidate_games_reject_entire_pseudo_game() -> None:
    first = _trackman_game()
    duplicate = first.assign(
        trackman_game_id="tm-game-2",
        trackman_id=[f"duplicate-{index}" for index in range(len(first))],
    )

    matched = match_current_pitch_rows(
        _main_game(), pd.concat([first, duplicate], ignore_index=True), id_maps=_id_maps()
    )

    assert matched["lupi_match_accepted"].eq(0).all()
    assert matched["trackman_id"].isna().all()
    assert matched["lupi_match_candidate_margin"].eq(0.0).all()


def test_future_history_mutation_cannot_change_cutoff_result() -> None:
    main = _main_game()
    history = _trackman_game()
    future = history.assign(
        season=2024,
        trackman_game_id="future",
        rel_speed=9999.0,
    )
    combined = pd.concat([history, future], ignore_index=True)
    expected = fit_lupi_matches(main, combined, cutoff_year=2023, id_maps=_id_maps())
    changed = combined.copy(deep=True)
    changed.loc[changed["season"].eq(2024), "rel_speed"] = -9999.0
    changed.loc[changed["season"].eq(2024), "trackman_id"] = history[
        "trackman_id"
    ].to_list()

    replay = fit_lupi_matches(main, changed, cutoff_year=2023, id_maps=_id_maps())

    pd.testing.assert_frame_equal(replay, expected)


@pytest.mark.parametrize("history", [_trackman_game(rows=4), _trackman_game().iloc[::-1]])
def test_partial_or_nonmonotonic_alignment_rejects_without_forcing(
    history: pd.DataFrame,
) -> None:
    matched = match_current_pitch_rows(_main_game(), history, id_maps=_id_maps())

    assert matched["lupi_match_accepted"].eq(0).all()
    assert matched["trackman_id"].isna().all()


def test_nonexact_aligned_row_remains_explicitly_unmatched() -> None:
    main = _main_game(rows=10)
    history = _trackman_game(rows=10)
    history.loc[9, "batter_trackman_id"] = 999

    matched = match_current_pitch_rows(main, history, id_maps=_id_maps())

    assert matched["lupi_match_mean_cost"].eq(0.10).all()
    assert matched["lupi_match_accepted"].tolist() == [1] * 9 + [0]
    assert pd.isna(matched.iloc[-1]["trackman_id"])


def test_target_mutation_has_no_effect_and_unlabeled_main_is_rejected() -> None:
    main = _main_game()
    expected = fit_lupi_matches(main, _trackman_game(), cutoff_year=2023, id_maps=_id_maps())
    changed = main.assign(control_success=1 - main["control_success"], target=999)

    replay = fit_lupi_matches(changed, _trackman_game(), cutoff_year=2023, id_maps=_id_maps())

    pd.testing.assert_frame_equal(replay, expected)
    with pytest.raises(LupiMatchingError, match="labeled training"):
        fit_lupi_matches(
            main.drop(columns="control_success"),
            _trackman_game(),
            cutoff_year=2023,
            id_maps=_id_maps(),
        )


class _DuplicateItemsMapping(Mapping[object, object]):
    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self):
        return iter((1, 1))

    def __len__(self) -> int:
        return 2

    def items(self):
        return ((1, 101), (1, 102))


@pytest.mark.parametrize(
    "factory_kwargs",
    [
        {"pitchers": {11: 101, 12: 101}, "batters": {21: 201}, "teams": {}},
        {"pitchers": _DuplicateItemsMapping(), "batters": {21: 201}, "teams": {}},
        {"pitchers": {11: 101}, "batters": {21: 201}, "teams": {10: "A", 20: "A"}},
    ],
)
def test_entity_maps_reject_ambiguous_non_one_to_one_mappings(
    factory_kwargs: dict[str, object],
) -> None:
    with pytest.raises(LupiMatchingError, match="one-to-one|duplicate"):
        EntityMaps.from_mappings(**factory_kwargs)


def test_entity_maps_are_sealed_and_duplicate_trackman_pitch_ids_fail_closed() -> None:
    with pytest.raises(TypeError, match="from_mappings"):
        EntityMaps()
    duplicated = _trackman_game()
    duplicated.loc[1, "trackman_id"] = duplicated.loc[0, "trackman_id"]

    with pytest.raises(LupiMatchingError, match="trackman_id.*unique"):
        fit_lupi_matches(
            _main_game(), duplicated, cutoff_year=2023, id_maps=_id_maps()
        )


@pytest.mark.parametrize("cutoff", [2023.0, True, "2023"])
def test_fit_requires_exact_integer_cutoff(cutoff: object) -> None:
    with pytest.raises(LupiMatchingError, match="cutoff_year"):
        fit_lupi_matches(
            _main_game(), _trackman_game(), cutoff_year=cutoff, id_maps=_id_maps()
        )


def test_fit_rejects_evaluation_rows_and_invalid_main_identity() -> None:
    future_main = _main_game().assign(season=2024)
    with pytest.raises(LupiMatchingError, match="beyond cutoff"):
        fit_lupi_matches(
            future_main, _trackman_game(), cutoff_year=2023, id_maps=_id_maps()
        )

    duplicate_ids = _main_game()
    duplicate_ids.loc[duplicate_ids.index[1], "row_id"] = duplicate_ids.iloc[0]["row_id"]
    with pytest.raises(LupiMatchingError, match="row_id.*unique"):
        fit_lupi_matches(
            duplicate_ids, _trackman_game(), cutoff_year=2023, id_maps=_id_maps()
        )
