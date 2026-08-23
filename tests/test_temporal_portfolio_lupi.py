from __future__ import annotations

from collections.abc import Mapping
import sys
import types

import numpy as np
import pandas as pd
import pytest

from experiments.temporal_portfolio.lupi_matching import (
    EntityMaps,
    LUPI_MATCH_COLUMNS,
    LupiMatchingError,
    fit_lupi_matches,
    match_current_pitch_rows,
)
from experiments.temporal_portfolio.lupi_teacher import (
    TEACHER_FEATURES,
    CatBoostTeacherBackend,
    TeacherError,
    build_teacher_vector,
    crossfit_teacher,
    teacher_fold,
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
            "pitch_no": list(range(1, rows + 1)),
            "season": main["season"].to_list(),
            "game_month": main["game_month"].to_list(),
            "game_dayofweek": main["game_dayofweek"].to_list(),
            "pitcher_team": ["B"] * rows,
            "batter_team": ["A"] * rows,
            "inning": main["inning"].to_list(),
            "top_bottom": ["Top"] * rows,
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


def _canonically_nonmonotonic_history() -> pd.DataFrame:
    history = _trackman_game()
    for column in ("inning", "balls_before", "strikes_before", "outs_before"):
        history[column] = history[column].iloc[::-1].to_list()
    return history


def _alternating_two_inning_game() -> tuple[pd.DataFrame, pd.DataFrame, EntityMaps]:
    segments = (
        (1, "T", 20, 10, 11, 21),
        (1, "B", 10, 20, 12, 22),
        (2, "T", 20, 10, 11, 21),
        (2, "B", 10, 20, 12, 22),
    )
    rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    pitch_no = 0
    for inning, side, pitcher_team, batter_team, pitcher, batter in segments:
        for balls, strikes in ((0, 0), (1, 0), (1, 1)):
            pitch_no += 1
            rows.append(
                {
                    "row_id": f"main-{pitch_no}",
                    "season": 2023,
                    "game_month": 5,
                    "game_dayofweek": 2,
                    "pitcher_team_id": pitcher_team,
                    "batter_team_id": batter_team,
                    "inning": inning,
                    "top_bottom": side,
                    "balls_before": balls,
                    "strikes_before": strikes,
                    "outs_before": 0,
                    "pitcher_id": pitcher,
                    "batter_id": batter,
                    "control_success": pitch_no % 2,
                }
            )
            history_rows.append(
                {
                    "trackman_id": f"pitch-{pitch_no}",
                    "trackman_game_id": "official-game",
                    "pitch_no": pitch_no,
                    "season": 2023,
                    "game_month": 5,
                    "game_dayofweek": 2,
                    "pitcher_team": "B" if pitcher_team == 20 else "A",
                    "batter_team": "A" if batter_team == 10 else "B",
                    "inning": inning,
                    "top_bottom": "Top" if side == "T" else "Bottom",
                    "balls_before": balls,
                    "strikes_before": strikes,
                    "outs_before": 0,
                    "pitcher_trackman_id": 101 if pitcher == 11 else 102,
                    "batter_trackman_id": 201 if batter == 21 else 202,
                }
            )
    maps = EntityMaps.from_mappings(
        pitchers={11: 101, 12: 102},
        batters={21: 201, 22: 202},
        teams={10: "A", 20: "B"},
    )
    return pd.DataFrame(rows), pd.DataFrame(history_rows), maps


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


def test_alternating_two_inning_game_matches_each_exact_half_inning_scope() -> None:
    main, history, maps = _alternating_two_inning_game()
    shuffled = history.iloc[[8, 0, 11, 3, 6, 1, 9, 5, 2, 10, 4, 7]]

    matched = match_current_pitch_rows(main, shuffled, id_maps=maps)

    assert matched["row_id"].tolist() == main["row_id"].tolist()
    assert matched["trackman_id"].tolist() == history["trackman_id"].tolist()
    assert matched["lupi_match_accepted"].eq(1).all()
    assert matched["lupi_match_coverage"].eq(1.0).all()
    assert matched["lupi_match_mean_cost"].eq(0.0).all()


def test_wrong_inning_rows_are_not_candidate_evidence() -> None:
    main = _main_game()
    wrong_inning = _trackman_game().assign(inning=2)

    matched = match_current_pitch_rows(main, wrong_inning, id_maps=_id_maps())

    assert matched["lupi_match_accepted"].eq(0).all()
    assert matched["trackman_id"].isna().all()
    assert matched["lupi_match_coverage"].eq(0.0).all()
    assert matched["lupi_match_mean_cost"].isna().all()


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
    combined = pd.concat([history, future], ignore_index=True).iloc[
        [2, 7, 0, 10, 5, 6, 1, 11, 4, 8, 3, 9]
    ]
    expected = fit_lupi_matches(main, combined, cutoff_year=2023, id_maps=_id_maps())
    changed = combined.copy(deep=True)
    changed.loc[changed["season"].eq(2024), "rel_speed"] = -9999.0
    changed.loc[changed["season"].eq(2024), "trackman_id"] = [
        f"pitch-{index}" for index in changed.loc[changed["season"].eq(2024), "pitch_no"] - 1
    ]

    replay = fit_lupi_matches(main, changed, cutoff_year=2023, id_maps=_id_maps())

    pd.testing.assert_frame_equal(replay, expected)


def test_official_side_and_pitch_number_canonicalize_shuffled_interleaved_games() -> None:
    main = _main_game()
    official = _trackman_game().assign(top_bottom="Top")
    decoy = _trackman_game(game_id="decoy").assign(
        game_month=6,
        top_bottom="Top",
        trackman_id=[f"decoy-{index}" for index in range(6)],
    )
    interleaved = pd.concat([official, decoy], ignore_index=True).iloc[
        [2, 7, 0, 10, 5, 6, 1, 11, 4, 8, 3, 9]
    ]

    expected = fit_lupi_matches(
        main, official, cutoff_year=2023, id_maps=_id_maps()
    )
    replay = fit_lupi_matches(
        main, interleaved, cutoff_year=2023, id_maps=_id_maps()
    )

    assert expected["lupi_match_accepted"].eq(1).all()
    assert expected["trackman_id"].tolist() == [f"pitch-{index}" for index in range(6)]
    pd.testing.assert_frame_equal(replay, expected)


@pytest.mark.parametrize(
    ("source", "side"),
    [("main", "Top"), ("history", "T"), ("main", "X"), ("history", "Unknown")],
)
def test_side_values_are_strict_for_each_official_schema(source: str, side: str) -> None:
    main = _main_game()
    history = _trackman_game().assign(top_bottom="Top")
    if source == "main":
        main["top_bottom"] = side
    else:
        history["top_bottom"] = side

    with pytest.raises(LupiMatchingError, match="top_bottom"):
        fit_lupi_matches(main, history, cutoff_year=2023, id_maps=_id_maps())


def test_duplicate_pitch_number_within_trackman_game_fails_closed() -> None:
    history = _trackman_game().assign(top_bottom="Top")
    history.loc[1, "pitch_no"] = history.loc[0, "pitch_no"]

    with pytest.raises(LupiMatchingError, match="pitch_no.*unique"):
        fit_lupi_matches(
            _main_game(), history, cutoff_year=2023, id_maps=_id_maps()
        )


@pytest.mark.parametrize(
    "history", [_trackman_game(rows=4), _canonically_nonmonotonic_history()]
)
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


class _RecordingTeacherBackend:
    def __init__(self, prediction: object | None = None) -> None:
        self.fit_calls: list[dict[str, object]] = []
        self.predict_calls: list[dict[str, object]] = []
        self.prediction = prediction

    def fit(
        self, features: pd.DataFrame, target: pd.Series, *, seed: int
    ) -> dict[str, object]:
        model = {
            "pitchers": frozenset(features.index.map(lambda index: index[0])),
            "seed": seed,
            "target": tuple(target),
            "columns": tuple(features.columns),
        }
        self.fit_calls.append(model)
        return model

    def predict(self, model: object, features: pd.DataFrame) -> np.ndarray:
        assert isinstance(model, dict)
        call = {
            "train_pitchers": model["pitchers"],
            "valid_pitchers": frozenset(features.index.map(lambda index: index[0])),
            "seed": model["seed"],
            "columns": tuple(features.columns),
        }
        self.predict_calls.append(call)
        if self.prediction is not None:
            value = self.prediction
            return np.asarray(value(len(features)) if callable(value) else value)
        return np.linspace(0.2, 0.8, len(features), dtype="float64")

    @property
    def evidence(self) -> Mapping[str, object]:
        return {"name": "recording", "fit_count": len(self.fit_calls)}


def _teacher_rows(*, folds: int = 3, seed: int = 3407) -> pd.DataFrame:
    pitchers_by_fold: dict[int, int] = {}
    candidate = 10
    while len(pitchers_by_fold) < folds:
        assigned = teacher_fold(candidate, seed, folds)
        pitchers_by_fold.setdefault(assigned, candidate)
        candidate += 1
    records: list[dict[str, object]] = []
    for fold, pitcher_id in sorted(pitchers_by_fold.items()):
        for repeat in range(2):
            record: dict[str, object] = {
                "row_id": f"row-{fold}-{repeat}",
                "pitcher_id": pitcher_id,
                "control_success": repeat,
                "lupi_match_accepted": 1,
            }
            for offset, feature in enumerate(TEACHER_FEATURES):
                record[feature] = float(100 * fold + 10 * repeat + offset)
            records.append(record)
    result = pd.DataFrame(records)
    result.index = pd.MultiIndex.from_arrays(
        [result["pitcher_id"], np.arange(len(result))]
    )
    return result


def test_teacher_probability_is_genuinely_held_out_by_pitcher_group() -> None:
    rows = _teacher_rows()
    backend = _RecordingTeacherBackend()

    result = crossfit_teacher(rows, seed=3407, folds=3, backend=backend)

    assert result.probability.between(0.0, 1.0).all()
    assert tuple(result.row_ids) == tuple(rows["row_id"])
    assert set(result.predicted_by_fold) == set(rows["row_id"])
    assert len(backend.predict_calls) == 3
    assert all(
        call["train_pitchers"].isdisjoint(call["valid_pitchers"])
        for call in backend.predict_calls
    )
    assert all(
        result.fold_by_row_id[row_id] == predicted_fold
        for row_id, predicted_fold in result.predicted_by_fold.items()
    )


def test_teacher_fold_is_stable_typed_and_independent_of_row_order() -> None:
    ids = ["pitcher-z", 91, "pitcher-a", 91]
    first = [teacher_fold(value, 3407, 5) for value in ids]
    replay = {
        value: teacher_fold(value, 3407, 5) for value in reversed(ids)
    }

    assert first == [replay[value] for value in ids]
    assert first[1] == first[3]
    assert teacher_fold(91, 3407, 5) == teacher_fold(91, 3407, 5)
    with pytest.raises(TeacherError, match="pitcher_id"):
        teacher_fold(True, 3407, 5)
    with pytest.raises(TeacherError, match="pitcher_id"):
        teacher_fold(float("nan"), 3407, 5)
    with pytest.raises(TeacherError, match="pitcher_id"):
        teacher_fold(object(), 3407, 5)
    with pytest.raises(TeacherError, match="seed"):
        teacher_fold(91, True, 5)
    with pytest.raises(TeacherError, match="folds"):
        teacher_fold(91, 3407, 1)


def test_teacher_vector_preserves_requested_order_and_nan_for_unmatched() -> None:
    rows = _teacher_rows()
    oof = crossfit_teacher(
        rows, seed=3407, folds=3, backend=_RecordingTeacherBackend()
    )
    requested = [
        rows.iloc[2]["row_id"],
        "unmatched",
        rows.iloc[0]["row_id"],
        *rows.iloc[[1, 3, 4, 5]]["row_id"],
    ]

    vector = build_teacher_vector(requested, oof)
    expected = oof.probability_by_row_id

    assert vector.dtype == np.dtype("float32")
    assert vector[0] == pytest.approx(expected[requested[0]])
    assert np.isnan(vector[1])
    assert vector[2] == pytest.approx(expected[requested[2]])
    vector[0] = 0.99
    assert oof.probability_by_row_id[requested[0]] == pytest.approx(expected[requested[0]])


def test_teacher_validates_schema_target_identity_and_backend_probabilities() -> None:
    rows = _teacher_rows()
    backend = _RecordingTeacherBackend()
    changed_target = rows.assign(control_success=1 - rows["control_success"])
    crossfit_teacher(changed_target, seed=3407, folds=3, backend=backend)
    assert all(call["columns"] == TEACHER_FEATURES for call in backend.fit_calls)
    assert any(call["target"] != (0, 1, 0, 1) for call in backend.fit_calls)

    duplicate = rows.copy(deep=True)
    duplicate.iloc[1, duplicate.columns.get_loc("row_id")] = duplicate.iloc[0]["row_id"]
    with pytest.raises(TeacherError, match="row_id.*unique"):
        crossfit_teacher(duplicate, seed=3407, folds=3, backend=_RecordingTeacherBackend())
    null_id = rows.copy(deep=True)
    null_id.iloc[0, null_id.columns.get_loc("pitcher_id")] = None
    with pytest.raises(TeacherError, match="pitcher_id"):
        crossfit_teacher(null_id, seed=3407, folds=3, backend=_RecordingTeacherBackend())
    with pytest.raises(TeacherError, match="schema"):
        crossfit_teacher(rows.assign(student_validation=1), seed=3407, folds=3, backend=_RecordingTeacherBackend())

    for prediction in (
        lambda size: np.full(size + 1, 0.5),
        lambda size: np.full(size, 1.01),
        lambda size: np.full(size, np.nan),
    ):
        with pytest.raises(TeacherError, match="probabil"):
            crossfit_teacher(
                rows,
                seed=3407,
                folds=3,
                backend=_RecordingTeacherBackend(prediction),
            )


def test_teacher_impossible_group_split_fails_without_in_sample_fallback() -> None:
    rows = _teacher_rows().loc[lambda frame: frame["pitcher_id"].eq(frame["pitcher_id"].iloc[0])]
    backend = _RecordingTeacherBackend()

    with pytest.raises(TeacherError, match="disjoint.*train.*validation|split"):
        crossfit_teacher(rows, seed=3407, folds=3, backend=backend)

    assert backend.fit_calls == []
    assert backend.predict_calls == []


def test_catboost_teacher_backend_has_sealed_config_and_records_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[object] = []

    class FakeCatBoostClassifier:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.fit_kwargs: dict[str, object] | None = None
            instances.append(self)

        def fit(self, features: pd.DataFrame, target: pd.Series, **kwargs: object) -> object:
            self.fit_kwargs = kwargs
            return self

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.column_stack((np.full(len(features), 0.4), np.full(len(features), 0.6)))

    monkeypatch.setitem(
        sys.modules,
        "catboost",
        types.SimpleNamespace(CatBoostClassifier=FakeCatBoostClassifier),
    )
    backend = CatBoostTeacherBackend(device="GPU")
    config = backend.config
    with pytest.raises(TypeError):
        config["iterations"] = 1  # type: ignore[index]
    features = _teacher_rows().iloc[:2].loc[:, TEACHER_FEATURES]
    model = backend.fit(features, pd.Series([0, 1]), seed=3410)
    prediction = backend.predict(model, features)

    assert prediction.tolist() == pytest.approx([0.6, 0.6])
    assert instances[0].kwargs["task_type"] == "GPU"  # type: ignore[attr-defined]
    assert instances[0].kwargs["random_seed"] == 3410  # type: ignore[attr-defined]
    assert "early_stopping_rounds" not in instances[0].kwargs  # type: ignore[attr-defined]
    assert instances[0].fit_kwargs == {}  # type: ignore[attr-defined]
    assert backend.evidence["device"] == "GPU"
    assert backend.evidence["fit_seeds"] == (3410,)
    with pytest.raises(TeacherError, match="device"):
        CatBoostTeacherBackend(device="gpu")


def test_teacher_oof_is_detached_and_revalidates_on_access() -> None:
    rows = _teacher_rows()
    oof = crossfit_teacher(rows, seed=3407, folds=3, backend=_RecordingTeacherBackend())
    probability = oof.probability
    probability.iloc[0] = -1.0
    folds = oof.fold_by_row_id
    with pytest.raises(TypeError):
        folds[rows.iloc[0]["row_id"]] = 99  # type: ignore[index]

    assert oof.probability.iloc[0] >= 0.0
    with pytest.raises(TypeError, match="crossfit_teacher"):
        type(oof)()

    with pytest.raises(TeacherError, match="duplicate"):
        build_teacher_vector([rows.iloc[0]["row_id"]] * 2, oof)

    object.__setattr__(oof, "_metadata_json", 1)
    with pytest.raises(TeacherError, match="integrity|invalid"):
        _ = oof.metadata
