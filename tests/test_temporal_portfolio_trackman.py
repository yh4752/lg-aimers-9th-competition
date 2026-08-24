from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import experiments.temporal_portfolio.trackman_pitcher as pitcher_module
from experiments.independent_dl.feature_sources.trackman import (
    PITCHER_LOOKUP_COLUMNS,
    PITCH_GROUPS,
    TrackmanBuildResult,
)
from experiments.temporal_portfolio.trackman_pitcher import (
    P0,
    P1_PREFIXES,
    P2_EXACT,
    P2_PREFIXES,
    P3_PREFIXES,
    PitcherTrackmanError,
    PitcherTrackmanState,
    fit_pitcher_trackman,
    select_columns,
)
from experiments.temporal_portfolio.trackman_batter import (
    BATTER_EXPOSURE_COLUMNS,
    BATTER_MAPPING_COLUMNS,
    BatterTrackmanError,
    BatterTrackmanState,
    attach_batter_exposure,
    build_matchup_features,
    coverage_status,
    fit_batter_trackman,
)


def _main() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "pitcher_id": [11, 12, 11, 12],
            "pitcher_hand": [1, 2, 1, 2],
            "pitcher_team_id": [7, 8, 7, 8],
            "asof_pitcher_n": [99, 79, 100, 80],
            "asof_pitcher_fastball_rate": [0.50, 0.25, 0.01, 0.98],
            "asof_pitcher_breaking_rate": [0.30, 0.50, 0.01, 0.01],
            "asof_pitcher_offspeed_rate": [0.20, 0.25, 0.98, 0.01],
        }
    )


def _history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        (101, "Left", "A", (50, 30, 20), 145.0),
        (102, "Right", "B", (20, 40, 20), 151.0),
    )
    for pitcher_id, hand, team, counts, speed in specifications:
        groups = (
            ["fastball"] * counts[0]
            + ["breaking"] * counts[1]
            + ["offspeed"] * counts[2]
        )
        for index, group in enumerate(groups):
            rows.append(
                {
                    "season": 2023,
                    "pitcher_trackman_id": pitcher_id,
                    "pitch_type_group": group,
                    "pitcher_hand": hand,
                    "pitcher_team": team,
                    "rel_speed": speed - (2.0 if group != "fastball" else 0.0),
                    "spin_rate": 2100.0 + index,
                    "induced_vert_break": 15.0,
                    "horz_break": -3.0 if hand == "Left" else 3.0,
                    "extension": 1.8,
                    "rel_height": 1.7,
                    "rel_side": -0.2 if hand == "Left" else 0.2,
                    "zone_speed": speed - 12.0,
                }
            )
    future = dict(rows[0])
    future.update(season=2024, rel_speed=999.0, spin_rate=9999.0)
    rows.append(future)
    return pd.DataFrame(rows)


def _batter_main() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2023, 2022, 2023, 2022, 2023, 2024],
            "batter_id": [21, 21, 22, 22, 23, 23, 21],
            "batter_hand": [1, 1, 2, 2, 1, 1, 2],
            "batter_team_id": [7, 7, 8, 8, 9, 9, 99],
            "asof_batter_n": [19, 49, 29, 79, 24, 59, 9999],
        }
    )


def _batter_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        (201, "Left", "A", 50, 145.0),
        (202, "Right", "B", 80, 151.0),
        (203, "Left", "C", 60, 142.0),
        (204, "Left", "C", 60, 142.0),
    )
    groups = ("fastball", "breaking", "offspeed", "other")
    for batter_trackman_id, hand, team, count, speed in specifications:
        for index in range(count):
            group = groups[index % len(groups)]
            rows.append(
                {
                    "season": 2023,
                    "batter_trackman_id": batter_trackman_id,
                    "pitch_type_group": group,
                    "batter_hand": hand,
                    "batter_team": team,
                    "rel_speed": speed - float(index % 3),
                    "spin_rate": 2100.0 + index,
                    "induced_vert_break": 15.0 + (index % 2),
                    "horz_break": -3.0 if hand == "Left" else 3.0,
                }
            )
    future = dict(rows[0])
    future.update(season=2024, rel_speed=999.0, spin_rate=9999.0)
    rows.append(future)
    return pd.DataFrame(rows)


def _matchup_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3"],
            "pitcher_id": [11, 12, 999],
            "batter_id": [21, 22, 999],
            "pitcher_hand": [1, 2, 1],
            "batter_hand": [1, 1, 2],
        },
        index=[8, 3, 5],
    )


def _expected_columns() -> dict[str, tuple[str, ...]]:
    columns = tuple(PITCHER_LOOKUP_COLUMNS)
    p1 = tuple(
        column for column in columns if column.startswith(P1_PREFIXES)
    )
    p2 = tuple(
        column
        for column in columns
        if column in P2_EXACT or column.startswith(P2_PREFIXES)
    )
    p3 = tuple(
        column for column in columns if column.startswith(P3_PREFIXES)
    )
    return {
        "P0": ("pitcher_id", *P0),
        "P1": ("pitcher_id", *p1),
        "P2": ("pitcher_id", *p2),
        "P3": ("pitcher_id", *p3),
    }


def _portfolio_pitcher_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _main().copy(deep=True)
    frame["row_id"] = ["p0", "p1", "v0", "v1"]
    frame["control_success"] = [1, 0, 1, 0]
    frame["batter_id"] = [21, 22, 21, 22]
    frame["batter_hand"] = [1, 2, 1, 2]
    frame["batter_team_id"] = [9, 10, 9, 10]
    return (
        frame.loc[frame["season"].eq(2023)].copy(deep=True),
        frame.loc[frame["season"].eq(2024)].copy(deep=True),
    )


def _empty_valid_result(cutoff_year: int = 2023) -> TrackmanBuildResult:
    lookup = pd.DataFrame(columns=PITCHER_LOOKUP_COLUMNS)
    return TrackmanBuildResult(
        cutoff_year=cutoff_year,
        lookup=lookup,
        lookup_schema=PITCHER_LOOKUP_COLUMNS,
        mapping=pd.DataFrame(),
        team_mapping=pd.DataFrame(),
    )


def _forge_state(state: PitcherTrackmanState, **changes: object) -> PitcherTrackmanState:
    forged = object.__new__(PitcherTrackmanState)
    for field in PitcherTrackmanState.__dataclass_fields__:
        object.__setattr__(forged, field, object.__getattribute__(state, field))
    for field, value in changes.items():
        object.__setattr__(forged, field, value)
    return forged


def test_pitcher_trackman_exact_partition_is_complete_and_disjoint() -> None:
    state = fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)
    expected = _expected_columns()

    assert isinstance(state.bundles, Mapping)
    assert tuple(state.bundles) == ("P0", "P1", "P2", "P3")
    assert {name: tuple(frame.columns) for name, frame in state.bundles.items()} == expected
    non_keys = [set(columns) - {"pitcher_id"} for columns in expected.values()]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(non_keys)
        for right in non_keys[index + 1 :]
    )
    assert set().union(*non_keys) == set(PITCHER_LOOKUP_COLUMNS) - {"pitcher_id"}
    assert all(frame.columns.is_unique for frame in state.bundles.values())
    assert tuple(select_columns(state.lookup, exact=P0).columns) == expected["P0"]


@pytest.mark.parametrize("empty_lookup", [False, True])
def test_portfolio_trackman_cache_roundtrip_preserves_exact_state_and_batches(
    tmp_path, monkeypatch: pytest.MonkeyPatch, empty_lookup: bool
) -> None:
    from experiments.temporal_portfolio.feature_cache import materialize_fold_cache
    from experiments.temporal_portfolio.features import PortfolioFeatureSpec

    if empty_lookup:
        monkeypatch.setattr(
            pitcher_module,
            "build_trackman_lookup",
            lambda *_args, **_kwargs: _empty_valid_result(),
        )
    train, valid = _portfolio_pitcher_frames()
    kwargs = dict(
        train=train,
        valid=valid,
        history=_history(),
        spec=PortfolioFeatureSpec(("base", "P0", "P2"), "dl_standard"),
        valid_year=2024,
    )

    fresh = materialize_fold_cache(tmp_path / "cache", **kwargs)
    reused = materialize_fold_cache(tmp_path / "cache", **kwargs)

    assert fresh.reused is False
    assert reused.reused is True
    assert fresh.state.source_hashes == reused.state.source_hashes
    pitcher = reused.state.fitted_sources["pitcher"]
    assert pitcher.lookup_sha256 == reused.state.source_hashes["pitcher"]
    assert pitcher.lookup.empty is empty_lookup
    np.testing.assert_array_equal(fresh.train.x_num, reused.train.x_num)
    np.testing.assert_array_equal(fresh.train.x_cat, reused.train.x_cat)
    np.testing.assert_array_equal(fresh.valid.x_num, reused.valid.x_num)
    np.testing.assert_array_equal(fresh.valid.x_cat, reused.valid.x_cat)


def test_pitcher_trackman_is_cutoff_bound_under_future_source_mutation() -> None:
    main = _main()
    history = _history()
    expected = fit_pitcher_trackman(main, history, cutoff_year=2023)
    changed_main = main.copy(deep=True)
    changed_history = history.copy(deep=True)
    changed_main.loc[changed_main["season"].gt(2023), "asof_pitcher_n"] = 999999
    changed_main.loc[changed_main["season"].gt(2023), "asof_pitcher_fastball_rate"] = 0.0
    changed_history.loc[changed_history["season"].gt(2023), "rel_speed"] = -9999.0
    changed_history.loc[changed_history["season"].gt(2023), "spin_rate"] = 1.0

    replay = fit_pitcher_trackman(changed_main, changed_history, cutoff_year=2023)

    assert replay.lookup_sha256 == expected.lookup_sha256
    assert dict(replay.bundle_sha256) == dict(expected.bundle_sha256)
    pd.testing.assert_frame_equal(replay.lookup, expected.lookup)
    for name in expected.bundles:
        pd.testing.assert_frame_equal(replay.bundles[name], expected.bundles[name])


def test_pitcher_trackman_lookup_and_identity_are_row_order_deterministic() -> None:
    expected = fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)
    replay = fit_pitcher_trackman(
        _main().sample(frac=1.0, random_state=7).reset_index(drop=True),
        _history().sample(frac=1.0, random_state=11).reset_index(drop=True),
        cutoff_year=2023,
    )

    assert expected.lookup["pitcher_id"].tolist() == [11, 12]
    pd.testing.assert_frame_equal(expected.lookup, replay.lookup)
    assert expected.lookup_sha256 == replay.lookup_sha256
    assert dict(expected.bundle_sha256) == dict(replay.bundle_sha256)


@pytest.mark.parametrize(
    ("column", "conflicting_value"),
    [
        ("pitcher_hand", 2),
        ("pitcher_team_id", 8),
        ("asof_pitcher_fastball_rate", 0.25),
        ("asof_pitcher_breaking_rate", 0.50),
        ("asof_pitcher_offspeed_rate", 0.25),
    ],
)
def test_pitcher_trackman_rejects_conflicting_maximum_main_signatures_in_any_order(
    column: str, conflicting_value: object
) -> None:
    main = _main()
    tied = main.iloc[[0]].copy(deep=True)
    assert tied.iloc[0]["asof_pitcher_n"] == main.iloc[0]["asof_pitcher_n"]
    tied.loc[:, column] = conflicting_value
    conflicting = pd.concat([main, tied], ignore_index=True)

    for ordered in (
        conflicting,
        conflicting.iloc[::-1].reset_index(drop=True),
    ):
        with pytest.raises(
            PitcherTrackmanError,
            match="conflicting maximum.*signature",
        ):
            fit_pitcher_trackman(ordered, _history(), cutoff_year=2023)


@pytest.mark.parametrize("nan_rate", [False, True])
def test_pitcher_trackman_allows_equivalent_maximum_main_signatures(
    nan_rate: bool,
) -> None:
    main = _main()
    if nan_rate:
        main.loc[0, "asof_pitcher_fastball_rate"] = np.nan
    tied = main.iloc[[0]].copy(deep=True)
    duplicated = pd.concat([main, tied], ignore_index=True)

    expected = fit_pitcher_trackman(duplicated, _history(), cutoff_year=2023)
    shuffled = fit_pitcher_trackman(
        duplicated.iloc[::-1].reset_index(drop=True),
        _history(),
        cutoff_year=2023,
    )

    pd.testing.assert_frame_equal(expected.lookup, shuffled.lookup)
    assert expected.lookup_sha256 == shuffled.lookup_sha256
    assert dict(expected.bundle_sha256) == dict(shuffled.bundle_sha256)


def test_pitcher_trackman_ignores_future_maximum_signature_conflicts() -> None:
    main = _main()
    future_tie = main.loc[
        main["season"].eq(2024) & main["pitcher_id"].eq(11)
    ].copy(deep=True)
    future_tie.loc[:, "pitcher_hand"] = 2
    future_tie.loc[:, "pitcher_team_id"] = 8
    future_tie.loc[:, "asof_pitcher_fastball_rate"] = 0.25
    future_tie.loc[:, "asof_pitcher_breaking_rate"] = 0.50
    future_tie.loc[:, "asof_pitcher_offspeed_rate"] = 0.25
    conflicting = pd.concat([main, future_tie], ignore_index=True)

    expected = fit_pitcher_trackman(main, _history(), cutoff_year=2023)
    replay = fit_pitcher_trackman(
        conflicting.iloc[::-1].reset_index(drop=True),
        _history(),
        cutoff_year=2023,
    )

    pd.testing.assert_frame_equal(expected.lookup, replay.lookup)
    assert expected.lookup_sha256 == replay.lookup_sha256


def test_pitcher_trackman_state_and_exposed_frames_are_immutable() -> None:
    state = fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)
    expected_lookup = state.lookup
    expected_p0 = state.bundles["P0"]

    with pytest.raises(FrozenInstanceError):
        state.cutoff_year = 2024  # type: ignore[misc]
    with pytest.raises(TypeError):
        state.bundles["P4"] = expected_p0  # type: ignore[index]
    with pytest.raises(TypeError):
        state.bundle_sha256["P0"] = "0" * 64  # type: ignore[index]

    exposed_lookup = state.lookup
    exposed_lookup.iloc[0, 0] = 999
    exposed = state.bundles["P0"]
    exposed.iloc[0, 0] = 999
    exposed.drop(columns=exposed.columns[-1], inplace=True)

    pd.testing.assert_frame_equal(state.lookup, expected_lookup)
    pd.testing.assert_frame_equal(state.bundles["P0"], expected_p0)
    assert state.lookup is not state.lookup
    assert state.bundles is not state.bundles


def test_pitcher_trackman_state_requires_the_fit_factory_and_detects_forgery() -> None:
    with pytest.raises(TypeError, match="fit_pitcher_trackman"):
        PitcherTrackmanState()  # type: ignore[call-arg]

    state = fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)
    rows = object.__getattribute__(state, "_lookup_rows")
    changed = list(rows[0])
    changed[-1] = 12345.0
    forged = _forge_state(state, _lookup_rows=(tuple(changed), *rows[1:]))

    with pytest.raises(PitcherTrackmanError, match="state"):
        _ = forged.lookup
    with pytest.raises(PitcherTrackmanError, match="state"):
        _ = forged.bundles


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_lookup_sha256", "z" * 64),
        ("_bundle_sha256_items", ("malformed",)),
    ],
)
def test_pitcher_trackman_state_normalizes_forged_identity_payloads(
    field: str, value: object
) -> None:
    state = fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)

    with pytest.raises(PitcherTrackmanError, match="state"):
        _ = _forge_state(state, **{field: value}).lookup


@pytest.mark.parametrize("cutoff_year", [True, 2023.0, np.int64(2023), 999, 10000, "2023"])
def test_pitcher_trackman_cutoff_must_be_an_exact_four_digit_int(cutoff_year: object) -> None:
    with pytest.raises(PitcherTrackmanError, match="cutoff_year"):
        fit_pitcher_trackman(_main(), _history(), cutoff_year=cutoff_year)  # type: ignore[arg-type]


@pytest.mark.parametrize("which", ["train", "history"])
def test_pitcher_trackman_requires_actual_dataframes_and_unique_string_columns(which: str) -> None:
    train: object = _main()
    history: object = _history()
    if which == "train":
        train = _main().to_dict("list")
    else:
        history = _history().to_dict("list")
    with pytest.raises(PitcherTrackmanError, match="DataFrame"):
        fit_pitcher_trackman(train, history, cutoff_year=2023)  # type: ignore[arg-type]

    frame = _main() if which == "train" else _history()
    duplicate = pd.concat([frame, frame.iloc[:, [0]]], axis=1)
    duplicate.columns = [*frame.columns, frame.columns[0]]
    with pytest.raises(PitcherTrackmanError, match="duplicate columns"):
        fit_pitcher_trackman(
            duplicate if which == "train" else _main(),
            duplicate if which == "history" else _history(),
            cutoff_year=2023,
        )

    renamed = frame.copy(deep=True)
    renamed.columns = [7, *renamed.columns[1:]]
    with pytest.raises(PitcherTrackmanError, match="column names"):
        fit_pitcher_trackman(
            renamed if which == "train" else _main(),
            renamed if which == "history" else _history(),
            cutoff_year=2023,
        )


@pytest.mark.parametrize(
    ("which", "column"),
    [("train", "pitcher_id"), ("history", "pitcher_trackman_id"), ("history", "rel_speed")],
)
def test_pitcher_trackman_reports_missing_required_schema(which: str, column: str) -> None:
    train = _main()
    history = _history()
    if which == "train":
        train = train.drop(columns=column)
    else:
        history = history.drop(columns=column)

    with pytest.raises(PitcherTrackmanError, match=f"required columns.*{column}"):
        fit_pitcher_trackman(train, history, cutoff_year=2023)


@pytest.mark.parametrize("bad_season", [None, np.nan, np.inf, 2023.5, True, "2023"])
def test_pitcher_trackman_rejects_malformed_seasons(bad_season: object) -> None:
    train = _main().copy(deep=True)
    train.loc[train.index[0], "season"] = bad_season

    with pytest.raises(PitcherTrackmanError, match="season"):
        fit_pitcher_trackman(train, _history(), cutoff_year=2023)


@pytest.mark.parametrize("empty_side", ["train", "history", "both"])
def test_pitcher_trackman_fails_closed_without_both_sources_through_cutoff(empty_side: str) -> None:
    train = _main()
    history = _history()
    if empty_side in {"train", "both"}:
        train = train.assign(season=2024)
    if empty_side in {"history", "both"}:
        history = history.assign(season=2024)

    with pytest.raises(PitcherTrackmanError, match="through cutoff"):
        fit_pitcher_trackman(train, history, cutoff_year=2023)


def test_pitcher_trackman_represents_no_accepted_mappings_with_empty_fixed_bundles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pitcher_module,
        "build_trackman_lookup",
        lambda *_args, **_kwargs: _empty_valid_result(),
    )

    state = fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)

    assert state.lookup.empty
    assert tuple(state.lookup.columns) == PITCHER_LOOKUP_COLUMNS
    assert all(frame.empty for frame in state.bundles.values())
    assert {
        name: tuple(frame.columns) for name, frame in state.bundles.items()
    } == _expected_columns()


def test_pitcher_trackman_ambiguous_candidates_may_produce_no_accepted_mapping() -> None:
    main = _main().iloc[[0]].copy(deep=True)
    main["asof_pitcher_fastball_rate"] = 0.44
    main["asof_pitcher_breaking_rate"] = 0.36
    history = _history()
    first = history.loc[
        history["season"].eq(2023) & history["pitcher_trackman_id"].eq(101)
    ].copy(deep=True)
    second = first.assign(pitcher_trackman_id=102)
    history = pd.concat([first, second], ignore_index=True)

    state = fit_pitcher_trackman(main, history, cutoff_year=2023)

    assert state.lookup.empty
    assert all(frame.empty for frame in state.bundles.values())


@pytest.mark.parametrize(
    "failure", [KeyError("missing"), ValueError("bad"), TypeError("bad")]
)
def test_pitcher_trackman_normalizes_legacy_builder_errors(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(pitcher_module, "build_trackman_lookup", fail)

    with pytest.raises(PitcherTrackmanError, match="construction failed") as caught:
        fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "corruption", ["wrong_type", "mutated_lookup", "wrong_cutoff", "wrong_schema"]
)
def test_pitcher_trackman_normalizes_corrupt_legacy_results(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    result: object = _empty_valid_result()
    if corruption == "wrong_type":
        result = object()
    elif corruption == "mutated_lookup":
        assert isinstance(result, TrackmanBuildResult)
        result.lookup["pitcher_id"] = [11]
    elif corruption == "wrong_cutoff":
        result = _empty_valid_result(2022)
    else:
        result = TrackmanBuildResult(
            cutoff_year=2023,
            lookup=pd.DataFrame(columns=PITCHER_LOOKUP_COLUMNS),
            lookup_schema=tuple(PITCHER_LOOKUP_COLUMNS[:-1]),
            mapping=pd.DataFrame(),
            team_mapping=pd.DataFrame(),
        )
    monkeypatch.setattr(pitcher_module, "build_trackman_lookup", lambda *_args, **_kwargs: result)

    with pytest.raises(PitcherTrackmanError, match="construction failed"):
        fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)


def test_pitcher_trackman_allows_nan_aggregates_but_rejects_infinity() -> None:
    history = _history()
    history.loc[history["season"].le(2023), "extension"] = np.nan

    state = fit_pitcher_trackman(_main(), history, cutoff_year=2023)

    assert state.bundles["P1"].filter(like="extension").isna().all().all()

    infinite = _history()
    infinite.loc[infinite.index[0], "rel_speed"] = np.inf
    with pytest.raises(PitcherTrackmanError, match="rel_speed.*infinity"):
        fit_pitcher_trackman(_main(), infinite, cutoff_year=2023)


def test_pitcher_trackman_rejects_nonfinite_legacy_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _empty_valid_result()
    row = {column: np.nan for column in PITCHER_LOOKUP_COLUMNS}
    row["pitcher_id"] = 11
    row["tm_match_cost"] = np.inf
    lookup = pd.DataFrame([row], columns=PITCHER_LOOKUP_COLUMNS)
    result = TrackmanBuildResult(
        2023,
        lookup,
        PITCHER_LOOKUP_COLUMNS,
        pd.DataFrame(),
        pd.DataFrame(),
    )
    monkeypatch.setattr(pitcher_module, "build_trackman_lookup", lambda *_args, **_kwargs: result)

    with pytest.raises(PitcherTrackmanError, match="infinity"):
        fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)


def _forge_batter_state(
    state: BatterTrackmanState, **changes: object
) -> BatterTrackmanState:
    forged = object.__new__(BatterTrackmanState)
    for field in BatterTrackmanState.__dataclass_fields__:
        object.__setattr__(forged, field, object.__getattribute__(state, field))
    for field, value in changes.items():
        object.__setattr__(forged, field, value)
    return forged


def test_batter_mapping_is_one_to_one_and_ambiguous_candidates_stay_unmatched() -> None:
    state = fit_batter_trackman(_batter_main(), _batter_history(), cutoff_year=2023)
    accepted = state.mapping.loc[state.mapping["tm_batter_match_accepted"].eq(1)]

    assert accepted["batter_id"].is_unique
    assert accepted["batter_trackman_id"].is_unique
    assert set(accepted["batter_id"]) == {21, 22}
    assert state.mapping.loc[
        state.mapping["batter_id"].eq(23), "tm_batter_match_accepted"
    ].item() == 0
    assert state.coverage == pytest.approx(2 / 3)
    assert state.status == "eligible"


def test_batter_exposure_and_matchup_have_fixed_schema_and_preserve_rows() -> None:
    batter_state = fit_batter_trackman(
        _batter_main(), _batter_history(), cutoff_year=2023
    )
    pitcher_state = fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)
    rows = _matchup_rows()

    exposure = attach_batter_exposure(rows, batter_state)
    matchup = build_matchup_features(
        rows, pitcher_state=pitcher_state, batter_state=batter_state
    )

    assert exposure.index.equals(rows.index)
    assert matchup.index.equals(rows.index)
    assert exposure["row_id"].tolist() == rows["row_id"].tolist()
    assert matchup["row_id"].tolist() == rows["row_id"].tolist()
    assert tuple(exposure.columns[-(len(BATTER_EXPOSURE_COLUMNS) - 1) :]) == tuple(
        column for column in BATTER_EXPOSURE_COLUMNS if column != "batter_id"
    )
    assert any(column.startswith("tm_batter_seen_fastball_") for column in exposure)
    assert any(column.startswith("tm_matchup_fastball_") for column in matchup)
    assert "tm_matchup_left_left" in matchup
    assert matchup.loc[8, "tm_matchup_left_left"] == 1
    assert matchup.loc[3, "tm_matchup_right_left"] == 1


def test_batter_exposure_contains_required_counts_confidence_rates_and_moments() -> None:
    state = fit_batter_trackman(_batter_main(), _batter_history(), cutoff_year=2023)
    columns = set(state.exposure.columns)

    assert {
        "tm_batter_seen_history_n",
        "tm_batter_seen_recent_n",
        "tm_batter_match_confidence",
        "tm_batter_match_missing",
    }.issubset(columns)
    for group in ("fastball", "breaking", "offspeed", "other"):
        assert f"tm_batter_seen_{group}_rate" in columns
    for metric in ("rel_speed", "spin_rate", "induced_vert_break", "horz_break"):
        assert f"tm_batter_seen_{metric}_mean" in columns
        assert f"tm_batter_seen_{metric}_std" in columns


def test_batter_trackman_is_cutoff_bound_and_row_order_deterministic() -> None:
    main = _batter_main()
    history = _batter_history()
    expected = fit_batter_trackman(main, history, cutoff_year=2023)
    changed_main = main.copy(deep=True)
    changed_history = history.copy(deep=True)
    changed_main.loc[changed_main["season"].gt(2023), "asof_batter_n"] = 1
    changed_history.loc[changed_history["season"].gt(2023), "rel_speed"] = -9999.0

    replay = fit_batter_trackman(
        changed_main.sample(frac=1.0, random_state=3).reset_index(drop=True),
        changed_history.sample(frac=1.0, random_state=5).reset_index(drop=True),
        cutoff_year=2023,
    )

    pd.testing.assert_frame_equal(replay.mapping, expected.mapping)
    pd.testing.assert_frame_equal(replay.exposure, expected.exposure)
    assert replay.mapping_sha256 == expected.mapping_sha256
    assert replay.exposure_sha256 == expected.exposure_sha256


def test_batter_recent_exposure_is_bound_to_requested_cutoff_year() -> None:
    main = _batter_main().loc[lambda frame: frame["season"].le(2023)].copy()
    history = _batter_history()

    state = fit_batter_trackman(main, history, cutoff_year=2024)
    exposure = state.exposure.set_index("batter_id")

    assert exposure.loc[21, "tm_batter_seen_history_n"] == 51
    assert exposure.loc[21, "tm_batter_seen_recent_n"] == 1


def test_batter_trackman_rejects_conflicting_tied_maximum_signatures() -> None:
    main = _batter_main()
    tied = main.loc[main["batter_id"].eq(21) & main["season"].eq(2023)].copy()
    tied.loc[:, "batter_team_id"] = 99
    conflict = pd.concat([main, tied], ignore_index=True)

    for ordered in (conflict, conflict.iloc[::-1].reset_index(drop=True)):
        with pytest.raises(BatterTrackmanError, match="conflicting maximum.*signature"):
            fit_batter_trackman(ordered, _batter_history(), cutoff_year=2023)


def test_batter_trackman_exact_hand_gate_and_low_coverage_status() -> None:
    main = _batter_main().copy(deep=True)
    main["batter_hand"] = 1
    history = _batter_history().copy(deep=True)
    history["batter_hand"] = "Right"
    state = fit_batter_trackman(main, history, cutoff_year=2023)

    assert state.mapping["tm_batter_match_accepted"].eq(0).all()
    assert state.exposure["tm_batter_match_missing"].eq(1).all()
    assert state.coverage == 0.0
    assert state.status == "insufficient_mapping"


@pytest.mark.parametrize(
    ("coverage", "expected"),
    [
        (0.0, "insufficient_mapping"),
        (0.299999, "insufficient_mapping"),
        (0.30, "exploratory"),
        (0.599999, "exploratory"),
        (0.60, "eligible"),
        (1.0, "eligible"),
    ],
)
def test_batter_mapping_coverage_status_thresholds(
    coverage: float, expected: str
) -> None:
    assert coverage_status(coverage) == expected


def test_batter_zero_match_state_has_fixed_schema() -> None:
    main = _batter_main().copy(deep=True)
    main["batter_hand"] = 1
    history = _batter_history().copy(deep=True)
    history["batter_hand"] = "Right"
    state = fit_batter_trackman(main, history, cutoff_year=2023)

    assert tuple(state.mapping.columns) == BATTER_MAPPING_COLUMNS
    assert tuple(state.exposure.columns) == BATTER_EXPOSURE_COLUMNS
    assert len(state.mapping) == len(state.exposure) == 3
    assert state.exposure.filter(like="_mean").isna().all().all()


def test_batter_state_is_immutable_defensive_and_detects_forgery() -> None:
    with pytest.raises(TypeError, match="fit_batter_trackman"):
        BatterTrackmanState()  # type: ignore[call-arg]
    state = fit_batter_trackman(_batter_main(), _batter_history(), cutoff_year=2023)
    expected_mapping = state.mapping
    expected_exposure = state.exposure

    with pytest.raises(FrozenInstanceError):
        state.cutoff_year = 2024  # type: ignore[misc]
    changed_mapping = state.mapping
    changed_mapping.iloc[0, 0] = 999
    changed_exposure = state.exposure
    changed_exposure.iloc[0, -1] = 999.0
    pd.testing.assert_frame_equal(state.mapping, expected_mapping)
    pd.testing.assert_frame_equal(state.exposure, expected_exposure)

    rows = object.__getattribute__(state, "_exposure_rows")
    changed = list(rows[0])
    changed[-1] = 12345.0
    forged = _forge_batter_state(state, _exposure_rows=(tuple(changed), *rows[1:]))
    with pytest.raises(BatterTrackmanError, match="state"):
        _ = forged.exposure

    mapping_rows = object.__getattribute__(state, "_mapping_rows")
    reordered = _forge_batter_state(state, _mapping_rows=tuple(reversed(mapping_rows)))
    with pytest.raises(BatterTrackmanError, match="state"):
        _ = reordered.mapping


def test_matchup_is_invariant_to_other_evaluation_rows_and_current_pitch_values() -> None:
    batter_state = fit_batter_trackman(
        _batter_main(), _batter_history(), cutoff_year=2023
    )
    pitcher_state = fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)
    rows = _matchup_rows().assign(rel_speed=[1.0, 2.0, 3.0], spin_rate=[4.0, 5.0, 6.0])
    expected = build_matchup_features(
        rows, pitcher_state=pitcher_state, batter_state=batter_state
    ).set_index("row_id")
    changed = rows.copy(deep=True)
    changed.loc[changed["row_id"].ne("r1"), :] = changed.loc[
        changed["row_id"].ne("r1"), :
    ].assign(pitcher_id=12, batter_id=22, rel_speed=9999.0, spin_rate=-9999.0)
    changed = changed.iloc[::-1]
    replay = build_matchup_features(
        changed, pitcher_state=pitcher_state, batter_state=batter_state
    ).set_index("row_id")

    pd.testing.assert_series_equal(
        expected.loc["r1"], replay.loc["r1"], check_names=False
    )


@pytest.mark.parametrize("composition", ["B1", "P2", "B1+P2"])
def test_matchup_composes_after_b1_and_p2_without_trusting_attached_values(
    composition: str,
) -> None:
    batter_state = fit_batter_trackman(
        _batter_main(), _batter_history(), cutoff_year=2023
    )
    pitcher_state = fit_pitcher_trackman(_main(), _history(), cutoff_year=2023)
    rows = _matchup_rows()
    if "B1" in composition:
        rows = attach_batter_exposure(rows, batter_state)
    if "P2" in composition:
        p2 = pitcher_state.bundles["P2"].set_index("pitcher_id")
        for column in p2.columns:
            rows[column] = rows["pitcher_id"].map(p2[column])
    original = rows.copy(deep=True)

    result = build_matchup_features(
        rows, pitcher_state=pitcher_state, batter_state=batter_state
    )

    assert result.index.equals(original.index)
    assert result.columns.is_unique
    assert tuple(result.columns[: len(original.columns)]) == tuple(original.columns)
    pd.testing.assert_frame_equal(result.loc[:, original.columns], original)
    added = tuple(result.columns[len(original.columns) :])
    assert added
    assert all(column.startswith("tm_matchup_") for column in added)

    changed = original.copy(deep=True)
    source_columns = [
        column
        for column in changed
        if column.startswith("tm_batter_")
        or column.startswith("tm_history_")
        or (
            column.startswith("tm_")
            and any(f"_{group}_" in column for group in PITCH_GROUPS)
        )
    ]
    changed.loc[:, source_columns] = -9999.0
    replay = build_matchup_features(
        changed, pitcher_state=pitcher_state, batter_state=batter_state
    )

    pd.testing.assert_frame_equal(
        result.loc[:, list(added)], replay.loc[:, list(added)]
    )


@pytest.mark.parametrize("which", ["train", "history", "rows"])
def test_batter_trackman_normalizes_input_failures(which: str) -> None:
    main: object = _batter_main()
    history: object = _batter_history()
    if which == "train":
        main = _batter_main().drop(columns="batter_hand")
    elif which == "history":
        history = _batter_history().assign(batter_trackman_id=np.inf)
    else:
        state = fit_batter_trackman(_batter_main(), _batter_history(), cutoff_year=2023)
        with pytest.raises(BatterTrackmanError):
            attach_batter_exposure({"batter_id": [21]}, state)  # type: ignore[arg-type]
        return

    with pytest.raises(BatterTrackmanError):
        fit_batter_trackman(main, history, cutoff_year=2023)  # type: ignore[arg-type]
