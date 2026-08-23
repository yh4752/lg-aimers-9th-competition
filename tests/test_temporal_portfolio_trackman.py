from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import experiments.temporal_portfolio.trackman_pitcher as pitcher_module
from experiments.independent_dl.feature_sources.trackman import (
    PITCHER_LOOKUP_COLUMNS,
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
