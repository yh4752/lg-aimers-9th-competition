from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from experiments.temporal_portfolio.seasonal_features import (
    S1State,
    SeasonalFeatureError,
    fit_s1_state,
    transform_s1,
)


def _s1_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["h0", "h1", "h2", "v0", "v1", "v1"],
            "season": [2022, 2023, 2023, 2024, 2024, 2024],
            "pitcher_id": [11, 11, 12, 11, 12, 999],
            "batter_id": [21, 21, 22, 21, 22, 999],
            "asof_pitcher_n": [8, 9, 4, 10, 5, 0],
            "asof_pitcher_success_rate": [0.50, 5 / 9, 0.50, 0.60, 0.60, 0.50],
            "asof_pitcher_reverse_rate": [0.10, 1 / 9, 0.25, 0.20, 0.20, 0.0],
            "asof_pitcher_middle_rate": [0.25, 2 / 9, 0.25, 0.30, 0.20, 0.0],
            "asof_pitcher_ball_rate": [0.40, 4 / 9, 0.50, 0.40, 0.40, 0.0],
            "asof_pitcher_strike_rate": [0.60, 5 / 9, 0.50, 0.60, 0.60, 0.0],
            "asof_pitcher_pitchmix_n": [8, 9, 4, 10, 5, 0],
            "asof_pitcher_fastball_rate": [0.50, 5 / 9, 0.50, 0.60, 0.40, 0.0],
            "asof_pitcher_breaking_rate": [0.25, 2 / 9, 0.25, 0.20, 0.40, 0.0],
            "asof_pitcher_offspeed_rate": [0.25, 2 / 9, 0.25, 0.20, 0.20, 0.0],
            "asof_pitcher_prev1_game_success_rate": [
                0.5,
                0.5,
                0.5,
                0.7,
                np.nan,
                np.nan,
            ],
            "asof_pitcher_prev3_game_success_rate": [0.5, 0.5, 0.5, 0.6, 0.5, np.nan],
            "asof_pitcher_prev5_game_success_rate": [0.5, 0.5, 0.5, 0.5, 0.4, np.nan],
            "asof_batter_n": [6, 7, 3, 8, 4, 0],
            "asof_batter_success_rate": [0.50, 4 / 7, 1 / 3, 0.625, 0.50, 0.50],
            "asof_batter_middle_rate": [1 / 6, 1 / 7, 1 / 3, 0.25, 0.25, 0.0],
            "control_success": [1, 0, 1, 1, 0, 1],
        },
        index=[30, 10, 20, 8, 8, 3],
    )


def _fit_and_valid() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _s1_frame()
    train = frame.loc[frame["season"].lt(2024)].copy(deep=True)
    valid = frame.loc[frame["season"].eq(2024)].drop(columns="control_success")
    return train, valid


def _by_row_position(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reset_index(drop=True)


def _forge_state(state: S1State, **changes: object) -> S1State:
    forged = object.__new__(S1State)
    for name in S1State.__dataclass_fields__:
        object.__setattr__(forged, name, object.__getattribute__(state, name))
    for name, value in changes.items():
        object.__setattr__(forged, name, value)
    return forged


def test_s1_uses_only_the_previous_season_snapshot() -> None:
    train, valid = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    first = transform_s1(valid, state)
    mutated_source = _s1_frame()
    mutated_source.loc[mutated_source["season"].eq(2024), "control_success"] = [0, 1, 0]
    replay_train = mutated_source.loc[mutated_source["season"].lt(2024)]
    second = transform_s1(valid, fit_s1_state(replay_train, valid_year=2024))

    pd.testing.assert_frame_equal(first, second)
    assert state.snapshot.cutoff_year == 2023 == state.valid_year - 1
    assert {
        "season_pitcher_n",
        "season_batter_n",
        "season_vs_career_success",
    }.issubset(first)


def test_s1_transform_is_separable_for_single_shuffle_and_subset() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    whole = transform_s1(rows, state)

    single = transform_s1(rows.iloc[[0]], state)
    pd.testing.assert_frame_equal(
        _by_row_position(whole.iloc[[0]]), _by_row_position(single)
    )

    order = [2, 0, 1]
    shuffled = transform_s1(rows.iloc[order], state)
    pd.testing.assert_frame_equal(
        _by_row_position(whole.iloc[order]), _by_row_position(shuffled)
    )

    subset = transform_s1(rows.iloc[[2, 1]], state)
    pd.testing.assert_frame_equal(
        _by_row_position(whole.iloc[[2, 1]]), _by_row_position(subset)
    )


def test_s1_preserves_duplicate_index_row_ids_order_count_and_caller() -> None:
    train, rows = _fit_and_valid()
    original = rows.copy(deep=True)

    result = transform_s1(rows, fit_s1_state(train, valid_year=2024))

    pd.testing.assert_frame_equal(rows, original)
    assert result.index.tolist() == [8, 8, 3]
    assert len(result) == len(rows)
    assert result.columns.is_unique
    assert not set(rows.columns) & set(result.columns)
    assert not any(column.startswith("snapshot_") for column in result)


def test_s1_output_has_only_finite_features_and_exact_categorical_metadata() -> None:
    train, rows = _fit_and_valid()

    result = transform_s1(rows, fit_s1_state(train, valid_year=2024))

    numeric = result.select_dtypes(include="number")
    assert np.isfinite(numeric.to_numpy()).all()
    assert result.attrs == {
        "categorical_columns": (
            "season_pitcher_n_bucket",
            "season_batter_n_bucket",
        )
    }
    assert type(result.attrs["categorical_columns"]) is tuple
    assert result.iloc[-1]["season_pitcher_n"] == pytest.approx(0.0)
    assert result.iloc[-1]["season_batter_n"] == pytest.approx(0.0)


def test_s1_fit_is_defensive_against_later_history_mutation() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    expected = transform_s1(rows, state)

    train.loc[:, "control_success"] = 1 - train["control_success"]
    train.loc[:, "asof_pitcher_success_rate"] = 0.0
    exposed = state.snapshot
    exposed.pitcher.loc[:, "snapshot_pitcher_success_count"] = 9999.0
    exposed.batter.loc[:, "snapshot_batter_success_count"] = 9999.0

    pd.testing.assert_frame_equal(expected, transform_s1(rows, state))
    assert not state.snapshot.pitcher["snapshot_pitcher_success_count"].eq(9999.0).any()
    assert not state.snapshot.batter["snapshot_batter_success_count"].eq(9999.0).any()


def test_s1_state_is_frozen_and_transform_requires_exact_state_type() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    with pytest.raises(FrozenInstanceError):
        state.valid_year = 2025  # type: ignore[misc]
    with pytest.raises(SeasonalFeatureError, match="state type"):
        transform_s1(rows, object())  # type: ignore[arg-type]

    class DerivedState(S1State):
        pass

    derived = object.__new__(DerivedState)
    with pytest.raises(SeasonalFeatureError, match="state type"):
        transform_s1(rows, derived)


@pytest.mark.parametrize(
    "valid_year", [True, 2024.0, np.int64(2024), 999, 10000, "2024"]
)
def test_s1_valid_year_must_be_an_exact_four_digit_int(valid_year: object) -> None:
    train, _ = _fit_and_valid()

    with pytest.raises(SeasonalFeatureError, match="valid_year"):
        fit_s1_state(train, valid_year=valid_year)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_season", [2023.5, np.nan, np.inf, True, "2023"])
def test_s1_fit_rejects_malformed_seasons(bad_season: object) -> None:
    train, _ = _fit_and_valid()
    train.iloc[0, train.columns.get_loc("season")] = bad_season

    with pytest.raises(SeasonalFeatureError, match="season"):
        fit_s1_state(train, valid_year=2024)


def test_s1_fit_rejects_validation_and_future_rows() -> None:
    frame = _s1_frame()

    with pytest.raises(SeasonalFeatureError, match="reach validation season"):
        fit_s1_state(frame, valid_year=2024)


def test_s1_fit_requires_rows_from_the_immediate_previous_season() -> None:
    train, _ = _fit_and_valid()
    earlier_only = train.loc[train["season"].eq(2022)]

    with pytest.raises(SeasonalFeatureError, match="immediate previous season"):
        fit_s1_state(earlier_only, valid_year=2024)


def test_s1_fit_does_not_require_each_entity_in_the_previous_season() -> None:
    train, _ = _fit_and_valid()
    train = train.copy(deep=True)
    train.loc[train["season"].eq(2022), ["pitcher_id", "batter_id"]] = [91, 92]

    state = fit_s1_state(train, valid_year=2024)

    assert state.snapshot.cutoff_year == 2023


@pytest.mark.parametrize("target", [np.nan, np.inf, -np.inf, -1, 0.5, 2, "1"])
def test_s1_fit_requires_a_finite_numeric_binary_target(target: object) -> None:
    train, _ = _fit_and_valid()
    train.iloc[0, train.columns.get_loc("control_success")] = target

    with pytest.raises(SeasonalFeatureError, match="target"):
        fit_s1_state(train, valid_year=2024)


def test_s1_transform_rejects_target_and_wrong_or_mixed_seasons() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    with pytest.raises(SeasonalFeatureError, match="target"):
        transform_s1(rows.assign(control_success=0), state)
    with pytest.raises(SeasonalFeatureError, match="transform season"):
        transform_s1(rows.assign(season=[2024, 2023, 2024]), state)
    with pytest.raises(SeasonalFeatureError, match="season"):
        transform_s1(rows.assign(season=[2024, "2024", 2024]), state)


@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_s1_requires_an_actual_dataframe_and_unique_columns(operation: str) -> None:
    train, rows = _fit_and_valid()
    if operation == "fit":
        function = lambda value: fit_s1_state(value, valid_year=2024)
        valid = train
    else:
        state = fit_s1_state(train, valid_year=2024)
        function = lambda value: transform_s1(value, state)
        valid = rows

    with pytest.raises(SeasonalFeatureError, match="DataFrame"):
        function(valid.to_dict("list"))
    duplicate = pd.concat([valid, valid.iloc[:, [-1]]], axis=1)
    duplicate.columns = [*valid.columns, valid.columns[-1]]
    with pytest.raises(SeasonalFeatureError, match="duplicate columns"):
        function(duplicate)


@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_s1_reports_missing_required_schema(operation: str) -> None:
    train, rows = _fit_and_valid()
    if operation == "fit":
        with pytest.raises(SeasonalFeatureError, match="required columns.*pitcher_id"):
            fit_s1_state(train.drop(columns="pitcher_id"), valid_year=2024)
    else:
        state = fit_s1_state(train, valid_year=2024)
        with pytest.raises(
            SeasonalFeatureError, match="required columns.*asof_pitcher_n"
        ):
            transform_s1(rows.drop(columns="asof_pitcher_n"), state)


def test_s1_rejects_non_string_column_names_with_a_schema_error() -> None:
    train, rows = _fit_and_valid()
    rows = rows.copy(deep=True)
    rows[7] = "extra"

    with pytest.raises(SeasonalFeatureError, match="column names"):
        transform_s1(rows, fit_s1_state(train, valid_year=2024))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("pitcher_id", ["unhashable"]),
        ("asof_pitcher_n", "not-a-number"),
        ("asof_pitcher_success_rate", np.inf),
        ("asof_pitcher_success_rate", 0.5 + 0.1j),
    ],
)
def test_s1_transform_rejects_malformed_entity_and_numeric_values(
    column: str, value: object
) -> None:
    train, rows = _fit_and_valid()
    rows = rows.copy(deep=True)
    if column == "pitcher_id":
        rows[column] = rows[column].astype("object")
    rows.iat[0, rows.columns.get_loc(column)] = value

    with pytest.raises(SeasonalFeatureError, match=column):
        transform_s1(rows, fit_s1_state(train, valid_year=2024))


def test_s1_rejects_preexisting_generated_or_snapshot_columns() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    with pytest.raises(SeasonalFeatureError, match="reserved"):
        transform_s1(rows.assign(season_pitcher_n=123), state)
    with pytest.raises(SeasonalFeatureError, match="reserved"):
        transform_s1(rows.assign(snapshot_pitcher_success_n=123), state)


def test_s1_ignores_unrelated_season_prefixed_columns() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    expected = transform_s1(rows, state)
    actual = transform_s1(rows.assign(season_note=["a", "b", "c"]), state)

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("valid_year", 2024.0),
        ("valid_year", 999),
        ("_cutoff_year", 2022),
        ("_cutoff_year", 2023.0),
        ("prior_rate", np.float64(0.5)),
        ("prior_rate", np.nan),
        ("prior_rate", -0.1),
        ("prior_rate", 1.1),
    ],
)
def test_s1_transform_rejects_forged_scalar_state_fields(
    field: str, value: object
) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, **{field: value}))


def test_s1_transform_normalizes_missing_state_fields() -> None:
    _, rows = _fit_and_valid()
    forged = object.__new__(S1State)

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, forged)


@pytest.mark.parametrize("side", ["pitcher", "batter"])
def test_s1_transform_rejects_forged_snapshot_schema(side: str) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    field = f"_{side}_columns"
    columns = object.__getattribute__(state, field)

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, **{field: columns[:-1]}))


@pytest.mark.parametrize("side", ["pitcher", "batter"])
def test_s1_transform_rejects_forged_snapshot_row_shapes(side: str) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    field = f"_{side}_rows"
    snapshot_rows = object.__getattribute__(state, field)
    malformed = (snapshot_rows[0][:-1], *snapshot_rows[1:])

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, **{field: malformed}))
    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, **{field: list(snapshot_rows)}))
    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(
            rows,
            _forge_state(state, **{field: (list(snapshot_rows[0]),)}),
        )


@pytest.mark.parametrize("side", ["pitcher", "batter"])
def test_s1_transform_rejects_duplicate_or_malformed_snapshot_entities(
    side: str,
) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    field = f"_{side}_rows"
    snapshot_rows = object.__getattribute__(state, field)

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(
            rows,
            _forge_state(state, **{field: (*snapshot_rows, snapshot_rows[0])}),
        )
    for bad_entity in (None, ["unhashable"]):
        malformed = ((bad_entity, *snapshot_rows[0][1:]), *snapshot_rows[1:])
        with pytest.raises(SeasonalFeatureError, match="state"):
            transform_s1(rows, _forge_state(state, **{field: malformed}))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("snapshot_pitcher_success_n", -1.0),
        ("snapshot_pitcher_success_count", np.nan),
        ("snapshot_pitcher_success_count", "1"),
        ("snapshot_pitcher_success_count", 999.0),
    ],
)
def test_s1_transform_rejects_forged_snapshot_numeric_payloads(
    column: str, value: object
) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    snapshot_rows = object.__getattribute__(state, "_pitcher_rows")
    columns = object.__getattribute__(state, "_pitcher_columns")
    changed = list(snapshot_rows[0])
    changed[columns.index(column)] = value
    forged_rows = (tuple(changed), *snapshot_rows[1:])

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, _pitcher_rows=forged_rows))
