from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.row_features import (
    ROW_FEATURE_BUNDLES,
    RowFeatureError,
    add_row_feature_bundle,
    row_segment_labels,
)


EXPECTED_COLUMNS = {
    "count_context": ("rf_count_state", "rf_count_out_state", "rf_base_out_state"),
    "pressure_context": (
        "rf_inning_bucket",
        "rf_pitcher_score_bucket",
        "rf_leverage_bucket",
        "rf_pressure_state",
        "rf_pitcher_team_win_expectancy",
    ),
    "hand_state_interactions": (
        "rf_hand_count_state",
        "rf_hand_base_state",
        "rf_game_hand_matchup",
    ),
    "pitcher_batter_gap": ("rf_success_gap", "rf_middle_gap", "rf_log_count_gap"),
    "recent_trend": (
        "rf_success_prev1_prev5",
        "rf_success_prev3_prev5",
        "rf_success_prev1_career",
        "rf_middle_prev1_prev5",
        "rf_middle_prev3_prev5",
        "rf_middle_prev1_career",
    ),
    "pitchmix_shape": (
        "rf_pitchmix_max",
        "rf_pitchmix_min",
        "rf_pitchmix_top2_margin",
        "rf_pitchmix_entropy",
        "rf_fastball_breaking_gap",
        "rf_fastball_offspeed_gap",
        "rf_breaking_offspeed_gap",
    ),
}


def test_count_context_has_exact_public_contract(
    preprocessing_frame: pd.DataFrame,
) -> None:
    assert ROW_FEATURE_BUNDLES == (
        "count_context",
        "pressure_context",
        "hand_state_interactions",
        "pitcher_batter_gap",
        "recent_trend",
        "pitchmix_shape",
    )

    result = add_row_feature_bundle(preprocessing_frame, "count_context")

    added = result.columns.difference(preprocessing_frame.columns).tolist()
    assert added == ["rf_base_out_state", "rf_count_out_state", "rf_count_state"]
    assert result["rf_count_state"].tolist() == ["0_0", "1_1", "2_2", "3_1", "1_2"]
    assert result["rf_count_out_state"].tolist() == [
        "0_0_0",
        "1_1_1",
        "2_2_2",
        "3_1_0",
        "1_2_1",
    ]
    assert result["rf_base_out_state"].tolist() == [
        "000_0",
        "100_1",
        "010_2",
        "001_0",
        "110_1",
    ]
    assert result.index.equals(preprocessing_frame.index)
    assert result is not preprocessing_frame


def test_pressure_context_has_exact_values(preprocessing_frame: pd.DataFrame) -> None:
    result = add_row_feature_bundle(preprocessing_frame, "pressure_context")

    added = result.columns.difference(preprocessing_frame.columns).tolist()
    assert added == [
        "rf_inning_bucket",
        "rf_leverage_bucket",
        "rf_pitcher_score_bucket",
        "rf_pitcher_team_win_expectancy",
        "rf_pressure_state",
    ]
    assert result["rf_inning_bucket"].tolist() == ["1-3", "4-6", "7-9", "10+", "7-9"]
    assert result["rf_pitcher_score_bucket"].tolist() == [
        "-1:1",
        "-1:1",
        "-1:1",
        "2:3",
        "-3:-2",
    ]
    assert result["rf_leverage_bucket"].tolist() == [
        "0.7:1.5",
        "0.7:1.5",
        ">=1.5",
        ">=1.5",
        ">=1.5",
    ]
    assert result["rf_pressure_state"].tolist() == [
        "1-3_-1:1_0.7:1.5",
        "4-6_-1:1_0.7:1.5",
        "7-9_-1:1_>=1.5",
        "10+_2:3_>=1.5",
        "7-9_-3:-2_>=1.5",
    ]
    assert result["rf_pitcher_team_win_expectancy"].tolist() == [0.5, 0.4, 0.4, 0.3, 0.3]


def test_hand_state_interactions_can_derive_hand_matchup(
    preprocessing_frame: pd.DataFrame,
) -> None:
    result = add_row_feature_bundle(preprocessing_frame, "hand_state_interactions")

    assert "hand_matchup" not in result
    assert result["rf_hand_count_state"].tolist() == [
        "1_2_0_0",
        "2_1_1_1",
        "1_2_2_2",
        "2_1_3_1",
        "1_1_1_2",
    ]
    assert result["rf_hand_base_state"].tolist() == [
        "1_2_000",
        "2_1_100",
        "1_2_010",
        "2_1_001",
        "1_1_110",
    ]
    assert result["rf_game_hand_matchup"].tolist() == [
        "R_1_2",
        "F_2_1",
        "R_1_2",
        "F_2_1",
        "R_1_1",
    ]


def test_pitcher_batter_gap_has_exact_values(preprocessing_frame: pd.DataFrame) -> None:
    result = add_row_feature_bundle(preprocessing_frame, "pitcher_batter_gap")

    np.testing.assert_allclose(result["rf_success_gap"], [0.03, -0.03, 0.02, -0.02, 0.0])
    np.testing.assert_allclose(result["rf_middle_gap"], [0.06, 0.17, 0.25, 0.38, 0.44])
    np.testing.assert_allclose(
        result["rf_log_count_gap"],
        np.log1p([100, 50, 25, 10, 5]) - np.log1p([80, 40, 20, 10, 5]),
    )


def test_recent_trend_has_exact_values(preprocessing_frame: pd.DataFrame) -> None:
    result = add_row_feature_bundle(preprocessing_frame, "recent_trend")

    np.testing.assert_allclose(result["rf_success_prev1_prev5"], 0.0)
    np.testing.assert_allclose(result["rf_success_prev3_prev5"], 0.0)
    np.testing.assert_allclose(result["rf_success_prev1_career"], [-0.05, -0.05, 0.0, -0.10, 0.0])
    np.testing.assert_allclose(result["rf_middle_prev1_prev5"], 0.0)
    np.testing.assert_allclose(result["rf_middle_prev3_prev5"], 0.0)
    np.testing.assert_allclose(result["rf_middle_prev1_career"], [0.30, 0.10, 0.20, -0.20, -0.10])


def test_pitchmix_shape_normalizes_before_exact_derivation() -> None:
    frame = pd.DataFrame(
        {
            "asof_pitcher_fastball_rate": [0.6, 2.0],
            "asof_pitcher_breaking_rate": [0.3, 1.0],
            "asof_pitcher_offspeed_rate": [0.1, 1.0],
        },
        index=[17, 4],
    )

    result = add_row_feature_bundle(frame, "pitchmix_shape")

    expected_entropy = [
        -sum(value * np.log(value) for value in (0.6, 0.3, 0.1)) / np.log(3),
        -sum(value * np.log(value) for value in (0.5, 0.25, 0.25)) / np.log(3),
    ]
    np.testing.assert_allclose(result["rf_pitchmix_max"], [0.6, 0.5])
    np.testing.assert_allclose(result["rf_pitchmix_min"], [0.1, 0.25])
    np.testing.assert_allclose(result["rf_pitchmix_top2_margin"], [0.3, 0.25])
    np.testing.assert_allclose(result["rf_pitchmix_entropy"], expected_entropy)
    np.testing.assert_allclose(result["rf_fastball_breaking_gap"], [0.3, 0.25])
    np.testing.assert_allclose(result["rf_fastball_offspeed_gap"], [0.5, 0.25])
    np.testing.assert_allclose(result["rf_breaking_offspeed_gap"], [0.2, 0.0])
    assert result.index.tolist() == [17, 4]


@pytest.mark.parametrize("bundle", ROW_FEATURE_BUNDLES)
def test_bundle_is_invariant_to_row_context(
    preprocessing_frame: pd.DataFrame, bundle: str
) -> None:
    columns = list(EXPECTED_COLUMNS[bundle])
    expected = add_row_feature_bundle(preprocessing_frame, bundle).loc[2, columns]

    reversed_result = add_row_feature_bundle(preprocessing_frame.iloc[::-1], bundle)
    pd.testing.assert_series_equal(reversed_result.loc[2, columns], expected)

    unrelated = preprocessing_frame.iloc[[0]].copy()
    unrelated.index = [99]
    unrelated["row_id"] = "unrelated"
    inserted_result = add_row_feature_bundle(
        pd.concat([preprocessing_frame.iloc[:2], unrelated, preprocessing_frame.iloc[2:]]),
        bundle,
    )
    pd.testing.assert_series_equal(inserted_result.loc[2, columns], expected)

    single_result = add_row_feature_bundle(preprocessing_frame.loc[[2]], bundle)
    pd.testing.assert_series_equal(single_result.loc[2, columns], expected)

    split_result = pd.concat(
        [
            add_row_feature_bundle(preprocessing_frame.iloc[:2], bundle),
            add_row_feature_bundle(preprocessing_frame.iloc[2:], bundle),
        ]
    )
    pd.testing.assert_series_equal(split_result.loc[2, columns], expected)


@pytest.mark.parametrize("bundle", ROW_FEATURE_BUNDLES)
def test_bundle_does_not_require_or_inspect_target(
    preprocessing_frame: pd.DataFrame, bundle: str
) -> None:
    columns = list(EXPECTED_COLUMNS[bundle])
    with_target = add_row_feature_bundle(preprocessing_frame, bundle)
    without_target = add_row_feature_bundle(
        preprocessing_frame.drop(columns="control_success"), bundle
    )

    pd.testing.assert_frame_equal(with_target[columns], without_target[columns])


@pytest.mark.parametrize(
    ("bundle", "source"),
    (
        ("count_context", "outs_before"),
        ("pressure_context", "inning"),
        ("hand_state_interactions", "game_type"),
        ("pitcher_batter_gap", "asof_batter_n"),
        ("recent_trend", "asof_pitcher_prev5_game_success_rate"),
        ("pitchmix_shape", "asof_pitcher_offspeed_rate"),
    ),
)
def test_missing_source_names_bundle_and_column(
    preprocessing_frame: pd.DataFrame, bundle: str, source: str
) -> None:
    with pytest.raises(RowFeatureError, match=f"{bundle}.*{source}"):
        add_row_feature_bundle(preprocessing_frame.drop(columns=source), bundle)


def test_unknown_bundle_is_rejected(preprocessing_frame: pd.DataFrame) -> None:
    with pytest.raises(RowFeatureError, match="unknown.*made_up"):
        add_row_feature_bundle(preprocessing_frame, "made_up")


@pytest.mark.parametrize("bundle", ROW_FEATURE_BUNDLES)
def test_existing_output_column_is_rejected(
    preprocessing_frame: pd.DataFrame, bundle: str
) -> None:
    collision = EXPECTED_COLUMNS[bundle][0]
    frame = preprocessing_frame.assign(**{collision: "keep-me"})

    with pytest.raises(RowFeatureError, match=f"{bundle}.*{collision}"):
        add_row_feature_bundle(frame, bundle)


@pytest.mark.parametrize(
    ("bundle", "source"),
    (
        ("count_context", "balls_before"),
        ("pressure_context", "li"),
        ("hand_state_interactions", "strikes_before"),
        ("pitcher_batter_gap", "asof_pitcher_success_rate"),
        ("recent_trend", "asof_pitcher_prev1_game_middle_rate"),
        ("pitchmix_shape", "asof_pitcher_fastball_rate"),
    ),
)
def test_nonnumeric_required_value_is_rejected(
    preprocessing_frame: pd.DataFrame, bundle: str, source: str
) -> None:
    frame = preprocessing_frame.copy()
    frame.loc[0, source] = "not-a-number"

    with pytest.raises(RowFeatureError, match=source):
        add_row_feature_bundle(frame, bundle)


@pytest.mark.parametrize("source", ("balls_before", "strikes_before", "outs_before"))
def test_count_context_rejects_negative_or_fractional_counts(
    preprocessing_frame: pd.DataFrame, source: str
) -> None:
    negative = preprocessing_frame.copy()
    negative.loc[0, source] = -1
    fractional = preprocessing_frame.copy()
    fractional.loc[0, source] = 0.5

    with pytest.raises(RowFeatureError, match=source):
        add_row_feature_bundle(negative, "count_context")
    with pytest.raises(RowFeatureError, match=source):
        add_row_feature_bundle(fractional, "count_context")


@pytest.mark.parametrize("source", ("asof_pitcher_n", "asof_batter_n"))
def test_pitcher_batter_gap_rejects_negative_counts(
    preprocessing_frame: pd.DataFrame, source: str
) -> None:
    frame = preprocessing_frame.copy()
    frame.loc[0, source] = -1

    with pytest.raises(RowFeatureError, match=source):
        add_row_feature_bundle(frame, "pitcher_batter_gap")


@pytest.mark.parametrize(
    "source",
    (
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ),
)
def test_pitchmix_shape_rejects_negative_finite_rates(
    preprocessing_frame: pd.DataFrame, source: str
) -> None:
    frame = preprocessing_frame.copy()
    frame.loc[0, source] = -0.1

    with pytest.raises(RowFeatureError, match="pitchmix.*nonnegative"):
        add_row_feature_bundle(frame, "pitchmix_shape")


def test_pitchmix_missing_nonfinite_or_zero_rows_are_all_nan() -> None:
    frame = pd.DataFrame(
        {
            "asof_pitcher_fastball_rate": [np.nan, 0.0, np.inf],
            "asof_pitcher_breaking_rate": [0.4, 0.0, 0.3],
            "asof_pitcher_offspeed_rate": [0.6, 0.0, 0.7],
        }
    )

    result = add_row_feature_bundle(frame, "pitchmix_shape")

    assert result[list(EXPECTED_COLUMNS["pitchmix_shape"])].isna().all().all()


def test_numeric_differences_preserve_missing_operands(
    preprocessing_frame: pd.DataFrame,
) -> None:
    gap_frame = preprocessing_frame.copy()
    gap_frame.loc[0, "asof_batter_success_rate"] = np.nan
    gap_frame.loc[1, "asof_pitcher_n"] = np.nan
    gap = add_row_feature_bundle(gap_frame, "pitcher_batter_gap")
    assert pd.isna(gap.loc[0, "rf_success_gap"])
    assert pd.isna(gap.loc[1, "rf_log_count_gap"])

    recent_frame = preprocessing_frame.copy()
    recent_frame.loc[0, "asof_pitcher_prev5_game_success_rate"] = np.nan
    recent_frame.loc[1, "asof_pitcher_middle_rate"] = np.nan
    recent = add_row_feature_bundle(recent_frame, "recent_trend")
    assert pd.isna(recent.loc[0, "rf_success_prev1_prev5"])
    assert pd.isna(recent.loc[1, "rf_middle_prev1_career"])


def test_categorical_missing_values_are_explicit(
    preprocessing_frame: pd.DataFrame,
) -> None:
    count_frame = preprocessing_frame.copy()
    count_frame.loc[0, ["balls_before", "base_state"]] = np.nan
    count = add_row_feature_bundle(count_frame, "count_context")
    assert count.loc[0, "rf_count_state"] == "__MISSING___0"
    assert count.loc[0, "rf_base_out_state"] == "__MISSING___0"

    pressure_frame = preprocessing_frame.copy()
    pressure_frame.loc[0, ["inning", "score_diff_pitcher_team", "li"]] = np.nan
    pressure = add_row_feature_bundle(pressure_frame, "pressure_context")
    assert pressure.loc[0, "rf_inning_bucket"] == "__MISSING__"
    assert pressure.loc[0, "rf_pitcher_score_bucket"] == "__MISSING__"
    assert pressure.loc[0, "rf_leverage_bucket"] == "__MISSING__"


def test_pressure_context_rejects_unknown_half_inning(
    preprocessing_frame: pd.DataFrame,
) -> None:
    frame = preprocessing_frame.copy()
    frame.loc[0, "top_bottom"] = "X"

    with pytest.raises(RowFeatureError, match="top_bottom"):
        add_row_feature_bundle(frame, "pressure_context")


def test_pressure_context_preserves_duplicate_index_alignment() -> None:
    frame = pd.DataFrame(
        {
            "inning": [1, 4],
            "score_diff_pitcher_team": [0, 1],
            "li": [1.0, 2.0],
            "top_bottom": ["T", "B"],
            "home_win_expectancy": [0.6, 0.7],
            "away_win_expectancy": [0.4, 0.3],
        },
        index=[5, 5],
    )

    result = add_row_feature_bundle(frame, "pressure_context")

    assert result.index.tolist() == [5, 5]
    assert result["rf_pitcher_team_win_expectancy"].tolist() == [0.6, 0.3]


def test_row_segment_labels_have_exact_outputs_and_values() -> None:
    index = [8, 2, 15, 3]
    frame = pd.DataFrame(
        {
            "game_type": ["R", None, "F", "R"],
            "pitcher_hand": pd.Series([1, 1, None, 2], index=index, dtype=object),
            "batter_hand": pd.Series([2, None, 1, 2], index=index, dtype=object),
            "pitcher_id": [11, 99, None, None],
            "batter_id": [21, 22, None, 23],
            "li": [1.5, 1.49, None, 2.0],
            "inning": [7, 6, None, 10],
            "runner_on_2b": [1, 1, None, 0],
            "runner_on_3b": [0, None, 1, 0],
        },
        index=index,
    )

    result = row_segment_labels(
        frame,
        train_pitcher_ids={11, "__MISSING__"},
        train_batter_ids={"21", 23},
    )

    assert result.columns.tolist() == [
        "game_type_segment",
        "hand_matchup_segment",
        "pitcher_id_segment",
        "batter_id_segment",
        "li_high",
        "late_inning",
        "runner_in_scoring_position",
    ]
    assert result.index.tolist() == [8, 2, 15, 3]
    assert result["game_type_segment"].tolist() == ["R", "__MISSING__", "F", "R"]
    assert result["hand_matchup_segment"].tolist() == [
        "1_2",
        "1___MISSING__",
        "__MISSING___1",
        "2_2",
    ]
    assert result["pitcher_id_segment"].tolist() == ["known", "oov", "known", "known"]
    assert result["batter_id_segment"].tolist() == ["known", "oov", "oov", "known"]
    assert result["li_high"].tolist() == [True, False, "__MISSING__", True]
    assert result["late_inning"].tolist() == [True, False, "__MISSING__", True]
    assert result["runner_in_scoring_position"].tolist() == [
        True,
        "__MISSING__",
        "__MISSING__",
        False,
    ]


def test_row_segment_labels_depend_only_on_row_and_explicit_train_sets(
    preprocessing_frame: pd.DataFrame,
) -> None:
    pitcher_ids = {11, 12}
    batter_ids = {21, 22}
    expected = row_segment_labels(preprocessing_frame.loc[[2]], pitcher_ids, batter_ids)

    changed_other_rows = preprocessing_frame.copy()
    changed_other_rows.loc[[0, 1, 3, 4], "li"] = 99
    changed_other_rows.loc[[0, 1, 3, 4], "pitcher_id"] = 999
    result = row_segment_labels(changed_other_rows.iloc[::-1], pitcher_ids, batter_ids)

    pd.testing.assert_frame_equal(result.loc[[2]], expected)

    alternate_sets = row_segment_labels(preprocessing_frame.loc[[2]], {11}, {23})
    non_id_columns = [
        column
        for column in expected.columns
        if column not in {"pitcher_id_segment", "batter_id_segment"}
    ]
    pd.testing.assert_frame_equal(alternate_sets[non_id_columns], expected[non_id_columns])
    assert alternate_sets.loc[2, "pitcher_id_segment"] == "known"
    assert alternate_sets.loc[2, "batter_id_segment"] == "known"


def test_row_segment_labels_require_row_local_sources(
    preprocessing_frame: pd.DataFrame,
) -> None:
    with pytest.raises(RowFeatureError, match="row_segment_labels.*runner_on_3b"):
        row_segment_labels(
            preprocessing_frame.drop(columns="runner_on_3b"),
            train_pitcher_ids=set(),
            train_batter_ids=set(),
        )


def test_row_segment_labels_reject_nonnumeric_threshold_inputs(
    preprocessing_frame: pd.DataFrame,
) -> None:
    frame = preprocessing_frame.copy()
    frame.loc[0, "li"] = "high"

    with pytest.raises(RowFeatureError, match="li"):
        row_segment_labels(frame, set(), set())


def test_missing_id_is_known_only_for_explicit_missing_marker(
    preprocessing_frame: pd.DataFrame,
) -> None:
    frame = preprocessing_frame.loc[[0]].copy()
    frame.loc[0, ["pitcher_id", "batter_id"]] = np.nan

    implicit = row_segment_labels(frame, {None}, {np.nan})
    explicit = row_segment_labels(frame, {"__MISSING__"}, {"__MISSING__"})

    assert implicit.loc[0, "pitcher_id_segment"] == "oov"
    assert implicit.loc[0, "batter_id_segment"] == "oov"
    assert explicit.loc[0, "pitcher_id_segment"] == "known"
    assert explicit.loc[0, "batter_id_segment"] == "known"
