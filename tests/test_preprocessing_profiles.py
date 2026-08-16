from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.features import materialize_preprocessed_fold_cache
from experiments.independent_dl.preprocessing import (
    PreprocessingError,
    PreprocessingSpec,
    fit_preprocessor,
    normalize_spec,
    transform_preprocessor,
)
from experiments.independent_dl.row_features import ROW_FEATURE_BUNDLES, RowFeatureError


ROW_FEATURE_OUTPUTS = {
    "count_context": (
        "rf_count_state",
        "rf_count_out_state",
        "rf_base_out_state",
    ),
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
    "pitcher_batter_gap": (
        "rf_success_gap",
        "rf_middle_gap",
        "rf_log_count_gap",
    ),
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
ROW_FEATURE_CATEGORICAL = {
    "count_context": ROW_FEATURE_OUTPUTS["count_context"],
    "pressure_context": ROW_FEATURE_OUTPUTS["pressure_context"][:4],
    "hand_state_interactions": ROW_FEATURE_OUTPUTS["hand_state_interactions"],
    "pitcher_batter_gap": (),
    "recent_trend": (),
    "pitchmix_shape": (),
}


def test_dl_standard_uses_train_median_and_never_refits_on_validation(
    preprocessing_frame: pd.DataFrame,
) -> None:
    train = preprocessing_frame.iloc[:3].copy()
    valid = preprocessing_frame.iloc[3:].copy()
    train.loc[1, "li"] = np.nan
    valid["li"] = [999.0, np.nan]

    state, _ = fit_preprocessor(train, PreprocessingSpec("dl_standard", ()))
    transformed = transform_preprocessor(valid, state)

    assert state.numeric_median["li"] == pytest.approx(train["li"].median())
    expected = (
        state.numeric_median["li"] - state.numeric_mean["li"]
    ) / state.numeric_std["li"]
    assert transformed.loc[valid["li"].isna(), "li"].iloc[0] == pytest.approx(
        expected
    )


def test_grouped_missing_indicators_are_three_group_flags(
    preprocessing_frame: pd.DataFrame,
) -> None:
    train = preprocessing_frame.copy()
    train.loc[0, "asof_pitcher_prev3_game_success_rate"] = np.nan
    train.loc[1, "asof_pitcher_ball_rate"] = np.nan
    train.loc[2, "asof_batter_middle_rate"] = np.nan

    _, result = fit_preprocessor(
        train,
        PreprocessingSpec("dl_standard", ("grouped_missing_indicators",)),
    )

    assert result[
        [
            "pitcher_recent_missing",
            "pitcher_career_missing",
            "batter_career_missing",
        ]
    ].to_numpy().tolist()[:3] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


@pytest.mark.parametrize(
    ("component", "rate", "count", "k", "output"),
    [
        (
            "pitcher_smooth_k100",
            "asof_pitcher_success_rate",
            "asof_pitcher_n",
            100.0,
            "pitcher_success_smooth_100",
        ),
        (
            "batter_smooth_k250",
            "asof_batter_success_rate",
            "asof_batter_n",
            250.0,
            "batter_success_smooth_250",
        ),
    ],
)
def test_smoothing_uses_fold_target_prior(
    preprocessing_frame: pd.DataFrame,
    component: str,
    rate: str,
    count: str,
    k: float,
    output: str,
) -> None:
    state, result = fit_preprocessor(
        preprocessing_frame,
        PreprocessingSpec("tree_native", (component,)),
    )
    prior = preprocessing_frame["control_success"].mean()
    expected = (
        preprocessing_frame[count].iloc[0] * preprocessing_frame[rate].iloc[0]
        + k * prior
    ) / (preprocessing_frame[count].iloc[0] + k)

    assert state.target_prior == pytest.approx(prior)
    assert result[output].iloc[0] == pytest.approx(expected)


def test_tree_native_preserves_numeric_nan(
    preprocessing_frame: pd.DataFrame,
) -> None:
    frame = preprocessing_frame.copy()
    frame.loc[0, "li"] = np.nan

    _, result = fit_preprocessor(frame, PreprocessingSpec("tree_native", ()))

    assert np.isnan(result.loc[0, "li"])


def test_duplicate_pitchmix_count_must_match_in_every_transformed_frame(
    preprocessing_frame: pd.DataFrame,
) -> None:
    state, _ = fit_preprocessor(
        preprocessing_frame, PreprocessingSpec("dl_standard", ())
    )
    changed = preprocessing_frame.copy()
    changed.loc[0, "asof_pitcher_pitchmix_n"] += 1

    with pytest.raises(PreprocessingError, match="pitchmix_n"):
        transform_preprocessor(changed, state)


def test_selective_transform_changes_only_fixed_eligible_columns(
    preprocessing_frame: pd.DataFrame,
) -> None:
    standard_state, standard = fit_preprocessor(
        preprocessing_frame, PreprocessingSpec("dl_standard", ())
    )
    selective_state, selective = fit_preprocessor(
        preprocessing_frame, PreprocessingSpec("dl_selective_transform", ())
    )

    assert standard.columns.tolist() == selective.columns.tolist()
    assert set(selective_state.yeo_johnson_lambda) == {
        "li",
        "run_top_before",
        "run_bot_before",
        "run_total_before",
        "score_diff_home",
        "score_diff_pitcher_team",
        "asof_pitcher_middle_rate",
        "asof_batter_middle_rate",
    }
    assert standard_state.yeo_johnson_lambda == {}
    assert not np.allclose(standard["li"], selective["li"])
    assert np.allclose(
        standard["asof_pitcher_success_rate"],
        selective["asof_pitcher_success_rate"],
    )


def test_entity_frequency_is_fit_on_train_and_unseen_is_zero(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
) -> None:
    state, _ = fit_preprocessor(
        preprocessing_train,
        PreprocessingSpec("tree_native", ("entity_frequency_log1p",)),
    )
    transformed = transform_preprocessor(preprocessing_valid, state)

    assert transformed["pitcher_id_frequency"].tolist() == [2.0, 0.0]
    assert transformed["batter_id_frequency"].tolist() == [0.0, 1.0]
    assert transformed["pitcher_id_frequency_log1p"].iloc[1] == 0.0


def test_entity_frequency_and_oov_keeps_ids_and_marks_unseen_values(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
) -> None:
    state, fitted = fit_preprocessor(
        preprocessing_train,
        PreprocessingSpec("dl_standard", ("entity_frequency_and_oov",)),
    )
    transformed = transform_preprocessor(preprocessing_valid, state)

    assert "pitcher_id" in fitted
    assert transformed["pitcher_id_oov"].tolist() == [0.0, 1.0]
    assert transformed["batter_id_oov"].tolist() == [1.0, 0.0]
    assert "unseen" not in state.entity_frequency["pitcher_id"]
    assert "pitcher_id_oov" in state.numeric_columns


@pytest.mark.parametrize("bundle", ROW_FEATURE_BUNDLES)
def test_row_feature_bundle_has_stable_fold_schema_and_partition(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    bundle: str,
) -> None:
    spec = PreprocessingSpec("dl_standard", ("hand_matchup", bundle))

    state, fitted = fit_preprocessor(preprocessing_train, spec)
    transformed = transform_preprocessor(preprocessing_valid, state)

    assert state.spec.components == ("hand_matchup", bundle)
    assert tuple(fitted.columns) == state.output_columns
    assert tuple(transformed.columns) == state.output_columns
    assert set(ROW_FEATURE_OUTPUTS[bundle]).issubset(state.output_columns)
    expected_categorical = set(ROW_FEATURE_CATEGORICAL[bundle])
    assert expected_categorical.issubset(state.categorical_columns)
    assert (
        set(ROW_FEATURE_OUTPUTS[bundle]) - expected_categorical
    ).issubset(state.numeric_columns)
    assert "control_success" not in state.source_columns
    assert "control_success" not in state.output_columns


def test_row_feature_components_sort_after_hand_matchup_and_remain_sealed() -> None:
    for bundle in ROW_FEATURE_BUNDLES:
        normalized = normalize_spec(
            PreprocessingSpec("dl_standard", (bundle, "hand_matchup"))
        )
        assert normalized.components == ("hand_matchup", bundle)

    with pytest.raises(PreprocessingError, match="unknown preprocessing component"):
        normalize_spec(PreprocessingSpec("dl_standard", ("hand_matchup", "unknown")))
    with pytest.raises(PreprocessingError, match="must be unique"):
        normalize_spec(
            PreprocessingSpec(
                "dl_standard",
                ("hand_matchup", "count_context", "count_context"),
            )
        )


def test_row_feature_errors_preserve_message_and_cause(
    preprocessing_frame: pd.DataFrame,
) -> None:
    with pytest.raises(PreprocessingError, match="count_context.*outs_before") as caught:
        fit_preprocessor(
            preprocessing_frame.drop(columns="outs_before"),
            PreprocessingSpec("dl_standard", ("hand_matchup", "count_context")),
        )

    assert isinstance(caught.value.__cause__, RowFeatureError)


def test_derived_category_map_is_train_only_and_unseen_validation_is_oov(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
    tmp_path: Path,
) -> None:
    valid = preprocessing_valid.copy()
    valid.loc[0, ["balls_before", "strikes_before", "outs_before"]] = [3, 2, 2]
    cache = materialize_preprocessed_fold_cache(
        tmp_path,
        preprocessing_train,
        valid,
        preprocessing_history,
        "raw_typed",
        PreprocessingSpec("dl_standard", ("hand_matchup", "count_context")),
        2023,
        2024,
    )

    mapping = cache.state.category_maps["rf_count_out_state"]
    column_index = cache.state.categorical_columns.index("rf_count_out_state")
    expected_levels = {
        f"{balls}_{strikes}_{outs}"
        for balls, strikes, outs in preprocessing_train[
            ["balls_before", "strikes_before", "outs_before"]
        ].itertuples(index=False, name=None)
    }
    assert set(mapping) == expected_levels
    assert "3_2_2" not in mapping
    assert cache.valid.x_cat[0, column_index] == 0


@pytest.mark.parametrize(
    ("bundle", "output", "source"),
    (
        ("pressure_context", "rf_pitcher_team_win_expectancy", "home_win_expectancy"),
        ("pitcher_batter_gap", "rf_success_gap", "asof_pitcher_success_rate"),
        (
            "recent_trend",
            "rf_success_prev1_prev5",
            "asof_pitcher_prev1_game_success_rate",
        ),
        ("pitchmix_shape", "rf_pitchmix_max", "asof_pitcher_fastball_rate"),
    ),
)
def test_row_feature_numeric_state_is_train_only(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    bundle: str,
    output: str,
    source: str,
) -> None:
    state, _ = fit_preprocessor(
        preprocessing_train,
        PreprocessingSpec("dl_standard", ("hand_matchup", bundle)),
    )
    fitted_state = (
        dict(state.numeric_median),
        dict(state.numeric_mean),
        dict(state.numeric_std),
    )
    original = transform_preprocessor(preprocessing_valid, state)
    changed = preprocessing_valid.copy()
    changed.loc[changed.index[0], source] = 999.0
    modified = transform_preprocessor(changed, state)

    assert not np.isclose(
        original.loc[original.index[0], output],
        modified.loc[modified.index[0], output],
    )
    assert (
        dict(state.numeric_median),
        dict(state.numeric_mean),
        dict(state.numeric_std),
    ) == fitted_state


def test_hand_matchup_baseline_preserves_existing_preprocessing(
    preprocessing_frame: pd.DataFrame,
) -> None:
    plain_state, plain = fit_preprocessor(
        preprocessing_frame, PreprocessingSpec("dl_standard", ())
    )
    state, result = fit_preprocessor(
        preprocessing_frame,
        PreprocessingSpec("dl_standard", ("hand_matchup",)),
    )

    pd.testing.assert_frame_equal(result.loc[:, plain.columns], plain)
    assert state.source_columns == plain_state.source_columns
    assert state.output_columns == plain_state.output_columns + ("hand_matchup",)
    assert state.numeric_columns == plain_state.numeric_columns
    assert state.categorical_columns == plain_state.categorical_columns + ("hand_matchup",)
    assert dict(state.numeric_median) == dict(plain_state.numeric_median)
    assert dict(state.numeric_mean) == dict(plain_state.numeric_mean)
    assert dict(state.numeric_std) == dict(plain_state.numeric_std)
    assert state.target_prior == plain_state.target_prior
    assert result["hand_matchup"].tolist() == ["1_2", "2_1", "1_2", "2_1", "1_1"]
