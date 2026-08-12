from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.preprocessing import (
    PreprocessingError,
    PreprocessingSpec,
    fit_preprocessor,
    transform_preprocessor,
)


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
