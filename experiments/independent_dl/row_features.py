"""Pure row-local feature derivations for the TabM campaign."""

from __future__ import annotations

from numbers import Real
from typing import AbstractSet

import numpy as np
import pandas as pd


MISSING_CATEGORY = "__MISSING__"

ROW_FEATURE_BUNDLES = (
    "count_context",
    "pressure_context",
    "hand_state_interactions",
    "pitcher_batter_gap",
    "recent_trend",
    "pitchmix_shape",
)


class RowFeatureError(ValueError):
    """Raised when a row-local feature cannot be derived safely."""


def _required(frame: pd.DataFrame, columns: tuple[str, ...], bundle: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise RowFeatureError(f"{bundle} requires missing columns: {missing}")


def _numeric(values: pd.Series, column: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & (result.isna() | ~np.isfinite(result))
    if invalid.any():
        raise RowFeatureError(f"{column} contains a non-numeric or non-finite value")
    return result.astype("float64")


def _numeric_allow_nonfinite(values: pd.Series, column: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    if (values.notna() & result.isna()).any():
        raise RowFeatureError(f"{column} contains a non-numeric value")
    return result.astype("float64")


def _count(values: pd.Series, column: str) -> pd.Series:
    result = _numeric(values, column)
    invalid = result.notna() & ((result < 0) | result.mod(1).ne(0))
    if invalid.any():
        raise RowFeatureError(f"{column} must contain nonnegative integers")
    return result.astype("Int64")


def _nonnegative(values: pd.Series, column: str) -> pd.Series:
    result = _numeric(values, column)
    if (result.dropna() < 0).any():
        raise RowFeatureError(f"{column} must be nonnegative")
    return result


def _category(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna(MISSING_CATEGORY)


def _join(*values: pd.Series) -> pd.Series:
    result = _category(values[0])
    for item in values[1:]:
        result = result + "_" + _category(item)
    return result.astype(object)


def _add_count_context(frame: pd.DataFrame) -> None:
    bundle = "count_context"
    sources = ("balls_before", "strikes_before", "outs_before", "base_state")
    _required(frame, sources, bundle)
    balls = _count(frame["balls_before"], "balls_before")
    strikes = _count(frame["strikes_before"], "strikes_before")
    outs = _count(frame["outs_before"], "outs_before")
    frame["rf_count_state"] = _join(balls, strikes)
    frame["rf_count_out_state"] = _join(balls, strikes, outs)
    frame["rf_base_out_state"] = _join(frame["base_state"], outs)


def _bucket(values: pd.Series, choices: tuple[tuple[pd.Series, str], ...]) -> pd.Series:
    result = pd.Series(MISSING_CATEGORY, index=values.index, dtype=object)
    for condition, label in choices:
        result.loc[values.notna() & condition] = label
    return result


def _add_pressure_context(frame: pd.DataFrame) -> None:
    bundle = "pressure_context"
    sources = (
        "inning",
        "score_diff_pitcher_team",
        "li",
        "top_bottom",
        "home_win_expectancy",
        "away_win_expectancy",
    )
    _required(frame, sources, bundle)
    inning = _numeric(frame["inning"], "inning")
    score = _numeric(frame["score_diff_pitcher_team"], "score_diff_pitcher_team")
    leverage = _numeric(frame["li"], "li")
    home = _numeric(frame["home_win_expectancy"], "home_win_expectancy")
    away = _numeric(frame["away_win_expectancy"], "away_win_expectancy")
    top_bottom = frame["top_bottom"].astype("string")
    invalid_side = top_bottom.notna() & ~top_bottom.isin(("T", "B"))
    if invalid_side.any():
        raise RowFeatureError("top_bottom must contain only T, B, or missing values")

    frame["rf_inning_bucket"] = _bucket(
        inning,
        (
            (inning <= 3, "1-3"),
            (inning.between(4, 6), "4-6"),
            (inning.between(7, 9), "7-9"),
            (inning >= 10, "10+"),
        ),
    )
    frame["rf_pitcher_score_bucket"] = _bucket(
        score,
        (
            (score <= -4, "<=-4"),
            (score.between(-3, -2), "-3:-2"),
            (score.between(-1, 1), "-1:1"),
            (score.between(2, 3), "2:3"),
            (score >= 4, ">=4"),
        ),
    )
    frame["rf_leverage_bucket"] = _bucket(
        leverage,
        (
            (leverage < 0.7, "<0.7"),
            ((leverage >= 0.7) & (leverage < 1.5), "0.7:1.5"),
            (leverage >= 1.5, ">=1.5"),
        ),
    )
    frame["rf_pressure_state"] = _join(
        frame["rf_inning_bucket"],
        frame["rf_pitcher_score_bucket"],
        frame["rf_leverage_bucket"],
    )
    frame["rf_pitcher_team_win_expectancy"] = np.where(
        top_bottom.eq("T").fillna(False).to_numpy(),
        home.to_numpy(),
        np.where(
            top_bottom.eq("B").fillna(False).to_numpy(),
            away.to_numpy(),
            np.nan,
        ),
    )


def _add_hand_state_interactions(frame: pd.DataFrame) -> None:
    bundle = "hand_state_interactions"
    sources = ("balls_before", "strikes_before", "base_state", "game_type")
    _required(frame, sources, bundle)
    balls = _count(frame["balls_before"], "balls_before")
    strikes = _count(frame["strikes_before"], "strikes_before")
    if "hand_matchup" in frame:
        hand_matchup = frame["hand_matchup"]
    else:
        _required(frame, ("pitcher_hand", "batter_hand"), bundle)
        hand_matchup = _join(frame["pitcher_hand"], frame["batter_hand"])
    frame["rf_hand_count_state"] = _join(hand_matchup, balls, strikes)
    frame["rf_hand_base_state"] = _join(hand_matchup, frame["base_state"])
    frame["rf_game_hand_matchup"] = _join(frame["game_type"], hand_matchup)


def _add_pitcher_batter_gap(frame: pd.DataFrame) -> None:
    bundle = "pitcher_batter_gap"
    sources = (
        "asof_pitcher_success_rate",
        "asof_batter_success_rate",
        "asof_pitcher_middle_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_n",
        "asof_batter_n",
    )
    _required(frame, sources, bundle)
    numeric = {column: _numeric(frame[column], column) for column in sources[:-2]}
    pitcher_n = _nonnegative(frame["asof_pitcher_n"], "asof_pitcher_n")
    batter_n = _nonnegative(frame["asof_batter_n"], "asof_batter_n")
    frame["rf_success_gap"] = (
        numeric["asof_pitcher_success_rate"] - numeric["asof_batter_success_rate"]
    )
    frame["rf_middle_gap"] = (
        numeric["asof_pitcher_middle_rate"] - numeric["asof_batter_middle_rate"]
    )
    frame["rf_log_count_gap"] = np.log1p(pitcher_n) - np.log1p(batter_n)


def _add_recent_trend(frame: pd.DataFrame) -> None:
    bundle = "recent_trend"
    sources = tuple(
        f"asof_pitcher_prev{k}_game_{kind}_rate"
        for kind in ("success", "middle")
        for k in (1, 3, 5)
    ) + ("asof_pitcher_success_rate", "asof_pitcher_middle_rate")
    _required(frame, sources, bundle)
    values = {column: _numeric(frame[column], column) for column in sources}
    for kind in ("success", "middle"):
        prefix = f"asof_pitcher_prev"
        prev1 = values[f"{prefix}1_game_{kind}_rate"]
        prev3 = values[f"{prefix}3_game_{kind}_rate"]
        prev5 = values[f"{prefix}5_game_{kind}_rate"]
        career = values[f"asof_pitcher_{kind}_rate"]
        frame[f"rf_{kind}_prev1_prev5"] = prev1 - prev5
        frame[f"rf_{kind}_prev3_prev5"] = prev3 - prev5
        frame[f"rf_{kind}_prev1_career"] = prev1 - career


def _add_pitchmix_shape(frame: pd.DataFrame) -> None:
    bundle = "pitchmix_shape"
    sources = (
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    )
    outputs = (
        "rf_pitchmix_max",
        "rf_pitchmix_min",
        "rf_pitchmix_top2_margin",
        "rf_pitchmix_entropy",
        "rf_fastball_breaking_gap",
        "rf_fastball_offspeed_gap",
        "rf_breaking_offspeed_gap",
    )
    _required(frame, sources, bundle)
    values = np.column_stack(
        [_numeric_allow_nonfinite(frame[column], column).to_numpy() for column in sources]
    )
    finite = np.isfinite(values)
    if (finite & (values < 0)).any():
        raise RowFeatureError("pitchmix rates must be nonnegative")
    valid = finite.all(axis=1) & (values.sum(axis=1) > 0)
    derived = np.full((len(frame), len(outputs)), np.nan, dtype="float64")
    if valid.any():
        normalized = values[valid] / values[valid].sum(axis=1, keepdims=True)
        ordered = np.sort(normalized, axis=1)
        log_values = np.zeros_like(normalized)
        np.log(normalized, out=log_values, where=normalized > 0)
        derived[valid] = np.column_stack(
            (
                normalized.max(axis=1),
                normalized.min(axis=1),
                ordered[:, -1] - ordered[:, -2],
                -(normalized * log_values).sum(axis=1) / np.log(3),
                normalized[:, 0] - normalized[:, 1],
                normalized[:, 0] - normalized[:, 2],
                normalized[:, 1] - normalized[:, 2],
            )
        )
    for index, column in enumerate(outputs):
        frame[column] = derived[:, index]


_DERIVERS = {
    "count_context": _add_count_context,
    "pressure_context": _add_pressure_context,
    "hand_state_interactions": _add_hand_state_interactions,
    "pitcher_batter_gap": _add_pitcher_batter_gap,
    "recent_trend": _add_recent_trend,
    "pitchmix_shape": _add_pitchmix_shape,
}

_OUTPUT_COLUMNS = {
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


def add_row_feature_bundle(frame: pd.DataFrame, bundle: str) -> pd.DataFrame:
    """Return a copy with one sealed row-local feature bundle attached."""

    if bundle not in ROW_FEATURE_BUNDLES:
        raise RowFeatureError(f"unknown row feature bundle: {bundle}")
    collisions = [column for column in _OUTPUT_COLUMNS[bundle] if column in frame]
    if collisions:
        raise RowFeatureError(f"{bundle} output columns already exist: {collisions}")
    result = frame.copy()
    original_index = result.index.copy()
    _DERIVERS[bundle](result)
    if not result.index.equals(original_index):
        raise RowFeatureError(f"{bundle} changed row alignment")
    return result


def _identifier_text(value: object) -> str:
    if pd.isna(value):
        return MISSING_CATEGORY
    if isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _known_identifier_text(values: AbstractSet[object]) -> set[str]:
    result: set[str] = set()
    for value in values:
        if isinstance(value, str) and value == MISSING_CATEGORY:
            result.add(MISSING_CATEGORY)
        elif not pd.isna(value):
            result.add(_identifier_text(value))
    return result


def _boolean_segment(values: pd.Series, condition: pd.Series) -> pd.Series:
    result = pd.Series(MISSING_CATEGORY, index=values.index, dtype=object)
    present = values.notna()
    result.loc[present] = condition.loc[present].astype(bool)
    return result


def row_segment_labels(
    frame: pd.DataFrame,
    train_pitcher_ids: AbstractSet[object],
    train_batter_ids: AbstractSet[object],
) -> pd.DataFrame:
    """Return predeclared segment labels using only each row and frozen ID sets."""

    sources = (
        "game_type",
        "pitcher_hand",
        "batter_hand",
        "pitcher_id",
        "batter_id",
        "li",
        "inning",
        "runner_on_2b",
        "runner_on_3b",
    )
    _required(frame, sources, "row_segment_labels")
    leverage = _numeric(frame["li"], "li")
    inning = _numeric(frame["inning"], "inning")
    runner_on_2b = _numeric(frame["runner_on_2b"], "runner_on_2b")
    runner_on_3b = _numeric(frame["runner_on_3b"], "runner_on_3b")
    pitcher_known = _known_identifier_text(train_pitcher_ids)
    batter_known = _known_identifier_text(train_batter_ids)
    pitcher_ids = frame["pitcher_id"].map(_identifier_text)
    batter_ids = frame["batter_id"].map(_identifier_text)

    result = pd.DataFrame(index=frame.index)
    result["game_type_segment"] = _category(frame["game_type"]).astype(object)
    result["hand_matchup_segment"] = _join(
        frame["pitcher_hand"], frame["batter_hand"]
    )
    result["pitcher_id_segment"] = np.where(
        pitcher_ids.isin(pitcher_known), "known", "oov"
    )
    result["batter_id_segment"] = np.where(
        batter_ids.isin(batter_known), "known", "oov"
    )
    result["li_high"] = _boolean_segment(leverage, leverage >= 1.5)
    result["late_inning"] = _boolean_segment(inning, inning >= 7)
    runners_present = runner_on_2b.notna() & runner_on_3b.notna()
    runner_condition = runner_on_2b.ne(0) | runner_on_3b.ne(0)
    result["runner_in_scoring_position"] = _boolean_segment(
        runners_present.where(runners_present), runner_condition
    )
    return result
