from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from experiments.temporal_portfolio.trackman_batter import (
    BATTER_EXPOSURE_COLUMNS,
    BatterTrackmanError,
    BatterTrackmanState,
    attach_batter_exposure,
    fit_batter_trackman,
)
from experiments.tree_expert.features import (
    TreeFeatureBatch,
    TreeFeatureError,
    TreeFeatureSkip,
    TreeFeatureState,
    fit_tree_features,
    transform_tree_features,
)


class DirectFeatureError(ValueError):
    pass


_TARGET = "control_success"
_MISSING = "__MISSING__"
_HIGH_CTR_COLUMNS = (
    "pitcher_batter",
    "pitcher_batter_count",
    "pitcher_batter_base",
    "pitcher_batter_hands",
    "batter_count",
    "team_matchup_game_type",
    "count_inning_score",
    "count_base_game_type",
    "pitcher_leverage_count",
    "batter_pitcher_hand_count",
    "leverage_runners_count_game_type",
)


@dataclass(frozen=True)
class DirectFeatureState:
    valid_year: int
    tree_state: TreeFeatureState
    batter_trackman_state: BatterTrackmanState | None
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    high_ctr_columns: tuple[str, ...]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class DirectFeatureBatch:
    frame: pd.DataFrame
    row_id: np.ndarray
    target: np.ndarray | None
    season: np.ndarray
    game_type: np.ndarray


def smoothed_rate(
    count: pd.Series,
    total: pd.Series,
    prior: float,
    strength: float,
) -> pd.Series:
    if not np.isfinite(prior) or not np.isfinite(strength) or strength <= 0:
        raise DirectFeatureError("smoothing parameters differ")
    numerator = pd.to_numeric(count, errors="coerce").fillna(0).clip(lower=0)
    denominator = pd.to_numeric(total, errors="coerce").fillna(0).clip(lower=0)
    return (numerator + strength * prior) / (denominator + strength)


def _category(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows:
        raise DirectFeatureError(f"interaction source is absent: {column}")
    return rows[column].astype("string").fillna(_MISSING)


def _join(*parts: pd.Series) -> pd.Series:
    result = parts[0].astype("string").fillna(_MISSING)
    for part in parts[1:]:
        result = result.str.cat(part.astype("string").fillna(_MISSING), sep="|")
    return result.astype(object)


def _append_interactions(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy(deep=True)
    count = _join(_category(rows, "balls_before"), _category(rows, "strikes_before"))
    pitcher = _category(rows, "pitcher_id")
    batter = _category(rows, "batter_id")
    pitcher_batter = _join(pitcher, batter)
    additions = {
        "pitcher_batter": pitcher_batter,
        "pitcher_batter_count": _join(pitcher_batter, count),
        "pitcher_batter_base": _join(pitcher_batter, _category(rows, "base_state")),
        "pitcher_batter_hands": _join(
            pitcher_batter,
            _category(rows, "pitcher_hand"),
            _category(rows, "batter_hand"),
        ),
        "batter_count": _join(batter, count),
        "team_matchup_game_type": _join(
            _category(rows, "pitcher_team_id"),
            _category(rows, "batter_team_id"),
            _category(rows, "game_type"),
        ),
        "count_inning_score": _join(count, frame["inning_bin"], frame["score_bin"]),
        "count_base_game_type": _join(
            count,
            _category(rows, "base_state"),
            _category(rows, "game_type"),
        ),
        "pitcher_leverage_count": _join(pitcher, frame["leverage_bin"], count),
        "batter_pitcher_hand_count": _join(
            batter,
            _category(rows, "pitcher_hand"),
            count,
        ),
        "leverage_runners_count_game_type": _join(
            frame["leverage_bin"],
            _category(rows, "num_runners_on"),
            count,
            _category(rows, "game_type"),
        ),
    }
    for name, values in additions.items():
        if name in output:
            raise DirectFeatureError(f"direct feature collision: {name}")
        output[name] = values
    return output


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise DirectFeatureError(f"numeric source is absent: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    if np.isinf(values.to_numpy()).any():
        raise DirectFeatureError(f"numeric source contains infinity: {column}")
    return values


def _append_current_season_features(frame: pd.DataFrame, prior: float) -> pd.DataFrame:
    output = frame.copy(deep=True)
    pitcher_n = _numeric(frame, "asof_pitcher_n").fillna(0).clip(lower=0)
    batter_n = _numeric(frame, "asof_batter_n").fillna(0).clip(lower=0)
    season_pitcher_n = _numeric(frame, "season_pitcher_n").fillna(0).clip(lower=0)
    season_batter_n = _numeric(frame, "season_batter_n").fillna(0).clip(lower=0)
    pitcher_rate = _numeric(frame, "asof_pitcher_success_rate").fillna(prior)
    batter_rate = _numeric(frame, "asof_batter_success_rate").fillna(prior)
    season_pitcher_rate = _numeric(frame, "season_pitcher_success_smooth_100").fillna(prior)
    season_batter_rate = _numeric(frame, "season_batter_success_smooth_100").fillna(prior)
    output["current_pitcher_n_delta"] = (pitcher_n - season_pitcher_n).clip(lower=0)
    output["current_batter_n_delta"] = (batter_n - season_batter_n).clip(lower=0)
    output["current_pitcher_rate_delta"] = pitcher_rate - season_pitcher_rate
    output["current_batter_rate_delta"] = batter_rate - season_batter_rate
    output["current_pitcher_success_delta"] = (
        pitcher_n * pitcher_rate - season_pitcher_n * season_pitcher_rate
    )
    output["current_batter_success_delta"] = (
        batter_n * batter_rate - season_batter_n * season_batter_rate
    )
    output["current_pitcher_rate_shrunk"] = smoothed_rate(
        pitcher_n * pitcher_rate,
        pitcher_n,
        prior,
        100.0,
    )
    output["current_batter_rate_shrunk"] = smoothed_rate(
        batter_n * batter_rate,
        batter_n,
        prior,
        100.0,
    )
    output["pitcher_recent_long_gap"] = (
        _numeric(frame, "recent_success_mean").fillna(prior) - pitcher_rate
    )
    output["pitcher_middle_long_gap"] = (
        _numeric(frame, "recent_middle_mean").fillna(0)
        - _numeric(frame, "asof_pitcher_middle_rate").fillna(0)
    )
    return output


def _fit_batter_state(
    rows: pd.DataFrame,
    history: pd.DataFrame | None,
    valid_year: int,
) -> BatterTrackmanState | None:
    if type(history) is not pd.DataFrame:
        return None
    try:
        state = fit_batter_trackman(rows, history, cutoff_year=valid_year - 1)
    except BatterTrackmanError:
        return None
    return state if state.status in {"exploratory", "eligible"} else None


def _append_batter_trackman(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    state: BatterTrackmanState | None,
) -> pd.DataFrame:
    output = frame.copy(deep=True)
    if state is None:
        output["tm_batter_unavailable"] = np.ones(len(output), dtype="float32")
        return output
    try:
        attached = attach_batter_exposure(rows, state)
    except BatterTrackmanError as error:
        raise DirectFeatureError(f"batter TrackMan transform failed: {error}") from error
    for column in BATTER_EXPOSURE_COLUMNS:
        if column == "batter_id":
            continue
        if column in output:
            raise DirectFeatureError(f"batter TrackMan feature collision: {column}")
        output[column] = attached[column].to_numpy(copy=True)
    output["tm_batter_unavailable"] = np.zeros(len(output), dtype="float32")
    return output


def _finalize(
    frame: pd.DataFrame,
    categorical_columns: tuple[str, ...],
) -> pd.DataFrame:
    output = frame.copy(deep=True)
    categorical = set(categorical_columns)
    for column in output:
        if column in categorical:
            output[column] = output[column].astype("string").fillna(_MISSING).astype(object)
            continue
        values = pd.to_numeric(output[column], errors="coerce").fillna(0).astype("float32")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise DirectFeatureError(f"feature contains non-finite values: {column}")
        output[column] = values
    return output


def _batch(
    rows: pd.DataFrame,
    tree_batch: TreeFeatureBatch,
    tree_state: TreeFeatureState,
    batter_state: BatterTrackmanState | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = _append_interactions(tree_batch.frame, rows)
    frame = _append_current_season_features(frame, tree_state.prior_rate)
    frame = _append_batter_trackman(frame, rows, batter_state)
    categorical = tuple(dict.fromkeys((*tree_state.categorical_columns, *_HIGH_CTR_COLUMNS)))
    return _finalize(frame, categorical), categorical


def _arrays(
    rows: pd.DataFrame,
    tree_batch: TreeFeatureBatch,
    frame: pd.DataFrame,
) -> DirectFeatureBatch:
    season = pd.to_numeric(rows["season"], errors="coerce").to_numpy(dtype="int16", copy=True)
    if len(season) != len(rows):
        raise DirectFeatureError("season values differ")
    game_type = rows["game_type"].astype(str).to_numpy(copy=True)
    season.setflags(write=False)
    game_type.setflags(write=False)
    return DirectFeatureBatch(
        frame=frame,
        row_id=tree_batch.row_id,
        target=tree_batch.target,
        season=season,
        game_type=game_type,
    )


def fit_direct_features(
    train: pd.DataFrame,
    history: pd.DataFrame | None,
    *,
    valid_year: int,
) -> tuple[DirectFeatureState, DirectFeatureBatch]:
    if type(train) is not pd.DataFrame or "season" not in train:
        raise DirectFeatureError("training rows differ")
    if type(valid_year) is not int or isinstance(valid_year, bool):
        raise DirectFeatureError("valid_year differs")
    seasons = pd.to_numeric(train["season"], errors="coerce")
    if seasons.isna().any() or seasons.ge(valid_year).any():
        raise DirectFeatureError("training rows reach validation season")
    try:
        try:
            tree_state, tree_batch = fit_tree_features(
                train,
                history,
                valid_year=valid_year,
                use_trackman=True,
            )
        except TreeFeatureSkip:
            tree_state, tree_batch = fit_tree_features(
                train,
                history,
                valid_year=valid_year,
                use_trackman=False,
            )
    except TreeFeatureError as error:
        raise DirectFeatureError(str(error)) from error
    batter_state = _fit_batter_state(train, history, valid_year)
    frame, categorical = _batch(train, tree_batch, tree_state, batter_state)
    source_hashes = dict(tree_state.source_hashes)
    source_hashes["BatterTrackMan"] = (
        batter_state.exposure_sha256 if batter_state is not None else "unavailable"
    )
    state = DirectFeatureState(
        valid_year=valid_year,
        tree_state=tree_state,
        batter_trackman_state=batter_state,
        feature_columns=tuple(frame.columns),
        categorical_columns=categorical,
        high_ctr_columns=_HIGH_CTR_COLUMNS,
        source_hashes=MappingProxyType(source_hashes),
    )
    return state, _arrays(train, tree_batch, frame)


def transform_direct_features(
    rows: pd.DataFrame,
    state: DirectFeatureState,
) -> DirectFeatureBatch:
    if type(state) is not DirectFeatureState:
        raise DirectFeatureError("direct feature state type differs")
    if type(rows) is not pd.DataFrame:
        raise DirectFeatureError("evaluation rows differ")
    if _TARGET in rows:
        raise DirectFeatureError("evaluation rows contain target")
    try:
        tree_batch = transform_tree_features(rows, state.tree_state)
    except TreeFeatureError as error:
        raise DirectFeatureError(str(error)) from error
    frame, categorical = _batch(
        rows,
        tree_batch,
        state.tree_state,
        state.batter_trackman_state,
    )
    if tuple(frame.columns) != state.feature_columns:
        raise DirectFeatureError("evaluation feature schema differs")
    if categorical != state.categorical_columns:
        raise DirectFeatureError("evaluation categorical schema differs")
    return _arrays(rows, tree_batch, frame)
