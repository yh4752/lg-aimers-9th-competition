from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from experiments.temporal_portfolio.seasonal_features import (
    S1State,
    SeasonalFeatureError,
    build_training_s1,
    transform_s1,
)
from experiments.temporal_portfolio.trackman_pitcher import PitcherTrackmanState
from experiments.temporal_portfolio.trackman_pitcher import (
    PitcherTrackmanError,
    fit_pitcher_trackman,
)


class TreeFeatureError(ValueError):
    """Raised when tree features cross a cutoff or change their schema."""


class TreeFeatureSkip(TreeFeatureError):
    """Raised when one candidate-local optional feature gate is not met."""


_TARGET = "control_success"
_ROW_ID = "row_id"
_MISSING_CATEGORY = "__MISSING__"
_RAW_CATEGORICAL = (
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "top_bottom",
    "game_type",
    "base_state",
    "balls_before",
    "strikes_before",
    "outs_before",
)
_CROSS_CATEGORICAL = (
    "count_state",
    "hand_matchup",
    "pitcher_batter_hand",
    "pitcher_count",
    "pitcher_base",
    "pitcher_game_type",
    "batter_pitcher_hand",
    "team_count",
    "count_base",
    "hand_count",
    "inning_bin",
    "score_bin",
    "leverage_bin",
    "inning_score",
    "leverage_score",
    "leverage_runners_count",
)
_SUCCESS_RECENT = tuple(
    f"asof_pitcher_prev{games}_game_success_rate" for games in (1, 3, 5)
)
_MIDDLE_RECENT = tuple(
    f"asof_pitcher_prev{games}_game_middle_rate" for games in (1, 3, 5)
)
_REQUIRED = {
    _ROW_ID,
    "inning",
    "score_diff_pitcher_team",
    "li",
    "num_runners_on",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    *_RAW_CATEGORICAL,
    *_SUCCESS_RECENT,
    *_MIDDLE_RECENT,
}


@dataclass(frozen=True)
class TreeFeatureState:
    valid_year: int
    prior_rate: float
    categorical_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    s1_state: S1State
    trackman_state: PitcherTrackmanState | None
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class TreeFeatureBatch:
    frame: pd.DataFrame
    anchor: np.ndarray
    row_id: np.ndarray
    target: np.ndarray | None


def _validate_frame(rows: object, *, training: bool) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame:
        raise TreeFeatureError("tree feature rows must be an actual pandas DataFrame")
    if rows.columns.has_duplicates or any(type(column) is not str for column in rows.columns):
        raise TreeFeatureError("tree feature columns must be unique strings")
    missing = sorted(_REQUIRED.difference(rows.columns))
    if missing:
        raise TreeFeatureError(f"tree feature rows are missing required columns: {missing}")
    if training and _TARGET not in rows:
        raise TreeFeatureError("training rows are missing target")
    if not training and _TARGET in rows:
        raise TreeFeatureError("evaluation rows contain target")
    row_id = rows[_ROW_ID]
    if row_id.isna().any() or not row_id.is_unique:
        raise TreeFeatureError("row_id values must be non-null and unique")
    return rows.copy(deep=True)


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(rows[column], errors="coerce").astype("float64")
    if np.isinf(values.to_numpy()).any():
        raise TreeFeatureError(f"{column} contains infinity")
    return values


def _category(rows: pd.DataFrame, column: str) -> pd.Series:
    return rows[column].astype("string").fillna(_MISSING_CATEGORY).astype(object)


def _join(*parts: pd.Series) -> pd.Series:
    result = parts[0].astype("string").fillna(_MISSING_CATEGORY)
    for part in parts[1:]:
        result = result.str.cat(
            part.astype("string").fillna(_MISSING_CATEGORY), sep="|"
        )
    return result.astype(object)


def _row_local_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = rows.drop(columns=[_ROW_ID, _TARGET], errors="ignore").copy(deep=True)
    for column in _RAW_CATEGORICAL:
        frame[column] = _category(rows, column)

    frame["count_state"] = _join(
        _category(rows, "balls_before"), _category(rows, "strikes_before")
    )
    frame["hand_matchup"] = _join(
        _category(rows, "pitcher_hand"), _category(rows, "batter_hand")
    )
    frame["pitcher_batter_hand"] = _join(
        _category(rows, "pitcher_id"), _category(rows, "batter_hand")
    )
    frame["pitcher_count"] = _join(
        _category(rows, "pitcher_id"), frame["count_state"]
    )
    frame["pitcher_base"] = _join(
        _category(rows, "pitcher_id"), _category(rows, "base_state")
    )
    frame["pitcher_game_type"] = _join(
        _category(rows, "pitcher_id"), _category(rows, "game_type")
    )
    frame["batter_pitcher_hand"] = _join(
        _category(rows, "batter_id"), _category(rows, "pitcher_hand")
    )
    frame["team_count"] = _join(
        _category(rows, "pitcher_team_id"), frame["count_state"]
    )
    frame["count_base"] = _join(frame["count_state"], _category(rows, "base_state"))
    frame["hand_count"] = _join(frame["hand_matchup"], frame["count_state"])

    inning = _numeric(rows, "inning")
    score = _numeric(rows, "score_diff_pitcher_team")
    leverage = _numeric(rows, "li")
    inning_bin = pd.cut(
        inning,
        [-np.inf, 3, 6, 9, np.inf],
        labels=["early", "middle", "late", "extra"],
    )
    score_bin = pd.cut(
        score,
        [-np.inf, -3, -1, 1, 3, np.inf],
        labels=["far_behind", "behind", "close", "ahead", "far_ahead"],
    )
    leverage_bin = pd.cut(
        leverage,
        [-np.inf, 0.75, 1.5, 3, np.inf],
        labels=["low", "normal", "high", "extreme"],
    )
    frame["inning_bin"] = inning_bin.astype("string").fillna(_MISSING_CATEGORY).astype(object)
    frame["score_bin"] = score_bin.astype("string").fillna(_MISSING_CATEGORY).astype(object)
    frame["leverage_bin"] = leverage_bin.astype("string").fillna(_MISSING_CATEGORY).astype(object)
    frame["inning_score"] = _join(frame["inning_bin"], frame["score_bin"])
    frame["leverage_score"] = _join(frame["leverage_bin"], frame["score_bin"])
    frame["leverage_runners_count"] = _join(
        frame["leverage_bin"],
        _category(rows, "num_runners_on"),
        frame["count_state"],
    )

    recent_success = pd.concat(
        [_numeric(rows, column) for column in _SUCCESS_RECENT], axis=1
    )
    recent_middle = pd.concat(
        [_numeric(rows, column) for column in _MIDDLE_RECENT], axis=1
    )
    frame["recent_success_mean"] = recent_success.mean(axis=1)
    frame["recent_success_std"] = recent_success.std(axis=1, ddof=0)
    frame["recent_success_slope"] = (
        recent_success.iloc[:, 0] - recent_success.iloc[:, 2]
    ) / 4.0
    frame["recent_middle_mean"] = recent_middle.mean(axis=1)
    frame["recent_middle_slope"] = (
        recent_middle.iloc[:, 0] - recent_middle.iloc[:, 2]
    ) / 4.0

    pitcher_n = _numeric(rows, "asof_pitcher_n").clip(lower=0)
    batter_n = _numeric(rows, "asof_batter_n").clip(lower=0)
    pitcher_rate = _numeric(rows, "asof_pitcher_success_rate")
    batter_rate = _numeric(rows, "asof_batter_success_rate")
    frame["pitcher_reliability_100"] = pitcher_n / (pitcher_n + 100.0)
    frame["batter_reliability_100"] = batter_n / (batter_n + 100.0)
    frame["pitcher_batter_success_gap"] = pitcher_rate - batter_rate
    clipped_rate = pitcher_rate.clip(1e-5, 1 - 1e-5)
    frame["pitcher_success_logit"] = np.log(clipped_rate / (1.0 - clipped_rate))

    pitchmix = pd.concat(
        [
            _numeric(rows, "asof_pitcher_fastball_rate"),
            _numeric(rows, "asof_pitcher_breaking_rate"),
            _numeric(rows, "asof_pitcher_offspeed_rate"),
        ],
        axis=1,
    ).clip(lower=0)
    frame["pitchmix_entropy"] = -(pitchmix * np.log(pitchmix.clip(lower=1e-12))).sum(axis=1)
    runners = _numeric(rows, "num_runners_on").clip(lower=0)
    frame["pressure_index"] = (
        np.log1p(leverage.clip(lower=0))
        * (1.0 + score.abs())
        * (1.0 + runners / 3.0)
    )
    return frame, (*_RAW_CATEGORICAL, *_CROSS_CATEGORICAL)


def _append_s1(
    frame: pd.DataFrame,
    s1: pd.DataFrame,
    categorical: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    output = frame.copy(deep=True)
    for column in s1.columns:
        if column in output:
            raise TreeFeatureError(f"S1 feature collision: {column}")
        output[column] = s1[column].to_numpy(copy=True)
    s1_categories = tuple(s1.attrs.get("categorical_columns", ()))
    all_categories = tuple(dict.fromkeys((*categorical, *s1_categories)))
    for column in all_categories:
        output[column] = output[column].astype("string").fillna(_MISSING_CATEGORY).astype(object)
    for column in output.columns:
        if column in all_categories:
            continue
        values = pd.to_numeric(output[column], errors="coerce").astype("float32")
        if np.isinf(values.to_numpy()).any():
            raise TreeFeatureError(f"feature contains infinity: {column}")
        output[column] = values
    return output, all_categories


def _anchor(rows: pd.DataFrame, s1: pd.DataFrame, prior: float) -> np.ndarray:
    n = _numeric(rows, "asof_pitcher_n").clip(lower=0)
    career = _numeric(rows, "asof_pitcher_success_rate").fillna(prior)
    recent = pd.concat(
        [_numeric(rows, column) for column in _SUCCESS_RECENT], axis=1
    ).mean(axis=1).fillna(prior)
    season_n = pd.to_numeric(s1["season_pitcher_n"], errors="coerce").clip(lower=0)
    season_rate = pd.to_numeric(
        s1["season_pitcher_success_smooth_100"], errors="coerce"
    ).fillna(prior)
    career_weight = n / (n + 100.0)
    season_weight = 0.15 + 0.30 * season_n / (season_n + 80.0)
    career_anchor = prior + career_weight * (career - prior)
    anchor = np.clip(
        career_anchor
        + season_weight * (season_rate - career_anchor)
        + 0.10 * (recent - career_anchor),
        1e-5,
        1 - 1e-5,
    ).to_numpy(dtype="float64")
    if not np.isfinite(anchor).all():
        raise TreeFeatureError("anchor contains non-finite values")
    anchor.setflags(write=False)
    return anchor


def _s1_sha256(state: S1State) -> str:
    snapshot = state.snapshot
    digest = sha256()
    digest.update(f"{state.valid_year}|{state.prior_rate:.17g}\n".encode("utf-8"))
    digest.update(snapshot.pitcher.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    digest.update(snapshot.batter.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()


def _fit_trackman(
    rows: pd.DataFrame,
    history: pd.DataFrame | None,
    *,
    valid_year: int,
    minimum_coverage: float,
) -> PitcherTrackmanState:
    if type(history) is not pd.DataFrame:
        raise TreeFeatureSkip("TrackMan history is unavailable")
    if type(minimum_coverage) not in {float, int} or type(minimum_coverage) is bool:
        raise TreeFeatureError("minimum TrackMan coverage must be numeric")
    threshold = float(minimum_coverage)
    if not 0.0 <= threshold <= 1.0:
        raise TreeFeatureError("minimum TrackMan coverage must be between zero and one")
    try:
        state = fit_pitcher_trackman(rows, history, cutoff_year=valid_year - 1)
    except PitcherTrackmanError as error:
        raise TreeFeatureSkip(f"TrackMan unavailable: {error}") from error
    lookup = state.lookup
    accepted = pd.to_numeric(
        lookup["tm_match_accepted"], errors="coerce"
    ).fillna(0)
    coverage = float(accepted.eq(1).mean()) if len(accepted) else 0.0
    if coverage < threshold:
        raise TreeFeatureSkip(f"trackman_coverage={coverage:.6f}")
    return state


def _attach_trackman(
    frame: pd.DataFrame,
    pitcher_ids: pd.Series,
    state: PitcherTrackmanState,
) -> pd.DataFrame:
    output = frame.copy(deep=True)
    lookup = state.lookup.set_index("pitcher_id")
    added: list[str] = []
    for bundle_name in ("P0", "P1", "P2", "P3"):
        bundle = state.bundles[bundle_name].set_index("pitcher_id")
        for column in bundle.columns:
            if column in output:
                raise TreeFeatureError(f"TrackMan feature collision: {column}")
            output[column] = pd.to_numeric(
                pitcher_ids.map(bundle[column]), errors="coerce"
            ).astype("float32")
            added.append(column)
    accepted = pd.to_numeric(
        pitcher_ids.map(lookup["tm_match_accepted"]), errors="coerce"
    )
    output["tm_pitcher_mapping_missing"] = accepted.ne(1).astype("float32")
    if len(added) != len(set(added)):
        raise TreeFeatureError("TrackMan bundles overlap")
    return output


def fit_tree_features(
    train: pd.DataFrame,
    history: pd.DataFrame | None,
    *,
    valid_year: int,
    use_trackman: bool,
    minimum_trackman_coverage: float = 0.30,
) -> tuple[TreeFeatureState, TreeFeatureBatch]:
    if type(use_trackman) is not bool:
        raise TreeFeatureError("use_trackman must be an exact bool")
    rows = _validate_frame(train, training=True)
    target = pd.to_numeric(rows[_TARGET], errors="coerce")
    if not target.isin([0, 1]).all():
        raise TreeFeatureError("training target must be binary")
    try:
        s1_state, training_s1 = build_training_s1(rows, valid_year=valid_year)
    except SeasonalFeatureError as error:
        raise TreeFeatureError(str(error)) from error
    raw, categorical = _row_local_features(rows)
    frame, categorical = _append_s1(raw, training_s1, categorical)
    trackman_state = (
        _fit_trackman(
            rows,
            history,
            valid_year=valid_year,
            minimum_coverage=minimum_trackman_coverage,
        )
        if use_trackman
        else None
    )
    if trackman_state is not None:
        frame = _attach_trackman(frame, rows["pitcher_id"], trackman_state)
    source_hashes = {"S1": _s1_sha256(s1_state)}
    if trackman_state is not None:
        source_hashes["TrackMan"] = trackman_state.lookup_sha256
    state = TreeFeatureState(
        valid_year=valid_year,
        prior_rate=s1_state.prior_rate,
        categorical_columns=categorical,
        feature_columns=tuple(frame.columns),
        s1_state=s1_state,
        trackman_state=trackman_state,
        source_hashes=MappingProxyType(source_hashes),
    )
    row_id = rows[_ROW_ID].astype(str).to_numpy(copy=True)
    row_id.setflags(write=False)
    target_array = target.to_numpy(dtype="int8", copy=True)
    target_array.setflags(write=False)
    return state, TreeFeatureBatch(
        frame=frame,
        anchor=_anchor(rows, training_s1, state.prior_rate),
        row_id=row_id,
        target=target_array,
    )


def transform_tree_features(
    rows: pd.DataFrame,
    state: TreeFeatureState,
) -> TreeFeatureBatch:
    if type(state) is not TreeFeatureState:
        raise TreeFeatureError("tree feature state type differs")
    source = _validate_frame(rows, training=False)
    try:
        s1 = transform_s1(source, state.s1_state)
    except SeasonalFeatureError as error:
        raise TreeFeatureError(str(error)) from error
    raw, categorical = _row_local_features(source)
    frame, categorical = _append_s1(raw, s1, categorical)
    if state.trackman_state is not None:
        frame = _attach_trackman(frame, source["pitcher_id"], state.trackman_state)
    if tuple(frame.columns) != state.feature_columns:
        raise TreeFeatureError("evaluation feature schema differs")
    if categorical != state.categorical_columns:
        raise TreeFeatureError("evaluation categorical schema differs")
    row_id = source[_ROW_ID].astype(str).to_numpy(copy=True)
    row_id.setflags(write=False)
    return TreeFeatureBatch(
        frame=frame,
        anchor=_anchor(source, s1, state.prior_rate),
        row_id=row_id,
        target=None,
    )
