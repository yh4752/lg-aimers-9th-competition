"""Fold-fitted preprocessing profiles for tree and deep tabular models."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from .row_features import ROW_FEATURE_BUNDLES, RowFeatureError, add_row_feature_bundle


TARGET_COLUMN = "control_success"
ROW_ID_COLUMN = "row_id"
MISSING_CATEGORY = "__MISSING__"
VALID_PROFILES = ("tree_native", "dl_standard", "dl_selective_transform")
BASE_CATEGORICAL_COLUMNS = (
    "game_type",
    "top_bottom",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
)
RECENT_MISSING_COLUMNS = tuple(
    f"asof_pitcher_prev{k}_game_{kind}_rate"
    for k in (1, 3, 5)
    for kind in ("success", "middle")
)
PITCHER_CAREER_MISSING_COLUMNS = tuple(
    f"asof_pitcher_{kind}_rate"
    for kind in (
        "ball",
        "breaking",
        "fastball",
        "middle",
        "offspeed",
        "reverse",
        "strike",
        "success",
    )
)
BATTER_CAREER_MISSING_COLUMNS = (
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
)
YEO_JOHNSON_COLUMNS = (
    "li",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "asof_pitcher_middle_rate",
    "asof_batter_middle_rate",
)
PASSTHROUGH_NUMERIC_COLUMNS = (
    "pitcher_recent_missing",
    "pitcher_career_missing",
    "batter_career_missing",
    "pitcher_id_oov",
    "batter_id_oov",
)
_SIMPLE_COMPONENTS = (
    "asof_count_log1p",
    "entity_frequency_log1p",
    "entity_frequency_and_oov",
    "grouped_missing_indicators",
    "hand_matchup",
    *ROW_FEATURE_BUNDLES,
    "count_state",
    "pitcher_team_win_expectancy",
)
_PITCHER_SMOOTH_RE = re.compile(r"pitcher_smooth_k([1-9][0-9]*)\Z")
_BATTER_SMOOTH_RE = re.compile(r"batter_smooth_k([1-9][0-9]*)\Z")


class PreprocessingError(ValueError):
    """Raised when a preprocessing state would be ambiguous or unsafe."""


@dataclass(frozen=True)
class PreprocessingSpec:
    profile: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessingState:
    spec: PreprocessingSpec
    source_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    numeric_median: Mapping[str, float]
    numeric_mean: Mapping[str, float]
    numeric_std: Mapping[str, float]
    yeo_johnson_lambda: Mapping[str, float]
    entity_frequency: Mapping[str, Mapping[str, int]]
    target_prior: float


def _component_key(component: str) -> tuple[int, int | str]:
    if component in _SIMPLE_COMPONENTS:
        return _SIMPLE_COMPONENTS.index(component), component
    match = _PITCHER_SMOOTH_RE.fullmatch(component)
    if match:
        return len(_SIMPLE_COMPONENTS), int(match.group(1))
    match = _BATTER_SMOOTH_RE.fullmatch(component)
    if match:
        return len(_SIMPLE_COMPONENTS) + 1, int(match.group(1))
    raise PreprocessingError(f"unknown preprocessing component: {component}")


def normalize_spec(spec: PreprocessingSpec) -> PreprocessingSpec:
    if spec.profile not in VALID_PROFILES:
        raise PreprocessingError(f"unknown preprocessing profile: {spec.profile}")
    if isinstance(spec.components, str):
        raise PreprocessingError("preprocessing components must be a tuple")
    components = tuple(sorted(set(spec.components), key=_component_key))
    if len(components) != len(spec.components):
        raise PreprocessingError("preprocessing components must be unique")
    pitcher = [item for item in components if _PITCHER_SMOOTH_RE.fullmatch(item)]
    batter = [item for item in components if _BATTER_SMOOTH_RE.fullmatch(item)]
    if len(pitcher) > 1 or len(batter) > 1:
        raise PreprocessingError("one pitcher K and one batter K are allowed")
    return PreprocessingSpec(spec.profile, components)


def _required(frame: pd.DataFrame, columns: tuple[str, ...], component: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise PreprocessingError(
            f"{component} requires source columns: {', '.join(missing)}"
        )


def _numeric(values: pd.Series, column: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & result.isna()
    if invalid.any():
        raise PreprocessingError(f"{column} contains a non-numeric value")
    return result.astype("float64")


def _category(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna(MISSING_CATEGORY).astype(object)


def _verify_duplicate_count(frame: pd.DataFrame) -> pd.DataFrame:
    left_name = "asof_pitcher_n"
    right_name = "asof_pitcher_pitchmix_n"
    if right_name not in frame:
        return frame.copy()
    _required(frame, (left_name,), "duplicate pitchmix_n check")
    left = _numeric(frame[left_name], left_name)
    right = _numeric(frame[right_name], right_name)
    same_missing = left.isna().equals(right.isna())
    observed = left.notna() & right.notna()
    same_values = np.array_equal(
        left.loc[observed].to_numpy(), right.loc[observed].to_numpy()
    )
    if not same_missing or not same_values:
        raise PreprocessingError(
            "asof_pitcher_pitchmix_n differs from asof_pitcher_n"
        )
    return frame.drop(columns=[right_name]).copy()


def _frequency(values: pd.Series) -> Mapping[str, int]:
    counts = _category(values).value_counts(dropna=False)
    return MappingProxyType({str(key): int(value) for key, value in counts.items()})


def _add_grouped_missing(frame: pd.DataFrame) -> None:
    groups = {
        "pitcher_recent_missing": RECENT_MISSING_COLUMNS,
        "pitcher_career_missing": PITCHER_CAREER_MISSING_COLUMNS,
        "batter_career_missing": BATTER_CAREER_MISSING_COLUMNS,
    }
    for output, columns in groups.items():
        _required(frame, columns, "grouped_missing_indicators")
        frame[output] = frame.loc[:, columns].isna().any(axis=1).astype("float64")


def _add_smoothing(
    frame: pd.DataFrame,
    *,
    entity: str,
    k: int,
    prior: float,
) -> None:
    count_name = f"asof_{entity}_n"
    rate_name = f"asof_{entity}_success_rate"
    _required(frame, (count_name, rate_name), f"{entity}_smooth_k{k}")
    count = _numeric(frame[count_name], count_name).to_numpy(dtype="float64")
    rate = _numeric(frame[rate_name], rate_name).to_numpy(dtype="float64")
    valid = np.isfinite(rate) & np.isfinite(count) & (count > 0)
    result = np.full(len(frame), prior, dtype="float64")
    result[valid] = (
        count[valid] * rate[valid] + float(k) * prior
    ) / (count[valid] + float(k))
    frame[f"{entity}_success_smooth_{k}"] = result


def _add_components(
    frame: pd.DataFrame,
    spec: PreprocessingSpec,
    *,
    entity_frequency: Mapping[str, Mapping[str, int]],
    target_prior: float,
) -> pd.DataFrame:
    result = frame.copy()
    for component in spec.components:
        if component == "asof_count_log1p":
            _required(result, ("asof_pitcher_n", "asof_batter_n"), component)
            for column in ("asof_pitcher_n", "asof_batter_n"):
                values = _numeric(result[column], column).clip(lower=0)
                result[f"{column}_log1p"] = np.log1p(values)
        elif component in {"entity_frequency_log1p", "entity_frequency_and_oov"}:
            _required(result, ("pitcher_id", "batter_id"), component)
            for entity in ("pitcher_id", "batter_id"):
                mapping = entity_frequency[entity]
                frequency = _category(result[entity]).map(mapping).fillna(0).astype(float)
                result[f"{entity}_frequency"] = frequency
                result[f"{entity}_frequency_log1p"] = np.log1p(frequency)
                if component == "entity_frequency_and_oov":
                    result[f"{entity}_oov"] = frequency.eq(0).astype("float64")
        elif component == "grouped_missing_indicators":
            _add_grouped_missing(result)
        elif component == "hand_matchup":
            _required(result, ("pitcher_hand", "batter_hand"), component)
            result["hand_matchup"] = (
                _category(result["pitcher_hand"])
                + "_"
                + _category(result["batter_hand"])
            )
        elif component in ROW_FEATURE_BUNDLES:
            try:
                result = add_row_feature_bundle(result, component)
            except RowFeatureError as error:
                raise PreprocessingError(str(error)) from error
        elif component == "count_state":
            _required(result, ("balls_before", "strikes_before"), component)
            balls = _numeric(result["balls_before"], "balls_before").astype("Int64")
            strikes = _numeric(result["strikes_before"], "strikes_before").astype("Int64")
            result["count_state"] = (
                balls.astype("string").fillna(MISSING_CATEGORY)
                + "_"
                + strikes.astype("string").fillna(MISSING_CATEGORY)
            ).astype(object)
        elif component == "pitcher_team_win_expectancy":
            _required(
                result,
                ("top_bottom", "home_win_expectancy", "away_win_expectancy"),
                component,
            )
            home = _numeric(result["home_win_expectancy"], "home_win_expectancy")
            away = _numeric(result["away_win_expectancy"], "away_win_expectancy")
            top = result["top_bottom"].astype("string").eq("T").fillna(False)
            result["pitcher_team_win_expectancy"] = np.where(top, home, away)
        else:
            match = _PITCHER_SMOOTH_RE.fullmatch(component)
            if match:
                _add_smoothing(
                    result,
                    entity="pitcher",
                    k=int(match.group(1)),
                    prior=target_prior,
                )
                continue
            match = _BATTER_SMOOTH_RE.fullmatch(component)
            if match:
                _add_smoothing(
                    result,
                    entity="batter",
                    k=int(match.group(1)),
                    prior=target_prior,
                )
                continue
            raise PreprocessingError(f"unknown preprocessing component: {component}")
    return result


def _categorical_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in frame.columns
        if column in BASE_CATEGORICAL_COLUMNS
        or column in {"hand_matchup", "count_state"}
        or not pd.api.types.is_numeric_dtype(frame[column])
    )


def _prepare_source(frame: pd.DataFrame) -> pd.DataFrame:
    source = _verify_duplicate_count(frame)
    return source.drop(
        columns=[
            column
            for column in (ROW_ID_COLUMN, TARGET_COLUMN)
            if column in source
        ]
    )


def _target_prior(frame: pd.DataFrame) -> float:
    if TARGET_COLUMN not in frame:
        raise PreprocessingError("fit requires control_success")
    target = _numeric(frame[TARGET_COLUMN], TARGET_COLUMN)
    if not np.isin(target.to_numpy(), (0.0, 1.0)).all():
        raise PreprocessingError("control_success must contain only 0 and 1")
    prior = float(target.mean())
    if not math.isfinite(prior):
        raise PreprocessingError("control_success mean must be finite")
    return prior


def fit_preprocessor(
    train: pd.DataFrame, spec: PreprocessingSpec
) -> tuple[PreprocessingState, pd.DataFrame]:
    """Fit a preprocessing state only from the supplied training rows."""

    normalized = normalize_spec(spec)
    prior = _target_prior(train)
    source = _prepare_source(train)
    entity_frequency: dict[str, Mapping[str, int]] = {}
    if {"entity_frequency_log1p", "entity_frequency_and_oov"}.intersection(
        normalized.components
    ):
        _required(source, ("pitcher_id", "batter_id"), "entity frequency")
        entity_frequency = {
            entity: _frequency(source[entity])
            for entity in ("pitcher_id", "batter_id")
        }
    prepared = _add_components(
        source,
        normalized,
        entity_frequency=entity_frequency,
        target_prior=prior,
    )
    categorical = _categorical_columns(prepared)
    numeric = tuple(column for column in prepared if column not in categorical)
    median: dict[str, float] = {}
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    lambdas: dict[str, float] = {}

    result = prepared.copy()
    for column in categorical:
        result[column] = _category(result[column])
    if normalized.profile != "tree_native":
        for column in numeric:
            values = _numeric(prepared[column], column).replace([np.inf, -np.inf], np.nan)
            if column in PASSTHROUGH_NUMERIC_COLUMNS:
                if values.isna().any() or not values.isin((0.0, 1.0)).all():
                    raise PreprocessingError(f"binary indicator is invalid: {column}")
                result[column] = values.to_numpy(dtype="float64")
                continue
            column_median = float(values.median()) if values.notna().any() else 0.0
            imputed = values.fillna(column_median).to_numpy(dtype="float64")
            if (
                normalized.profile == "dl_selective_transform"
                and column in YEO_JOHNSON_COLUMNS
                and np.unique(imputed).size > 1
            ):
                try:
                    transformed, fitted_lambda = stats.yeojohnson(imputed)
                except (ValueError, FloatingPointError) as error:
                    raise PreprocessingError(
                        f"Yeo-Johnson fit failed for {column}: {error}"
                    ) from error
                imputed = np.asarray(transformed, dtype="float64")
                lambdas[column] = float(fitted_lambda)
            column_mean = float(np.mean(imputed))
            column_std = float(np.std(imputed, ddof=0))
            if not math.isfinite(column_std) or column_std <= 0:
                column_std = 1.0
            if not math.isfinite(column_mean) or not np.isfinite(imputed).all():
                raise PreprocessingError(f"non-finite fitted numeric state: {column}")
            median[column] = column_median
            mean[column] = column_mean
            std[column] = column_std
            result[column] = (imputed - column_mean) / column_std

    state = PreprocessingState(
        spec=normalized,
        source_columns=tuple(source.columns),
        output_columns=tuple(result.columns),
        categorical_columns=categorical,
        numeric_columns=numeric,
        numeric_median=MappingProxyType(median),
        numeric_mean=MappingProxyType(mean),
        numeric_std=MappingProxyType(std),
        yeo_johnson_lambda=MappingProxyType(lambdas),
        entity_frequency=MappingProxyType(entity_frequency),
        target_prior=prior,
    )
    return state, result


def transform_preprocessor(
    frame: pd.DataFrame, state: PreprocessingState
) -> pd.DataFrame:
    """Transform rows without changing any fitted preprocessing value."""

    source = _prepare_source(frame)
    if tuple(source.columns) != state.source_columns:
        missing = sorted(set(state.source_columns) - set(source.columns))
        extra = sorted(set(source.columns) - set(state.source_columns))
        raise PreprocessingError(
            f"preprocessing source schema differs: missing={missing}, extra={extra}"
        )
    prepared = _add_components(
        source,
        state.spec,
        entity_frequency=state.entity_frequency,
        target_prior=state.target_prior,
    )
    if tuple(prepared.columns) != state.output_columns:
        raise PreprocessingError("preprocessing output schema differs from fitted state")
    result = prepared.copy()
    for column in state.categorical_columns:
        result[column] = _category(result[column])
    if state.spec.profile != "tree_native":
        for column in state.numeric_columns:
            values = _numeric(prepared[column], column).replace([np.inf, -np.inf], np.nan)
            if column in PASSTHROUGH_NUMERIC_COLUMNS:
                if values.isna().any() or not values.isin((0.0, 1.0)).all():
                    raise PreprocessingError(f"binary indicator is invalid: {column}")
                result[column] = values.to_numpy(dtype="float64")
                continue
            imputed = values.fillna(state.numeric_median[column]).to_numpy(dtype="float64")
            if column in state.yeo_johnson_lambda:
                try:
                    imputed = np.asarray(
                        stats.yeojohnson(
                            imputed, lmbda=state.yeo_johnson_lambda[column]
                        ),
                        dtype="float64",
                    )
                except (ValueError, FloatingPointError) as error:
                    raise PreprocessingError(
                        f"Yeo-Johnson transform failed for {column}: {error}"
                    ) from error
            transformed = (
                imputed - state.numeric_mean[column]
            ) / state.numeric_std[column]
            if not np.isfinite(transformed).all():
                raise PreprocessingError(f"non-finite transformed values: {column}")
            result[column] = transformed
    if len(result) != len(frame) or not result.index.equals(frame.index):
        raise PreprocessingError("preprocessing must preserve row count and index")
    return result
