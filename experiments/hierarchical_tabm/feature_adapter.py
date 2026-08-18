from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.common import ModelMetadata
from experiments.independent_dl.preprocessing import (
    PreprocessingSpec,
    PreprocessingState,
    fit_preprocessor,
    transform_preprocessor,
)

from .context_features import (
    ContextState,
    context_state_from_payload,
    context_state_payload,
    context_state_sha256,
    transform_frozen,
    transform_training_loo,
)


class HierarchicalFeatureError(ValueError):
    """Raised when the hierarchy-to-TabM feature bridge is not reproducible."""


EXPECTED_HIERARCHY_COLUMNS = (
    "hier_context_rate",
    "hier_context_logit",
    "hier_pitcher_reliability",
    "hier_batter_reliability",
    "hier_pitcher_context_gap",
    "hier_batter_context_gap",
    "hier_pitcher_weighted_gap",
    "hier_batter_weighted_gap",
)
MISSING_CATEGORY = "__MISSING__"
_ENTITY_COLUMNS = {
    "pitcher": ("asof_pitcher_n", "asof_pitcher_success_rate"),
    "batter": ("asof_batter_n", "asof_batter_success_rate"),
}


@dataclass(frozen=True)
class HierarchicalFeatureState:
    context: ContextState
    preprocessing_payload: Mapping[str, object]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_maps: Mapping[str, Mapping[str, int]]
    pitcher_k: float
    batter_k: float


@dataclass(frozen=True)
class PreparedFold:
    train: FeatureBatch
    valid: FeatureBatch
    state: HierarchicalFeatureState
    metadata: ModelMetadata


def _positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HierarchicalFeatureError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise HierarchicalFeatureError(f"{label} must be finite and positive")
    return result


def _numeric(values: pd.Series, column: str) -> np.ndarray:
    converted = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & converted.isna()
    if invalid.any():
        raise HierarchicalFeatureError(f"{column} contains a non-numeric value")
    return converted.to_numpy(dtype="float64")


def _entity_values(
    frame: pd.DataFrame, entity: str, context: np.ndarray, k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count_column, rate_column = _ENTITY_COLUMNS[entity]
    missing = [column for column in (count_column, rate_column) if column not in frame]
    if missing:
        raise HierarchicalFeatureError(f"missing required columns: {', '.join(missing)}")
    counts = _numeric(frame[count_column], count_column)
    if np.isinf(counts).any() or (np.isfinite(counts) & (counts < 0)).any():
        raise HierarchicalFeatureError(f"{count_column} must be nonnegative and finite")
    finite_counts = np.nan_to_num(counts, nan=0.0)
    if not np.equal(finite_counts, np.floor(finite_counts)).all():
        raise HierarchicalFeatureError(f"{count_column} must contain integer counts")
    rates = _numeric(frame[rate_column], rate_column)
    observed = np.isfinite(rates)
    if np.isinf(rates).any() or ((rates[observed] < 0) | (rates[observed] > 1)).any():
        raise HierarchicalFeatureError(f"{rate_column} must be in [0, 1] when present")
    reliability = finite_counts / (finite_counts + k)
    effective_rate = np.where(observed, rates, context)
    gap = effective_rate - context
    return reliability, gap, reliability * gap


def attach_hierarchical_numeric(
    frame: pd.DataFrame,
    context_rate: Sequence[float],
    *,
    pitcher_k: float,
    batter_k: float,
) -> pd.DataFrame:
    collisions = sorted(set(EXPECTED_HIERARCHY_COLUMNS).intersection(frame.columns))
    if collisions:
        raise HierarchicalFeatureError(f"hierarchical output collision: {collisions}")
    pitcher_smoothing = _positive_finite(pitcher_k, "pitcher_k")
    batter_smoothing = _positive_finite(batter_k, "batter_k")
    if isinstance(context_rate, pd.Series) and not context_rate.index.equals(frame.index):
        raise HierarchicalFeatureError("context index differs from frame index")
    context = np.asarray(context_rate, dtype="float64")
    if context.ndim != 1 or len(context) != len(frame):
        raise HierarchicalFeatureError("context length differs from frame")
    if not np.isfinite(context).all() or ((context < 0) | (context > 1)).any():
        raise HierarchicalFeatureError("context rate must be finite and in [0, 1]")
    pitcher_rel, pitcher_gap, pitcher_weighted = _entity_values(
        frame, "pitcher", context, pitcher_smoothing
    )
    batter_rel, batter_gap, batter_weighted = _entity_values(
        frame, "batter", context, batter_smoothing
    )
    clipped = np.clip(context, 1e-6, 1.0 - 1e-6)
    output = frame.copy()
    values = (
        context,
        np.log(clipped / (1.0 - clipped)),
        pitcher_rel,
        batter_rel,
        pitcher_gap,
        batter_gap,
        pitcher_weighted,
        batter_weighted,
    )
    for column, column_values in zip(EXPECTED_HIERARCHY_COLUMNS, values, strict=True):
        if not np.isfinite(column_values).all():
            raise HierarchicalFeatureError(f"non-finite hierarchical feature: {column}")
        output[column] = column_values
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise HierarchicalFeatureError("hierarchical features must preserve rows and index")
    return output


def _preprocessing_payload(state: PreprocessingState) -> dict[str, object]:
    return {
        "spec": {"profile": state.spec.profile, "components": list(state.spec.components)},
        "source_columns": list(state.source_columns),
        "output_columns": list(state.output_columns),
        "categorical_columns": list(state.categorical_columns),
        "numeric_columns": list(state.numeric_columns),
        "numeric_median": dict(state.numeric_median),
        "numeric_mean": dict(state.numeric_mean),
        "numeric_std": dict(state.numeric_std),
        "yeo_johnson_lambda": dict(state.yeo_johnson_lambda),
        "entity_frequency": {
            entity: dict(mapping) for entity, mapping in state.entity_frequency.items()
        },
        "target_prior": state.target_prior,
    }


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise HierarchicalFeatureError(f"{label} keys differ")


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise HierarchicalFeatureError(f"{label} must be a string list")
    if len(value) != len(set(value)):
        raise HierarchicalFeatureError(f"{label} contains duplicates")
    return tuple(value)


def _float_mapping(value: object, label: str, *, positive: bool = False) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise HierarchicalFeatureError(f"{label} is invalid")
    result: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise HierarchicalFeatureError(f"{label} contains an invalid value")
        number = float(item)
        if not math.isfinite(number) or (positive and number <= 0):
            raise HierarchicalFeatureError(f"{label} contains an invalid value")
        result[key] = number
    return MappingProxyType(result)


def _preprocessing_from_payload(payload: object) -> PreprocessingState:
    if not isinstance(payload, Mapping):
        raise HierarchicalFeatureError("preprocessing payload is invalid")
    _exact_keys(
        payload,
        {
            "spec", "source_columns", "output_columns", "categorical_columns",
            "numeric_columns", "numeric_median", "numeric_mean", "numeric_std",
            "yeo_johnson_lambda", "entity_frequency", "target_prior",
        },
        "preprocessing",
    )
    spec = payload["spec"]
    if not isinstance(spec, Mapping):
        raise HierarchicalFeatureError("preprocessing spec is invalid")
    _exact_keys(spec, {"profile", "components"}, "preprocessing spec")
    if spec["profile"] != "dl_standard" or spec["components"] != ["hand_matchup"]:
        raise HierarchicalFeatureError("preprocessing spec differs")
    source = _string_list(payload["source_columns"], "source_columns")
    output = _string_list(payload["output_columns"], "output_columns")
    categorical = _string_list(payload["categorical_columns"], "categorical_columns")
    numeric = _string_list(payload["numeric_columns"], "numeric_columns")
    if tuple(column for column in output if column in categorical) != categorical:
        raise HierarchicalFeatureError("categorical output order differs")
    if tuple(column for column in output if column in numeric) != numeric:
        raise HierarchicalFeatureError("numeric output order differs")
    if set(output) != set(categorical).union(numeric):
        raise HierarchicalFeatureError("preprocessing output partition differs")
    median = _float_mapping(payload["numeric_median"], "numeric_median")
    mean = _float_mapping(payload["numeric_mean"], "numeric_mean")
    std = _float_mapping(payload["numeric_std"], "numeric_std", positive=True)
    lambdas = _float_mapping(payload["yeo_johnson_lambda"], "yeo_johnson_lambda")
    if set(median) != set(mean) or set(mean) != set(std) or not set(mean).issubset(numeric):
        raise HierarchicalFeatureError("numeric preprocessing state differs")
    if not set(lambdas).issubset(numeric):
        raise HierarchicalFeatureError("Yeo-Johnson state differs")
    frequencies = payload["entity_frequency"]
    if not isinstance(frequencies, Mapping):
        raise HierarchicalFeatureError("entity frequencies are invalid")
    frozen_frequencies: dict[str, Mapping[str, int]] = {}
    for entity, mapping in frequencies.items():
        if not isinstance(entity, str) or not isinstance(mapping, Mapping):
            raise HierarchicalFeatureError("entity frequencies are invalid")
        parsed: dict[str, int] = {}
        for key, count in mapping.items():
            if not isinstance(key, str) or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise HierarchicalFeatureError("entity frequencies are invalid")
            parsed[key] = count
        frozen_frequencies[entity] = MappingProxyType(parsed)
    prior = payload["target_prior"]
    if isinstance(prior, bool) or not isinstance(prior, (int, float)):
        raise HierarchicalFeatureError("target_prior is invalid")
    prior_float = float(prior)
    if not math.isfinite(prior_float) or not 0 <= prior_float <= 1:
        raise HierarchicalFeatureError("target_prior is invalid")
    return PreprocessingState(
        PreprocessingSpec("dl_standard", ("hand_matchup",)),
        source,
        output,
        categorical,
        numeric,
        median,
        mean,
        std,
        lambdas,
        MappingProxyType(frozen_frequencies),
        prior_float,
    )


def _category_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna(MISSING_CATEGORY).astype(str)


def _category_maps(frame: pd.DataFrame, columns: tuple[str, ...]) -> Mapping[str, Mapping[str, int]]:
    result: dict[str, Mapping[str, int]] = {}
    for column in columns:
        levels = sorted(set(_category_text(frame[column]).tolist()))
        result[column] = MappingProxyType(
            {level: position + 1 for position, level in enumerate(levels)}
        )
    return MappingProxyType(result)


def _build_batch(
    source: pd.DataFrame,
    prepared: pd.DataFrame,
    *,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    category_maps: Mapping[str, Mapping[str, int]],
) -> FeatureBatch:
    if "row_id" not in source or source["row_id"].isna().any():
        raise HierarchicalFeatureError("row_id must be present and non-null")
    row_id = source["row_id"].astype(str).to_numpy(dtype=str)
    if len(set(row_id.tolist())) != len(row_id):
        raise HierarchicalFeatureError("row_id must be unique")
    if "season" not in source:
        raise HierarchicalFeatureError("season is missing")
    season_values = _numeric(source["season"], "season")
    if not np.isfinite(season_values).all() or not np.equal(season_values, np.floor(season_values)).all():
        raise HierarchicalFeatureError("season must contain finite integers")
    numeric = prepared.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    x_num = numeric.to_numpy(dtype="float32")
    if not np.isfinite(x_num).all():
        raise HierarchicalFeatureError("preprocessed numerical features must be finite")
    encoded = [
        _category_text(prepared[column]).map(category_maps[column]).fillna(0).to_numpy(dtype="int64")
        for column in categorical_columns
    ]
    x_cat = (
        np.column_stack(encoded).astype("int64", copy=False)
        if encoded else np.empty((len(source), 0), dtype="int64")
    )
    target = None
    if "control_success" in source:
        target_values = _numeric(source["control_success"], "control_success")
        if not np.isfinite(target_values).all() or not np.isin(target_values, (0.0, 1.0)).all():
            raise HierarchicalFeatureError("control_success must contain only 0 and 1")
        target = target_values.astype("float32")
    game_type = (
        _category_text(source["game_type"]).to_numpy(dtype=str)
        if "game_type" in source else np.full(len(source), MISSING_CATEGORY, dtype=str)
    )
    return FeatureBatch(row_id, season_values.astype("int64"), game_type, x_num, x_cat, target)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _feature_body(state: HierarchicalFeatureState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "context": context_state_payload(state.context),
        "context_sha256": context_state_sha256(state.context),
        "preprocessing": dict(state.preprocessing_payload),
        "numeric_columns": list(state.numeric_columns),
        "categorical_columns": list(state.categorical_columns),
        "category_maps": {
            column: dict(state.category_maps[column]) for column in state.categorical_columns
        },
        "pitcher_k": state.pitcher_k,
        "batter_k": state.batter_k,
    }


def feature_state_payload(state: HierarchicalFeatureState) -> dict[str, object]:
    body = _feature_body(state)
    return {**body, "state_sha256": sha256(_canonical_json(body)).hexdigest()}


def feature_state_sha256(state: HierarchicalFeatureState) -> str:
    return str(feature_state_payload(state)["state_sha256"])


def feature_state_from_payload(payload: Mapping[str, object]) -> HierarchicalFeatureState:
    if not isinstance(payload, Mapping):
        raise HierarchicalFeatureError("feature state payload is invalid")
    expected = {
        "schema_version", "context", "context_sha256", "preprocessing",
        "numeric_columns", "categorical_columns", "category_maps", "pitcher_k",
        "batter_k", "state_sha256",
    }
    _exact_keys(payload, expected, "feature state")
    if payload["schema_version"] != 1:
        raise HierarchicalFeatureError("feature state schema differs")
    state_digest = payload["state_sha256"]
    body = {key: payload[key] for key in payload if key != "state_sha256"}
    if not isinstance(state_digest, str) or state_digest != sha256(_canonical_json(body)).hexdigest():
        raise HierarchicalFeatureError("feature state digest differs")
    context_payload = payload["context"]
    if not isinstance(context_payload, Mapping):
        raise HierarchicalFeatureError("context payload is invalid")
    try:
        context = context_state_from_payload(context_payload)
    except (TypeError, ValueError, KeyError) as error:
        raise HierarchicalFeatureError(f"context payload is invalid: {error}") from error
    if payload["context_sha256"] != context_state_sha256(context):
        raise HierarchicalFeatureError("context digest differs")
    preprocessing = _preprocessing_from_payload(payload["preprocessing"])
    numeric = _string_list(payload["numeric_columns"], "numeric_columns")
    categorical = _string_list(payload["categorical_columns"], "categorical_columns")
    if numeric != preprocessing.numeric_columns or categorical != preprocessing.categorical_columns:
        raise HierarchicalFeatureError("feature columns differ from preprocessing")
    maps_payload = payload["category_maps"]
    if not isinstance(maps_payload, Mapping) or tuple(maps_payload) != categorical:
        raise HierarchicalFeatureError("category map columns differ")
    maps: dict[str, Mapping[str, int]] = {}
    for column in categorical:
        mapping = maps_payload[column]
        if not isinstance(mapping, Mapping) or any(not isinstance(key, str) for key in mapping):
            raise HierarchicalFeatureError("category map is invalid")
        parsed = dict(mapping)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in parsed.values()):
            raise HierarchicalFeatureError("category indices are invalid")
        if list(parsed) != sorted(parsed) or list(parsed.values()) != list(range(1, len(parsed) + 1)):
            raise HierarchicalFeatureError("category indices are invalid")
        maps[column] = MappingProxyType(parsed)
    state = HierarchicalFeatureState(
        context,
        MappingProxyType(_preprocessing_payload(preprocessing)),
        numeric,
        categorical,
        MappingProxyType(maps),
        _positive_finite(payload["pitcher_k"], "pitcher_k"),
        _positive_finite(payload["batter_k"], "batter_k"),
    )
    if feature_state_payload(state) != dict(payload):
        raise HierarchicalFeatureError("feature state is not canonical")
    return state


def _attach(frame: pd.DataFrame, rates: pd.DataFrame, *, pitcher_k: float, batter_k: float) -> pd.DataFrame:
    return attach_hierarchical_numeric(
        frame,
        rates["hier_context_rate"],
        pitcher_k=pitcher_k,
        batter_k=batter_k,
    )


def prepare_fold(
    fit_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    *,
    context_state: ContextState,
    pitcher_k: float,
    batter_k: float,
) -> PreparedFold:
    pitcher_smoothing = _positive_finite(pitcher_k, "pitcher_k")
    batter_smoothing = _positive_finite(batter_k, "batter_k")
    fit_source = _attach(
        fit_rows,
        transform_training_loo(fit_rows, context_state),
        pitcher_k=pitcher_smoothing,
        batter_k=batter_smoothing,
    )
    valid_source = _attach(
        valid_rows,
        transform_frozen(valid_rows, context_state),
        pitcher_k=pitcher_smoothing,
        batter_k=batter_smoothing,
    )
    preprocessing, fit_prepared = fit_preprocessor(
        fit_source, PreprocessingSpec("dl_standard", ("hand_matchup",))
    )
    valid_prepared = transform_preprocessor(valid_source, preprocessing)
    maps = _category_maps(fit_prepared, preprocessing.categorical_columns)
    state = HierarchicalFeatureState(
        context_state,
        MappingProxyType(_preprocessing_payload(preprocessing)),
        preprocessing.numeric_columns,
        preprocessing.categorical_columns,
        maps,
        pitcher_smoothing,
        batter_smoothing,
    )
    train_batch = _build_batch(
        fit_rows, fit_prepared,
        numeric_columns=state.numeric_columns,
        categorical_columns=state.categorical_columns,
        category_maps=state.category_maps,
    )
    valid_batch = _build_batch(
        valid_rows, valid_prepared,
        numeric_columns=state.numeric_columns,
        categorical_columns=state.categorical_columns,
        category_maps=state.category_maps,
    )
    metadata = ModelMetadata(
        n_num_features=train_batch.x_num.shape[1],
        categorical_cardinalities=tuple(len(maps[column]) + 1 for column in state.categorical_columns),
        train_x_num=train_batch.x_num,
    )
    return PreparedFold(train_batch, valid_batch, state, metadata)


def transform_with_state(frame: pd.DataFrame, state: HierarchicalFeatureState) -> FeatureBatch:
    preprocessing = _preprocessing_from_payload(state.preprocessing_payload)
    source = _attach(
        frame,
        transform_frozen(frame, state.context),
        pitcher_k=state.pitcher_k,
        batter_k=state.batter_k,
    )
    prepared = transform_preprocessor(source, preprocessing)
    return _build_batch(
        frame,
        prepared,
        numeric_columns=state.numeric_columns,
        categorical_columns=state.categorical_columns,
        category_maps=state.category_maps,
    )
