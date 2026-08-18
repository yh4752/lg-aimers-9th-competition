from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


class ContextFeatureError(ValueError):
    """Raised when a hierarchical context state would be unsafe or ambiguous."""


HIERARCHY_LEVELS = (
    ("balls_before", "strikes_before"),
    ("balls_before", "strikes_before", "pitcher_hand", "batter_hand"),
    (
        "balls_before",
        "strikes_before",
        "pitcher_hand",
        "batter_hand",
        "base_state",
        "outs_before",
    ),
    (
        "balls_before",
        "strikes_before",
        "pitcher_hand",
        "batter_hand",
        "base_state",
        "outs_before",
        "game_type",
    ),
)
MISSING_CATEGORY = "__MISSING__"
TARGET = "control_success"
_COUNT_LIMITS = {
    "balls_before": (0, 3),
    "strikes_before": (0, 2),
    "outs_before": (0, 2),
}
_CATEGORY_COLUMNS = ("pitcher_hand", "batter_hand", "base_state", "game_type")
_BASE_STATE = re.compile(r"[01]{3}")


@dataclass(frozen=True)
class LevelState:
    columns: tuple[str, ...]
    counts: Mapping[tuple[str, ...], int]
    successes: Mapping[tuple[str, ...], float]
    rates: Mapping[tuple[str, ...], float]


@dataclass(frozen=True)
class ContextState:
    schema_version: int
    smoothing_k: float
    row_count: int
    target_sum: float
    global_rate: float
    levels: tuple[LevelState, ...]


def _required(frame: pd.DataFrame, *, target: bool) -> None:
    columns = {"row_id", *HIERARCHY_LEVELS[-1]}
    if target:
        columns.add(TARGET)
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ContextFeatureError(f"missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ContextFeatureError("context frame is empty")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise ContextFeatureError("row_id must be non-null and unique")


def _integer_column(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise ContextFeatureError(f"{column} must contain finite integers")
    values = numeric.to_numpy(dtype="float64")
    if not np.equal(values, np.floor(values)).all():
        raise ContextFeatureError(f"{column} must contain integers")
    lower, upper = _COUNT_LIMITS[column]
    if ((values < lower) | (values > upper)).any():
        raise ContextFeatureError(f"{column} is outside [{lower}, {upper}]")
    return pd.Series(values.astype("int64").astype(str), index=frame.index)


def _category_column(frame: pd.DataFrame, column: str) -> pd.Series:
    result = frame[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
    if column == "base_state":
        invalid = result.ne(MISSING_CATEGORY) & ~result.str.fullmatch(_BASE_STATE)
        if invalid.any():
            raise ContextFeatureError("base_state must be a three-bit string")
    return result


def _target(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise ContextFeatureError("control_success must contain only 0 and 1")
    return values


def _normalized(frame: pd.DataFrame, *, require_target: bool) -> tuple[pd.DataFrame, np.ndarray | None]:
    _required(frame, target=require_target)
    output = pd.DataFrame(index=frame.index)
    output["row_id"] = frame["row_id"].astype(str).to_numpy()
    for column in _COUNT_LIMITS:
        output[column] = _integer_column(frame, column)
    for column in _CATEGORY_COLUMNS:
        output[column] = _category_column(frame, column)
    target = _target(frame) if require_target else None
    if target is not None:
        output[TARGET] = target
    return output, target


def _key(row: pd.Series, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


def _freeze_level(
    columns: tuple[str, ...],
    counts: Mapping[tuple[str, ...], int],
    successes: Mapping[tuple[str, ...], float],
    rates: Mapping[tuple[str, ...], float],
) -> LevelState:
    return LevelState(
        columns,
        MappingProxyType(dict(counts)),
        MappingProxyType(dict(successes)),
        MappingProxyType(dict(rates)),
    )


def fit_context_state(frame: pd.DataFrame, *, smoothing_k: float) -> ContextState:
    if isinstance(smoothing_k, bool) or not isinstance(smoothing_k, (int, float)):
        raise ContextFeatureError("smoothing_k must be finite and positive")
    smoothing = float(smoothing_k)
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ContextFeatureError("smoothing_k must be finite and positive")
    normalized, target = _normalized(frame, require_target=True)
    assert target is not None
    row_count = len(normalized)
    target_sum = float(target.sum())
    global_rate = target_sum / row_count
    levels: list[LevelState] = []
    parent_rates: Mapping[tuple[str, ...], float] | None = None
    parent_width = 0
    for columns in HIERARCHY_LEVELS:
        grouped = (
            normalized.groupby(list(columns), sort=True, observed=True, dropna=False)[TARGET]
            .agg(["size", "sum"])
            .reset_index()
        )
        counts: dict[tuple[str, ...], int] = {}
        successes: dict[tuple[str, ...], float] = {}
        rates: dict[tuple[str, ...], float] = {}
        for _, row in grouped.iterrows():
            key = _key(row, columns)
            count = int(row["size"])
            success = float(row["sum"])
            parent = global_rate if parent_rates is None else parent_rates[key[:parent_width]]
            counts[key] = count
            successes[key] = success
            rates[key] = (success + smoothing * parent) / (count + smoothing)
        levels.append(_freeze_level(columns, counts, successes, rates))
        parent_rates = rates
        parent_width = len(columns)
    return ContextState(1, smoothing, row_count, target_sum, global_rate, tuple(levels))


def transform_training_loo(frame: pd.DataFrame, state: ContextState) -> pd.DataFrame:
    normalized, target = _normalized(frame, require_target=True)
    assert target is not None
    fitted = fit_context_state(frame, smoothing_k=state.smoothing_k)
    if context_state_payload(fitted) != context_state_payload(state):
        raise ContextFeatureError("training frame differs from fitted context state")
    output = np.empty(len(normalized), dtype="float64")
    for position, (_, row) in enumerate(normalized.iterrows()):
        y = float(target[position])
        parent = (
            (state.target_sum - y) / (state.row_count - 1)
            if state.row_count > 1
            else 0.5
        )
        for level in state.levels:
            key = _key(row, level.columns)
            count = level.counts[key] - 1
            success = level.successes[key] - y
            parent = (success + state.smoothing_k * parent) / (count + state.smoothing_k)
        output[position] = parent
    if not np.isfinite(output).all() or ((output < 0.0) | (output > 1.0)).any():
        raise ContextFeatureError("leave-one-out context rate is invalid")
    return pd.DataFrame({"hier_context_rate": output}, index=frame.index)


def transform_frozen(frame: pd.DataFrame, state: ContextState) -> pd.DataFrame:
    normalized, _ = _normalized(frame, require_target=False)
    output = np.empty(len(normalized), dtype="float64")
    for position, (_, row) in enumerate(normalized.iterrows()):
        rate = state.global_rate
        for level in state.levels:
            key = _key(row, level.columns)
            observed = level.rates.get(key)
            if observed is None:
                break
            rate = observed
        output[position] = rate
    if not np.isfinite(output).all() or ((output < 0.0) | (output > 1.0)).any():
        raise ContextFeatureError("frozen context rate is invalid")
    return pd.DataFrame({"hier_context_rate": output}, index=frame.index)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def context_state_payload(state: ContextState) -> dict[str, object]:
    levels: list[dict[str, object]] = []
    for level in state.levels:
        entries = [
            {
                "key": list(key),
                "count": level.counts[key],
                "success": level.successes[key],
                "rate": level.rates[key],
            }
            for key in sorted(level.counts)
        ]
        levels.append({"columns": list(level.columns), "entries": entries})
    return {
        "schema_version": state.schema_version,
        "smoothing_k": state.smoothing_k,
        "row_count": state.row_count,
        "target_sum": state.target_sum,
        "global_rate": state.global_rate,
        "levels": levels,
    }


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ContextFeatureError(f"{label} keys differ")


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ContextFeatureError(f"{label} must be a finite float")
    return value


def context_state_from_payload(payload: Mapping[str, object]) -> ContextState:
    if type(payload) is not dict:
        raise ContextFeatureError("context state must be an object")
    _exact_keys(
        payload,
        {"schema_version", "smoothing_k", "row_count", "target_sum", "global_rate", "levels"},
        "context state",
    )
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise ContextFeatureError("context state schema differs")
    smoothing = _finite_float(payload["smoothing_k"], "smoothing_k")
    row_count = payload["row_count"]
    if type(row_count) is not int or row_count <= 0:
        raise ContextFeatureError("row_count must be positive")
    target_sum = _finite_float(payload["target_sum"], "target_sum")
    global_rate = _finite_float(payload["global_rate"], "global_rate")
    if not 0 <= target_sum <= row_count or not math.isclose(
        global_rate, target_sum / row_count, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ContextFeatureError("global context evidence differs")
    raw_levels = payload["levels"]
    if type(raw_levels) is not list or len(raw_levels) != len(HIERARCHY_LEVELS):
        raise ContextFeatureError("context hierarchy levels differ")
    levels: list[LevelState] = []
    parent_rates: Mapping[tuple[str, ...], float] | None = None
    parent_width = 0
    for index, (raw_level, columns) in enumerate(zip(raw_levels, HIERARCHY_LEVELS, strict=True)):
        if type(raw_level) is not dict:
            raise ContextFeatureError("context level must be an object")
        _exact_keys(raw_level, {"columns", "entries"}, f"level {index}")
        if raw_level["columns"] != list(columns):
            raise ContextFeatureError("context hierarchy column order differs")
        entries = raw_level["entries"]
        if type(entries) is not list or not entries:
            raise ContextFeatureError("context level entries are invalid")
        counts: dict[tuple[str, ...], int] = {}
        successes: dict[tuple[str, ...], float] = {}
        rates: dict[tuple[str, ...], float] = {}
        observed_order: list[tuple[str, ...]] = []
        for entry in entries:
            if type(entry) is not dict:
                raise ContextFeatureError("context entry must be an object")
            _exact_keys(entry, {"key", "count", "success", "rate"}, "context entry")
            raw_key = entry["key"]
            if (
                type(raw_key) is not list
                or len(raw_key) != len(columns)
                or any(type(item) is not str for item in raw_key)
            ):
                raise ContextFeatureError("context key is invalid")
            key = tuple(raw_key)
            if key in counts:
                raise ContextFeatureError("duplicate context key")
            count = entry["count"]
            if type(count) is not int or count <= 0:
                raise ContextFeatureError("context count is invalid")
            success = _finite_float(entry["success"], "context success")
            rate = _finite_float(entry["rate"], "context rate")
            if not 0 <= success <= count or not 0 <= rate <= 1:
                raise ContextFeatureError("context entry range is invalid")
            parent = global_rate if parent_rates is None else parent_rates.get(key[:parent_width])
            if parent is None:
                raise ContextFeatureError("context parent is missing")
            expected_rate = (success + smoothing * parent) / (count + smoothing)
            if not math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-15):
                raise ContextFeatureError("context rate differs from evidence")
            counts[key] = count
            successes[key] = success
            rates[key] = rate
            observed_order.append(key)
        if observed_order != sorted(observed_order):
            raise ContextFeatureError("context entries are not canonical")
        if sum(counts.values()) != row_count or not math.isclose(
            sum(successes.values()), target_sum, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ContextFeatureError("context level totals differ")
        levels.append(_freeze_level(columns, counts, successes, rates))
        parent_rates = rates
        parent_width = len(columns)
    return ContextState(1, smoothing, row_count, target_sum, global_rate, tuple(levels))


def context_state_sha256(state: ContextState) -> str:
    return sha256(_canonical_json(context_state_payload(state))).hexdigest()

