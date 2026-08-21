from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class CalibrationError(ValueError):
    """Raised when calibration input or persisted state is unsafe."""


CALIBRATION_EFFECTS = (
    "game_type",
    "count_state",
    "hand_matchup",
    "base_out_state",
)
MISSING_CATEGORY = "__MISSING__"


@dataclass(frozen=True)
class CalibrationState:
    schema_version: int
    kind: str
    regularization: float
    clip: float
    bias: float
    slope: float
    effects: Mapping[str, Mapping[str, float]]
    fit_row_ids_sha256: str


@dataclass(frozen=True)
class CalibrationSelection:
    kind: str
    selected_regularization: float
    validation_brier: Mapping[float, float]
    state: CalibrationState
    selection_row_ids: tuple[str, ...]


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise CalibrationError(f"{label} must be finite and positive")
    return result


def _clip(value: object) -> float:
    result = _positive(value, "clip")
    if result >= 0.5:
        raise CalibrationError("clip must be below 0.5")
    return result


def _arrays(
    probability: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probability, dtype="float64")
    y = np.asarray(target, dtype="float64")
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y) or not len(p):
        raise CalibrationError("probability and target length must match and be non-empty")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise CalibrationError("probability must be finite and in [0, 1]")
    if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise CalibrationError("target must contain only 0 and 1")
    return p, y


def _row_ids(segments: pd.DataFrame | None, length: int) -> tuple[str, ...]:
    if segments is None:
        return tuple(str(position) for position in range(length))
    if len(segments) != length:
        raise CalibrationError("segment length differs from probability")
    if "row_id" not in segments or segments["row_id"].isna().any():
        raise CalibrationError("row_id must be present and non-null")
    values = tuple(segments["row_id"].astype(str))
    if len(set(values)) != len(values):
        raise CalibrationError("row_id must be unique")
    return values


def _row_id_sha256(row_ids: tuple[str, ...]) -> str:
    return sha256("\n".join(row_ids).encode("utf-8")).hexdigest()


def _category(values: pd.Series) -> np.ndarray:
    return values.astype("string").fillna(MISSING_CATEGORY).astype(str).to_numpy()


def _effect_layout(
    segments: pd.DataFrame, length: int
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], tuple[tuple[str, str], ...]]:
    _row_ids(segments, length)
    missing = [column for column in CALIBRATION_EFFECTS if column not in segments]
    if missing:
        raise CalibrationError(f"missing calibration effect: {', '.join(missing)}")
    encoded: dict[str, np.ndarray] = {}
    levels: dict[str, tuple[str, ...]] = {}
    flattened: list[tuple[str, str]] = []
    for column in CALIBRATION_EFFECTS:
        values = _category(segments[column])
        column_levels = tuple(sorted(set(values.tolist())))
        lookup = {level: index for index, level in enumerate(column_levels)}
        encoded[column] = np.asarray([lookup[value] for value in values], dtype="int64")
        levels[column] = column_levels
        flattened.extend((column, level) for level in column_levels)
    return encoded, levels, tuple(flattened)


def _logit(probability: np.ndarray, clip: float) -> np.ndarray:
    p = np.clip(probability, clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    output = np.empty_like(logits, dtype="float64")
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def _run_optimizer(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    optimizer: Callable,
) -> np.ndarray:
    try:
        result = optimizer(
            objective,
            initial,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-10},
        )
    except Exception as error:
        raise CalibrationError(f"optimizer failed: {error}") from error
    values = np.asarray(getattr(result, "x", None), dtype="float64")
    if values.shape != initial.shape or not np.isfinite(values).all():
        raise CalibrationError("optimizer returned invalid parameters")
    objective_value = float(objective(values))
    if not math.isfinite(objective_value):
        raise CalibrationError("optimizer returned a non-finite objective")
    return values


def _fit(
    probability: np.ndarray,
    target: np.ndarray,
    segments: pd.DataFrame | None,
    *,
    kind: str,
    regularization: float,
    clip: float,
    optimizer: Callable,
) -> CalibrationState:
    p, y = _arrays(probability, target)
    penalty = _positive(regularization, "regularization")
    clipping = _clip(clip)
    row_ids = _row_ids(segments, len(p))
    logits = _logit(p, clipping)
    encoded: dict[str, np.ndarray] = {}
    levels: dict[str, tuple[str, ...]] = {}
    flat_levels: tuple[tuple[str, str], ...] = ()
    if kind == "H3":
        if segments is None:
            raise CalibrationError("H3 requires segment rows")
        encoded, levels, flat_levels = _effect_layout(segments, len(p))
    elif kind != "H2":
        raise CalibrationError("kind must be H2 or H3")
    offsets_by_column: dict[str, tuple[int, int]] = {}
    position = 2
    for column in CALIBRATION_EFFECTS if kind == "H3" else ():
        end = position + len(levels[column])
        offsets_by_column[column] = (position, end)
        position = end
    initial = np.zeros(position, dtype="float64")
    initial[1] = 1.0

    def objective(parameters: np.ndarray) -> float:
        score = parameters[0] + parameters[1] * logits
        for column, (start, end) in offsets_by_column.items():
            score = score + parameters[start:end][encoded[column]]
        calibrated = _sigmoid(score)
        base = float(np.mean(np.square(y - calibrated), dtype=np.float64))
        shrinkage = parameters[0] ** 2 + (parameters[1] - 1.0) ** 2
        if len(parameters) > 2:
            shrinkage += float(np.mean(np.square(parameters[2:]), dtype=np.float64))
        return base + penalty * shrinkage

    fitted = _run_optimizer(objective, initial, optimizer)
    effects: dict[str, Mapping[str, float]] = {}
    for column in CALIBRATION_EFFECTS if kind == "H3" else ():
        start, end = offsets_by_column[column]
        effects[column] = MappingProxyType(
            {
                level: float(value)
                for level, value in zip(levels[column], fitted[start:end], strict=True)
            }
        )
    return CalibrationState(
        1,
        kind,
        penalty,
        clipping,
        float(fitted[0]),
        float(fitted[1]),
        MappingProxyType(effects),
        _row_id_sha256(row_ids),
    )


def fit_h2(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    regularization: float,
    clip: float,
) -> CalibrationState:
    return _fit(
        probability, target, None, kind="H2", regularization=regularization,
        clip=clip, optimizer=minimize,
    )


def fit_h3(
    probability: np.ndarray,
    target: np.ndarray,
    segments: pd.DataFrame,
    *,
    regularization: float,
    clip: float,
) -> CalibrationState:
    return _fit(
        probability, target, segments, kind="H3", regularization=regularization,
        clip=clip, optimizer=minimize,
    )


def apply_calibration(
    probability: np.ndarray,
    segments: pd.DataFrame,
    state: CalibrationState,
) -> np.ndarray:
    p = np.asarray(probability, dtype="float64")
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise CalibrationError("probability must be finite and in [0, 1]")
    if len(segments) != len(p):
        raise CalibrationError("segment length differs from probability")
    score = state.bias + state.slope * _logit(p, state.clip)
    if state.kind == "H3":
        missing = [column for column in CALIBRATION_EFFECTS if column not in segments]
        if missing:
            raise CalibrationError(f"missing calibration effect: {', '.join(missing)}")
        for column in CALIBRATION_EFFECTS:
            mapping = state.effects[column]
            offsets = _category(segments[column])
            score = score + np.asarray([mapping.get(value, 0.0) for value in offsets])
    elif state.kind != "H2":
        raise CalibrationError("calibration kind is invalid")
    return np.clip(_sigmoid(score), state.clip, 1.0 - state.clip)


def _selection_grid(grid: object) -> tuple[float, ...]:
    if not isinstance(grid, tuple) or not grid:
        raise CalibrationError("regularization grid must be a non-empty tuple")
    parsed = tuple(_positive(value, "regularization") for value in grid)
    if parsed != tuple(sorted(set(parsed))):
        raise CalibrationError("regularization grid must be unique and increasing")
    return parsed


def select_calibration(
    oof_2023: pd.DataFrame,
    *,
    kind: str,
    grid: tuple[float, ...],
    fit_month_max: int,
    validation_month_min: int,
    optimizer: Callable = minimize,
) -> CalibrationSelection:
    regularizations = _selection_grid(grid)
    required = {"row_id", "probability", "control_success", "game_month"}
    if kind == "H3":
        required.update(CALIBRATION_EFFECTS)
    missing = sorted(required.difference(oof_2023.columns))
    if missing:
        raise CalibrationError(f"missing OOF columns: {', '.join(missing)}")
    row_ids = _row_ids(oof_2023, len(oof_2023))
    probability, target = _arrays(
        oof_2023["probability"].to_numpy(),
        oof_2023["control_success"].to_numpy(),
    )
    months = pd.to_numeric(oof_2023["game_month"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(months).all() or not np.equal(months, np.floor(months)).all() or ((months < 1) | (months > 12)).any():
        raise CalibrationError("game_month must contain integers in [1, 12]")
    if "season" in oof_2023 and not oof_2023["season"].eq(2023).all():
        raise CalibrationError("calibration selection requires 2023 OOF rows")
    if not isinstance(fit_month_max, int) or not isinstance(validation_month_min, int) or not 1 <= fit_month_max < validation_month_min <= 12:
        raise CalibrationError("calibration month boundary is invalid")
    fit_mask = months <= fit_month_max
    valid_mask = months >= validation_month_min
    if not fit_mask.any() or not valid_mask.any():
        raise CalibrationError("calibration split must contain fit and validation rows")

    def fit_one(mask: np.ndarray, regularization: float) -> CalibrationState:
        segment_rows = oof_2023.loc[mask].reset_index(drop=True)
        return _fit(
            probability[mask], target[mask], segment_rows if kind == "H3" else None,
            kind=kind, regularization=regularization, clip=1e-6, optimizer=optimizer,
        )

    validation_scores: dict[float, float] = {}
    valid_segments = oof_2023.loc[valid_mask].reset_index(drop=True)
    for regularization in regularizations:
        state = fit_one(fit_mask, regularization)
        calibrated = apply_calibration(probability[valid_mask], valid_segments, state)
        validation_scores[regularization] = float(
            np.mean(np.square(target[valid_mask] - calibrated), dtype=np.float64)
        )
    best = min(validation_scores.values())
    selected = max(
        regularization
        for regularization, score in validation_scores.items()
        if score <= best + 1e-12
    )
    all_segments = oof_2023.reset_index(drop=True)
    final = _fit(
        probability, target, all_segments if kind == "H3" else None,
        kind=kind, regularization=selected, clip=1e-6, optimizer=optimizer,
    )
    if kind == "H2":
        final = CalibrationState(
            final.schema_version, final.kind, final.regularization, final.clip,
            final.bias, final.slope, final.effects, _row_id_sha256(row_ids),
        )
    return CalibrationSelection(
        kind,
        selected,
        MappingProxyType(validation_scores),
        final,
        tuple(oof_2023.loc[valid_mask, "row_id"].astype(str)),
    )


def _state_body(state: CalibrationState) -> dict[str, object]:
    return {
        "schema_version": state.schema_version,
        "kind": state.kind,
        "regularization": state.regularization,
        "clip": state.clip,
        "bias": state.bias,
        "slope": state.slope,
        "effects": {
            column: dict(state.effects[column]) for column in state.effects
        },
        "fit_row_ids_sha256": state.fit_row_ids_sha256,
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def calibration_state_payload(state: CalibrationState) -> dict[str, object]:
    body = _state_body(state)
    return {**body, "state_sha256": sha256(_canonical_json(body)).hexdigest()}


def calibration_state_sha256(state: CalibrationState) -> str:
    return str(calibration_state_payload(state)["state_sha256"])


def canonical_state_json(state: CalibrationState) -> bytes:
    return _canonical_json(calibration_state_payload(state))


def calibration_state_from_payload(payload: Mapping[str, object]) -> CalibrationState:
    if not isinstance(payload, Mapping):
        raise CalibrationError("calibration state is invalid")
    expected = {
        "schema_version", "kind", "regularization", "clip", "bias", "slope",
        "effects", "fit_row_ids_sha256", "state_sha256",
    }
    if set(payload) != expected:
        raise CalibrationError("calibration state keys differ")
    body = {key: payload[key] for key in payload if key != "state_sha256"}
    if not isinstance(payload["state_sha256"], str) or payload["state_sha256"] != sha256(_canonical_json(body)).hexdigest():
        raise CalibrationError("calibration state digest differs")
    if payload["schema_version"] != 1 or payload["kind"] not in {"H2", "H3"}:
        raise CalibrationError("calibration state identity differs")
    regularization = _positive(payload["regularization"], "regularization")
    clipping = _clip(payload["clip"])
    numeric: list[float] = []
    for key in ("bias", "slope"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise CalibrationError(f"{key} is invalid")
        numeric.append(float(value))
    digest = payload["fit_row_ids_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise CalibrationError("fit row digest is invalid")
    effects_payload = payload["effects"]
    expected_effects = CALIBRATION_EFFECTS if payload["kind"] == "H3" else ()
    if (
        not isinstance(effects_payload, Mapping)
        or set(effects_payload) != set(expected_effects)
    ):
        raise CalibrationError("calibration effects differ")
    effects: dict[str, Mapping[str, float]] = {}
    for column in expected_effects:
        mapping = effects_payload[column]
        if not isinstance(mapping, Mapping) or list(mapping) != sorted(mapping):
            raise CalibrationError("calibration effect levels differ")
        parsed: dict[str, float] = {}
        for level, value in mapping.items():
            if not isinstance(level, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise CalibrationError("calibration effect value is invalid")
            parsed[level] = float(value)
        effects[column] = MappingProxyType(parsed)
    state = CalibrationState(
        1, str(payload["kind"]), regularization, clipping, numeric[0], numeric[1],
        MappingProxyType(effects), digest,
    )
    if calibration_state_payload(state) != dict(payload):
        raise CalibrationError("calibration state is not canonical")
    return state
