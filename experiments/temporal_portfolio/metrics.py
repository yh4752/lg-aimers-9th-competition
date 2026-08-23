"""Strict, deterministic metrics for temporal portfolio OOF evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Integral, Real
from typing import Sequence

import numpy as np
import pandas as pd

from .contracts import PortfolioContract


class PortfolioMetricError(ValueError):
    """Raised when metric inputs cannot support an exact paired comparison."""


_KEY_COLUMNS = ("row_id", "valid_year")
_BASE_COLUMNS = (*_KEY_COLUMNS, "target", "probability")
_RESERVED_OUTPUT_COLUMNS = set(_BASE_COLUMNS)
_MODES = ("probability", "logit")
_DECAYS = (Decimal("0.40"), Decimal("0.55"), Decimal("0.70"), Decimal("1.00"))
_RECENT_WEIGHTS = (
    Decimal("0.50"),
    Decimal("0.65"),
    Decimal("0.75"),
    Decimal("0.85"),
    Decimal("1.00"),
)
_ANCHOR_BETAS = (
    Decimal("0"),
    Decimal("0.025"),
    Decimal("0.05"),
    Decimal("0.10"),
)
_EPS = np.finfo("float64").eps


def _exact_decimal(value: object, label: str) -> Decimal:
    if type(value) is not Decimal:
        raise PortfolioMetricError(f"{label} must be an exact Decimal")
    if not value.is_finite() or value < Decimal("0") or value > Decimal("1"):
        raise PortfolioMetricError(f"{label} Decimal must be finite and in [0, 1]")
    return value


def _numeric_vector(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise PortfolioMetricError(f"{label} must be a non-empty one-dimensional array")
    if array.dtype.kind not in "iuf":
        raise PortfolioMetricError(f"{label} must be a numeric array")
    result = np.array(array, dtype="float64", copy=True)
    if not np.isfinite(result).all():
        raise PortfolioMetricError(f"{label} must be finite")
    return result


def _binary_vector(value: object, label: str) -> np.ndarray:
    result = _numeric_vector(value, label)
    if not np.isin(result, (0.0, 1.0)).all():
        raise PortfolioMetricError(f"{label} must contain only 0 and 1")
    return result


def _probability_vector(value: object, label: str) -> np.ndarray:
    result = _numeric_vector(value, label)
    if np.any((result < 0.0) | (result > 1.0)):
        raise PortfolioMetricError(f"{label} must be in [0, 1]")
    return result


def _paired_probabilities(
    left: object, right: object
) -> tuple[np.ndarray, np.ndarray]:
    first = _probability_vector(left, "left probability")
    second = _probability_vector(right, "right probability")
    if len(first) != len(second):
        raise PortfolioMetricError("probability array lengths differ")
    return first, second


def brier(target: object, probability: object) -> float:
    """Return mean binary squared error after strict vector validation."""

    y = _binary_vector(target, "target")
    p = _probability_vector(probability, "probability")
    if len(y) != len(p):
        raise PortfolioMetricError("target and probability array lengths differ")
    return float(np.mean(np.square(p - y), dtype=np.float64))


def blend_probability(left: object, right: object, left_weight: Decimal) -> np.ndarray:
    """Blend two probability vectors using an exact preregistered weight."""

    weight = _exact_decimal(left_weight, "left weight")
    first, second = _paired_probabilities(left, right)
    if weight == Decimal("1"):
        return first.copy()
    if weight == Decimal("0"):
        return second.copy()
    numeric_weight = float(weight)
    return numeric_weight * first + (1.0 - numeric_weight) * second


def _expit(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype="float64")
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def blend_logit(left: object, right: object, left_weight: Decimal) -> np.ndarray:
    """Blend logits without overflowing, preserving exact weight endpoints."""

    weight = _exact_decimal(left_weight, "left weight")
    first, second = _paired_probabilities(left, right)
    if weight == Decimal("1"):
        return first.copy()
    if weight == Decimal("0"):
        return second.copy()
    first_clipped = np.clip(first, _EPS, 1.0 - _EPS)
    second_clipped = np.clip(second, _EPS, 1.0 - _EPS)
    first_logit = np.log(first_clipped) - np.log1p(-first_clipped)
    second_logit = np.log(second_clipped) - np.log1p(-second_clipped)
    numeric_weight = float(weight)
    return _expit(
        numeric_weight * first_logit + (1.0 - numeric_weight) * second_logit
    )


def apply_anchor(
    probability: object, *, anchor_rate: float, beta: Decimal
) -> np.ndarray:
    """Shrink predictions toward a constant anchor in logit space."""

    anchor_beta = _exact_decimal(beta, "anchor beta")
    if isinstance(anchor_rate, bool) or not isinstance(anchor_rate, Real):
        raise PortfolioMetricError("anchor rate must be a finite numeric probability")
    numeric_anchor = float(anchor_rate)
    if not np.isfinite(numeric_anchor) or not 0.0 <= numeric_anchor <= 1.0:
        raise PortfolioMetricError("anchor rate must be in [0, 1]")
    values = _probability_vector(probability, "probability")
    anchor = np.full(values.shape, numeric_anchor, dtype="float64")
    return blend_logit(values, anchor, Decimal("1") - anchor_beta)


def score_gain_from_brier_gain(gain: Decimal, baseline: Decimal) -> Decimal:
    """Map Brier improvement to the competition's 100,000-point skill scale."""

    if type(gain) is not Decimal or not gain.is_finite():
        raise PortfolioMetricError("gain must be a finite exact Decimal")
    if type(baseline) is not Decimal or not baseline.is_finite() or baseline <= 0:
        raise PortfolioMetricError("baseline must be a positive finite exact Decimal")
    return gain / baseline * Decimal("100000")


def score_tier(gain: Decimal) -> str:
    """Classify an exact Brier gain at the sealed campaign boundaries."""

    if type(gain) is not Decimal or not gain.is_finite():
        raise PortfolioMetricError("gain must be a finite exact Decimal")
    if gain >= Decimal("0.00045"):
        return "breakthrough"
    if gain >= Decimal("0.00025"):
        return "competitive"
    if gain >= Decimal("0.00005"):
        return "incremental"
    return "below_incremental"


def _id_token(value: object, label: str) -> tuple[str, object]:
    if isinstance(value, bool):
        raise PortfolioMetricError(f"{label} row_id is invalid")
    if isinstance(value, Integral):
        return "int", int(value)
    if type(value) is str and value:
        return "str", value
    raise PortfolioMetricError(f"{label} row_id is invalid")


def _year(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PortfolioMetricError(f"{label} valid_year is invalid")
    year = int(value)
    if not 1000 <= year <= 9999:
        raise PortfolioMetricError(f"{label} valid_year is invalid")
    return year


def _segment_equal(left: object, right: object) -> bool:
    left_missing = bool(pd.isna(left))
    right_missing = bool(pd.isna(right))
    if left_missing or right_missing:
        return left_missing and right_missing
    return type(left) is type(right) and bool(left == right)


def _validate_segment_columns(segment_columns: object) -> tuple[str, ...]:
    if type(segment_columns) is not tuple or any(
        type(column) is not str or not column for column in segment_columns
    ):
        raise PortfolioMetricError("segment columns must be a tuple of canonical names")
    if len(set(segment_columns)) != len(segment_columns):
        raise PortfolioMetricError("segment columns must be unique")
    if set(segment_columns).intersection(_BASE_COLUMNS):
        raise PortfolioMetricError("segment columns overlap the base OOF schema")
    return segment_columns


def _validated_oof(
    frame: object, *, label: str, segment_columns: tuple[str, ...]
) -> tuple[pd.DataFrame, tuple[tuple[tuple[str, object], int], ...]]:
    if type(frame) is not pd.DataFrame or frame.empty:
        raise PortfolioMetricError(f"{label} must be a non-empty pandas DataFrame")
    if not frame.columns.is_unique:
        raise PortfolioMetricError(f"{label} columns must be unique")
    expected = {*_BASE_COLUMNS, *segment_columns}
    if set(frame.columns) != expected:
        raise PortfolioMetricError(f"{label} schema differs")
    output = frame.loc[:, (*_BASE_COLUMNS, *segment_columns)].copy(deep=True)
    id_tokens = tuple(_id_token(value, label) for value in output["row_id"])
    if len(set(id_tokens)) != len(id_tokens):
        raise PortfolioMetricError(f"{label} row_id must be unique")
    years = tuple(_year(value, label) for value in output["valid_year"])
    keys = tuple(zip(id_tokens, years, strict=True))
    if len(set(keys)) != len(keys):
        raise PortfolioMetricError(f"{label} OOF keys must be unique")
    target = _binary_vector(output["target"].to_numpy(), f"{label} target")
    probability = _probability_vector(
        output["probability"].to_numpy(), f"{label} probability"
    )
    output["valid_year"] = np.asarray(years, dtype="int64")
    output["target"] = target.astype("int8")
    output["probability"] = probability
    return output, keys


def align_oof(
    candidates: Sequence[tuple[str, pd.DataFrame]],
    *,
    segment_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Strictly align candidate-named OOF frames on canonical paired keys.

    Each frame must contain exactly ``row_id``, ``valid_year``, ``target``,
    ``probability``, and the explicitly registered segment columns. No rows are
    joined or discarded.
    """

    segments = _validate_segment_columns(segment_columns)
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise PortfolioMetricError("candidates must be a sequence of name/frame pairs")
    entries = tuple(candidates)
    if not entries:
        raise PortfolioMetricError("at least one candidate is required")
    names: list[str] = []
    for entry in entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise PortfolioMetricError("candidates must contain exact name/frame pairs")
        name = entry[0]
        if type(name) is not str or not name or name in _RESERVED_OUTPUT_COLUMNS or name in segments:
            raise PortfolioMetricError("candidate name is invalid or reserved")
        names.append(name)
    if len(set(names)) != len(names):
        raise PortfolioMetricError("candidate names must be unique")

    validated = [
        _validated_oof(frame, label=f"candidate {name}", segment_columns=segments)
        for name, frame in entries
    ]
    canonical_keys = set(validated[0][1])
    if any(set(keys) != canonical_keys for _, keys in validated[1:]):
        raise PortfolioMetricError("OOF key set differs between candidates")
    ordered_keys = tuple(sorted(canonical_keys, key=lambda key: (key[0], key[1])))

    ordered_frames: list[pd.DataFrame] = []
    for frame, keys in validated:
        positions = {key: position for position, key in enumerate(keys)}
        ordered_frames.append(
            frame.iloc[[positions[key] for key in ordered_keys]].reset_index(drop=True)
        )
    reference = ordered_frames[0]
    for candidate in ordered_frames[1:]:
        if not np.array_equal(reference["target"].to_numpy(), candidate["target"].to_numpy()):
            raise PortfolioMetricError("target differs on paired OOF rows")
        for column in segments:
            if not all(
                _segment_equal(left, right)
                for left, right in zip(reference[column], candidate[column], strict=True)
            ):
                raise PortfolioMetricError(f"segment differs on paired OOF rows: {column}")

    result = reference.loc[:, (*_KEY_COLUMNS, "target", *segments)].copy(deep=True)
    for name, frame in zip(names, ordered_frames, strict=True):
        result[name] = frame["probability"].to_numpy(dtype="float64", copy=True)
    return result


def _decimal_tuple_matches(
    actual: object, expected: tuple[Decimal, ...]
) -> bool:
    return (
        type(actual) is tuple
        and len(actual) == len(expected)
        and all(
            type(value) is Decimal and value.as_tuple() == sealed.as_tuple()
            for value, sealed in zip(actual, expected, strict=True)
        )
    )


def _decimal_id(value: Decimal) -> str:
    return str(value).replace(".", "p")


def _is_exact_grid_member(value: object, grid: tuple[Decimal, ...]) -> bool:
    return type(value) is Decimal and any(
        value.as_tuple() == sealed.as_tuple() for sealed in grid
    )


@dataclass(frozen=True, slots=True)
class T1Recipe:
    """One sealed, exactly reproducible T1 blend recipe."""

    decay: Decimal
    recent_weight: Decimal
    mode: str
    beta: Decimal
    recipe_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not _is_exact_grid_member(self.decay, _DECAYS):
            raise PortfolioMetricError("recipe decay is not sealed")
        if not _is_exact_grid_member(self.recent_weight, _RECENT_WEIGHTS):
            raise PortfolioMetricError("recipe recent weight is not sealed")
        if type(self.mode) is not str or self.mode not in _MODES:
            raise PortfolioMetricError("recipe blend mode is not sealed")
        if not _is_exact_grid_member(self.beta, _ANCHOR_BETAS):
            raise PortfolioMetricError("recipe anchor beta is not sealed")
        object.__setattr__(
            self,
            "recipe_id",
            "__".join(
                (
                    "t1",
                    f"d{_decimal_id(self.decay)}",
                    f"rw{_decimal_id(self.recent_weight)}",
                    self.mode,
                    f"b{_decimal_id(self.beta)}",
                )
            ),
        )


def build_t1_recipes(contract: PortfolioContract) -> tuple[T1Recipe, ...]:
    """Build the complete sealed Cartesian T1 recipe registration."""

    if type(contract) is not PortfolioContract:
        raise PortfolioMetricError("T1 recipe contract type differs")
    dimensions = (
        ("decays", contract.decays, _DECAYS),
        ("recent weights", contract.recent_weights, _RECENT_WEIGHTS),
        ("anchor betas", contract.anchor_betas, _ANCHOR_BETAS),
    )
    for label, actual, expected in dimensions:
        if not _decimal_tuple_matches(actual, expected):
            raise PortfolioMetricError(f"contract {label} differ")
    expected_count = (
        len(contract.decays)
        * len(contract.recent_weights)
        * len(_MODES)
        * len(contract.anchor_betas)
    )
    recipes = tuple(
        T1Recipe(decay, recent_weight, mode, beta)
        for decay in contract.decays
        for recent_weight in contract.recent_weights
        for mode in _MODES
        for beta in contract.anchor_betas
    )
    if len(recipes) != expected_count or len({item.recipe_id for item in recipes}) != expected_count:
        raise PortfolioMetricError("T1 recipe Cartesian registration differs")
    return recipes
