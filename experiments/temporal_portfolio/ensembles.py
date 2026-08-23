"""Bounded ensemble recipes and forward-only probability calibration."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .metrics import blend_logit, blend_probability, brier


class EnsembleError(ValueError):
    """Raised when an ensemble or calibration contract is violated."""


PAIR_WEIGHTS = (Decimal("0.90"), Decimal("0.80"), Decimal("0.70"))
CALIBRATION_METHODS = ("intercept", "platt")
_EPS = np.finfo("float64").eps


@dataclass(frozen=True, init=False)
class OOFStream:
    row_id: tuple[str, ...]
    valid_year: tuple[int, ...]
    target: tuple[int, ...]
    _probability_bytes: bytes

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("OOFStream must be created with from_frame()")

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "OOFStream":
        work = _oof_frame(frame)
        instance = object.__new__(cls)
        object.__setattr__(instance, "row_id", tuple(work["row_id"]))
        object.__setattr__(instance, "valid_year", tuple(int(value) for value in work["valid_year"]))
        object.__setattr__(instance, "target", tuple(int(value) for value in work["target"]))
        object.__setattr__(instance, "_probability_bytes", work["probability"].to_numpy(dtype="float64").tobytes())
        _validate_stream(instance)
        return instance

    @property
    def probability(self) -> np.ndarray:
        return np.frombuffer(self._probability_bytes, dtype="float64")


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    components: tuple[str, ...]
    weights: tuple[Decimal, ...]
    mode: str
    anchor: Decimal | None = None
    calibration: str | None = None

    def __post_init__(self) -> None:
        if type(self.recipe_id) is not str or not self.recipe_id:
            raise EnsembleError("recipe_id must be a nonempty string")
        if not self.components or len(self.components) != len(self.weights) or len(set(self.components)) != len(self.components):
            raise EnsembleError("recipe components and weights differ")
        if any(type(name) is not str or not name for name in self.components):
            raise EnsembleError("recipe components must be canonical strings")
        if any(type(weight) is not Decimal or not weight.is_finite() or weight < 0 for weight in self.weights) or sum(self.weights) != Decimal("1"):
            raise EnsembleError("recipe weights must be exact nonnegative Decimals summing to one")
        if self.mode not in ("probability", "logit"):
            raise EnsembleError("recipe mode is invalid")
        if self.anchor is not None and (type(self.anchor) is not Decimal or not self.anchor.is_finite() or not Decimal("0") <= self.anchor <= Decimal("1")):
            raise EnsembleError("recipe anchor is invalid")
        if self.calibration is not None and self.calibration not in CALIBRATION_METHODS:
            raise EnsembleError("recipe calibration is invalid")
        if self.anchor is not None and self.calibration is not None:
            raise EnsembleError("anchor and calibration cannot be stacked")


@dataclass(frozen=True)
class AnchorSpec:
    beta: Decimal

    def __post_init__(self) -> None:
        if type(self.beta) is not Decimal or not self.beta.is_finite() or not Decimal("0") <= self.beta <= Decimal("1"):
            raise EnsembleError("anchor beta must be a bounded Decimal")


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    fit_years_by_score_year: Mapping[int, tuple[int, ...]]
    calibrated_probability_by_year: Mapping[int, np.ndarray]
    coefficient_by_score_year: Mapping[int, tuple[float, float]]
    gain_by_score_year: Mapping[int, float]
    weighted_gain: float
    approved: bool
    final_coefficient: tuple[float, float] | None


def build_raw_recipes(
    streams: Mapping[str, OOFStream], *, primary: str
) -> tuple[Recipe, ...]:
    snapshot = _streams(streams)
    if type(primary) is not str or primary not in snapshot:
        raise EnsembleError("primary stream is missing")
    _aligned_streams(snapshot)
    secondaries = tuple(sorted(set(snapshot).difference((primary,))))
    output = [_single(primary)]
    pair_scores: list[tuple[float, str]] = []
    primary_stream = snapshot[primary]
    primary_brier = brier(primary_stream.target, primary_stream.probability)
    for secondary in secondaries:
        best_gain = float("-inf")
        for weight in PAIR_WEIGHTS:
            for mode in ("probability", "logit"):
                recipe = _pair(primary, secondary, weight, mode)
                output.append(recipe)
                probability = _apply_recipe(recipe, snapshot)
                best_gain = max(
                    best_gain,
                    primary_brier - brier(primary_stream.target, probability),
                )
        pair_scores.append((best_gain, secondary))
    survivors = [name for gain, name in sorted(pair_scores, key=lambda item: (-item[0], item[1])) if gain > 0]
    if len(survivors) >= 2:
        for mode in ("probability", "logit"):
            output.append(_triple(primary, survivors[0], survivors[1], mode))
    if len(output) > 24:
        raise EnsembleError("raw recipe grid exceeded its contract")
    return tuple(output)


def build_correction_recipes(
    *, top_raw: tuple[Recipe, ...] | list[Recipe], approved_anchor: AnchorSpec
) -> tuple[Recipe, ...]:
    if not isinstance(top_raw, (tuple, list)) or not 1 <= len(top_raw) <= 5:
        raise EnsembleError("top_raw must contain between one and five recipes")
    if type(approved_anchor) is not AnchorSpec:
        raise EnsembleError("approved_anchor must be an AnchorSpec")
    output: list[Recipe] = []
    for raw in top_raw:
        if type(raw) is not Recipe or raw.anchor is not None or raw.calibration is not None:
            raise EnsembleError("top_raw contains a corrected recipe")
        output.append(raw)
        output.append(
            Recipe(
                f"{raw.recipe_id}__anchor_{approved_anchor.beta}",
                raw.components,
                raw.weights,
                raw.mode,
                anchor=approved_anchor.beta,
            )
        )
        for method in CALIBRATION_METHODS:
            output.append(
                Recipe(
                    f"{raw.recipe_id}__cal_{method}",
                    raw.components,
                    raw.weights,
                    raw.mode,
                    calibration=method,
                )
            )
    if len(output) > 20 or any(item.anchor is not None and item.calibration is not None for item in output):
        raise EnsembleError("correction recipe grid exceeded its contract")
    if len({item.recipe_id for item in output}) != len(output):
        raise EnsembleError("correction recipe IDs are not unique")
    return tuple(output)


def evaluate_forward_calibration(
    frame: pd.DataFrame, *, method: str
) -> CalibrationResult:
    if method not in CALIBRATION_METHODS:
        raise EnsembleError("calibration method is not preregistered")
    work = _oof_frame(frame)
    years = tuple(sorted(set(work["valid_year"])))
    if len(years) < 3:
        raise EnsembleError("forward calibration requires at least three OOF years")
    fit_years: dict[int, tuple[int, ...]] = {}
    probabilities: dict[int, np.ndarray] = {}
    coefficients: dict[int, tuple[float, float]] = {}
    gains: dict[int, float] = {}
    for score_year in years[1:]:
        prior_years = tuple(year for year in years if year < score_year)
        fit = work.loc[work["valid_year"].isin(prior_years)]
        scored = work.loc[work["valid_year"].eq(score_year)]
        slope, intercept = _fit_calibrator(fit, method)
        calibrated = _calibrate(scored["probability"].to_numpy(), slope, intercept)
        gain = brier(scored["target"], scored["probability"]) - brier(scored["target"], calibrated)
        fit_years[score_year] = prior_years
        probabilities[score_year] = np.array(calibrated, copy=True)
        probabilities[score_year].setflags(write=False)
        coefficients[score_year] = (slope, intercept)
        gains[score_year] = gain
    scored_rows = work.loc[work["valid_year"].isin(years[1:])]
    weighted_gain = float(
        sum(gains[year] * int((work["valid_year"] == year).sum()) for year in years[1:])
        / len(scored_rows)
    )
    approved = (
        all(0.8 <= slope <= 1.2 and -0.15 <= intercept <= 0.15 for slope, intercept in coefficients.values())
        and all(gain >= 0.0 for gain in gains.values())
        and weighted_gain >= 0.00003
    )
    final = _fit_calibrator(work, method) if approved else None
    return CalibrationResult(
        method,
        MappingProxyType(fit_years),
        MappingProxyType(probabilities),
        MappingProxyType(coefficients),
        MappingProxyType(gains),
        weighted_gain,
        approved,
        final,
    )


def _fit_calibrator(frame: pd.DataFrame, method: str) -> tuple[float, float]:
    target = frame["target"].to_numpy(dtype="int8")
    if len(np.unique(target)) != 2:
        raise EnsembleError("calibration fit rows must contain both target classes")
    logits = _logit(frame["probability"].to_numpy()).reshape(-1, 1)
    if method == "intercept":
        model = LogisticRegression(C=1e12, solver="lbfgs", fit_intercept=True)
        model.fit(np.zeros_like(logits), target)
        return 1.0, float(model.intercept_[0])
    model = LogisticRegression(C=1.0, solver="lbfgs", fit_intercept=True)
    model.fit(logits, target)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _calibrate(probability: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    value = slope * _logit(probability) + intercept
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype="float64"), _EPS, 1.0 - _EPS)
    return np.log(clipped) - np.log1p(-clipped)


def _streams(streams: Mapping[str, OOFStream]) -> dict[str, OOFStream]:
    if not isinstance(streams, Mapping):
        raise EnsembleError("streams must be a mapping")
    snapshot = dict(streams)
    if not snapshot or any(type(name) is not str or not name or type(stream) is not OOFStream for name, stream in snapshot.items()):
        raise EnsembleError("streams contain invalid names or values")
    for stream in snapshot.values():
        _validate_stream(stream)
    return snapshot


def _validate_stream(stream: object) -> None:
    if type(stream) is not OOFStream:
        raise EnsembleError("OOF stream type is invalid")
    size = len(stream.row_id)
    probability = stream.probability
    if (
        size == 0
        or len(stream.valid_year) != size
        or len(stream.target) != size
        or probability.shape != (size,)
        or len(set(stream.row_id)) != size
        or any(type(value) is not str or not value for value in stream.row_id)
        or any(type(value) is not int or not 1000 <= value <= 9999 for value in stream.valid_year)
        or any(type(value) is not int or value not in (0, 1) for value in stream.target)
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
    ):
        raise EnsembleError("OOF stream state is invalid")


def _aligned_streams(streams: Mapping[str, OOFStream]) -> None:
    reference = next(iter(streams.values()))
    for stream in streams.values():
        if (stream.row_id, stream.valid_year, stream.target) != (reference.row_id, reference.valid_year, reference.target):
            raise EnsembleError("OOF streams are not exactly aligned")


def _single(primary: str) -> Recipe:
    return Recipe(f"single__{primary}", (primary,), (Decimal("1"),), "probability")


def _pair(primary: str, secondary: str, weight: Decimal, mode: str) -> Recipe:
    return Recipe(f"pair__{primary}__{secondary}__{weight}__{mode}", (primary, secondary), (weight, Decimal("1") - weight), mode)


def _triple(primary: str, left: str, right: str, mode: str) -> Recipe:
    return Recipe(f"triple__{primary}__{left}__{right}__{mode}", (primary, left, right), (Decimal("0.70"), Decimal("0.15"), Decimal("0.15")), mode)


def _apply_recipe(recipe: Recipe, streams: Mapping[str, OOFStream]) -> np.ndarray:
    values = [streams[name].probability for name in recipe.components]
    if len(values) == 1:
        return np.array(values[0], copy=True)
    if len(values) == 2:
        return (blend_probability if recipe.mode == "probability" else blend_logit)(values[0], values[1], recipe.weights[0])
    if recipe.mode == "probability":
        return sum(float(weight) * value for weight, value in zip(recipe.weights, values, strict=True))
    logits = sum(float(weight) * _logit(value) for weight, value in zip(recipe.weights, values, strict=True))
    return _calibrate(1.0 / (1.0 + np.exp(-logits)), 1.0, 0.0)


def _oof_frame(frame: object) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or frame.empty or not frame.columns.is_unique:
        raise EnsembleError("OOF must be a nonempty DataFrame with unique columns")
    if set(frame.columns) != {"row_id", "valid_year", "target", "probability"}:
        raise EnsembleError("OOF calibration schema differs")
    work = frame.loc[:, ["row_id", "valid_year", "target", "probability"]].copy(deep=True)
    if any(type(value) is not str or not value or value != value.strip() for value in work["row_id"]) or work["row_id"].duplicated().any():
        raise EnsembleError("OOF row_id must be nonnull and unique")
    if any(isinstance(value, bool) or not isinstance(value, Integral) or not 1000 <= int(value) <= 9999 for value in work["valid_year"]):
        raise EnsembleError("valid_year must contain exact years")
    if any(isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in (0, 1) for value in work["target"]):
        raise EnsembleError("target must contain exact binary integers")
    probability = work["probability"].to_numpy()
    if np.issubdtype(probability.dtype, np.bool_) or not (np.issubdtype(probability.dtype, np.integer) or np.issubdtype(probability.dtype, np.floating)):
        raise EnsembleError("probability must be real numeric")
    probability = np.asarray(probability, dtype="float64")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise EnsembleError("probability must be finite and in [0, 1]")
    work["valid_year"] = work["valid_year"].astype("int64")
    work["target"] = work["target"].astype("int8")
    work["probability"] = probability
    return work.sort_values(["valid_year", "row_id"], kind="stable").reset_index(drop=True)
