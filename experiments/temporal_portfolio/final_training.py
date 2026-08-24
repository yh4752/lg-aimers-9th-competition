"""Fail-closed resolution of approved temporal full-fit work."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from numbers import Integral
import time
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .features import PortfolioFeatureSpec, normalize_feature_spec


class FinalTrainingError(RuntimeError):
    """Raised when temporal evidence cannot authorize a final fit."""


@dataclass(frozen=True)
class T4Decision:
    status: str
    decision_sha256: str
    decay: Decimal
    recent_weight: Decimal
    best_epoch_by_year: Mapping[int, int]
    feature_spec: PortfolioFeatureSpec
    model_spec: Mapping[str, object]
    catboost_prefix: int | None

    def __post_init__(self) -> None:
        if self.status not in ("accepted", "rejected", "budget_inconclusive"):
            raise FinalTrainingError("T4 decision status is invalid")
        if not _is_sha256(self.decision_sha256):
            raise FinalTrainingError("T4 decision SHA-256 is invalid")
        if self.decay not in tuple(map(Decimal, ("0.40", "0.55", "0.70", "1.00"))):
            raise FinalTrainingError("T4 decay is invalid")
        if not Decimal("0") < self.recent_weight <= Decimal("1"):
            raise FinalTrainingError("T4 recent weight is invalid")
        epochs = _epochs(self.best_epoch_by_year)
        spec = normalize_feature_spec(self.feature_spec)
        if not isinstance(self.model_spec, Mapping) or not self.model_spec:
            raise FinalTrainingError("T4 model spec is invalid")
        model = dict(self.model_spec)
        if model.get("family") not in ("tabm", "catboost"):
            raise FinalTrainingError("T4 model family is invalid")
        if model["family"] == "catboost" and self.catboost_prefix not in (16, 64, 192, 384):
            raise FinalTrainingError("approved CatBoost prefix is missing")
        if model["family"] != "catboost" and self.catboost_prefix is not None:
            raise FinalTrainingError("non-CatBoost decision has a prefix")
        object.__setattr__(self, "best_epoch_by_year", MappingProxyType(epochs))
        object.__setattr__(self, "feature_spec", spec)
        object.__setattr__(self, "model_spec", MappingProxyType(model))


@dataclass(frozen=True)
class FullFitPlan:
    decision_sha256: str
    recent_years: tuple[int, ...]
    multi_years: tuple[int, ...]
    multi_weight_by_year: Mapping[int, float]
    decay: Decimal
    recent_weight: Decimal
    epochs: int
    feature_spec: PortfolioFeatureSpec
    model_spec: Mapping[str, object]
    catboost_prefix: int | None


@dataclass(frozen=True)
class InferenceBenchmark:
    rows: int
    elapsed_seconds: float
    accepted: bool


def resolve_final_epoch(best_epoch_by_year: Mapping[int, int]) -> int:
    values = _epochs(best_epoch_by_year)
    minimum = min(values.values())
    if max(values.values()) > max(12, 4 * minimum):
        raise FinalTrainingError("fold epochs are unstable")
    weights = {2022: Decimal("0.2"), 2023: Decimal("0.3"), 2024: Decimal("0.5")}
    cumulative = Decimal("0")
    for epoch, weight in sorted((values[year], weights[year]) for year in weights):
        cumulative += weight
        if cumulative >= Decimal("0.5"):
            return max(2, epoch)
    raise AssertionError("unreachable weighted median")


def build_full_fit_plan(train: pd.DataFrame, decision: T4Decision) -> FullFitPlan:
    if type(decision) is not T4Decision or decision.status != "accepted":
        raise FinalTrainingError("T4 acceptance evidence is required")
    if type(train) is not pd.DataFrame or "season" not in train or train.empty:
        raise FinalTrainingError("official training seasons are missing")
    seasons = train["season"]
    if seasons.isna().any() or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in seasons
    ):
        raise FinalTrainingError("official training seasons are invalid")
    required = (2021, 2022, 2023, 2024)
    if not set(required).issubset({int(value) for value in seasons}):
        raise FinalTrainingError("official training seasons are incomplete")
    decay = float(decision.decay)
    weights = MappingProxyType(
        {year: decay ** (2024 - year) for year in required}
    )
    return FullFitPlan(
        decision.decision_sha256,
        (2024,),
        required,
        weights,
        decision.decay,
        decision.recent_weight,
        resolve_final_epoch(decision.best_epoch_by_year),
        decision.feature_spec,
        decision.model_spec,
        decision.catboost_prefix,
    )


def benchmark_inference(
    predictor: object,
    rows: pd.DataFrame,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> InferenceBenchmark:
    if type(rows) is not pd.DataFrame or rows.empty:
        raise FinalTrainingError("inference rows are invalid")
    start = float(clock())
    try:
        probability = np.asarray(predictor.predict(rows), dtype="float64")
    except Exception as error:
        raise FinalTrainingError("full inference failed") from error
    elapsed = float(clock()) - start
    if probability.shape != (len(rows),) or not np.isfinite(probability).all():
        raise FinalTrainingError("full inference probabilities are invalid")
    if np.any((probability < 0) | (probability > 1)):
        raise FinalTrainingError("full inference probabilities are invalid")
    if not math.isfinite(elapsed) or elapsed < 0 or elapsed > 480:
        raise FinalTrainingError("full inference budget exceeded")
    return InferenceBenchmark(len(rows), elapsed, True)


def _epochs(value: Mapping[int, int]) -> dict[int, int]:
    if not isinstance(value, Mapping) or set(value) != {2022, 2023, 2024}:
        raise FinalTrainingError("final epoch evidence differs")
    result = dict(value)
    if any(
        isinstance(epoch, bool) or not isinstance(epoch, Integral) or int(epoch) < 1
        for epoch in result.values()
    ):
        raise FinalTrainingError("final epoch evidence is invalid")
    return {int(year): int(epoch) for year, epoch in result.items()}


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
