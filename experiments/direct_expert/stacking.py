from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .inputs import canonical_json


class DirectExpertStackingError(ValueError):
    pass


def required_experts(expert_ids: tuple[str, ...]) -> tuple[str, ...]:
    required = set(expert_ids)
    if required.intersection({"D5", "D6"}):
        required.add("D0")
    return tuple(sorted(required, key=lambda item: int(item[1:])))


@dataclass(frozen=True)
class StackRecipe:
    method: str
    weights: Mapping[str, float]
    selection_years: tuple[int, ...]
    e2_weight: float = 0.0
    recipe_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.method not in {"probability", "logit"}:
            raise DirectExpertStackingError("stacking method differs")
        weights = {key: float(value) for key, value in self.weights.items()}
        if (
            not weights
            or any(key not in {f"D{i}" for i in range(8)} for key in weights)
            or any(not np.isfinite(value) or value < 0.05 for value in weights.values())
            or not np.isfinite(self.e2_weight)
            or self.e2_weight < 0
            or abs(sum(weights.values()) + self.e2_weight - 1.0) > 1e-12
        ):
            raise DirectExpertStackingError("stacking weights differ")
        dependencies = required_experts(tuple(weights))
        maximum = 2 if self.e2_weight > 0 else 3
        if len(dependencies) > maximum:
            raise DirectExpertStackingError("deployment dependency budget exceeded")
        if self.selection_years != (2022, 2023):
            raise DirectExpertStackingError("stacking selection years differ")
        frozen = MappingProxyType(dict(sorted(weights.items())))
        object.__setattr__(self, "weights", frozen)
        payload = {
            "method": self.method,
            "weights": dict(frozen),
            "selection_years": list(self.selection_years),
            "e2_weight": self.e2_weight,
        }
        object.__setattr__(self, "recipe_sha256", sha256(canonical_json(payload)).hexdigest())


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def apply_stack(
    recipe: StackRecipe,
    predictions: Mapping[str, np.ndarray],
    *,
    e2: np.ndarray | None = None,
) -> np.ndarray:
    if set(predictions) != set(recipe.weights):
        raise DirectExpertStackingError("stack prediction members differ")
    arrays = {name: np.asarray(value, dtype="float64") for name, value in predictions.items()}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or any(value.ndim != 1 for value in arrays.values()):
        raise DirectExpertStackingError("stack prediction shapes differ")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise DirectExpertStackingError("stack prediction contains non-finite values")
    if recipe.e2_weight:
        if e2 is None or np.asarray(e2).shape != next(iter(arrays.values())).shape:
            raise DirectExpertStackingError("E2 prediction differs")
        baseline = np.asarray(e2, dtype="float64")
        if not np.isfinite(baseline).all():
            raise DirectExpertStackingError("E2 prediction contains non-finite values")
    else:
        baseline = None
    if recipe.method == "probability":
        combined = sum(recipe.weights[name] * arrays[name] for name in recipe.weights)
        if baseline is not None:
            combined = combined + recipe.e2_weight * baseline
    else:
        combined_logit = sum(recipe.weights[name] * _logit(arrays[name]) for name in recipe.weights)
        if baseline is not None:
            combined_logit = combined_logit + recipe.e2_weight * _logit(baseline)
        combined = 1.0 / (1.0 + np.exp(-combined_logit))
    return np.clip(combined, 1e-6, 1 - 1e-6)


def _aligned_streams(
    streams: Mapping[str, pd.DataFrame],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not 1 <= len(streams) <= 4:
        raise DirectExpertStackingError("locked stream count differs")
    target: np.ndarray | None = None
    row_ids: tuple[str, ...] | None = None
    predictions: dict[str, np.ndarray] = {}
    for expert_id in sorted(streams, key=lambda item: int(item[1:])):
        frame = streams[expert_id]
        required = {"row_id", "target", "probability", "oof_year"}
        if type(frame) is not pd.DataFrame or set(frame) != required or frame.empty:
            raise DirectExpertStackingError("stack stream columns differ")
        aligned = frame.copy(deep=True).sort_values("row_id", kind="stable")
        if aligned["row_id"].isna().any() or not aligned["row_id"].is_unique:
            raise DirectExpertStackingError("stack row identity differs")
        if set(pd.to_numeric(aligned["oof_year"], errors="coerce")) != {2022, 2023}:
            raise DirectExpertStackingError("stack stream years differ")
        observed_ids = tuple(aligned["row_id"].astype(str))
        observed_target = pd.to_numeric(aligned["target"], errors="coerce").to_numpy()
        probability = pd.to_numeric(aligned["probability"], errors="coerce").to_numpy(dtype="float64")
        if not np.isin(observed_target, (0, 1)).all() or not np.isfinite(probability).all():
            raise DirectExpertStackingError("stack stream values differ")
        if row_ids is None:
            row_ids = observed_ids
            target = observed_target.astype("int8")
        elif row_ids != observed_ids or not np.array_equal(target, observed_target):
            raise DirectExpertStackingError("stack stream alignment differs")
        predictions[expert_id] = probability
    assert target is not None
    return target, predictions


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
    elif parts == 2:
        for left in range(1, total):
            yield (left, total - left)
    else:
        for first in range(1, total - 1):
            for second in range(1, total - first):
                yield (first, second, total - first - second)


def _score(
    method: str,
    weights: Mapping[str, float],
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> float:
    recipe = StackRecipe(method, weights, (2022, 2023))
    probability = apply_stack(recipe, {name: predictions[name] for name in weights})
    return float(np.mean(np.square(target - probability)))


def fit_stack(
    streams: Mapping[str, pd.DataFrame],
    *,
    method: str,
    confirmation: Mapping[str, pd.DataFrame] | None = None,
) -> StackRecipe:
    del confirmation
    target, predictions = _aligned_streams(streams)
    names = tuple(sorted(predictions, key=lambda item: int(item[1:])))
    best: tuple[float, tuple[str, ...], tuple[float, ...]] | None = None
    for size in range(1, min(3, len(names)) + 1):
        for active in combinations(names, size):
            if len(required_experts(active)) > 3:
                continue
            for units in _compositions(20, size):
                values = tuple(unit / 20.0 for unit in units)
                weights = dict(zip(active, values, strict=True))
                score = _score(method, weights, target, predictions)
                candidate = (score, active, values)
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        raise DirectExpertStackingError("no feasible stack recipe")
    _, active, values = best
    weights = dict(zip(active, values, strict=True))
    best_score = _score(method, weights, target, predictions)
    for step in (0.02, 0.01, 0.005):
        improved = True
        while improved:
            improved = False
            for source in active:
                for target_name in active:
                    if source == target_name or weights[source] - step < 0.05 - 1e-12:
                        continue
                    trial = dict(weights)
                    trial[source] -= step
                    trial[target_name] += step
                    score = _score(method, trial, target, predictions)
                    if score < best_score - 1e-15:
                        weights, best_score, improved = trial, score, True
    normalized = {name: value / sum(weights.values()) for name, value in weights.items()}
    return StackRecipe(method, normalized, (2022, 2023))


def fit_e2_safety_blend(
    direct: StackRecipe,
    streams: Mapping[str, pd.DataFrame],
    e2: np.ndarray,
) -> StackRecipe:
    dependencies = required_experts(tuple(direct.weights))
    if len(dependencies) > 2:
        raise DirectExpertStackingError("E2 blend direct dependency budget exceeded")
    target, predictions = _aligned_streams({name: streams[name] for name in direct.weights})
    baseline = np.asarray(e2, dtype="float64")
    if baseline.shape != target.shape or not np.isfinite(baseline).all():
        raise DirectExpertStackingError("E2 baseline differs")
    best: tuple[float, float, StackRecipe] | None = None
    for direct_weight in (0.60, 0.70, 0.80, 0.90):
        recipe = StackRecipe(
            direct.method,
            {name: weight * direct_weight for name, weight in direct.weights.items()},
            (2022, 2023),
            e2_weight=round(1.0 - direct_weight, 2),
        )
        probability = apply_stack(
            recipe,
            {name: predictions[name] for name in recipe.weights},
            e2=baseline,
        )
        score = float(np.mean(np.square(target - probability)))
        candidate = (score, -direct_weight, recipe)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2]
