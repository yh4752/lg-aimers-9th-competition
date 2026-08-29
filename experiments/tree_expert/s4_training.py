from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .s4_contracts import S4Contract, load_s4_contract
from .s4_temporal import probability_vector


class S4TrainingError(ValueError):
    pass


class Regressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: object) -> object: ...
    def predict(self, x: np.ndarray) -> object: ...


@dataclass(frozen=True)
class S4FitResult:
    prediction: np.ndarray
    best_iteration: int
    model: object


def _finite_vector(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype="float64")
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise S4TrainingError(f"{label} must be a finite vector")
    return result


def binary_target(value: object) -> np.ndarray:
    result = _finite_vector(value, "target")
    if not np.isin(result, [0.0, 1.0]).all():
        raise S4TrainingError("target must be binary")
    return result


def residual_target(target: object, anchor: object) -> np.ndarray:
    y = binary_target(target)
    p = probability_vector(anchor, "anchor")
    if y.shape != p.shape:
        raise S4TrainingError("residual rows differ")
    return y - p


def corrected_probability(anchor: object, correction: object, *, alpha: float) -> np.ndarray:
    p = probability_vector(anchor, "anchor")
    r = _finite_vector(correction, "correction")
    if p.shape != r.shape or type(alpha) not in {int, float} or not np.isfinite(alpha) or alpha <= 0.0:
        raise S4TrainingError("correction differs")
    return np.clip(p + float(alpha) * r, 1e-5, 1.0 - 1e-5)


def route_rf(game_type: object, r_prediction: object, f_prediction: object) -> np.ndarray:
    route = np.asarray(game_type, dtype=str)
    r = _finite_vector(r_prediction, "R prediction")
    f = _finite_vector(f_prediction, "F prediction")
    if route.ndim != 1 or route.shape != r.shape or r.shape != f.shape or not np.isin(route, ["R", "F"]).all():
        raise S4TrainingError("R/F routing differs")
    return np.where(route == "F", f, r)


def model_parameters(contract: S4Contract, family: str, *, gpu_id: int, seed: int = 3407) -> dict[str, object]:
    base_family = "catboost" if family in {"catboost", "catboost_rf", "dual_temporal"} else family
    if base_family not in contract.parameters or gpu_id not in {0, 1}:
        raise S4TrainingError("model identity differs")
    values = dict(contract.parameters[base_family])
    if base_family == "catboost":
        values.update({
            "loss_function": "RMSE", "random_seed": seed, "task_type": "GPU",
            "devices": str(gpu_id), "allow_writing_files": False, "verbose": False,
        })
    elif base_family == "xgboost":
        values.update({"objective": "reg:squarederror", "random_state": seed, "tree_method": "hist", "device": f"cuda:{gpu_id}"})
    else:
        values.update({"objective": "regression", "random_state": seed, "verbosity": -1})
    return values


def default_model_factory(family: str, parameters: dict[str, object]) -> Regressor:
    base = "catboost" if family in {"catboost", "catboost_rf", "dual_temporal"} else family
    if base == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**parameters)
    if base == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**parameters)
    if base == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**parameters)
    raise S4TrainingError("model family differs")


def fit_residual_estimator(
    *,
    family: str,
    train_matrix: np.ndarray,
    train_target: object,
    valid_matrix: np.ndarray,
    gpu_id: int,
    seed: int,
    sample_weight: np.ndarray | None = None,
    contract: S4Contract | None = None,
    model_factory: Callable[[str, dict[str, object]], Regressor] = default_model_factory,
) -> S4FitResult:
    active = load_s4_contract() if contract is None else contract
    x_train = np.asarray(train_matrix, dtype="float32")
    x_valid = np.asarray(valid_matrix, dtype="float32")
    y = _finite_vector(train_target, "residual target")
    if x_train.ndim != 2 or x_valid.ndim != 2 or len(x_train) != len(y) or x_train.shape[1] != x_valid.shape[1]:
        raise S4TrainingError("training matrices differ")
    model = model_factory(family, model_parameters(active, family, gpu_id=gpu_id, seed=seed))
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        weights = _finite_vector(sample_weight, "sample weight")
        if weights.shape != y.shape or np.any(weights <= 0.0):
            raise S4TrainingError("sample weights differ")
        kwargs["sample_weight"] = weights
    model.fit(x_train, y, **kwargs)
    prediction = _finite_vector(model.predict(x_valid), "model prediction")
    if len(prediction) != len(x_valid):
        raise S4TrainingError("prediction rows differ")
    best = int(getattr(model, "best_iteration_", getattr(model, "best_iteration", -1)))
    return S4FitResult(prediction, best, model)
