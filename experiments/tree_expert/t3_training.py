from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .features import TreeFeatureBatch, TreeFeatureError, fit_tree_features, transform_tree_features
from .t3_contracts import T3Contract, T3Job, load_t3_contract
from .t3_temporal import temporal_training_weights


class T3TrainingError(ValueError):
    pass


class TrainableRegressor(Protocol):
    def fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs: object) -> object: ...
    def predict(self, x: pd.DataFrame) -> np.ndarray: ...
    def get_best_iteration(self) -> int: ...
    def save_model(self, path: str) -> None: ...


ModelFactory = Callable[[dict[str, object]], TrainableRegressor]
FeatureBuilder = Callable[..., tuple[object, TreeFeatureBatch]]
FeatureTransformer = Callable[[pd.DataFrame, object], TreeFeatureBatch]


@dataclass(frozen=True)
class T3JobResult:
    status: str
    job_id: str
    predictions: pd.DataFrame
    brier: float | None
    baseline_brier: float | None
    gain: float | None
    best_iteration: int | None
    model_path: Path | None
    predictions_path: Path | None
    failure: str | None


def _default_model_factory(parameters: dict[str, object]) -> TrainableRegressor:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise T3TrainingError("catboost==1.2.10 is required") from error
    return CatBoostRegressor(**parameters)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8"))


def _validate_rows(frame: pd.DataFrame, *, target: bool) -> None:
    if type(frame) is not pd.DataFrame or "row_id" not in frame or "season" not in frame:
        raise T3TrainingError("training frame schema differs")
    if target and "control_success" not in frame:
        raise T3TrainingError("target is missing")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise TreeFeatureError("row_id values must be non-null and unique")


def _aligned_baseline(baseline: pd.DataFrame, row_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    required = {"row_id", "target", "probability"}
    if type(baseline) is not pd.DataFrame or not required.issubset(baseline.columns):
        raise T3TrainingError("baseline schema differs")
    expected = np.asarray(row_id, dtype=str)
    actual = baseline["row_id"].astype(str).to_numpy()
    if not np.array_equal(expected, actual):
        raise T3TrainingError("baseline row identity differs")
    target = pd.to_numeric(baseline["target"], errors="raise").to_numpy(dtype="float64")
    probability = pd.to_numeric(baseline["probability"], errors="raise").to_numpy(dtype="float64")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise T3TrainingError("baseline probability differs")
    return target, probability


def _parameters(contract: T3Contract, job: T3Job, output_dir: Path, gpu_id: int) -> dict[str, object]:
    parameters = dict(contract.catboost)
    parameters.update(
        random_seed=job.seed,
        devices=str(gpu_id),
        train_dir=str(output_dir / "catboost_info"),
        loss_function="RMSE",
        eval_metric="RMSE",
    )
    return parameters


def _save_model(model: TrainableRegressor, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        model.save_model(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise T3TrainingError("CatBoost model output is empty")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _failure(output: Path, job: T3Job, error: Exception) -> T3JobResult:
    message = f"{type(error).__name__}: {error}"
    payload = {"status": "failed", "job_id": job.job_id, "failure": message}
    _atomic_bytes(output / "metrics.json", _json_bytes(payload))
    _atomic_bytes(output / "worker.log", f"status=failed failure={message}\n".encode("utf-8"))
    return T3JobResult("failed", job.job_id, pd.DataFrame(), None, None, None, None, None, None, message)


def run_t3_job(
    *,
    job: T3Job,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    baseline: pd.DataFrame,
    output_dir: Path,
    gpu_id: int = 0,
    contract: T3Contract | None = None,
    model_factory: ModelFactory = _default_model_factory,
    feature_builder: FeatureBuilder = fit_tree_features,
    feature_transformer: FeatureTransformer = transform_tree_features,
) -> T3JobResult:
    if type(job) is not T3Job:
        raise T3TrainingError("T3 job type differs")
    _validate_rows(train, target=True)
    _validate_rows(valid, target=True)
    active = load_t3_contract() if contract is None else contract
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    prefix = train.loc[pd.to_numeric(train["season"], errors="raise").le(job.train_end_year)].copy()
    validation = valid.loc[pd.to_numeric(valid["season"], errors="raise").eq(job.valid_year)].copy()
    if prefix.empty or validation.empty:
        raise T3TrainingError("fold rows are empty")
    try:
        state, train_batch = feature_builder(
            prefix, None, valid_year=job.valid_year, use_trackman=False,
        )
        valid_batch = feature_transformer(validation.drop(columns="control_success"), state)
        if train_batch.target is None:
            raise T3TrainingError("training target differs")
        sample_weight = temporal_training_weights(
            prefix["season"], valid_year=job.valid_year, head=job.head, decay=job.decay,
        )
        selected = sample_weight > 0
        residual = np.asarray(train_batch.target, dtype="float64")[selected] - np.asarray(train_batch.anchor)[selected]
        model = model_factory(_parameters(active, job, output, gpu_id))
        model.fit(
            train_batch.frame.loc[selected], residual,
            sample_weight=sample_weight[selected],
            cat_features=list(getattr(state, "categorical_columns", ())),
            eval_set=(valid_batch.frame, validation["control_success"].to_numpy(dtype="float64") - valid_batch.anchor),
            use_best_model=True,
            verbose=50,
        )
        probability = np.clip(
            np.asarray(valid_batch.anchor, dtype="float64") + np.asarray(model.predict(valid_batch.frame), dtype="float64"),
            1e-5, 1 - 1e-5,
        )
        baseline_target, baseline_probability = _aligned_baseline(baseline, valid_batch.row_id)
        target = validation["control_success"].to_numpy(dtype="float64")
        if not np.array_equal(target, baseline_target):
            raise T3TrainingError("baseline target differs")
        brier = float(np.mean(np.square(probability - target)))
        baseline_brier = float(np.mean(np.square(baseline_probability - target)))
        if not math.isfinite(brier):
            raise T3TrainingError("T3 Brier is not finite")
        prediction = baseline.copy(deep=True)
        prediction["probability"] = probability
        predictions_path = output / "predictions.csv"
        model_path = output / "checkpoint.cbm"
        _save_model(model, model_path)
        _atomic_frame(predictions_path, prediction)
        best_iteration = max(0, int(model.get_best_iteration()))
        gain = baseline_brier - brier
        metrics = {
            "status": "completed", "job_id": job.job_id, "head": job.head,
            "decay": job.decay, "seed": job.seed, "row_count": len(prediction),
            "weight_sum": float(sample_weight.sum()), "brier": brier,
            "baseline_brier": baseline_brier, "gain": gain,
            "best_iteration": best_iteration,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_bytes(output / "job.json", _json_bytes(job.__dict__))
        _atomic_bytes(output / "metrics.json", _json_bytes(metrics))
        _atomic_bytes(output / "worker.log", f"status=completed brier={brier:.12f} gain={gain:.12f}\n".encode("utf-8"))
        return T3JobResult(
            "completed", job.job_id, prediction, brier, baseline_brier, gain,
            best_iteration, model_path, predictions_path, None,
        )
    except (TreeFeatureError, T3TrainingError):
        raise
    except Exception as error:
        return _failure(output, job, error)
