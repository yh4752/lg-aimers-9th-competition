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

from .hetero_contracts import HeteroContract, HeteroJob, load_hetero_contract
from .hetero_features import (
    HeteroFeatureBatch,
    HeteroFeatureError,
    fit_hetero_features,
    transform_hetero_features,
)


class HeteroTrainingError(ValueError):
    pass


class BinaryClassifier(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: object) -> object: ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


ModelFactory = Callable[[str, dict[str, object]], BinaryClassifier]
FeatureBuilder = Callable[..., tuple[object, HeteroFeatureBatch]]
FeatureTransformer = Callable[[pd.DataFrame, object], HeteroFeatureBatch]


@dataclass(frozen=True)
class HeteroJobResult:
    status: str
    job_id: str
    predictions: pd.DataFrame
    brier: float | None
    baseline_brier: float | None
    best_iteration: int | None
    model_path: Path | None
    predictions_path: Path | None
    failure: str | None


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
    _atomic_bytes(path, payload)


def _parameters(contract: HeteroContract, job: HeteroJob, gpu_id: int) -> dict[str, object]:
    parameters = dict(contract.parameters[job.family])
    parameters["random_state"] = job.seed
    if job.family == "xgboost":
        parameters.update(
            objective="binary:logistic", eval_metric="logloss",
            device=f"cuda:{gpu_id}", early_stopping_rounds=80,
        )
    else:
        parameters.update(objective="binary", device_type="cpu", verbosity=-1)
    return parameters


def _default_model_factory(family: str, parameters: dict[str, object]) -> BinaryClassifier:
    if family == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise HeteroTrainingError("xgboost==3.0.2 is required") from error
        return XGBClassifier(**parameters)
    if family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise HeteroTrainingError("lightgbm==4.6.0 is required") from error
        parameters = dict(parameters)
        parameters.pop("early_stopping_rounds", None)
        return LGBMClassifier(**parameters)
    raise HeteroTrainingError("model family differs")


def _baseline(frame: pd.DataFrame, row_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    required = {"row_id", "target", "probability"}
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns):
        raise HeteroTrainingError("baseline schema differs")
    expected = np.asarray(row_id, dtype=str)
    actual = frame["row_id"].astype(str).to_numpy()
    if not np.array_equal(expected, actual):
        raise HeteroTrainingError("baseline row identity differs")
    target = pd.to_numeric(frame["target"], errors="raise").to_numpy(dtype="float64")
    probability = pd.to_numeric(frame["probability"], errors="raise").to_numpy(dtype="float64")
    if not np.isin(target, [0, 1]).all() or not np.isfinite(probability).all():
        raise HeteroTrainingError("baseline values differ")
    if np.any((probability < 0) | (probability > 1)):
        raise HeteroTrainingError("baseline probability differs")
    return target, probability


def _fit(model: BinaryClassifier, family: str, train: HeteroFeatureBatch, valid: HeteroFeatureBatch, target: np.ndarray) -> None:
    kwargs: dict[str, object] = {"eval_set": [(valid.matrix, target)]}
    if family == "xgboost":
        kwargs["verbose"] = 50
    else:
        if model.__class__.__module__.split(".")[0] == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as error:
                raise HeteroTrainingError("lightgbm==4.6.0 is required") from error
            kwargs["callbacks"] = [lgb.early_stopping(80), lgb.log_evaluation(50)]
    model.fit(train.matrix, np.asarray(train.target, dtype="int8"), **kwargs)


def _save_model(model: BinaryClassifier, family: str, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    if hasattr(model, "save_model"):
        model.save_model(str(temporary))  # type: ignore[attr-defined]
    elif family == "lightgbm" and hasattr(model, "booster_"):
        model.booster_.save_model(str(temporary))  # type: ignore[attr-defined]
    else:
        raise HeteroTrainingError("model cannot be saved")
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise HeteroTrainingError("model output is empty")
    os.replace(temporary, path)


def run_hetero_job(
    *,
    job: HeteroJob,
    train: pd.DataFrame,
    baseline: pd.DataFrame,
    output_dir: Path,
    gpu_id: int = 0,
    contract: HeteroContract | None = None,
    model_factory: ModelFactory = _default_model_factory,
    feature_builder: FeatureBuilder = fit_hetero_features,
    feature_transformer: FeatureTransformer = transform_hetero_features,
) -> HeteroJobResult:
    active = load_hetero_contract() if contract is None else contract
    if type(job) is not HeteroJob or job.family not in active.families:
        raise HeteroTrainingError("hetero job identity differs")
    required = {"row_id", "season", "control_success"}
    if type(train) is not pd.DataFrame or not required.issubset(train.columns):
        raise HeteroTrainingError("training frame schema differs")
    if train["row_id"].isna().any() or not train["row_id"].is_unique:
        raise HeteroTrainingError("training row identity differs")
    season = pd.to_numeric(train["season"], errors="raise")
    prefix = train.loc[season.le(job.train_end_year)].copy()
    validation = train.loc[season.eq(job.valid_year)].copy()
    if prefix.empty or validation.empty:
        raise HeteroTrainingError("fold rows are empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        state, train_batch = feature_builder(prefix, valid_year=job.valid_year)
        valid_batch = feature_transformer(validation.drop(columns="control_success"), state)
        if train_batch.target is None or valid_batch.target is not None:
            raise HeteroTrainingError("feature target boundary differs")
        baseline_target, baseline_probability = _baseline(baseline, valid_batch.row_id)
        target = validation["control_success"].to_numpy(dtype="float64")
        if not np.array_equal(target, baseline_target):
            raise HeteroTrainingError("baseline target differs")
        model = model_factory(job.family, _parameters(active, job, gpu_id))
        _fit(model, job.family, train_batch, valid_batch, target)
        probability = np.asarray(model.predict_proba(valid_batch.matrix), dtype="float64")
        if probability.ndim != 2 or probability.shape != (len(validation), 2):
            raise HeteroTrainingError("model probability shape differs")
        model_probability = probability[:, 1]
        if not np.isfinite(model_probability).all() or np.any((model_probability < 0) | (model_probability > 1)):
            raise HeteroTrainingError("model probability differs")
        prediction = baseline.rename(columns={"probability": "baseline_probability"}).copy()
        prediction.insert(3, "model_probability", model_probability)
        predictions_path = output / "predictions.csv"
        model_path = output / ("model.json" if job.family == "xgboost" else "model.txt")
        _save_model(model, job.family, model_path)
        _atomic_frame(predictions_path, prediction)
        brier = float(np.mean(np.square(model_probability - target)))
        baseline_brier = float(np.mean(np.square(baseline_probability - target)))
        if not math.isfinite(brier):
            raise HeteroTrainingError("model Brier is not finite")
        best_iteration = int(getattr(model, "best_iteration", getattr(model, "best_iteration_", -1)))
        metrics = {
            "status": "completed", "job_id": job.job_id, "family": job.family,
            "seed": job.seed, "brier": brier, "baseline_brier": baseline_brier,
            "best_iteration": best_iteration, "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_bytes(output / "job.json", json.dumps(job.__dict__, sort_keys=True, separators=(",", ":")).encode())
        _atomic_bytes(output / "metrics.json", json.dumps(metrics, sort_keys=True, separators=(",", ":")).encode())
        _atomic_bytes(output / "worker.log", f"status=completed brier={brier:.12f}\n".encode())
        return HeteroJobResult(
            "completed", job.job_id, prediction, brier, baseline_brier,
            best_iteration, model_path, predictions_path, None,
        )
    except (HeteroFeatureError, HeteroTrainingError):
        raise
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        _atomic_bytes(output / "metrics.json", json.dumps({
            "status": "failed", "job_id": job.job_id, "failure": failure,
        }, sort_keys=True, separators=(",", ":")).encode())
        _atomic_bytes(output / "worker.log", f"status=failed failure={failure}\n".encode())
        return HeteroJobResult("failed", job.job_id, pd.DataFrame(), None, None, None, None, None, failure)
