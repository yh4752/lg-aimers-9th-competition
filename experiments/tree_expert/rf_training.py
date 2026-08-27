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
from .rf_contracts import RFContract, RFJob, load_rf_contract


class RFTrainingError(ValueError):
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
class RFJobResult:
    status: str
    job_id: str
    predictions: pd.DataFrame
    brier: float | None
    segment_brier: float | None
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
        raise RFTrainingError("catboost==1.2.10 is required") from error
    return CatBoostRegressor(**parameters)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode())


def _validate_rows(frame: pd.DataFrame, *, target: bool) -> None:
    required = {"row_id", "season", "game_type"}
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns):
        raise RFTrainingError("training frame schema differs")
    if target and "control_success" not in frame:
        raise RFTrainingError("target is missing")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise TreeFeatureError("row_id values must be non-null and unique")
    if not frame["game_type"].astype(str).isin(["R", "F"]).all():
        raise RFTrainingError("game_type values differ")


def _aligned_baseline(
    baseline: pd.DataFrame,
    row_id: np.ndarray,
    game_type: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    required = {"row_id", "target", "probability", "game_type"}
    if type(baseline) is not pd.DataFrame or not required.issubset(baseline.columns):
        raise RFTrainingError("baseline schema differs")
    expected = np.asarray(row_id, dtype=str)
    actual = baseline["row_id"].astype(str).to_numpy()
    if not np.array_equal(expected, actual):
        raise RFTrainingError("baseline row identity differs")
    if not np.array_equal(baseline["game_type"].astype(str).to_numpy(), np.asarray(game_type, dtype=str)):
        raise RFTrainingError("baseline game_type differs")
    target = pd.to_numeric(baseline["target"], errors="raise").to_numpy(dtype="float64")
    probability = pd.to_numeric(baseline["probability"], errors="raise").to_numpy(dtype="float64")
    if (
        not np.isin(target, [0.0, 1.0]).all()
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
    ):
        raise RFTrainingError("baseline probability differs")
    return target, probability


def _parameters(contract: RFContract, job: RFJob, output_dir: Path, gpu_id: int) -> dict[str, object]:
    parameters = dict(contract.catboost_common)
    parameters.update(contract.experts[job.head])
    parameters.pop("segment", None)
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
            raise RFTrainingError("CatBoost model output is empty")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _failure(output: Path, job: RFJob, error: Exception) -> RFJobResult:
    message = f"{type(error).__name__}: {error}"
    _atomic_bytes(
        output / "metrics.json",
        _json_bytes({"status": "failed", "job_id": job.job_id, "failure": message}),
    )
    _atomic_bytes(output / "worker.log", f"status=failed failure={message}\n".encode())
    return RFJobResult(
        "failed", job.job_id, pd.DataFrame(), None, None, None, None,
        None, None, None, message,
    )


def run_rf_job(
    *,
    job: RFJob,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    baseline: pd.DataFrame,
    output_dir: Path,
    gpu_id: int = 0,
    contract: RFContract | None = None,
    model_factory: ModelFactory = _default_model_factory,
    feature_builder: FeatureBuilder = fit_tree_features,
    feature_transformer: FeatureTransformer = transform_tree_features,
) -> RFJobResult:
    if type(job) is not RFJob:
        raise RFTrainingError("RF job type differs")
    active = load_rf_contract() if contract is None else contract
    if job.head not in active.experts or active.experts[job.head]["segment"] != job.segment:
        raise RFTrainingError("RF job identity differs")
    _validate_rows(train, target=True)
    _validate_rows(valid, target=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    prefix = train.loc[pd.to_numeric(train["season"], errors="raise").le(job.train_end_year)].copy()
    validation = valid.loc[pd.to_numeric(valid["season"], errors="raise").eq(job.valid_year)].copy()
    segment_train = prefix.loc[prefix["game_type"].astype(str).eq(job.segment)].copy()
    if segment_train.empty:
        raise RFTrainingError("segment training rows are empty")
    if len(segment_train) < active.gates.minimum_segment_rows:
        raise RFTrainingError("segment training rows are below minimum")
    if validation.empty:
        raise RFTrainingError("validation rows are empty")
    try:
        state, train_batch = feature_builder(
            segment_train,
            None,
            valid_year=job.valid_year,
            use_trackman=False,
        )
        valid_batch = feature_transformer(validation.drop(columns="control_success"), state)
        if train_batch.target is None:
            raise RFTrainingError("training target differs")
        residual = np.asarray(train_batch.target, dtype="float64") - np.asarray(
            train_batch.anchor, dtype="float64"
        )
        valid_segment = validation["game_type"].astype(str).to_numpy() == job.segment
        if not valid_segment.any():
            raise RFTrainingError("validation segment rows are empty")
        model = model_factory(_parameters(active, job, output, gpu_id))
        model.fit(
            train_batch.frame,
            residual,
            cat_features=list(getattr(state, "categorical_columns", ())),
            eval_set=(
                valid_batch.frame.loc[valid_segment],
                validation["control_success"].to_numpy(dtype="float64")[valid_segment]
                - np.asarray(valid_batch.anchor, dtype="float64")[valid_segment],
            ),
            use_best_model=True,
            verbose=50,
        )
        probability = np.clip(
            np.asarray(valid_batch.anchor, dtype="float64")
            + np.asarray(model.predict(valid_batch.frame), dtype="float64"),
            1e-5,
            1 - 1e-5,
        )
        if len(probability) != len(validation) or not np.isfinite(probability).all():
            raise RFTrainingError("expert probability differs")
        baseline_target, baseline_probability = _aligned_baseline(
            baseline,
            valid_batch.row_id,
            validation["game_type"].astype(str).to_numpy(),
        )
        target = validation["control_success"].to_numpy(dtype="float64")
        if not np.array_equal(target, baseline_target):
            raise RFTrainingError("baseline target differs")
        brier = float(np.mean(np.square(probability - target)))
        segment_brier = float(np.mean(np.square(probability[valid_segment] - target[valid_segment])))
        baseline_brier = float(np.mean(np.square(baseline_probability - target)))
        if not all(math.isfinite(value) for value in (brier, segment_brier, baseline_brier)):
            raise RFTrainingError("RF Brier is not finite")
        prediction = baseline.copy(deep=True)
        prediction["probability"] = probability
        predictions_path = output / "predictions.csv"
        model_path = output / "checkpoint.cbm"
        _save_model(model, model_path)
        _atomic_frame(predictions_path, prediction)
        best_iteration = max(0, int(model.get_best_iteration()))
        gain = baseline_brier - brier
        metrics = {
            "status": "completed",
            "job_id": job.job_id,
            "head": job.head,
            "segment": job.segment,
            "seed": job.seed,
            "training_rows": len(segment_train),
            "validation_rows": len(validation),
            "validation_segment_rows": int(valid_segment.sum()),
            "brier": brier,
            "segment_brier": segment_brier,
            "baseline_brier": baseline_brier,
            "gain": gain,
            "best_iteration": best_iteration,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_bytes(output / "job.json", _json_bytes(job.__dict__))
        _atomic_bytes(output / "metrics.json", _json_bytes(metrics))
        _atomic_bytes(
            output / "worker.log",
            f"status=completed brier={brier:.12f} segment_brier={segment_brier:.12f} gain={gain:.12f}\n".encode(),
        )
        return RFJobResult(
            "completed", job.job_id, prediction, brier, segment_brier,
            baseline_brier, gain, best_iteration, model_path, predictions_path, None,
        )
    except (TreeFeatureError, RFTrainingError):
        raise
    except Exception as error:
        return _failure(output, job, error)


def load_rf_job_result(output_dir: Path, expected_job_id: str) -> RFJobResult:
    output = Path(output_dir)
    try:
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RFTrainingError("job metrics cannot be loaded") from error
    if type(metrics) is not dict or metrics.get("job_id") != expected_job_id:
        raise RFTrainingError("job metrics identity differs")
    if metrics.get("status") == "failed":
        return RFJobResult(
            "failed", expected_job_id, pd.DataFrame(), None, None, None,
            None, None, None, None, str(metrics.get("failure")),
        )
    predictions_path = output / "predictions.csv"
    model_path = output / "checkpoint.cbm"
    if metrics.get("status") != "completed" or not predictions_path.is_file() or not model_path.is_file():
        raise RFTrainingError("completed job files differ")
    return RFJobResult(
        "completed",
        expected_job_id,
        pd.read_csv(predictions_path),
        float(metrics["brier"]),
        float(metrics["segment_brier"]),
        float(metrics["baseline_brier"]),
        float(metrics["gain"]),
        int(metrics["best_iteration"]),
        model_path,
        predictions_path,
        None,
    )
