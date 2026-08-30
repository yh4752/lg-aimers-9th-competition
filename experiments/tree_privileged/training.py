"""One fold/seed CatBoost residual student job."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Protocol
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from experiments.tree_expert.inputs import PREDICTION_COLUMNS, VerifiedOfficialData

from .contracts import load_contract
from .features import CandidateFeatureBatch, CandidateFeatureSkip, fit_candidate_features, transform_candidate_features
from .profiles import ProfileStrengths
from .teacher import TeacherEvidence


class PrivilegedTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrivilegedJob:
    job_id: str
    candidate_id: str
    train_end_year: int
    valid_year: int
    seed: int


@dataclass(frozen=True)
class CandidateJobResult:
    job_id: str
    candidate_id: str
    status: str
    brier: float | None
    model_path: Path | None
    predictions_path: Path | None
    failure: str | None
    best_iteration: int | None


class Regressor(Protocol):
    def fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs: object) -> object: ...
    def predict(self, x: pd.DataFrame) -> object: ...
    def save_model(self, path: str) -> None: ...
    def get_best_iteration(self) -> int: ...


ModelFactory = Callable[[dict[str, object]], Regressor]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    _atomic(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _baseline(frame: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or tuple(frame.columns) != PREDICTION_COLUMNS:
        raise PrivilegedTrainingError("baseline schema differs")
    if frame["row_id"].astype(str).tolist() != valid["row_id"].astype(str).tolist():
        raise PrivilegedTrainingError("baseline row order differs")
    if not np.array_equal(frame["target"].to_numpy(), valid["control_success"].to_numpy()):
        raise PrivilegedTrainingError("baseline target differs")
    return frame.copy(deep=True)


def _default_factory(parameters: dict[str, object]) -> Regressor:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise PrivilegedTrainingError("catboost==1.2.10 is required") from error
    return CatBoostRegressor(**parameters)


def _parameters(job: PrivilegedJob, output: Path, gpu_id: int) -> dict[str, object]:
    contract = load_contract().catboost
    return {
        "iterations": contract.iterations, "depth": contract.depth,
        "learning_rate": float(contract.learning_rate), "l2_leaf_reg": float(contract.l2_leaf_reg),
        "random_strength": float(contract.random_strength),
        "bagging_temperature": float(contract.bagging_temperature), "border_count": contract.border_count,
        "max_ctr_complexity": contract.max_ctr_complexity, "loss_function": "RMSE", "eval_metric": "RMSE",
        "random_seed": job.seed, "task_type": "GPU", "devices": str(gpu_id),
        "allow_writing_files": True, "train_dir": str(output / "catboost_info"), "verbose": 50,
    }


def _terminal(output: Path, result: CandidateJobResult, metrics: dict[str, object]) -> None:
    _atomic(output / "metrics.json", _canonical(metrics))
    _atomic(output / "worker_result.json", _canonical({
        "job_id": result.job_id, "candidate_id": result.candidate_id, "status": result.status,
        "brier": result.brier, "model": None if result.model_path is None else result.model_path.name,
        "predictions": None if result.predictions_path is None else result.predictions_path.name,
        "failure": result.failure, "best_iteration": result.best_iteration,
    }))


def _state_zip(output: Path, state: object) -> None:
    payload = _canonical({
        "candidate_id": getattr(state, "candidate_id"),
        "feature_columns": list(getattr(state, "feature_columns")),
        "categorical_columns": list(getattr(state, "categorical_columns")),
        "profile_columns": list(getattr(state, "profile_columns")),
        "teacher_evidence_hashes": dict(getattr(state, "teacher_evidence_hashes")),
    })
    temporary = output / ".feature_state.zip.tmp"
    info = ZipInfo("feature_state.json", date_time=(2026, 1, 1, 0, 0, 0)); info.compress_type = ZIP_DEFLATED
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(info, payload)
    os.replace(temporary, output / "feature_state.zip")


def run_candidate_job(
    *,
    job: PrivilegedJob,
    data: VerifiedOfficialData,
    baseline: pd.DataFrame,
    output_dir: Path,
    absolute_deadline: float,
    gpu_id: int,
    model_factory: ModelFactory | None = None,
    feature_builder=fit_candidate_features,
    feature_transformer=transform_candidate_features,
    teacher_evidence: TeacherEvidence | None = None,
    strengths: ProfileStrengths | tuple[int, int, int] = (75, 150, 300),
) -> CandidateJobResult:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    _atomic(output / "job.json", _canonical({**asdict(job), "gpu_id": gpu_id}))
    try:
        if time.time() >= absolute_deadline:
            raise TimeoutError("candidate deadline reached")
        all_rows = pd.read_csv(data.train)
        history = pd.read_csv(data.history)
        fit_rows = all_rows.loc[all_rows["season"].le(job.train_end_year)].copy()
        valid_labeled = all_rows.loc[all_rows["season"].eq(job.valid_year)].copy()
        aligned = _baseline(baseline, valid_labeled)
        state, train_batch = feature_builder(
            fit_rows, history, valid_year=job.valid_year, candidate_id=job.candidate_id,
            teacher_evidence=teacher_evidence, strengths=strengths,
        )
        valid_batch = feature_transformer(valid_labeled.drop(columns="control_success"), state)
        if train_batch.soft_target is None or valid_batch.soft_target is not None:
            raise PrivilegedTrainingError("soft-target boundary differs")
        factory = model_factory or _default_factory
        model = factory(_parameters(job, output, gpu_id))
        residual = train_batch.soft_target - train_batch.anchor
        valid_target = valid_labeled["control_success"].to_numpy(dtype="float64")
        model.fit(
            train_batch.frame, residual,
            eval_set=(valid_batch.frame, valid_target - valid_batch.anchor),
            cat_features=list(state.categorical_columns), use_best_model=True,
            early_stopping_rounds=load_contract().catboost.od_wait,
            save_snapshot=True, snapshot_file=str(output / "experiment.cbsnapshot"), snapshot_interval=600,
        )
        probability = np.clip(valid_batch.anchor + np.asarray(model.predict(valid_batch.frame), dtype="float64"), 1e-5, 1 - 1e-5)
        if probability.shape != valid_target.shape or not np.isfinite(probability).all():
            raise PrivilegedTrainingError("candidate probabilities differ")
        predictions = aligned.copy(); predictions["probability"] = probability
        predictions_path = output / "predictions.csv"; _atomic_frame(predictions_path, predictions.loc[:, PREDICTION_COLUMNS])
        model_path = output / "model.cbm"; temporary = output / ".model.cbm.tmp"
        model.save_model(str(temporary)); os.replace(temporary, model_path)
        _state_zip(output, state)
        _atomic(output / "teacher_evidence.json", _canonical(dict(state.teacher_evidence_hashes)))
        _atomic(output / "profile_evidence.json", _canonical({
            "profile_columns": list(state.profile_columns),
            "strengths": None if state.selected_strengths is None else asdict(state.selected_strengths),
        }))
        brier = float(np.mean(np.square(probability - valid_target)))
        best_iteration = int(model.get_best_iteration())
        result = CandidateJobResult(job.job_id, job.candidate_id, "completed", brier, model_path,
                                    predictions_path, None, best_iteration)
        _terminal(output, result, {"brier": brier, "baseline_brier": float(np.mean(np.square(aligned["probability"] - valid_target))),
                                   "best_iteration": best_iteration, "row_count": len(valid_target)})
        return result
    except CandidateFeatureSkip as error:
        result = CandidateJobResult(job.job_id, job.candidate_id, "skipped", None, None, None, str(error), None)
        _terminal(output, result, {"status": "skipped", "reason": str(error)})
        return result
    except Exception as error:
        result = CandidateJobResult(job.job_id, job.candidate_id, "failed", None, None, None,
                                    f"{type(error).__name__}: {error}", None)
        _terminal(output, result, {"status": "failed", "reason": result.failure})
        return result
