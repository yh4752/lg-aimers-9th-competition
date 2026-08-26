from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .contracts import E1Contract, E1Job, load_e1_contract
from .failure_labels import FailureLabelAudit, audit_failure_labels
from .features import (
    TreeFeatureBatch,
    TreeFeatureSkip,
    fit_tree_features,
    transform_tree_features,
)
from .inputs import PREDICTION_COLUMNS, VerifiedOfficialData


class TreeTrainingError(RuntimeError):
    """Raised when an E1 fold worker cannot produce aligned evidence."""


class TrainableModel(Protocol):
    classes_: np.ndarray

    def fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs: object) -> object: ...
    def predict(self, x: pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...
    def save_model(self, path: str) -> None: ...
    def get_best_iteration(self) -> int: ...


ModelFactory = Callable[[str, dict[str, object]], TrainableModel]
FeatureBuilder = Callable[..., tuple[object, TreeFeatureBatch]]
FeatureTransformer = Callable[[pd.DataFrame, object], TreeFeatureBatch]
FailureAuditor = Callable[..., FailureLabelAudit]


@dataclass(frozen=True)
class FoldResult:
    job_id: str
    candidate_id: str
    status: str
    brier: float | None
    model_path: Path | None
    predictions_path: Path | None
    snapshot_path: Path | None
    failure: str | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _default_model_factory(kind: str, parameters: dict[str, object]) -> TrainableModel:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as error:
        raise TreeTrainingError("catboost==1.2.10 is required") from error
    if kind == "residual":
        return CatBoostRegressor(**parameters)
    return CatBoostClassifier(**parameters)


def _check_deadline(deadline: float) -> None:
    if time.time() >= deadline:
        raise TimeoutError("E1 job deadline reached")


def _aligned_baseline(
    baseline: pd.DataFrame,
    valid_rows: pd.DataFrame,
) -> pd.DataFrame:
    if type(baseline) is not pd.DataFrame or tuple(baseline.columns) != PREDICTION_COLUMNS:
        raise TreeTrainingError("baseline prediction schema differs")
    if baseline["row_id"].astype(str).tolist() != valid_rows["row_id"].astype(str).tolist():
        raise TreeTrainingError("baseline row_id alignment differs")
    target = pd.to_numeric(valid_rows["control_success"], errors="coerce").to_numpy()
    baseline_target = pd.to_numeric(baseline["target"], errors="coerce").to_numpy()
    if not np.array_equal(target, baseline_target):
        raise TreeTrainingError("baseline target alignment differs")
    probability = pd.to_numeric(baseline["probability"], errors="coerce")
    if probability.isna().any() or not probability.between(0, 1).all():
        raise TreeTrainingError("baseline probability values are invalid")
    return baseline.copy(deep=True)


def _cat_columns(state: object, batch: TreeFeatureBatch) -> list[str]:
    declared = getattr(state, "categorical_columns", None)
    if isinstance(declared, tuple):
        columns = list(declared)
    else:
        columns = [
            column
            for column in batch.frame.columns
            if batch.frame[column].dtype == object
            or isinstance(batch.frame[column].dtype, pd.StringDtype)
        ]
    if any(column not in batch.frame for column in columns):
        raise TreeTrainingError("categorical feature schema differs")
    return columns


def _parameters(
    contract: E1Contract,
    job: E1Job,
    output_dir: Path,
    gpu_id: int,
) -> dict[str, object]:
    common = dict(contract.catboost_parameters)
    common.update(
        random_seed=job.seed,
        devices=str(gpu_id),
        train_dir=str(output_dir / "catboost_info"),
    )
    if job.objective == "binary":
        common.update(loss_function="Logloss", eval_metric="Logloss")
    elif job.objective == "residual":
        common.update(loss_function="RMSE", eval_metric="RMSE")
    elif job.objective == "multiclass":
        common.update(loss_function="MultiClass", eval_metric="MultiClass")
    else:
        raise TreeTrainingError(f"unsupported E1 objective: {job.objective}")
    return common


def _save_model(model: TrainableModel, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        model.save_model(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise TreeTrainingError("CatBoost model output is empty")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_terminal(
    output_dir: Path,
    result: FoldResult,
    metrics: dict[str, object],
) -> None:
    _atomic_bytes(output_dir / "metrics.json", _canonical_json(metrics))
    payload = {
        "job_id": result.job_id,
        "candidate_id": result.candidate_id,
        "status": result.status,
        "brier": result.brier,
        "model": result.model_path.name if result.model_path else None,
        "predictions": result.predictions_path.name if result.predictions_path else None,
        "snapshot": result.snapshot_path.name if result.snapshot_path else None,
        "failure": result.failure,
    }
    _atomic_bytes(output_dir / "worker_result.json", _canonical_json(payload))
    _atomic_bytes(
        output_dir / "worker.log",
        (
            f"TREE_E1_JOB_END job={result.job_id} status={result.status} "
            f"failure={result.failure or 'none'}\n"
        ).encode("utf-8"),
    )


def run_e1_job(
    *,
    job: E1Job,
    data: VerifiedOfficialData,
    baseline: pd.DataFrame,
    output_dir: Path,
    absolute_deadline: float,
    gpu_id: int,
    contract: E1Contract | None = None,
    model_factory: ModelFactory | None = None,
    feature_builder: FeatureBuilder = fit_tree_features,
    feature_transformer: FeatureTransformer = transform_tree_features,
    failure_auditor: FailureAuditor = audit_failure_labels,
) -> FoldResult:
    active_contract = load_e1_contract() if contract is None else contract
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(
        output / "job.json",
        _canonical_json({**asdict(job), "gpu_id": gpu_id}),
    )
    snapshot_path = output / "experiment.cbsnapshot"
    try:
        _check_deadline(absolute_deadline)
        all_rows = pd.read_csv(data.train)
        history = pd.read_csv(data.history) if job.use_trackman else None
        train_rows = all_rows.loc[all_rows["season"].le(job.train_end_year)].copy(deep=True)
        valid_labeled = all_rows.loc[all_rows["season"].eq(job.valid_year)].copy(deep=True)
        if train_rows.empty or valid_labeled.empty:
            raise TreeTrainingError("fold rows are empty")
        aligned = _aligned_baseline(baseline, valid_labeled)
        valid_rows = valid_labeled.drop(columns="control_success")

        _check_deadline(absolute_deadline)
        state, train_batch = feature_builder(
            train_rows,
            history,
            valid_year=job.valid_year,
            use_trackman=job.use_trackman,
            minimum_trackman_coverage=float(
                active_contract.trackman_gate["minimum_accepted_coverage"]
            ),
        )
        valid_batch = feature_transformer(valid_rows, state)
        if train_batch.target is None or valid_batch.target is not None:
            raise TreeTrainingError("feature target boundary differs")
        if valid_batch.row_id.astype(str).tolist() != aligned["row_id"].astype(str).tolist():
            raise TreeTrainingError("feature row_id alignment differs")
        cat_columns = _cat_columns(state, train_batch)

        factory = _default_model_factory if model_factory is None else model_factory
        model = factory(
            job.objective,
            _parameters(active_contract, job, output, gpu_id),
        )
        common_fit = {
            "cat_features": cat_columns,
            "save_snapshot": True,
            "snapshot_file": str(snapshot_path),
            "snapshot_interval": active_contract.snapshot_interval_seconds,
            "verbose": 50,
        }
        valid_target = pd.to_numeric(
            valid_labeled["control_success"], errors="coerce"
        ).to_numpy(dtype="float64")

        if job.objective == "binary":
            model.fit(
                train_batch.frame,
                train_batch.target,
                eval_set=(valid_batch.frame, valid_target),
                use_best_model=True,
                early_stopping_rounds=60,
                **common_fit,
            )
            probability = np.asarray(model.predict_proba(valid_batch.frame))[:, 1]
        elif job.objective == "residual":
            residual = train_batch.target.astype("float64") - train_batch.anchor
            model.fit(
                train_batch.frame,
                residual,
                eval_set=(valid_batch.frame, valid_target - valid_batch.anchor),
                use_best_model=True,
                early_stopping_rounds=60,
                **common_fit,
            )
            probability = np.clip(
                valid_batch.anchor + np.asarray(model.predict(valid_batch.frame)),
                1e-5,
                1 - 1e-5,
            )
        else:
            audit = failure_auditor(
                train_rows,
                gate=active_contract.failure_label_gate,
                valid_year=job.valid_year,
            )
            _atomic_bytes(
                output / "failure_label_audit.json",
                _canonical_json(
                    {
                        "status": audit.status,
                        "reason": audit.reason,
                        "coverage": audit.coverage,
                        "binary_delta_fraction": audit.binary_delta_fraction,
                        "success_agreement": audit.success_agreement,
                        "middle_reverse_overlap": audit.middle_reverse_overlap,
                        "class_counts": dict(audit.class_counts),
                    }
                ),
            )
            if audit.status != "passed":
                raise TreeFeatureSkip(audit.reason)
            class_map = {"success": 0, "middle": 1, "reverse": 2, "other_failure": 3}
            class_target = np.asarray([class_map[str(label)] for label in audit.labels], dtype="int8")
            model.fit(
                train_batch.frame.iloc[audit.source_positions],
                class_target,
                use_best_model=False,
                **common_fit,
            )
            classes = np.asarray(model.classes_)
            success_columns = np.flatnonzero(classes == 0)
            if success_columns.size != 1:
                raise TreeTrainingError("multiclass success column differs")
            probability = np.asarray(model.predict_proba(valid_batch.frame))[
                :, int(success_columns[0])
            ]

        _check_deadline(absolute_deadline)
        probability = np.asarray(probability, dtype="float64")
        if probability.shape != valid_target.shape or not np.isfinite(probability).all():
            raise TreeTrainingError("candidate probability shape or values differ")
        if np.any((probability < 0) | (probability > 1)):
            raise TreeTrainingError("candidate probability is outside [0, 1]")
        predictions = aligned.copy(deep=True)
        predictions["probability"] = probability
        brier = float(np.mean(np.square(probability - valid_target)))
        baseline_probability = aligned["probability"].to_numpy(dtype="float64")
        baseline_brier = float(np.mean(np.square(baseline_probability - valid_target)))
        predictions_path = output / "predictions.csv"
        model_path = output / "model.cbm"
        _atomic_frame(predictions_path, predictions.loc[:, PREDICTION_COLUMNS])
        _save_model(model, model_path)
        result = FoldResult(
            job_id=job.job_id,
            candidate_id=job.candidate_id,
            status="completed",
            brier=brier,
            model_path=model_path,
            predictions_path=predictions_path,
            snapshot_path=snapshot_path if snapshot_path.is_file() else None,
            failure=None,
        )
        _write_terminal(
            output,
            result,
            {
                "objective": job.objective,
                "brier": brier,
                "baseline_brier": baseline_brier,
                "gain": baseline_brier - brier,
                "best_iteration": int(model.get_best_iteration()),
                "row_count": len(predictions),
            },
        )
        return result
    except TreeFeatureSkip as error:
        result = FoldResult(
            job_id=job.job_id,
            candidate_id=job.candidate_id,
            status="skipped",
            brier=None,
            model_path=None,
            predictions_path=None,
            snapshot_path=snapshot_path if snapshot_path.is_file() else None,
            failure=str(error),
        )
        _write_terminal(
            output,
            result,
            {"objective": job.objective, "status": "skipped", "reason": str(error)},
        )
        return result
    except Exception as error:
        result = FoldResult(
            job_id=job.job_id,
            candidate_id=job.candidate_id,
            status="failed",
            brier=None,
            model_path=None,
            predictions_path=None,
            snapshot_path=snapshot_path if snapshot_path.is_file() else None,
            failure=f"{type(error).__name__}: {error}",
        )
        _write_terminal(
            output,
            result,
            {"objective": job.objective, "status": "failed", "reason": result.failure},
        )
        return result
