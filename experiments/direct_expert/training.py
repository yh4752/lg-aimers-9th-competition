from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np
import pandas as pd

from .contracts import DirectExpertContract, ExpertJob, ExpertSpec, expert_spec
from .features import DirectFeatureBatch, feature_profile
from .inputs import canonical_json


class DirectExpertTrainingError(ValueError):
    pass


class Estimator(Protocol):
    def fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs: object) -> None: ...
    def save_model(self, path: str) -> None: ...


ModelFactory = Callable[[str, dict[str, object]], Estimator]


@dataclass(frozen=True)
class FoldData:
    train: DirectFeatureBatch
    valid: DirectFeatureBatch
    valid_target: np.ndarray
    valid_metadata: pd.DataFrame
    categorical_columns: tuple[str, ...]
    bindings: Mapping[str, str]


@dataclass(frozen=True)
class FoldJobResult:
    status: str
    job_id: str
    output_dir: Path
    predictions: pd.DataFrame
    brier: float
    best_iteration: int


_REQUIRED_OUTPUTS = {
    "predictions.csv",
    "metrics.json",
    "job_identity.json",
    "model.cbm",
    "worker.log",
}


def season_weights(spec: ExpertSpec, seasons: np.ndarray, valid_year: int) -> np.ndarray:
    values = np.asarray(seasons)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise DirectExpertTrainingError("season values differ")
    if type(valid_year) is not int or isinstance(valid_year, bool):
        raise DirectExpertTrainingError("valid_year differs")
    age = valid_year - 1 - values.astype("int64")
    if (age < 0).any():
        raise DirectExpertTrainingError("season reaches validation year")
    if spec.recent_seasons is not None:
        return (age < spec.recent_seasons).astype("float64")
    if spec.decay is not None:
        return np.power(spec.decay, age, dtype="float64")
    return np.ones(len(values), dtype="float64")


def training_mask(spec: ExpertSpec, batch: DirectFeatureBatch) -> np.ndarray:
    if spec.game_type is None:
        return np.ones(len(batch.frame), dtype=bool)
    return np.asarray(batch.game_type == spec.game_type, dtype=bool)


def catboost_parameters(
    spec: ExpertSpec,
    contract: DirectExpertContract,
    *,
    seed: int,
    gpu: int,
    phase: str,
) -> dict[str, object]:
    if phase not in {"screening", "confirmation", "extra_seeds", "full_fit"}:
        raise DirectExpertTrainingError("training phase differs")
    source = (
        contract.screening_parameters
        if phase == "screening"
        else contract.final_parameters
    )
    parameters: dict[str, object] = dict(source)
    parameters.update(
        {
            "random_seed": seed,
            "task_type": "GPU",
            "devices": str(gpu),
            "allow_writing_files": True,
            "save_snapshot": True,
            "snapshot_interval": int(contract.runtime["snapshot_interval_seconds"]),
            "verbose": 100,
        }
    )
    if spec.objective == "Logloss":
        parameters.update(loss_function="Logloss", eval_metric="BrierScore")
    elif spec.objective == "RMSE":
        parameters.update(loss_function="RMSE", eval_metric="RMSE")
    else:
        raise DirectExpertTrainingError("expert objective differs")
    return parameters


def _default_model_factory(objective: str, parameters: dict[str, object]) -> Estimator:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as error:
        raise DirectExpertTrainingError("CatBoost 1.2.10 is required") from error
    if objective == "Logloss":
        return CatBoostClassifier(**parameters)
    return CatBoostRegressor(**parameters)


def _validate_fold(data: FoldData, job: ExpertJob) -> None:
    if type(data) is not FoldData:
        raise DirectExpertTrainingError("fold data type differs")
    if data.train.target is None or data.valid.target is not None:
        raise DirectExpertTrainingError("fold target placement differs")
    target = np.asarray(data.valid_target)
    if target.shape != (len(data.valid.frame),) or not np.isin(target, (0, 1)).all():
        raise DirectExpertTrainingError("validation target differs")
    required = {"row_id", "game_type", "pitcher_id"}
    if set(data.valid_metadata) != required or len(data.valid_metadata) != len(target):
        raise DirectExpertTrainingError("validation metadata differs")
    if tuple(data.valid_metadata["row_id"].astype(str)) != tuple(map(str, data.valid.row_id)):
        raise DirectExpertTrainingError("validation row alignment differs")
    if job.fold[1] <= job.fold[0]:
        raise DirectExpertTrainingError("job fold differs")
    for name, value in data.bindings.items():
        if type(name) is not str or type(value) is not str or len(value) != 64:
            raise DirectExpertTrainingError("fold bindings differ")


def _identity(job: ExpertJob, data: FoldData, parameters: Mapping[str, object]) -> dict[str, object]:
    spec = expert_spec(job.expert_id)
    feature_columns, categorical_columns = feature_profile(
        data.train.frame,
        data.categorical_columns,
        spec.interaction_profile,
    )
    payload = {
        "schema_version": 1,
        "job": asdict(job),
        "bindings": dict(sorted(data.bindings.items())),
        "parameters": dict(sorted(parameters.items())),
        "categorical_columns": list(categorical_columns),
        "feature_columns": list(feature_columns),
    }
    payload["identity_sha256"] = sha256(canonical_json(payload)).hexdigest()
    return payload


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_reusable(output: Path, identity: Mapping[str, object]) -> FoldJobResult | None:
    if not output.exists():
        return None
    if not output.is_dir() or not _REQUIRED_OUTPUTS.issubset(
        path.name for path in output.iterdir() if path.is_file()
    ):
        raise DirectExpertTrainingError("existing job output is incomplete")
    try:
        observed = json.loads((output / "job_identity.json").read_text(encoding="utf-8"))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(output / "predictions.csv")
    except Exception as error:
        raise DirectExpertTrainingError("existing job output is unreadable") from error
    if observed != identity:
        raise DirectExpertTrainingError("existing job identity differs")
    return FoldJobResult(
        status="completed",
        job_id=str(metrics["job_id"]),
        output_dir=output,
        predictions=predictions,
        brier=float(metrics["brier"]),
        best_iteration=int(metrics["best_iteration"]),
    )


def run_fold_job(
    job: ExpertJob,
    data: FoldData,
    output_dir: Path,
    *,
    model_factory: ModelFactory | None = None,
    gpu: int = 0,
    contract: DirectExpertContract | None = None,
) -> FoldJobResult:
    from .contracts import load_contract

    active = contract or load_contract()
    spec = expert_spec(job.expert_id)
    _validate_fold(data, job)
    parameters = catboost_parameters(spec, active, seed=job.seed, gpu=gpu, phase=job.phase)
    output = Path(output_dir)
    parameters["train_dir"] = str(output / "catboost_info")
    parameters["snapshot_file"] = str(output / "catboost_snapshot.cbsnapshot")
    identity = _identity(job, data, parameters)
    reusable = _load_reusable(output, identity)
    if reusable is not None:
        return reusable
    output.mkdir(parents=True, exist_ok=False)
    weights = season_weights(spec, data.train.season, job.fold[1])
    mask = training_mask(spec, data.train) & (weights > 0)
    if not mask.any() or len(np.unique(data.train.target[mask])) < 2:
        raise DirectExpertTrainingError("expert training subset is not binary")
    feature_columns, categorical_columns = feature_profile(
        data.train.frame,
        data.categorical_columns,
        spec.interaction_profile,
    )
    categories = [feature_columns.index(name) for name in categorical_columns]
    factory = model_factory or _default_model_factory
    model = factory(spec.objective, parameters)
    model.fit(
        data.train.frame.loc[mask, feature_columns],
        data.train.target[mask],
        sample_weight=weights[mask],
        cat_features=categories,
        eval_set=(data.valid.frame.loc[:, feature_columns], data.valid_target),
        use_best_model=True,
    )
    if spec.objective == "Logloss":
        raw = np.asarray(model.predict_proba(data.valid.frame.loc[:, feature_columns]), dtype="float64")[:, 1]
    else:
        raw = np.asarray(model.predict(data.valid.frame.loc[:, feature_columns]), dtype="float64")
    probability = np.clip(raw, 1e-6, 1 - 1e-6)
    if probability.shape != data.valid_target.shape or not np.isfinite(probability).all():
        raise DirectExpertTrainingError("validation probability differs")
    predictions = data.valid_metadata.copy(deep=True)
    predictions.insert(1, "target", data.valid_target.astype("int8"))
    predictions.insert(2, "probability", probability)
    predictions["oof_year"] = job.fold[1]
    predictions = predictions.loc[:, ["row_id", "target", "probability", "game_type", "pitcher_id", "oof_year"]]
    brier = float(np.mean(np.square(data.valid_target - probability)))
    best_iteration = int(getattr(model, "get_best_iteration", lambda: -1)())
    model.save_model(str(output / "model.cbm"))
    _write(output / "predictions.csv", predictions.to_csv(index=False, lineterminator="\n").encode())
    _write(
        output / "metrics.json",
        canonical_json(
            {
                "schema_version": 1,
                "job_id": job.job_id,
                "brier": brier,
                "best_iteration": best_iteration,
                "rows": len(predictions),
            }
        ),
    )
    _write(output / "job_identity.json", canonical_json(identity))
    _write(output / "worker.log", f"DIRECT_EXPERT_JOB_END candidate={job.job_id} status=completed\n".encode())
    return FoldJobResult("completed", job.job_id, output, predictions, brier, best_iteration)
