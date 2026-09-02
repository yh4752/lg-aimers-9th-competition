from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np
import pandas as pd

from experiments.direct_expert.features import DirectFeatureBatch, feature_profile

from .contracts import RoleSpec, load_contract, role_spec


class E3TrainingError(ValueError):
    pass


class Estimator(Protocol):
    def fit(self, frame: pd.DataFrame, target: np.ndarray, **kwargs: object) -> None: ...
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...
    def save_model(self, path: str) -> None: ...


ModelFactory = Callable[[dict[str, object]], Estimator]


@dataclass(frozen=True)
class E3Job:
    job_id: str
    role_id: str
    fold: tuple[int, int]
    seed: int
    phase: str


@dataclass(frozen=True)
class FoldData:
    train: DirectFeatureBatch
    valid: DirectFeatureBatch
    valid_target: np.ndarray
    valid_metadata: pd.DataFrame
    categorical_columns: tuple[str, ...]
    subtype_targets: pd.DataFrame
    bindings: Mapping[str, str]


@dataclass(frozen=True)
class FoldJobResult:
    status: str
    job_id: str
    output_dir: Path
    predictions: pd.DataFrame
    brier: float | None
    best_iteration: int


_OUTPUTS = {"model.cbm", "predictions.csv", "metrics.json", "job_identity.json", "worker.log"}
_OPTIONAL_OUTPUTS = {"training.snapshot", "active_identity.json"}


def season_weights(spec: RoleSpec, seasons: np.ndarray, valid_year: int) -> np.ndarray:
    values = np.asarray(seasons)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise E3TrainingError("season values differ")
    age = valid_year - 1 - values.astype("int64")
    if (age < 0).any() or spec.decay is None:
        raise E3TrainingError("season cutoff differs")
    return np.power(float(spec.decay), age, dtype="float64")


def role_training_target(
    spec: RoleSpec,
    batch: DirectFeatureBatch,
    subtype_targets: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if batch.target is None:
        raise E3TrainingError("training target is absent")
    success = np.asarray(batch.target, dtype="float64")
    if success.shape != (len(batch.frame),) or not np.isin(success, (0, 1)).all():
        raise E3TrainingError("training target differs")
    if spec.target == "success":
        target = success
        mask = np.ones(len(batch.frame), dtype=bool)
    else:
        required = {"valid", "middle", "wild", "reverse"}
        if type(subtype_targets) is not pd.DataFrame or not required.issubset(subtype_targets.columns) or len(subtype_targets) != len(batch.frame):
            raise E3TrainingError("subtype target frame differs")
        mask = subtype_targets["valid"].to_numpy(dtype=bool)
        raw = pd.to_numeric(subtype_targets[spec.target], errors="coerce")
        if raw.loc[mask].isna().any() or not raw.loc[mask].isin((0, 1)).all():
            raise E3TrainingError("subtype target values differ")
        target = raw.fillna(0).to_numpy(dtype="float64")
    if spec.game_type is not None:
        mask &= np.asarray(batch.game_type == spec.game_type, dtype=bool)
    return target, mask


def catboost_parameters(spec: RoleSpec, *, seed: int, gpu: int, output: Path) -> dict[str, object]:
    if type(seed) is not int or type(gpu) is not int or gpu < 0:
        raise E3TrainingError("worker identity differs")
    contract = load_contract()
    source = contract.success_parameters if spec.target == "success" else contract.subtype_parameters
    parameters: dict[str, object] = dict(source)
    parameters.update(
        loss_function="Logloss",
        eval_metric="BrierScore",
        random_seed=seed,
        task_type="GPU",
        devices=str(gpu),
        allow_writing_files=True,
        train_dir=str(Path(output) / "catboost_info"),
        save_snapshot=True,
        snapshot_file=str(Path(output) / "training.snapshot"),
        snapshot_interval=int(contract.runtime["snapshot_interval_seconds"]),
        verbose=100,
    )
    return parameters


def _default_factory(parameters: dict[str, object]) -> Estimator:
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise E3TrainingError("CatBoost 1.2.10 is required") from error
    return CatBoostClassifier(**parameters)


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise E3TrainingError("job evidence is not canonicalizable") from error


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _identity(job: E3Job, data: FoldData, parameters: Mapping[str, object]) -> dict[str, object]:
    spec = role_spec(job.role_id)
    columns, categorical = feature_profile(data.train.frame, data.categorical_columns, spec.profile)
    payload = {
        "schema_version": 1,
        "job": {
            "job_id": job.job_id,
            "role_id": job.role_id,
            "fold": list(job.fold),
            "seed": job.seed,
            "phase": job.phase,
        },
        "bindings": dict(sorted(data.bindings.items())),
        "parameters": dict(sorted(parameters.items())),
        "feature_columns": list(columns),
        "categorical_columns": list(categorical),
    }
    payload["identity_sha256"] = sha256(_canonical(payload)).hexdigest()
    return payload


def _load(output: Path, identity: Mapping[str, object]) -> FoldJobResult | None:
    if not output.exists():
        return None
    if not output.is_dir():
        raise E3TrainingError("existing job output differs")
    files = {path.name for path in output.iterdir() if path.is_file()}
    directories = {path.name for path in output.iterdir() if path.is_dir()}
    if files - _OUTPUTS - _OPTIONAL_OUTPUTS or directories - {"catboost_info"}:
        raise E3TrainingError("existing job output differs")
    if not _OUTPUTS.issubset(files):
        if files.intersection(_OUTPUTS) or "active_identity.json" not in files:
            raise E3TrainingError("existing job output differs")
        active = json.loads((output / "active_identity.json").read_text(encoding="utf-8"))
        if active != identity:
            raise E3TrainingError("active job identity differs")
        return None
    observed = json.loads((output / "job_identity.json").read_text(encoding="utf-8"))
    if observed != identity:
        raise E3TrainingError("existing job identity differs")
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output / "predictions.csv")
    return FoldJobResult(
        "completed", metrics["job_id"], output, predictions,
        None if metrics["brier"] is None else float(metrics["brier"]),
        int(metrics["best_iteration"]),
    )


def run_fold_job(
    job: E3Job,
    data: FoldData,
    output_dir: Path,
    *,
    gpu: int,
    model_factory: ModelFactory | None = None,
) -> FoldJobResult:
    contract = load_contract()
    if job.fold not in contract.folds or job.seed not in contract.seeds or job.phase not in {"screening", "confirmation", "extra_seeds"}:
        raise E3TrainingError("job contract differs")
    spec = role_spec(job.role_id, contract)
    if data.train.target is None or data.valid.target is not None:
        raise E3TrainingError("fold target placement differs")
    if set(data.valid_metadata) != {"row_id", "game_type", "pitcher_id", "oof_year"}:
        raise E3TrainingError("validation metadata differs")
    if tuple(data.valid_metadata["row_id"].astype(str)) != tuple(map(str, data.valid.row_id)):
        raise E3TrainingError("validation row alignment differs")
    if any(type(value) is not str or len(value) != 64 for value in data.bindings.values()):
        raise E3TrainingError("fold bindings differ")
    output = Path(output_dir)
    parameters = catboost_parameters(spec, seed=job.seed, gpu=gpu, output=output)
    identity = _identity(job, data, parameters)
    reusable = _load(output, identity)
    if reusable is not None:
        return reusable
    output.mkdir(parents=True, exist_ok=True)
    _atomic(output / "active_identity.json", _canonical(identity))
    target, mask = role_training_target(spec, data.train, data.subtype_targets)
    weights = season_weights(spec, data.train.season, job.fold[1])
    mask &= weights > 0
    if mask.sum() < 2 or len(np.unique(target[mask])) != 2:
        raise E3TrainingError("role training sample differs")
    columns, categorical = feature_profile(data.train.frame, data.categorical_columns, spec.profile)
    category_indices = [columns.index(name) for name in categorical]
    model = (model_factory or _default_factory)(parameters)
    fit_kwargs: dict[str, object] = {
        "sample_weight": weights[mask],
        "cat_features": category_indices,
    }
    valid_target = np.asarray(data.valid_target, dtype="int8")
    if spec.target == "success":
        active_valid = np.ones(len(data.valid.frame), dtype=bool)
        if spec.game_type is not None:
            active_valid &= data.valid_metadata["game_type"].eq(spec.game_type).to_numpy()
        if active_valid.any():
            fit_kwargs["eval_set"] = (
                data.valid.frame.loc[active_valid, columns], valid_target[active_valid],
            )
            fit_kwargs["use_best_model"] = True
    else:
        fit_kwargs["use_best_model"] = False
    model.fit(data.train.frame.loc[mask, columns], target[mask], **fit_kwargs)
    raw = np.asarray(model.predict_proba(data.valid.frame.loc[:, columns]), dtype="float64")
    if raw.shape != (len(data.valid.frame), 2) or not np.isfinite(raw).all():
        raise E3TrainingError("validation probability differs")
    probability = np.clip(raw[:, 1], 1e-6, 1.0 - 1e-6)
    predictions = data.valid_metadata.copy(deep=True)
    predictions["target"] = valid_target
    predictions["probability"] = probability
    predictions["role_id"] = job.role_id
    predictions["seed"] = job.seed
    brier = float(np.mean(np.square(valid_target - probability))) if spec.target == "success" else None
    observed_best_iteration = getattr(model, "get_best_iteration", lambda: -1)()
    best_iteration = -1 if observed_best_iteration is None else int(observed_best_iteration)
    if best_iteration < 0:
        best_iteration = int(parameters["iterations"]) - 1
    model.save_model(str(output / "model.cbm"))
    predictions.to_csv(output / "predictions.csv", index=False)
    _atomic(output / "job_identity.json", _canonical(identity))
    _atomic(output / "metrics.json", _canonical({
        "schema_version": 1,
        "job_id": job.job_id,
        "role_id": job.role_id,
        "brier": brier,
        "best_iteration": best_iteration,
        "rows": len(predictions),
    }))
    _atomic(output / "worker.log", f"status=completed job={job.job_id}\n".encode("utf-8"))
    return FoldJobResult("completed", job.job_id, output, predictions, brier, best_iteration)
