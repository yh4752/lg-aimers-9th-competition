from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.independent_dl.models.tabm import TabMAdapter
from experiments.independent_dl.training import TrainRequest, fit_candidate

from .context_features import fit_context_state
from .contracts import HierarchicalJob
from .feature_adapter import (
    feature_state_payload,
    prepare_fold,
)
from .metrics import build_segment_columns


class HierarchicalTrainingError(ValueError):
    """Raised before an unsafe hierarchical TabM job can start."""


PREDICTION_COLUMNS = (
    "row_id",
    "target",
    "probability",
    "season",
    "game_month",
    "game_type",
    "count_state",
    "hand_matchup",
    "base_out_state",
    "pitcher_known",
    "batter_known",
)


@dataclass(frozen=True)
class TrainingJobResult:
    job_id: str
    kind: str
    status: str
    train_rows: int
    valid_rows: int | None
    best_epoch: int | None
    best_brier: float | None
    completed_epochs: int
    predictions_path: Path | None
    checkpoint_path: Path | None
    final_model_path: Path | None
    feature_state_path: Path | None
    failure: str | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise HierarchicalTrainingError("identity must be a non-empty mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise HierarchicalTrainingError("identity must contain lowercase SHA-256 values")
        result[key] = digest
    return MappingProxyType(result)


def _load_train(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / "train.csv"
    if path.is_symlink() or not path.is_file():
        raise HierarchicalTrainingError("official train.csv is missing")
    try:
        frame = pd.read_csv(path, dtype={"base_state": "string", "row_id": "string"})
    except Exception as error:
        raise HierarchicalTrainingError(f"cannot read train.csv: {error}") from error
    required = {
        "row_id", "season", "control_success", "game_month", "game_type",
        "pitcher_id", "batter_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise HierarchicalTrainingError(f"train.csv missing columns: {', '.join(missing)}")
    if frame.empty or frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise HierarchicalTrainingError("train.csv row_id is invalid")
    season = pd.to_numeric(frame["season"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(season).all() or not np.equal(season, np.floor(season)).all():
        raise HierarchicalTrainingError("season is invalid")
    if ((season < 2019) | (season > 2024)).any():
        raise HierarchicalTrainingError("train.csv season is outside 2019..2024")
    return frame


def _resume_meta_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".meta.json")


def _validate_resume_checkpoint(
    checkpoint: Path, job: HierarchicalJob, identity: Mapping[str, str]
) -> int:
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise HierarchicalTrainingError("resume checkpoint is missing")
    meta_path = _resume_meta_path(checkpoint)
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HierarchicalTrainingError(f"resume checkpoint metadata is invalid: {error}") from error
    expected = {
        "schema_version", "job_id", "identity", "checkpoint_sha256", "completed_epochs"
    }
    if type(payload) is not dict or set(payload) != expected:
        raise HierarchicalTrainingError("resume checkpoint metadata differs")
    if payload["schema_version"] != 1 or payload["job_id"] != job.job_id:
        raise HierarchicalTrainingError("resume checkpoint job identity differs")
    if payload["identity"] != dict(identity):
        raise HierarchicalTrainingError("resume checkpoint identity differs")
    if payload["checkpoint_sha256"] != _file_sha256(checkpoint):
        raise HierarchicalTrainingError("resume checkpoint SHA-256 differs")
    epochs = payload["completed_epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise HierarchicalTrainingError("resume completed_epochs is invalid")
    return epochs


def _model_config() -> dict[str, object]:
    return {
        "architecture": "tabm",
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.1,
        "num_embedding": "piecewise_linear",
    }


def _training_config(*, epochs: int, full_fit: bool) -> dict[str, object]:
    return {
        "learning_rate": 0.0006,
        "weight_decay": 0.0001,
        "effective_batch_size": 4096,
        "micro_batch_size": 512,
        "amp": True,
        "patience": epochs + 1 if full_fit else 10,
        "scheduler": "constant" if full_fit else "plateau",
    }


def _default_fit(**kwargs: object) -> Mapping[str, object]:
    job = kwargs["job"]
    prepared = kwargs["prepared"]
    output_dir = Path(kwargs["output_dir"])
    identity = kwargs["identity"]
    final_epochs = kwargs.get("final_epochs")
    if not isinstance(job, HierarchicalJob):
        raise HierarchicalTrainingError("fit job is invalid")
    full_fit = job.kind == "full_fit"
    epochs = int(final_epochs) if full_fit else 40
    valid_batch = prepared.train if full_fit else prepared.valid
    request = TrainRequest(
        candidate_id=job.job_id,
        family="tabm",
        seed=3407,
        epochs=epochs,
        model_config=_model_config(),
        training_config=_training_config(epochs=epochs, full_fit=full_fit),
        train=prepared.train,
        valid=valid_batch,
        min_epochs=epochs if full_fit else 3,
        checkpoint_binding=identity,
        model_metadata=prepared.metadata,
    )
    previous_deadline = os.environ.get("PREPROCESSING_SESSION_DEADLINE_UNIX")
    os.environ["PREPROCESSING_SESSION_DEADLINE_UNIX"] = str(kwargs["absolute_deadline"])
    try:
        result = fit_candidate(request, TabMAdapter("bce"), output_dir)
    finally:
        if previous_deadline is None:
            os.environ.pop("PREPROCESSING_SESSION_DEADLINE_UNIX", None)
        else:
            os.environ["PREPROCESSING_SESSION_DEADLINE_UNIX"] = previous_deadline
    if full_fit:
        final_model = output_dir / "final_checkpoint.pt"
        shutil.copyfile(result.checkpoint, final_model)
        return {
            "status": "incomplete" if result.budget_reached else "completed",
            "completed_epochs": result.completed_epochs,
            "checkpoint_path": result.checkpoint,
            "final_model_path": final_model,
        }
    return {
        "status": "incomplete" if result.budget_reached else "completed",
        "best_epoch": result.best_epoch,
        "best_brier": result.best_brier,
        "completed_epochs": result.completed_epochs,
        "probability": result.predictions,
        "checkpoint_path": result.checkpoint,
    }


def _prediction_frame(
    fit_rows: pd.DataFrame, valid_rows: pd.DataFrame, probability: np.ndarray
) -> pd.DataFrame:
    if probability.shape != (len(valid_rows),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise HierarchicalTrainingError("validation probabilities are invalid")
    target = pd.to_numeric(valid_rows["control_success"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise HierarchicalTrainingError("validation target is invalid")
    segments = build_segment_columns(
        valid_rows,
        fit_pitcher_ids=set(fit_rows["pitcher_id"].astype(str)),
        fit_batter_ids=set(fit_rows["batter_id"].astype(str)),
    )
    output = pd.DataFrame(
        {
            "row_id": valid_rows["row_id"].astype(str).to_numpy(),
            "target": target.astype("int64"),
            "probability": probability,
            "season": pd.to_numeric(valid_rows["season"]).astype("int64").to_numpy(),
            "game_month": pd.to_numeric(valid_rows["game_month"]).astype("int64").to_numpy(),
        },
        index=valid_rows.index,
    )
    for column in (
        "game_type", "count_state", "hand_matchup", "base_out_state",
        "pitcher_known", "batter_known",
    ):
        output[column] = segments[column].to_numpy()
    output = output.reset_index(drop=True)
    if tuple(output.columns) != PREDICTION_COLUMNS:
        raise HierarchicalTrainingError("prediction schema differs")
    return output


def training_result_payload(result: TrainingJobResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "job_id": result.job_id,
        "kind": result.kind,
        "status": result.status,
        "train_rows": result.train_rows,
        "valid_rows": result.valid_rows,
        "best_epoch": result.best_epoch,
        "best_brier": result.best_brier,
        "completed_epochs": result.completed_epochs,
        "predictions": None if result.predictions_path is None else result.predictions_path.name,
        "checkpoint": None if result.checkpoint_path is None else result.checkpoint_path.name,
        "final_model": None if result.final_model_path is None else result.final_model_path.name,
        "feature_state": None if result.feature_state_path is None else result.feature_state_path.name,
        "failure": result.failure,
    }
    for key, path in (
        ("predictions_sha256", result.predictions_path),
        ("checkpoint_sha256", result.checkpoint_path),
        ("final_model_sha256", result.final_model_path),
        ("feature_state_sha256", result.feature_state_path),
    ):
        payload[key] = None if path is None else _file_sha256(path)
    return payload


def run_training_job(
    job: HierarchicalJob,
    *,
    data_dir: Path,
    output_dir: Path,
    selected_k: float,
    absolute_deadline: float,
    identity: Mapping[str, str],
    final_epochs: int | None = None,
    resume_checkpoint: Path | None = None,
    on_epoch_checkpoint: Callable[[Path, int], None] | None = None,
    fit: Callable = _default_fit,
) -> TrainingJobResult:
    if job.kind not in {"oof", "full_fit"}:
        raise HierarchicalTrainingError("job kind differs")
    if not math.isfinite(float(absolute_deadline)) or absolute_deadline <= 0:
        raise HierarchicalTrainingError("absolute deadline is invalid")
    identity_value = _sha_mapping(identity)
    if isinstance(selected_k, bool) or not isinstance(selected_k, (int, float)) or not math.isfinite(float(selected_k)) or selected_k <= 0:
        raise HierarchicalTrainingError("selected_k is invalid")
    if job.kind == "full_fit":
        if isinstance(final_epochs, bool) or not isinstance(final_epochs, int) or not 2 <= final_epochs <= 8:
            raise HierarchicalTrainingError("final_epochs must be in [2, 8]")
    elif final_epochs is not None:
        raise HierarchicalTrainingError("OOF job cannot set final_epochs")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise HierarchicalTrainingError("output directory already exists")
    output.mkdir(parents=True)
    resume_epochs = 0
    if resume_checkpoint is not None:
        resume_epochs = _validate_resume_checkpoint(Path(resume_checkpoint), job, identity_value)
    frame = _load_train(Path(data_dir))
    fit_rows = frame.loc[frame["season"] <= job.train_end_year].copy()
    valid_rows = (
        None if job.kind == "full_fit"
        else frame.loc[frame["season"] == job.valid_year].copy()
    )
    if fit_rows.empty or (valid_rows is not None and valid_rows.empty):
        raise HierarchicalTrainingError("fold split is empty")
    context = fit_context_state(fit_rows, smoothing_k=float(selected_k))
    preparation_valid = fit_rows.copy() if valid_rows is None else valid_rows
    prepared_fold = prepare_fold(
        fit_rows, preparation_valid, context_state=context,
        pitcher_k=100.0, batter_k=250.0,
    )
    prepared = (
        SimpleNamespace(
            train=prepared_fold.train, valid=None, state=prepared_fold.state,
            metadata=prepared_fold.metadata,
        )
        if valid_rows is None else prepared_fold
    )
    feature_state_path = output / "feature_state.json"
    _atomic_json(feature_state_path, feature_state_payload(prepared.state))
    if time.time() >= absolute_deadline:
        result = TrainingJobResult(
            job.job_id, job.kind, "incomplete", len(fit_rows),
            None if valid_rows is None else len(valid_rows), None, None,
            resume_epochs, None, Path(resume_checkpoint) if resume_checkpoint else None,
            None, feature_state_path, "deadline_reached_before_training",
        )
        _atomic_json(output / "worker_result.json", training_result_payload(result))
        return result
    if resume_checkpoint is not None and fit is _default_fit:
        shutil.copyfile(resume_checkpoint, output / "checkpoint.pt")
        old_meta = Path(resume_checkpoint).parent / "checkpoint_meta.json"
        if old_meta.is_file():
            shutil.copyfile(old_meta, output / "checkpoint_meta.json")
    raw = fit(
        job=job,
        fit_rows=fit_rows,
        valid_rows=valid_rows,
        prepared=prepared,
        context_state=context,
        output_dir=output,
        identity=identity_value,
        selected_k=float(selected_k),
        absolute_deadline=absolute_deadline,
        final_epochs=final_epochs,
        resume_checkpoint=resume_checkpoint,
        on_epoch_checkpoint=on_epoch_checkpoint,
    )
    if not isinstance(raw, Mapping) or raw.get("status") not in {"completed", "incomplete"}:
        raise HierarchicalTrainingError("fit result status differs")
    checkpoint = raw.get("checkpoint_path")
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    if checkpoint_path is not None and (checkpoint_path.is_symlink() or not checkpoint_path.is_file()):
        raise HierarchicalTrainingError("fit checkpoint is missing")
    completed_epochs = raw.get("completed_epochs")
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int) or completed_epochs < 0:
        raise HierarchicalTrainingError("completed_epochs is invalid")
    predictions_path = None
    best_epoch = None
    best_brier = None
    final_model_path = None
    if job.kind == "oof":
        assert valid_rows is not None
        probability = np.asarray(raw.get("probability"), dtype="float64")
        prediction_frame = _prediction_frame(fit_rows, valid_rows, probability)
        predictions_path = output / "predictions.csv"
        prediction_frame.to_csv(predictions_path, index=False)
        best_epoch_raw = raw.get("best_epoch")
        best_brier_raw = raw.get("best_brier")
        if isinstance(best_epoch_raw, bool) or not isinstance(best_epoch_raw, int) or best_epoch_raw < 0:
            raise HierarchicalTrainingError("best_epoch is invalid")
        if isinstance(best_brier_raw, bool) or not isinstance(best_brier_raw, (int, float)) or not math.isfinite(float(best_brier_raw)):
            raise HierarchicalTrainingError("best_brier is invalid")
        recomputed = float(np.mean(np.square(prediction_frame["target"] - probability)))
        if not math.isclose(float(best_brier_raw), recomputed, rel_tol=0, abs_tol=1e-12):
            raise HierarchicalTrainingError("best_brier differs from predictions")
        best_epoch = best_epoch_raw
        best_brier = float(best_brier_raw)
    else:
        model = raw.get("final_model_path")
        final_model_path = Path(model) if model is not None else None
        if raw["status"] == "completed" and (
            final_model_path is None or final_model_path.is_symlink() or not final_model_path.is_file()
        ):
            raise HierarchicalTrainingError("completed full fit model is missing")
    result = TrainingJobResult(
        job.job_id, job.kind, str(raw["status"]), len(fit_rows),
        None if valid_rows is None else len(valid_rows), best_epoch, best_brier,
        completed_epochs, predictions_path, checkpoint_path, final_model_path,
        feature_state_path, None if raw["status"] == "completed" else "deadline_reached",
    )
    if checkpoint_path is not None:
        _atomic_json(
            _resume_meta_path(checkpoint_path),
            {
                "schema_version": 1,
                "job_id": job.job_id,
                "identity": dict(identity_value),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "completed_epochs": completed_epochs,
            },
        )
    _atomic_json(output / "worker_result.json", training_result_payload(result))
    return result


def _job_from_json(path: Path) -> HierarchicalJob:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return HierarchicalJob(**payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-k", type=float, required=True)
    parser.add_argument("--absolute-deadline", type=float, required=True)
    parser.add_argument("--identity", type=str, required=True)
    parser.add_argument("--final-epochs", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    result = run_training_job(
        _job_from_json(args.job), data_dir=args.data_dir, output_dir=args.output_dir,
        selected_k=args.selected_k, absolute_deadline=args.absolute_deadline,
        identity=json.loads(args.identity), final_epochs=args.final_epochs,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(_canonical_json(training_result_payload(result)).decode(), flush=True)
    return 0 if result.status in {"completed", "incomplete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
