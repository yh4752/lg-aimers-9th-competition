from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd

from experiments.catboost_preprocessing.features import fit_catboost_features, transform_catboost_features

from .contracts import BlendContract, BlendJob, build_jobs, load_contract


class BlendTrainingError(RuntimeError):
    """Raised when one sealed CatBoost fold cannot run safely."""


@dataclass(frozen=True)
class FoldResult:
    job_id: str
    status: str
    train_rows: int
    valid_rows: int
    brier: float | None
    model_path: Path | None
    predictions_path: Path | None
    snapshot_path: Path | None
    elapsed_seconds: float
    failure: str | None


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _ProgressWriter:
    _ITERATION = re.compile(r"^\s*(\d+):")

    def __init__(self, job_id: str, stream) -> None:
        self.job_id = job_id
        self.stream = stream
        self.started = time.monotonic()

    def write(self, value: str) -> int:
        for line in value.splitlines():
            match = self._ITERATION.match(line)
            if match:
                marker = (
                    f"CATBOOST_PROGRESS job={self.job_id} iteration={int(match.group(1))} "
                    f"elapsed_seconds={time.monotonic() - self.started:.1f}"
                )
                print(marker, flush=True)
                self.stream.write(marker + "\n")
            elif line.strip():
                self.stream.write(line + "\n")
        self.stream.flush()
        return len(value)

    def flush(self) -> None:
        self.stream.flush()


def _default_factory(**parameters):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**parameters)


def _job_payload(
    job: BlendJob,
    contract_sha256: str,
    input_manifest_sha256: str,
    code_sha256: str,
) -> dict[str, object]:
    return {
        **asdict(job),
        "contract_sha256": contract_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "code_sha256": code_sha256,
    }


def _result_payload(result: FoldResult) -> dict[str, object]:
    return {
        "job_id": result.job_id,
        "status": result.status,
        "train_rows": result.train_rows,
        "valid_rows": result.valid_rows,
        "brier": result.brier,
        "model": None if result.model_path is None else result.model_path.name,
        "predictions": (
            None if result.predictions_path is None else result.predictions_path.name
        ),
        "snapshot": None if result.snapshot_path is None else result.snapshot_path.name,
        "elapsed_seconds": result.elapsed_seconds,
        "failure": result.failure,
    }


def run_fold(
    *,
    job: BlendJob,
    contract: BlendContract,
    data_dir: Path,
    output_dir: Path,
    contract_sha256: str,
    input_manifest_sha256: str,
    code_sha256: str,
    absolute_deadline: float,
    model_factory: Callable[..., object] | None = None,
) -> FoldResult:
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_path = output_dir / "job.json"
    expected_job = _job_payload(
        job, contract_sha256, input_manifest_sha256, code_sha256
    )
    if job_path.exists():
        try:
            existing = json.loads(job_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise BlendTrainingError("existing job identity is unreadable") from error
        if existing != expected_job:
            raise BlendTrainingError("existing job identity differs")
    else:
        _atomic_json(job_path, expected_job)

    snapshot_path = output_dir / "experiment.cbsnapshot"
    if time.time() >= absolute_deadline:
        result = FoldResult(
            job_id=job.job_id,
            status="budget_inconclusive",
            train_rows=0,
            valid_rows=0,
            brier=None,
            model_path=None,
            predictions_path=None,
            snapshot_path=snapshot_path if snapshot_path.is_file() else None,
            elapsed_seconds=time.monotonic() - started,
            failure="absolute deadline expired before training",
        )
        _atomic_json(output_dir / "worker_result.json", _result_payload(result))
        return result

    train_rows = 0
    valid_rows = 0
    try:
        full = pd.read_csv(Path(data_dir) / "train.csv")
        required = {"row_id", "season", "control_success", "game_type", "game_month"}
        if not required <= set(full):
            raise BlendTrainingError("train.csv schema is missing required columns")
        seasons = pd.to_numeric(full["season"], errors="raise")
        train = full.loc[seasons.le(job.train_end_year)].reset_index(drop=True)
        valid = full.loc[seasons.eq(job.valid_year)].reset_index(drop=True)
        train_rows, valid_rows = len(train), len(valid)
        if train.empty or valid.empty:
            raise BlendTrainingError("training and validation folds must be nonempty")
        if train["row_id"].astype(str).duplicated().any() or valid["row_id"].astype(str).duplicated().any():
            raise BlendTrainingError("row_id must be unique within each fold")
        y_train = pd.to_numeric(train["control_success"], errors="raise").to_numpy(
            dtype="float64"
        )
        target = pd.to_numeric(valid["control_success"], errors="raise").to_numpy(
            dtype="float64"
        )
        if not np.isin(y_train, (0.0, 1.0)).all() or not np.isin(target, (0.0, 1.0)).all():
            raise BlendTrainingError("target must be binary")
        state, x_train = fit_catboost_features(
            train, components=contract.preprocessing_components
        )
        x_valid = transform_catboost_features(valid, state)
        catboost_info = output_dir / "catboost_info"
        catboost_info.mkdir(exist_ok=True)
        parameters = {
            **contract.catboost_parameters,
            "allow_writing_files": True,
            "train_dir": str(catboost_info),
        }
        model = (model_factory or _default_factory)(**parameters)
        log_path = output_dir / "worker.log"
        with log_path.open("a", encoding="utf-8") as log:
            writer = _ProgressWriter(job.job_id, log)
            model.fit(
                x_train,
                y_train,
                cat_features=list(state.categorical_columns),
                eval_set=(x_valid, target),
                use_best_model=True,
                early_stopping_rounds=50,
                save_snapshot=True,
                snapshot_file=str(snapshot_path),
                snapshot_interval=contract.snapshot_interval_seconds,
                verbose=50,
                log_cout=writer,
                log_cerr=writer,
            )
        raw_probability = np.asarray(model.predict(x_valid), dtype="float64").reshape(-1)
        if len(raw_probability) != len(valid) or not np.isfinite(raw_probability).all():
            raise BlendTrainingError("model predictions are invalid")
        probability = np.clip(raw_probability, 0.0, 1.0)
        model_temporary = output_dir / "model.tmp.cbm"
        model_path = output_dir / "model.cbm"
        model.save_model(str(model_temporary))
        os.replace(model_temporary, model_path)
        predictions = pd.DataFrame(
            {
                "row_id": valid["row_id"].astype(str),
                "target": target.astype(int),
                "probability": probability,
                "game_type": valid["game_type"].astype(str),
                "game_month": valid["game_month"],
                "pitcher_id_known": np.where(
                    x_valid["pitcher_id"].astype(str).isin(
                        state.category_values["pitcher_id"]
                    ),
                    "known",
                    "oov",
                ),
                "batter_id_known": np.where(
                    x_valid["batter_id"].astype(str).isin(
                        state.category_values["batter_id"]
                    ),
                    "known",
                    "oov",
                ),
            }
        )
        predictions_path = output_dir / "predictions.csv"
        predictions_temporary = output_dir / "predictions.csv.tmp"
        predictions.to_csv(predictions_temporary, index=False)
        os.replace(predictions_temporary, predictions_path)
        brier = float(np.mean(np.square(probability - target), dtype=np.float64))
        if not math.isfinite(brier):
            raise BlendTrainingError("Brier is not finite")
        metrics = {
            "job_id": job.job_id,
            "train_end_year": job.train_end_year,
            "valid_year": job.valid_year,
            "seed": job.seed,
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "brier": brier,
            "raw_prediction_min": float(raw_probability.min()),
            "raw_prediction_max": float(raw_probability.max()),
            "model_sha256": _file_sha(model_path),
            "predictions_sha256": _file_sha(predictions_path),
            "snapshot_sha256": _file_sha(snapshot_path) if snapshot_path.is_file() else None,
        }
        _atomic_json(output_dir / "metrics.json", metrics)
        result = FoldResult(
            job_id=job.job_id,
            status="completed",
            train_rows=train_rows,
            valid_rows=valid_rows,
            brier=brier,
            model_path=model_path,
            predictions_path=predictions_path,
            snapshot_path=snapshot_path if snapshot_path.is_file() else None,
            elapsed_seconds=time.monotonic() - started,
            failure=None,
        )
    except Exception as error:
        result = FoldResult(
            job_id=job.job_id,
            status="failed",
            train_rows=train_rows,
            valid_rows=valid_rows,
            brier=None,
            model_path=None,
            predictions_path=None,
            snapshot_path=snapshot_path if snapshot_path.is_file() else None,
            elapsed_seconds=time.monotonic() - started,
            failure=f"{type(error).__name__}: {error}",
        )
    _atomic_json(output_dir / "worker_result.json", _result_payload(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one sealed CatBoost blend fold.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--absolute-deadline", type=float, required=True)
    args = parser.parse_args(argv)
    contract = load_contract()
    jobs = {job.job_id: job for job in build_jobs(contract)}
    if args.job_id not in jobs:
        raise BlendTrainingError("unknown job identifier")
    result = run_fold(
        job=jobs[args.job_id],
        contract=contract,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        contract_sha256=args.contract_sha256,
        input_manifest_sha256=args.input_manifest_sha256,
        code_sha256=args.code_sha256,
        absolute_deadline=args.absolute_deadline,
    )
    print(
        f"CATBOOST_JOB_END job={result.job_id} status={result.status}", flush=True
    )
    return 0 if result.status in {"completed", "budget_inconclusive"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
