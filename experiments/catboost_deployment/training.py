from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np
import pandas as pd

from experiments.catboost_preprocessing.features import (
    fit_catboost_features,
    transform_catboost_features,
)

from .contracts import DeploymentContract, DeploymentJob, build_jobs, load_contract
from .metrics import CATBOOST_PREDICTION_COLUMNS, SEGMENT_COLUMNS
from .state import serialize_feature_state


class DeploymentTrainingError(RuntimeError):
    """Raised when a sealed CatBoost deployment job cannot run safely."""


@dataclass(frozen=True)
class DeploymentJobResult:
    job_id: str
    kind: str
    status: str
    train_rows: int
    valid_rows: int | None
    predictions_path: Path | None
    model_path: Path | None
    preprocessing_path: Path | None
    snapshot_path: Path | None
    elapsed_seconds: float
    failure: str | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _canonical_json(payload))


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
                    f"CATBOOST_DEPLOY_PROGRESS job={self.job_id} "
                    f"iteration={int(match.group(1))} "
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


def _result_payload(result: DeploymentJobResult) -> dict[str, object]:
    return {
        "job_id": result.job_id,
        "kind": result.kind,
        "status": result.status,
        "train_rows": result.train_rows,
        "valid_rows": result.valid_rows,
        "predictions": None if result.predictions_path is None else result.predictions_path.name,
        "model": None if result.model_path is None else result.model_path.name,
        "preprocessing": (
            None if result.preprocessing_path is None else result.preprocessing_path.name
        ),
        "snapshot": None if result.snapshot_path is None else result.snapshot_path.name,
        "elapsed_seconds": result.elapsed_seconds,
        "failure": result.failure,
    }


def _job_payload(
    job: DeploymentJob,
    *,
    contract_sha256: str,
    input_manifest_sha256: str,
    code_sha256: str,
    selected_tree_count: int | None,
    alignment_decision_sha256: str | None,
) -> dict[str, object]:
    return {
        **asdict(job),
        "contract_sha256": contract_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "code_sha256": code_sha256,
        "selected_tree_count": selected_tree_count,
        "alignment_decision_sha256": alignment_decision_sha256,
    }


def _validate_target(frame: pd.DataFrame) -> np.ndarray:
    target = pd.to_numeric(frame["control_success"], errors="raise").to_numpy(
        dtype="float64"
    )
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise DeploymentTrainingError("target must be finite and binary")
    return target


def _known_labels(frame: pd.DataFrame, state, column: str) -> np.ndarray:
    return np.where(
        frame[column].astype(str).isin(state.category_values[column]), "known", "oov"
    )


def _prepare_job_identity(output_dir: Path, expected: dict[str, object]) -> None:
    path = output_dir / "job.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise DeploymentTrainingError("existing job identity is unreadable") from error
        if existing != expected:
            raise DeploymentTrainingError("existing job identity differs")
    else:
        _atomic_json(path, expected)


def _failed_result(
    job: DeploymentJob,
    started: float,
    snapshot: Path,
    error: Exception,
    *,
    train_rows: int = 0,
    valid_rows: int | None = None,
) -> DeploymentJobResult:
    return DeploymentJobResult(
        job_id=job.job_id,
        kind=job.kind,
        status="failed",
        train_rows=train_rows,
        valid_rows=valid_rows,
        predictions_path=None,
        model_path=None,
        preprocessing_path=None,
        snapshot_path=snapshot if snapshot.is_file() else None,
        elapsed_seconds=time.monotonic() - started,
        failure=f"{type(error).__name__}: {error}",
    )


def _run_job(
    *,
    job: DeploymentJob,
    contract: DeploymentContract,
    data_dir: Path,
    output_dir: Path,
    contract_sha256: str,
    input_manifest_sha256: str,
    code_sha256: str,
    absolute_deadline: float,
    selected_tree_count: int | None,
    alignment_decision_sha256: str | None,
    model_factory: Callable[..., object] | None,
) -> DeploymentJobResult:
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "experiment.cbsnapshot"
    train_rows = 0
    valid_rows: int | None = None
    try:
        expected_job = _job_payload(
            job,
            contract_sha256=contract_sha256,
            input_manifest_sha256=input_manifest_sha256,
            code_sha256=code_sha256,
            selected_tree_count=selected_tree_count,
            alignment_decision_sha256=alignment_decision_sha256,
        )
        _prepare_job_identity(output_dir, expected_job)
        if time.time() >= absolute_deadline:
            result = DeploymentJobResult(
                job_id=job.job_id,
                kind=job.kind,
                status="budget_inconclusive",
                train_rows=0,
                valid_rows=None if job.kind == "full_fit" else 0,
                predictions_path=None,
                model_path=None,
                preprocessing_path=None,
                snapshot_path=snapshot if snapshot.is_file() else None,
                elapsed_seconds=time.monotonic() - started,
                failure="absolute deadline expired before training",
            )
            _atomic_json(output_dir / "worker_result.json", _result_payload(result))
            return result

        full = pd.read_csv(Path(data_dir) / "train.csv")
        required = {
            "row_id",
            "season",
            "control_success",
            "game_type",
            "game_month",
            "pitcher_id",
            "batter_id",
        }
        if not required <= set(full):
            raise DeploymentTrainingError("train.csv schema is missing required columns")
        seasons = pd.to_numeric(full["season"], errors="raise")
        train = full.loc[seasons.le(job.train_end_year)].reset_index(drop=True)
        if job.kind == "alignment":
            if job.valid_year is None:
                raise DeploymentTrainingError("alignment job is missing validation year")
            valid = full.loc[seasons.eq(job.valid_year)].reset_index(drop=True)
            valid_rows = len(valid)
            if train.empty or valid.empty:
                raise DeploymentTrainingError("alignment folds must be nonempty")
        elif job.kind == "full_fit":
            if job.valid_year is not None or len(train) != len(full):
                raise DeploymentTrainingError("full fit must use every official training row")
            valid = None
        else:
            raise DeploymentTrainingError("unknown deployment job kind")
        train_rows = len(train)
        if train["row_id"].isna().any() or train["row_id"].astype(str).duplicated().any():
            raise DeploymentTrainingError("training row_id must be unique")
        if valid is not None and (
            valid["row_id"].isna().any() or valid["row_id"].astype(str).duplicated().any()
        ):
            raise DeploymentTrainingError("validation row_id must be unique")

        y_train = _validate_target(train)
        state, x_train = fit_catboost_features(train, components=("hand_matchup",))
        preprocessing_path = output_dir / "preprocessing_state.json"
        _atomic_bytes(preprocessing_path, serialize_feature_state(state))
        x_valid = None if valid is None else transform_catboost_features(valid, state)
        target = None if valid is None else _validate_target(valid)

        iterations = 400 if job.kind == "alignment" else selected_tree_count
        if type(iterations) is not int or iterations not in contract.tree_prefixes:
            raise DeploymentTrainingError("selected tree count is invalid")
        catboost_info = output_dir / "catboost_info"
        catboost_info.mkdir(exist_ok=True)
        parameters = {
            **contract.catboost_parameters,
            "iterations": iterations,
            "allow_writing_files": True,
            "train_dir": str(catboost_info),
        }
        model = (model_factory or _default_factory)(**parameters)
        log_path = output_dir / "worker.log"
        with log_path.open("a", encoding="utf-8") as log:
            writer = _ProgressWriter(job.job_id, log)
            fit_arguments: dict[str, object] = {
                "cat_features": list(state.categorical_columns),
                "use_best_model": False,
                "save_snapshot": True,
                "snapshot_file": str(snapshot),
                "snapshot_interval": contract.snapshot_interval_seconds,
                "verbose": 50,
                "log_cout": writer,
                "log_cerr": writer,
            }
            if x_valid is not None and target is not None:
                fit_arguments["eval_set"] = (x_valid, target)
            model.fit(x_train, y_train, **fit_arguments)

        model_temporary = output_dir / "model.tmp.cbm"
        model_path = output_dir / "model.cbm"
        model.save_model(str(model_temporary))
        os.replace(model_temporary, model_path)
        predictions_path: Path | None = None
        metrics: dict[str, object] = {
            "job_id": job.job_id,
            "kind": job.kind,
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "selected_tree_count": iterations,
            "model_sha256": _file_sha(model_path),
            "preprocessing_sha256": _file_sha(preprocessing_path),
            "snapshot_sha256": _file_sha(snapshot) if snapshot.is_file() else None,
        }
        if valid is not None and x_valid is not None and target is not None:
            predictions: dict[str, object] = {
                "row_id": valid["row_id"].astype(str),
                "target": target.astype(int),
            }
            prefix_brier: dict[str, float] = {}
            for prefix in contract.tree_prefixes:
                raw = np.asarray(model.predict(x_valid, ntree_end=prefix), dtype="float64").reshape(-1)
                if len(raw) != len(valid) or not np.isfinite(raw).all():
                    raise DeploymentTrainingError("model predictions are invalid")
                probability = np.clip(raw, 0.0, 1.0)
                predictions[f"p_{prefix}"] = probability
                brier = float(np.mean(np.square(probability - target), dtype=np.float64))
                if not math.isfinite(brier):
                    raise DeploymentTrainingError("prefix Brier is not finite")
                prefix_brier[str(prefix)] = brier
            predictions.update(
                {
                    "game_type": valid["game_type"].astype(str),
                    "game_month": valid["game_month"],
                    "pitcher_id_known": _known_labels(x_valid, state, "pitcher_id"),
                    "batter_id_known": _known_labels(x_valid, state, "batter_id"),
                }
            )
            prediction_frame = pd.DataFrame(predictions).loc[:, CATBOOST_PREDICTION_COLUMNS]
            predictions_path = output_dir / "predictions.csv"
            temporary_predictions = output_dir / "predictions.csv.tmp"
            prediction_frame.to_csv(temporary_predictions, index=False)
            os.replace(temporary_predictions, predictions_path)
            metrics["prefix_brier"] = prefix_brier
            metrics["predictions_sha256"] = _file_sha(predictions_path)
        else:
            metrics["alignment_decision_sha256"] = alignment_decision_sha256
        _atomic_json(output_dir / "metrics.json", metrics)
        result = DeploymentJobResult(
            job_id=job.job_id,
            kind=job.kind,
            status="completed",
            train_rows=train_rows,
            valid_rows=valid_rows,
            predictions_path=predictions_path,
            model_path=model_path,
            preprocessing_path=preprocessing_path,
            snapshot_path=snapshot if snapshot.is_file() else None,
            elapsed_seconds=time.monotonic() - started,
            failure=None,
        )
    except Exception as error:
        result = _failed_result(
            job,
            started,
            snapshot,
            error,
            train_rows=train_rows,
            valid_rows=valid_rows,
        )
    _atomic_json(output_dir / "worker_result.json", _result_payload(result))
    return result


def run_alignment_job(
    *,
    job: DeploymentJob,
    contract: DeploymentContract,
    data_dir: Path,
    output_dir: Path,
    contract_sha256: str,
    input_manifest_sha256: str,
    code_sha256: str,
    absolute_deadline: float,
    model_factory: Callable[..., object] | None = None,
) -> DeploymentJobResult:
    return _run_job(
        job=job,
        contract=contract,
        data_dir=data_dir,
        output_dir=output_dir,
        contract_sha256=contract_sha256,
        input_manifest_sha256=input_manifest_sha256,
        code_sha256=code_sha256,
        absolute_deadline=absolute_deadline,
        selected_tree_count=None,
        alignment_decision_sha256=None,
        model_factory=model_factory,
    )


def run_full_fit_job(
    *,
    job: DeploymentJob,
    contract: DeploymentContract,
    selected_tree_count: int,
    alignment_decision_sha256: str,
    data_dir: Path,
    output_dir: Path,
    contract_sha256: str,
    input_manifest_sha256: str,
    code_sha256: str,
    absolute_deadline: float,
    model_factory: Callable[..., object] | None = None,
) -> DeploymentJobResult:
    return _run_job(
        job=job,
        contract=contract,
        data_dir=data_dir,
        output_dir=output_dir,
        contract_sha256=contract_sha256,
        input_manifest_sha256=input_manifest_sha256,
        code_sha256=code_sha256,
        absolute_deadline=absolute_deadline,
        selected_tree_count=selected_tree_count,
        alignment_decision_sha256=alignment_decision_sha256,
        model_factory=model_factory,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one sealed CatBoost deployment job.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--absolute-deadline", type=float, required=True)
    parser.add_argument("--selected-tree-count", type=int)
    parser.add_argument("--alignment-decision-sha256")
    args = parser.parse_args(argv)
    contract = load_contract()
    jobs = {job.job_id: job for job in build_jobs(contract)}
    if args.job_id not in jobs:
        raise DeploymentTrainingError("unknown job identifier")
    job = jobs[args.job_id]
    common = dict(
        job=job,
        contract=contract,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        contract_sha256=args.contract_sha256,
        input_manifest_sha256=args.input_manifest_sha256,
        code_sha256=args.code_sha256,
        absolute_deadline=args.absolute_deadline,
    )
    if job.kind == "alignment":
        result = run_alignment_job(**common)
    else:
        if args.selected_tree_count is None or args.alignment_decision_sha256 is None:
            raise DeploymentTrainingError("full fit arguments are missing")
        result = run_full_fit_job(
            **common,
            selected_tree_count=args.selected_tree_count,
            alignment_decision_sha256=args.alignment_decision_sha256,
        )
    print(f"DEPLOY_JOB_END job={result.job_id} status={result.status}", flush=True)
    return 0 if result.status in {"completed", "budget_inconclusive"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
