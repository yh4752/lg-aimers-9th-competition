from __future__ import annotations

from dataclasses import dataclass
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

from experiments.catboost_deployment.state import serialize_feature_state
from experiments.catboost_preprocessing.features import (
    fit_catboost_features,
    transform_catboost_features,
)

from .contracts import contract_sha256, load_contract
from .metrics import (
    CATBOOST_COLUMNS,
    RealignDecision,
    decision_to_payload,
    passes_gates,
    select_candidate,
)


class RealignTrainingError(RuntimeError):
    """Raised when a sealed CatBoost job cannot run safely."""


@dataclass(frozen=True)
class CatBoostJobResult:
    job_id: str
    status: str
    model_path: Path | None
    preprocessing_path: Path | None
    predictions_path: Path | None
    snapshot_path: Path | None


_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RealignTrainingError(f"temporary output already exists: {temporary.name}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _canonical_json(payload))


def _default_factory(**parameters):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**parameters)


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
                    f"REALIGN_TRAINING_PROGRESS job={self.job_id} "
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


def _sha_or_default(value: str | None, default: str, label: str) -> str:
    result = default if value is None else value
    if type(result) is not str or _SHA_RE.fullmatch(result) is None:
        raise RealignTrainingError(f"{label} SHA-256 is invalid")
    return result


def _load_train(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / "train.csv"
    if path.is_symlink() or not path.is_file():
        raise RealignTrainingError("official train.csv is missing")
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise RealignTrainingError(f"official train.csv cannot be read: {error}") from error
    required = {
        "row_id",
        "season",
        "control_success",
        "game_type",
        "game_month",
        "pitcher_id",
        "batter_id",
    }
    if not required.issubset(frame.columns) or frame.empty:
        raise RealignTrainingError("official train.csv schema differs")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise RealignTrainingError("official row_id must be unique")
    return frame


def _target(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["control_success"], errors="coerce").to_numpy(
        "float64"
    )
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise RealignTrainingError("target must be finite and binary")
    return values


def _known(frame: pd.DataFrame, state, column: str) -> np.ndarray:
    return np.where(
        frame[column].astype(str).isin(state.category_values[column]), "known", "oov"
    )


def _prepare_identity(path: Path, payload: dict[str, object]) -> None:
    if path.is_symlink():
        raise RealignTrainingError("existing job identity is not a regular file")
    if path.exists() or path.is_symlink():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RealignTrainingError("existing job identity is unreadable") from error
        if existing != payload:
            raise RealignTrainingError("existing job identity differs")
        return
    _atomic_json(path, payload)


def _run(
    *,
    kind: str,
    data_dir: Path,
    output_dir: Path,
    absolute_deadline: float,
    decision: RealignDecision | None,
    input_manifest_sha256: str | None,
    code_sha256: str | None,
    model_factory: Callable[..., object] | None,
) -> CatBoostJobResult:
    if not math.isfinite(absolute_deadline):
        raise RealignTrainingError("absolute deadline must be finite")
    contract = load_contract()
    selected_tree_count: int | None = None
    decision_sha: str | None = None
    if kind == "full_fit":
        selected_evidence = (
            ()
            if decision is None
            else tuple(
                item
                for item in decision.candidates
                if item.tree_count == decision.selected_tree_count
            )
        )
        if (
            decision is None
            or decision.status != "promoted"
            or decision.selected_tree_count not in contract.tree_prefixes
            or decision.reason != "selected_by_preregistered_order"
            or len(selected_evidence) != 1
            or not selected_evidence[0].passed
            or not passes_gates(selected_evidence[0], contract)
        ):
            raise RealignTrainingError("promoted decision is required for full fit")
        try:
            selected = select_candidate(decision.candidates, contract)
        except Exception as error:
            raise RealignTrainingError("promoted decision is required for full fit") from error
        if selected.tree_count != decision.selected_tree_count:
            raise RealignTrainingError("promoted decision is required for full fit")
        selected_tree_count = decision.selected_tree_count
        decision_sha = sha256(_canonical_json(decision_to_payload(decision))).hexdigest()
    elif kind != "alignment":
        raise RealignTrainingError("unknown CatBoost job kind")

    output = Path(output_dir)
    if output.is_symlink():
        raise RealignTrainingError("training output directory is a symlink")
    output.mkdir(parents=True, exist_ok=True)
    train_path = Path(data_dir) / "train.csv"
    default_input_sha = _file_sha(train_path) if train_path.is_file() else "0" * 64
    identity = {
        "schema_version": 1,
        "campaign_id": contract.campaign_id,
        "job_id": "catboost_f1_2022" if kind == "alignment" else "catboost_full_2024",
        "kind": kind,
        "contract_sha256": contract_sha256(),
        "input_manifest_sha256": _sha_or_default(
            input_manifest_sha256, default_input_sha, "input manifest"
        ),
        "code_sha256": _sha_or_default(
            code_sha256, _file_sha(Path(__file__)), "code"
        ),
        "decision_sha256": decision_sha,
        "selected_tree_count": selected_tree_count,
    }
    _prepare_identity(output / "job.json", identity)
    snapshot = output / "experiment.cbsnapshot"
    if snapshot.is_symlink() or (snapshot.exists() and not snapshot.is_file()):
        raise RealignTrainingError("snapshot path is invalid")
    if time.time() >= absolute_deadline:
        return CatBoostJobResult(
            job_id=identity["job_id"],
            status="budget_inconclusive",
            model_path=None,
            preprocessing_path=None,
            predictions_path=None,
            snapshot_path=snapshot if snapshot.is_file() else None,
        )
    for name in ("model.cbm", "predictions.csv", "metrics.json"):
        path = output / name
        if path.exists() or path.is_symlink():
            raise RealignTrainingError("completed training output already exists")

    full = _load_train(Path(data_dir))
    seasons = pd.to_numeric(full["season"], errors="raise").astype("int64")
    if kind == "alignment":
        train = full.loc[seasons.le(2021)].reset_index(drop=True)
        valid = full.loc[seasons.eq(2022)].reset_index(drop=True)
        iterations = 400
        if train.empty or valid.empty:
            raise RealignTrainingError("2021->2022 alignment fold is empty")
    else:
        train = full.reset_index(drop=True)
        valid = None
        iterations = selected_tree_count
    if type(iterations) is not int or (
        kind == "full_fit" and iterations not in contract.tree_prefixes
    ):
        raise RealignTrainingError("selected tree count is invalid")

    y_train = _target(train)
    state, x_train = fit_catboost_features(train, components=("hand_matchup",))
    preprocessing_path = output / "preprocessing_state.json"
    preprocessing_payload = serialize_feature_state(state)
    if preprocessing_path.exists() or preprocessing_path.is_symlink():
        if (
            preprocessing_path.is_symlink()
            or not preprocessing_path.is_file()
            or preprocessing_path.read_bytes() != preprocessing_payload
        ):
            raise RealignTrainingError("existing preprocessing state differs")
    else:
        _atomic_bytes(preprocessing_path, preprocessing_payload)
    x_valid = None if valid is None else transform_catboost_features(valid, state)
    y_valid = None if valid is None else _target(valid)

    parameters = {
        **contract.catboost_parameters,
        "iterations": iterations,
        "allow_writing_files": True,
        "train_dir": str(output / "catboost_info"),
    }
    model = (model_factory or _default_factory)(**parameters)
    (output / "catboost_info").mkdir(exist_ok=True)
    with (output / "worker.log").open("a", encoding="utf-8") as log:
        writer = _ProgressWriter(str(identity["job_id"]), log)
        fit_arguments: dict[str, object] = {
            "cat_features": list(state.categorical_columns),
            "use_best_model": False,
            "save_snapshot": True,
            "snapshot_file": str(snapshot),
            "snapshot_interval": contract.snapshot_interval_seconds,
            "verbose": 4,
            "log_cout": writer,
            "log_cerr": writer,
        }
        if x_valid is not None and y_valid is not None:
            fit_arguments["eval_set"] = (x_valid, y_valid)
        model.fit(x_train, y_train, **fit_arguments)

    temporary_model = output / "model.tmp.cbm"
    model_path = output / "model.cbm"
    try:
        model.save_model(str(temporary_model))
        if not temporary_model.is_file() or temporary_model.stat().st_size <= 0:
            raise RealignTrainingError("CatBoost model was not written")
        os.replace(temporary_model, model_path)
    except Exception:
        temporary_model.unlink(missing_ok=True)
        raise

    predictions_path: Path | None = None
    metrics: dict[str, object] = {
        "schema_version": 1,
        "job_id": identity["job_id"],
        "kind": kind,
        "train_rows": len(train),
        "valid_rows": None if valid is None else len(valid),
        "selected_tree_count": iterations,
        "model_sha256": _file_sha(model_path),
        "preprocessing_sha256": _file_sha(preprocessing_path),
        "snapshot_sha256": _file_sha(snapshot) if snapshot.is_file() else None,
    }
    if valid is not None and x_valid is not None and y_valid is not None:
        predictions: dict[str, object] = {
            "row_id": valid["row_id"].astype(str),
            "target": y_valid.astype("int8"),
        }
        prefix_brier: dict[str, float] = {}
        for prefix in contract.tree_prefixes:
            raw = np.asarray(model.predict(x_valid, ntree_end=prefix), dtype="float64").reshape(-1)
            if len(raw) != len(valid) or not np.isfinite(raw).all():
                raise RealignTrainingError("CatBoost prefix prediction is invalid")
            probability = np.clip(raw, 0.0, 1.0)
            predictions[f"p_{prefix}"] = probability
            prefix_brier[str(prefix)] = float(
                np.mean(np.square(probability - y_valid), dtype=np.float64)
            )
        predictions.update(
            {
                "game_type": valid["game_type"].astype(str),
                "game_month": valid["game_month"],
                "pitcher_id_known": _known(x_valid, state, "pitcher_id"),
                "batter_id_known": _known(x_valid, state, "batter_id"),
            }
        )
        frame = pd.DataFrame(predictions).loc[:, CATBOOST_COLUMNS]
        predictions_path = output / "predictions.csv"
        temporary_predictions = output / "predictions.csv.tmp"
        try:
            frame.to_csv(temporary_predictions, index=False)
            os.replace(temporary_predictions, predictions_path)
        except Exception:
            temporary_predictions.unlink(missing_ok=True)
            raise
        metrics["prefix_brier"] = prefix_brier
        metrics["predictions_sha256"] = _file_sha(predictions_path)
    else:
        metrics["decision_sha256"] = decision_sha
    _atomic_json(output / "metrics.json", metrics)
    return CatBoostJobResult(
        job_id=str(identity["job_id"]),
        status="completed",
        model_path=model_path,
        preprocessing_path=preprocessing_path,
        predictions_path=predictions_path,
        snapshot_path=snapshot if snapshot.is_file() else None,
    )


def run_catboost_f1(
    *,
    data_dir: Path,
    output_dir: Path,
    absolute_deadline: float,
    input_manifest_sha256: str | None = None,
    code_sha256: str | None = None,
    model_factory: Callable[..., object] | None = None,
) -> CatBoostJobResult:
    return _run(
        kind="alignment",
        data_dir=data_dir,
        output_dir=output_dir,
        absolute_deadline=absolute_deadline,
        decision=None,
        input_manifest_sha256=input_manifest_sha256,
        code_sha256=code_sha256,
        model_factory=model_factory,
    )


def run_full_fit(
    *,
    data_dir: Path,
    output_dir: Path,
    decision: RealignDecision,
    absolute_deadline: float,
    input_manifest_sha256: str | None = None,
    code_sha256: str | None = None,
    model_factory: Callable[..., object] | None = None,
) -> CatBoostJobResult:
    return _run(
        kind="full_fit",
        data_dir=data_dir,
        output_dir=output_dir,
        absolute_deadline=absolute_deadline,
        decision=decision,
        input_manifest_sha256=input_manifest_sha256,
        code_sha256=code_sha256,
        model_factory=model_factory,
    )
