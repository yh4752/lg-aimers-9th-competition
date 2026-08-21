from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from experiments.tabm_campaign.row_feature_proxy import _validate_checkpoint_meta
from experiments.tabm_campaign.row_feature_runtime import CampaignJob, CampaignJobResult
from experiments.tabm_campaign.worker import _valid_completed_result, run_worker

from .contracts import RealignContract, load_contract
from .metrics import TABM_COLUMNS


class RealignTabMFoldError(ValueError):
    """Raised when the sealed 2022 TabM fold does not produce trusted evidence."""


@dataclass(frozen=True)
class TabMFoldResult:
    status: str
    checkpoint_path: Path | None
    predictions_path: Path | None
    completed_epochs: int
    brier: float | None


def build_tabm_f1_job(contract: RealignContract) -> CampaignJob:
    if (
        contract.campaign_id != "catboost_50_50_realign_v2"
        or tuple((fold.train_end_year, fold.valid_year) for fold in contract.folds)
        != ((2021, 2022), (2022, 2023), (2023, 2024))
    ):
        raise RealignTabMFoldError("realignment contract identity differs")
    config = contract.tabm
    return CampaignJob(
        candidate_id="realign_tabm__tr2021__va2022__s3407",
        capacity=str(config["capacity"]),
        k=int(config["k"]),
        width=int(config["width"]),
        blocks=int(config["blocks"]),
        dropout=float(config["dropout"]),
        num_embedding=str(config["num_embedding"]),
        loss=str(config["loss"]),
        scheduler=str(config["scheduler"]),
        learning_rate=float(config["learning_rate"]),
        seed=int(config["seed"]),
        train_end_year=2021,
        valid_year=2022,
        sample_mode="full",
        max_epochs=int(config["max_epochs"]),
        min_epochs=int(config["min_epochs"]),
        patience=int(config["patience"]),
        feature_bundle=None,
    )


def _load_training_rows(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path(data_dir) / "train.csv"
    if train_path.is_symlink() or not train_path.is_file():
        raise RealignTabMFoldError("official train.csv is missing")
    try:
        train = pd.read_csv(train_path)
    except Exception as error:
        raise RealignTabMFoldError(f"official train.csv cannot be read: {error}") from error
    required = {
        "row_id",
        "season",
        "control_success",
        "game_type",
        "game_month",
        "pitcher_id",
        "batter_id",
    }
    if not required.issubset(train.columns):
        raise RealignTabMFoldError("official train.csv columns differ")
    try:
        seasons = pd.to_numeric(train["season"], errors="raise").astype("int64")
    except Exception as error:
        raise RealignTabMFoldError("official season values are invalid") from error
    fit_rows = train.loc[seasons.le(2021)].reset_index(drop=True)
    valid_rows = train.loc[seasons.eq(2022)].reset_index(drop=True)
    if fit_rows.empty or valid_rows.empty:
        raise RealignTabMFoldError("2021->2022 temporal fold is empty")
    return fit_rows, valid_rows


def _expected_known(fit: pd.Series, valid: pd.Series) -> np.ndarray:
    known = set(fit.astype("string").dropna().astype(str))
    return np.where(
        valid.astype("string").fillna("__MISSING__").astype(str).isin(known),
        "known",
        "oov",
    )


def _same(left: pd.Series, right: pd.Series) -> bool:
    return bool((left.eq(right) | (left.isna() & right.isna())).all())


def _normalize_predictions(
    raw_path: Path,
    fit_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    output_path: Path,
    expected_brier: float,
) -> None:
    try:
        frame = pd.read_csv(raw_path)
    except Exception as error:
        raise RealignTabMFoldError(f"TabM predictions cannot be read: {error}") from error
    if not set(TABM_COLUMNS).issubset(frame.columns) or frame.empty:
        raise RealignTabMFoldError("TabM prediction columns differ")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise RealignTabMFoldError("TabM prediction row_id must be unique")
    expected_row_id = valid_rows["row_id"].astype(str).reset_index(drop=True)
    actual_row_id = frame["row_id"].astype(str).reset_index(drop=True)
    if not expected_row_id.equals(actual_row_id):
        raise RealignTabMFoldError("TabM prediction row alignment differs")
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy("float64")
    source_target = pd.to_numeric(
        valid_rows["control_success"], errors="coerce"
    ).to_numpy("float64")
    if (
        not np.isfinite(target).all()
        or not np.isin(target, (0.0, 1.0)).all()
        or not np.array_equal(target, source_target)
    ):
        raise RealignTabMFoldError("TabM prediction target differs")
    probability = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(
        "float64"
    )
    if (
        not np.isfinite(probability).all()
        or ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise RealignTabMFoldError("TabM prediction probability is invalid")
    brier = float(np.mean(np.square(probability - target), dtype=np.float64))
    if not math.isclose(brier, expected_brier, rel_tol=1e-12, abs_tol=1e-12):
        raise RealignTabMFoldError("TabM prediction Brier differs")
    expected_segments = {
        "game_type": valid_rows["game_type"].reset_index(drop=True),
        "game_month": valid_rows["game_month"].reset_index(drop=True),
        "pitcher_id_known": pd.Series(
            _expected_known(fit_rows["pitcher_id"], valid_rows["pitcher_id"])
        ),
        "batter_id_known": pd.Series(
            _expected_known(fit_rows["batter_id"], valid_rows["batter_id"])
        ),
    }
    for column, expected in expected_segments.items():
        actual = frame[column].reset_index(drop=True)
        if not _same(actual.astype("string"), expected.astype("string")):
            raise RealignTabMFoldError(f"TabM prediction segment differs: {column}")
    normalized = frame.loc[:, TABM_COLUMNS].copy()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() or output_path.is_symlink() or temporary.exists():
        raise RealignTabMFoldError("normalized TabM prediction output already exists")
    try:
        normalized.to_csv(temporary, index=False)
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_tabm_f1(
    *,
    data_dir: Path,
    output_dir: Path,
    cache_root: Path,
    absolute_deadline: float,
    worker: Callable[
        [CampaignJob, Path, Path, Path, float], CampaignJobResult
    ] = run_worker,
) -> TabMFoldResult:
    if not math.isfinite(absolute_deadline):
        raise RealignTabMFoldError("absolute deadline must be finite")
    contract = load_contract()
    job = build_tabm_f1_job(contract)
    output = Path(output_dir)
    raw = worker(job, Path(data_dir), output, Path(cache_root), absolute_deadline)
    if type(raw) is not CampaignJobResult:
        raise RealignTabMFoldError("worker result type differs")
    if raw.candidate_id != job.candidate_id:
        raise RealignTabMFoldError("worker result candidate differs")
    if raw.status != "completed":
        raise RealignTabMFoldError("completed result is required")
    if (
        raw.brier is None
        or not math.isfinite(raw.brier)
        or not 0.0 <= raw.brier <= 1.0
        or raw.completed_epochs < job.min_epochs
        or raw.completed_epochs > job.max_epochs
        or raw.failure is not None
    ):
        raise RealignTabMFoldError("completed worker result fields differ")
    if not _valid_completed_result(output, job, raw):
        raise RealignTabMFoldError("worker artifact binding differs")
    try:
        _validate_checkpoint_meta(job, raw, output)
    except Exception as error:
        raise RealignTabMFoldError(f"checkpoint is invalid: {error}") from error
    if raw.predictions_path is None or raw.checkpoint is None:
        raise RealignTabMFoldError("completed worker artifact is missing")
    fit_rows, valid_rows = _load_training_rows(Path(data_dir))
    normalized = output / "tabm_f1_predictions.csv"
    _normalize_predictions(
        raw.predictions_path, fit_rows, valid_rows, normalized, raw.brier
    )
    return TabMFoldResult(
        status="completed",
        checkpoint_path=raw.checkpoint,
        predictions_path=normalized,
        completed_epochs=raw.completed_epochs,
        brier=raw.brier,
    )
