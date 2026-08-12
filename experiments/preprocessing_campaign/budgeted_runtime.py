"""Official-data runtime for one sealed budgeted preprocessing job."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import time
from typing import Callable

import numpy as np
import pandas as pd

from experiments.independent_dl.preprocessing_campaign import PreprocessingJobResult

from .budgeted_contracts import BudgetedJob, deterministic_temporal_sample


@dataclass(frozen=True)
class PreparedFold:
    train: pd.DataFrame
    valid: pd.DataFrame
    sample_sha256: str


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _row_hash(frame: pd.DataFrame) -> str:
    digest = sha256()
    for value in frame["row_id"].astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class BudgetedRuntime:
    """Run one DL or CatBoost job without owning the shared campaign manifest."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        cache_root: str | Path,
        proxy_max_rows: int = 400_000,
        frame_loader: Callable[[], tuple[pd.DataFrame, pd.DataFrame]] | None = None,
        fit_function: object | None = None,
        adapter_factory: object | None = None,
        catboost_factory: object | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.proxy_max_rows = int(proxy_max_rows)
        self.frame_loader = frame_loader
        self.fit_function = fit_function
        self.adapter_factory = adapter_factory
        self.catboost_factory = catboost_factory
        self._frames: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._frames is None:
            if self.frame_loader is not None:
                self._frames = self.frame_loader()
            else:
                train_path = self.data_dir / "train.csv"
                history_path = self.data_dir / "trackman_history.csv"
                if not train_path.is_file() or not history_path.is_file():
                    raise ValueError("train.csv and trackman_history.csv are required")
                self._frames = (
                    pd.read_csv(train_path),
                    pd.read_csv(history_path),
                )
        return self._frames

    def prepare_fold(self, job: BudgetedJob) -> PreparedFold:
        full, _ = self._load()
        seasons = pd.to_numeric(full["season"], errors="raise").astype("int64")
        train = full.loc[seasons.le(job.train_end_year)].copy()
        valid = full.loc[seasons.eq(job.valid_year)].copy().reset_index(drop=True)
        if train.empty or valid.empty:
            raise ValueError("training and validation folds must both be nonempty")
        if job.sample_mode == "proxy":
            train = deterministic_temporal_sample(
                train,
                train_end_year=job.train_end_year,
                max_rows=self.proxy_max_rows,
                seed=job.seed,
            )
        elif job.sample_mode == "full":
            train = train.reset_index(drop=True)
        else:
            raise ValueError(f"unknown sample mode: {job.sample_mode}")
        return PreparedFold(train, valid, _row_hash(train))

    @staticmethod
    def _default_adapter(family: str) -> object:
        from experiments.independent_dl.campaign import OfficialCampaignRuntime

        return OfficialCampaignRuntime._default_adapter(family)

    def run_job(
        self, job: BudgetedJob, output_dir: str | Path
    ) -> PreprocessingJobResult:
        if job.family == "catboost":
            return self._run_catboost(job, Path(output_dir))
        return self._run_dl(job, Path(output_dir))

    def _run_dl(self, job: BudgetedJob, output_dir: Path) -> PreprocessingJobResult:
        from experiments.independent_dl.features import materialize_preprocessed_fold_cache
        from experiments.independent_dl.preprocessing import PreprocessingSpec
        from experiments.independent_dl.training import TrainRequest, fit_candidate

        started = time.monotonic()
        fold = self.prepare_fold(job)
        _, history = self._load()
        cache = materialize_preprocessed_fold_cache(
            self.cache_root / job.family,
            fold.train,
            fold.valid,
            history,
            "raw_typed",
            PreprocessingSpec(job.preprocessing_profile, job.components),
            job.train_end_year,
            job.valid_year,
        )
        request = TrainRequest(
            candidate_id=job.job_id,
            family=job.family,
            seed=job.seed,
            epochs=160,
            model_config=job.model,
            training_config=job.training,
            train=cache.train,
            valid=cache.valid,
        )
        adapter_factory = self.adapter_factory or self._default_adapter
        output_dir.mkdir(parents=True, exist_ok=True)
        result = (self.fit_function or fit_candidate)(
            request, adapter_factory(job.family), output_dir
        )
        probability = np.asarray(result.predictions, dtype="float64")
        target = np.asarray(cache.valid.y, dtype="float64")
        if probability.shape != target.shape:
            raise ValueError("predictions are not aligned to validation")
        pitcher_index = cache.state.categorical_columns.index("pitcher_id")
        batter_index = cache.state.categorical_columns.index("batter_id")
        predictions_path = output_dir / "predictions.csv"
        temporary = predictions_path.with_suffix(".csv.tmp")
        pd.DataFrame(
            {
                "row_id": cache.valid.row_id.astype(str),
                "fold": f"valid_{job.valid_year}",
                "season": cache.valid.season,
                "game_type": cache.valid.game_type.astype(str),
                "target": target.astype(int),
                "probability": probability,
                "family": job.family,
                "setting_id": job.setting_id,
                "seed": job.seed,
                "pitcher_oov": (cache.valid.x_cat[:, pitcher_index] == 0).astype(int),
                "batter_oov": (cache.valid.x_cat[:, batter_index] == 0).astype(int),
            }
        ).to_csv(temporary, index=False)
        os.replace(temporary, predictions_path)
        brier = float(np.mean(np.square(probability - target)))
        metrics_path = output_dir / "metrics.json"
        elapsed = time.monotonic() - started
        hardware = dict(result.hardware)
        _atomic_json(
            metrics_path,
            {
                "job_id": job.job_id,
                "family": job.family,
                "setting_id": job.setting_id,
                "train_end_year": job.train_end_year,
                "valid_year": job.valid_year,
                "seed": job.seed,
                "sample_mode": job.sample_mode,
                "sample_sha256": fold.sample_sha256,
                "train_rows": len(fold.train),
                "valid_rows": len(fold.valid),
                "brier": brier,
                "best_epoch": int(result.best_epoch),
                "completed_epochs": int(result.completed_epochs),
                "validation_points": len(result.validation_curve),
                "validation_curve": [list(item) for item in result.validation_curve],
                "elapsed_seconds": elapsed,
                "hardware": hardware,
            },
        )
        ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        gpu_gb = float(hardware.get("peak_gpu_gb", 0.0))
        return PreprocessingJobResult(
            metrics_path, predictions_path, brier, elapsed, ram_gb, gpu_gb
        )

    def _run_catboost(
        self, job: BudgetedJob, output_dir: Path
    ) -> PreprocessingJobResult:
        from experiments.catboost_preprocessing.features import (
            fit_catboost_features,
            transform_catboost_features,
        )

        started = time.monotonic()
        fold = self.prepare_fold(job)
        state, x_train = fit_catboost_features(fold.train, components=job.components)
        x_valid = transform_catboost_features(fold.valid, state)
        y_train = pd.to_numeric(fold.train["control_success"], errors="raise").to_numpy()
        target = pd.to_numeric(fold.valid["control_success"], errors="raise").to_numpy()
        if self.catboost_factory is None:
            from catboost import CatBoostRegressor

            factory = CatBoostRegressor
        else:
            factory = self.catboost_factory
        parameters = {
            **dict(job.model),
            "random_seed": job.seed,
            "task_type": "GPU",
            "allow_writing_files": False,
            "verbose": 100,
            "od_type": "Iter",
            "od_wait": 50,
        }
        model = factory(**parameters)
        model.fit(
            x_train,
            y_train,
            cat_features=list(state.categorical_columns),
            eval_set=(x_valid, target),
            use_best_model=True,
            verbose=False,
        )
        probability = np.clip(
            np.asarray(model.predict(x_valid), dtype="float64").reshape(-1), 0.0, 1.0
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(output_dir / "model.cbm"))
        predictions_path = output_dir / "predictions.csv"
        pd.DataFrame(
            {
                "row_id": fold.valid["row_id"].astype(str),
                "fold": f"valid_{job.valid_year}",
                "season": fold.valid["season"],
                "game_type": fold.valid["game_type"].astype(str),
                "target": target.astype(int),
                "probability": probability,
                "family": job.family,
                "setting_id": job.setting_id,
                "seed": job.seed,
                "pitcher_oov": ~x_valid["pitcher_id"].astype(str).isin(
                    state.category_values["pitcher_id"]
                ),
                "batter_oov": ~x_valid["batter_id"].astype(str).isin(
                    state.category_values["batter_id"]
                ),
            }
        ).to_csv(predictions_path, index=False)
        evaluations = model.get_evals_result()
        validation = evaluations.get("validation", evaluations.get("validation_0", {}))
        rmse = next(iter(validation.values()), [])
        curve = [(index, float(value) ** 2) for index, value in enumerate(rmse)]
        brier = float(np.mean(np.square(probability - target)))
        elapsed = time.monotonic() - started
        metrics_path = output_dir / "metrics.json"
        _atomic_json(
            metrics_path,
            {
                "job_id": job.job_id,
                "family": job.family,
                "setting_id": job.setting_id,
                "train_end_year": job.train_end_year,
                "valid_year": job.valid_year,
                "seed": job.seed,
                "sample_mode": job.sample_mode,
                "sample_sha256": fold.sample_sha256,
                "train_rows": len(fold.train),
                "valid_rows": len(fold.valid),
                "brier": brier,
                "best_epoch": int(model.get_best_iteration()),
                "completed_epochs": len(curve),
                "validation_points": len(curve),
                "validation_curve": [list(item) for item in curve],
                "elapsed_seconds": elapsed,
            },
        )
        ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        return PreprocessingJobResult(
            metrics_path, predictions_path, brier, elapsed, ram_gb, 0.0
        )
