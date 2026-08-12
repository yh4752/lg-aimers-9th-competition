"""Wave-E CatBoost campaign pinned to source commit 9454d68b93971627e3d3f613ce30be690cb5dce2."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import time
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from experiments.independent_dl.preprocessing_campaign import PreprocessingJobResult
from experiments.independent_dl.preprocessing_contracts import (
    PreprocessingCampaignSpec,
    PreprocessingSetting,
)

from .features import fit_catboost_features, transform_catboost_features


SOURCE_COMMIT = "9454d68b93971627e3d3f613ce30be690cb5dce2"
CATBOOST_STRUCTURES = {
    "champion": {"depth": 7, "iterations": 400},
    "depth5": {"depth": 5, "iterations": 600},
    "depth8": {"depth": 8, "iterations": 300},
    "lr003": {"learning_rate": 0.03, "iterations": 700},
}
CATBOOST_BASE_PARAMETERS = {
    "iterations": 400,
    "depth": 7,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "border_count": 128,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "l2_leaf_reg": 3.0,
    "model_size_reg": 0.5,
    "max_ctr_complexity": 1,
    "allow_writing_files": False,
    "verbose": 100,
}


@dataclass(frozen=True)
class CatBoostPreprocessingJob:
    job_id: str
    wave: str
    family: str
    structure_id: str
    parameters: Mapping[str, object]
    setting: PreprocessingSetting
    train_end_year: int
    valid_year: int
    seed: int


def expand_catboost_jobs(
    campaign: PreprocessingCampaignSpec,
) -> tuple[CatBoostPreprocessingJob, ...]:
    """Expand the sealed 4 × 17 × 5 × 3 Wave-E grid."""

    jobs: list[CatBoostPreprocessingJob] = []
    for structure_id, structure in campaign.catboost_structures.items():
        parameters = MappingProxyType({**CATBOOST_BASE_PARAMETERS, **structure})
        for setting in campaign.catboost_settings:
            for train_end_year, valid_year in campaign.folds:
                for seed in campaign.seeds:
                    job_id = (
                        f"e__{structure_id}__{setting.setting_id}__"
                        f"tr{train_end_year}__va{valid_year}__s{seed}"
                    )
                    jobs.append(
                        CatBoostPreprocessingJob(
                            job_id=job_id,
                            wave="e",
                            family="catboost",
                            structure_id=structure_id,
                            parameters=parameters,
                            setting=setting,
                            train_end_year=train_end_year,
                            valid_year=valid_year,
                            seed=seed,
                        )
                    )
    if len(jobs) != 1020 or len({job.job_id for job in jobs}) != 1020:
        raise ValueError("Wave E expansion is invalid")
    return tuple(jobs)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class CatBoostPreprocessingRuntime:
    """Fit one CatBoost fold; importing CatBoost is deferred until execution."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        model_factory: Callable[..., object] | None = None,
        task_type: str = "CPU",
    ) -> None:
        if task_type not in {"CPU", "GPU"}:
            raise ValueError("task_type must be CPU or GPU")
        self.data_dir = Path(data_dir).resolve()
        self.model_factory = model_factory
        self.task_type = task_type
        self._train: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._train is None:
            path = self.data_dir / "train.csv"
            if not path.is_file():
                raise ValueError("train.csv is required")
            self._train = pd.read_csv(path)
        return self._train

    @staticmethod
    def _default_factory(**parameters):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**parameters)

    def run_job(
        self, job: CatBoostPreprocessingJob, output_dir: Path
    ) -> PreprocessingJobResult:
        started = time.monotonic()
        full = self._load()
        seasons = pd.to_numeric(full["season"], errors="raise")
        train = full.loc[seasons.le(job.train_end_year)].reset_index(drop=True)
        valid = full.loc[seasons.eq(job.valid_year)].reset_index(drop=True)
        if train.empty or valid.empty:
            raise ValueError("training and validation folds must both be nonempty")
        state, x_train = fit_catboost_features(
            train, components=job.setting.components
        )
        x_valid = transform_catboost_features(valid, state)
        y_train = pd.to_numeric(train["control_success"], errors="raise").to_numpy()
        target = pd.to_numeric(valid["control_success"], errors="raise").to_numpy()
        parameters = {
            **job.parameters,
            "random_seed": job.seed,
            "task_type": self.task_type,
        }
        model = (self.model_factory or self._default_factory)(**parameters)
        model.fit(
            x_train,
            y_train,
            cat_features=list(state.categorical_columns),
            verbose=False,
        )
        probability = np.clip(
            np.asarray(model.predict(x_valid), dtype="float64").reshape(-1), 0.0, 1.0
        )
        if len(probability) != len(valid):
            raise ValueError("predictions are not aligned to validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.cbm"
        model.save_model(str(model_path))
        predictions_path = output_dir / "predictions.csv"
        temporary = predictions_path.with_suffix(".csv.tmp")
        pd.DataFrame(
            {
                "row_id": valid["row_id"].astype(str),
                "fold": f"valid_{job.valid_year}",
                "season": valid["season"],
                "game_type": valid["game_type"].astype(str),
                "target": target.astype(int),
                "probability": probability,
                "structure_id": job.structure_id,
                "preprocessing_id": job.setting.setting_id,
                "seed": job.seed,
                "pitcher_oov": ~x_valid["pitcher_id"].astype(str).isin(
                    state.category_values["pitcher_id"]
                ),
                "batter_oov": ~x_valid["batter_id"].astype(str).isin(
                    state.category_values["batter_id"]
                ),
            }
        ).to_csv(temporary, index=False)
        os.replace(temporary, predictions_path)
        brier = float(np.mean(np.square(probability - target)))
        elapsed = time.monotonic() - started
        ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        metrics_path = output_dir / "metrics.json"
        _atomic_json(
            metrics_path,
            {
                "job_id": job.job_id,
                "source_commit": SOURCE_COMMIT,
                "structure_id": job.structure_id,
                "preprocessing_id": job.setting.setting_id,
                "train_end_year": job.train_end_year,
                "valid_year": job.valid_year,
                "seed": job.seed,
                "train_rows": len(train),
                "valid_rows": len(valid),
                "brier": brier,
                "elapsed_seconds": elapsed,
                "peak_ram_gb": ram_gb,
            },
        )
        return PreprocessingJobResult(
            metrics_path, predictions_path, brier, elapsed, ram_gb, 0.0
        )
