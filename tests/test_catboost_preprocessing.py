from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.catboost_preprocessing.campaign import (
    CATBOOST_STRUCTURES,
    CatBoostPreprocessingRuntime,
    expand_catboost_jobs,
)
from experiments.catboost_preprocessing.features import (
    fit_catboost_features,
    transform_catboost_features,
)
from experiments.independent_dl.preprocessing_contracts import (
    load_preprocessing_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/independent_dl/configs/preprocessing_ablation_v1.json"


def test_catboost_structures_match_pinned_source() -> None:
    assert CATBOOST_STRUCTURES == {
        "champion": {"depth": 7, "iterations": 400},
        "depth5": {"depth": 5, "iterations": 600},
        "depth8": {"depth": 8, "iterations": 300},
        "lr003": {"learning_rate": 0.03, "iterations": 700},
    }


def test_catboost_native_keeps_nan_and_categorical_ids(preprocessing_frame) -> None:
    train = preprocessing_frame.copy()
    train.loc[0, "li"] = np.nan

    state, features = fit_catboost_features(train, components=())

    assert np.isnan(features.loc[0, "li"])
    assert features["pitcher_id"].dtype == object
    assert "control_success" not in features
    assert "asof_pitcher_pitchmix_n" not in features
    assert "pitcher_id" in state.categorical_columns


def test_catboost_transform_uses_fold_fitted_prior_and_oov(preprocessing_train, preprocessing_valid) -> None:
    preprocessing_valid = preprocessing_valid.copy()
    preprocessing_valid.loc[
        1,
        [
            "asof_pitcher_n",
            "asof_pitcher_pitchmix_n",
            "asof_pitcher_success_rate",
        ],
    ] = np.nan
    state, _ = fit_catboost_features(
        preprocessing_train, components=("pitcher_smooth_k25",)
    )
    valid = transform_catboost_features(preprocessing_valid, state)

    assert valid.loc[1, "pitcher_success_smooth_25"] == state.preprocessing.target_prior
    assert state.category_values["pitcher_id"] == ("11", "12", "13", "14")


def test_wave_e_expands_exact_1020_independent_jobs() -> None:
    campaign = load_preprocessing_campaign(CONFIG)
    jobs = expand_catboost_jobs(campaign)

    assert len(jobs) == 1020
    assert len({job.job_id for job in jobs}) == 1020
    assert {job.seed for job in jobs} == {42, 2026, 3407}
    assert {job.structure_id for job in jobs} == set(CATBOOST_STRUCTURES)
    assert {job.family for job in jobs} == {"catboost"}


class _FakeModel:
    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, x, y, *, cat_features, verbose):
        self.cat_features = tuple(cat_features)
        self.mean = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean)

    def save_model(self, path):
        Path(path).write_text(json.dumps(self.parameters), encoding="utf-8")


def test_runtime_lazy_model_factory_writes_required_artifacts(
    tmp_path, preprocessing_frame
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = preprocessing_frame.copy()
    frame["season"] = [2019, 2019, 2020, 2020, 2020]
    frame.to_csv(data_dir / "train.csv", index=False)
    campaign = load_preprocessing_campaign(CONFIG)
    job = next(
        job
        for job in expand_catboost_jobs(campaign)
        if job.train_end_year == 2019 and job.setting.setting_id == "tree_native"
    )
    runtime = CatBoostPreprocessingRuntime(
        data_dir, model_factory=lambda **parameters: _FakeModel(**parameters)
    )

    result = runtime.run_job(job, tmp_path / "output")

    assert result.metrics_path.is_file()
    assert result.predictions_path.is_file()
    assert (tmp_path / "output/model.cbm").is_file()
    prediction_columns = set(
        __import__("pandas").read_csv(result.predictions_path).columns
    )
    assert {"row_id", "target", "probability", "structure_id"} <= prediction_columns
