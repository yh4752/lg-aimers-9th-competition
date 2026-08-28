import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.hc_contracts import load_hc_contract
from experiments.tree_expert.hc_training import (
    HCTrainingError,
    build_c1_training_rows,
    ensemble_c1_predictions,
    materialize_oof_feature_rows,
    run_c1_job,
)


def _oof_rows() -> pd.DataFrame:
    records = []
    for year in (2021, 2022, 2023, 2024):
        for index in range(4):
            target = float(index % 2)
            p0 = 0.4 + index * 0.05
            records.append(
                {
                    "row_id": f"r{year}_{index}",
                    "oof_year": year,
                    "target": target,
                    "p0": p0,
                    "game_type": "R" if index % 2 else "F",
                    "pitcher_id": f"p{index % 2}",
                    "batter_id": f"b{index}",
                    "feature_num": float(index),
                    "feature_cat": str(index % 2),
                }
            )
    return pd.DataFrame(records)


@pytest.mark.parametrize(
    ("valid_year", "allowed"),
    [(2022, {2021}), (2023, {2021, 2022}), (2024, {2021, 2022, 2023})],
)
def test_c1_training_uses_only_earlier_oof(valid_year, allowed):
    frame = build_c1_training_rows(_oof_rows(), valid_year=valid_year)
    assert set(frame["oof_year"]) == allowed
    assert frame["oof_year"].max() < valid_year
    np.testing.assert_allclose(frame["residual_target"], frame["target"] - frame["p0"])


def test_c1_training_rejects_duplicate_row_identity():
    rows = pd.concat([_oof_rows(), _oof_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(HCTrainingError, match="row identity differs"):
        build_c1_training_rows(rows, valid_year=2023)


class _FakeRegressor:
    def __init__(self):
        self.parameters = None
        self.fit_target = None

    def fit(self, frame, target, **kwargs):
        self.fit_target = np.asarray(target)
        return self

    def predict(self, frame):
        return np.full(len(frame), 0.03)

    def save_model(self, path):
        Path(path).write_bytes(b"model")

    def get_best_iteration(self):
        return 7


def test_c1_job_fits_residual_and_clips_probability(tmp_path: Path):
    rows = _oof_rows()
    train = build_c1_training_rows(rows, valid_year=2024)
    valid = rows.loc[rows["oof_year"].eq(2024)].copy()
    valid.loc[valid.index[0], "p0"] = 0.99
    model = _FakeRegressor()

    result = run_c1_job(
        job_id="hc__test",
        seed=3407,
        profile_name="hc_balanced",
        train_rows=train,
        valid_rows=valid,
        feature_columns=("feature_num", "feature_cat", "p0"),
        categorical_columns=("feature_cat",),
        output_dir=tmp_path / "job",
        absolute_deadline=10**12,
        gpu_id=0,
        model_factory=lambda parameters: model,
    )

    predictions = pd.read_csv(result.predictions_path)
    assert model.fit_target.tolist() == pytest.approx(train["target"] - train["p0"])
    assert predictions.loc[0, "p1"] == pytest.approx(0.99999)
    assert result.best_iteration == 7
    schema = json.loads((tmp_path / "job/feature_schema.json").read_text())
    assert schema["feature_columns"] == ["feature_num", "feature_cat", "p0"]
    assert schema["categorical_columns"] == ["feature_cat"]


def test_c1_seed_ensemble_requires_exact_row_alignment(tmp_path: Path):
    paths = {}
    for seed, offset in ((3407, 0.00), (42, 0.03), (2026, -0.03)):
        frame = _oof_rows().loc[lambda value: value["oof_year"].eq(2024)].copy()
        frame["p1"] = frame["p0"] + offset
        path = tmp_path / f"{seed}.csv"
        frame.to_csv(path, index=False)
        paths[seed] = path
    ensemble = ensemble_c1_predictions(paths)
    assert ensemble["p1"].tolist() == pytest.approx(
        _oof_rows().loc[lambda value: value["oof_year"].eq(2024), "p0"]
    )
    changed = pd.read_csv(paths[42]).iloc[::-1]
    changed.to_csv(paths[42], index=False)
    with pytest.raises(HCTrainingError, match="row alignment differs"):
        ensemble_c1_predictions(paths)


def test_registered_catboost_is_residual_regression():
    parameters = load_hc_contract().residual_catboost
    assert parameters["loss_function"] == "RMSE"
    assert parameters["iterations"] == 800
    assert parameters["task_type"] == "GPU"


def test_oof_feature_materialization_freezes_each_year_at_previous_season():
    records = []
    predictions = {}
    for year in (2020, 2021, 2022, 2023, 2024):
        record = {
            "row_id": f"r{year}",
            "season": year,
            "control_success": year % 2,
            "game_type": "R",
            "pitcher_id": "p1",
            "batter_id": "b1",
            "pitcher_hand": "R",
            "batter_hand": "L",
            "balls_before": 1,
            "strikes_before": 1,
            "outs_before": 1,
            "base_state": "0",
        }
        records.append(record)
        if year >= 2021:
            predictions[year] = pd.DataFrame(
                {
                    "row_id": [f"r{year}"],
                    "target": [year % 2],
                    "probability": [0.5],
                    "pitcher_id_known": ["known"],
                    "batter_id_known": ["known"],
                }
            )
    official = pd.DataFrame(records)
    cutoffs = []

    def builder(fit_rows, history, **kwargs):
        cutoffs.append((kwargs["valid_year"], int(fit_rows["season"].max())))
        state = SimpleNamespace(categorical_columns=("cat",))
        return state, object()

    def transformer(valid_rows, state):
        return SimpleNamespace(
            frame=pd.DataFrame(
                {"num": np.arange(len(valid_rows), dtype=float), "cat": "x"},
                index=valid_rows.index,
            ),
            row_id=valid_rows["row_id"].astype(str).to_numpy(),
        )

    materialized = materialize_oof_feature_rows(
        official,
        history=pd.DataFrame(),
        p0_by_year=predictions,
        profile_name="hc_balanced",
        profile=load_hc_contract().profiles["hc_balanced"],
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
        tree_feature_builder=builder,
        tree_feature_transformer=transformer,
    )
    assert cutoffs == [(2021, 2020), (2022, 2021), (2023, 2022), (2024, 2023)]
    assert materialized.frame["oof_year"].tolist() == [2021, 2022, 2023, 2024]
    assert "tree__num" in materialized.feature_columns
    assert "tree__cat" in materialized.categorical_columns
    assert "p0" in materialized.feature_columns
    assert materialized.frame["pitcher_id_known"].tolist() == ["known"] * 4
    assert materialized.frame["batter_id_known"].tolist() == ["known"] * 4
