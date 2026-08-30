from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.direct_expert.contracts import ExpertJob, expert_spec, load_contract
from experiments.direct_expert.features import DirectFeatureBatch
from experiments.direct_expert.training import (
    FoldData,
    catboost_parameters,
    run_fold_job,
    season_weights,
    training_mask,
)


def _batch(game_type: tuple[str, ...], *, target: bool) -> DirectFeatureBatch:
    size = len(game_type)
    return DirectFeatureBatch(
        frame=pd.DataFrame(
            {
                "numeric": np.arange(size, dtype="float32"),
                "category": pd.Series(["a", "b", "a", "b"][:size], dtype=object),
            }
        ),
        row_id=np.asarray([f"r{i}" for i in range(size)]),
        target=np.asarray([0, 1, 0, 1][:size], dtype="int8") if target else None,
        season=np.asarray([2021, 2022, 2023, 2023][:size], dtype="int16"),
        game_type=np.asarray(game_type),
    )


@pytest.mark.parametrize(
    ("expert_id", "expected"),
    [
        ("D0", [1.0, 1.0, 1.0]),
        ("D1", [0.75**2, 0.75, 1.0]),
        ("D2", [0.55**2, 0.55, 1.0]),
        ("D3", [0.0, 1.0, 1.0]),
    ],
)
def test_registered_season_weights(expert_id: str, expected: list[float]) -> None:
    seasons = np.asarray([2021, 2022, 2023])
    np.testing.assert_allclose(
        season_weights(expert_spec(expert_id), seasons, 2024),
        expected,
    )


def test_r_and_f_specialists_train_only_their_rows() -> None:
    batch = _batch(("R", "F", "R", "F"), target=True)
    assert training_mask(expert_spec("D5"), batch).tolist() == [True, False, True, False]
    assert training_mask(expert_spec("D6"), batch).tolist() == [False, True, False, True]


@dataclass
class FakeModel:
    objective: str

    def fit(self, x, y, **kwargs) -> None:
        self.fit_rows = len(x)
        self.kwargs = kwargs

    def predict_proba(self, x) -> np.ndarray:
        probability = np.linspace(0.2, 0.8, len(x))
        return np.column_stack((1.0 - probability, probability))

    def predict(self, x) -> np.ndarray:
        return np.linspace(-0.2, 1.2, len(x))

    def save_model(self, path: str) -> None:
        Path(path).write_text(self.objective, encoding="utf-8")

    def get_best_iteration(self) -> int:
        return 7


class FakeFactory:
    def __init__(self) -> None:
        self.models: list[FakeModel] = []

    def __call__(self, objective: str, parameters: dict[str, object]) -> FakeModel:
        model = FakeModel(objective)
        self.models.append(model)
        return model


def _fold_data() -> FoldData:
    return FoldData(
        train=_batch(("R", "F", "R", "F"), target=True),
        valid=_batch(("R", "F"), target=False),
        valid_target=np.asarray([0, 1], dtype="int8"),
        valid_metadata=pd.DataFrame(
            {
                "row_id": ["r0", "r1"],
                "game_type": ["R", "F"],
                "pitcher_id": [10, 11],
            }
        ),
        categorical_columns=("category",),
        bindings={name: str(index) * 64 for index, name in enumerate(("contract", "code", "train", "history", "input"), 1)},
    )


def _job(expert_id: str) -> ExpertJob:
    return ExpertJob(
        job_id=f"screen__{expert_id}__2023_2024__s3407",
        expert_id=expert_id,
        fold=(2023, 2024),
        seed=3407,
        phase="screening",
    )


def test_classifier_and_regressor_emit_probabilities(tmp_path: Path) -> None:
    factory = FakeFactory()
    left = run_fold_job(_job("D0"), _fold_data(), tmp_path / "d0", model_factory=factory)
    right = run_fold_job(_job("D4"), _fold_data(), tmp_path / "d4", model_factory=factory)

    assert left.status == right.status == "completed"
    assert left.predictions["probability"].between(0, 1).all()
    assert right.predictions["probability"].between(0, 1).all()
    assert right.predictions["probability"].tolist() == pytest.approx([1e-6, 1 - 1e-6])
    assert (tmp_path / "d0/model.cbm").is_file()
    assert (tmp_path / "d0/job_identity.json").is_file()


def test_parameter_factory_uses_sealed_capacity() -> None:
    parameters = catboost_parameters(
        expert_spec("D7"),
        load_contract(),
        seed=3407,
        gpu=1,
        phase="screening",
    )
    assert parameters["iterations"] == 1800
    assert parameters["depth"] == 9
    assert parameters["max_ctr_complexity"] == 3
    assert parameters["task_type"] == "GPU"
    assert parameters["devices"] == "1"
