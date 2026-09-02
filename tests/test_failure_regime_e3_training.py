from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.direct_expert.features import DirectFeatureBatch
from experiments.failure_regime_e3.contracts import role_spec
from experiments.failure_regime_e3.training import (
    E3Job,
    FoldData,
    catboost_parameters,
    role_training_target,
    run_fold_job,
    season_weights,
)


def _batch(*, target: np.ndarray | None) -> DirectFeatureBatch:
    frame = pd.DataFrame(
        {
            "numeric": [0.1, 0.2, 0.3, 0.4],
            "category": ["a", "b", "a", "b"],
        }
    )
    row_id = np.asarray(["r0", "r1", "r2", "r3"], dtype=object)
    season = np.asarray([2021, 2022, 2022, 2023], dtype="int16")
    game_type = np.asarray(["R", "F", "R", "F"], dtype=object)
    return DirectFeatureBatch(frame, row_id, target, season, game_type)


def _subtypes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valid": [True, True, False, True],
            "middle": [1.0, 0.0, np.nan, 1.0],
            "wild": [0.0, 1.0, np.nan, 0.0],
            "reverse": [1.0, 0.0, np.nan, 1.0],
        }
    )


def test_role_targets_keep_overlap_and_active_game_type() -> None:
    success = np.asarray([0, 1, 1, 0], dtype="int8")
    batch = _batch(target=success)
    middle_target, middle_mask = role_training_target(role_spec("MIDDLE"), batch, _subtypes())
    f_target, f_mask = role_training_target(role_spec("S_F"), batch, _subtypes())

    np.testing.assert_array_equal(middle_target, [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(middle_mask, [True, True, False, True])
    np.testing.assert_array_equal(f_target, success)
    np.testing.assert_array_equal(f_mask, [False, True, False, True])


def test_season_weights_and_capacity_are_registered() -> None:
    weights = season_weights(role_spec("S_GLOBAL"), np.asarray([2021, 2022, 2023]), 2024)
    np.testing.assert_allclose(weights, [0.55**2, 0.55, 1.0])
    success = catboost_parameters(role_spec("S_GLOBAL"), seed=3407, gpu=0, output=Path("job"))
    subtype = catboost_parameters(role_spec("MIDDLE"), seed=42, gpu=1, output=Path("job2"))

    assert success["iterations"] == 2400 and success["depth"] == 10
    assert subtype["iterations"] == 1800 and subtype["depth"] == 9
    assert success["devices"] == "0" and subtype["devices"] == "1"
    assert success["eval_metric"] == "BrierScore"


class FakeModel:
    fits = 0

    def __init__(self, parameters: dict[str, object]):
        self.parameters = parameters

    def fit(self, frame: pd.DataFrame, target: np.ndarray, **kwargs: object) -> None:
        del frame, target, kwargs
        type(self).fits += 1

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability = np.full(len(frame), 0.6, dtype="float64")
        return np.column_stack([1.0 - probability, probability])

    def save_model(self, path: str) -> None:
        Path(path).write_bytes(b"fake-model")


class FlakyModel(FakeModel):
    failures = 1

    def fit(self, frame: pd.DataFrame, target: np.ndarray, **kwargs: object) -> None:
        if type(self).failures:
            type(self).failures -= 1
            raise RuntimeError("interrupted")
        super().fit(frame, target, **kwargs)


class NoBestIterationModel(FakeModel):
    def get_best_iteration(self):
        return None


def test_fold_job_writes_complete_identity_and_reuses_exact_output(tmp_path: Path) -> None:
    FakeModel.fits = 0
    train_target = np.asarray([0, 1, 1, 0], dtype="int8")
    train = _batch(target=train_target)
    valid = _batch(target=None)
    metadata = pd.DataFrame(
        {
            "row_id": valid.row_id,
            "game_type": valid.game_type,
            "pitcher_id": ["p0", "p1", "p2", "p3"],
            "oof_year": [2024] * 4,
        }
    )
    data = FoldData(
        train=train,
        valid=valid,
        valid_target=train_target,
        valid_metadata=metadata,
        categorical_columns=("category",),
        subtype_targets=_subtypes(),
        bindings={"train": "a" * 64, "history": "b" * 64},
    )
    job = E3Job("screen__S_GLOBAL__2023_2024__s3407", "S_GLOBAL", (2023, 2024), 3407, "screening")
    factory = lambda parameters: FakeModel(parameters)

    first = run_fold_job(job, data, tmp_path / "job", gpu=0, model_factory=factory)
    # CatBoost can leave its restart snapshot next to completed evidence.
    # A verified completed job must remain reusable in that case.
    (tmp_path / "job" / "training.snapshot").write_bytes(b"snapshot")
    second = run_fold_job(job, data, tmp_path / "job", gpu=0, model_factory=factory)

    assert first.status == second.status == "completed"
    assert first.brier == second.brier
    assert FakeModel.fits == 1
    assert {path.name for path in (tmp_path / "job").iterdir()} >= {
        "model.cbm", "predictions.csv", "metrics.json", "job_identity.json", "worker.log",
    }
    assert first.predictions["probability"].tolist() == [0.6] * 4


def test_interrupted_job_is_retryable_only_with_bound_active_identity(tmp_path: Path) -> None:
    FlakyModel.failures = 1
    target = np.asarray([0, 1, 1, 0], dtype="int8")
    train = _batch(target=target)
    valid = _batch(target=None)
    data = FoldData(
        train=train,
        valid=valid,
        valid_target=target,
        valid_metadata=pd.DataFrame({
            "row_id": valid.row_id,
            "game_type": valid.game_type,
            "pitcher_id": ["p0", "p1", "p2", "p3"],
            "oof_year": [2024] * 4,
        }),
        categorical_columns=("category",),
        subtype_targets=_subtypes(),
        bindings={"train": "a" * 64, "history": "b" * 64},
    )
    job = E3Job("screen__S_GLOBAL__2023_2024__s3407", "S_GLOBAL", (2023, 2024), 3407, "screening")
    factory = lambda parameters: FlakyModel(parameters)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_fold_job(job, data, tmp_path / "job", gpu=0, model_factory=factory)
    assert (tmp_path / "job" / "active_identity.json").is_file()
    result = run_fold_job(job, data, tmp_path / "job", gpu=0, model_factory=factory)
    assert result.status == "completed"


def test_subtype_job_accepts_catboost_none_best_iteration(tmp_path: Path) -> None:
    target = np.asarray([0, 1, 1, 0], dtype="int8")
    train = _batch(target=target)
    valid = _batch(target=None)
    data = FoldData(
        train=train,
        valid=valid,
        valid_target=target,
        valid_metadata=pd.DataFrame({
            "row_id": valid.row_id,
            "game_type": valid.game_type,
            "pitcher_id": ["p0", "p1", "p2", "p3"],
            "oof_year": [2024] * 4,
        }),
        categorical_columns=("category",),
        subtype_targets=_subtypes(),
        bindings={"train": "a" * 64, "history": "b" * 64},
    )
    job = E3Job("confirmation__MIDDLE__tr2023__va2024__s3407", "MIDDLE", (2023, 2024), 3407, "confirmation")

    result = run_fold_job(
        job,
        data,
        tmp_path / "job",
        gpu=0,
        model_factory=lambda parameters: NoBestIterationModel(parameters),
    )

    assert result.status == "completed"
    assert result.best_iteration == 1799
