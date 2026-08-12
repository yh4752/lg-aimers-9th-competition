from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.tabicl_v2 import (
    TabICLv2ContractError,
    fit_predict_tabicl_v2,
)


def _batch(prefix: str, *, target: bool) -> FeatureBatch:
    return FeatureBatch(
        row_id=np.array([f"{prefix}-0", f"{prefix}-1"]),
        season=np.array([2023, 2023], dtype="int64"),
        game_type=np.array(["R", "F"]),
        x_num=np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32"),
        x_cat=np.array([[1], [2]], dtype="int64"),
        y=np.array([0, 1], dtype="float32") if target else None,
    )


class _FakeClassifier:
    last: "_FakeClassifier | None" = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.classes_ = np.array([0, 1])
        _FakeClassifier.last = self

    def fit(self, x, y):
        self.fit_frame = x.copy()
        self.fit_target = np.asarray(y).copy()
        return self

    def predict_proba(self, x):
        self.valid_frame = x.copy()
        return np.array([[0.6, 0.4], [0.3, 0.7]], dtype="float64")


def test_tabicl_v2_preserves_typed_columns_and_returns_class_one_probability(
    tmp_path: Path,
) -> None:
    result = fit_predict_tabicl_v2(
        _batch("train", target=True),
        _batch("valid", target=True),
        {
            "n_estimators": 32,
            "kv_cache": True,
            "offload_mode": "auto",
            "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
        },
        tmp_path,
        seed=42,
        estimator_factory=_FakeClassifier,
        device="cuda",
    )

    classifier = _FakeClassifier.last
    assert classifier is not None
    assert classifier.kwargs["n_estimators"] == 32
    assert classifier.kwargs["random_state"] == 42
    assert classifier.fit_frame.columns.tolist() == ["num_0", "num_1", "cat_0"]
    assert isinstance(classifier.fit_frame["cat_0"].dtype, pd.CategoricalDtype)
    assert classifier.fit_frame.index.tolist() == ["train-0", "train-1"]
    assert classifier.valid_frame.index.tolist() == ["valid-0", "valid-1"]
    assert result.predictions.tolist() == [0.4, 0.7]
    assert result.submission_eligibility == "research_only"
    assert result.metadata_path.is_file()


@pytest.mark.parametrize(
    "probabilities",
    (
        np.array([[0.5], [0.5]], dtype="float64"),
        np.array([[0.5, np.nan], [0.5, 0.5]], dtype="float64"),
    ),
)
def test_tabicl_v2_rejects_invalid_probability_matrix(
    tmp_path: Path, probabilities: np.ndarray
) -> None:
    class InvalidClassifier(_FakeClassifier):
        def predict_proba(self, x):
            return probabilities

    with pytest.raises(TabICLv2ContractError):
        fit_predict_tabicl_v2(
            _batch("train", target=True),
            _batch("valid", target=True),
            {
                "n_estimators": 32,
                "kv_cache": True,
                "offload_mode": "auto",
                "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
            },
            tmp_path,
            seed=42,
            estimator_factory=InvalidClassifier,
            device="cpu",
        )
