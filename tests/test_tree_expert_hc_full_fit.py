from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.hc_full_fit import (
    HCFullFitError,
    accepted_full_fit_token,
    fit_full_c1_seed,
    full_fit_iterations,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{index}" for index in range(8)],
            "oof_year": [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            "target": [0, 1] * 4,
            "p0": [0.4, 0.6] * 4,
            "tree__num": np.arange(8, dtype=float),
            "tree__cat": ["a", "b"] * 4,
            "hc_rate": np.linspace(0.4, 0.6, 8),
        }
    )


class _Model:
    def __init__(self):
        self.fit_kwargs = None

    def fit(self, frame, target, **kwargs):
        self.fit_kwargs = kwargs
        self.target = np.asarray(target)
        return self

    def save_model(self, path):
        Path(path).write_bytes(b"full-model")


def test_full_fit_iterations_use_each_seed_temporal_median_plus_one():
    assert full_fit_iterations([0, 7, 900]) == 50
    assert full_fit_iterations([120, 99, 140]) == 121
    assert full_fit_iterations([900, 950, 1000]) == 800
    with pytest.raises(HCFullFitError, match="three"):
        full_fit_iterations([1, 2])


def test_full_fit_requires_accepted_c1_or_c2_and_exact_seed_evidence():
    token = accepted_full_fit_token(
        {"status": "accepted", "candidate": "C2"},
        profile_name="hc_balanced",
        calibration_alpha=0.5,
        best_iterations={3407: [5, 6, 7], 42: [10, 11, 12], 2026: [20, 21, 22]},
    )
    assert token.winner == "C2"
    assert dict(token.iterations) == {3407: 50, 42: 50, 2026: 50}
    with pytest.raises(HCFullFitError, match="accepted"):
        accepted_full_fit_token(
            {"status": "fallback", "candidate": "C0"},
            profile_name="hc_balanced",
            calibration_alpha=None,
            best_iterations={3407: [5, 6, 7], 42: [10, 11, 12], 2026: [20, 21, 22]},
        )


def test_full_fit_trains_residual_without_validation_or_early_stopping(tmp_path: Path):
    token = accepted_full_fit_token(
        {"status": "accepted", "candidate": "C1"},
        profile_name="hc_strong",
        calibration_alpha=None,
        best_iterations={3407: [100, 120, 140], 42: [80, 90, 100], 2026: [60, 70, 80]},
    )
    model = _Model()
    result = fit_full_c1_seed(
        token=token,
        seed=3407,
        oof_rows=_rows(),
        feature_columns=("tree__num", "tree__cat", "hc_rate", "p0"),
        categorical_columns=("tree__cat",),
        output_dir=tmp_path / "seed",
        gpu_id=1,
        model_factory=lambda parameters: model,
    )
    np.testing.assert_allclose(model.target, _rows()["target"] - _rows()["p0"])
    assert "eval_set" not in model.fit_kwargs
    assert "early_stopping_rounds" not in model.fit_kwargs
    assert result.iterations == 121
    assert result.model_path.read_bytes() == b"full-model"
