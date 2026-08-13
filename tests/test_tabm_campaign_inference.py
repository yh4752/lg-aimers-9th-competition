from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tabm_campaign.inference_runtime import audit_frozen_predictor


class _FrozenLinear:
    def __init__(self) -> None:
        self.digest = "a" * 64

    def state_digest(self) -> str:
        return self.digest

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return frame[["x"]].to_numpy(dtype="float32"), np.empty((len(frame), 0), dtype="int64")

    def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 2048) -> np.ndarray:
        del batch_size
        return 1.0 / (1.0 + np.exp(-frame["x"].to_numpy(dtype="float32")))


def test_inference_module_has_no_training_surface() -> None:
    source = Path("experiments/tabm_campaign/inference_runtime.py").read_text(encoding="utf-8")
    assert "fit_preprocessor" not in source
    assert "optimizer" not in source
    assert "backward(" not in source


def test_row_independence_with_fp32_tolerance() -> None:
    frame = pd.DataFrame({"row_id": [f"r-{i}" for i in range(11)], "x": np.linspace(-2, 2, 11)})
    report = audit_frozen_predictor(_FrozenLinear(), frame, batch_sizes=(1, 3, 7))
    assert report.max_abs_probability_delta <= 1e-6
    assert report.features_exact
    assert report.state_digest_before == report.state_digest_after
