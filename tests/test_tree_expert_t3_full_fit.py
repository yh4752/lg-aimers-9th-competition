from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.features import TreeFeatureBatch
from experiments.tree_expert.t3_decisions import T3AcceptanceDecision
from experiments.tree_expert.t3_full_fit import T3FullFitError, full_fit_iterations, full_fit_t3


class FakeModel:
    def fit(self, _x, _y, **kwargs):
        self.weights = np.asarray(kwargs["sample_weight"])
        return self

    def save_model(self, path):
        Path(path).write_bytes(b"model")


def decision(status="accepted"):
    return T3AcceptanceDecision(
        status=status, reason="fixture", decay=0.55, recent_weight=0.8,
        weighted_gain=0.001, fold_gains=MappingProxyType({(2021, 2022): 0.001, (2022, 2023): 0.001, (2023, 2024): 0.001}),
        maximum_segment_regression=0.0, non_worse_seed_count=3,
    )


def train_rows():
    return pd.DataFrame({
        "row_id": [f"r{i}" for i in range(6)], "season": list(range(2019, 2025)),
        "control_success": [0, 1, 0, 1, 0, 1],
    })


def feature_builder(rows, _history, **_kwargs):
    state = type("State", (), {"categorical_columns": ()})()
    batch = TreeFeatureBatch(
        frame=pd.DataFrame({"x": np.arange(len(rows), dtype="float32")}),
        anchor=np.full(len(rows), 0.5), row_id=rows["row_id"].to_numpy(),
        target=rows["control_success"].to_numpy(dtype="int8"),
    )
    return state, batch


def state_exporter(_state, destination, **_kwargs):
    destination.mkdir(parents=True)
    (destination / "feature_state.json").write_text("{}")
    return destination


def test_full_fit_rejects_nonaccepted_decision(tmp_path):
    with pytest.raises(T3FullFitError, match="candidate is not accepted"):
        full_fit_t3(
            decision=decision("rejected"), train=train_rows(), output_dir=tmp_path,
            iteration_evidence={}, model_factory=lambda _: FakeModel(),
            feature_builder=feature_builder, state_exporter=state_exporter,
        )


def test_full_fit_creates_six_bound_models(tmp_path):
    iterations = {
        head: {seed: (60, 70, 80) for seed in (42, 2026, 3407)}
        for head in ("recent", "multi")
    }
    result = full_fit_t3(
        decision=decision(), train=train_rows(), output_dir=tmp_path,
        iteration_evidence=iterations, model_factory=lambda _: FakeModel(),
        feature_builder=feature_builder, state_exporter=state_exporter,
    )
    assert len(result.model_paths) == 6
    assert all(path.is_file() for path in result.model_paths.values())
    assert result.manifest_path.is_file()
    assert full_fit_iterations((60, 70, 80)) == 71
