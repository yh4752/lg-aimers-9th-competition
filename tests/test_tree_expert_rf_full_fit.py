from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.features import TreeFeatureBatch
from experiments.tree_expert.rf_contracts import load_rf_contract
from experiments.tree_expert.rf_decisions import RFAcceptanceDecision
from experiments.tree_expert.rf_full_fit import (
    RFFullFitError,
    create_rf_full_fit_token,
    full_fit_iterations,
    full_fit_rf,
)


def _decision(status: str = "accepted") -> RFAcceptanceDecision:
    return RFAcceptanceDecision(
        status=status,
        reason="all_rf_gates_passed" if status == "accepted" else "weighted_gain_failed",
        f_head="f_small",
        include_r=True,
        alpha_r=0.25,
        alpha_f=0.5,
        fold_gains=MappingProxyType({(2021, 2022): 0.1, (2022, 2023): 0.1, (2023, 2024): 0.1}),
        segment_gains=MappingProxyType({"R": 0.1, "F": 0.1}),
        recent_f_gain=0.1,
        weighted_gain=0.1,
        improved_fold_count=3,
        maximum_segment_regression=0.0,
    )


def _evidence():
    return {
        head: {seed: (4, 8, 6) for seed in (42, 2026, 3407)}
        for head in ("f_small", "r_expert")
    }


def _contract():
    contract = load_rf_contract()
    return replace(contract, gates=replace(contract.gates, minimum_segment_rows=1))


def test_full_fit_iterations_uses_median_plus_one_and_clamps() -> None:
    assert full_fit_iterations((4, 8, 6), maximum=800) == 7
    assert full_fit_iterations((0, 0, 0), maximum=800) == 1
    assert full_fit_iterations((999, 999, 999), maximum=800) == 800


def test_full_fit_token_is_blocked_for_rejected_decision() -> None:
    with pytest.raises(RFFullFitError, match="accepted decision is required"):
        create_rf_full_fit_token(_decision("rejected"), _evidence())


class RecordingModel:
    def fit(self, frame, target, **_kwargs):
        self.rows = len(frame)
        self.target = np.asarray(target)
        return self

    def save_model(self, path: str):
        Path(path).write_bytes(b"model")


class Harness:
    @staticmethod
    def fit(rows, _history, **_kwargs):
        state = type("State", (), {"categorical_columns": ("category",)})()
        batch = TreeFeatureBatch(
            frame=pd.DataFrame({"category": rows["row_id"].astype(str)}),
            anchor=np.full(len(rows), 0.5),
            row_id=rows["row_id"].astype(str).to_numpy(),
            target=rows["control_success"].to_numpy(dtype="float64"),
        )
        return state, batch


def test_full_fit_writes_one_state_per_head_and_three_models(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "row_id": ["r1", "r2", "f1", "f2"],
            "season": [2023, 2024, 2023, 2024],
            "game_type": ["R", "R", "F", "F"],
            "control_success": [0, 1, 1, 0],
        }
    )

    def exporter(_state, destination, **_kwargs):
        Path(destination).mkdir(parents=True)
        (Path(destination) / "feature_state.json").write_text("{}")
        return Path(destination)

    result = full_fit_rf(
        decision=_decision(),
        train=train,
        output_dir=tmp_path,
        iteration_evidence=_evidence(),
        contract=_contract(),
        model_factory=lambda _parameters: RecordingModel(),
        feature_builder=Harness.fit,
        state_exporter=exporter,
    )

    assert set(result.frozen_states) == {"f_small", "r_expert"}
    assert len(result.model_paths) == 6
    assert all(path.is_file() for path in result.model_paths.values())
    assert result.manifest_path.is_file()
