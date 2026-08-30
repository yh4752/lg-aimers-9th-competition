from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd

from experiments.direct_expert.stage_b_runtime import ProductionStageBRuntime


def _frame(expert: str, year: int, seed: int) -> pd.DataFrame:
    size = 120
    target = np.asarray([0, 1] * (size // 2), dtype="int8")
    anchor = np.where(target == 1, 0.54, 0.46)
    bonus = {"D0": 0.035, "D1": 0.030, "D2": 0.025, "D3": 0.020}[expert]
    probability = anchor + np.where(target == 1, bonus, -bonus)
    probability += (seed % 7 - 3) * 0.0001
    return pd.DataFrame(
        {
            "row_id": [f"{year}-{index:03d}" for index in range(size)],
            "target": target,
            "probability": probability,
            "game_type": ["R", "F"] * (size // 2),
            "pitcher_id": np.repeat(np.arange(30), 4),
            "oof_year": year,
            "p_anchor": anchor,
        }
    )


def test_production_decision_keeps_all_gate_evidence() -> None:
    runtime = object.__new__(ProductionStageBRuntime)
    runtime._experts = ("D0", "D1", "D2", "D3")
    runtime._chosen_recipe = None
    runtime._decision = None
    runtime._frame_cache = {}
    runtime._averaged_cache = {}
    runtime.base = SimpleNamespace(clear_fold_cache=lambda: None)
    runtime._frame = MethodType(lambda _self, expert, year, seed: _frame(expert, year, seed), runtime)
    decision = runtime.decide({})
    assert decision["status"] in {"accepted_stable", "accepted_aggressive"}
    assert decision["evaluated"]
    for candidate in decision["evaluated"]:
        assert {"candidate_id", "status", "failed_gates", "recipe", "evidence"}.issubset(candidate)
        assert set(candidate["evidence"]["fold_gains"]) == {2022, 2023, 2024}
