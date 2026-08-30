from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.gated_residual_final.calibration import fit_temporal_calibrator
from experiments.gated_residual_final.inference import combine_final_probabilities
from experiments.gated_residual_final.selection import CandidateConfig


def _rows() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": ["r1", "r2"], "game_type": ["R", "F"],
        "pitcher_id": ["p1", "p2"], "batter_id": ["b1", "b2"],
        "pitcher_hand": ["R", "L"], "batter_hand": ["L", "R"],
    })


def _zero_calibrator():
    history = pd.DataFrame({
        "target": [0, 1], "probability": [0.5, 0.5], "oof_year": [2024, 2024],
        "game_type": ["R", "F"], "hand_matchup": ["RL", "LR"],
        "pitcher_id": ["p1", "p2"], "batter_id": ["b1", "b2"],
    })
    return fit_temporal_calibrator(history, validation_year=2025, hierarchy="full", ridge=100)


def test_regular_row_receives_shrunk_d5_residual() -> None:
    result = combine_final_probabilities(
        rows=_rows(), anchor=np.array([0.5, 0.5]), d0=np.array([0.4, 0.4]),
        d5=np.array([0.8, 0.8]), pitcher_counts={"p1": 100, "p2": 100},
        batter_counts={"b1": 100, "b2": 100},
        config=CandidateConfig("G1", 0.1, 25, None, None), calibrator=None,
    )

    assert 0.5 < result[0] < 0.8
    assert result[1] == 0.5


def test_same_row_prediction_does_not_depend_on_companion_rows() -> None:
    rows = _rows()
    kwargs = dict(
        anchor=np.array([0.5, 0.5]), d0=np.array([0.4, 0.4]), d5=np.array([0.8, 0.8]),
        pitcher_counts={"p1": 100, "p2": 100}, batter_counts={"b1": 100, "b2": 100},
        config=CandidateConfig("G3", 0.1, 25, 0.1, 100), calibrator=_zero_calibrator(),
    )
    together = combine_final_probabilities(rows=rows, **kwargs)
    alone = combine_final_probabilities(
        rows=rows.iloc[[0]], anchor=kwargs["anchor"][[0]], d0=kwargs["d0"][[0]],
        d5=kwargs["d5"][[0]], pitcher_counts=kwargs["pitcher_counts"],
        batter_counts=kwargs["batter_counts"], config=kwargs["config"], calibrator=kwargs["calibrator"],
    )

    np.testing.assert_allclose(alone, together[[0]], atol=1e-12)
