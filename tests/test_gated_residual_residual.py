from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.gated_residual_final.residual import (
    gated_residual,
    reliability,
    row_reliability,
    temporal_entity_counts,
)


def test_non_regular_row_backs_off_exactly_to_anchor() -> None:
    result = gated_residual(
        anchor=np.array([0.2]), direct=np.array([0.9]),
        row_reliability=np.array([0.0]), alpha=0.3,
    )

    np.testing.assert_allclose(result, [0.2], atol=1e-12)


def test_alpha_zero_backs_off_exactly_to_anchor() -> None:
    result = gated_residual(
        anchor=np.array([0.001, 0.9]), direct=np.array([0.8, 0.1]),
        row_reliability=np.array([1.0, 1.0]), alpha=0.0,
    )

    np.testing.assert_allclose(result, [0.001, 0.9], atol=1e-12)


def test_validation_counts_use_only_prior_years() -> None:
    frame = pd.DataFrame({
        "season": [2022, 2023, 2024],
        "pitcher_id": ["p1", "p1", "p_future"],
        "batter_id": ["b1", "b2", "b_future"],
    })

    counts = temporal_entity_counts(frame, validation_year=2024)

    assert counts.pitcher == {"p1": 2}
    assert counts.batter == {"b1": 1, "b2": 1}


def test_unknown_entity_has_zero_reliability() -> None:
    assert reliability(0, 100, k=25) == 0.0
    assert reliability(100, 0, k=25) == 0.0


def test_row_reliability_is_zero_outside_regular_season() -> None:
    frame = pd.DataFrame({
        "game_type": ["R", "F"], "pitcher_id": ["p", "p"], "batter_id": ["b", "b"],
    })
    result = row_reliability(frame, pitcher_counts={"p": 100}, batter_counts={"b": 100}, k=25)

    assert result[0] > 0
    assert result[1] == 0.0
