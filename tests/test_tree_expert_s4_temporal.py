import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.s4_temporal import (
    S4TemporalError,
    align_probability_frames,
    blend_anchor,
    season_decay_weights,
    structure_fold_frames,
)


def test_decay_weights_use_only_training_seasons() -> None:
    years = np.array([2020, 2021, 2022, 2023])
    weights = season_decay_weights(years, cutoff_year=2023, decay=0.55)
    np.testing.assert_allclose(weights, [0.55**3, 0.55**2, 0.55, 1.0])


def test_future_training_season_is_rejected() -> None:
    with pytest.raises(S4TemporalError, match="exceed cutoff"):
        season_decay_weights([2023, 2024], cutoff_year=2023, decay=0.55)


def test_anchor_probability_matches_fixed_mixture() -> None:
    actual = blend_anchor(np.array([0.2, 0.8]), np.array([0.6, 0.4]), recent_weight=0.75)
    np.testing.assert_allclose(actual, [0.3, 0.7])


def test_structure_frames_never_read_confirmation_fold() -> None:
    frames = {
        (2021, 2022): pd.DataFrame({"row_id": ["a"], "probability": [0.5]}),
        (2022, 2023): pd.DataFrame({"row_id": ["b"], "probability": [0.5]}),
        (2023, 2024): pd.DataFrame({"row_id": ["c"], "probability": [0.5]}),
    }
    seen: list[tuple[int, int]] = []
    structure_fold_frames(frames, on_read=seen.append)
    assert seen == [(2021, 2022), (2022, 2023)]


def test_alignment_rejects_reordered_rows() -> None:
    left = pd.DataFrame({"row_id": ["a", "b"], "probability": [0.4, 0.6]})
    right = pd.DataFrame({"row_id": ["b", "a"], "probability": [0.6, 0.4]})
    with pytest.raises(S4TemporalError, match="row identity differs"):
        align_probability_frames(left, right)
