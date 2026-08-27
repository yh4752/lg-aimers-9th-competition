import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.t3_temporal import (
    T3TemporalError,
    temporal_training_weights,
)


def test_recent_head_uses_only_the_immediate_previous_season():
    seasons = pd.Series([2019, 2021, 2022, 2023, 2023])
    weights = temporal_training_weights(seasons, valid_year=2024, head="recent")
    np.testing.assert_array_equal(weights, [0.0, 0.0, 0.0, 1.0, 1.0])
    assert not weights.flags.writeable


def test_multi_head_applies_decay_from_the_immediate_previous_season():
    seasons = pd.Series([2021, 2022, 2023])
    weights = temporal_training_weights(
        seasons, valid_year=2024, head="multi", decay=0.55,
    )
    np.testing.assert_allclose(weights, [0.55**2, 0.55, 1.0])


@pytest.mark.parametrize(
    ("seasons", "head", "decay", "message"),
    [
        ([2023, 2024], "recent", None, "must precede validation"),
        ([2022, 2023], "multi", 0.50, "head or decay differs"),
        ([2021, 2022], "recent", None, "previous season is missing"),
        ([2022.5, 2023], "multi", 0.55, "finite integers"),
    ],
)
def test_temporal_weights_reject_invalid_inputs(seasons, head, decay, message):
    with pytest.raises(T3TemporalError, match=message):
        temporal_training_weights(
            pd.Series(seasons), valid_year=2024, head=head, decay=decay,
        )
