from __future__ import annotations

import json

import pandas as pd
import pytest

from experiments.catboost_deployment.state import (
    DeploymentStateError,
    deserialize_feature_state,
    serialize_feature_state,
)
from experiments.catboost_preprocessing.features import (
    fit_catboost_features,
    transform_catboost_features,
)


def test_preprocessing_state_round_trip_is_exact(
    preprocessing_frame: pd.DataFrame,
) -> None:
    fitted, expected = fit_catboost_features(
        preprocessing_frame, components=("hand_matchup",)
    )

    payload = serialize_feature_state(fitted)
    restored = deserialize_feature_state(payload)
    actual = transform_catboost_features(preprocessing_frame, restored)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert restored.category_values == fitted.category_values
    assert serialize_feature_state(restored) == payload


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value.pop("preprocessing"),
        lambda value: value["preprocessing"].update(target_prior=float("nan")),
        lambda value: value["preprocessing"].update(
            source_columns=list(reversed(value["preprocessing"]["source_columns"]))
        ),
        lambda value: value.update(categorical_columns=["unknown"]),
        lambda value: value["category_values"].update(pitcher_id=["changed"]),
        lambda value: value["preprocessing"]["spec"].update(profile="unknown"),
    ],
)
def test_rejects_tampered_preprocessing_state(
    preprocessing_frame: pd.DataFrame, mutate
) -> None:
    fitted, _ = fit_catboost_features(
        preprocessing_frame, components=("hand_matchup",)
    )
    value = json.loads(serialize_feature_state(fitted))
    mutate(value)

    with pytest.raises(DeploymentStateError):
        deserialize_feature_state(
            json.dumps(value, allow_nan=True, sort_keys=True).encode()
        )
