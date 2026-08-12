"""CatBoost feature contract ported from source commit 9454d68b93971627e3d3f613ce30be690cb5dce2."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from experiments.independent_dl.preprocessing import (
    PreprocessingSpec,
    PreprocessingState,
    fit_preprocessor,
    transform_preprocessor,
)


@dataclass(frozen=True)
class CatBoostFeatureState:
    preprocessing: PreprocessingState
    categorical_columns: tuple[str, ...]
    category_values: Mapping[str, tuple[str, ...]]


def _category_values(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType(
        {
            column: tuple(sorted(set(frame[column].astype(str))))
            for column in columns
        }
    )


def fit_catboost_features(
    train: pd.DataFrame, *, components: tuple[str, ...]
) -> tuple[CatBoostFeatureState, pd.DataFrame]:
    """Fit optional features on one training fold while preserving native NaNs."""

    preprocessing, features = fit_preprocessor(
        train, PreprocessingSpec("tree_native", tuple(components))
    )
    state = CatBoostFeatureState(
        preprocessing=preprocessing,
        categorical_columns=preprocessing.categorical_columns,
        category_values=_category_values(features, preprocessing.categorical_columns),
    )
    return state, features


def transform_catboost_features(
    frame: pd.DataFrame, state: CatBoostFeatureState
) -> pd.DataFrame:
    """Apply only the state learned from the corresponding training fold."""

    return transform_preprocessor(frame, state.preprocessing)

