from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .features import (
    TreeFeatureBatch,
    TreeFeatureState,
    fit_tree_features,
    transform_tree_features,
)


class HeteroFeatureError(ValueError):
    pass


_MISSING = "__MISSING__"


@dataclass(frozen=True)
class HeteroFeatureState:
    tree_state: object
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_maps: Mapping[str, Mapping[str, int]]
    column_index: Mapping[str, int]


@dataclass(frozen=True)
class HeteroFeatureBatch:
    matrix: np.ndarray
    row_id: np.ndarray
    target: np.ndarray | None


def _category_values(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(_MISSING)


def _encode(batch: TreeFeatureBatch, state: HeteroFeatureState) -> HeteroFeatureBatch:
    frame = batch.frame
    if tuple(frame.columns) != state.feature_columns:
        raise HeteroFeatureError("feature schema differs")
    columns: list[np.ndarray] = []
    categories = set(state.categorical_columns)
    for column in state.feature_columns:
        if column in categories:
            mapping = state.category_maps[column]
            values = _category_values(frame[column]).map(mapping).fillna(-1)
            encoded = values.to_numpy(dtype="float32")
        else:
            encoded = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float32")
            if np.isinf(encoded).any():
                raise HeteroFeatureError(f"feature contains infinity: {column}")
        columns.append(encoded)
    matrix = np.column_stack(columns).astype("float32", copy=False)
    if len(matrix) != len(batch.row_id):
        raise HeteroFeatureError("feature row identity differs")
    matrix.setflags(write=False)
    row_id = np.asarray(batch.row_id, dtype=str).copy()
    row_id.setflags(write=False)
    target = None
    if batch.target is not None:
        target = np.asarray(batch.target, dtype="int8").copy()
        target.setflags(write=False)
    return HeteroFeatureBatch(matrix, row_id, target)


def fit_hetero_features(
    train: pd.DataFrame,
    *,
    valid_year: int,
) -> tuple[HeteroFeatureState, HeteroFeatureBatch]:
    tree_state, batch = fit_tree_features(
        train,
        None,
        valid_year=valid_year,
        use_trackman=False,
    )
    feature_columns = tuple(batch.frame.columns)
    categorical_columns = tuple(getattr(tree_state, "categorical_columns", ()))
    if not set(categorical_columns).issubset(feature_columns):
        raise HeteroFeatureError("categorical schema differs")
    category_maps: dict[str, Mapping[str, int]] = {}
    for column in categorical_columns:
        unique = sorted(set(_category_values(batch.frame[column]).tolist()))
        category_maps[column] = MappingProxyType({value: index for index, value in enumerate(unique)})
    state = HeteroFeatureState(
        tree_state=tree_state,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        category_maps=MappingProxyType(category_maps),
        column_index=MappingProxyType({column: index for index, column in enumerate(feature_columns)}),
    )
    encoded = _encode(batch, state)
    if encoded.target is None:
        raise HeteroFeatureError("training target is absent")
    return state, encoded


def transform_hetero_features(
    rows: pd.DataFrame,
    state: HeteroFeatureState,
) -> HeteroFeatureBatch:
    if type(state) is not HeteroFeatureState:
        raise HeteroFeatureError("hetero feature state type differs")
    batch = transform_tree_features(rows, state.tree_state)
    if batch.target is not None:
        raise HeteroFeatureError("evaluation target is present")
    return _encode(batch, state)
