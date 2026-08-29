from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .hc_contracts import load_hc_contract
from .hc_features import (
    HCFeatureError,
    HierarchyState,
    fit_hierarchy,
    hierarchy_state_from_payload,
    hierarchy_state_payload,
    transform_hierarchy,
)


class S4FeatureError(ValueError):
    pass


@dataclass(frozen=True)
class S4ContextState:
    cutoff_year: int
    hierarchy: HierarchyState


def fit_s4_context(rows: pd.DataFrame, *, cutoff_year: int) -> S4ContextState:
    contract = load_hc_contract()
    try:
        state = fit_hierarchy(
            rows,
            cutoff_year=cutoff_year,
            profile_name="hc_strong",
            profile=contract.profiles["hc_strong"],
            minimum_group_rows=contract.minimum_group_rows,
        )
    except HCFeatureError as error:
        raise S4FeatureError(str(error)) from error
    return S4ContextState(int(cutoff_year), state)


def transform_s4_context(rows: pd.DataFrame, state: S4ContextState) -> pd.DataFrame:
    if type(state) is not S4ContextState:
        raise S4FeatureError("S4 context state differs")
    try:
        output = transform_hierarchy(rows, state.hierarchy)
    except HCFeatureError as error:
        raise S4FeatureError(str(error)) from error
    frame = rows.loc[:, ["row_id"]].reset_index(drop=True).copy(deep=True)
    return pd.concat([frame, output.reset_index(drop=True)], axis=1)


def context_state_payload(state: S4ContextState) -> dict[str, object]:
    if type(state) is not S4ContextState:
        raise S4FeatureError("S4 context state differs")
    return {
        "schema_version": 1,
        "cutoff_year": state.cutoff_year,
        "hierarchy": hierarchy_state_payload(state.hierarchy),
    }


def context_state_from_payload(payload: Mapping[str, object]) -> S4ContextState:
    if type(payload) is not dict or set(payload) != {"schema_version", "cutoff_year", "hierarchy"}:
        raise S4FeatureError("S4 context payload differs")
    cutoff = payload["cutoff_year"]
    if payload["schema_version"] != 1 or type(cutoff) is not int or type(cutoff) is bool:
        raise S4FeatureError("S4 context identity differs")
    try:
        hierarchy = hierarchy_state_from_payload(payload["hierarchy"])
    except HCFeatureError as error:
        raise S4FeatureError(str(error)) from error
    if hierarchy.cutoff_year != cutoff:
        raise S4FeatureError("S4 context cutoff differs")
    return S4ContextState(cutoff, hierarchy)
