from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


class E3LabelError(ValueError):
    pass


_REQUIRED = (
    "season", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_reverse_rate", "control_success",
)


@dataclass(frozen=True)
class FailureTargets:
    frame: pd.DataFrame
    valid_mask: np.ndarray
    coverage: float
    binary_fraction: float
    success_agreement: float
    overlap_rate: float
    positive_counts: Mapping[str, int]


def _snap(values: np.ndarray, tolerance: float) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype="float64")
    output[np.isfinite(values) & (np.abs(values) <= tolerance)] = 0.0
    output[np.isfinite(values) & (np.abs(values - 1.0) <= tolerance)] = 1.0
    return output


def _working(rows: pd.DataFrame, valid_year: int) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame or rows.empty or rows.columns.has_duplicates:
        raise E3LabelError("training rows differ")
    missing = sorted(set(_REQUIRED).difference(rows.columns))
    if missing:
        raise E3LabelError(f"training rows are missing columns: {missing}")
    if type(valid_year) is not int or isinstance(valid_year, bool):
        raise E3LabelError("valid_year differs")
    output = rows.loc[:, _REQUIRED].copy(deep=True)
    output["source_position"] = np.arange(len(output), dtype="int64")
    if output["pitcher_id"].isna().any():
        raise E3LabelError("pitcher_id contains missing values")
    for column in ("season", *_REQUIRED[2:]):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if output["season"].isna().any() or output["season"].ge(valid_year).any():
        raise E3LabelError("training rows reach validation season")
    if not output["control_success"].isin((0, 1)).all():
        raise E3LabelError("control_success differs")
    return output.sort_values(
        ["pitcher_id", "asof_pitcher_n", "source_position"], kind="stable"
    ).reset_index(drop=True)


def recover_failure_targets(
    rows: pd.DataFrame,
    *,
    valid_year: int,
    tolerance: float,
) -> FailureTargets:
    if type(tolerance) not in {int, float} or isinstance(tolerance, bool) or not 0 <= float(tolerance) < 0.5:
        raise E3LabelError("tolerance differs")
    work = _working(rows, valid_year)
    duplicate = work.duplicated(["pitcher_id", "asof_pitcher_n"], keep=False)
    next_duplicate = duplicate.groupby(work["pitcher_id"], sort=False).shift(-1).fillna(True)
    n = work["asof_pitcher_n"].to_numpy(dtype="float64")
    next_n = work["asof_pitcher_n"].shift(-1).to_numpy(dtype="float64")
    same = work["pitcher_id"].eq(work["pitcher_id"].shift(-1)).to_numpy()
    linked = (
        same
        & np.isfinite(n)
        & np.isfinite(next_n)
        & np.isclose(next_n, n + 1.0)
        & ~duplicate.to_numpy()
        & ~next_duplicate.to_numpy(dtype=bool)
    )
    positions = np.flatnonzero(linked)
    snapped: dict[str, np.ndarray] = {}
    for name in ("success", "middle", "ball", "reverse"):
        column = f"asof_pitcher_{name}_rate"
        current = work[column].to_numpy(dtype="float64")[positions]
        successor = work[column].shift(-1).to_numpy(dtype="float64")[positions]
        delta = next_n[positions] * successor - n[positions] * current
        snapped[name] = _snap(delta, float(tolerance))
    binary = np.logical_and.reduce([np.isfinite(values) for values in snapped.values()])
    target = work["control_success"].to_numpy(dtype="int8")[positions]
    agreement = binary & (snapped["success"] == target)
    valid_sorted = np.zeros(len(work), dtype=bool)
    valid_sorted[positions[agreement]] = True
    output = pd.DataFrame(
        {
            "source_position": work["source_position"].to_numpy(dtype="int64"),
            "valid": valid_sorted,
            "success": np.nan,
            "middle": np.nan,
            "ball_result": np.nan,
            "wild": np.nan,
            "reverse": np.nan,
        }
    )
    accepted_positions = positions[agreement]
    success = snapped["success"][agreement]
    middle = snapped["middle"][agreement]
    ball = snapped["ball"][agreement]
    reverse = snapped["reverse"][agreement]
    wild = (1.0 - success) * (1.0 - middle) * (1.0 - reverse)
    for column, values in (
        ("success", success), ("middle", middle), ("ball_result", ball),
        ("wild", wild), ("reverse", reverse),
    ):
        output.loc[accepted_positions, column] = values
    output = output.sort_values("source_position", kind="stable", ignore_index=True)
    valid = output["valid"].to_numpy(dtype=bool, copy=True)
    failures = valid & output["success"].eq(0).to_numpy()
    overlap = failures & output["middle"].eq(1).to_numpy() & output["reverse"].eq(1).to_numpy()
    positive_counts = {
        name: int(output.loc[valid, name].sum()) for name in ("middle", "wild", "reverse")
    }
    valid.setflags(write=False)
    return FailureTargets(
        frame=output,
        valid_mask=valid,
        coverage=float(valid.mean()),
        binary_fraction=float(binary.mean()) if len(binary) else 0.0,
        success_agreement=float(agreement[binary].mean()) if binary.any() else 0.0,
        overlap_rate=float(overlap.sum() / failures.sum()) if failures.any() else 0.0,
        positive_counts=MappingProxyType(positive_counts),
    )
