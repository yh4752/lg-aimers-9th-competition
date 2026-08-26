from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


class FailureLabelError(ValueError):
    """Raised when failure labels cannot be audited from official training rows."""


_REQUIRED = (
    "season",
    "pitcher_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate",
    "control_success",
)
_GATE_KEYS = {
    "minimum_coverage",
    "minimum_binary_delta_fraction",
    "minimum_success_agreement",
    "maximum_middle_reverse_overlap",
    "minimum_class_rows",
    "delta_tolerance",
}
_CLASSES = ("success", "middle", "reverse", "other_failure")


@dataclass(frozen=True)
class FailureLabelAudit:
    status: str
    reason: str
    labels: np.ndarray
    source_positions: np.ndarray
    coverage: float
    binary_delta_fraction: float
    success_agreement: float
    middle_reverse_overlap: float
    class_counts: Mapping[str, int]


def _gate_values(gate: Mapping[str, object]) -> dict[str, float | int]:
    if not isinstance(gate, Mapping) or set(gate) != _GATE_KEYS:
        raise FailureLabelError("failure label gate keys differ")
    values = dict(gate)
    for name in (
        "minimum_coverage",
        "minimum_binary_delta_fraction",
        "minimum_success_agreement",
        "maximum_middle_reverse_overlap",
    ):
        value = values[name]
        if type(value) not in {float, int} or type(value) is bool or not 0 <= float(value) <= 1:
            raise FailureLabelError(f"{name} must be between zero and one")
    minimum_rows = values["minimum_class_rows"]
    if type(minimum_rows) is not int or minimum_rows < 1:
        raise FailureLabelError("minimum_class_rows must be a positive int")
    tolerance = values["delta_tolerance"]
    if type(tolerance) not in {float, int} or type(tolerance) is bool or not 0 <= float(tolerance) < 0.5:
        raise FailureLabelError("delta_tolerance must be in [0, 0.5)")
    return values


def _empty() -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([], dtype=object)
    positions = np.asarray([], dtype="int64")
    labels.setflags(write=False)
    positions.setflags(write=False)
    return labels, positions


def _snap(values: np.ndarray, tolerance: float) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype="float64")
    zero = np.isfinite(values) & (np.abs(values) <= tolerance)
    one = np.isfinite(values) & (np.abs(values - 1.0) <= tolerance)
    output[zero] = 0.0
    output[one] = 1.0
    return output


def audit_failure_labels(
    rows: pd.DataFrame,
    *,
    gate: Mapping[str, object],
    valid_year: int | None = None,
) -> FailureLabelAudit:
    if type(rows) is not pd.DataFrame or rows.columns.has_duplicates:
        raise FailureLabelError("failure label rows must be a DataFrame with unique columns")
    missing = sorted(set(_REQUIRED).difference(rows.columns))
    if missing:
        raise FailureLabelError(f"failure label rows are missing columns: {missing}")
    if rows.empty:
        raise FailureLabelError("failure label rows are empty")
    gates = _gate_values(gate)
    if valid_year is not None:
        if type(valid_year) is not int:
            raise FailureLabelError("valid_year must be an int")
        seasons = pd.to_numeric(rows["season"], errors="coerce")
        if seasons.isna().any() or not np.equal(seasons, np.floor(seasons)).all():
            raise FailureLabelError("season values are invalid")
        if seasons.ge(valid_year).any():
            raise FailureLabelError("rows reach validation season")

    working = rows.loc[:, _REQUIRED].copy(deep=True)
    working["__position__"] = np.arange(len(working), dtype="int64")
    if working["pitcher_id"].isna().any():
        raise FailureLabelError("pitcher_id contains missing values")
    numeric_columns = _REQUIRED[2:]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    target = working["control_success"]
    if not target.isin([0, 1]).all():
        raise FailureLabelError("control_success must be binary")

    working = working.sort_values(
        ["pitcher_id", "asof_pitcher_n", "__position__"], kind="stable"
    ).reset_index(drop=True)
    duplicate_count = working.duplicated(
        ["pitcher_id", "asof_pitcher_n"], keep=False
    )
    next_duplicate = duplicate_count.groupby(working["pitcher_id"], sort=False).shift(-1)
    next_pitcher = working["pitcher_id"].shift(-1)
    next_n = working["asof_pitcher_n"].shift(-1)
    n = working["asof_pitcher_n"]
    same_pitcher = working["pitcher_id"].eq(next_pitcher)
    linked = (
        same_pitcher
        & n.notna()
        & next_n.notna()
        & np.isclose(next_n.to_numpy(dtype="float64"), n.to_numpy(dtype="float64") + 1.0)
        & ~duplicate_count
        & ~next_duplicate.fillna(True).astype(bool)
    )

    selected = np.flatnonzero(linked.to_numpy())
    current_n = n.to_numpy(dtype="float64")[selected]
    successor_n = next_n.to_numpy(dtype="float64")[selected]
    deltas: dict[str, np.ndarray] = {}
    for component in ("success", "middle", "reverse"):
        column = f"asof_pitcher_{component}_rate"
        current = working[column].to_numpy(dtype="float64")[selected]
        successor = working[column].shift(-1).to_numpy(dtype="float64")[selected]
        deltas[component] = successor_n * successor - current_n * current

    tolerance = float(gates["delta_tolerance"])
    snapped = {name: _snap(values, tolerance) for name, values in deltas.items()}
    binary = np.logical_and.reduce([np.isfinite(values) for values in snapped.values()])
    linked_count = len(selected)
    binary_delta_fraction = float(binary.mean()) if linked_count else 0.0

    linked_target = working["control_success"].to_numpy(dtype="int8")[selected]
    agreement = binary & (snapped["success"] == linked_target)
    success_agreement = (
        float((snapped["success"][binary] == linked_target[binary]).mean())
        if binary.any()
        else 0.0
    )
    target_is_success = linked_target.astype(bool)
    failures = binary & ~target_is_success
    overlap = failures & snapped["middle"].astype(bool) & snapped["reverse"].astype(bool)
    middle_reverse_overlap = (
        float(overlap.sum() / failures.sum()) if failures.any() else 0.0
    )
    subtype_on_success = (
        binary
        & target_is_success
        & ((snapped["middle"] == 1.0) | (snapped["reverse"] == 1.0))
    )
    usable = agreement & ~overlap & ~subtype_on_success

    labels = np.full(linked_count, "", dtype=object)
    labels[usable & target_is_success] = "success"
    labels[usable & ~target_is_success & (snapped["middle"] == 1.0)] = "middle"
    labels[usable & ~target_is_success & (snapped["reverse"] == 1.0)] = "reverse"
    labels[
        usable
        & ~target_is_success
        & (snapped["middle"] == 0.0)
        & (snapped["reverse"] == 0.0)
    ] = "other_failure"
    labeled = usable & (labels != "")
    coverage = float(labeled.sum() / len(working))
    class_counts = {name: int(np.count_nonzero(labels[labeled] == name)) for name in _CLASSES}

    failed: list[str] = []
    if coverage < float(gates["minimum_coverage"]):
        failed.append("coverage")
    if binary_delta_fraction < float(gates["minimum_binary_delta_fraction"]):
        failed.append("binary_delta_fraction")
    if success_agreement < float(gates["minimum_success_agreement"]):
        failed.append("success_agreement")
    if middle_reverse_overlap > float(gates["maximum_middle_reverse_overlap"]):
        failed.append("middle_reverse_overlap")
    if any(count < int(gates["minimum_class_rows"]) for count in class_counts.values()):
        failed.append("minimum_class_rows")

    if failed:
        empty_labels, empty_positions = _empty()
        return FailureLabelAudit(
            status="skipped_unreliable_labels",
            reason="gate_failed:" + ",".join(failed),
            labels=empty_labels,
            source_positions=empty_positions,
            coverage=coverage,
            binary_delta_fraction=binary_delta_fraction,
            success_agreement=success_agreement,
            middle_reverse_overlap=middle_reverse_overlap,
            class_counts=MappingProxyType(class_counts),
        )

    output_labels = labels[labeled].copy()
    output_positions = working["__position__"].to_numpy(dtype="int64")[selected][labeled].copy()
    order = np.argsort(output_positions, kind="stable")
    output_labels = output_labels[order]
    output_positions = output_positions[order]
    output_labels.setflags(write=False)
    output_positions.setflags(write=False)
    return FailureLabelAudit(
        status="passed",
        reason="all_gates_passed",
        labels=output_labels,
        source_positions=output_positions,
        coverage=coverage,
        binary_delta_fraction=binary_delta_fraction,
        success_agreement=success_agreement,
        middle_reverse_overlap=middle_reverse_overlap,
        class_counts=MappingProxyType(class_counts),
    )
