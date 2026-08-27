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
EXCLUSION_REASONS = (
    "no_successor",
    "duplicate_count",
    "skipped_count",
    "non_binary_delta",
    "success_disagreement",
    "middle_reverse_overlap",
    "subtype_on_success",
)


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


@dataclass(frozen=True)
class FailureLabelRecovery:
    rows: pd.DataFrame
    linked_count: int
    labeled_count: int
    coverage: float
    binary_delta_fraction: float
    success_agreement: float
    middle_reverse_overlap: float
    class_counts: Mapping[str, int]
    exclusion_counts: Mapping[str, int]


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


def _working_rows(rows: pd.DataFrame, valid_year: int | None) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame or rows.columns.has_duplicates:
        raise FailureLabelError("failure label rows must be a DataFrame with unique columns")
    missing = sorted(set(_REQUIRED).difference(rows.columns))
    if missing:
        raise FailureLabelError(f"failure label rows are missing columns: {missing}")
    if rows.empty:
        raise FailureLabelError("failure label rows are empty")
    if valid_year is not None and type(valid_year) is not int:
        raise FailureLabelError("valid_year must be an int")

    working = rows.loc[:, _REQUIRED].copy(deep=True)
    working["__position__"] = np.arange(len(working), dtype="int64")
    if working["pitcher_id"].isna().any():
        raise FailureLabelError("pitcher_id contains missing values")
    for column in ("season", *_REQUIRED[2:]):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    seasons = working["season"]
    if seasons.isna().any() or not np.equal(seasons, np.floor(seasons)).all():
        raise FailureLabelError("season values are invalid")
    if valid_year is not None and seasons.ge(valid_year).any():
        raise FailureLabelError("rows reach validation season")
    if not working["control_success"].isin([0, 1]).all():
        raise FailureLabelError("control_success must be binary")
    return working


def recover_failure_labels(
    rows: pd.DataFrame,
    *,
    delta_tolerance: float,
    valid_year: int | None = None,
) -> FailureLabelRecovery:
    if (
        type(delta_tolerance) not in {float, int}
        or type(delta_tolerance) is bool
        or not 0 <= float(delta_tolerance) < 0.5
    ):
        raise FailureLabelError("delta_tolerance must be in [0, 0.5)")
    working = _working_rows(rows, valid_year)
    working = working.sort_values(
        ["pitcher_id", "asof_pitcher_n", "__position__"], kind="stable"
    ).reset_index(drop=True)

    duplicate = working.duplicated(["pitcher_id", "asof_pitcher_n"], keep=False)
    next_duplicate = duplicate.groupby(working["pitcher_id"], sort=False).shift(-1)
    next_pitcher = working["pitcher_id"].shift(-1)
    n = working["asof_pitcher_n"]
    next_n = n.shift(-1)
    same_pitcher = working["pitcher_id"].eq(next_pitcher)
    consecutive = (
        same_pitcher
        & n.notna()
        & next_n.notna()
        & np.isclose(
            next_n.to_numpy(dtype="float64"),
            n.to_numpy(dtype="float64") + 1.0,
        )
    )
    linked = (
        consecutive
        & ~duplicate
        & ~next_duplicate.fillna(True).astype(bool)
    )

    status = np.full(len(working), "excluded", dtype=object)
    labels = np.full(len(working), "", dtype=object)
    reasons = np.full(len(working), "no_successor", dtype=object)
    reasons[same_pitcher.to_numpy() & ~consecutive.to_numpy()] = "skipped_count"
    duplicate_affected = duplicate | next_duplicate.fillna(False).astype(bool)
    reasons[duplicate_affected.to_numpy()] = "duplicate_count"

    selected = np.flatnonzero(linked.to_numpy())
    current_n = n.to_numpy(dtype="float64")[selected]
    successor_n = next_n.to_numpy(dtype="float64")[selected]
    deltas: dict[str, np.ndarray] = {}
    for component in ("success", "middle", "reverse"):
        column = f"asof_pitcher_{component}_rate"
        current = working[column].to_numpy(dtype="float64")[selected]
        successor = working[column].shift(-1).to_numpy(dtype="float64")[selected]
        deltas[component] = successor_n * successor - current_n * current

    snapped = {
        name: _snap(values, float(delta_tolerance)) for name, values in deltas.items()
    }
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
    overlap = failures & (snapped["middle"] == 1.0) & (snapped["reverse"] == 1.0)
    middle_reverse_overlap = float(overlap.sum() / failures.sum()) if failures.any() else 0.0
    subtype_on_success = (
        binary
        & target_is_success
        & ((snapped["middle"] == 1.0) | (snapped["reverse"] == 1.0))
    )
    usable = agreement & ~overlap & ~subtype_on_success

    selected_reasons = np.full(linked_count, "non_binary_delta", dtype=object)
    selected_reasons[binary & ~agreement] = "success_disagreement"
    selected_reasons[overlap] = "middle_reverse_overlap"
    selected_reasons[subtype_on_success] = "subtype_on_success"
    selected_reasons[usable] = ""
    reasons[selected] = selected_reasons

    selected_labels = np.full(linked_count, "", dtype=object)
    selected_labels[usable & target_is_success] = "success"
    selected_labels[usable & ~target_is_success & (snapped["middle"] == 1.0)] = "middle"
    selected_labels[usable & ~target_is_success & (snapped["reverse"] == 1.0)] = "reverse"
    selected_labels[
        usable
        & ~target_is_success
        & (snapped["middle"] == 0.0)
        & (snapped["reverse"] == 0.0)
    ] = "other_failure"
    labeled = usable & (selected_labels != "")
    labels[selected[labeled]] = selected_labels[labeled]
    status[selected[labeled]] = "labeled"

    ordered = pd.DataFrame(
        {
            "source_position": working["__position__"].to_numpy(dtype="int64"),
            "season": working["season"].to_numpy(dtype="int64"),
            "status": status,
            "label": labels,
            "exclusion_reason": reasons,
        }
    ).sort_values("source_position", kind="stable", ignore_index=True)
    labeled_count = int(labeled.sum())
    class_counts = {
        name: int(np.count_nonzero(selected_labels[labeled] == name)) for name in _CLASSES
    }
    exclusion_counts = {
        name: int(np.count_nonzero(reasons == name)) for name in EXCLUSION_REASONS
    }
    return FailureLabelRecovery(
        rows=ordered,
        linked_count=linked_count,
        labeled_count=labeled_count,
        coverage=float(labeled_count / len(working)),
        binary_delta_fraction=binary_delta_fraction,
        success_agreement=success_agreement,
        middle_reverse_overlap=middle_reverse_overlap,
        class_counts=MappingProxyType(class_counts),
        exclusion_counts=MappingProxyType(exclusion_counts),
    )


def audit_failure_labels(
    rows: pd.DataFrame,
    *,
    gate: Mapping[str, object],
    valid_year: int | None = None,
) -> FailureLabelAudit:
    gates = _gate_values(gate)
    recovery = recover_failure_labels(
        rows,
        valid_year=valid_year,
        delta_tolerance=float(gates["delta_tolerance"]),
    )

    failed: list[str] = []
    if recovery.coverage < float(gates["minimum_coverage"]):
        failed.append("coverage")
    if recovery.binary_delta_fraction < float(gates["minimum_binary_delta_fraction"]):
        failed.append("binary_delta_fraction")
    if recovery.success_agreement < float(gates["minimum_success_agreement"]):
        failed.append("success_agreement")
    if recovery.middle_reverse_overlap > float(gates["maximum_middle_reverse_overlap"]):
        failed.append("middle_reverse_overlap")
    if any(
        count < int(gates["minimum_class_rows"])
        for count in recovery.class_counts.values()
    ):
        failed.append("minimum_class_rows")

    if failed:
        empty_labels, empty_positions = _empty()
        return FailureLabelAudit(
            status="skipped_unreliable_labels",
            reason="gate_failed:" + ",".join(failed),
            labels=empty_labels,
            source_positions=empty_positions,
            coverage=recovery.coverage,
            binary_delta_fraction=recovery.binary_delta_fraction,
            success_agreement=recovery.success_agreement,
            middle_reverse_overlap=recovery.middle_reverse_overlap,
            class_counts=recovery.class_counts,
        )

    labeled_rows = recovery.rows.loc[recovery.rows["status"].eq("labeled")]
    output_labels = labeled_rows["label"].to_numpy(dtype=object, copy=True)
    output_positions = labeled_rows["source_position"].to_numpy(dtype="int64", copy=True)
    output_labels.setflags(write=False)
    output_positions.setflags(write=False)
    return FailureLabelAudit(
        status="passed",
        reason="all_gates_passed",
        labels=output_labels,
        source_positions=output_positions,
        coverage=recovery.coverage,
        binary_delta_fraction=recovery.binary_delta_fraction,
        success_agreement=recovery.success_agreement,
        middle_reverse_overlap=recovery.middle_reverse_overlap,
        class_counts=recovery.class_counts,
    )
