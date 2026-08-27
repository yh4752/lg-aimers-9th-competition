from __future__ import annotations

from dataclasses import dataclass
import time
from types import MappingProxyType
from typing import Callable, Mapping

import pandas as pd

from .failure_audit_contracts import FailureAuditContract
from .failure_labels import (
    EXCLUSION_REASONS,
    FailureLabelRecovery,
    recover_failure_labels,
)


class FailureAuditError(ValueError):
    pass


_REQUIRED = {
    "row_id",
    "season",
    "pitcher_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate",
    "control_success",
}
_KNOWN_SEASONS = frozenset(range(2019, 2025))


@dataclass(frozen=True)
class CutoffAudit:
    audit_id: str
    cutoff_year: int
    status: str
    failed_gates: tuple[str, ...]
    maximum_source_season: int
    row_count: int
    linked_count: int
    labeled_count: int
    coverage: float
    binary_delta_fraction: float
    success_agreement: float
    middle_reverse_overlap: float
    class_counts: Mapping[str, int]
    exclusion_counts: Mapping[str, int]


@dataclass(frozen=True)
class TypeDecision:
    failure_type: str
    status: str
    reason: str
    positive_rows: Mapping[str, int]
    negative_rows: Mapping[str, int]


@dataclass(frozen=True)
class FailureAuditResult:
    cutoffs: Mapping[str, CutoffAudit]
    type_decisions: Mapping[str, TypeDecision]
    class_counts: pd.DataFrame
    exclusion_counts: pd.DataFrame


def _validated_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame or rows.columns.has_duplicates:
        raise FailureAuditError("audit rows must be a DataFrame with unique columns")
    missing = sorted(_REQUIRED.difference(rows.columns))
    if missing:
        raise FailureAuditError(f"audit rows are missing columns: {missing}")
    if rows.empty:
        raise FailureAuditError("audit rows are empty")
    if rows["row_id"].isna().any() or not rows["row_id"].is_unique:
        raise FailureAuditError("row_id must be unique and non-null")
    seasons = pd.to_numeric(rows["season"], errors="coerce")
    if seasons.isna().any() or not seasons.isin(_KNOWN_SEASONS).all():
        raise FailureAuditError("season values differ")
    target = pd.to_numeric(rows["control_success"], errors="coerce")
    if not target.isin([0, 1]).all():
        raise FailureAuditError("control_success must be binary")
    result = rows.copy(deep=True)
    result["season"] = seasons.astype("int64")
    result["control_success"] = target.astype("int8")
    return result


def _quality_failures(
    recovery: FailureLabelRecovery, contract: FailureAuditContract
) -> tuple[str, ...]:
    failed: list[str] = []
    if recovery.coverage < contract.minimum_coverage:
        failed.append("coverage")
    if recovery.binary_delta_fraction < contract.minimum_binary_delta_fraction:
        failed.append("binary_delta_fraction")
    if recovery.success_agreement < contract.minimum_success_agreement:
        failed.append("success_agreement")
    if recovery.middle_reverse_overlap > contract.maximum_middle_reverse_overlap:
        failed.append("middle_reverse_overlap")
    return tuple(failed)


def run_failure_label_audit(
    rows: pd.DataFrame,
    *,
    contract: FailureAuditContract,
    wall_deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] | None = None,
) -> FailureAuditResult:
    source = _validated_rows(rows)
    cutoff_results: dict[str, CutoffAudit] = {}
    class_records: list[dict[str, object]] = []
    exclusion_records: list[dict[str, object]] = []

    for audit_id, cutoff_year in contract.cutoffs:
        if wall_deadline is not None and clock() >= wall_deadline:
            raise FailureAuditError("wall deadline reached")
        if log is not None:
            log(f"FAIL_AUDIT_CUTOFF_START cutoff={audit_id}")
        prefix = source.loc[source["season"].le(cutoff_year)].copy(deep=True)
        if prefix.empty:
            raise FailureAuditError(f"audit prefix is empty: {audit_id}")
        recovery = recover_failure_labels(
            prefix,
            valid_year=cutoff_year + 1,
            delta_tolerance=contract.delta_tolerance,
        )
        failed = _quality_failures(recovery, contract)
        if wall_deadline is not None and clock() >= wall_deadline:
            raise FailureAuditError("wall deadline reached")
        cutoff_results[audit_id] = CutoffAudit(
            audit_id=audit_id,
            cutoff_year=cutoff_year,
            status="passed" if not failed else "failed",
            failed_gates=failed,
            maximum_source_season=int(prefix["season"].max()),
            row_count=len(prefix),
            linked_count=recovery.linked_count,
            labeled_count=recovery.labeled_count,
            coverage=recovery.coverage,
            binary_delta_fraction=recovery.binary_delta_fraction,
            success_agreement=recovery.success_agreement,
            middle_reverse_overlap=recovery.middle_reverse_overlap,
            class_counts=recovery.class_counts,
            exclusion_counts=recovery.exclusion_counts,
        )
        if log is not None:
            log(
                f"FAIL_AUDIT_CUTOFF_RESULT cutoff={audit_id} "
                f"status={'passed' if not failed else 'failed'}"
            )
        for season in sorted(recovery.rows["season"].unique()):
            season_rows = recovery.rows.loc[recovery.rows["season"].eq(season)]
            labeled = season_rows.loc[season_rows["status"].eq("labeled")]
            for label in ("success", *contract.types):
                class_records.append(
                    {
                        "audit_id": audit_id,
                        "cutoff_year": cutoff_year,
                        "source_season": int(season),
                        "label": label,
                        "count": int(labeled["label"].eq(label).sum()),
                    }
                )
            for reason in EXCLUSION_REASONS:
                exclusion_records.append(
                    {
                        "audit_id": audit_id,
                        "cutoff_year": cutoff_year,
                        "source_season": int(season),
                        "exclusion_reason": reason,
                        "count": int(season_rows["exclusion_reason"].eq(reason).sum()),
                    }
                )

    type_decisions: dict[str, TypeDecision] = {}
    common_failures = tuple(
        audit_id
        for audit_id, result in cutoff_results.items()
        if result.status != "passed"
    )
    for failure_type in contract.types:
        positive = {
            audit_id: result.class_counts[failure_type]
            for audit_id, result in cutoff_results.items()
        }
        negative = {
            audit_id: result.labeled_count - positive[audit_id]
            for audit_id, result in cutoff_results.items()
        }
        reasons: list[str] = []
        if common_failures:
            reasons.append("common_quality_gates:" + ",".join(common_failures))
        low_positive = tuple(
            name for name, count in positive.items() if count < contract.minimum_positive_rows
        )
        low_negative = tuple(
            name for name, count in negative.items() if count < contract.minimum_negative_rows
        )
        if low_positive:
            reasons.append("positive_rows:" + ",".join(low_positive))
        if low_negative:
            reasons.append("negative_rows:" + ",".join(low_negative))
        type_decisions[failure_type] = TypeDecision(
            failure_type=failure_type,
            status="ineligible" if reasons else "eligible",
            reason=";".join(reasons) if reasons else "all_gates_passed",
            positive_rows=MappingProxyType(positive),
            negative_rows=MappingProxyType(negative),
        )

    return FailureAuditResult(
        cutoffs=MappingProxyType(cutoff_results),
        type_decisions=MappingProxyType(type_decisions),
        class_counts=pd.DataFrame(class_records),
        exclusion_counts=pd.DataFrame(exclusion_records),
    )
