from __future__ import annotations

from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from experiments.tree_expert.failure_audit import (
    CutoffAudit,
    FailureAuditResult,
    TypeDecision,
)
from experiments.tree_expert.failure_audit_artifacts import (
    FailureAuditArtifactError,
    FailureAuditBindings,
    create_failure_audit_review,
    verify_failure_audit_review,
)


BINDINGS = FailureAuditBindings("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _result() -> FailureAuditResult:
    cutoffs = {
        f"A{index}": CutoffAudit(
            audit_id=f"A{index}",
            cutoff_year=2020 + index,
            status="passed",
            failed_gates=(),
            maximum_source_season=2020 + index,
            row_count=100,
            linked_count=99,
            labeled_count=99,
            coverage=0.99,
            binary_delta_fraction=1.0,
            success_agreement=1.0,
            middle_reverse_overlap=0.0,
            class_counts=MappingProxyType(
                {"success": 60, "middle": 15, "reverse": 12, "other_failure": 12}
            ),
            exclusion_counts=MappingProxyType({"no_successor": 1}),
        )
        for index in range(1, 5)
    }
    decisions = {
        name: TypeDecision(
            failure_type=name,
            status="eligible",
            reason="all_gates_passed",
            positive_rows=MappingProxyType({f"A{i}": 12 for i in range(1, 5)}),
            negative_rows=MappingProxyType({f"A{i}": 87 for i in range(1, 5)}),
        )
        for name in ("middle", "reverse", "other_failure")
    }
    return FailureAuditResult(
        cutoffs=MappingProxyType(cutoffs),
        type_decisions=MappingProxyType(decisions),
        class_counts=pd.DataFrame(
            [{"audit_id": "A1", "source_season": 2021, "label": "middle", "count": 15}]
        ),
        exclusion_counts=pd.DataFrame(
            [{"audit_id": "A1", "source_season": 2021, "exclusion_reason": "no_successor", "count": 1}]
        ),
    )


def _rewrite_member(source, destination, name: str, payload: bytes):
    with ZipFile(source, "r") as archive, ZipFile(
        destination, "w", compression=ZIP_DEFLATED
    ) as output:
        for info in archive.infolist():
            output.writestr(info, payload if info.filename == name else archive.read(info))
    return destination


def test_review_is_deterministic_and_contains_only_aggregate_evidence(tmp_path) -> None:
    first = create_failure_audit_review(_result(), tmp_path / "first.zip", BINDINGS)
    second = create_failure_audit_review(_result(), tmp_path / "second.zip", BINDINGS)

    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        assert set(archive.namelist()) == {
            "audit_summary.json",
            "class_counts.csv",
            "exclusion_counts.csv",
            "cutoffs/A1.json",
            "cutoffs/A2.json",
            "cutoffs/A3.json",
            "cutoffs/A4.json",
            "audit.log",
            "manifest.json",
        }
        assert "row_id" not in archive.read("class_counts.csv").decode()

    verified = verify_failure_audit_review(first, BINDINGS)
    assert verified.archive_sha256


def test_review_rejects_changed_member(tmp_path) -> None:
    review = create_failure_audit_review(_result(), tmp_path / "review.zip", BINDINGS)
    tampered = _rewrite_member(
        review, tmp_path / "tampered.zip", "cutoffs/A2.json", b"{}"
    )

    with pytest.raises(FailureAuditArtifactError, match="member evidence differs"):
        verify_failure_audit_review(tampered, BINDINGS)
