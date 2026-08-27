from __future__ import annotations

from pathlib import Path

import pytest

from experiments.tree_expert.failure_audit_contracts import (
    FailureAuditContractError,
    contract_sha256,
    load_failure_audit_contract,
)


def test_contract_fixes_cutoffs_and_gates() -> None:
    contract = load_failure_audit_contract()

    assert contract.cutoffs == (
        ("A1", 2021),
        ("A2", 2022),
        ("A3", 2023),
        ("A4", 2024),
    )
    assert contract.delta_tolerance == 0.02
    assert contract.minimum_coverage == 0.98
    assert contract.minimum_binary_delta_fraction == 0.999
    assert contract.minimum_success_agreement == 0.999
    assert contract.maximum_middle_reverse_overlap == 0.001
    assert contract.minimum_positive_rows == 5_000
    assert contract.minimum_negative_rows == 5_000
    assert contract.types == ("middle", "reverse", "other_failure")


def test_contract_rejects_changed_tolerance(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "experiments/tree_expert/failure_audit_contract.json"
    )
    changed = source.read_text(encoding="utf-8").replace(
        '"delta_tolerance": 0.02', '"delta_tolerance": 0.03'
    )
    path = tmp_path / "contract.json"
    path.write_text(changed, encoding="utf-8")

    with pytest.raises(FailureAuditContractError, match="fixed audit contract differs"):
        load_failure_audit_contract(path)


def test_contract_hash_is_sha256() -> None:
    assert len(contract_sha256()) == 64
