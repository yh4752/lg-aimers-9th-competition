from __future__ import annotations

from pathlib import Path

import pytest

from experiments.tree_expert.s4_recovery import (
    S4RecoveryError,
    load_recovery_contract,
    parse_recovery_contract,
)


SOURCE_SHA = "c8becb4037e511ff5d28d0fd146e1a9ec73923125cb4fb9c4467ba3cab6f8e9d"
PREDECESSOR_SHA = "5e73e15269723679d421f88ddbb04141941b5760a9d7257c550b42183acd35c8"


def _contract_payload() -> dict[str, object]:
    contract = load_recovery_contract()
    return {
        "schema_version": contract.schema_version,
        "artifact_kind": contract.artifact_kind,
        "source_handoff_sha256": contract.source_handoff_sha256,
        "predecessor_code_sha256": contract.predecessor_code_sha256,
        "s4_contract_sha256": contract.s4_contract_sha256,
        "source_bindings": dict(contract.source_bindings),
    }


def test_recovery_contract_seals_source_and_predecessor() -> None:
    contract = load_recovery_contract()
    assert contract.source_handoff_sha256 == SOURCE_SHA
    assert contract.predecessor_code_sha256 == PREDECESSOR_SHA
    assert contract.source_bindings["code_sha256"] == PREDECESSOR_SHA


def test_recovery_contract_rejects_unknown_keys() -> None:
    payload = _contract_payload()
    payload["unexpected"] = True
    with pytest.raises(S4RecoveryError, match="contract keys differ"):
        parse_recovery_contract(payload)


def test_recovery_contract_rejects_malformed_sha() -> None:
    payload = _contract_payload()
    payload["source_handoff_sha256"] = "A" * 64
    with pytest.raises(S4RecoveryError, match="SHA-256 differs"):
        parse_recovery_contract(payload)
