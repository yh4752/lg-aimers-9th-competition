from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
from types import MappingProxyType

import pytest

from experiments.tree_expert.s4_recovery import (
    S4RecoveryContract,
    S4RecoveryError,
    compact_recovery_handoff,
    load_recovery_contract,
    parse_recovery_contract,
    verify_recovery_input,
)
from experiments.tree_expert.s4_artifacts import S4Bindings, create_s4_handoff, file_sha256
from experiments.tree_expert.s4_state import S4State, save_s4_state


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


def _source_fixture(
    tmp_path: Path, *, omit: str | None = None
) -> tuple[Path, S4RecoveryContract]:
    bindings = S4Bindings(*("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, "f" * 64))
    root = tmp_path / "campaign"
    completed = tuple(f"full_chains__{index:02d}" for index in range(14))
    save_s4_state(
        S4State("full_chains", completed, (), MappingProxyType({})),
        root / "state/state.json",
    )
    payloads = {
        "diagnostics/campaign.json": b"{}",
        "decisions/anchor_coverage.json": b"{}",
        "anchors/e2/2021.csv": b"row_id,p_anchor\n1,0.5\n",
        "residual_predictions/c00/s3407/2022.csv": b"row_id,raw_correction\n1,0.1\n",
        "anchor_basis/2021_2022/recent/predictions.csv": b"discard",
        "verified_e2/fold_2021_2022.csv": b"discard",
        "verified_e2_input.zip": b"discard",
        "jobs/full_chains__00/model.cbm": b"discard-model",
        "jobs/full_chains__00/result.json": b"{}",
        "s4_campaign.log": b"ok\n",
    }
    for name, payload in payloads.items():
        if name == omit:
            continue
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for index in range(14):
        base = root / "full_chains" / f"c{index:02d}"
        base.mkdir(parents=True, exist_ok=True)
        (base / "config.json").write_text("{}", encoding="utf-8")
        for year in (2022, 2023, 2024):
            if f"full_chains/c{index:02d}/{year}.csv" == omit:
                continue
            (base / f"{year}.csv").write_text("row_id,p_candidate\n1,0.5\n", encoding="utf-8")
    source = create_s4_handoff(root, tmp_path / "source_handoff.zip", bindings)
    contract = S4RecoveryContract(
        schema_version=1,
        artifact_kind="tree_s4_recovery_contract_v1",
        source_handoff_sha256=file_sha256(source),
        predecessor_code_sha256=bindings.code_sha256,
        s4_contract_sha256=bindings.contract_sha256,
        source_bindings=MappingProxyType(asdict(bindings)),
    )
    return source, contract


def test_compaction_drops_reproducible_payload_and_rebinds(tmp_path: Path) -> None:
    source, contract = _source_fixture(tmp_path)
    output = tmp_path / "recovery_input.zip"
    result = compact_recovery_handoff(
        source,
        output,
        destination_code_sha256="9" * 64,
        contract=contract,
    )
    verified = verify_recovery_input(
        output,
        expected_code_sha256="9" * 64,
        contract=contract,
    )
    assert result.path == output
    assert result.dropped_bytes > 0
    assert verified.state_phase == "full_chains"
    assert verified.completed_full_chains == tuple(range(14))
    assert not any(
        name.startswith(("jobs/", "anchor_basis/", "verified_e2/"))
        for name in verified.resume_members
    )
    assert "verified_e2_input.zip" not in verified.resume_members
    assert "full_chains/c13/config.json" in verified.resume_members


def test_compaction_rejects_unknown_source_identity(tmp_path: Path) -> None:
    source, contract = _source_fixture(tmp_path)
    changed = replace(contract, source_handoff_sha256="0" * 64)
    with pytest.raises(S4RecoveryError, match="source handoff SHA-256 differs"):
        compact_recovery_handoff(
            source,
            tmp_path / "recovery.zip",
            destination_code_sha256="9" * 64,
            contract=changed,
        )


def test_compaction_requires_every_completed_full_chain_output(tmp_path: Path) -> None:
    source, contract = _source_fixture(
        tmp_path, omit="full_chains/c13/2024.csv"
    )
    with pytest.raises(S4RecoveryError, match="completed full-chain output is absent"):
        compact_recovery_handoff(
            source,
            tmp_path / "recovery.zip",
            destination_code_sha256="9" * 64,
            contract=contract,
        )
