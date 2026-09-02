from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.failure_regime_e3.contracts import E3ContractError, load_contract


ROOT = Path(__file__).resolve().parents[1]


def test_registered_contract_keeps_aggressive_capacity_and_temporal_lock() -> None:
    contract = load_contract()

    assert contract.campaign_id == "failure_regime_e3_v1"
    assert contract.policy_version == "dacon-236743-2026-08-15"
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.selection_years == (2022, 2023)
    assert contract.confirmation_year == 2024
    assert contract.seeds == (42, 2026, 3407)
    assert tuple(role.role_id for role in contract.roles) == (
        "S_GLOBAL", "S_FAST", "S_R", "S_F", "MIDDLE", "WILD", "REVERSE",
    )
    assert contract.success_parameters["iterations"] == 2400
    assert contract.success_parameters["depth"] == 10
    assert contract.subtype_parameters["iterations"] == 1800
    assert contract.subtype_parameters["depth"] == 9
    assert contract.gate_parameters["depth"] == 3
    assert contract.gates["weighted_gain"] == pytest.approx(0.00025)
    assert contract.gates["latest_gain"] == pytest.approx(0.00015)
    assert contract.gates["bootstrap_lower"] == pytest.approx(0.00010)
    assert contract.inference_max_seconds == 480
    assert contract.runtime["wall_seconds"] == 36_000


def test_contract_rejects_unregistered_capacity(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "experiments/failure_regime_e3/contract.json").read_text())
    payload["model"]["success"]["depth"] = 8
    source = tmp_path / "contract.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(E3ContractError, match="registered contract differs"):
        load_contract(source)
