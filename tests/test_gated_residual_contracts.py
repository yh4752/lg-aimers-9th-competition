from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.gated_residual_final.contracts import FinalContractError, load_contract
from experiments.gated_residual_final.runtime_inventory import runtime_members


def test_contract_locks_search_gates_and_runtime() -> None:
    contract = load_contract()

    assert contract.alpha_grid == (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    assert contract.k_grid == (25, 100, 400)
    assert contract.beta_grid == (0.05, 0.10, 0.20)
    assert contract.lambda_grid == (100, 500, 2000)
    assert contract.gates["weighted_gain"] == 0.00005
    assert contract.gates["latest_gain"] == 0.0
    assert contract.maximum_runtime_seconds == 28_800
    assert contract.full_fit_seeds == (42, 2026, 3407)


def test_unknown_contract_key_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(Path("experiments/gated_residual_final/contract.json").read_text())
    payload["surprise"] = True
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload))

    with pytest.raises(FinalContractError, match="contract keys differ"):
        load_contract(changed)


def test_runtime_inventory_contains_final_and_shared_dependencies() -> None:
    members = runtime_members(Path.cwd())

    assert "experiments/gated_residual_final/contracts.py" in members
    assert "experiments/direct_expert/features.py" in members
    assert "experiments/temporal_portfolio/trackman_pitcher.py" in members
    assert len(members) == len(set(members))
