from pathlib import Path

import pytest

from experiments.tree_privileged.contracts import PrivilegedContractError, load_contract


def test_contract_seals_budget_candidates_and_submission_boundary() -> None:
    contract = load_contract()
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.screen_seed == 3407
    assert contract.confirm_seeds == (42, 2026)
    assert contract.candidates == ("P",)
    assert contract.teacher_lambdas == (0.15, 0.35)
    assert contract.wall_seconds == 37_800
    assert contract.submission_package is False
    assert contract.maximum_confirmed_candidates == 2


def test_contract_rejects_any_local_change(tmp_path: Path) -> None:
    source = Path("experiments/tree_privileged/contract.json")
    changed = tmp_path / "contract.json"
    changed.write_text(source.read_text().replace('"0.00005"', '"0.00004"', 1))
    with pytest.raises(PrivilegedContractError, match="gates differ"):
        load_contract(changed)
