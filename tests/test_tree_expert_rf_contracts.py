from __future__ import annotations

from pathlib import Path

import pytest

from experiments.tree_expert.rf_contracts import (
    RFContractError,
    confirmation_jobs,
    contract_sha256,
    load_rf_contract,
    structure_jobs,
)


def test_rf_contract_has_fixed_search_and_budget() -> None:
    contract = load_rf_contract()

    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.structure_seed == 3407
    assert contract.confirmation_seeds == (42, 2026)
    assert contract.alpha_values == (0.25, 0.50, 0.75, 1.00)
    assert contract.wall_seconds == 19_800
    assert contract.full_fit_guard_seconds == 1_800
    assert contract.probability_tolerance == 1e-6


def test_structure_grid_has_three_heads_across_three_folds() -> None:
    jobs = structure_jobs(load_rf_contract())

    assert len(jobs) == 9
    assert {job.head for job in jobs} == {"f_small", "f_wide", "r_expert"}
    assert len({job.job_id for job in jobs}) == 9


def test_confirmation_grid_only_uses_selected_heads() -> None:
    jobs = confirmation_jobs(load_rf_contract(), f_head="f_small", include_r=True)

    assert len(jobs) == 12
    assert {job.seed for job in jobs} == {42, 2026}
    assert {job.head for job in jobs} == {"f_small", "r_expert"}


def test_confirmation_grid_can_skip_r_head() -> None:
    jobs = confirmation_jobs(load_rf_contract(), f_head="f_wide", include_r=False)

    assert len(jobs) == 6
    assert {job.head for job in jobs} == {"f_wide"}


def test_confirmation_rejects_unknown_head() -> None:
    with pytest.raises(RFContractError, match="F head differs"):
        confirmation_jobs(load_rf_contract(), f_head="other", include_r=True)


def test_contract_parser_rejects_changed_grid(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "experiments/tree_expert/rf_contract.json"
    changed = source.read_text(encoding="utf-8").replace(
        '"alpha_values": [0.25, 0.5, 0.75, 1.0]',
        '"alpha_values": [0.2, 0.5, 0.75, 1.0]',
    )
    path = tmp_path / "contract.json"
    path.write_text(changed, encoding="utf-8")

    with pytest.raises(RFContractError, match="search grid differs"):
        load_rf_contract(path)


def test_contract_hash_is_sha256() -> None:
    assert len(contract_sha256()) == 64
