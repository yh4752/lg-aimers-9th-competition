from dataclasses import replace

import pytest

from experiments.tree_expert.s4_contracts import (
    S4ContractError,
    anchor_specs,
    load_s4_contract,
    residual_specs,
)


def test_contract_seals_aggressive_search_and_runtime() -> None:
    contract = load_s4_contract()
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.recent_weights == (0.60, 0.75, 0.90)
    assert contract.decays == (0.30, 0.55, 0.75)
    assert contract.residual_alphas == (0.25, 0.50, 0.75, 1.00)
    assert contract.calibration_betas == (0.10, 0.25, 0.50, 0.75)
    assert contract.minimum_full_chains == 12
    assert contract.runtime.wall_seconds == 43200
    assert contract.runtime.artifact_reserve_seconds >= 2700


def test_anchor_and_residual_registries_are_unique_and_cover_external_template() -> None:
    contract = load_s4_contract()
    anchors = anchor_specs(contract)
    residuals = residual_specs(contract)
    assert len({item.candidate_id for item in anchors}) == len(anchors)
    assert any(item.recent_weight == 0.75 and item.decay == 0.55 for item in anchors)
    assert {item.family for item in residuals} == {
        "catboost", "catboost_rf", "xgboost", "lightgbm", "dual_temporal",
    }


def test_minimum_full_chain_change_is_rejected() -> None:
    contract = load_s4_contract()
    bad = replace(contract, minimum_full_chains=11)
    with pytest.raises(S4ContractError, match="minimum_full_chains"):
        anchor_specs(bad)
