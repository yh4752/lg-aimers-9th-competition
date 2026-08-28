from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tree_expert.hetero_contracts import (
    HeteroContractError,
    confirmation_jobs,
    load_hetero_contract,
    structure_jobs,
)


def test_contract_has_fixed_search_and_runtime() -> None:
    contract = load_hetero_contract()

    assert contract.families == ("xgboost", "lightgbm")
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.correction_weights == (0.05, 0.10, 0.15)
    assert contract.structure_seed == 3407
    assert contract.confirmation_seeds == (42, 2026)
    assert contract.versions == {"xgboost": "3.0.2", "lightgbm": "4.6.0"}
    assert contract.gates.weighted_gain == 0.00005
    assert contract.gates.maximum_fold_regression == 0.00003
    assert contract.gates.maximum_segment_regression == 0.00030
    assert contract.gates.maximum_residual_correlation == 0.995
    assert contract.wall_seconds == 18_000


def test_job_registries_are_complete_and_unique() -> None:
    contract = load_hetero_contract()
    structure = structure_jobs(contract)
    confirmation = confirmation_jobs(contract, "xgboost")

    assert len(structure) == 6
    assert len(confirmation) == 6
    assert len({job.job_id for job in (*structure, *confirmation)}) == 12
    assert {job.family for job in structure} == {"xgboost", "lightgbm"}
    assert {job.seed for job in confirmation} == {42, 2026}


def test_contract_rejects_changed_weight_grid(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "experiments/tree_expert/hetero_contract.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["correction_weights"] = [0.05, 0.20]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HeteroContractError, match="search grid differs"):
        load_hetero_contract(changed)


def test_confirmation_rejects_unknown_family() -> None:
    with pytest.raises(HeteroContractError, match="family differs"):
        confirmation_jobs(load_hetero_contract(), "catboost")
