from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from experiments.tree_expert.e2_contracts import (
    E2ContractError,
    build_structure_jobs,
    load_e2_contract,
)


CONTRACT_PATH = Path("experiments/tree_expert/e2_contract.json")


def test_e2_contract_seals_campaign_scope() -> None:
    contract = load_e2_contract()

    assert contract.campaign_id == "tree_expert_e2_v1"
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.seeds == (42, 2026, 3407)
    assert contract.structures == (
        "c1_anchor_residual",
        "c2_trackman_residual",
    )
    assert contract.review_only is False
    assert contract.submission_package is False
    assert contract.wall_seconds == 21_600
    with pytest.raises(FrozenInstanceError):
        contract.wall_seconds = 1


def test_structure_jobs_are_deterministic() -> None:
    jobs = build_structure_jobs(
        load_e2_contract(),
        seed=3407,
        folds=((2021, 2022),),
    )

    assert [job.job_id for job in jobs] == [
        "e2__c1_anchor_residual__tr2021__va2022__s3407",
        "e2__c2_trackman_residual__tr2021__va2022__s3407",
    ]
    assert [(job.objective, job.use_trackman) for job in jobs] == [
        ("residual", False),
        ("residual", True),
    ]


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("inputs", "e1_handoff_sha256", "0" * 64, "input hashes differ"),
        ("gates", "accept_weighted_gain", 0.5, "gates differ"),
        ("runtime", "wall_seconds", 1, "runtime differs"),
    ],
)
def test_e2_contract_rejects_changed_values(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    source = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source[section][key] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(E2ContractError, match=message):
        load_e2_contract(path)


def test_structure_job_builder_rejects_unregistered_fold_or_seed() -> None:
    contract = load_e2_contract()

    with pytest.raises(E2ContractError, match="seed is not registered"):
        build_structure_jobs(contract, seed=7, folds=((2021, 2022),))
    with pytest.raises(E2ContractError, match="fold is not registered"):
        build_structure_jobs(contract, seed=3407, folds=((2020, 2021),))
