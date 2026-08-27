from dataclasses import FrozenInstanceError

import pytest

from experiments.tree_expert.t3_contracts import load_t3_contract, structure_jobs


def test_t3_contract_fixes_the_small_search_space():
    contract = load_t3_contract()
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.decays == (0.35, 0.55, 0.75)
    assert contract.recent_weights == (0.70, 0.80, 0.90)
    assert contract.structure_seed == 3407
    assert contract.confirmation_seeds == (42, 2026)
    assert contract.wall_seconds == 28_800
    with pytest.raises(FrozenInstanceError):
        contract.structure_seed = 42


def test_structure_jobs_have_unique_stable_ids():
    jobs = structure_jobs(load_t3_contract())
    assert len(jobs) == 12
    assert len({job.job_id for job in jobs}) == 12
    assert jobs[0].job_id == "t3__recent__tr2021__va2022__s3407"
    assert jobs[-1].job_id == "t3__multi_d075__tr2023__va2024__s3407"
