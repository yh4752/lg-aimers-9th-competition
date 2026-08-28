from dataclasses import FrozenInstanceError
import json

import pytest

from experiments.tree_expert.hc_contracts import (
    HCContractError,
    build_hc_jobs,
    load_hc_contract,
)


def test_contract_matches_registered_experiment():
    contract = load_hc_contract()
    assert contract.campaign_id == "tree_hierarchical_residual_v1"
    assert contract.profiles["hc_balanced"].interaction_k == 800.0
    assert contract.calibration_alphas == (0.25, 0.5, 0.75, 1.0)
    assert contract.profile_tie_order == (
        "hc_strong",
        "hc_balanced",
        "hc_light",
    )
    assert contract.seed_ensemble == "arithmetic_probability_mean"
    assert contract.runtime.h1_wall_seconds == 21_600
    assert contract.submission_package is False
    with pytest.raises(FrozenInstanceError):
        contract.campaign_id = "changed"


def test_jobs_are_finite_unique_and_stage_registered():
    jobs = build_hc_jobs(load_hc_contract())
    assert {job.stage for job in jobs} == {"H1", "H2", "H3"}
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert all(job.seed in {42, 2026, 3407} for job in jobs)
    assert sum(job.kind == "source_tabm" for job in jobs) == 1
    assert sum(job.kind == "source_e2" for job in jobs) == 3
    assert sum(job.stage == "H1" and job.kind == "c1_residual" for job in jobs) == 9
    assert sum(job.stage == "H2" and job.kind == "c1_residual" for job in jobs) == 6
    assert sum(job.stage == "H2" and job.kind == "c2_calibration" for job in jobs) == 4
    assert sum(job.stage == "H3" and job.kind == "full_fit" for job in jobs) == 3
    assert sum(job.stage == "H3" and job.kind == "c2_state" for job in jobs) == 1


def test_contract_rejects_extra_or_changed_registry_value(tmp_path):
    original = json.loads(load_hc_contract().source_path.read_text(encoding="utf-8"))
    original["profiles"]["hc_balanced"]["interaction_k"] = 799.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(HCContractError, match="registered contract differs"):
        load_hc_contract(changed)


def test_contract_rejects_boolean_as_integer(tmp_path):
    original = json.loads(load_hc_contract().source_path.read_text(encoding="utf-8"))
    original["runtime"]["h1_wall_seconds"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(HCContractError, match="registered contract differs"):
        load_hc_contract(changed)
