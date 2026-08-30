from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.direct_expert.contracts import (
    DirectExpertContractError,
    expert_spec,
    expert_specs,
    load_contract,
    screening_jobs,
)


def test_contract_seals_deep_direct_campaign() -> None:
    contract = load_contract()

    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.screening_seed == 3407
    assert contract.confirmation_seeds == (42, 2026)
    assert contract.screening_parameters["iterations"] == 1800
    assert contract.screening_parameters["depth"] == 9
    assert contract.final_parameters["iterations"] == 2400
    assert contract.final_parameters["depth"] == 10
    assert contract.maximum_deployed_models == 9
    assert contract.runtime["full_fit_guard_seconds"] == 7200


def test_exact_eight_experts_and_sixteen_stage_a_jobs() -> None:
    specs = expert_specs(load_contract())
    assert tuple(item.expert_id for item in specs) == tuple(f"D{i}" for i in range(8))
    assert expert_spec("D6").game_type == "F"

    jobs = screening_jobs(load_contract())
    assert len(jobs) == 16
    assert len({job.job_id for job in jobs}) == 16
    assert {job.fold for job in jobs} == {(2021, 2022), (2022, 2023)}


def test_changed_contract_is_rejected() -> None:
    contract = load_contract()
    with pytest.raises(DirectExpertContractError, match="final depth differs"):
        replace(contract, final_parameters={**contract.final_parameters, "depth": 8})


def test_unknown_contract_key_is_rejected(tmp_path: Path) -> None:
    source = Path("experiments/direct_expert/contract.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["surprise"] = True
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DirectExpertContractError, match="contract keys differ"):
        load_contract(changed)


def test_boolean_is_not_accepted_as_integer(tmp_path: Path) -> None:
    source = Path("experiments/direct_expert/contract.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["screening_seed"] = True
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DirectExpertContractError, match="screening_seed"):
        load_contract(changed)
