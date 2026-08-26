from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from experiments.tree_expert.contracts import (
    TreeExpertContractError,
    build_e1_jobs,
    load_e1_contract,
)


CONTRACT_PATH = Path("experiments/tree_expert/e1_contract.json")


def test_e1_contract_is_review_only_and_has_four_fixed_candidates() -> None:
    contract = load_e1_contract()

    assert contract.review_only is True
    assert contract.submission_package is False
    assert contract.fold == (2023, 2024)
    assert contract.seed == 3407
    assert tuple(job.candidate_id for job in build_e1_jobs(contract)) == (
        "c0_native_ctr",
        "c1_anchor_residual",
        "c2_trackman_residual",
        "c3_failure_aware",
    )
    with pytest.raises(FrozenInstanceError):
        contract.seed = 1


def test_e1_contract_rejects_one_changed_gate(tmp_path: Path) -> None:
    source = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source["gates"]["stop_if_all_regress_more_than"] = 0.5
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(TreeExpertContractError, match="gates differ"):
        load_e1_contract(path)
