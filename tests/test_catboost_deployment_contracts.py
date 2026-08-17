from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.catboost_deployment.contracts import (
    DeploymentContractError,
    build_jobs,
    contract_sha256,
    load_contract,
)


def test_contract_is_preregistered() -> None:
    contract = load_contract()

    assert contract.source_blend_delivery_sha256 == (
        "ab7ca41e98e2b94b997369d7c777f8293c64acf61114f314110d17bf743edcfa"
    )
    assert contract.source_stage_c_delivery_sha256 == (
        "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a"
    )
    assert contract.tabm_weight == 0.70
    assert contract.tree_prefixes == (4, 32, 64, 128, 192, 296, 400)
    assert contract.minimum_weighted_gain == 0.00003
    assert contract.maximum_fold_regression == 0.00003
    assert contract.catboost_parameters["iterations"] == 400
    assert contract.catboost_parameters["random_seed"] == 42
    assert contract.catboost_parameters["task_type"] == "GPU"
    assert contract.snapshot_interval_seconds == 300
    assert contract.emergency_interval_seconds == 1200
    assert contract.session_seconds == 10800
    assert contract.new_job_guard_seconds == 900
    assert len(contract_sha256()) == 64


def test_jobs_are_two_alignment_folds_then_full_fit() -> None:
    jobs = build_jobs(load_contract())

    assert [(job.kind, job.train_end_year, job.valid_year) for job in jobs] == [
        ("alignment", 2022, 2023),
        ("alignment", 2023, 2024),
        ("full_fit", 2024, None),
    ]
    assert [job.job_id for job in jobs] == [
        "align_2022_2023",
        "align_2023_2024",
        "full_2024",
    ]
    assert {job.seed for job in jobs} == {42}


def _mutated_contract(tmp_path: Path, mutate) -> Path:
    source = Path("experiments/catboost_deployment/contract.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutate(payload)
    destination = tmp_path / "contract.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version=2),
        lambda value: value.update(campaign_id="changed"),
        lambda value: value.update(extra=True),
        lambda value: value.pop("tree_prefixes"),
        lambda value: value.update(source_blend_delivery_sha256="A" * 64),
        lambda value: value.update(source_stage_c_delivery_sha256="0" * 64),
        lambda value: value.update(tabm_weight=0.8),
        lambda value: value.update(tree_prefixes=[4, 32, 64, 128, 192, 296]),
        lambda value: value.update(tree_prefixes=[4, 32, 64, 128, 128, 296, 400]),
        lambda value: value["catboost_parameters"].update(iterations=True),
        lambda value: value["catboost_parameters"].update(depth=8),
        lambda value: value["gates"].update(minimum_weighted_gain=0.0),
        lambda value: value["gates"].update(maximum_fold_regression=0.0),
        lambda value: value["budget"].update(snapshot_interval_seconds=True),
        lambda value: value["budget"].update(emergency_interval_seconds=600),
        lambda value: value["budget"].update(session_seconds=36000),
        lambda value: value["budget"].update(new_job_guard_seconds=0),
    ],
)
def test_contract_rejects_identity_drift(tmp_path: Path, mutate) -> None:
    path = _mutated_contract(tmp_path, mutate)

    with pytest.raises(DeploymentContractError):
        load_contract(path)
