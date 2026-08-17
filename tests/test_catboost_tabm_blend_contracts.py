from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.catboost_tabm_blend.contracts import (
    BlendContractError,
    build_jobs,
    contract_sha256,
    load_contract,
)


def test_contract_seals_two_folds_one_seed_and_three_weights() -> None:
    contract = load_contract()

    assert [(job.train_end_year, job.valid_year) for job in build_jobs(contract)] == [
        (2022, 2023),
        (2023, 2024),
    ]
    assert {job.seed for job in build_jobs(contract)} == {42}
    assert contract.tabm_weights == (0.9, 0.8, 0.7)
    assert contract.minimum_weighted_gain == 0.00003
    assert contract.maximum_fold_regression == 0.00003
    assert contract.review_only is True
    assert contract.submission_package is False
    assert len(contract_sha256()) == 64


def test_jobs_have_exact_stable_identifiers() -> None:
    assert tuple(job.job_id for job in build_jobs(load_contract())) == (
        "catboost__hand_matchup__tr2022__va2023__s42",
        "catboost__hand_matchup__tr2023__va2024__s42",
    )


def _mutated_contract(tmp_path: Path, mutate) -> Path:
    source = Path("experiments/catboost_tabm_blend/contract.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutate(payload)
    destination = tmp_path / "contract.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version=2),
        lambda value: value.update(extra=True),
        lambda value: value.pop("folds"),
        lambda value: value.update(folds=list(reversed(value["folds"]))),
        lambda value: value["catboost"].update(random_seed=3407),
        lambda value: value.update(tabm_weights=[0.95, 0.8, 0.7]),
        lambda value: value.update(review_only=False),
        lambda value: value.update(submission_package=True),
        lambda value: value.update(official_train_sha256="A" * 64),
        lambda value: value["catboost"].update(depth=8),
        lambda value: value["budget"].update(snapshot_interval_seconds=600),
        lambda value: value["budget"].update(download_interval_seconds=600),
        lambda value: value["budget"].update(wall_seconds=36000),
    ],
)
def test_contract_rejects_identity_drift(tmp_path: Path, mutate) -> None:
    path = _mutated_contract(tmp_path, mutate)

    with pytest.raises(BlendContractError):
        load_contract(path)
