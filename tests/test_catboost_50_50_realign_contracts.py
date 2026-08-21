from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

import pytest

from experiments.catboost_50_50_realign.contracts import (
    DEFAULT_CONTRACT,
    RealignContractError,
    build_jobs,
    contract_sha256,
    load_contract,
)


def test_contract_seals_fixed_grid_weight_and_three_folds() -> None:
    contract = load_contract()

    assert contract.campaign_id == "catboost_50_50_realign_v2"
    assert contract.review_only is True
    assert contract.tabm_weight == Decimal("0.50")
    assert contract.tree_prefixes == (4, 8, 12, 16, 20, 24, 28, 32)
    assert tuple((fold.train_end_year, fold.valid_year) for fold in contract.folds) == (
        (2021, 2022),
        (2022, 2023),
        (2023, 2024),
    )
    assert contract.minimum_weighted_gain == Decimal("0.00003")
    assert contract.latest_bootstrap_lower_minimum == Decimal("0")
    assert contract.maximum_segment_regression == Decimal("0.00075")
    assert contract.selection_tolerance == Decimal("1e-12")


def test_contract_builds_only_new_fold_and_full_fit_jobs() -> None:
    jobs = build_jobs(load_contract())

    assert tuple(asdict(job) for job in jobs) == (
        {
            "job_id": "tabm_f1_2022",
            "kind": "tabm_alignment",
            "train_end_year": 2021,
            "valid_year": 2022,
            "seed": 3407,
        },
        {
            "job_id": "catboost_f1_2022",
            "kind": "catboost_alignment",
            "train_end_year": 2021,
            "valid_year": 2022,
            "seed": 42,
        },
        {
            "job_id": "catboost_full_2024",
            "kind": "full_fit",
            "train_end_year": 2024,
            "valid_year": None,
            "seed": 42,
        },
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("campaign_id",), "other"),
        (("source_sha256", "oof_audit"), "0" * 64),
        (("folds",), [[2022, 2023], [2023, 2024]]),
        (("tabm_weight",), "0.49"),
        (("tree_prefixes",), [4, 8, 16, 32]),
        (("tabm", "seed"), 42),
        (("catboost", "depth"), 8),
        (("gates", "minimum_weighted_gain"), "0.00002"),
        (("selection_tolerance",), "1e-9"),
        (("bootstrap", "seed"), 1),
        (("budget", "session_seconds"), 7200),
    ),
)
def test_contract_rejects_any_sealed_value_change(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    payload = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RealignContractError):
        load_contract(changed)


def test_contract_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(RealignContractError, match="duplicate JSON key"):
        load_contract(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        DEFAULT_CONTRACT.read_text(encoding="utf-8").replace('"1e-12"', "NaN"),
        encoding="utf-8",
    )
    with pytest.raises(RealignContractError, match="non-finite"):
        load_contract(nonfinite)


def test_contract_sha_is_bound_to_validated_bytes() -> None:
    digest = contract_sha256()
    assert len(digest) == 64
    assert int(digest, 16) >= 0
