from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.hierarchical_tabm.contracts import (
    DEFAULT_CONTRACT,
    Fold,
    HierarchicalContractError,
    ModelPolicy,
    build_jobs,
    contract_sha256,
    load_contract,
)


TRAIN_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
HISTORY_SHA = "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9"
STAGE_C_SHA = "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a"


def _payload() -> dict[str, object]:
    return json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def test_contract_seals_three_hypotheses_and_current_tabm() -> None:
    contract = load_contract()
    assert contract.schema_version == 1
    assert contract.campaign_id == "hierarchical_tabm_score_push_v1"
    assert contract.review_only is True
    assert contract.submission_package is False
    assert contract.official_train_sha256 == TRAIN_SHA
    assert contract.official_history_sha256 == HISTORY_SHA
    assert contract.source_stage_c_delivery_sha256 == STAGE_C_SHA
    assert contract.k_candidates == (32.0, 128.0, 512.0)
    assert contract.development_fold == Fold(2021, 2022)
    assert contract.oof_folds == (Fold(2022, 2023), Fold(2023, 2024))
    assert contract.candidate_ids == ("H1", "H2", "H3")
    assert contract.model == ModelPolicy(
        k=32,
        width=512,
        blocks=4,
        dropout=0.1,
        num_embedding="piecewise_linear",
        loss="bce",
        scheduler="plateau",
        learning_rate=0.0006,
        weight_decay=0.0001,
        effective_batch_size=4096,
        micro_batch_size=512,
        seed=3407,
    )


def test_contract_seals_hierarchy_training_calibration_and_limits() -> None:
    contract = load_contract()
    assert contract.k_tie_tolerance == 0.000001
    assert contract.pitcher_reliability_k == 100.0
    assert contract.batter_reliability_k == 250.0
    assert contract.hierarchy_columns == (
        "hier_context_rate",
        "hier_context_logit",
        "hier_pitcher_reliability",
        "hier_batter_reliability",
        "hier_pitcher_context_gap",
        "hier_batter_context_gap",
        "hier_pitcher_weighted_gap",
        "hier_batter_weighted_gap",
    )
    assert (contract.max_epochs, contract.min_epochs, contract.patience) == (40, 3, 10)
    assert (contract.final_min_epochs, contract.final_max_epochs) == (2, 8)
    assert contract.calibration_grid == (0.0001, 0.001, 0.01, 0.1)
    assert contract.probability_clip == 0.000001
    assert contract.segment_min_rows == 5000
    assert dict(contract.inference_limits) == {
        "python_version": "3.11.15",
        "inference_seconds": 480,
        "gpu_bytes": 21474836480,
        "rss_bytes": 22000000000,
        "artifact_bytes": 2000000000,
    }
    assert (
        contract.session_seconds,
        contract.new_job_guard_seconds,
        contract.snapshot_interval_seconds,
        contract.download_interval_seconds,
    ) == (10800, 900, 300, 1200)


def test_contract_seals_candidate_gates() -> None:
    gates = {name: dict(values) for name, values in load_contract().gates.items()}
    assert gates == {
        "H1_strong": {
            "minimum_weighted_gain": 0.00010,
            "minimum_latest_gain": 0.00005,
            "maximum_old_fold_regression": 0.00015,
            "maximum_segment_regression": 0.00075,
        },
        "H1_frontier": {
            "maximum_old_fold_regression": 0.00025,
        },
        "H2": {
            "minimum_latest_gain_vs_h1": 0.00003,
            "minimum_latest_gain_vs_anchor": 0.00008,
            "maximum_segment_regression_vs_h1": 0.00020,
        },
        "H3": {
            "minimum_latest_gain_vs_h1": 0.00003,
            "minimum_latest_gain_vs_anchor": 0.00008,
            "maximum_segment_regression_vs_h1": 0.00020,
            "minimum_latest_gain_vs_h2": 0.00002,
        },
    }


def test_contract_builds_only_two_oof_jobs_then_one_full_job() -> None:
    jobs = build_jobs(load_contract())
    assert [(job.job_id, job.kind, job.train_end_year, job.valid_year) for job in jobs] == [
        ("h1__tr2022__va2023__s3407", "oof", 2022, 2023),
        ("h1__tr2023__va2024__s3407", "oof", 2023, 2024),
        ("h1__full__tr2024__s3407", "full_fit", 2024, None),
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("campaign_id",), "other"),
        (("review_only",), False),
        (("submission_package",), True),
        (("official_train_sha256",), "0" * 64),
        (("official_history_sha256",), "0" * 64),
        (("source_stage_c_delivery_sha256",), "0" * 64),
        (("candidate_ids",), ["H1", "H3", "H2"]),
        (("k_candidates",), [32.0, 512.0, 128.0]),
        (("k_tie_tolerance",), 0.000002),
        (("reliability", "pitcher_k"), 101.0),
        (("model", "seed"), 42),
        (("training", "max_epochs"), 41),
        (("calibration", "grid"), [0.0001, 0.001, 0.1]),
        (("segments", "minimum_rows"), 4999),
        (("budget", "session_seconds"), 10799),
        (("gates", "H3", "minimum_latest_gain_vs_h2"), 0.00003),
        (("inference_limits", "python_version"), "3.12.0"),
    ],
)
def test_rejects_any_preregistered_value_change(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    payload = _payload()
    target = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(HierarchicalContractError):
        load_contract(_write(tmp_path, payload))


def test_rejects_unknown_missing_and_duplicate_keys(tmp_path: Path) -> None:
    extra = _payload()
    extra["extra"] = 1
    with pytest.raises(HierarchicalContractError, match="keys"):
        load_contract(_write(tmp_path, extra))

    missing = _payload()
    del missing["budget"]
    with pytest.raises(HierarchicalContractError, match="keys"):
        load_contract(_write(tmp_path, missing))

    duplicate = DEFAULT_CONTRACT.read_text(encoding="utf-8").replace(
        '"schema_version":1', '"schema_version":1,"schema_version":1', 1
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(HierarchicalContractError, match="duplicate"):
        load_contract(path)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf")])
def test_rejects_boolean_and_nonfinite_numeric_values(tmp_path: Path, bad: object) -> None:
    payload = _payload()
    payload["training"]["max_epochs"] = bad  # type: ignore[index]
    with pytest.raises(HierarchicalContractError):
        load_contract(_write(tmp_path, payload))


def test_contract_hash_validates_then_hashes_exact_bytes(tmp_path: Path) -> None:
    expected = contract_sha256()
    assert len(expected) == 64
    copied = tmp_path / "contract.json"
    copied.write_bytes(DEFAULT_CONTRACT.read_bytes())
    assert contract_sha256(copied) == expected
    copied.write_bytes(copied.read_bytes() + b"\n")
    assert contract_sha256(copied) != expected

