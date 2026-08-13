from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from competition_rules.contract import (
    RulesContractError,
    validate_experiment_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_active_experiment_contracts_are_current_and_hash_bound() -> None:
    expected = {
        "experiments/independent_dl/experiment_contract.json": {
            "experiments/independent_dl/configs/campaign_v1.json":
                "a3a7f82b6d20d59764f0c7e2964a3c2b9489b452ebd59f5a9c137cf25e6d5c55",
            "experiments/independent_dl/configs/preprocessing_ablation_v1.json":
                "e322a0d5b8e0759ad189528c24a353d2dfa9342d6115cc4b361a1cd77939c09f",
        },
        "experiments/preprocessing_campaign/experiment_contract.json": {
            "experiments/preprocessing_campaign/configs/budgeted_campaign_v1.json":
                "99deaa32e5d9616a0f50625ac38eda73df2d472dd54d75ff26fda00ef9f3958b",
        },
        "experiments/catboost_preprocessing/experiment_contract.json": {
            "experiments/independent_dl/configs/preprocessing_ablation_v1.json":
                "e322a0d5b8e0759ad189528c24a353d2dfa9342d6115cc4b361a1cd77939c09f",
        },
    }

    for relative, bindings in expected.items():
        contract = validate_experiment_contract(ROOT / relative, project_root=ROOT)
        assert contract["rules_version"] == "dacon-236743-2026-08-13"
        assert contract["config_sha256"] == bindings
        assert contract["evaluation_scope"] == "current_row_only"
        assert contract["external_api"] is False


def test_tabicl_is_not_covered_by_the_campaign_contract() -> None:
    path = ROOT / "experiments/independent_dl/experiment_contract.json"

    assert validate_experiment_contract(
        path,
        project_root=ROOT,
        candidate_id="tabm__raw_typed__p1__s42",
        config_path=ROOT / "experiments/independent_dl/configs/campaign_v1.json",
    )["candidate_count"] == 64
    with pytest.raises(RulesContractError, match="candidate is not covered"):
        validate_experiment_contract(
            path,
            project_root=ROOT,
            candidate_id="tabicl_v2__raw_typed__frontier32__s42",
            config_path=ROOT / "experiments/independent_dl/configs/campaign_v1.json",
        )


def _write_valid_contract_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config = tmp_path / "config.json"
    config.write_text('{"candidate":"safe"}\n', encoding="utf-8")
    source = tmp_path / "predict.py"
    source.write_text("def predict(frame):\n    return frame['x']\n", encoding="utf-8")
    candidate_ids = ["safe"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_id": "fixture_contract_v1",
        "rules_version": "dacon-236743-2026-08-13",
        "scope": "config_bound_campaign",
        "candidate_source": "explicit",
        "candidate_ids": candidate_ids,
        "candidate_count": 1,
        "candidate_ids_sha256": sha256(
            json.dumps(candidate_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "allowed_derivations": [],
        "config_sha256": {"config.json": sha256(config.read_bytes()).hexdigest()},
        "inference_source_paths": ["predict.py"],
        "data_sources": ["official_train", "official_trackman"],
        "fit_scope": "training_rows_only",
        "evaluation_scope": "current_row_only",
        "time_scope": "pre_pitch_only",
        "external_api": False,
        "pretrained_models": [],
        "retrieval_corpora": [],
    }
    path = tmp_path / "experiment_contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fit_scope", "evaluation_rows"),
        ("evaluation_scope", "whole_evaluation_set"),
        ("time_scope", "post_pitch"),
        ("external_api", True),
    ],
)
def test_rule_relevant_mutations_are_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    path, payload = _write_valid_contract_fixture(tmp_path)
    assert validate_experiment_contract(path, project_root=tmp_path)

    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RulesContractError, match=field):
        validate_experiment_contract(path, project_root=tmp_path)


def test_config_hash_and_source_path_are_live_bound(tmp_path: Path) -> None:
    path, _ = _write_valid_contract_fixture(tmp_path)
    assert validate_experiment_contract(path, project_root=tmp_path)

    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RulesContractError, match="config SHA-256"):
        validate_experiment_contract(path, project_root=tmp_path)


def test_candidate_digest_must_match_the_declared_ids(tmp_path: Path) -> None:
    path, payload = _write_valid_contract_fixture(tmp_path)
    ids = ["safe", "unsafe"]
    payload["candidate_ids"] = ids
    payload["candidate_count"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RulesContractError, match="candidate_ids_sha256"):
        validate_experiment_contract(path, project_root=tmp_path)
