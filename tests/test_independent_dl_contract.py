from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.independent_dl.campaign import _config_sha256
from experiments.independent_dl.contracts import CampaignContractError, load_campaign


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/independent_dl/configs/campaign_v1.json"


EXPECTED_FRONTIER_PRIORITY = (
    "tabm__raw_typed__p1__s42",
    "tabm__raw_typed__p2__s42",
    "tabm__engineered__p2__s42",
    "tabm__entity_context__p2__s42",
    "tabm__trackman_augmented__p2__s42",
    "tabm__raw_typed__p3__s42",
    "tabm__engineered__p3__s42",
    "tabm__entity_context__p3__s42",
    "tabm__trackman_augmented__p3__s42",
    "mlp_resnet__raw_typed__p3__s42",
    "mlp_resnet__engineered__p3__s42",
    "mlp_resnet__entity_context__p3__s42",
    "mlp_resnet__trackman_augmented__p3__s42",
    "ft_transformer__raw_typed__p3__s42",
    "ft_transformer__engineered__p3__s42",
    "ft_transformer__entity_context__p3__s42",
    "ft_transformer__trackman_augmented__p3__s42",
    "tabicl_v2__raw_typed__frontier32__s42",
)


def test_campaign_expands_trainable_grid_and_frontier_candidate() -> None:
    campaign = load_campaign(CONFIG)

    assert campaign.campaign_id == "independent_dl_campaign_v1"
    assert {candidate.family for candidate in campaign.candidates} == {
        "tabm",
        "mlp_resnet",
        "ft_transformer",
        "tabr",
        "tabicl_v2",
    }
    assert {candidate.feature_view for candidate in campaign.candidates} == {
        "raw_typed",
        "engineered",
        "entity_context",
        "trackman_augmented",
    }
    assert len(campaign.candidates) == 65
    assert len({candidate.candidate_id for candidate in campaign.candidates}) == 65


def test_campaign_uses_frontier_priority_before_remaining_grid() -> None:
    campaign = load_campaign(CONFIG)

    assert tuple(
        candidate.candidate_id for candidate in campaign.candidates[:18]
    ) == EXPECTED_FRONTIER_PRIORITY
    assert campaign.candidates[17].stage == "research_only"


def test_existing_candidate_hashes_survive_execution_wave_upgrade() -> None:
    campaign = load_campaign(CONFIG)
    by_id = {candidate.candidate_id: candidate for candidate in campaign.candidates}

    assert _config_sha256(by_id["tabm__raw_typed__p1__s42"]) == (
        "57c86f873a4eadbe69214f28259c0b5f3feb27dcedbdccb487b9d87c28771801"
    )
    assert _config_sha256(by_id["tabm__raw_typed__p2__s42"]) == (
        "49043e2018df2af03469f0606908806f3a82174abb32a7e823b6b6e8d7ddab34"
    )


def test_frontier_candidate_values_are_sealed(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["frontier_candidates"][0]["model"]["n_estimators"] = 16
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CampaignContractError, match="frontier candidate model"):
        load_campaign(path)


def test_campaign_contains_full_scale_and_boundary_expansion_contracts() -> None:
    campaign = load_campaign(CONFIG)

    assert campaign.exploration_fold == (2023, 2024)
    assert campaign.oof_folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert max(candidate.epochs for candidate in campaign.candidates) == 400
    assert campaign.confirmation_seeds == (2026, 3407)
    assert campaign.boundary_expansion["tabm"]["k"] == (96, 128)
    assert campaign.boundary_expansion["mlp_resnet"]["width"] == (3072, 4096)
    assert campaign.boundary_expansion["ft_transformer"]["layers"] == (16, 20)
    assert campaign.boundary_expansion["tabr"]["retrieval"] == (384, 512)
    assert {
        candidate.model["architecture"]
        for candidate in campaign.candidates
        if candidate.family == "mlp_resnet"
    } == {"mlp", "resnet"}


def test_campaign_keeps_performance_first_training_profiles() -> None:
    campaign = load_campaign(CONFIG)
    trainable = [item for item in campaign.candidates if item.family != "tabicl_v2"]

    assert {candidate.training["scheduler"] for candidate in trainable} == {
        "cosine",
        "plateau",
        "one_cycle",
        "cosine_warmup",
    }
    assert {candidate.training["effective_batch_size"] for candidate in trainable} == {
        4096
    }
    assert {candidate.training["amp"] for candidate in trainable} == {True}
    assert {candidate.training["patience"] for candidate in trainable} == {
        30,
        40,
        60,
        80,
    }


@pytest.mark.parametrize(
    "payload",
    ('{"campaign_id":NaN}', '{"campaign_id":"a","campaign_id":"b"}'),
)
def test_campaign_rejects_nonfinite_and_duplicate_json(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(CampaignContractError):
        load_campaign(path)
