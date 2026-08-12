from __future__ import annotations

from pathlib import Path

import pytest

from experiments.independent_dl.contracts import CampaignContractError, load_campaign


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/independent_dl/configs/campaign_v1.json"


def test_campaign_expands_four_families_four_views_and_64_candidates() -> None:
    campaign = load_campaign(CONFIG)

    assert campaign.campaign_id == "independent_dl_campaign_v1"
    assert {candidate.family for candidate in campaign.candidates} == {
        "tabm",
        "mlp_resnet",
        "ft_transformer",
        "tabr",
    }
    assert {candidate.feature_view for candidate in campaign.candidates} == {
        "raw_typed",
        "engineered",
        "entity_context",
        "trackman_augmented",
    }
    assert len(campaign.candidates) == 64
    assert len({candidate.candidate_id for candidate in campaign.candidates}) == 64


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

    assert {candidate.training["scheduler"] for candidate in campaign.candidates} == {
        "cosine",
        "plateau",
        "one_cycle",
        "cosine_warmup",
    }
    assert {candidate.training["effective_batch_size"] for candidate in campaign.candidates} == {
        4096
    }
    assert {candidate.training["amp"] for candidate in campaign.candidates} == {True}
    assert {candidate.training["patience"] for candidate in campaign.candidates} == {
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
