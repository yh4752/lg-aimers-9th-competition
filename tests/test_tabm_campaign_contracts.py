from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tabm_campaign.contracts import CampaignContractError, load_campaign


CONFIG = Path("experiments/tabm_campaign/configs/champion_v1.json")


def test_champion_contract_has_fixed_preprocessing_and_four_versions() -> None:
    campaign = load_campaign(CONFIG)

    assert campaign.preprocessing.profile == "dl_standard"
    assert campaign.preprocessing.components == ("hand_matchup",)
    assert campaign.preprocessing.fit_scope == "official_train_only"
    assert dict(campaign.wall_seconds) == {"A": 7200, "B": 10800, "C": 10800, "D": 7200}
    assert len(campaign.version_a_candidates) == 24
    assert {candidate.loss for candidate in campaign.version_a_candidates} == {"bce", "brier"}
    assert {candidate.scheduler for candidate in campaign.version_a_candidates} == {
        "one_cycle",
        "plateau",
    }
    assert campaign.seeds == (42, 2026, 3407)
    assert campaign.max_final_weights == 3
    assert len(campaign.refinements) == 9


@pytest.mark.parametrize(
    ("path", "value", "replacement", "message"),
    [
        (("preprocessing", "fit_scope"), "official_train_only", "test", "official_train_only"),
        (("candidates", 0, "family"), "tabm", "tabnet", "tabm"),
        (("wall_seconds", "A"), 7200, 7201, "wall_seconds"),
    ],
)
def test_contract_rejects_mutated_sealed_fields(
    tmp_path: Path,
    path: tuple[object, ...],
    value: object,
    replacement: object,
    message: str,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    cursor = payload
    for part in path[:-1]:
        cursor = cursor[part]
    assert cursor[path[-1]] == value
    cursor[path[-1]] = replacement
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CampaignContractError, match=message):
        load_campaign(bad)


def test_contract_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    bad = tmp_path / "duplicate.json"
    bad.write_text('{"campaign_id":"one","campaign_id":"two"}', encoding="utf-8")
    with pytest.raises(CampaignContractError, match="duplicate JSON key"):
        load_campaign(bad)


def test_contract_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["surprise"] = True
    bad = tmp_path / "unknown.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CampaignContractError, match="unknown keys"):
        load_campaign(bad)
