from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from experiments.independent_dl.preprocessing_contracts import (
    PreprocessingContractError,
    load_preprocessing_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/independent_dl/configs/preprocessing_ablation_v1.json"
SOURCE_CONFIG = ROOT / "experiments/independent_dl/configs/campaign_v1.json"


def test_wave_a_expands_eight_anchors_nineteen_settings_and_five_folds() -> None:
    campaign = load_preprocessing_campaign(CONFIG)

    assert campaign.campaign_id == "preprocessing_campaign_v1"
    assert campaign.folds == (
        (2019, 2020),
        (2020, 2021),
        (2021, 2022),
        (2022, 2023),
        (2023, 2024),
    )
    assert len(campaign.anchors) == 8
    assert len(campaign.dl_settings) == 19
    assert len(campaign.wave_a_jobs) == 760
    assert {job.seed for job in campaign.wave_a_jobs} == {42}
    assert {job.profile_id for job in campaign.wave_a_jobs} == {"p3", "p4"}
    assert len({job.job_id for job in campaign.wave_a_jobs}) == 760


def test_contract_pins_the_existing_campaign_bytes(tmp_path: Path) -> None:
    shutil.copy2(SOURCE_CONFIG, tmp_path / SOURCE_CONFIG.name)
    config = CONFIG.read_text(encoding="utf-8").replace(
        "bd8116773b3f77255315262277461cd27764b9fc37d73a5e7c1f72b2a7f98b53",
        "0" * 64,
    )
    changed = tmp_path / CONFIG.name
    changed.write_text(config, encoding="utf-8")

    with pytest.raises(PreprocessingContractError, match="source campaign SHA-256"):
        load_preprocessing_campaign(changed)


def test_settings_keep_full_smoothing_ranges_and_three_seeds() -> None:
    campaign = load_preprocessing_campaign(CONFIG)

    assert campaign.seeds == (42, 2026, 3407)
    assert campaign.pitcher_smoothing_k == (25, 50, 100, 250)
    assert campaign.batter_smoothing_k == (10, 25, 100, 250, 500, 1000, 2500)
    assert len(campaign.catboost_settings) == 17
    assert tuple(campaign.catboost_structures) == (
        "champion",
        "depth5",
        "depth8",
        "lr003",
    )


@pytest.mark.parametrize(
    "payload",
    ('{"campaign_id":NaN}', '{"campaign_id":"a","campaign_id":"b"}'),
)
def test_contract_rejects_nonfinite_and_duplicate_json(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(PreprocessingContractError):
        load_preprocessing_campaign(path)
