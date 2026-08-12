from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.preprocessing_campaign.budgeted_contracts import (
    deterministic_temporal_sample,
    load_budgeted_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments/preprocessing_campaign/configs/budgeted_campaign_v1.json"
)


def _season_frame(rows_per_season: int = 120) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in range(2019, 2025):
        rows.extend(
            {
                "row_id": f"{season}-{index:04d}",
                "season": season,
                "control_success": index % 2,
            }
            for index in range(rows_per_season)
        )
    return pd.DataFrame(rows)


def test_budgeted_config_registers_only_predeclared_models_and_settings() -> None:
    campaign = load_budgeted_campaign(CONFIG)

    assert campaign.campaign_id == "budgeted_preprocessing_campaign_v1"
    assert campaign.session_seconds == 6300
    assert campaign.stop_new_jobs_seconds == 900
    assert campaign.archive_reserve_seconds == 600
    assert [job.family for job in campaign.stage_jobs(1)] == [
        "tabm",
        "ft_transformer",
        "tabnet",
        "catboost",
    ]
    assert {job.setting_id for job in campaign.stage_jobs(2)} == {
        "dl_standard",
        "selective_yeo_johnson",
        "pitcher_smoothing_k100",
        "batter_smoothing_k250",
    }
    assert {job.setting_id for job in campaign.stage_jobs(3)} == {
        "id_frequency_and_oov",
        "asof_count_log1p",
        "grouped_recent_missing",
        "hand_matchup",
    }


def test_temporal_proxy_is_deterministic_proportional_and_train_only() -> None:
    frame = _season_frame()

    left = deterministic_temporal_sample(
        frame, train_end_year=2023, max_rows=400, seed=42
    )
    right = deterministic_temporal_sample(
        frame.sample(frac=1.0, random_state=7),
        train_end_year=2023,
        max_rows=400,
        seed=42,
    )

    assert len(left) == 400
    assert left["row_id"].tolist() == right["row_id"].tolist()
    assert left["season"].max() == 2023
    assert left.groupby("season").size().to_dict() == {
        2019: 80,
        2020: 80,
        2021: 80,
        2022: 80,
        2023: 80,
    }


def test_temporal_proxy_returns_all_rows_when_below_cap() -> None:
    frame = _season_frame(rows_per_season=10)

    sampled = deterministic_temporal_sample(
        frame, train_end_year=2020, max_rows=400, seed=42
    )

    assert len(sampled) == 20
    assert set(sampled["season"]) == {2019, 2020}
