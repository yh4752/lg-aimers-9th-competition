from __future__ import annotations

from hashlib import sha256

import pandas as pd

from experiments.temporal_portfolio.contracts import build_stage_jobs, load_contract
from experiments.temporal_portfolio.job_materialization import materialize_t1_job


def _train() -> pd.DataFrame:
    rows = []
    for year in (2019, 2020, 2021, 2022, 2023, 2024):
        for index in range(4):
            rows.append(
                {
                    "row_id": f"{year}_{index}",
                    "season": year,
                    "game_type": "regular",
                    "pitcher_id": 10 + index,
                    "batter_id": 20 + index,
                    "pitcher_hand": "R",
                    "batter_hand": "L" if index % 2 else "R",
                    "asof_pitcher_n": 10 + index,
                    "base_state": "empty",
                    "li": 1.0,
                    "numeric": float(index),
                    "control_success": index % 2,
                }
            )
    return pd.DataFrame(rows)


def test_materialize_t1_job_binds_cache_request_audit_and_identity(tmp_path) -> None:
    contract = load_contract()
    spec = build_stage_jobs(contract, "T1")[0]
    train = _train()
    data_rows_sha256 = sha256(b"fixture rows").hexdigest()

    materialized = materialize_t1_job(
        spec,
        data_rows_sha256=data_rows_sha256,
        train=train,
        history=pd.DataFrame(),
        cache_root=tmp_path / "cache",
    )

    assert materialized.plan.identity == materialized.training.identity
    assert materialized.training.job_id == spec.job_id
    assert materialized.training.valid_rows == 4
    assert materialized.training.sample_weight.tolist() == [1.0] * 4
    materialized.training.validate_seals()


def test_materialize_multi_job_applies_season_decay(tmp_path) -> None:
    spec = next(
        item
        for item in build_stage_jobs(load_contract(), "T1")
        if item.expert == "multi" and item.fold.valid_year == 2022
    )
    materialized = materialize_t1_job(
        spec,
        data_rows_sha256=sha256(b"fixture rows").hexdigest(),
        train=_train(),
        history=pd.DataFrame(),
        cache_root=tmp_path / "cache",
    )
    expected = [float(spec.decay) ** 2] * 4 + [float(spec.decay)] * 4 + [1.0] * 4
    assert materialized.training.sample_weight.tolist() == expected
    assert tuple(materialized.training.identity.payload["train_seasons"]) == (
        2019,
        2020,
        2021,
    )
