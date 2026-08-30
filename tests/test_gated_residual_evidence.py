from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from experiments.gated_residual_final.evidence import (
    average_seed_frames,
    build_seed_frame,
    candidate_probability,
)
from experiments.gated_residual_final.selection import CandidateConfig


def _prediction(role: str, year: int, seed: int) -> pd.DataFrame:
    base = 0.2 if role == "D0" else 0.8
    return pd.DataFrame({
        "row_id": [f"TRAIN_{year}_R", f"TRAIN_{year}_F"],
        "target": [1, 0], "probability": [base + seed * 0.0, base + 0.1],
        "game_type": ["R", "F"], "pitcher_id": ["p1", "p2"], "oof_year": [year, year],
    })


def _archives(tmp_path: Path) -> tuple[Path, Path]:
    stage_a = tmp_path / "a.zip"
    stage_b = tmp_path / "b.zip"
    with ZipFile(stage_a, "w", ZIP_DEFLATED) as archive:
        for role in ("D0", "D5"):
            for year in (2022, 2023):
                name = f"jobs/screen__{role}__{year - 1}_{year}__s3407/predictions.csv"
                archive.writestr(name, _prediction(role, year, 3407).to_csv(index=False))
    with ZipFile(stage_b, "w", ZIP_DEFLATED) as archive:
        for role in ("D0", "D5"):
            name = f"jobs/confirm__{role}__2023_2024__s3407/predictions.csv"
            archive.writestr(name, _prediction(role, 2024, 3407).to_csv(index=False))
            for seed in (42, 2026):
                for year in (2022, 2023, 2024):
                    name = f"jobs/extra__{role}__{year - 1}_{year}__s{seed}/predictions.csv"
                    archive.writestr(name, _prediction(role, year, seed).to_csv(index=False))
    return stage_a, stage_b


def _e2(year: int) -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": [f"TRAIN_{year}_R", f"TRAIN_{year}_F"],
        "target": [1, 0], "p_anchor": [0.5, 0.5], "oof_year": [year, year],
        "game_type": ["R", "F"], "pitcher_id": ["p1", "p2"], "batter_id": ["b1", "b2"],
        "pitcher_hand": ["R", "L"], "batter_hand": ["L", "R"],
    })


def test_seed_frame_routes_d5_only_on_regular_rows(tmp_path: Path) -> None:
    stage_a, stage_b = _archives(tmp_path)

    frame = build_seed_frame(
        e2_by_year={year: _e2(year) for year in (2022, 2023, 2024)},
        stage_a=stage_a, stage_b=stage_b, seed=3407,
        train_for_counts=pd.concat([_e2(year).assign(season=year) for year in (2022, 2023, 2024)]),
    )

    regular = frame.loc[frame["game_type"].eq("R")]
    finals = frame.loc[frame["game_type"].eq("F")]
    np.testing.assert_allclose(regular["p_direct"], 0.8)
    np.testing.assert_allclose(finals["p_direct"], 0.3)
    assert set(frame["hand_matchup"]) == {"RL", "LR"}


def test_seed_average_compares_values_not_dtypes(tmp_path: Path) -> None:
    stage_a, stage_b = _archives(tmp_path)
    sources = []
    for seed in (42, 2026, 3407):
        frame = build_seed_frame(
            e2_by_year={year: _e2(year) for year in (2022, 2023, 2024)},
            stage_a=stage_a, stage_b=stage_b, seed=seed,
            train_for_counts=pd.concat([_e2(year).assign(season=year) for year in (2022, 2023, 2024)]),
        )
        frame["target"] = frame["target"].astype("int64" if seed == 42 else "int8")
        sources.append(frame)

    averaged = average_seed_frames(dict(zip((42, 2026, 3407), sources)))

    assert len(averaged) == 6
    np.testing.assert_allclose(averaged["p_direct"], sources[0]["p_direct"])


def test_candidate_probability_is_temporal_and_finite(tmp_path: Path) -> None:
    stage_a, stage_b = _archives(tmp_path)
    frame = build_seed_frame(
        e2_by_year={year: _e2(year) for year in (2022, 2023, 2024)},
        stage_a=stage_a, stage_b=stage_b, seed=3407,
        train_for_counts=pd.concat([_e2(year).assign(season=year) for year in (2022, 2023, 2024)]),
    )
    config = CandidateConfig("G3", alpha=0.1, k=25, beta=0.1, ridge=100)

    probability, sources = candidate_probability(frame, config)

    assert probability.shape == (len(frame),)
    assert np.isfinite(probability).all()
    assert sources == {2022: (), 2023: (2022,), 2024: (2022, 2023)}
