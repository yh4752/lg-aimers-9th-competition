from __future__ import annotations

from pathlib import Path
from typing import Mapping
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .calibration import apply_calibration, fit_temporal_calibrator
from .oof import direct_probability
from .residual import gated_residual, row_reliability, temporal_entity_counts
from .selection import CandidateConfig


class EvidenceError(ValueError):
    pass


_SEEDS = (42, 2026, 3407)
_YEARS = (2022, 2023, 2024)


def _job_name(role: str, year: int, seed: int) -> tuple[str, str]:
    if role not in {"D0", "D5"} or year not in _YEARS or seed not in _SEEDS:
        raise EvidenceError("OOF job identity differs")
    if seed == 3407 and year in (2022, 2023):
        return "stage_a", f"jobs/screen__{role}__{year - 1}_{year}__s3407/predictions.csv"
    phase = "confirm" if seed == 3407 else "extra"
    return "stage_b", f"jobs/{phase}__{role}__{year - 1}_{year}__s{seed}/predictions.csv"


def _read_prediction(stage_a: Path, stage_b: Path, role: str, year: int, seed: int) -> pd.DataFrame:
    source_name, member = _job_name(role, year, seed)
    source = Path(stage_a) if source_name == "stage_a" else Path(stage_b)
    try:
        with ZipFile(source) as archive:
            frame = pd.read_csv(archive.open(member))
    except (OSError, KeyError, BadZipFile) as error:
        raise EvidenceError(f"OOF prediction is absent: {member}") from error
    required = {"row_id", "target", "probability", "game_type", "pitcher_id", "oof_year"}
    if set(frame) != required or frame.empty:
        raise EvidenceError(f"OOF prediction columns differ: {member}")
    if not pd.to_numeric(frame["oof_year"], errors="coerce").eq(year).all():
        raise EvidenceError(f"OOF prediction year differs: {member}")
    return frame


def _metadata(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    required = {
        "row_id", "target", "p_anchor", "oof_year", "game_type", "pitcher_id",
        "batter_id", "pitcher_hand", "batter_hand",
    }
    if type(frame) is not pd.DataFrame or frame.empty or not required.issubset(frame.columns):
        raise EvidenceError(f"E2 OOF columns differ: {year}")
    optional = {"base_state", "balls_before", "strikes_before"}.intersection(frame.columns)
    output = frame.loc[:, sorted(required | optional)].copy()
    if not pd.to_numeric(output["oof_year"], errors="coerce").eq(year).all():
        raise EvidenceError(f"E2 OOF year differs: {year}")
    output["row_id"] = output["row_id"].astype(str)
    output["hand_matchup"] = output["pitcher_hand"].astype(str) + output["batter_hand"].astype(str)
    if {"balls_before", "strikes_before"}.issubset(output.columns):
        output["count_state"] = (
            pd.to_numeric(output["balls_before"], errors="raise").astype(int).astype(str)
            + "-"
            + pd.to_numeric(output["strikes_before"], errors="raise").astype(int).astype(str)
        )
    return output


def build_seed_frame(
    *,
    e2_by_year: Mapping[int, pd.DataFrame],
    stage_a: Path,
    stage_b: Path,
    seed: int,
    train_for_counts: pd.DataFrame,
) -> pd.DataFrame:
    if set(e2_by_year) != set(_YEARS):
        raise EvidenceError("E2 OOF year set differs")
    frames = []
    for year in _YEARS:
        e2 = _metadata(e2_by_year[year], year)
        d0 = _read_prediction(stage_a, stage_b, "D0", year, seed).rename(columns={"probability": "p_d0"})
        d5 = _read_prediction(stage_a, stage_b, "D5", year, seed).rename(columns={"probability": "p_d5"})
        d0 = d0.loc[:, ["row_id", "target", "p_d0"]].assign(row_id=lambda value: value["row_id"].astype(str))
        d5 = d5.loc[:, ["row_id", "target", "p_d5"]].assign(row_id=lambda value: value["row_id"].astype(str))
        merged = e2.merge(d0, on=["row_id", "target"], validate="one_to_one").merge(
            d5, on=["row_id", "target"], validate="one_to_one"
        )
        if len(merged) != len(e2):
            raise EvidenceError(f"OOF row alignment differs: {year} seed={seed}")
        merged["p_direct"] = direct_probability(
            game_type=merged["game_type"].to_numpy(),
            d0=merged["p_d0"].to_numpy(),
            d5=merged["p_d5"].to_numpy(),
        )
        counts = temporal_entity_counts(train_for_counts, validation_year=year)
        for k in (25, 100, 400):
            merged[f"reliability_k{k}"] = row_reliability(
                merged, pitcher_counts=counts.pitcher, batter_counts=counts.batter, k=k,
            )
        merged["seed"] = int(seed)
        frames.append(merged)
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values("row_id", kind="stable").reset_index(drop=True)


def average_seed_frames(frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    if set(frames) != set(_SEEDS):
        raise EvidenceError("seed frame set differs")
    normalized = {
        seed: frame.sort_values("row_id", kind="stable").reset_index(drop=True).copy(deep=True)
        for seed, frame in frames.items()
    }
    base = normalized[42].copy(deep=True)
    value_columns = ("p_d0", "p_d5", "p_direct")
    metadata_columns = [column for column in base if column not in {*value_columns, "seed"}]
    for seed in (2026, 3407):
        other = normalized[seed]
        if len(other) != len(base):
            raise EvidenceError(f"seed row count differs: {seed}")
        for column in metadata_columns:
            left = base[column].astype(str).to_numpy() if column in {"row_id", "game_type", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand", "hand_matchup"} else pd.to_numeric(base[column], errors="coerce").to_numpy(dtype="float64")
            right = other[column].astype(str).to_numpy() if column in {"row_id", "game_type", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand", "hand_matchup"} else pd.to_numeric(other[column], errors="coerce").to_numpy(dtype="float64")
            same = np.array_equal(left, right) if left.dtype.kind in {"O", "U", "S"} else np.array_equal(left, right, equal_nan=True)
            if not same:
                raise EvidenceError(f"seed metadata values differ: {seed} column={column}")
    for column in value_columns:
        base[column] = np.mean(
            [normalized[seed][column].to_numpy(dtype="float64") for seed in _SEEDS], axis=0
        )
    base["seed"] = -1
    return base


def candidate_probability(
    frame: pd.DataFrame, config: CandidateConfig,
) -> tuple[np.ndarray, dict[int, tuple[int, ...]]]:
    required = {
        "target", "p_anchor", "p_direct", "oof_year", "game_type", "hand_matchup",
        "pitcher_id", "batter_id",
    }
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns):
        raise EvidenceError("candidate evidence columns differ")
    if config.archetype == "G0":
        strength = frame["game_type"].astype(str).eq("R").to_numpy(dtype="float64")
    else:
        if config.k not in (25, 100, 400) or f"reliability_k{config.k}" not in frame:
            raise EvidenceError("candidate reliability differs")
        strength = frame[f"reliability_k{config.k}"].to_numpy(dtype="float64")
    base = gated_residual(
        anchor=frame["p_anchor"].to_numpy(dtype="float64"),
        direct=frame["p_direct"].to_numpy(dtype="float64"),
        row_reliability=strength,
        alpha=config.alpha,
    )
    years = tuple(sorted(int(value) for value in pd.to_numeric(frame["oof_year"]).unique()))
    sources: dict[int, tuple[int, ...]] = {}
    if config.archetype in {"G0", "G1"}:
        for year in years:
            sources[year] = ()
        return base, sources
    hierarchy = "global_game" if config.archetype == "G2" else "full"
    work = frame.copy(deep=True)
    work["probability"] = base
    output = base.copy()
    for year in years:
        calibrator = fit_temporal_calibrator(
            work, validation_year=year, hierarchy=hierarchy, ridge=int(config.ridge),
        )
        mask = work["oof_year"].eq(year).to_numpy()
        output[mask] = apply_calibration(
            calibrator, work.loc[mask], beta=float(config.beta),
        )
        sources[year] = calibrator.source_years
    return output, sources
