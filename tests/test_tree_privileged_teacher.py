from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.temporal_portfolio.lupi_teacher import TEACHER_FEATURES
from experiments.tree_privileged.matching import MATCH_COLUMNS
from experiments.tree_privileged.teacher import build_teacher_oof


class RecordingBackend:
    def fit(self, features: pd.DataFrame, target: pd.Series, *, seed: int):
        return float(target.mean())

    def predict(self, model: float, features: pd.DataFrame):
        return np.full(len(features), model, dtype="float64")


def _fixture(rows: int = 20, matched: int = 20):
    row_ids = [f"r{index}" for index in range(rows)]
    train = pd.DataFrame({
        "row_id": row_ids, "season": [2024] * rows, "game_type": ["R", "F"] * (rows // 2),
        "pitcher_id": [index % 10 for index in range(rows)], "control_success": [index % 2 for index in range(rows)],
    })
    match_rows = []
    for index, row_id in enumerate(row_ids):
        accepted = index < matched
        match_rows.append({
            "row_id": row_id, "trackman_id": index if accepted else pd.NA,
            "lupi_match_accepted": int(accepted), "matched_game_type": "R" if accepted else pd.NA,
            "lupi_match_coverage": 1.0 if accepted else 0.0, "lupi_match_mean_cost": 0.0,
            "lupi_match_candidate_margin": np.inf, "lupi_match_exact_token_agreement": 1.0,
            "lupi_match_exact_token_evidence": 1,
        })
    matches = pd.DataFrame(match_rows).loc[:, MATCH_COLUMNS]
    history = pd.DataFrame({"trackman_id": range(rows), **{
        feature: np.linspace(0.0, 1.0, rows) for feature in TEACHER_FEATURES
    }})
    return train, history, matches


def test_teacher_vector_is_pitcher_held_out_and_nan_for_unmatched() -> None:
    train, history, matches = _fixture(matched=19)
    result = build_teacher_oof(train, history, cutoff_year=2024, backend=RecordingBackend(), matches=matches)
    assert result.probability.shape == (len(train),)
    assert np.isnan(result.probability[-1])
    for train_pitchers, valid_pitchers in result.split_evidence:
        assert set(train_pitchers).isdisjoint(valid_pitchers)


def test_teacher_coverage_gate_skips_only_distillation() -> None:
    train, history, matches = _fixture(matched=2)
    result = build_teacher_oof(train, history, cutoff_year=2024, backend=RecordingBackend(), matches=matches)
    assert result.status == "insufficient_coverage"
    assert result.distillation_allowed is False
    assert np.isnan(result.probability).all()
