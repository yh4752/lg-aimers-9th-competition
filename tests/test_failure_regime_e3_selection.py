from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.failure_regime_e3.selection import (
    candidate_evidence,
    decide,
    fit_frozen_recipe,
)


class PassthroughGate:
    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> None:
        assert len(frame) == len(target)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability = frame["p_s_global"].to_numpy(dtype="float64")
        return np.column_stack([1.0 - probability, probability])


def _factory(_seed: int) -> PassthroughGate:
    return PassthroughGate()


def _frame() -> pd.DataFrame:
    records = []
    for year in (2022, 2023, 2024):
        for index in range(60):
            target = (index + year) % 2
            direct = 0.85 if target else 0.15
            records.append(
                {
                    "row_id": f"{year}_{index}",
                    "target": target,
                    "oof_year": year,
                    "pitcher_id": f"p{index % 12}",
                    "game_type": "F" if index % 4 == 0 else "R",
                    "p_e2": 0.52 if target else 0.48,
                    "p_s_global": direct,
                    "feature": float(index % 3),
                }
            )
    return pd.DataFrame(records)


def test_recipe_is_locked_without_reading_2024_values() -> None:
    source = _frame()
    first = fit_frozen_recipe(source, estimator_factory=_factory)
    modified = source.copy(deep=True)
    mask = modified["oof_year"].eq(2024)
    modified.loc[mask, "target"] = 1 - modified.loc[mask, "target"]
    modified.loc[mask, "p_s_global"] = 1 - modified.loc[mask, "p_s_global"]
    second = fit_frozen_recipe(modified, estimator_factory=_factory)

    assert first.recipe == second.recipe
    assert first.recipe.selection_years == (2022, 2023)
    assert first.selection_probability.shape == (120,)


def test_evidence_accepts_only_when_every_temporal_gate_passes() -> None:
    frame = _frame()
    candidate = np.where(frame["target"].to_numpy() == 1, 0.80, 0.20)
    seeds = {seed: candidate.copy() for seed in (42, 2026, 3407)}
    evidence = candidate_evidence(frame, candidate, seed_probabilities=seeds, bootstrap_repetitions=100)

    decision = decide(evidence)

    assert decision.status == "accepted"
    assert decision.failed_gates == ()
    assert evidence.latest_gain >= 0.00015
    assert evidence.minimum_fold_gain >= 0.0
    assert evidence.r_gain >= 0.0
    assert evidence.f_gain >= 0.0


def test_negative_f_segment_blocks_candidate() -> None:
    frame = _frame()
    candidate = np.where(frame["target"].to_numpy() == 1, 0.80, 0.20).astype("float64")
    f_mask = frame["game_type"].eq("F").to_numpy()
    candidate[f_mask] = 1.0 - candidate[f_mask]
    evidence = candidate_evidence(
        frame,
        candidate,
        seed_probabilities={seed: candidate for seed in (42, 2026, 3407)},
        bootstrap_repetitions=100,
        minimum_segment_rows=10,
    )

    decision = decide(evidence)

    assert decision.status == "rejected"
    assert "maximum_segment_regression" in decision.failed_gates
