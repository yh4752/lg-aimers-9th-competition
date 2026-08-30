from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.gated_residual_final.runtime import evaluate_candidates


def _seed_frame(seed: int) -> pd.DataFrame:
    rows = []
    for year in (2022, 2023, 2024):
        for index in range(24):
            target = index % 2
            rows.append({
                "row_id": f"{year}_{index}", "target": target, "p_anchor": 0.5,
                "p_d0": 0.8 if target else 0.2,
                "p_d5": 0.82 if target else 0.18,
                "p_direct": 0.82 if target else 0.18,
                "oof_year": year, "game_type": "R", "pitcher_id": f"p{index % 4}",
                "batter_id": f"b{index % 6}", "pitcher_hand": "R", "batter_hand": "L",
                "hand_matchup": "RL", "reliability_k25": 0.8,
                "reliability_k100": 0.5, "reliability_k400": 0.2, "seed": seed,
            })
    return pd.DataFrame(rows)


def test_candidate_evaluation_writes_complete_acceptance_evidence(tmp_path: Path) -> None:
    frames = {seed: _seed_frame(seed) for seed in (42, 2026, 3407)}

    outcome, frozen = evaluate_candidates(frames, tmp_path, bootstrap_repetitions=100)

    assert outcome.decision.status == "accepted"
    assert len(frozen) == 4
    payload = json.loads(outcome.evidence_path.read_text())
    assert payload["selection_years"] == [2022, 2023]
    assert payload["confirmation_year"] == 2024
    assert len(payload["candidates"]) == 4
    assert payload["chosen_candidate_id"] == outcome.decision.candidate_id


def test_candidate_evaluation_records_hard_rejection(tmp_path: Path) -> None:
    frames = {}
    for seed in (42, 2026, 3407):
        frame = _seed_frame(seed).assign(p_direct=0.5, p_d0=0.5, p_d5=0.5)
        frame["p_anchor"] = frame["target"].astype(float)
        frames[seed] = frame

    outcome, _ = evaluate_candidates(frames, tmp_path, bootstrap_repetitions=50)

    assert outcome.decision.status == "rejected"
    assert "weighted_gain" in outcome.decision.failed_gates
    assert json.loads(outcome.evidence_path.read_text())["status"] == "completed_no_candidate"
