from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from experiments.gated_residual_final.selection import (
    CandidateEvidence,
    decide,
    evidence_from_predictions,
    freeze_archetypes,
)


def _evidence(**changes) -> CandidateEvidence:
    base = CandidateEvidence(
        candidate_id="G1_x",
        fold_gains={2022: 0.00008, 2023: 0.00009, 2024: 0.00006},
        weighted_gain=0.00007,
        latest_gain=0.00006,
        minimum_fold_gain=0.00006,
        bootstrap_lower=0.00001,
        non_worse_seed_count=3,
        latest_non_worse_seed_count=3,
        maximum_segment_regression=0.0,
        finite_probabilities=True,
    )
    return replace(base, **changes)


def test_2024_is_not_passed_to_structure_search() -> None:
    frame = pd.DataFrame({
        "oof_year": [2022, 2023, 2024], "target": [0, 1, 0], "p_anchor": [0.5] * 3,
    })
    seen: list[tuple[int, ...]] = []

    def predictor(part: pd.DataFrame, config) -> np.ndarray:
        seen.append(tuple(sorted(int(year) for year in part["oof_year"].unique())))
        return np.full(len(part), 0.49)

    frozen = freeze_archetypes(frame, predictor=predictor)

    assert {item.archetype for item in frozen} == {"G0", "G1", "G2", "G3"}
    assert all(2024 not in years for years in seen)
    assert all(item.locked_years == (2022, 2023) for item in frozen)


def test_latest_gate_cannot_be_relaxed_by_floating_point() -> None:
    decision = decide(_evidence(latest_gain=-1e-12))

    assert decision.status == "rejected"
    assert "latest_gain" in decision.failed_gates


def test_seed_and_segment_gates_are_mandatory() -> None:
    decision = decide(_evidence(
        non_worse_seed_count=1,
        latest_non_worse_seed_count=1,
        maximum_segment_regression=0.00031,
    ))

    assert {"non_worse_seed_count", "latest_non_worse_seed_count", "maximum_segment_regression"}.issubset(decision.failed_gates)


def test_all_gates_are_required_for_acceptance() -> None:
    assert decide(_evidence()).status == "accepted"
    assert decide(_evidence(bootstrap_lower=-1e-9)).status == "rejected"
    assert decide(_evidence(finite_probabilities=False)).status == "rejected"


def test_evidence_computes_seed_and_segment_gates() -> None:
    rows = []
    for year in (2022, 2023, 2024):
        for index in range(12):
            target = index % 2
            rows.append({
                "row_id": f"{year}_{index}", "target": target,
                "p_anchor": 0.55 if target else 0.45,
                "probability": 0.56 if target else 0.44,
                "oof_year": year, "pitcher_id": f"p{index % 3}",
                "game_type": "R", "hand_matchup": "RL",
            })
    frame = pd.DataFrame(rows)

    evidence = evidence_from_predictions(
        "candidate", frame, seed_predictions={42: frame, 2026: frame, 3407: frame},
        minimum_segment_rows=1, bootstrap_repetitions=100,
    )

    assert evidence.weighted_gain > 0
    assert evidence.non_worse_seed_count == 3
    assert evidence.latest_non_worse_seed_count == 3
    assert evidence.maximum_segment_regression == 0.0


def test_base_and_count_segments_are_included_in_regression_gate() -> None:
    frame = pd.DataFrame({
        "row_id": ["a", "b", "c", "d"], "target": [1, 1, 0, 0],
        "p_anchor": [0.5] * 4, "probability": [0.6, 0.6, 0.6, 0.6],
        "oof_year": [2024] * 4, "pitcher_id": ["p1", "p2", "p3", "p4"],
        "game_type": ["R"] * 4, "hand_matchup": ["RL"] * 4,
        "base_state": ["empty", "empty", "loaded", "loaded"],
        "count_state": ["0-0", "0-0", "3-2", "3-2"],
    })

    evidence = evidence_from_predictions(
        "candidate", frame, seed_predictions={42: frame, 2026: frame, 3407: frame},
        minimum_segment_rows=1, bootstrap_repetitions=50,
    )

    assert evidence.maximum_segment_regression > 0.1
