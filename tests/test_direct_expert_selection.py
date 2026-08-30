from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.direct_expert.selection import (
    CandidateEvidence,
    decide,
    evidence_from_predictions,
    select_structure_experts,
)


def accepted_stable_evidence() -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id="stable",
        fold_gains={2022: 0.00005, 2023: 0.00005, 2024: 0.00005},
        weighted_gain=0.00005,
        latest_gain=0.00005,
        recent_heavy_gain=0.00005,
        minimum_fold_gain=-0.00003,
        maximum_segment_regression=0.0003,
        bootstrap_lower=0.0,
        latest_bootstrap_lower=0.0,
        non_worse_seed_count=2,
        improving_latest_seed_count=2,
    )


def accepted_aggressive_evidence() -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id="aggressive",
        fold_gains={2022: -0.0002, 2023: 0.00005, 2024: 0.0002},
        weighted_gain=-0.00001,
        latest_gain=0.0002,
        recent_heavy_gain=0.0001,
        minimum_fold_gain=-0.0002,
        maximum_segment_regression=0.0005,
        bootstrap_lower=-0.0002,
        latest_bootstrap_lower=-0.00004,
        non_worse_seed_count=1,
        improving_latest_seed_count=2,
    )


def test_stable_gate_requires_every_boundary() -> None:
    evidence = accepted_stable_evidence()
    assert decide(evidence).status == "accepted_stable"
    assert "weighted_gain" in decide(replace(evidence, weighted_gain=0.000049999)).failed_gates
    assert "minimum_fold_gain" in decide(replace(evidence, minimum_fold_gain=-0.000030001)).failed_gates
    assert "bootstrap_lower" in decide(replace(evidence, bootstrap_lower=-1e-12)).failed_gates


def test_aggressive_gate_does_not_accept_s4_c00() -> None:
    evidence = accepted_aggressive_evidence()
    result = decide(replace(evidence, recent_heavy_gain=0.0000473396))
    assert result.status == "rejected"
    assert result.failed_gates == ("recent_heavy_gain",)


class ExplodingMapping(dict):
    def __iter__(self):
        raise AssertionError("confirmation data was opened")

    def __getitem__(self, key):
        raise AssertionError("confirmation data was opened")


def _structure_frames(prefer: tuple[str, ...] = ()) -> dict[str, pd.DataFrame]:
    output = {}
    target = np.asarray([0, 1, 0, 1] * 4)
    anchor = np.asarray([0.45, 0.55, 0.45, 0.55] * 4)
    for index, expert_id in enumerate(f"D{i}" for i in range(8)):
        bonus = 0.10 if expert_id in prefer else 0.02 + index * 0.001
        probability = anchor + np.where(target == 1, bonus, -bonus)
        output[expert_id] = pd.DataFrame(
            {
                "row_id": [f"r{i}" for i in range(len(target))],
                "target": target,
                "probability": probability,
                "p_anchor": anchor,
                "game_type": ["R", "F"] * 8,
                "pitcher_id": np.repeat(np.arange(8), 2),
                "oof_year": [2022] * 8 + [2023] * 8,
            }
        )
    return output


def test_structure_selection_never_reads_2024() -> None:
    selected = select_structure_experts(
        _structure_frames(),
        confirmation_frames=ExplodingMapping(),
    )
    assert len(selected.expert_ids) == 4
    assert selected.locked_on_years == (2022, 2023)


def test_specialist_selection_is_dependency_closed() -> None:
    selected = select_structure_experts(_structure_frames(prefer=("D5", "D6")))
    assert len(selected.expert_ids) == 4
    assert {"D0", "D5", "D6"}.issubset(selected.expert_ids)


def test_prediction_evidence_aligns_by_row_id() -> None:
    frame = _structure_frames()["D0"]
    evidence = evidence_from_predictions("D0", frame.sample(frac=1, random_state=7))
    assert evidence.fold_gains.keys() == {2022, 2023}
    assert evidence.weighted_gain > 0
