from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd

from experiments.tree_privileged.decisions import (
    ScreenCandidate, ScreenEvidence, decide_candidate, select_confirmation_candidates, select_rf_blend,
)


def test_screen_keeps_at_most_two_diverse_candidates() -> None:
    gains = {"P": .00008, "D15": .00007, "D35": -.00001, "PD15": .00012, "PD35": .00011}
    candidates = {name: ScreenCandidate(name, gain, gain, gain, (gain, gain)) for name, gain in gains.items()}
    correlations = {tuple(sorted(("PD15", "PD35"))): .999, tuple(sorted(("PD15", "P"))): .991}
    evidence = ScreenEvidence(MappingProxyType(candidates), MappingProxyType(correlations))
    assert select_confirmation_candidates(evidence) == ("PD15", "P")


def _confirmation(latest_gain: float) -> pd.DataFrame:
    rows = []
    for seed in (42, 2026, 3407):
        for year in (2022, 2023, 2024):
            for index, target in enumerate((0, 1) * 100):
                baseline = .5
                gain = .00008 if year != 2024 else latest_gain
                # A symmetric move toward/away from the label with approximately requested gain.
                delta = np.sign(target - .5) * gain
                rows.append({"candidate_id": "PD15", "valid_year": year, "seed": seed,
                             "row_id": f"{seed}-{year}-{index}", "target": target,
                             "baseline_probability": baseline, "probability": baseline + delta,
                             "game_type": "R" if index % 2 else "F"})
    return pd.DataFrame(rows)


def test_acceptance_requires_weighted_recent_fold_seed_and_segment_gates() -> None:
    accepted = decide_candidate(_confirmation(.00008))
    rejected = decide_candidate(_confirmation(-.00003))
    assert accepted.status == "accepted"
    assert rejected.status == "rejected"
    assert "latest_min_gain" in rejected.failed_gates


def test_rf_alpha_is_chosen_on_2022_2023_only() -> None:
    source = _confirmation(.00008)
    first = select_rf_blend(source)
    changed = source.copy()
    changed.loc[changed["valid_year"].eq(2024), "target"] ^= 1
    assert select_rf_blend(changed) == first
