import numpy as np
import pandas as pd

from experiments.tree_expert.hc_contracts import load_hc_contract
from experiments.tree_expert.hc_evaluation import build_c1_evidence, build_c2_evidence


def _frame() -> pd.DataFrame:
    records = []
    for year in (2022, 2023, 2024):
        for index in range(40):
            target = float(index % 2)
            p0 = 0.55 if target else 0.45
            records.append(
                {
                    "row_id": f"r{year}_{index}",
                    "oof_year": year,
                    "target": target,
                    "p0": p0,
                    "p1": 0.60 if target else 0.40,
                    "p2": 0.62 if target else 0.38,
                    "pitcher_id": f"p{index % 5}",
                    "game_type": "R" if index % 3 else "F",
                    "pitcher_id_known": "known",
                    "batter_id_known": "known",
                    "pitcher_hand": "R",
                    "batter_hand": "L",
                    "balls_before": index % 4,
                    "strikes_before": index % 3,
                    "base_state": str(index % 4),
                }
            )
    return pd.DataFrame(records)


def test_candidate_evidence_is_paired_and_seed_consistent():
    frame = _frame()
    seeds = {
        seed: frame.loc[:, ["row_id", "oof_year", "target", "p0"]].assign(
            p1=frame["p1"].to_numpy() + offset
        )
        for seed, offset in ((3407, 0.0), (42, 0.001), (2026, -0.001))
    }
    c1 = build_c1_evidence(frame.drop(columns="p2"), seeds, load_hc_contract())
    c2 = build_c2_evidence(frame, load_hc_contract())
    assert c1.weighted_gain > 0
    assert c1.confirmation_gain > 0
    assert c1.non_worse_seed_counts == {2022: 3, 2023: 3, 2024: 3}
    assert c2.weighted_gain > c1.weighted_gain
    assert c2.incremental_gain > 0
    assert np.isfinite(c1.bootstrap_lower_95)
