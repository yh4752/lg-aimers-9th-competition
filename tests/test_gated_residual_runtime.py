from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

import experiments.gated_residual_final.runtime as runtime_module
from experiments.direct_expert.features import fit_direct_features, transform_direct_features
from experiments.gated_residual_final.runtime import evaluate_candidates
from tests.direct_expert_fixtures import make_history_rows, make_train_rows


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


def test_production_feature_state_roundtrips_read_only_mappings() -> None:
    train = make_train_rows()
    prefix = train.loc[train["season"].lt(2024)]
    state, _ = fit_direct_features(prefix, make_history_rows(), valid_year=2024)
    serializer = getattr(runtime_module, "_serialize_feature_state", None)

    assert callable(serializer)
    restored = pickle.loads(serializer(state))
    valid = train.loc[train["season"].eq(2024)].drop(columns="control_success")
    expected = transform_direct_features(valid, state)
    observed = transform_direct_features(valid, restored)

    pd.testing.assert_frame_equal(observed.frame, expected.frame)
    assert dict(restored.source_hashes) == dict(state.source_hashes)
    assert dict(restored.tree_state.source_hashes) == dict(state.tree_state.source_hashes)


def test_audit_rows_are_loaded_from_the_unlabeled_validation_season(tmp_path: Path) -> None:
    test_path = tmp_path / "test.csv"
    rows = make_train_rows().drop(columns="control_success").head(8).copy()
    rows["season"] = 2025
    rows.to_csv(test_path, index=False)
    loader = getattr(runtime_module, "_load_audit_rows", None)

    assert callable(loader)
    observed = loader(test_path, valid_year=2025)

    assert len(observed) == len(rows)
    assert observed["season"].eq(2025).all()
    assert observed["row_id"].is_unique
    assert "control_success" not in observed
