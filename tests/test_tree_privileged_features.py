from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import experiments.tree_privileged.features as feature_module
from experiments.tree_expert.features import TreeFeatureBatch
from experiments.tree_privileged.features import fit_candidate_features, transform_candidate_features
from experiments.tree_privileged.teacher import TeacherEvidence


def _rows() -> pd.DataFrame:
    rows = []
    for season in (2021, 2022):
        for index in range(30):
            rows.append({
                "row_id": f"{season}-{index}", "season": season, "pitcher_id": index % 3,
                "batter_id": index % 5, "balls_before": index % 4, "strikes_before": index % 3,
                "batter_hand": "L", "pitcher_hand": "R", "base_state": str(index % 2),
                "game_type": "R", "pitcher_team_id": 1, "control_success": index % 2,
            })
    return pd.DataFrame(rows)


def _base_builder(train, _history, *, valid_year, use_trackman):
    state = SimpleNamespace(categorical_columns=("cat",))
    frame = pd.DataFrame({"cat": ["x"] * len(train), "numeric": np.arange(len(train), dtype="float32")})
    return state, TreeFeatureBatch(frame, np.full(len(train), 0.4), train["row_id"].to_numpy(), train["control_success"].to_numpy())


def _teacher(train, history, *, cutoff_year):
    probability = np.linspace(0.1, 0.9, len(train)); probability[-1] = np.nan
    mask = np.isfinite(probability)
    return TeacherEvidence(probability, mask, float(mask.mean()), 0.9, MappingProxyType({}), "ready", True,
                           (), "1" * 64, "2" * 64, "3" * 64)


def test_candidate_feature_and_target_boundaries(monkeypatch) -> None:
    fit_rows = _rows()
    state, batch = fit_candidate_features(
        fit_rows, pd.DataFrame(), valid_year=2023, candidate_id="PD35",
        base_builder=_base_builder, teacher_builder=_teacher,
    )
    assert state.profile_columns and all(name.startswith("profile_") for name in state.profile_columns)
    expected = fit_rows["control_success"].to_numpy(dtype="float64")
    mask = np.isfinite(batch.teacher_probability)
    expected[mask] = 0.65 * expected[mask] + 0.35 * batch.teacher_probability[mask]
    assert_allclose(batch.soft_target, expected)

    def fake_transform(rows, _state):
        return TreeFeatureBatch(
            pd.DataFrame({"cat": ["x"] * len(rows), "numeric": np.arange(len(rows), dtype="float32")}),
            np.full(len(rows), 0.4), rows["row_id"].to_numpy(), None,
        )
    monkeypatch.setattr(feature_module, "transform_tree_features", fake_transform)
    valid = fit_rows.iloc[:2].drop(columns="control_success")
    transformed = transform_candidate_features(valid, state)
    assert transformed.soft_target is None
    assert transformed.teacher_probability is None


def test_inference_rejects_current_pitch_trackman() -> None:
    fit_rows = _rows()
    state, _ = fit_candidate_features(
        fit_rows, pd.DataFrame(), valid_year=2023, candidate_id="P", base_builder=_base_builder,
    )
    invalid = fit_rows.iloc[:1].drop(columns="control_success")
    invalid["rel_speed"] = 145.0
    try:
        transform_candidate_features(invalid, state)
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("current-pitch TrackMan was accepted at inference")

