from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
import json

import pandas as pd
import pytest

from experiments.temporal_portfolio.seasonal_features import fit_s1_state
from experiments.tree_expert.e2_decisions import AcceptanceDecision
from experiments.tree_expert.e2_full_fit import (
    E2FullFitError,
    accepted_full_fit_token,
    export_frozen_tree_state,
    fit_full_seed,
    full_fit_iterations,
    load_frozen_tree_state,
)
from experiments.tree_expert.features import TreeFeatureBatch, TreeFeatureState


def _s1_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023],
            "pitcher_id": [1, 2],
            "batter_id": [11, 12],
            "asof_pitcher_n": [10, 20],
            "asof_pitcher_success_rate": [0.5, 0.6],
            "asof_pitcher_reverse_rate": [0.1, 0.2],
            "asof_pitcher_middle_rate": [0.2, 0.2],
            "asof_pitcher_ball_rate": [0.4, 0.3],
            "asof_pitcher_strike_rate": [0.6, 0.7],
            "asof_pitcher_pitchmix_n": [10, 20],
            "asof_pitcher_fastball_rate": [0.5, 0.6],
            "asof_pitcher_breaking_rate": [0.3, 0.2],
            "asof_pitcher_offspeed_rate": [0.2, 0.2],
            "asof_batter_n": [5, 8],
            "asof_batter_success_rate": [0.4, 0.5],
            "asof_batter_middle_rate": [0.2, 0.25],
            "control_success": [0, 1],
        }
    )


def _tree_state() -> TreeFeatureState:
    s1 = fit_s1_state(_s1_rows(), valid_year=2024)
    return TreeFeatureState(
        valid_year=2024,
        prior_rate=s1.prior_rate,
        categorical_columns=("pitcher_id",),
        feature_columns=("pitcher_id", "value"),
        s1_state=s1,
        trackman_state=None,
        source_hashes=MappingProxyType({"S1": "a" * 64}),
    )


def _acceptance(status: str = "accepted") -> AcceptanceDecision:
    return AcceptanceDecision(
        status=status,
        predictor="catboost",
        weighted_gain=0.0002,
        f3_gain=0.0002,
        worst_fold_gain=0.0001,
        bootstrap_lower=0.00001,
        bootstrap_upper=0.0003,
        maximum_segment_regression=0.0001,
        performance_grade="incremental",
        reason="standalone_catboost_gates_passed" if status == "accepted" else "failed",
    )


def test_full_fit_iterations_use_median_plus_one_and_clip() -> None:
    assert full_fit_iterations((3, 0, 8)) == 50
    assert full_fit_iterations((90, 120, 100)) == 101
    assert full_fit_iterations((500, 410, 430)) == 400


def test_rejected_acceptance_cannot_create_full_fit_token() -> None:
    with pytest.raises(E2FullFitError, match="not accepted"):
        accepted_full_fit_token(
            _acceptance("rejected"),
            candidate_id="c1_anchor_residual",
            best_iterations={42: (1, 2, 3), 2026: (1, 2, 3), 3407: (1, 2, 3)},
            decision_sha256="b" * 64,
        )


def test_token_fixes_three_seed_iteration_counts() -> None:
    token = accepted_full_fit_token(
        _acceptance(),
        candidate_id="c1_anchor_residual",
        best_iterations={42: (60, 80, 70), 2026: (90, 100, 110), 3407: (3, 0, 8)},
        decision_sha256="b" * 64,
    )

    assert token.seeds == (42, 2026, 3407)
    assert token.iterations == {42: 71, 2026: 101, 3407: 50}


def test_frozen_c1_state_round_trips_without_trackman(tmp_path: Path) -> None:
    state = _tree_state()

    export_frozen_tree_state(
        state,
        tmp_path / "state",
        candidate_id="c1_anchor_residual",
    )
    restored = load_frozen_tree_state(tmp_path / "state")

    assert restored.valid_year == state.valid_year
    assert restored.prior_rate == state.prior_rate
    assert restored.feature_columns == state.feature_columns
    assert restored.categorical_columns == state.categorical_columns
    assert restored.trackman_state is None
    pd.testing.assert_frame_equal(
        restored.s1_state.snapshot.pitcher,
        state.s1_state.snapshot.pitcher,
        check_dtype=True,
    )
    pd.testing.assert_frame_equal(
        restored.s1_state.snapshot.batter,
        state.s1_state.snapshot.batter,
        check_dtype=True,
    )
    assert not (tmp_path / "state" / "trackman_lookup.csv").exists()


def test_c2_state_requires_cutoff_2024_trackman_lookup(tmp_path: Path) -> None:
    with pytest.raises(E2FullFitError, match="TrackMan state"):
        export_frozen_tree_state(
            _tree_state(),
            tmp_path / "state",
            candidate_id="c2_trackman_residual",
        )


def test_frozen_state_rejects_tampered_snapshot(tmp_path: Path) -> None:
    export_frozen_tree_state(
        _tree_state(),
        tmp_path / "state",
        candidate_id="c1_anchor_residual",
    )
    with (tmp_path / "state" / "s1_pitcher.csv").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(E2FullFitError, match="SHA-256"):
        load_frozen_tree_state(tmp_path / "state")


def test_loader_rejects_c2_identity_without_trackman(tmp_path: Path) -> None:
    root = export_frozen_tree_state(
        _tree_state(),
        tmp_path / "state",
        candidate_id="c1_anchor_residual",
    )
    manifest_path = root / "feature_state.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidate_id"] = "c2_trackman_residual"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(E2FullFitError, match="candidate and TrackMan"):
        load_frozen_tree_state(root)


def test_full_seed_fit_uses_fixed_iterations_and_residual_target(tmp_path: Path) -> None:
    token = accepted_full_fit_token(
        _acceptance(),
        candidate_id="c1_anchor_residual",
        best_iterations={42: (60, 80, 70), 2026: (90, 100, 110), 3407: (3, 0, 8)},
        decision_sha256="b" * 64,
    )
    state = _tree_state()
    batch = TreeFeatureBatch(
        frame=pd.DataFrame({"pitcher_id": ["1", "2"], "value": [0.2, 0.3]}),
        anchor=pd.Series([0.4, 0.6]).to_numpy(),
        row_id=pd.Series(["r1", "r2"]).to_numpy(),
        target=pd.Series([0, 1]).to_numpy(),
    )
    captured: dict[str, object] = {}

    class FakeModel:
        def fit(self, frame, target, **kwargs):
            captured["frame"] = frame.copy()
            captured["target"] = target.copy()
            captured["fit"] = kwargs

        def save_model(self, path: str) -> None:
            Path(path).write_bytes(b"model")

    def model_factory(parameters: dict[str, object]):
        captured["parameters"] = parameters
        return FakeModel()

    result = fit_full_seed(
        token=token,
        seed=42,
        state=state,
        batch=batch,
        output_dir=tmp_path,
        gpu_id=0,
        model_factory=model_factory,
    )

    assert captured["parameters"]["iterations"] == 71
    assert captured["parameters"]["random_seed"] == 42
    assert captured["parameters"]["devices"] == "0"
    assert list(captured["target"]) == pytest.approx([-0.4, 0.4])
    assert captured["fit"]["use_best_model"] is False
    assert result.model_path.is_file()
