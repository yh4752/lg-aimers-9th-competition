from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from experiments.tree_expert.e2_full_fit import export_frozen_tree_state
from experiments.tree_expert.features import fit_tree_features, transform_tree_features


def _fixture_rows() -> pd.DataFrame:
    namespace = runpy.run_path(str(Path(__file__).with_name("test_tree_expert_features.py")))
    return namespace["_rows"]()


def test_standalone_features_match_reviewed_e2_transform(tmp_path: Path) -> None:
    from submission import tree_expert_e2_script as runtime

    rows = _fixture_rows()
    train = rows.loc[rows["season"].lt(2024)].copy(deep=True)
    valid = rows.loc[rows["season"].eq(2024)].drop(columns="control_success")
    state, _ = fit_tree_features(
        train, history=None, valid_year=2024, use_trackman=False
    )
    frozen = export_frozen_tree_state(
        state, tmp_path / "frozen_state", candidate_id="c1_anchor_residual"
    )
    expected = transform_tree_features(valid, state)

    restored = runtime.load_frozen_state(frozen)
    frame, anchor, row_id = runtime.transform_features(valid, restored)

    pd.testing.assert_frame_equal(frame, expected.frame)
    np.testing.assert_array_equal(row_id, expected.row_id)
    np.testing.assert_allclose(anchor, expected.anchor, rtol=0, atol=1e-12)


def test_predictor_is_row_order_batch_and_singleton_independent(tmp_path: Path) -> None:
    from submission import tree_expert_e2_script as runtime

    rows = _fixture_rows()
    train = rows.loc[rows["season"].lt(2024)].copy(deep=True)
    valid = rows.loc[rows["season"].eq(2024)].drop(columns="control_success")
    state, _ = fit_tree_features(
        train, history=None, valid_year=2024, use_trackman=False
    )
    frozen = export_frozen_tree_state(
        state, tmp_path / "frozen_state", candidate_id="c1_anchor_residual"
    )

    class ResidualModel:
        def __init__(self, offset: float) -> None:
            self.offset = offset

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return frame["inning"].to_numpy(dtype="float64") * 0.001 + self.offset

    predictor = runtime.TreeE2Predictor(
        runtime.load_frozen_state(frozen),
        (ResidualModel(-0.01), ResidualModel(0.0), ResidualModel(0.01)),
        state_sha256="a" * 64,
    )
    baseline = predictor.predict_batch(valid, batch_size=4096)
    reverse = valid.iloc[::-1].reset_index(drop=True)
    reversed_values = predictor.predict_batch(reverse, batch_size=1)
    by_id = dict(zip(reverse["row_id"].astype(str), reversed_values, strict=True))

    np.testing.assert_allclose(
        baseline,
        [by_id[row_id] for row_id in valid["row_id"].astype(str)],
        rtol=0,
        atol=1e-12,
    )
    for index in range(len(valid)):
        one = predictor.predict_batch(valid.iloc[[index]], batch_size=1)
        np.testing.assert_allclose(one, baseline[[index]], rtol=0, atol=1e-12)
    assert predictor.state_digest() == "a" * 64


def test_bound_runtime_metadata_is_exact() -> None:
    from submission.tree_expert_e2_candidate import (
        TREE_E2_ADAPTER_ID,
        TREE_E2_CANDIDATE_ID,
        render_bound_script,
    )

    members = {
        "frozen_state/feature_state.json": "1" * 64,
        "frozen_state/s1_batter.csv": "2" * 64,
        "frozen_state/s1_pitcher.csv": "3" * 64,
        "models/catboost_seed_42.cbm": "4" * 64,
        "models/catboost_seed_2026.cbm": "5" * 64,
        "models/catboost_seed_3407.cbm": "6" * 64,
    }
    metadata = {
        "candidate_id": TREE_E2_CANDIDATE_ID,
        "adapter_id": TREE_E2_ADAPTER_ID,
        "handoff_sha256": "7" * 64,
        "delivery_sha256": "8" * 64,
        "delivery_manifest_sha256": "9" * 64,
        "model_sha256": "a" * 64,
        "members": members,
        "seeds": [42, 2026, 3407],
        "iterations": {"42": 78, "2026": 86, "3407": 149},
    }

    source = render_bound_script(metadata)
    namespace = {"__name__": "tree_e2_rendered"}
    exec(compile(source, "script.py", "exec"), namespace)

    assert namespace["EMBEDDED_METADATA"] == json.loads(
        json.dumps(metadata, sort_keys=True)
    )


def test_e2_adapter_is_registered_and_rendered() -> None:
    from submission.adapters import resolve_adapter_factory
    from submission.runtime import render_script
    from submission.tree_expert_e2_candidate import TREE_E2_ADAPTER_ID

    metadata = {
        "candidate_id": "c1_anchor_residual",
        "adapter_id": TREE_E2_ADAPTER_ID,
        "handoff_sha256": "7" * 64,
        "delivery_sha256": "8" * 64,
        "delivery_manifest_sha256": "9" * 64,
        "model_sha256": "a" * 64,
        "members": {
            "frozen_state/feature_state.json": "1" * 64,
            "frozen_state/s1_batter.csv": "2" * 64,
            "frozen_state/s1_pitcher.csv": "3" * 64,
            "models/catboost_seed_42.cbm": "4" * 64,
            "models/catboost_seed_2026.cbm": "5" * 64,
            "models/catboost_seed_3407.cbm": "6" * 64,
        },
        "seeds": [42, 2026, 3407],
        "iterations": {"42": 78, "2026": 86, "3407": 149},
    }

    assert callable(resolve_adapter_factory(TREE_E2_ADAPTER_ID))
    source = render_script(adapter_id=TREE_E2_ADAPTER_ID, artifact_metadata=metadata)
    assert b"c1_anchor_residual" in source
