from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from experiments.hierarchical_tabm.calibration import fit_h2, fit_h3
from experiments.hierarchical_tabm.artifacts import DeliveryEvidence, write_candidate_delivery
from experiments.hierarchical_tabm.context_features import fit_context_state
from experiments.hierarchical_tabm.contracts import load_contract
from experiments.hierarchical_tabm.feature_adapter import feature_state_payload, prepare_fold
from experiments.hierarchical_tabm.inference import (
    FrozenHierarchicalPredictor,
    audit_frozen_predictor,
    audit_inference_limits,
    load_candidate_predictor,
)
from experiments.hierarchical_tabm.inputs import EXPECTED_BINDING_KEYS
from experiments.independent_dl.models.tabm import TabMAdapter


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{index}" for index in range(8)],
            "season": [2021] * 5 + [2024] * 3,
            "game_type": ["R", "R", "P", "R", "P", "R", "X", "R"],
            "balls_before": [0, 1, 2, 3, 0, 1, 2, 3],
            "strikes_before": [0, 1, 2, 0, 1, 2, 0, 1],
            "pitcher_hand": ["R", "L", "R", "L", "R", "R", "S", "L"],
            "batter_hand": ["L", "R", "L", "R", "L", "L", "R", "S"],
            "base_state": ["000", "100", "010", "001", "110", "000", "111", "101"],
            "outs_before": [0, 1, 2, 0, 1, 2, 0, 1],
            "pitcher_id": ["p1", "p1", "p2", "p2", "p3", "p1", "new-p", "p2"],
            "batter_id": ["b1", "b2", "b1", "b2", "b3", "b1", "new-b", "b2"],
            "asof_pitcher_n": [10, 11, 8, 9, 7, 12, 0, 10],
            "asof_pitcher_success_rate": [.5, .55, .4, .45, .6, .52, np.nan, .5],
            "asof_batter_n": [9, 8, 7, 6, 5, 10, 0, 8],
            "asof_batter_success_rate": [.4, .45, .5, .55, .6, .42, np.nan, .48],
            "li": [.1, .2, .3, .4, .5, .6, 9.0, .8],
            "control_success": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def _passed_report(candidate_id: str, members: dict[str, Path]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "delivery_role": "final_candidate",
        "passed": True,
        "independence": {
            "row_count": 3,
            "features_exact": True,
            "maximum_probability_delta": 0.0,
            "state_before": "1" * 64,
            "state_after": "1" * 64,
            "batch_sizes": [1, 257, 2048],
        },
        "resources": {
            "row_count": 245789,
            "python_version": "3.11.15",
            "elapsed_seconds": 1.0,
            "peak_gpu_bytes": 1,
            "peak_rss_bytes": 1,
            "artifact_bytes": 1,
            "passed": True,
        },
        "validated_members": {
            name: sha256(path.read_bytes()).hexdigest() for name, path in members.items()
        },
        "failure": None,
    }


def _predictor(kind: str = "H1") -> tuple[FrozenHierarchicalPredictor, pd.DataFrame]:
    rows = _rows()
    train = rows.iloc[:5].copy()
    valid = rows.iloc[5:].copy()
    context = fit_context_state(train, smoothing_k=32.0)
    prepared = prepare_fold(
        train, valid, context_state=context, pitcher_k=100.0, batter_k=250.0
    )

    def raw(x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        score = x_num[:, 0].astype("float64") * 0.05
        if x_cat.shape[1]:
            score += x_cat[:, 0].astype("float64") * 0.01
        return 1.0 / (1.0 + np.exp(-score))

    calibration = None
    if kind == "H2":
        calibration = fit_h2(
            np.array([.2, .4, .6, .8]), np.array([0, 0, 1, 1]),
            regularization=.01, clip=1e-6,
        )
    elif kind == "H3":
        segments = pd.DataFrame(
            {
                "row_id": ["c0", "c1", "c2", "c3"],
                "game_type": ["R", "R", "P", "P"],
                "count_state": ["0_0", "1_1", "0_0", "1_1"],
                "hand_matchup": ["R_L", "L_R", "R_L", "L_R"],
                "base_out_state": ["000_0", "100_1", "000_0", "100_1"],
            }
        )
        calibration = fit_h3(
            np.array([.2, .4, .6, .8]), np.array([0, 0, 1, 1]), segments,
            regularization=.01, clip=1e-6,
        )
    predictor = FrozenHierarchicalPredictor(
        feature_state=prepared.state,
        candidate_id=kind,
        raw_predict=raw,
        model_digest="1" * 64,
        calibration_state=calibration,
        artifact_bytes=1234,
    )
    return predictor, valid


@pytest.mark.parametrize("candidate", ["H1", "H2", "H3"])
def test_frozen_predictor_is_target_blind_and_does_not_mutate(candidate: str) -> None:
    predictor, frame = _predictor(candidate)
    before = predictor.state_digest()
    without_target = predictor.predict_batch(
        frame.drop(columns="control_success"), batch_size=2
    )
    changed_target = frame.assign(control_success=1 - frame["control_success"])
    with_target = predictor.predict_batch(changed_target, batch_size=1)
    assert np.array_equal(without_target, with_target)
    assert np.isfinite(without_target).all()
    assert ((without_target >= 0) & (without_target <= 1)).all()
    assert predictor.state_digest() == before


def test_prediction_is_row_order_batch_and_population_invariant() -> None:
    predictor, frame = _predictor("H3")
    report = audit_frozen_predictor(
        predictor, frame.drop(columns="control_success"), batch_sizes=(1, 2, 257)
    )
    assert report.features_exact is True
    assert report.maximum_probability_delta <= 1e-12
    assert report.state_before == report.state_after

    one = predictor.predict_batch(frame.iloc[[0]], batch_size=1)[0]
    changed = pd.concat(
        [frame.iloc[::-1], frame.iloc[[1]].assign(row_id="unrelated")],
        ignore_index=True,
    )
    current = predictor.predict_batch(changed, batch_size=3)
    row_position = changed.index[changed["row_id"].eq(frame.iloc[0]["row_id"])][0]
    assert current[row_position] == pytest.approx(one, abs=1e-12)


def test_prediction_path_does_not_use_population_operations(monkeypatch) -> None:
    predictor, frame = _predictor("H1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("population-dependent operation called")

    monkeypatch.setattr(pd.DataFrame, "groupby", forbidden)
    monkeypatch.setattr(pd.DataFrame, "rank", forbidden)
    monkeypatch.setattr(pd.DataFrame, "rolling", forbidden)
    monkeypatch.setattr(pd.DataFrame, "cumsum", forbidden)
    monkeypatch.setattr(pd.Series, "groupby", forbidden)
    monkeypatch.setattr(pd.Series, "rank", forbidden)
    monkeypatch.setattr(pd.Series, "rolling", forbidden)
    monkeypatch.setattr(pd.Series, "cumsum", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    probability = predictor.predict_batch(frame.drop(columns="control_success"), batch_size=2)
    assert probability.shape == (len(frame),)


def test_scale_gate_uses_sealed_limits() -> None:
    predictor, frame = _predictor("H1")
    report = audit_inference_limits(
        predictor,
        frame.drop(columns="control_success"),
        load_contract(),
        resource_probe=lambda: (120.0, 2_000_000_000, 3_000_000_000),
    )
    assert report.passed is True
    assert report.row_count == len(frame)
    assert report.artifact_bytes == 1234

    failed = audit_inference_limits(
        predictor,
        frame.drop(columns="control_success"),
        load_contract(),
        resource_probe=lambda: (481.0, 2_000_000_000, 3_000_000_000),
    )
    assert failed.passed is False


def test_invalid_batch_size_and_duplicate_row_id_are_rejected() -> None:
    predictor, frame = _predictor("H1")
    with pytest.raises(ValueError, match="batch_size"):
        predictor.predict_batch(frame, batch_size=0)
    duplicate = frame.copy()
    duplicate.loc[duplicate.index[1], "row_id"] = duplicate.iloc[0]["row_id"]
    with pytest.raises(ValueError, match="row_id"):
        predictor.predict_batch(duplicate)


def test_candidate_delivery_round_trips_through_real_tabm_adapter(tmp_path: Path) -> None:
    rows = _rows()
    train = rows.iloc[:5].copy()
    valid = rows.iloc[5:].copy()
    prepared = prepare_fold(
        train,
        valid,
        context_state=fit_context_state(train, smoothing_k=32.0),
        pitcher_k=100.0,
        batter_k=250.0,
    )
    contract = load_contract()
    adapter = TabMAdapter("bce")
    model = adapter.build(
        {
            "architecture": "tabm",
            "k": contract.model.k,
            "width": contract.model.width,
            "blocks": contract.model.blocks,
            "dropout": contract.model.dropout,
            "num_embedding": contract.model.num_embedding,
        },
        prepared.metadata,
        "cpu",
    )
    torch = __import__("torch")
    source = tmp_path / "source"; source.mkdir()
    checkpoint = source / "final_checkpoint.pt"
    torch.save({"model": model.state_dict(), "epoch": 0}, checkpoint)
    feature = source / "feature_state.json"
    feature.write_text(
        json.dumps(feature_state_payload(prepared.state), sort_keys=True, separators=(",", ":"))
    )
    bindings = {key: sha256(key.encode()).hexdigest() for key in EXPECTED_BINDING_KEYS}
    bindings["train_sha256"] = contract.official_train_sha256
    bindings["history_sha256"] = contract.official_history_sha256
    bindings["stage_c_delivery_sha256"] = contract.source_stage_c_delivery_sha256
    report = source / "decision_H1.json"
    report.write_text(json.dumps(_passed_report("H1", {
        "model/final_checkpoint.pt": checkpoint,
        "state/feature_state.json": feature,
    })))
    delivery = write_candidate_delivery(
        DeliveryEvidence(
            MappingProxyType(bindings), MappingProxyType({"H1": "final_candidate"}),
            checkpoint, feature, MappingProxyType({}),
            MappingProxyType({"H1": report}),
        ),
        tmp_path / "delivery",
    )

    predictor = load_candidate_predictor(
        delivery, candidate_id="H1", device="cpu", expected_bindings=bindings
    )
    probability = predictor.predict_batch(valid.drop(columns="control_success"), batch_size=2)
    assert probability.shape == (len(valid),)
    assert np.isfinite(probability).all()
    resources = audit_inference_limits(
        predictor, valid.drop(columns="control_success"), contract
    )
    assert resources.passed is True
