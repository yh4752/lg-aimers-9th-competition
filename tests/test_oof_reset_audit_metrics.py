from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.oof_reset_audit.metrics import (
    block_bootstrap_interval,
    calibration_deciles,
    paired_metrics,
    segment_metrics,
    single_model_metrics,
    validate_prediction_frame,
)
from experiments.oof_reset_audit.types import PredictionSet, TrustClass


def _frame(probability: list[float], *, reverse: bool = False) -> pd.DataFrame:
    value = pd.DataFrame(
        {
            "row_id": [f"r{index}" for index in range(8)],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "probability": probability,
            "game_month": [4, 4, 5, 5, 6, 6, 7, 7],
            "game_type": ["R", "R", "F", "F", "R", "R", "F", "F"],
        }
    )
    return value.iloc[::-1].reset_index(drop=True) if reverse else value


def _prediction(
    model_id: str, values: list[float], *, reverse: bool = False
) -> PredictionSet:
    return PredictionSet(
        artifact_path=Path(f"{model_id}.zip"),
        artifact_sha256="a" * 64,
        source_member=f"predictions/{model_id}.csv",
        prediction_sha256="b" * 64,
        model_id=model_id,
        fold="2023->2024",
        trust=TrustClass.RULE_SAFE,
        frame=_frame(values, reverse=reverse),
    )


def test_validation_rejects_duplicate_ids_targets_and_probability_range() -> None:
    duplicate = _frame([0.1] * 8)
    duplicate.loc[1, "row_id"] = "r0"
    with pytest.raises(ValueError, match="row_id must be unique"):
        validate_prediction_frame(duplicate)

    invalid_target = _frame([0.1] * 8)
    invalid_target.loc[1, "target"] = 2
    with pytest.raises(ValueError, match="target"):
        validate_prediction_frame(invalid_target)

    invalid_probability = _frame([0.1] * 7 + [1.1])
    with pytest.raises(ValueError, match="probability"):
        validate_prediction_frame(invalid_probability)


def test_single_and_paired_metrics_have_exact_values_and_ignore_row_order() -> None:
    anchor = _prediction(
        "anchor", [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55]
    )
    candidate = _prediction(
        "candidate", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6], reverse=True
    )

    single = single_model_metrics(anchor)
    paired = paired_metrics(anchor, candidate)

    assert math.isclose(single["brier"], 0.123125, abs_tol=1e-12)
    assert math.isclose(single["local_bss"], 50_750.0, abs_tol=1e-9)
    assert math.isclose(paired["candidate_brier"], 0.075, abs_tol=1e-12)
    assert math.isclose(paired["gain_vs_anchor"], 0.048125, abs_tol=1e-12)
    assert paired["rows"] == 8


def test_paired_metrics_reject_different_row_sets_and_targets() -> None:
    anchor = _prediction("anchor", [0.2] * 8)
    different_rows = _prediction("candidate", [0.2] * 8)
    different_rows.frame.loc[7, "row_id"] = "different"
    with pytest.raises(ValueError, match="row_id set differs"):
        paired_metrics(anchor, different_rows)

    different_target = _prediction("candidate", [0.2] * 8)
    different_target.frame.loc[6, "target"] = 1
    with pytest.raises(ValueError, match="target differs"):
        paired_metrics(anchor, different_target)


def test_calibration_segments_and_block_interval_are_deterministic() -> None:
    anchor = _prediction(
        "anchor", [0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95]
    )
    candidate = _prediction(
        "candidate", [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    )
    calibration = calibration_deciles(anchor)
    segments = segment_metrics(anchor, candidate, minimum_rows=2)
    assert calibration["rows"].sum() == 8
    assert set(segments["segment"]) == {"game_month", "game_type"}
    assert bool(segments["eligible"].all())

    losses = pd.DataFrame(
        {
            "block": [f"2024-{month}" for month in range(4, 10) for _ in range(2)],
            "loss_delta": np.arange(12, dtype="float64") / 1000,
        }
    )
    first = block_bootstrap_interval(losses)
    second = block_bootstrap_interval(losses)
    assert first == second
    assert first["status"] == "completed"
    assert first["block_count"] == 6
    assert block_bootstrap_interval(losses[losses.block != "2024-9"])["status"] == (
        "insufficient_blocks"
    )
