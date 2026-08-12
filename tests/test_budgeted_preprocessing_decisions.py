from __future__ import annotations

import numpy as np
import pytest

from experiments.preprocessing_campaign.budgeted_decisions import (
    DecisionError,
    choose_dl_representative,
    decide_tabnet,
    final_preprocessing_status,
    fixed_blend_metrics,
    promote_single,
)


def _metric(
    family: str,
    *,
    brier: float,
    seconds: float = 2500.0,
    valid_epochs: int = 30,
    validation_points: int = 30,
) -> dict[str, object]:
    return {
        "candidate_id": family,
        "family": family,
        "brier": brier,
        "elapsed_seconds": seconds,
        "completed_epochs": valid_epochs,
        "validation_points": validation_points,
        "hashes_valid": True,
    }


def test_model_tie_chooses_faster_candidate_within_two_e_minus_four() -> None:
    rows = [
        _metric("tabm", brier=0.24750, seconds=2500),
        _metric("ft_transformer", brier=0.24738, seconds=2700),
    ]

    decision = choose_dl_representative(rows, blend_rows=[])

    assert decision.selected_family == "tabm"
    assert decision.reason == "brier_tie_faster"


def test_tabnet_under_minimum_training_is_inconclusive() -> None:
    result = decide_tabnet(
        _metric("tabnet", brier=0.2470, valid_epochs=9),
        best_dl_brier=0.2472,
        blend_gain=0.0,
        oov_gain=0.0,
        overall_delta=-0.0002,
    )

    assert result.status == "inconclusive"


def test_tabnet_advances_when_fixed_blend_gain_crosses_threshold() -> None:
    result = decide_tabnet(
        _metric("tabnet", brier=0.2480),
        best_dl_brier=0.2472,
        blend_gain=0.00011,
        oov_gain=0.0,
        overall_delta=0.0004,
    )

    assert result.status == "promoted"
    assert result.reason == "blend_gain"


def test_preprocessing_requires_preregistered_gain_or_segment_or_blend_rule() -> None:
    assert promote_single(delta=-0.00011, oov_delta=0.0, blend_gain=0.0)
    assert promote_single(delta=-0.00004, oov_delta=-0.00021, blend_gain=0.0)
    assert promote_single(delta=0.00004, oov_delta=0.0, blend_gain=0.00011)
    assert not promote_single(delta=0.00006, oov_delta=-0.001, blend_gain=0.001)


def test_fixed_blend_rejects_row_misalignment() -> None:
    with pytest.raises(DecisionError, match="row alignment"):
        fixed_blend_metrics(
            row_id=np.array(["a", "b"]),
            target=np.array([0.0, 1.0]),
            catboost_row_id=np.array(["b", "a"]),
            catboost_probability=np.array([0.3, 0.8]),
            candidate_probability=np.array([0.2, 0.9]),
            weights=(0.1, 0.25, 0.5),
        )


def test_final_recommendation_requires_two_fold_direction_full_gain_and_segments() -> None:
    decision = final_preprocessing_status(
        proxy_deltas={2023: -0.00012, 2024: -0.00014},
        proxy_rows={2023: 245_525, 2024: 253_507},
        full_2024_delta=-0.00003,
        segment_deltas={
            "pitcher_oov": 0.0001,
            "batter_oov": 0.0,
            "game_type": 0.00019,
        },
        hashes_valid=True,
    )

    assert decision == "recommended"


def test_final_status_is_inconclusive_when_only_one_proxy_fold_improves() -> None:
    decision = final_preprocessing_status(
        proxy_deltas={2023: 0.00001, 2024: -0.0003},
        proxy_rows={2023: 245_525, 2024: 253_507},
        full_2024_delta=-0.0001,
        segment_deltas={"pitcher_oov": 0.0, "batter_oov": 0.0, "game_type": 0.0},
        hashes_valid=True,
    )

    assert decision == "inconclusive"
