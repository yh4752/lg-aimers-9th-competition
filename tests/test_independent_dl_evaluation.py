from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.evaluation import (
    EvaluationError,
    evaluate_aligned_blends,
    evaluate_predictions,
    select_survivors,
    write_evaluation_artifacts,
)


def _oof(probability: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d", "e", "f"],
            "fold": ["valid_2022"] * 2
            + ["valid_2023"] * 2
            + ["valid_2024"] * 2,
            "season": [2022, 2022, 2023, 2023, 2024, 2024],
            "game_type": ["R", "F", "R", "F", "R", "F"],
            "target": [0, 1, 0, 1, 0, 1],
            "probability": probability,
        }
    )


def test_prediction_metrics_cover_global_folds_and_game_type() -> None:
    report = evaluate_predictions(_oof([0.1, 0.8, 0.2, 0.7, 0.3, 0.6]))

    assert report["global"]["n"] == 6
    assert set(report["folds"]) == {"valid_2022", "valid_2023", "valid_2024"}
    assert set(report["game_type"]) == {"F", "R"}
    assert report["global"]["brier"] == pytest.approx(
        np.mean(np.square(np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.6]) - [0, 1, 0, 1, 0, 1]))
    )


def test_blend_requires_exact_row_and_fold_alignment() -> None:
    dl = _oof([0.1, 0.8, 0.2, 0.7, 0.3, 0.6])
    ml = _oof([0.2, 0.7, 0.1, 0.8, 0.4, 0.5]).iloc[::-1]

    with pytest.raises(EvaluationError, match="row alignment"):
        evaluate_aligned_blends(dl, ml, weights=(0.1, 0.5, 0.9))


def test_blend_grid_contains_probability_and_logit_families() -> None:
    dl = _oof([0.1, 0.8, 0.2, 0.7, 0.3, 0.6])
    ml = _oof([0.2, 0.7, 0.1, 0.8, 0.4, 0.5])

    grid = evaluate_aligned_blends(dl, ml, weights=(0.25, 0.75))

    assert len(grid) == 4
    assert set(grid["blend_family"]) == {"probability", "logit"}
    assert set(grid["dl_weight"]) == {0.25, 0.75}
    assert np.isfinite(grid["global_brier"]).all()


def test_survivors_include_absolute_blend_segment_diversity_and_family_top() -> None:
    table = pd.DataFrame(
        [
            {"candidate_id": "absolute", "family": "tabm", "brier": 0.249, "blend_gain": 0.0, "segment_gain": 0.0, "error_correlation": 0.99},
            {"candidate_id": "blend", "family": "mlp_resnet", "brier": 0.260, "blend_gain": 0.001, "segment_gain": 0.0, "error_correlation": 0.99},
            {"candidate_id": "segment", "family": "ft_transformer", "brier": 0.261, "blend_gain": 0.0, "segment_gain": 0.002, "error_correlation": 0.99},
            {"candidate_id": "diverse", "family": "tabr", "brier": 0.262, "blend_gain": 0.0, "segment_gain": 0.0, "error_correlation": 0.90},
            {"candidate_id": "family-top", "family": "tabm", "brier": 0.251, "blend_gain": 0.0, "segment_gain": 0.0, "error_correlation": 0.99},
        ]
    )

    survivors = select_survivors(
        table,
        ml_brier=0.248,
        proximity_delta=0.002,
        max_error_correlation=0.98,
        top_k_per_family=2,
    )

    reasons = set().union(*(row["reasons"] for row in survivors))
    assert {"absolute", "blend", "segment", "diversity", "family_top"} <= reasons
    assert [row["candidate_id"] for row in survivors] == table["candidate_id"].tolist()


def test_survival_rejects_missing_metric_even_with_unrelated_columns() -> None:
    incomplete = pd.DataFrame(
        [
            {
                "candidate_id": "candidate",
                "family": "tabm",
                "brier": 0.25,
                "segment_gain": 0.0,
                "error_correlation": 0.9,
                "unrelated": 1.0,
            }
        ]
    )

    with pytest.raises(EvaluationError, match="missing survival"):
        select_survivors(
            incomplete,
            ml_brier=0.248,
            proximity_delta=0.002,
            max_error_correlation=0.98,
            top_k_per_family=2,
        )


def test_evaluation_artifacts_use_fixed_names(tmp_path) -> None:
    frame = pd.DataFrame([{"candidate_id": "one", "brier": 0.25}])
    write_evaluation_artifacts(
        tmp_path,
        standalone=frame,
        diversity=frame,
        blend_contribution=frame,
        survivors=[{"candidate_id": "one", "family": "tabm", "reasons": ("family_top",)}],
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "blend_contribution.csv",
        "diversity.csv",
        "standalone.csv",
        "survivors.json",
    ]
