from __future__ import annotations

import itertools

import pandas as pd
import pytest

from experiments.independent_dl.preprocessing_evaluation import (
    evaluate_paired_setting,
    generate_pairwise_settings,
    next_beam,
)


def _metric_rows(delta: float = -0.0002) -> pd.DataFrame:
    rows = []
    for seed, fold in itertools.product((42, 2026, 3407), range(2020, 2025)):
        for setting_id, brier in (
            ("baseline", 0.2500),
            ("candidate", 0.2500 + delta),
        ):
            rows.append(
                {
                    "anchor_id": "tabm-p3",
                    "preprocessing_id": setting_id,
                    "seed": seed,
                    "fold": f"valid_{fold}",
                    "valid_rows": 100,
                    "brier": brier,
                    "pitcher_oov_brier": brier,
                    "batter_oov_brier": brier,
                }
            )
    return pd.DataFrame(rows)


def test_paired_evaluation_requires_same_anchor_fold_and_seed() -> None:
    changed = _metric_rows().drop(index=0)

    with pytest.raises(ValueError, match="paired baseline"):
        evaluate_paired_setting(
            changed, candidate_id="candidate", baseline_id="baseline"
        )


def test_global_adoption_requires_all_fixed_gates() -> None:
    result = evaluate_paired_setting(
        _metric_rows(), candidate_id="candidate", baseline_id="baseline"
    )

    assert result["gates"] == {
        "weighted_mean_improved": True,
        "four_of_five_folds_improved": True,
        "two_of_three_seeds_improved": True,
        "worst_fold_delta_lte_0_0001": True,
        "pitcher_oov_delta_lte_0_0002": True,
        "batter_oov_delta_lte_0_0002": True,
    }
    assert result["adopt_global"] is True


def test_pairwise_generation_keeps_all_nine_components() -> None:
    settings = generate_pairwise_settings(
        pitcher_component="pitcher_smooth_k100",
        batter_component="batter_smooth_k250",
    )

    assert len(settings) == 36
    assert len({setting.components for setting in settings}) == 36


def test_beam_keeps_eight_and_adds_one_component() -> None:
    components = (
        "pitcher_smooth_k100",
        "batter_smooth_k250",
        "dl_selective_transform",
        "asof_count_log1p",
        "entity_frequency_log1p",
        "grouped_missing_indicators",
        "hand_matchup",
        "count_state",
        "pitcher_team_win_expectancy",
    )
    ranked = pd.DataFrame(
        [
            {
                "preprocessing_id": f"pair-{index}",
                "components": pair,
                "weighted_delta": -0.001 + index * 0.00001,
                "improved_folds": 5,
                "worst_fold_delta": -0.0001,
                "max_oov_delta": 0.0,
            }
            for index, pair in enumerate(
                itertools.islice(itertools.combinations(components, 2), 10)
            )
        ]
    )

    expanded = next_beam(ranked, beam_width=8, max_components=5)

    assert len({item.parent_id for item in expanded}) <= 8
    assert {len(item.components) for item in expanded} == {3}


def test_bad_worst_fold_blocks_global_adoption() -> None:
    rows = _metric_rows()
    mask = (
        rows["preprocessing_id"].eq("candidate")
        & rows["fold"].eq("valid_2024")
    )
    rows.loc[mask, ["brier", "pitcher_oov_brier", "batter_oov_brier"]] = 0.2502

    result = evaluate_paired_setting(
        rows, candidate_id="candidate", baseline_id="baseline"
    )

    assert result["gates"]["worst_fold_delta_lte_0_0001"] is False
    assert result["adopt_global"] is False
