from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.direct_expert.stacking import (
    DirectExpertStackingError,
    StackRecipe,
    apply_stack,
    fit_e2_safety_blend,
    fit_stack,
    required_experts,
)


class ExplodingMapping(dict):
    def __iter__(self):
        raise AssertionError("confirmation stream was opened")


def structure_streams() -> dict[str, pd.DataFrame]:
    target = np.asarray([0, 1] * 8)
    output = {}
    for index, expert_id in enumerate(("D0", "D2", "D5", "D7")):
        probability = np.where(target == 1, 0.55 + 0.02 * index, 0.45 - 0.02 * index)
        output[expert_id] = pd.DataFrame(
            {
                "row_id": [f"r{i}" for i in range(len(target))],
                "target": target,
                "probability": probability,
                "oof_year": [2022] * 8 + [2023] * 8,
            }
        )
    return output


def test_probability_stack_is_nonnegative_sparse_and_normalized() -> None:
    recipe = fit_stack(structure_streams(), method="probability")
    assert len(recipe.weights) <= 3
    assert all(weight >= 0.05 for weight in recipe.weights.values())
    assert sum(recipe.weights.values()) == pytest.approx(1.0)


def test_logit_stack_clips_before_transform() -> None:
    recipe = StackRecipe("logit", {"D0": 0.5, "D2": 0.5}, (2022, 2023))
    result = apply_stack(recipe, {"D0": np.array([0.0]), "D2": np.array([1.0])})
    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(0.5)


def test_2024_stream_is_never_opened_while_fitting() -> None:
    recipe = fit_stack(
        structure_streams(),
        method="probability",
        confirmation=ExplodingMapping(),
    )
    assert recipe.selection_years == (2022, 2023)


def test_specialist_dependency_counts_toward_budget() -> None:
    assert required_experts(("D5",)) == ("D0", "D5")
    with pytest.raises(DirectExpertStackingError, match="deployment dependency budget"):
        StackRecipe("probability", {"D5": 0.34, "D6": 0.33, "D7": 0.33}, (2022, 2023))


def test_e2_safety_blend_uses_at_most_two_direct_dependencies() -> None:
    direct = fit_stack({key: value for key, value in structure_streams().items() if key in {"D0", "D2"}}, method="probability")
    baseline = np.asarray([0.45, 0.55] * 8)
    recipe = fit_e2_safety_blend(direct, structure_streams(), baseline)
    assert recipe.e2_weight in {0.1, 0.2, 0.3, 0.4}
    assert len(required_experts(tuple(recipe.weights))) <= 2
