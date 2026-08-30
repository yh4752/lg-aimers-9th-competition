import pytest

from experiments.direct_expert.full_fit import (
    DirectExpertFullFitError,
    accepted_token,
    full_fit_iterations,
)
from experiments.direct_expert.stacking import StackRecipe


def test_only_accepted_decision_can_issue_full_fit_token() -> None:
    with pytest.raises(DirectExpertFullFitError, match="candidate is not accepted"):
        accepted_token(
            {"candidate_id": "x", "status": "rejected"},
            StackRecipe("probability", {"D0": 1.0}, (2022, 2023)),
            {"contract_sha256": "1" * 64},
            seeds=(42, 2026, 3407),
            iterations={"D0": 1500},
        )


def test_full_fit_iterations_use_median_and_clip() -> None:
    assert full_fit_iterations([1200, 1500, 2100], maximum=2400) == 1500
    assert full_fit_iterations([2300, 2400, 2400], maximum=2400) == 2400
