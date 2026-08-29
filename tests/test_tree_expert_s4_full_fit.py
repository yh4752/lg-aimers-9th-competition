from dataclasses import replace

import pytest

from experiments.tree_expert.s4_artifacts import S4Bindings
from experiments.tree_expert.s4_decisions import S4Decision
from experiments.tree_expert.s4_full_fit import (
    S4FullFitError,
    issue_full_fit_token,
    median_plus_one_iterations,
)


def _decision(status="accepted"):
    return S4Decision("candidate", status, (), 0.001, 0.001, 0.0, 0.0, 0.0001, 3)


def _bindings():
    return S4Bindings(*("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, "f" * 64))


def test_rejected_decision_cannot_issue_full_fit_token():
    with pytest.raises(S4FullFitError, match="accepted decision required"):
        issue_full_fit_token(_decision("rejected"), _bindings(), fold_iterations={(2021, 2022): 10, (2022, 2023): 12, (2023, 2024): 11})


def test_token_uses_median_plus_one_and_seals_bindings():
    token = issue_full_fit_token(
        _decision(), _bindings(),
        fold_iterations={(2021, 2022): 10, (2022, 2023): 12, (2023, 2024): 11},
    )
    assert token.full_fit_iterations == 12
    assert token.bindings == _bindings()


def test_iteration_registry_requires_all_folds_and_clips():
    assert median_plus_one_iterations(
        {(2021, 2022): 3, (2022, 2023): 9, (2023, 2024): 5}, maximum=5
    ) == 5
    with pytest.raises(S4FullFitError):
        median_plus_one_iterations({(2021, 2022): 3}, maximum=5)

