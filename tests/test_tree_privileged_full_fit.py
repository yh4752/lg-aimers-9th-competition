from __future__ import annotations

from types import MappingProxyType

import pytest

from experiments.tree_privileged.contracts import load_contract
from experiments.tree_privileged.decisions import CandidateDecision
from experiments.tree_privileged.full_fit import PrivilegedFullFitError, accepted_full_fit_token
from experiments.tree_privileged.profiles import ProfileStrengths


def _decision(status: str) -> CandidateDecision:
    return CandidateDecision("PD15", status, () if status == "accepted" else ("weighted_gain",), MappingProxyType({}))


def _bindings() -> dict[str, str]:
    contract = load_contract()
    return {
        "official_train_sha256": contract.inputs["official_train_sha256"],
        "official_history_sha256": contract.inputs["official_history_sha256"],
        "e2_handoff_sha256": contract.inputs["e2_handoff_sha256"],
    }


def test_full_fit_refuses_rejected_or_unbound_decision() -> None:
    evidence = {seed: (99, 100, 101) for seed in (42, 2026, 3407)}
    with pytest.raises(PrivilegedFullFitError, match="accepted decision"):
        accepted_full_fit_token(_decision("rejected"), bindings=_bindings(), best_iterations=evidence,
                                strengths=ProfileStrengths(25, 50, 100))
    changed = _bindings(); changed["e2_handoff_sha256"] = "0" * 64
    with pytest.raises(PrivilegedFullFitError, match="bindings differ"):
        accepted_full_fit_token(_decision("accepted"), bindings=changed, best_iterations=evidence,
                                strengths=ProfileStrengths(25, 50, 100))


def test_full_fit_iterations_are_bound_to_three_seed_medians() -> None:
    token = accepted_full_fit_token(
        _decision("accepted"), bindings=_bindings(),
        best_iterations={42: (70, 80, 90), 2026: (40, 60, 80), 3407: (1199, 1200, 1200)},
        strengths=ProfileStrengths(25, 50, 100),
    )
    assert dict(token.iterations) == {42: 81, 2026: 61, 3407: 1200}

