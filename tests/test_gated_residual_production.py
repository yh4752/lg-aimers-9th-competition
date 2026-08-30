from __future__ import annotations

from pathlib import Path

from experiments.gated_residual_final.production import build_production_candidate
from experiments.gated_residual_final.selection import CandidateDecision


def test_rejected_decision_never_calls_full_fit(tmp_path: Path) -> None:
    calls = []

    result = build_production_candidate(
        decision=CandidateDecision("G1_x", "rejected", ("latest_gain",)),
        fit_model=lambda **kwargs: calls.append(kwargs),
        iterations={"D0": 100, "D5": 120},
        output_dir=tmp_path,
    )

    assert result is None
    assert calls == []


def test_accepted_candidate_trains_exactly_six_direct_models(tmp_path: Path) -> None:
    calls = []

    def fit_model(*, role: str, seed: int, iterations: int, output: Path) -> Path:
        calls.append((role, seed, iterations))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"{role}-{seed}".encode())
        return output

    result = build_production_candidate(
        decision=CandidateDecision("G1_x", "accepted", ()),
        fit_model=fit_model,
        iterations={"D0": 100, "D5": 120},
        output_dir=tmp_path,
    )

    assert calls == [
        (role, seed, {"D0": 100, "D5": 120}[role])
        for role in ("D0", "D5") for seed in (42, 2026, 3407)
    ]
    assert result is not None
    assert len(result.models) == 6


def test_invalid_iteration_budget_is_rejected(tmp_path: Path) -> None:
    from experiments.gated_residual_final.production import ProductionError
    import pytest

    with pytest.raises(ProductionError, match="iteration evidence differs"):
        build_production_candidate(
            decision=CandidateDecision("G1_x", "accepted", ()),
            fit_model=lambda **kwargs: kwargs["output"],
            iterations={"D0": 2401, "D5": 120},
            output_dir=tmp_path,
        )
