from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .hc_contracts import HCContract


@dataclass(frozen=True)
class C1Evidence:
    weighted_brier: float
    weighted_gain: float
    confirmation_gain: float
    minimum_fold_gain: float
    maximum_segment_regression: float
    bootstrap_lower_95: float
    non_worse_seed_counts: Mapping[int, int]


@dataclass(frozen=True)
class C2Evidence:
    weighted_brier: float
    weighted_gain: float
    incremental_gain: float
    minimum_fold_gain: float
    c0_calibration_gap: float
    candidate_calibration_gap: float
    c0_ece: float
    candidate_ece: float
    maximum_segment_regression: float
    bootstrap_lower_95: float


@dataclass(frozen=True)
class CandidateDecision:
    candidate: str
    status: str
    weighted_brier: float
    failed_gates: tuple[str, ...]
    evidence: object


@dataclass(frozen=True)
class WinnerDecision:
    status: str
    candidate: str
    reason: str


def decide_c1(evidence: C1Evidence, contract: HCContract) -> CandidateDecision:
    gates = contract.gates
    failed: list[str] = []
    if evidence.weighted_gain < float(gates["c1_weighted_gain"]):
        failed.append("weighted_gain")
    if evidence.confirmation_gain <= float(gates["c1_confirmation_gain"]):
        failed.append("confirmation_gain")
    if evidence.minimum_fold_gain < float(gates["c1_min_fold_gain"]):
        failed.append("minimum_fold_gain")
    if evidence.maximum_segment_regression > float(gates["c1_max_segment_regression"]):
        failed.append("segment_regression")
    if evidence.bootstrap_lower_95 <= 0:
        failed.append("bootstrap_lower")
    required = int(gates["required_non_worse_seeds"])
    if set(evidence.non_worse_seed_counts) != {2022, 2023, 2024} or any(
        count < required for count in evidence.non_worse_seed_counts.values()
    ):
        failed.append("seed_consistency")
    return CandidateDecision(
        candidate="C1",
        status="accepted" if not failed else "rejected",
        weighted_brier=float(evidence.weighted_brier),
        failed_gates=tuple(failed),
        evidence=evidence,
    )


def decide_c2(evidence: C2Evidence, contract: HCContract) -> CandidateDecision:
    gates = contract.gates
    failed: list[str] = []
    if evidence.weighted_gain < float(gates["c2_weighted_gain"]):
        failed.append("weighted_gain")
    if evidence.incremental_gain < float(gates["c2_incremental_gain"]):
        failed.append("incremental_gain")
    if evidence.minimum_fold_gain < float(gates["c2_min_fold_gain"]):
        failed.append("minimum_fold_gain")
    if evidence.candidate_calibration_gap > evidence.c0_calibration_gap:
        failed.append("calibration_gap")
    if evidence.candidate_ece > evidence.c0_ece:
        failed.append("ece")
    if evidence.maximum_segment_regression > float(gates["c2_max_segment_regression"]):
        failed.append("segment_regression")
    if evidence.bootstrap_lower_95 <= 0:
        failed.append("bootstrap_lower")
    return CandidateDecision(
        candidate="C2",
        status="accepted" if not failed else "rejected",
        weighted_brier=float(evidence.weighted_brier),
        failed_gates=tuple(failed),
        evidence=evidence,
    )


def choose_winner(
    c1: CandidateDecision, c2: CandidateDecision, contract: HCContract
) -> WinnerDecision:
    accepted = {item.candidate: item for item in (c1, c2) if item.status == "accepted"}
    if not accepted:
        return WinnerDecision("fallback", "C0", "no_new_candidate_passed")
    if set(accepted) == {"C1"}:
        return WinnerDecision("accepted", "C1", "only_accepted_candidate")
    if set(accepted) == {"C2"}:
        return WinnerDecision("accepted", "C2", "only_accepted_candidate")
    difference = abs(c1.weighted_brier - c2.weighted_brier)
    if difference <= float(contract.gates["simpler_tie_margin"]):
        return WinnerDecision("accepted", "C1", "simpler_within_tie_margin")
    candidate = "C1" if c1.weighted_brier < c2.weighted_brier else "C2"
    return WinnerDecision("accepted", candidate, "lower_weighted_brier")
