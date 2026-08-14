from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .metrics import SegmentMetric


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    capacity: str
    num_embedding: str
    loss: str
    scheduler: str
    brier: float
    parameter_count: int
    inference_seconds: float


@dataclass(frozen=True)
class Survivor:
    score: CandidateScore
    reason: str


@dataclass(frozen=True)
class TemporalEvidence:
    candidate_id: str
    delta_2024: float
    delta_2023: float


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reason: str
    candidate_id: str


@dataclass(frozen=True)
class ChampionVerdict:
    candidate_id: str
    source: str
    reason: str


def _score_order(score: CandidateScore) -> tuple[float, int, float, str]:
    if not math.isfinite(score.brier) or score.parameter_count < 0 or score.inference_seconds < 0:
        raise ValueError(f"invalid candidate evidence: {score.candidate_id}")
    return score.brier, score.parameter_count, score.inference_seconds, score.candidate_id


def choose_version_a_survivors(scores: list[CandidateScore]) -> tuple[Survivor, ...]:
    if len(scores) < 4 or len({score.candidate_id for score in scores}) != len(scores):
        raise ValueError("Version A requires at least four unique completed candidates")
    ordered = sorted(scores, key=_score_order)
    best = ordered[0]
    selected: list[Survivor] = [Survivor(best, "best_overall")]

    def add_first(predicate, reason: str) -> None:
        chosen_ids = {item.score.candidate_id for item in selected}
        for score in ordered:
            if score.candidate_id not in chosen_ids and predicate(score):
                selected.append(Survivor(score, reason))
                return

    add_first(lambda score: score.capacity == "p2", "best_p2")
    add_first(lambda score: score.capacity in {"p3_lite", "p3_full"}, "best_p3")
    add_first(
        lambda score: any(
            (score.num_embedding != best.num_embedding, score.loss != best.loss, score.scheduler != best.scheduler)
        ),
        "axis_diversity",
    )
    for score in ordered:
        if len(selected) == 4:
            break
        if score.candidate_id not in {item.score.candidate_id for item in selected}:
            selected.append(Survivor(score, "metric_fallback"))
    if len(selected) != 4:
        raise ValueError("four distinct survivors could not be selected")
    return tuple(selected)


def temporal_verdict(evidence: TemporalEvidence) -> Verdict:
    if not math.isfinite(evidence.delta_2024) or not math.isfinite(evidence.delta_2023):
        return Verdict(False, "non_finite_temporal_evidence", evidence.candidate_id)
    if evidence.delta_2024 > 0.00005:
        return Verdict(False, "primary_fold_regression", evidence.candidate_id)
    if evidence.delta_2023 > 0.00010:
        return Verdict(False, "older_fold_regression", evidence.candidate_id)
    if 0.70 * evidence.delta_2024 + 0.30 * evidence.delta_2023 >= 0:
        return Verdict(False, "weighted_delta_not_negative", evidence.candidate_id)
    return Verdict(True, "temporal_gates_passed", evidence.candidate_id)


def choose_temporal_champion(
    fold_briers: Mapping[str, tuple[float, float]],
    *,
    reference_id: str,
) -> tuple[str, float]:
    """Select against one declared reference using the temporal promotion gates."""

    if reference_id not in fold_briers:
        raise ValueError(f"temporal reference is missing: {reference_id}")
    reference_2024, reference_2023 = fold_briers[reference_id]
    if not math.isfinite(reference_2024) or not math.isfinite(reference_2023):
        raise ValueError(f"temporal reference is non-finite: {reference_id}")
    champion = reference_id
    champion_delta = 0.0
    for candidate_id in sorted(fold_briers):
        primary, older = fold_briers[candidate_id]
        evidence = TemporalEvidence(
            candidate_id,
            float(primary) - float(reference_2024),
            float(older) - float(reference_2023),
        )
        verdict = temporal_verdict(evidence)
        weighted = 0.70 * evidence.delta_2024 + 0.30 * evidence.delta_2023
        if verdict.accepted and weighted < champion_delta:
            champion = candidate_id
            champion_delta = weighted
    return champion, champion_delta


def choose_refined_champion(
    evidence: Mapping[tuple[str, int], float],
    *,
    refined_id: str,
    version_b_id: str,
) -> ChampionVerdict:
    required = ((refined_id, 2024), (refined_id, 2023), (version_b_id, 2024), (version_b_id, 2023))
    if not all(key in evidence for key in required):
        return ChampionVerdict(version_b_id, "version_b", "refinement_confirmation_incomplete")
    refined = TemporalEvidence(
        refined_id,
        float(evidence[(refined_id, 2024)] - evidence[(version_b_id, 2024)]),
        float(evidence[(refined_id, 2023)] - evidence[(version_b_id, 2023)]),
    )
    verdict = temporal_verdict(refined)
    return (
        ChampionVerdict(refined_id, "version_c", "refinement_temporal_gates_passed")
        if verdict.accepted
        else ChampionVerdict(version_b_id, "version_b", verdict.reason)
    )


def segment_gate(segments: tuple[SegmentMetric, ...]) -> bool:
    return not any(
        row.hard_gate_eligible and row.reference_delta is not None and row.reference_delta > 0.00050
        for row in segments
    )


def ensemble_verdict(
    *,
    candidate_id: str,
    primary_gain: float,
    older_delta: float | None,
    weighted_delta: float | None,
    segment_passed: bool,
) -> Verdict:
    if primary_gain < 0.00003:
        return Verdict(False, "primary_gain_below_0_00003", candidate_id)
    if older_delta is None or weighted_delta is None:
        return Verdict(False, "older_fold_confirmation_incomplete", candidate_id)
    if older_delta > 0.00005:
        return Verdict(False, "older_fold_regression", candidate_id)
    if weighted_delta >= 0:
        return Verdict(False, "weighted_delta_not_negative", candidate_id)
    if not segment_passed:
        return Verdict(False, "segment_gate_failed", candidate_id)
    return Verdict(True, "ensemble_gates_passed", candidate_id)
