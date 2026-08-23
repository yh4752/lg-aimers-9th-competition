"""Pure, preregistered promotion decisions for temporal campaign stages."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral

from .metrics import score_tier


class DecisionError(ValueError):
    """Raised when campaign evidence is incomplete or malformed."""


_STATUSES = frozenset(
    {
        "champion",
        "exploratory",
        "rejected",
        "insufficient_mapping",
        "budget_inconclusive",
        "stable_for_deployment",
        "unstable_for_deployment",
    }
)
_STRUCTURAL = frozenset({"B1", "M1"})
_CATBOOST_PREFIXES = (16, 64, 192, 384)


@dataclass(frozen=True)
class CandidateMetrics:
    candidate_id: str
    family: str
    temporal_gain: Decimal
    weighted_gain: Decimal
    latest_gain: Decimal
    bootstrap_lower: Decimal
    max_segment_regression: Decimal
    fold_regressions: tuple[Decimal, ...]
    improved_fold_count: int
    worst_fold_regression: Decimal
    latest_regression: Decimal
    evidence_complete: bool = True
    mapping_evidence: str = "accepted"

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise DecisionError("candidate_id must be a nonempty string")
        if type(self.family) is not str or not self.family:
            raise DecisionError("family must be a nonempty string")
        values = (
            self.temporal_gain,
            self.weighted_gain,
            self.latest_gain,
            self.bootstrap_lower,
            self.max_segment_regression,
            self.worst_fold_regression,
            self.latest_regression,
            *self.fold_regressions,
        )
        if not self.fold_regressions or any(type(value) is not Decimal or not value.is_finite() for value in values):
            raise DecisionError("candidate metrics must be finite Decimals")
        if type(self.improved_fold_count) is not int or not 0 <= self.improved_fold_count <= len(self.fold_regressions):
            raise DecisionError("improved_fold_count is invalid")
        if type(self.evidence_complete) is not bool:
            raise DecisionError("evidence_complete must be an exact bool")
        if self.mapping_evidence not in ("accepted", "insufficient_mapping"):
            raise DecisionError("mapping_evidence is invalid")


@dataclass(frozen=True)
class CandidateDecision:
    status: str
    tier: str
    candidate_id: str

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise DecisionError("decision status is invalid")


@dataclass(frozen=True)
class CatBoostFoldMetrics:
    candidate_id: str
    best_prefixes: tuple[int, ...]
    evidence_complete: bool = True

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise DecisionError("candidate_id must be a nonempty string")
        if type(self.best_prefixes) is not tuple or any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in _CATBOOST_PREFIXES
            for value in self.best_prefixes
        ):
            raise DecisionError("CatBoost prefixes differ from preregistration")
        if type(self.evidence_complete) is not bool:
            raise DecisionError("evidence_complete must be an exact bool")


@dataclass(frozen=True)
class CatBoostPrefixDecision:
    status: str
    candidate_id: str
    selected_prefix: int | None

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise DecisionError("prefix decision status is invalid")


def decide_candidate(metrics: CandidateMetrics) -> CandidateDecision:
    if type(metrics) is not CandidateMetrics:
        raise DecisionError("metrics must be CandidateMetrics")
    tier = score_tier(metrics.temporal_gain)
    if not metrics.evidence_complete:
        return CandidateDecision("budget_inconclusive", tier, metrics.candidate_id)
    if metrics.mapping_evidence == "insufficient_mapping":
        return CandidateDecision("insufficient_mapping", tier, metrics.candidate_id)
    champion = (
        metrics.weighted_gain >= Decimal("0.00005")
        and metrics.latest_gain >= Decimal("0.00003")
        and metrics.bootstrap_lower > 0
        and metrics.max_segment_regression <= Decimal("0.00050")
        and all(value <= Decimal("0.00003") for value in metrics.fold_regressions)
    )
    if champion:
        return CandidateDecision("champion", tier, metrics.candidate_id)
    exploratory = (
        metrics.improved_fold_count >= 2
        and metrics.weighted_gain >= Decimal("0.00003")
        and metrics.worst_fold_regression <= Decimal("0.00015")
        and metrics.latest_regression <= Decimal("0.00005")
        and metrics.max_segment_regression <= Decimal("0.00100")
    )
    return CandidateDecision(
        "exploratory" if exploratory else "rejected", tier, metrics.candidate_id
    )


def select_t1_survivors(candidates: tuple[CandidateMetrics, ...]) -> tuple[CandidateMetrics, ...]:
    return _rank_eligible(candidates, limit=1)


def select_t2a_survivors(candidates: tuple[CandidateMetrics, ...]) -> tuple[CandidateMetrics, ...]:
    eligible = list(_rank_eligible(candidates, limit=None))
    best = eligible[:2]
    structural = next((item for item in eligible if item.family in _STRUCTURAL and item not in best), None)
    if structural is not None:
        best.append(structural)
    return tuple(best[:3])


def select_t2b_survivors(candidates: tuple[CandidateMetrics, ...]) -> tuple[CandidateMetrics, ...]:
    return _rank_eligible(candidates, limit=3)


def select_t3_survivors(candidates: tuple[CandidateMetrics, ...]) -> tuple[CandidateMetrics, ...]:
    return _rank_eligible(candidates, limit=2)


def decide_catboost_prefix(metrics: CatBoostFoldMetrics) -> CatBoostPrefixDecision:
    if type(metrics) is not CatBoostFoldMetrics:
        raise DecisionError("metrics must be CatBoostFoldMetrics")
    if not metrics.evidence_complete or len(metrics.best_prefixes) < 2:
        return CatBoostPrefixDecision("budget_inconclusive", metrics.candidate_id, None)
    positions = [_CATBOOST_PREFIXES.index(int(value)) for value in metrics.best_prefixes]
    if max(positions) - min(positions) > 1:
        return CatBoostPrefixDecision("unstable_for_deployment", metrics.candidate_id, None)
    counts = {prefix: metrics.best_prefixes.count(prefix) for prefix in set(metrics.best_prefixes)}
    selected = min(counts, key=lambda prefix: (-counts[prefix], prefix))
    return CatBoostPrefixDecision("stable_for_deployment", metrics.candidate_id, int(selected))


def _rank_eligible(
    candidates: tuple[CandidateMetrics, ...], *, limit: int | None
) -> tuple[CandidateMetrics, ...]:
    if type(candidates) is not tuple or not candidates or any(type(item) is not CandidateMetrics for item in candidates):
        raise DecisionError("candidates must be a nonempty tuple of CandidateMetrics")
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise DecisionError("candidate IDs must be unique")
    eligible = [
        item
        for item in candidates
        if decide_candidate(item).status in ("champion", "exploratory")
    ]
    eligible.sort(
        key=lambda item: (
            0 if decide_candidate(item).status == "champion" else 1,
            -item.weighted_gain,
            -item.latest_gain,
            item.candidate_id,
        )
    )
    return tuple(eligible if limit is None else eligible[:limit])
