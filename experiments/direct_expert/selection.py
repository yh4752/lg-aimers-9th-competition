from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import load_contract
from .inputs import canonical_json


class DirectExpertSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    fold_gains: Mapping[int, float]
    weighted_gain: float
    latest_gain: float
    recent_heavy_gain: float
    minimum_fold_gain: float
    maximum_segment_regression: float
    bootstrap_lower: float
    latest_bootstrap_lower: float
    non_worse_seed_count: int
    improving_latest_seed_count: int


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    status: str
    gate: str | None
    failed_gates: tuple[str, ...]


@dataclass(frozen=True)
class LockedSelection:
    expert_ids: tuple[str, ...]
    roles: Mapping[str, str]
    locked_on_years: tuple[int, ...]
    selection_sha256: str


_REQUIRED = {
    "row_id",
    "target",
    "probability",
    "p_anchor",
    "game_type",
    "pitcher_id",
    "oof_year",
}


def _validated(frame: pd.DataFrame) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or set(frame) != _REQUIRED or frame.empty:
        raise DirectExpertSelectionError("prediction columns differ")
    output = frame.copy(deep=True).sort_values("row_id", kind="stable").reset_index(drop=True)
    if output["row_id"].isna().any() or not output["row_id"].is_unique:
        raise DirectExpertSelectionError("prediction row identity differs")
    target = pd.to_numeric(output["target"], errors="coerce")
    candidate = pd.to_numeric(output["probability"], errors="coerce")
    anchor = pd.to_numeric(output["p_anchor"], errors="coerce")
    year = pd.to_numeric(output["oof_year"], errors="coerce")
    if (
        not target.isin((0, 1)).all()
        or candidate.isna().any()
        or anchor.isna().any()
        or not candidate.between(0, 1).all()
        or not anchor.between(0, 1).all()
        or year.isna().any()
    ):
        raise DirectExpertSelectionError("prediction values differ")
    output["target"] = target.astype("int8")
    output["probability"] = candidate.astype("float64")
    output["p_anchor"] = anchor.astype("float64")
    output["oof_year"] = year.astype("int16")
    return output


def _weighted(values: Mapping[int, float], weights: Mapping[int, float]) -> float:
    available = [(weights[year], value) for year, value in values.items() if year in weights]
    if not available:
        raise DirectExpertSelectionError("fold weights have no overlap")
    denominator = sum(weight for weight, _ in available)
    return float(sum(weight * value for weight, value in available) / denominator)


def _bootstrap_lower(frame: pd.DataFrame, *, latest_only: bool) -> float:
    source = frame.loc[frame["oof_year"].eq(frame["oof_year"].max())] if latest_only else frame
    groups = [group.index.to_numpy() for _, group in source.groupby("pitcher_id", sort=True)]
    if not groups:
        raise DirectExpertSelectionError("pitcher clusters are absent")
    gain = (
        np.square(source["target"].to_numpy() - source["p_anchor"].to_numpy())
        - np.square(source["target"].to_numpy() - source["probability"].to_numpy())
    )
    rng = np.random.default_rng(3407)
    sampled = np.empty(1000, dtype="float64")
    for index in range(len(sampled)):
        choices = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[item] for item in choices])
        positions = source.index.get_indexer(rows)
        sampled[index] = float(gain[positions].mean())
    return float(np.quantile(sampled, 0.025))


def evidence_from_predictions(
    candidate_id: str,
    predictions: pd.DataFrame,
    *,
    non_worse_seed_count: int = 0,
    improving_latest_seed_count: int = 0,
    minimum_segment_rows: int = 5000,
) -> CandidateEvidence:
    frame = _validated(predictions)
    frame["gain"] = (
        np.square(frame["target"] - frame["p_anchor"])
        - np.square(frame["target"] - frame["probability"])
    )
    fold_gains = {
        int(year): float(group["gain"].mean())
        for year, group in frame.groupby("oof_year", sort=True)
    }
    latest_year = max(fold_gains)
    segment_regressions = []
    for _, group in frame.groupby("game_type", sort=True):
        if len(group) >= minimum_segment_rows:
            segment_regressions.append(max(0.0, -float(group["gain"].mean())))
    stable_weights = {2022: 0.60, 2023: 0.75, 2024: 0.90}
    recent_weights = {2022: 0.15, 2023: 0.25, 2024: 0.60}
    return CandidateEvidence(
        candidate_id=candidate_id,
        fold_gains=MappingProxyType(fold_gains),
        weighted_gain=_weighted(fold_gains, stable_weights),
        latest_gain=fold_gains[latest_year],
        recent_heavy_gain=_weighted(fold_gains, recent_weights),
        minimum_fold_gain=min(fold_gains.values()),
        maximum_segment_regression=max(segment_regressions, default=0.0),
        bootstrap_lower=_bootstrap_lower(frame, latest_only=False),
        latest_bootstrap_lower=_bootstrap_lower(frame, latest_only=True),
        non_worse_seed_count=int(non_worse_seed_count),
        improving_latest_seed_count=int(improving_latest_seed_count),
    )


def _failures(evidence: CandidateEvidence, gate: str) -> tuple[str, ...]:
    contract = load_contract()
    if gate == "stable":
        thresholds = contract.stable_gate
        checks = (
            ("weighted_gain", evidence.weighted_gain >= thresholds["weighted_gain"]),
            ("latest_gain", evidence.latest_gain >= thresholds["latest_gain"]),
            ("minimum_fold_gain", evidence.minimum_fold_gain >= thresholds["minimum_fold_gain"]),
            ("maximum_segment_regression", evidence.maximum_segment_regression <= thresholds["maximum_segment_regression"]),
            ("bootstrap_lower", evidence.bootstrap_lower >= thresholds["bootstrap_lower"]),
            ("non_worse_seed_count", evidence.non_worse_seed_count >= thresholds["minimum_non_worse_seeds"]),
        )
    else:
        thresholds = contract.aggressive_gate
        checks = (
            ("latest_gain", evidence.latest_gain >= thresholds["latest_gain"]),
            ("recent_heavy_gain", evidence.recent_heavy_gain >= thresholds["recent_heavy_gain"]),
            ("minimum_fold_gain", evidence.minimum_fold_gain >= thresholds["minimum_fold_gain"]),
            ("maximum_segment_regression", evidence.maximum_segment_regression <= thresholds["maximum_segment_regression"]),
            ("latest_bootstrap_lower", evidence.latest_bootstrap_lower >= thresholds["latest_bootstrap_lower"]),
            ("improving_latest_seed_count", evidence.improving_latest_seed_count >= thresholds["minimum_improving_latest_seeds"]),
        )
    return tuple(name for name, passed in checks if not passed)


def decide(evidence: CandidateEvidence) -> CandidateDecision:
    stable = _failures(evidence, "stable")
    aggressive = _failures(evidence, "aggressive")
    if not stable:
        return CandidateDecision(evidence.candidate_id, "accepted_stable", "stable", ())
    if not aggressive:
        return CandidateDecision(evidence.candidate_id, "accepted_aggressive", "aggressive", ())
    failed = stable if len(stable) <= len(aggressive) else aggressive
    return CandidateDecision(evidence.candidate_id, "rejected", None, failed)


def _correlation(left: pd.DataFrame, right: pd.DataFrame) -> float:
    a = _validated(left).set_index("row_id")
    b = _validated(right).set_index("row_id")
    if set(a.index) != set(b.index):
        raise DirectExpertSelectionError("candidate row sets differ")
    b = b.loc[a.index]
    x = a["target"].to_numpy() - a["probability"].to_numpy()
    y = b["target"].to_numpy() - b["probability"].to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return 1.0
    return abs(float(np.corrcoef(x, y)[0, 1]))


def select_structure_experts(
    structure_frames: Mapping[str, pd.DataFrame],
    *,
    confirmation_frames: Mapping[str, pd.DataFrame] | None = None,
) -> LockedSelection:
    del confirmation_frames
    expected = {f"D{i}" for i in range(8)}
    if set(structure_frames) != expected:
        raise DirectExpertSelectionError("structure expert set differs")
    evidence = {
        expert_id: evidence_from_predictions(expert_id, frame, minimum_segment_rows=1)
        for expert_id, frame in structure_frames.items()
    }
    roles: dict[str, str] = {}

    def add(role: str, expert_id: str) -> None:
        if expert_id not in roles.values():
            roles[role] = expert_id

    add("best_weighted", max(expected, key=lambda item: (evidence[item].weighted_gain, -int(item[1:]))))
    add("best_latest_structure", max(expected, key=lambda item: (evidence[item].latest_gain, -int(item[1:]))))
    selected = list(roles.values())
    remaining = expected - set(selected)
    if remaining:
        diverse = min(
            remaining,
            key=lambda item: (
                max(_correlation(structure_frames[item], structure_frames[chosen]) for chosen in selected),
                int(item[1:]),
            ),
        )
        add("most_diverse", diverse)
    wildcard_pool = [
        item
        for item in ("D4", "D5", "D6", "D7", "D0", "D1", "D2", "D3")
        if item not in roles.values() and evidence[item].minimum_fold_gain >= -0.001
    ]
    if wildcard_pool:
        add("wildcard", max(wildcard_pool, key=lambda item: (evidence[item].weighted_gain, -int(item[1:]))))
    for item in sorted(expected, key=lambda key: (-evidence[key].weighted_gain, int(key[1:]))):
        if len(roles) == 4:
            break
        add(f"fill_{len(roles)}", item)
    selected = list(roles.values())
    if any(item in {"D5", "D6"} for item in selected) and "D0" not in selected:
        removable = [item for item in reversed(selected) if item not in {"D5", "D6"}]
        if not removable:
            raise DirectExpertSelectionError("specialist dependency cannot be closed")
        removed = removable[0]
        role = next(name for name, value in roles.items() if value == removed)
        roles[role] = "D0"
        selected = list(roles.values())
    if len(selected) != 4 or len(set(selected)) != 4:
        raise DirectExpertSelectionError("locked expert count differs")
    payload = {
        "expert_ids": selected,
        "roles": roles,
        "locked_on_years": [2022, 2023],
    }
    return LockedSelection(
        expert_ids=tuple(selected),
        roles=MappingProxyType(dict(roles)),
        locked_on_years=(2022, 2023),
        selection_sha256=sha256(canonical_json(payload)).hexdigest(),
    )
