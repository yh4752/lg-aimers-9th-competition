from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .hc_metrics import calibration_gap, expected_calibration_error, paired_cluster_bootstrap
from .s4_contracts import S4Contract, load_s4_contract


class S4DecisionError(ValueError):
    pass


Fold = tuple[int, int]


@dataclass(frozen=True)
class AnchorEvidence:
    candidate_id: str
    mandatory_role: str | None
    weighted_gain: float
    fold_gains: Mapping[Fold, float]
    rf_gain: float
    residual_signature: tuple[float, ...]


@dataclass(frozen=True)
class SelectedAnchor:
    role: str
    candidate_id: str


@dataclass(frozen=True)
class FullChainArchetype:
    candidate_id: str
    anchor_role: str
    anchor_id: str
    residual_family: str
    residual_alpha: float
    calibration_profile: str
    calibration_beta: float


@dataclass(frozen=True)
class S4Evidence:
    candidate_id: str
    confirmed: bool
    weighted_gain: float
    recent_gain: float
    minimum_fold_gain: float
    maximum_segment_regression: float
    bootstrap_lower: float
    bootstrap_upper: float
    non_worse_seed_count: int
    residual_correlation: float
    calibration_gap: float
    ece: float


@dataclass(frozen=True)
class S4Decision:
    candidate_id: str
    status: str
    failed_gates: tuple[str, ...]
    weighted_gain: float
    recent_gain: float
    minimum_fold_gain: float
    maximum_segment_regression: float
    bootstrap_lower: float
    non_worse_seed_count: int


_ROLES = (
    "e2_control", "external_template", "best_weighted",
    "best_worst_fold", "most_diverse", "best_rf",
)


def _finite(value: object, label: str) -> float:
    if type(value) not in {int, float} or not np.isfinite(value):
        raise S4DecisionError(f"{label} differs")
    return float(value)


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    a = np.asarray(left, dtype="float64")
    b = np.asarray(right, dtype="float64")
    if a.ndim != 1 or a.shape != b.shape or a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise S4DecisionError("anchor residual signatures differ")
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    result = float(np.corrcoef(a, b)[0, 1])
    return abs(result) if np.isfinite(result) else 0.0


def select_anchor_coverage(evidence: tuple[AnchorEvidence, ...]) -> tuple[SelectedAnchor, ...]:
    if len(evidence) < 2 or len({item.candidate_id for item in evidence}) != len(evidence):
        raise S4DecisionError("anchor evidence differs")
    registered = set(load_s4_contract().folds[:2])
    for item in evidence:
        if set(item.fold_gains) != registered:
            raise S4DecisionError("anchor evidence folds differ")
        _finite(item.weighted_gain, "anchor weighted gain")
        _finite(item.rf_gain, "anchor R/F gain")
    mandatory: dict[str, AnchorEvidence] = {}
    for item in evidence:
        if item.mandatory_role is not None:
            if item.mandatory_role in mandatory:
                raise S4DecisionError("mandatory anchor role differs")
            mandatory[item.mandatory_role] = item
    if set(mandatory) != {"e2_control", "external_template"}:
        raise S4DecisionError("mandatory anchor coverage differs")
    used = {mandatory["e2_control"].candidate_id, mandatory["external_template"].candidate_id}
    remaining = [item for item in evidence if item.candidate_id not in used]
    if len(remaining) < 4:
        raise S4DecisionError("anchor coverage needs six distinct candidates")
    weighted = max(remaining, key=lambda item: (item.weighted_gain, item.candidate_id))
    used.add(weighted.candidate_id)
    remaining = [item for item in remaining if item.candidate_id not in used]
    worst = max(remaining, key=lambda item: (min(item.fold_gains.values()), item.candidate_id))
    used.add(worst.candidate_id)
    remaining = [item for item in remaining if item.candidate_id not in used]
    diverse = min(
        remaining,
        key=lambda item: (_correlation(weighted.residual_signature, item.residual_signature), item.candidate_id),
    )
    used.add(diverse.candidate_id)
    remaining = [item for item in remaining if item.candidate_id not in used]
    rf = max(remaining, key=lambda item: (item.rf_gain, item.candidate_id))
    chosen = {
        "e2_control": mandatory["e2_control"],
        "external_template": mandatory["external_template"],
        "best_weighted": weighted,
        "best_worst_fold": worst,
        "most_diverse": diverse,
        "best_rf": rf,
    }
    return tuple(SelectedAnchor(role, chosen[role].candidate_id) for role in _ROLES)


def full_chain_archetypes(
    contract: S4Contract, coverage: tuple[SelectedAnchor, ...]
) -> tuple[FullChainArchetype, ...]:
    by_role = {item.role: item.candidate_id for item in coverage}
    if set(by_role) != set(_ROLES) or len(coverage) != len(_ROLES):
        raise S4DecisionError("anchor coverage differs")
    templates = (
        ("e2_control", "catboost", "global_game"),
        ("e2_control", "lightgbm", "matchup"),
        ("external_template", "catboost", "pitcher"),
        ("external_template", "catboost_rf", "rf_matchup"),
        ("external_template", "xgboost", "batter"),
        ("best_weighted", "catboost", "matchup"),
        ("best_weighted", "dual_temporal", "pitcher"),
        ("best_weighted", "xgboost", "global_game"),
        ("best_worst_fold", "catboost", "batter"),
        ("best_worst_fold", "lightgbm", "matchup"),
        ("most_diverse", "xgboost", "pitcher"),
        ("most_diverse", "lightgbm", "batter"),
        ("best_rf", "catboost_rf", "rf_matchup"),
        ("best_rf", "dual_temporal", "matchup"),
        ("best_rf", "catboost", "pitcher"),
    )
    output = []
    for index, (role, family, profile) in enumerate(templates):
        alpha = contract.residual_alphas[index % len(contract.residual_alphas)]
        beta = contract.calibration_betas[index % len(contract.calibration_betas)]
        output.append(FullChainArchetype(
            candidate_id=f"s4__full__{index:02d}__{role}__{family}__{profile}",
            anchor_role=role,
            anchor_id=by_role[role],
            residual_family=family,
            residual_alpha=alpha,
            calibration_profile=profile,
            calibration_beta=beta,
        ))
    if len(output) < contract.minimum_full_chains:
        raise S4DecisionError("full-chain coverage is too small")
    return tuple(output)


def _validated_frame(frame: object, fold: Fold) -> pd.DataFrame:
    required = {"row_id", "target", "p_base", "p_candidate", "pitcher_id", "game_type"}
    if type(frame) is not pd.DataFrame or frame.empty or not required.issubset(frame.columns):
        raise S4DecisionError(f"prediction frame differs: {fold}")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise S4DecisionError("prediction row identity differs")
    result = frame.copy(deep=True)
    for column in ("target", "p_base", "p_candidate"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not result["target"].isin((0, 1)).all() or result[["p_base", "p_candidate"]].isna().any().any():
        raise S4DecisionError("prediction values differ")
    if not result[["p_base", "p_candidate"]].apply(lambda x: x.between(0, 1).all()).all():
        raise S4DecisionError("prediction probabilities differ")
    result["oof_year"] = fold[1]
    return result


def evaluate_full_chain(
    candidate_id: str,
    frames: Mapping[Fold, pd.DataFrame],
    *,
    confirmed: bool,
    non_worse_seed_count: int,
    contract: S4Contract | None = None,
) -> S4Evidence:
    active = load_s4_contract() if contract is None else contract
    if set(frames) != set(active.folds):
        raise S4DecisionError("full-chain fold evidence differs")
    folds: dict[Fold, pd.DataFrame] = {fold: _validated_frame(frames[fold], fold) for fold in active.folds}
    gains: dict[Fold, float] = {}
    joined = []
    segment_regressions = [0.0]
    weights = np.asarray((0.60, 0.75, 0.90), dtype="float64")
    for fold in active.folds:
        frame = folds[fold]
        target = frame["target"].to_numpy(dtype="float64")
        base_loss = np.square(frame["p_base"].to_numpy(dtype="float64") - target)
        candidate_loss = np.square(frame["p_candidate"].to_numpy(dtype="float64") - target)
        row_gain = base_loss - candidate_loss
        gains[fold] = float(row_gain.mean())
        joined.append(frame)
        for value in sorted(frame["game_type"].astype(str).unique()):
            mask = frame["game_type"].astype(str).eq(value).to_numpy()
            if int(mask.sum()) >= active.gates.minimum_segment_rows:
                segment_regressions.append(float(candidate_loss[mask].mean() - base_loss[mask].mean()))
    combined = pd.concat(joined, ignore_index=True).rename(columns={"p_base": "_base", "p_candidate": "_candidate"})
    bootstrap = paired_cluster_bootstrap(
        combined, "_base", "_candidate",
        repeats=active.gates.bootstrap_repeats,
        seed=active.gates.bootstrap_seed,
    )
    target = combined["target"].to_numpy(dtype="float64")
    base_error = target - combined["_base"].to_numpy(dtype="float64")
    candidate_error = target - combined["_candidate"].to_numpy(dtype="float64")
    correlation = 0.0 if np.std(base_error) == 0 or np.std(candidate_error) == 0 else abs(float(np.corrcoef(base_error, candidate_error)[0, 1]))
    probability = combined["_candidate"].to_numpy(dtype="float64")
    return S4Evidence(
        candidate_id=candidate_id,
        confirmed=bool(confirmed),
        weighted_gain=float(np.average([gains[fold] for fold in active.folds], weights=weights)),
        recent_gain=gains[active.folds[-1]],
        minimum_fold_gain=min(gains.values()),
        maximum_segment_regression=max(segment_regressions),
        bootstrap_lower=bootstrap.lower_95,
        bootstrap_upper=bootstrap.upper_95,
        non_worse_seed_count=int(non_worse_seed_count),
        residual_correlation=correlation if np.isfinite(correlation) else 0.0,
        calibration_gap=calibration_gap(target, probability),
        ece=expected_calibration_error(target, probability),
    )


def decide_submission_eligibility(
    evidence: S4Evidence, contract: S4Contract | None = None
) -> S4Decision:
    active = load_s4_contract() if contract is None else contract
    failed = []
    gates = active.gates
    if evidence.weighted_gain < gates.weighted_gain:
        failed.append("weighted_gain")
    if evidence.recent_gain < gates.recent_gain:
        failed.append("recent_gain")
    if evidence.minimum_fold_gain < -gates.maximum_fold_regression:
        failed.append("minimum_fold_gain")
    if evidence.maximum_segment_regression > gates.maximum_segment_regression:
        failed.append("maximum_segment_regression")
    if evidence.bootstrap_lower < 0.0:
        failed.append("bootstrap_lower")
    if evidence.non_worse_seed_count < gates.minimum_non_worse_seed_count:
        failed.append("non_worse_seed_count")
    status = "research_only" if not evidence.confirmed else "rejected" if failed else "accepted"
    return S4Decision(
        candidate_id=evidence.candidate_id,
        status=status,
        failed_gates=tuple(failed),
        weighted_gain=evidence.weighted_gain,
        recent_gain=evidence.recent_gain,
        minimum_fold_gain=evidence.minimum_fold_gain,
        maximum_segment_regression=evidence.maximum_segment_regression,
        bootstrap_lower=evidence.bootstrap_lower,
        non_worse_seed_count=evidence.non_worse_seed_count,
    )


def decision_payload(decision: S4Decision) -> dict[str, object]:
    return {
        "candidate_id": decision.candidate_id,
        "status": decision.status,
        "failed_gates": list(decision.failed_gates),
        "weighted_gain": decision.weighted_gain,
        "recent_gain": decision.recent_gain,
        "minimum_fold_gain": decision.minimum_fold_gain,
        "maximum_segment_regression": decision.maximum_segment_regression,
        "bootstrap_lower": decision.bootstrap_lower,
        "non_worse_seed_count": decision.non_worse_seed_count,
    }
