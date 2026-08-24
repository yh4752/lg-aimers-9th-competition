from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.oof_reset_audit.metrics import (
    block_bootstrap_interval,
    segment_metrics,
)
from experiments.oof_reset_audit.types import PredictionSet, TrustClass

from .contracts import RealignContract


TABM_COLUMNS = (
    "row_id",
    "target",
    "probability",
    "game_type",
    "game_month",
    "pitcher_id_known",
    "batter_id_known",
)
CATBOOST_COLUMNS = (
    "row_id",
    "target",
    "p_4",
    "p_8",
    "p_12",
    "p_16",
    "p_20",
    "p_24",
    "p_28",
    "p_32",
    "game_type",
    "game_month",
    "pitcher_id_known",
    "batter_id_known",
)
SELECTION_ORDER = (
    "maximum_minimum_fold_gain",
    "maximum_weighted_gain",
    "minimum_tree_count",
)
_FOLDS = ("2021->2022", "2022->2023", "2023->2024")
_LATEST_FOLD = "2023->2024"
_MINIMUM_SEGMENT_ROWS = 5_000


class RealignMetricError(ValueError):
    """Raised when OOF evidence cannot support a preregistered decision."""


@dataclass(frozen=True)
class PrefixEvidence:
    tree_count: int
    fold_gain: Mapping[str, float]
    weighted_gain: float
    latest_bootstrap_lower: float | None
    maximum_segment_regression: float
    eligible_segment_count: int
    bootstrap_status: str
    passed: bool


@dataclass(frozen=True)
class RealignDecision:
    status: str
    selected_tree_count: int | None
    candidates: tuple[PrefixEvidence, ...]
    reason: str
    selection_order: tuple[str, ...] = SELECTION_ORDER


def _validate_fold_set(values: Mapping[str, pd.DataFrame], label: str) -> None:
    if type(values) is not dict or set(values) != set(_FOLDS):
        raise RealignMetricError(f"{label} fold set differs")


def _validate_frame(
    frame: pd.DataFrame, columns: tuple[str, ...], label: str
) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or tuple(frame.columns) != columns or frame.empty:
        raise RealignMetricError(f"{label} columns differ")
    result = frame.copy()
    if result["row_id"].isna().any():
        raise RealignMetricError("row_id is missing")
    result["row_id"] = result["row_id"].astype(str)
    if result["row_id"].duplicated().any():
        raise RealignMetricError("row_id must be unique")
    target = pd.to_numeric(result["target"], errors="coerce").to_numpy("float64")
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise RealignMetricError("target must contain only 0 and 1")
    result["target"] = target.astype("int8")
    probability_columns = ("probability",) if label == "TabM" else columns[2:10]
    for column in probability_columns:
        probability = pd.to_numeric(result[column], errors="coerce").to_numpy("float64")
        if (
            not np.isfinite(probability).all()
            or ((probability < 0.0) | (probability > 1.0)).any()
        ):
            raise RealignMetricError(f"probability is invalid: {column}")
        result[column] = probability
    return result


def _align(
    tabm: pd.DataFrame, catboost: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = _validate_frame(tabm, TABM_COLUMNS, "TabM").reset_index(drop=True)
    right = _validate_frame(catboost, CATBOOST_COLUMNS, "CatBoost")
    if set(left["row_id"]) != set(right["row_id"]):
        raise RealignMetricError("row_id set differs")
    right = right.set_index("row_id", drop=False).loc[left["row_id"]].reset_index(drop=True)
    if not np.array_equal(left["target"].to_numpy(), right["target"].to_numpy()):
        raise RealignMetricError("target differs")
    for column in TABM_COLUMNS[3:]:
        matches = left[column].eq(right[column]) | (
            left[column].isna() & right[column].isna()
        )
        if not bool(matches.all()):
            raise RealignMetricError(f"segment differs: {column}")
    return left, right


def _blend(tabm: np.ndarray, catboost: np.ndarray) -> np.ndarray:
    return 0.5 * tabm + 0.5 * np.clip(catboost, 0.0, 1.0)


def _prediction_set(
    frame: pd.DataFrame, probability: np.ndarray, *, fold: str, model_id: str
) -> PredictionSet:
    evidence = frame[
        ["row_id", "target", "game_type", "game_month"]
    ].copy()
    evidence["pitcher_known"] = frame["pitcher_id_known"].to_numpy()
    evidence["batter_known"] = frame["batter_id_known"].to_numpy()
    evidence["probability"] = probability
    return PredictionSet(
        artifact_path=Path("fixture"),
        artifact_sha256="0" * 64,
        source_member=model_id,
        prediction_sha256="0" * 64,
        model_id=model_id,
        fold=fold,
        trust=TrustClass.RULE_SAFE,
        frame=evidence,
    )


def passes_gates(item: PrefixEvidence, contract: RealignContract) -> bool:
    numeric = (
        *item.fold_gain.values(),
        item.weighted_gain,
        item.maximum_segment_regression,
    )
    if item.latest_bootstrap_lower is not None:
        numeric += (item.latest_bootstrap_lower,)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        for value in numeric
    ):
        return False
    return (
        set(item.fold_gain) == set(_FOLDS)
        and all(Decimal(str(value)) > Decimal("0") for value in item.fold_gain.values())
        and Decimal(str(item.weighted_gain)) >= contract.minimum_weighted_gain
        and item.latest_bootstrap_lower is not None
        and Decimal(str(item.latest_bootstrap_lower))
        >= contract.latest_bootstrap_lower_minimum
        and Decimal(str(item.maximum_segment_regression))
        <= contract.maximum_segment_regression
        and item.eligible_segment_count > 0
        and item.bootstrap_status == "completed"
    )


def select_candidate(
    candidates: Sequence[PrefixEvidence], contract: RealignContract
) -> PrefixEvidence:
    eligible = [candidate for candidate in candidates if candidate.passed]
    if not eligible:
        raise RealignMetricError("no prefix passed all gates")
    tolerance = float(contract.selection_tolerance)
    best_minimum = max(min(item.fold_gain.values()) for item in eligible)
    eligible = [
        item
        for item in eligible
        if best_minimum - min(item.fold_gain.values()) <= tolerance
    ]
    best_weighted = max(item.weighted_gain for item in eligible)
    eligible = [
        item for item in eligible if best_weighted - item.weighted_gain <= tolerance
    ]
    return min(eligible, key=lambda item: item.tree_count)


def evaluate_prefixes(
    tabm_by_fold: Mapping[str, pd.DataFrame],
    catboost_by_fold: Mapping[str, pd.DataFrame],
    contract: RealignContract,
) -> RealignDecision:
    _validate_fold_set(tabm_by_fold, "TabM")
    _validate_fold_set(catboost_by_fold, "CatBoost")
    aligned = {
        fold: _align(tabm_by_fold[fold], catboost_by_fold[fold]) for fold in _FOLDS
    }
    candidates: list[PrefixEvidence] = []
    for tree_count in contract.tree_prefixes:
        fold_gain: dict[str, float] = {}
        total_gain = 0.0
        total_rows = 0
        latest_interval: dict[str, object] | None = None
        maximum_regression = -np.inf
        eligible_count = 0
        for fold in _FOLDS:
            tabm_frame, catboost_frame = aligned[fold]
            target = tabm_frame["target"].to_numpy("float64")
            tabm_probability = tabm_frame["probability"].to_numpy("float64")
            catboost_probability = catboost_frame[f"p_{tree_count}"].to_numpy("float64")
            blended = _blend(tabm_probability, catboost_probability)
            loss_delta = np.square(tabm_probability - target) - np.square(
                blended - target
            )
            gain = float(loss_delta.mean(dtype=np.float64))
            fold_gain[fold] = gain
            total_gain += float(loss_delta.sum(dtype=np.float64))
            total_rows += len(loss_delta)
            anchor = _prediction_set(
                tabm_frame, tabm_probability, fold=fold, model_id="tabm"
            )
            candidate = _prediction_set(
                tabm_frame,
                blended,
                fold=fold,
                model_id=f"blend_p_{tree_count}",
            )
            segments = segment_metrics(
                anchor, candidate, minimum_rows=_MINIMUM_SEGMENT_ROWS
            )
            if not segments.empty:
                eligible = segments[segments["eligible"]]
                if not eligible.empty:
                    eligible_count += len(eligible)
                    maximum_regression = max(
                        maximum_regression, float(eligible["regression"].max())
                    )
            if fold == _LATEST_FOLD:
                blocks = tabm_frame["game_month"].astype(str)
                latest_interval = block_bootstrap_interval(
                    pd.DataFrame(
                        {
                            "block": [f"{fold}:{value}" for value in blocks],
                            "loss_delta": loss_delta,
                        }
                    ),
                    repeats=contract.bootstrap_repeats,
                    seed=contract.bootstrap_seed,
                )
        status = (
            str(latest_interval["status"])
            if latest_interval is not None
            else "missing"
        )
        lower = (
            float(latest_interval["lower"])
            if latest_interval is not None
            and latest_interval.get("status") == "completed"
            else None
        )
        evidence = PrefixEvidence(
            tree_count=tree_count,
            fold_gain=MappingProxyType(fold_gain),
            weighted_gain=total_gain / total_rows,
            latest_bootstrap_lower=lower,
            maximum_segment_regression=(
                float(maximum_regression) if eligible_count else float("inf")
            ),
            eligible_segment_count=eligible_count,
            bootstrap_status=status,
            passed=False,
        )
        evidence = replace(evidence, passed=passes_gates(evidence, contract))
        candidates.append(evidence)
    immutable = tuple(candidates)
    passed = tuple(item for item in immutable if item.passed)
    if not passed:
        return RealignDecision(
            status="rejected",
            selected_tree_count=None,
            candidates=immutable,
            reason="no_prefix_passed_all_preregistered_gates",
        )
    selected = select_candidate(passed, contract)
    return RealignDecision(
        status="promoted",
        selected_tree_count=selected.tree_count,
        candidates=immutable,
        reason="selected_by_preregistered_order",
    )


def decision_to_payload_evidence(item: PrefixEvidence) -> dict[str, object]:
    return {
        "tree_count": item.tree_count,
        "fold_gain": dict(item.fold_gain),
        "weighted_gain": item.weighted_gain,
        "latest_bootstrap_lower": item.latest_bootstrap_lower,
        "maximum_segment_regression": item.maximum_segment_regression,
        "eligible_segment_count": item.eligible_segment_count,
        "bootstrap_status": item.bootstrap_status,
        "passed": item.passed,
    }


def decision_to_payload(decision: RealignDecision) -> dict[str, object]:
    return {
        "status": decision.status,
        "selected_tree_count": decision.selected_tree_count,
        "candidates": [decision_to_payload_evidence(item) for item in decision.candidates],
        "reason": decision.reason,
        "selection_order": list(decision.selection_order),
    }


def decision_from_payload(
    payload: Mapping[str, object], contract: RealignContract
) -> RealignDecision:
    def finite_float(value: object, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RealignMetricError(f"{label} must be numeric")
        result = float(value)
        if not np.isfinite(result):
            raise RealignMetricError(f"{label} must be finite")
        return result

    if type(payload) is not dict or set(payload) != {
        "status",
        "selected_tree_count",
        "candidates",
        "reason",
        "selection_order",
    }:
        raise RealignMetricError("decision payload differs")
    raw_candidates = payload["candidates"]
    if type(raw_candidates) is not list:
        raise RealignMetricError("decision candidates differ")
    candidates: list[PrefixEvidence] = []
    expected_keys = {
        "tree_count",
        "fold_gain",
        "weighted_gain",
        "latest_bootstrap_lower",
        "maximum_segment_regression",
        "eligible_segment_count",
        "bootstrap_status",
        "passed",
    }
    for raw in raw_candidates:
        if type(raw) is not dict or set(raw) != expected_keys:
            raise RealignMetricError("candidate payload differs")
        fold_gain = raw["fold_gain"]
        if type(fold_gain) is not dict or set(fold_gain) != set(_FOLDS):
            raise RealignMetricError("candidate fold gain differs")
        if (
            isinstance(raw["tree_count"], bool)
            or type(raw["tree_count"]) is not int
            or raw["tree_count"] not in contract.tree_prefixes
            or isinstance(raw["eligible_segment_count"], bool)
            or type(raw["eligible_segment_count"]) is not int
            or raw["eligible_segment_count"] < 0
            or type(raw["bootstrap_status"]) is not str
            or type(raw["passed"]) is not bool
        ):
            raise RealignMetricError("candidate scalar fields differ")
        candidate = PrefixEvidence(
            tree_count=raw["tree_count"],
            fold_gain=MappingProxyType(
                {
                    fold: finite_float(fold_gain[fold], f"fold gain {fold}")
                    for fold in _FOLDS
                }
            ),
            weighted_gain=finite_float(raw["weighted_gain"], "weighted gain"),
            latest_bootstrap_lower=(
                None
                if raw["latest_bootstrap_lower"] is None
                else finite_float(
                    raw["latest_bootstrap_lower"], "bootstrap lower"
                )
            ),
            maximum_segment_regression=finite_float(
                raw["maximum_segment_regression"], "segment regression"
            ),
            eligible_segment_count=raw["eligible_segment_count"],
            bootstrap_status=raw["bootstrap_status"],
            passed=raw["passed"],
        )
        if candidate.passed != passes_gates(candidate, contract):
            raise RealignMetricError("candidate gate result differs")
        candidates.append(candidate)
    if tuple(item.tree_count for item in candidates) != contract.tree_prefixes:
        raise RealignMetricError("candidate prefix set differs")
    if (
        type(payload["status"]) is not str
        or type(payload["reason"]) is not str
        or type(payload["selection_order"]) is not list
        or any(type(value) is not str for value in payload["selection_order"])
        or (
            payload["selected_tree_count"] is not None
            and (
                isinstance(payload["selected_tree_count"], bool)
                or type(payload["selected_tree_count"]) is not int
            )
        )
    ):
        raise RealignMetricError("decision scalar fields differ")
    decision = RealignDecision(
        status=payload["status"],
        selected_tree_count=(
            None
            if payload["selected_tree_count"] is None
            else payload["selected_tree_count"]
        ),
        candidates=tuple(candidates),
        reason=payload["reason"],
        selection_order=tuple(payload["selection_order"]),
    )
    if decision.selection_order != SELECTION_ORDER:
        raise RealignMetricError("selection order differs")
    if decision.status == "promoted":
        if decision.reason != "selected_by_preregistered_order":
            raise RealignMetricError("decision reason differs")
        selected = select_candidate(decision.candidates, contract)
        if decision.selected_tree_count != selected.tree_count:
            raise RealignMetricError("selected prefix differs")
    elif (
        decision.status != "rejected"
        or decision.selected_tree_count is not None
        or decision.reason != "no_prefix_passed_all_preregistered_gates"
        or any(item.passed for item in decision.candidates)
    ):
        raise RealignMetricError("decision status differs")
    return decision
