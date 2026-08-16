from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd


_CONTRACT_PATH = Path(__file__).with_name("score_improvement_contract.json")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "stage_c_campaign_config_sha256",
    "folds",
    "seeds",
    "ensembles",
    "min_weighted_gain",
    "max_fold_degrade",
}
_FOLDS = ("2022->2023", "2023->2024")
_SEEDS = (42, 2026, 3407)
_ENSEMBLE_SEEDS = {
    "mean_all": frozenset(_SEEDS),
    "mean_42_3407": frozenset((42, 3407)),
}
_STAGE_C_CANDIDATE_PREFIX = "c_final__a__p2__piecewise_linear__bce__plateau__s42"


class EnsembleAuditError(ValueError):
    """Raised when an ensemble contract or its evidence is not trustworthy."""


@dataclass(frozen=True)
class EnsembleAuditContract:
    stage_c_campaign_config_sha256: str
    folds: tuple[str, ...]
    seeds: tuple[int, ...]
    ensembles: Mapping[str, Mapping[int, float]]
    min_weighted_gain: float
    max_fold_degrade: float


@dataclass(frozen=True)
class CandidateAudit:
    candidate_id: str
    weights: Mapping[int, float]
    fold_brier: Mapping[str, float]
    weighted_brier: float
    weighted_gain: float
    worst_fold_degrade: float
    gate_passed: bool


@dataclass(frozen=True)
class EnsembleAuditResult:
    decision: str
    baseline_candidate_id: str
    selected_candidate_id: str
    row_counts: Mapping[str, int]
    candidates: Mapping[str, CandidateAudit]


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in items:
        if key in output:
            raise EnsembleAuditError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> object:
    raise EnsembleAuditError(f"contract number must be finite: {value}")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except EnsembleAuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EnsembleAuditError(f"cannot read ensemble contract: {error}") from error
    if not isinstance(value, dict):
        raise EnsembleAuditError("ensemble contract must be a JSON object")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnsembleAuditError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EnsembleAuditError(f"{label} must be finite")
    return result


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EnsembleAuditError(f"{label} must be a lowercase SHA-256")
    return value


def _ordered_tuple(
    value: object,
    label: str,
    convert: Callable[[object], object],
) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise EnsembleAuditError(f"{label} must be a list")
    converted = tuple(convert(item) for item in value)
    if len(converted) != len(set(converted)):
        raise EnsembleAuditError(f"{label} must not contain duplicates")
    return converted


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EnsembleAuditError("seeds must contain integers")
    return value


def _fold(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise EnsembleAuditError("folds must contain non-empty strings")
    return value


def _ensembles(value: object, seeds: tuple[int, ...]) -> Mapping[str, Mapping[int, float]]:
    if not isinstance(value, dict):
        raise EnsembleAuditError("ensembles must be an object")
    if set(value) != set(_ENSEMBLE_SEEDS):
        raise EnsembleAuditError("ensembles must contain the sealed candidate names")

    output: dict[str, Mapping[int, float]] = {}
    for name in sorted(value):
        raw_weights = value[name]
        if not isinstance(raw_weights, dict) or not raw_weights:
            raise EnsembleAuditError(f"ensemble weights must be an object: {name}")
        weights: dict[int, float] = {}
        for raw_seed, raw_weight in raw_weights.items():
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError) as error:
                raise EnsembleAuditError(f"ensemble seed must be an integer: {name}") from error
            if str(seed) != raw_seed or seed not in seeds:
                raise EnsembleAuditError(f"ensemble contains unknown seed: {name}/{raw_seed}")
            weight = _finite_number(raw_weight, f"ensemble weight {name}/{seed}")
            if weight <= 0:
                raise EnsembleAuditError(f"ensemble weights must be positive: {name}")
            weights[seed] = weight
        if frozenset(weights) != _ENSEMBLE_SEEDS[name]:
            raise EnsembleAuditError(f"ensemble seed set differs: {name}")
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise EnsembleAuditError(f"ensemble weights must sum to 1: {name}")
        output[name] = MappingProxyType(dict(sorted(weights.items())))
    return MappingProxyType(output)


def load_ensemble_contract(path: str | Path | None = None) -> EnsembleAuditContract:
    contract_path = _CONTRACT_PATH if path is None else Path(path)
    payload = _load_json(contract_path)
    unknown = set(payload) - _TOP_LEVEL_KEYS
    missing = _TOP_LEVEL_KEYS - set(payload)
    if unknown:
        raise EnsembleAuditError(f"contract has unknown keys: {sorted(unknown)}")
    if missing:
        raise EnsembleAuditError(f"contract is missing keys: {sorted(missing)}")
    if payload["schema_version"] != 1:
        raise EnsembleAuditError("schema_version must be 1")

    folds = _ordered_tuple(payload["folds"], "folds", _fold)
    seeds = _ordered_tuple(payload["seeds"], "seeds", _seed)
    if folds != _FOLDS:
        raise EnsembleAuditError("folds differ from the sealed folds")
    if seeds != _SEEDS:
        raise EnsembleAuditError("seeds differ from the sealed seeds")

    min_gain = _finite_number(payload["min_weighted_gain"], "min_weighted_gain")
    max_degrade = _finite_number(payload["max_fold_degrade"], "max_fold_degrade")
    if min_gain < 0 or max_degrade < 0:
        raise EnsembleAuditError("gates must be non-negative")

    return EnsembleAuditContract(
        stage_c_campaign_config_sha256=_sha256(
            payload["stage_c_campaign_config_sha256"],
            "stage_c_campaign_config_sha256",
        ),
        folds=folds,  # type: ignore[arg-type]
        seeds=seeds,  # type: ignore[arg-type]
        ensembles=_ensembles(payload["ensembles"], seeds),  # type: ignore[arg-type]
        min_weighted_gain=min_gain,
        max_fold_degrade=max_degrade,
    )


def prediction_member(fold: str, seed: int) -> str:
    if fold not in _FOLDS or seed not in _SEEDS:
        raise EnsembleAuditError(f"unknown fold or seed: {fold}/{seed}")
    train_year, validation_year = fold.split("->")
    candidate_id = (
        f"{_STAGE_C_CANDIDATE_PREFIX}__s{seed}"
        f"__tr{train_year}__va{validation_year}"
    )
    return f"predictions/{candidate_id}.csv"


def _required_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise EnsembleAuditError(f"{label} is missing columns: {sorted(missing)}")


def _row_ids(frame: pd.DataFrame, label: str) -> pd.Series:
    values = frame["row_id"]
    if values.isna().any() or not values.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise EnsembleAuditError(f"{label} has invalid row_id")
    if values.duplicated().any():
        raise EnsembleAuditError(f"{label} has duplicate row_id")
    return values


def _binary(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise EnsembleAuditError(f"{label} must be binary")
    return numeric


def _probabilities(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(numeric).all() or ((numeric < 0.0) | (numeric > 1.0)).any():
        raise EnsembleAuditError(f"{label} probability must be finite and within [0, 1]")
    return numeric


def _candidate_weights(contract: EnsembleAuditContract) -> Mapping[str, Mapping[int, float]]:
    output: dict[str, Mapping[int, float]] = {
        f"seed_{seed}": MappingProxyType({seed: 1.0}) for seed in contract.seeds
    }
    output.update(contract.ensembles)
    return MappingProxyType(output)


def audit_prediction_frames(
    contract: EnsembleAuditContract,
    predictions: Mapping[tuple[str, int], pd.DataFrame],
    truth: pd.DataFrame,
) -> EnsembleAuditResult:
    """Score sealed seed combinations after strict row and target alignment."""

    expected_keys = {(fold, seed) for fold in contract.folds for seed in contract.seeds}
    if set(predictions) != expected_keys:
        raise EnsembleAuditError("prediction fold and seed set differs from contract")

    _required_columns(truth, {"row_id", "season", "control_success"}, "truth")
    _row_ids(truth, "truth")
    truth_targets = _binary(truth["control_success"], "truth control_success")
    truth_frame = truth.loc[:, ["row_id", "season"]].copy()
    truth_frame["control_success"] = truth_targets

    aligned: dict[tuple[str, int], np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    row_counts: dict[str, int] = {}
    for fold in contract.folds:
        validation_year = int(fold.rsplit("->", 1)[1])
        fold_truth = truth_frame.loc[truth_frame["season"] == validation_year].copy()
        if fold_truth.empty:
            raise EnsembleAuditError(f"truth has no rows for fold: {fold}")
        expected_ids = fold_truth["row_id"].tolist()
        expected_id_set = set(expected_ids)
        fold_target = fold_truth["control_success"].to_numpy(dtype=np.float64, copy=True)
        targets[fold] = fold_target
        row_counts[fold] = len(fold_truth)

        for seed in contract.seeds:
            label = f"prediction {fold}/seed_{seed}"
            frame = predictions[(fold, seed)]
            _required_columns(frame, {"row_id", "target", "probability"}, label)
            ids = _row_ids(frame, label)
            if set(ids) != expected_id_set or len(frame) != len(fold_truth):
                raise EnsembleAuditError(f"{label} row_id set differs from truth")
            indexed = frame.set_index("row_id", verify_integrity=True).loc[expected_ids]
            prediction_target = _binary(indexed["target"], f"{label} target")
            if not np.array_equal(prediction_target, fold_target):
                raise EnsembleAuditError(f"{label} target differs from truth")
            aligned[(fold, seed)] = _probabilities(
                indexed["probability"],
                label,
            )

    weights_by_candidate = _candidate_weights(contract)
    fold_scores: dict[str, dict[str, float]] = {}
    weighted_scores: dict[str, float] = {}
    total_rows = sum(row_counts.values())
    for candidate_id, weights in weights_by_candidate.items():
        scores: dict[str, float] = {}
        for fold in contract.folds:
            probability = sum(
                aligned[(fold, seed)] * weight for seed, weight in weights.items()
            )
            scores[fold] = float(np.mean(np.square(probability - targets[fold])))
        fold_scores[candidate_id] = scores
        weighted_scores[candidate_id] = sum(
            scores[fold] * row_counts[fold] for fold in contract.folds
        ) / total_rows

    single_ids = tuple(f"seed_{seed}" for seed in contract.seeds)
    baseline_id = min(single_ids, key=lambda item: (weighted_scores[item], item))
    baseline_weighted = weighted_scores[baseline_id]
    baseline_folds = fold_scores[baseline_id]

    audits: dict[str, CandidateAudit] = {}
    for candidate_id, weights in weights_by_candidate.items():
        weighted_gain = baseline_weighted - weighted_scores[candidate_id]
        worst_degrade = max(
            fold_scores[candidate_id][fold] - baseline_folds[fold]
            for fold in contract.folds
        )
        gate_passed = (
            candidate_id in contract.ensembles
            and weighted_gain >= contract.min_weighted_gain
            and worst_degrade <= contract.max_fold_degrade
        )
        audits[candidate_id] = CandidateAudit(
            candidate_id=candidate_id,
            weights=weights,
            fold_brier=MappingProxyType(dict(fold_scores[candidate_id])),
            weighted_brier=weighted_scores[candidate_id],
            weighted_gain=weighted_gain,
            worst_fold_degrade=worst_degrade,
            gate_passed=gate_passed,
        )

    promoted = [
        candidate_id
        for candidate_id in contract.ensembles
        if audits[candidate_id].gate_passed
    ]
    if promoted:
        selected = min(promoted, key=lambda item: (weighted_scores[item], item))
        decision = "promoted"
    else:
        selected = baseline_id
        decision = "keep_single"

    return EnsembleAuditResult(
        decision=decision,
        baseline_candidate_id=baseline_id,
        selected_candidate_id=selected,
        row_counts=MappingProxyType(dict(row_counts)),
        candidates=MappingProxyType(audits),
    )
