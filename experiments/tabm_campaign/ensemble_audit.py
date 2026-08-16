from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping


_CONTRACT_PATH = Path(__file__).with_name("score_improvement_contract.json")
_TOP_LEVEL_KEYS = {
    "schema_version",
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


class EnsembleAuditError(ValueError):
    """Raised when an ensemble contract or its evidence is not trustworthy."""


@dataclass(frozen=True)
class EnsembleAuditContract:
    folds: tuple[str, ...]
    seeds: tuple[int, ...]
    ensembles: Mapping[str, Mapping[int, float]]
    min_weighted_gain: float
    max_fold_degrade: float


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
        folds=folds,  # type: ignore[arg-type]
        seeds=seeds,  # type: ignore[arg-type]
        ensembles=_ensembles(payload["ensembles"], seeds),  # type: ignore[arg-type]
        min_weighted_gain=min_gain,
        max_fold_degrade=max_degrade,
    )
