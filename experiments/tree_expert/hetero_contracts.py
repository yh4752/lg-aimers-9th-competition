from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class HeteroContractError(ValueError):
    pass


DEFAULT_HETERO_CONTRACT = Path(__file__).with_name("hetero_contract.json")
_FAMILIES = ("xgboost", "lightgbm")
_FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
_WEIGHTS = (0.05, 0.10, 0.15)
_CONFIRMATION_SEEDS = (42, 2026)


@dataclass(frozen=True)
class HeteroGates:
    weighted_gain: float
    recent_fold_gain: float
    maximum_fold_regression: float
    maximum_segment_regression: float
    maximum_residual_correlation: float
    minimum_segment_rows: int
    bootstrap_repeats: int
    bootstrap_seed: int
    minimum_non_worse_seed_count: int
    blend_incremental_gain: float


@dataclass(frozen=True)
class HeteroContract:
    campaign_id: str
    inputs: Mapping[str, str]
    families: tuple[str, ...]
    versions: Mapping[str, str]
    folds: tuple[tuple[int, int], ...]
    correction_weights: tuple[float, ...]
    structure_seed: int
    confirmation_seeds: tuple[int, ...]
    parameters: Mapping[str, Mapping[str, object]]
    gates: HeteroGates
    wall_seconds: int
    new_job_guard_seconds: int
    snapshot_interval_seconds: int
    probability_tolerance: float
    rss_max_bytes: int


@dataclass(frozen=True)
class HeteroJob:
    job_id: str
    family: str
    train_end_year: int
    valid_year: int
    seed: int


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise HeteroContractError(f"{label} keys differ")
    return dict(value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise HeteroContractError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise HeteroContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HeteroContractError(f"{label} must be finite")
    return result


def load_hetero_contract(path: Path = DEFAULT_HETERO_CONTRACT) -> HeteroContract:
    try:
        root = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeteroContractError("hetero contract cannot be loaded") from error
    root = _object(root, {
        "schema_version", "campaign_id", "inputs", "families", "versions",
        "folds", "correction_weights", "structure_seed", "confirmation_seeds",
        "xgboost", "lightgbm", "gates", "runtime",
    }, "hetero contract")
    if root["schema_version"] != 1 or root["campaign_id"] != "tree_hetero_residual_s3_v1":
        raise HeteroContractError("hetero contract identity differs")
    inputs = _object(root["inputs"], {
        "official_train_sha256", "official_history_sha256", "e2_handoff_sha256",
    }, "hetero inputs")
    if any(type(value) is not str or len(value) != 64 for value in inputs.values()):
        raise HeteroContractError("hetero input hashes differ")
    families = tuple(root["families"])
    folds = tuple(tuple(item) for item in root["folds"])
    weights = tuple(_number(item, "correction weight") for item in root["correction_weights"])
    confirmation = tuple(_integer(item, "confirmation seed") for item in root["confirmation_seeds"])
    if families != _FAMILIES or folds != _FOLDS or weights != _WEIGHTS or confirmation != _CONFIRMATION_SEEDS:
        raise HeteroContractError("search grid differs")
    if root["structure_seed"] != 3407:
        raise HeteroContractError("structure seed differs")
    versions = _object(root["versions"], set(_FAMILIES), "hetero versions")
    if versions != {"xgboost": "3.0.2", "lightgbm": "4.6.0"}:
        raise HeteroContractError("dependency versions differ")
    xgboost = _object(root["xgboost"], {
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda",
        "max_bin", "tree_method",
    }, "XGBoost parameters")
    lightgbm = _object(root["lightgbm"], {
        "n_estimators", "num_leaves", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_samples", "reg_alpha", "reg_lambda",
        "max_bin", "num_threads",
    }, "LightGBM parameters")
    gates = _object(root["gates"], {
        "weighted_gain", "recent_fold_gain", "maximum_fold_regression",
        "maximum_segment_regression", "maximum_residual_correlation",
        "minimum_segment_rows", "bootstrap_repeats", "bootstrap_seed",
        "minimum_non_worse_seed_count", "blend_incremental_gain",
    }, "hetero gates")
    runtime = _object(root["runtime"], {
        "wall_seconds", "new_job_guard_seconds", "snapshot_interval_seconds",
        "probability_tolerance", "rss_max_bytes",
    }, "hetero runtime")
    return HeteroContract(
        campaign_id=str(root["campaign_id"]),
        inputs=MappingProxyType({str(key): str(value) for key, value in inputs.items()}),
        families=families,
        versions=MappingProxyType({str(key): str(value) for key, value in versions.items()}),
        folds=folds,
        correction_weights=weights,
        structure_seed=_integer(root["structure_seed"], "structure seed"),
        confirmation_seeds=confirmation,
        parameters=MappingProxyType({
            "xgboost": MappingProxyType(xgboost),
            "lightgbm": MappingProxyType(lightgbm),
        }),
        gates=HeteroGates(
            weighted_gain=_number(gates["weighted_gain"], "weighted gain"),
            recent_fold_gain=_number(gates["recent_fold_gain"], "recent fold gain"),
            maximum_fold_regression=_number(gates["maximum_fold_regression"], "fold regression"),
            maximum_segment_regression=_number(gates["maximum_segment_regression"], "segment regression"),
            maximum_residual_correlation=_number(gates["maximum_residual_correlation"], "residual correlation"),
            minimum_segment_rows=_integer(gates["minimum_segment_rows"], "minimum segment rows"),
            bootstrap_repeats=_integer(gates["bootstrap_repeats"], "bootstrap repeats"),
            bootstrap_seed=_integer(gates["bootstrap_seed"], "bootstrap seed"),
            minimum_non_worse_seed_count=_integer(gates["minimum_non_worse_seed_count"], "minimum seed count"),
            blend_incremental_gain=_number(gates["blend_incremental_gain"], "blend gain"),
        ),
        wall_seconds=_integer(runtime["wall_seconds"], "wall seconds"),
        new_job_guard_seconds=_integer(runtime["new_job_guard_seconds"], "job guard"),
        snapshot_interval_seconds=_integer(runtime["snapshot_interval_seconds"], "snapshot interval"),
        probability_tolerance=_number(runtime["probability_tolerance"], "probability tolerance"),
        rss_max_bytes=_integer(runtime["rss_max_bytes"], "RSS bytes"),
    )


def contract_sha256(path: Path = DEFAULT_HETERO_CONTRACT) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _jobs(contract: HeteroContract, family: str, seeds: tuple[int, ...]) -> tuple[HeteroJob, ...]:
    if family not in contract.families:
        raise HeteroContractError("family differs")
    return tuple(
        HeteroJob(
            f"hetero__{family}__tr{train_end}__va{valid_year}__s{seed}",
            family, train_end, valid_year, seed,
        )
        for train_end, valid_year in contract.folds
        for seed in seeds
    )


def structure_jobs(contract: HeteroContract) -> tuple[HeteroJob, ...]:
    return tuple(
        job
        for family in contract.families
        for job in _jobs(contract, family, (contract.structure_seed,))
    )


def confirmation_jobs(contract: HeteroContract, family: str) -> tuple[HeteroJob, ...]:
    return _jobs(contract, family, contract.confirmation_seeds)
