from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class RFContractError(ValueError):
    pass


DEFAULT_RF_CONTRACT = Path(__file__).with_name("rf_contract.json")
_FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
_F_HEADS = ("f_small", "f_wide")
_HEADS = (*_F_HEADS, "r_expert")
_ALPHAS = (0.25, 0.50, 0.75, 1.00)
_CONFIRMATION_SEEDS = (42, 2026)


@dataclass(frozen=True)
class RFGates:
    weighted_gain: float
    minimum_improved_folds: int
    maximum_segment_regression: float
    recent_f_gain: float
    minimum_segment_rows: int
    probability_tolerance: float


@dataclass(frozen=True)
class RFContract:
    campaign_id: str
    inputs: Mapping[str, str]
    folds: tuple[tuple[int, int], ...]
    structure_seed: int
    confirmation_seeds: tuple[int, ...]
    alpha_values: tuple[float, ...]
    experts: Mapping[str, Mapping[str, object]]
    catboost_common: Mapping[str, object]
    gates: RFGates
    wall_seconds: int
    new_job_guard_seconds: int
    full_fit_guard_seconds: int
    snapshot_interval_seconds: int
    progress_interval_seconds: int
    inference_max_seconds: int
    rss_max_bytes: int
    probability_tolerance: float


@dataclass(frozen=True)
class RFJob:
    job_id: str
    head: str
    segment: str
    train_end_year: int
    valid_year: int
    seed: int


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise RFContractError(f"{label} keys differ")
    return dict(value)


def _int(value: object, label: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise RFContractError(f"{label} must be an integer")
    return value


def _float(value: object, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise RFContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RFContractError(f"{label} must be finite")
    return result


def load_rf_contract(path: Path = DEFAULT_RF_CONTRACT) -> RFContract:
    try:
        root = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RFContractError("RF contract cannot be loaded") from error
    root = _object(
        root,
        {
            "schema_version", "campaign_id", "inputs", "folds",
            "structure_seed", "confirmation_seeds", "alpha_values", "experts",
            "catboost_common", "gates", "runtime",
        },
        "RF contract",
    )
    if root["schema_version"] != 1 or root["campaign_id"] != "tree_expert_rf_v1":
        raise RFContractError("RF contract identity differs")
    inputs = _object(
        root["inputs"],
        {"official_train_sha256", "official_history_sha256", "e2_handoff_sha256"},
        "RF inputs",
    )
    if any(type(value) is not str or len(value) != 64 for value in inputs.values()):
        raise RFContractError("RF input hashes differ")
    folds = tuple(tuple(item) for item in root["folds"])
    seeds = tuple(_int(item, "confirmation seed") for item in root["confirmation_seeds"])
    alphas = tuple(_float(item, "alpha") for item in root["alpha_values"])
    if folds != _FOLDS or seeds != _CONFIRMATION_SEEDS or alphas != _ALPHAS:
        raise RFContractError("RF search grid differs")
    if _int(root["structure_seed"], "structure seed") != 3407:
        raise RFContractError("RF search grid differs")

    expert_root = _object(root["experts"], set(_HEADS), "RF experts")
    experts: dict[str, Mapping[str, object]] = {}
    expert_keys = {
        "segment", "iterations", "depth", "l2_leaf_reg",
        "random_strength", "bagging_temperature",
    }
    for head in _HEADS:
        values = _object(expert_root[head], expert_keys, f"RF expert {head}")
        expected_segment = "F" if head in _F_HEADS else "R"
        if values["segment"] != expected_segment:
            raise RFContractError("RF expert segment differs")
        for key in ("iterations", "depth"):
            if _int(values[key], f"{head} {key}") <= 0:
                raise RFContractError("RF expert value differs")
        for key in ("l2_leaf_reg", "random_strength", "bagging_temperature"):
            if _float(values[key], f"{head} {key}") < 0:
                raise RFContractError("RF expert value differs")
        experts[head] = MappingProxyType(values)

    common = _object(
        root["catboost_common"],
        {
            "learning_rate", "bootstrap_type", "border_count", "max_ctr_complexity",
            "one_hot_max_size", "od_type", "od_wait", "task_type", "allow_writing_files",
        },
        "RF CatBoost common",
    )
    gates = _object(
        root["gates"],
        {
            "weighted_gain", "minimum_improved_folds", "maximum_segment_regression",
            "recent_f_gain", "minimum_segment_rows", "probability_tolerance",
        },
        "RF gates",
    )
    runtime = _object(
        root["runtime"],
        {
            "wall_seconds", "new_job_guard_seconds", "full_fit_guard_seconds",
            "snapshot_interval_seconds", "progress_interval_seconds",
            "inference_max_seconds", "rss_max_bytes",
        },
        "RF runtime",
    )
    numeric_runtime = {key: _int(value, key) for key, value in runtime.items()}
    if any(value <= 0 for value in numeric_runtime.values()):
        raise RFContractError("RF runtime differs")
    tolerance = _float(gates["probability_tolerance"], "probability tolerance")
    if numeric_runtime["wall_seconds"] != 19_800 or numeric_runtime["full_fit_guard_seconds"] != 1_800:
        raise RFContractError("RF runtime differs")
    if tolerance != 1e-6:
        raise RFContractError("RF probability tolerance differs")
    return RFContract(
        campaign_id=str(root["campaign_id"]),
        inputs=MappingProxyType({key: str(value) for key, value in inputs.items()}),
        folds=folds,
        structure_seed=3407,
        confirmation_seeds=seeds,
        alpha_values=alphas,
        experts=MappingProxyType(experts),
        catboost_common=MappingProxyType(common),
        gates=RFGates(
            weighted_gain=_float(gates["weighted_gain"], "weighted gain"),
            minimum_improved_folds=_int(gates["minimum_improved_folds"], "minimum improved folds"),
            maximum_segment_regression=_float(gates["maximum_segment_regression"], "segment regression"),
            recent_f_gain=_float(gates["recent_f_gain"], "recent F gain"),
            minimum_segment_rows=_int(gates["minimum_segment_rows"], "minimum segment rows"),
            probability_tolerance=tolerance,
        ),
        wall_seconds=numeric_runtime["wall_seconds"],
        new_job_guard_seconds=numeric_runtime["new_job_guard_seconds"],
        full_fit_guard_seconds=numeric_runtime["full_fit_guard_seconds"],
        snapshot_interval_seconds=numeric_runtime["snapshot_interval_seconds"],
        progress_interval_seconds=numeric_runtime["progress_interval_seconds"],
        inference_max_seconds=numeric_runtime["inference_max_seconds"],
        rss_max_bytes=numeric_runtime["rss_max_bytes"],
        probability_tolerance=tolerance,
    )


def contract_sha256(path: Path = DEFAULT_RF_CONTRACT) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _job(contract: RFContract, head: str, fold: tuple[int, int], seed: int) -> RFJob:
    train_end, valid_year = fold
    return RFJob(
        job_id=f"rf__{head}__tr{train_end}__va{valid_year}__s{seed}",
        head=head,
        segment=str(contract.experts[head]["segment"]),
        train_end_year=train_end,
        valid_year=valid_year,
        seed=seed,
    )


def structure_jobs(contract: RFContract) -> tuple[RFJob, ...]:
    return tuple(
        _job(contract, head, fold, contract.structure_seed)
        for fold in contract.folds
        for head in _HEADS
    )


def confirmation_jobs(contract: RFContract, *, f_head: str, include_r: bool) -> tuple[RFJob, ...]:
    if f_head not in _F_HEADS:
        raise RFContractError("F head differs")
    heads = (f_head, "r_expert") if include_r else (f_head,)
    return tuple(
        _job(contract, head, fold, seed)
        for fold in contract.folds
        for seed in contract.confirmation_seeds
        for head in heads
    )
