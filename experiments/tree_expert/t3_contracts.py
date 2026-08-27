from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class T3ContractError(ValueError):
    pass


DEFAULT_T3_CONTRACT = Path(__file__).with_name("t3_contract.json")
_FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
_DECAYS = (0.35, 0.55, 0.75)
_RECENT_WEIGHTS = (0.70, 0.80, 0.90)
_CONFIRMATION_SEEDS = (42, 2026)
_TOP_KEYS = {
    "schema_version", "campaign_id", "inputs", "folds", "decays",
    "recent_weights", "structure_seed", "confirmation_seeds", "catboost",
    "gates", "runtime",
}


@dataclass(frozen=True)
class T3Gates:
    weighted_gain: float
    recent_fold_gain: float
    maximum_fold_regression: float
    maximum_segment_regression: float
    minimum_segment_rows: int
    minimum_non_worse_seed_count: int


@dataclass(frozen=True)
class T3Contract:
    campaign_id: str
    inputs: Mapping[str, str]
    folds: tuple[tuple[int, int], ...]
    decays: tuple[float, ...]
    recent_weights: tuple[float, ...]
    structure_seed: int
    confirmation_seeds: tuple[int, ...]
    catboost: Mapping[str, object]
    gates: T3Gates
    wall_seconds: int
    new_job_guard_seconds: int
    full_fit_guard_seconds: int
    snapshot_interval_seconds: int
    inference_max_seconds: int
    rss_max_bytes: int
    probability_tolerance: float


@dataclass(frozen=True)
class T3Job:
    job_id: str
    head: str
    decay: float | None
    train_end_year: int
    valid_year: int
    seed: int


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise T3ContractError(f"{label} keys differ")
    return dict(value)


def _float(value: object, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise T3ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise T3ContractError(f"{label} must be finite")
    return result


def _int(value: object, label: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise T3ContractError(f"{label} must be an integer")
    return value


def load_t3_contract(path: Path = DEFAULT_T3_CONTRACT) -> T3Contract:
    try:
        root = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise T3ContractError("T3 contract cannot be loaded") from error
    root = _object(root, _TOP_KEYS, "T3 contract")
    if root["schema_version"] != 1 or root["campaign_id"] != "tree_expert_t3_temporal_dual_v1":
        raise T3ContractError("T3 contract identity differs")
    inputs = _object(
        root["inputs"],
        {"official_train_sha256", "official_history_sha256", "e2_handoff_sha256"},
        "T3 inputs",
    )
    if any(type(value) is not str or len(value) != 64 for value in inputs.values()):
        raise T3ContractError("T3 input hashes differ")
    folds = tuple(tuple(item) for item in root["folds"])
    decays = tuple(_float(item, "decay") for item in root["decays"])
    recent_weights = tuple(_float(item, "recent weight") for item in root["recent_weights"])
    confirmation = tuple(_int(item, "confirmation seed") for item in root["confirmation_seeds"])
    if folds != _FOLDS or decays != _DECAYS or recent_weights != _RECENT_WEIGHTS:
        raise T3ContractError("T3 search grid differs")
    if confirmation != _CONFIRMATION_SEEDS or root["structure_seed"] != 3407:
        raise T3ContractError("T3 seeds differ")
    catboost = _object(
        root["catboost"],
        {
            "iterations", "depth", "learning_rate", "l2_leaf_reg",
            "random_strength", "bootstrap_type", "bagging_temperature",
            "border_count", "max_ctr_complexity", "one_hot_max_size",
            "od_type", "od_wait", "task_type", "allow_writing_files",
        },
        "T3 CatBoost",
    )
    gates_raw = _object(
        root["gates"],
        {
            "weighted_gain", "recent_fold_gain", "maximum_fold_regression",
            "maximum_segment_regression", "minimum_segment_rows",
            "minimum_non_worse_seed_count",
        },
        "T3 gates",
    )
    runtime = _object(
        root["runtime"],
        {
            "wall_seconds", "new_job_guard_seconds", "full_fit_guard_seconds",
            "snapshot_interval_seconds",
            "inference_max_seconds", "rss_max_bytes", "probability_tolerance",
        },
        "T3 runtime",
    )
    return T3Contract(
        campaign_id=str(root["campaign_id"]),
        inputs=MappingProxyType({key: str(value) for key, value in inputs.items()}),
        folds=folds,
        decays=decays,
        recent_weights=recent_weights,
        structure_seed=_int(root["structure_seed"], "structure seed"),
        confirmation_seeds=confirmation,
        catboost=MappingProxyType(catboost),
        gates=T3Gates(
            weighted_gain=_float(gates_raw["weighted_gain"], "weighted gain"),
            recent_fold_gain=_float(gates_raw["recent_fold_gain"], "recent fold gain"),
            maximum_fold_regression=_float(gates_raw["maximum_fold_regression"], "fold regression"),
            maximum_segment_regression=_float(gates_raw["maximum_segment_regression"], "segment regression"),
            minimum_segment_rows=_int(gates_raw["minimum_segment_rows"], "minimum segment rows"),
            minimum_non_worse_seed_count=_int(gates_raw["minimum_non_worse_seed_count"], "minimum seed count"),
        ),
        wall_seconds=_int(runtime["wall_seconds"], "wall seconds"),
        new_job_guard_seconds=_int(runtime["new_job_guard_seconds"], "job guard"),
        full_fit_guard_seconds=_int(runtime["full_fit_guard_seconds"], "full-fit guard"),
        snapshot_interval_seconds=_int(runtime["snapshot_interval_seconds"], "snapshot interval"),
        inference_max_seconds=_int(runtime["inference_max_seconds"], "inference seconds"),
        rss_max_bytes=_int(runtime["rss_max_bytes"], "RSS bytes"),
        probability_tolerance=_float(runtime["probability_tolerance"], "probability tolerance"),
    )


def contract_sha256(path: Path = DEFAULT_T3_CONTRACT) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _decay_label(decay: float) -> str:
    return f"d{int(round(decay * 100)):03d}"


def structure_jobs(contract: T3Contract) -> tuple[T3Job, ...]:
    jobs: list[T3Job] = []
    for train_end, valid_year in contract.folds:
        jobs.append(T3Job(
            f"t3__recent__tr{train_end}__va{valid_year}__s{contract.structure_seed}",
            "recent", None, train_end, valid_year, contract.structure_seed,
        ))
        for decay in contract.decays:
            label = _decay_label(decay)
            jobs.append(T3Job(
                f"t3__multi_{label}__tr{train_end}__va{valid_year}__s{contract.structure_seed}",
                "multi", decay, train_end, valid_year, contract.structure_seed,
            ))
    return tuple(jobs)


def confirmation_jobs(contract: T3Contract, decay: float) -> tuple[T3Job, ...]:
    if decay not in contract.decays:
        raise T3ContractError("confirmation decay differs")
    jobs: list[T3Job] = []
    for train_end, valid_year in contract.folds:
        for seed in contract.confirmation_seeds:
            jobs.append(T3Job(
                f"t3__recent__tr{train_end}__va{valid_year}__s{seed}",
                "recent", None, train_end, valid_year, seed,
            ))
            label = _decay_label(decay)
            jobs.append(T3Job(
                f"t3__multi_{label}__tr{train_end}__va{valid_year}__s{seed}",
                "multi", decay, train_end, valid_year, seed,
            ))
    return tuple(jobs)
