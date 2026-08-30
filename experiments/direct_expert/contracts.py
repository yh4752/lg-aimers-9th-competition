from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class DirectExpertContractError(ValueError):
    pass


@dataclass(frozen=True)
class ExpertSpec:
    expert_id: str
    objective: str
    decay: float | None
    recent_seasons: int | None
    game_type: str | None
    interaction_profile: str


@dataclass(frozen=True)
class ExpertJob:
    job_id: str
    expert_id: str
    fold: tuple[int, int]
    seed: int
    phase: str


@dataclass(frozen=True)
class DirectExpertContract:
    schema_version: int
    campaign_id: str
    folds: tuple[tuple[int, int], ...]
    screening_seed: int
    confirmation_seeds: tuple[int, ...]
    experts: tuple[str, ...]
    screening_parameters: Mapping[str, int | float]
    final_parameters: Mapping[str, int | float]
    maximum_selected_experts: int
    maximum_deployed_models: int
    probability_tolerance: float
    stable_gate: Mapping[str, int | float]
    aggressive_gate: Mapping[str, int | float]
    runtime: Mapping[str, int]
    versions: Mapping[str, str]

    def __post_init__(self) -> None:
        _validate_registered(self)


_ROOT_KEYS = {
    "schema_version",
    "campaign_id",
    "folds",
    "screening_seed",
    "confirmation_seeds",
    "experts",
    "screening_parameters",
    "final_parameters",
    "maximum_selected_experts",
    "maximum_deployed_models",
    "probability_tolerance",
    "stable_gate",
    "aggressive_gate",
    "runtime",
    "versions",
}
_MODEL_KEYS = {"iterations", "depth", "learning_rate", "max_ctr_complexity", "border_count", "od_wait"}
_STABLE_KEYS = {
    "weighted_gain",
    "latest_gain",
    "minimum_fold_gain",
    "maximum_segment_regression",
    "bootstrap_lower",
    "minimum_non_worse_seeds",
}
_AGGRESSIVE_KEYS = {
    "latest_gain",
    "recent_heavy_gain",
    "minimum_fold_gain",
    "maximum_segment_regression",
    "latest_bootstrap_lower",
    "minimum_improving_latest_seeds",
}
_RUNTIME_KEYS = {
    "stage_a_wall_seconds",
    "stage_b_wall_seconds",
    "new_job_guard_seconds",
    "artifact_reserve_seconds",
    "snapshot_interval_seconds",
    "minimum_free_bytes",
}
_VERSION_KEYS = {"catboost", "pandas", "numpy"}


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise DirectExpertContractError(f"{label} keys differ")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise DirectExpertContractError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DirectExpertContractError(f"{label} must be finite")
    return value


def _mapping(value: object, keys: set[str], label: str) -> Mapping[str, int | float]:
    payload = _exact_keys(value, keys, label)
    return MappingProxyType({key: _number(payload[key], f"{label}.{key}") for key in sorted(keys)})


def _validate_registered(contract: DirectExpertContract) -> None:
    if contract.schema_version != 1 or contract.campaign_id != "direct_target_expert_v1":
        raise DirectExpertContractError("contract identity differs")
    if contract.folds != ((2021, 2022), (2022, 2023), (2023, 2024)):
        raise DirectExpertContractError("folds differ")
    if contract.screening_seed != 3407 or contract.confirmation_seeds != (42, 2026):
        raise DirectExpertContractError("seeds differ")
    if contract.experts != tuple(f"D{i}" for i in range(8)):
        raise DirectExpertContractError("experts differ")
    if contract.screening_parameters["depth"] != 9:
        raise DirectExpertContractError("screening depth differs")
    if contract.final_parameters["depth"] != 10:
        raise DirectExpertContractError("final depth differs")
    if contract.screening_parameters["iterations"] != 1800:
        raise DirectExpertContractError("screening iterations differ")
    if contract.final_parameters["iterations"] != 2400:
        raise DirectExpertContractError("final iterations differ")
    if contract.maximum_selected_experts != 4 or contract.maximum_deployed_models != 9:
        raise DirectExpertContractError("deployment budget differs")


def load_contract(path: Path | None = None) -> DirectExpertContract:
    source = path or Path(__file__).with_name("contract.json")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DirectExpertContractError("contract cannot be read") from error
    payload = _exact_keys(raw, _ROOT_KEYS, "contract")
    folds_raw = payload["folds"]
    if type(folds_raw) is not list:
        raise DirectExpertContractError("folds must be a list")
    folds: list[tuple[int, int]] = []
    for index, fold in enumerate(folds_raw):
        if type(fold) is not list or len(fold) != 2:
            raise DirectExpertContractError(f"folds[{index}] differs")
        folds.append((_integer(fold[0], f"folds[{index}][0]"), _integer(fold[1], f"folds[{index}][1]")))
    seeds_raw = payload["confirmation_seeds"]
    experts_raw = payload["experts"]
    if type(seeds_raw) is not list:
        raise DirectExpertContractError("confirmation_seeds must be a list")
    if type(experts_raw) is not list or any(type(item) is not str for item in experts_raw):
        raise DirectExpertContractError("experts must be strings")
    versions_raw = _exact_keys(payload["versions"], _VERSION_KEYS, "versions")
    if any(type(versions_raw[key]) is not str for key in _VERSION_KEYS):
        raise DirectExpertContractError("versions must be strings")
    contract = DirectExpertContract(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        campaign_id=payload["campaign_id"] if type(payload["campaign_id"]) is str else "",
        folds=tuple(folds),
        screening_seed=_integer(payload["screening_seed"], "screening_seed"),
        confirmation_seeds=tuple(_integer(seed, "confirmation_seeds") for seed in seeds_raw),
        experts=tuple(experts_raw),
        screening_parameters=_mapping(payload["screening_parameters"], _MODEL_KEYS, "screening_parameters"),
        final_parameters=_mapping(payload["final_parameters"], _MODEL_KEYS, "final_parameters"),
        maximum_selected_experts=_integer(payload["maximum_selected_experts"], "maximum_selected_experts"),
        maximum_deployed_models=_integer(payload["maximum_deployed_models"], "maximum_deployed_models"),
        probability_tolerance=float(_number(payload["probability_tolerance"], "probability_tolerance")),
        stable_gate=_mapping(payload["stable_gate"], _STABLE_KEYS, "stable_gate"),
        aggressive_gate=_mapping(payload["aggressive_gate"], _AGGRESSIVE_KEYS, "aggressive_gate"),
        runtime=MappingProxyType({
            key: _integer(_exact_keys(payload["runtime"], _RUNTIME_KEYS, "runtime")[key], f"runtime.{key}")
            for key in sorted(_RUNTIME_KEYS)
        }),
        versions=MappingProxyType({key: versions_raw[key] for key in sorted(_VERSION_KEYS)}),
    )
    _validate_registered(contract)
    return contract


def expert_specs(contract: DirectExpertContract | None = None) -> tuple[ExpertSpec, ...]:
    if contract is not None:
        _validate_registered(contract)
    return (
        ExpertSpec("D0", "Logloss", None, None, None, "standard"),
        ExpertSpec("D1", "Logloss", 0.75, None, None, "standard"),
        ExpertSpec("D2", "Logloss", 0.55, None, None, "standard"),
        ExpertSpec("D3", "Logloss", None, 2, None, "standard"),
        ExpertSpec("D4", "RMSE", 0.55, None, None, "standard"),
        ExpertSpec("D5", "Logloss", 0.55, None, "R", "standard"),
        ExpertSpec("D6", "Logloss", 0.55, None, "F", "standard"),
        ExpertSpec("D7", "Logloss", None, None, None, "high_ctr"),
    )


def expert_spec(expert_id: str) -> ExpertSpec:
    for spec in expert_specs():
        if spec.expert_id == expert_id:
            return spec
    raise DirectExpertContractError(f"unknown expert: {expert_id}")


def screening_jobs(contract: DirectExpertContract | None = None) -> tuple[ExpertJob, ...]:
    active = contract or load_contract()
    _validate_registered(active)
    jobs = tuple(
        ExpertJob(
            job_id=f"screen__{spec.expert_id}__{train_year}_{valid_year}__s{active.screening_seed}",
            expert_id=spec.expert_id,
            fold=(train_year, valid_year),
            seed=active.screening_seed,
            phase="screening",
        )
        for train_year, valid_year in active.folds[:2]
        for spec in expert_specs(active)
    )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise DirectExpertContractError("duplicate jobs")
    return jobs
