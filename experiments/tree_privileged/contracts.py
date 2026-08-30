from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class PrivilegedContractError(ValueError):
    pass


DEFAULT_CONTRACT = Path(__file__).with_name("contract.json")
_FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
_CANDIDATES = ("P",)


@dataclass(frozen=True)
class TeacherContract:
    folds: int
    lambdas: tuple[Decimal, ...]
    minimum_total_coverage: Decimal
    minimum_latest_coverage: Decimal


@dataclass(frozen=True)
class ProfileContract:
    identity_strengths: tuple[int, ...]
    interaction_strengths: tuple[int, ...]
    matchup_strengths: tuple[int, ...]
    minimum_rows: Mapping[str, int]


@dataclass(frozen=True)
class CatBoostContract:
    iterations: int
    depth: int
    learning_rate: Decimal
    l2_leaf_reg: Decimal
    random_strength: Decimal
    bagging_temperature: Decimal
    border_count: int
    max_ctr_complexity: int
    od_wait: int


@dataclass(frozen=True)
class GateContract:
    weighted_gain: Decimal
    latest_min_gain: Decimal
    maximum_segment_regression: Decimal
    minimum_improved_folds: int
    minimum_non_worse_seeds: int
    maximum_screen_correlation: Decimal
    ensemble_incremental_gain: Decimal
    probability_tolerance: Decimal


@dataclass(frozen=True)
class RuntimeContract:
    wall_seconds: int
    new_job_guard_seconds: int
    artifact_reserve_seconds: int
    snapshot_interval_seconds: int
    maximum_confirmed_candidates: int
    gpu_count: int


@dataclass(frozen=True)
class PrivilegedContract:
    campaign_id: str
    review_only: bool
    submission_package: bool
    inputs: Mapping[str, str]
    folds: tuple[tuple[int, int], ...]
    candidates: tuple[str, ...]
    screen_seed: int
    confirm_seeds: tuple[int, ...]
    teacher: TeacherContract
    profiles: ProfileContract
    rf_alphas: tuple[Decimal, ...]
    catboost: CatBoostContract
    gates: GateContract
    runtime: RuntimeContract

    @property
    def teacher_lambdas(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.teacher.lambdas)

    @property
    def wall_seconds(self) -> int:
        return self.runtime.wall_seconds

    @property
    def maximum_confirmed_candidates(self) -> int:
        return self.runtime.maximum_confirmed_candidates


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise PrivilegedContractError(f"{label} keys differ")
    return dict(value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise PrivilegedContractError(f"{label} differs")
    return value


def _decimal(value: object, label: str) -> Decimal:
    if type(value) not in {str, int, float} or type(value) is bool:
        raise PrivilegedContractError(f"{label} differs")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise PrivilegedContractError(f"{label} differs") from error
    if not result.is_finite():
        raise PrivilegedContractError(f"{label} differs")
    return result


def _sealed(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise PrivilegedContractError(f"{label} differ")


def load_contract(path: Path = DEFAULT_CONTRACT) -> PrivilegedContract:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrivilegedContractError("contract cannot be loaded") from error
    root = _object(raw, {
        "schema_version", "campaign_id", "review_only", "submission_package", "inputs",
        "folds", "candidates", "screen_seed", "confirm_seeds", "teacher", "profiles",
        "rf_alphas", "catboost", "gates", "runtime",
    }, "contract")
    _sealed((root["schema_version"], root["campaign_id"], root["review_only"], root["submission_package"]),
            (1, "tree_privileged_profile_v1", True, False), "contract identity")

    inputs = _object(root["inputs"], {
        "official_train_sha256", "official_history_sha256", "e2_handoff_sha256",
    }, "inputs")
    expected_inputs = {
        "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
        "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
        "e2_handoff_sha256": "4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f",
    }
    _sealed(inputs, expected_inputs, "input hashes")

    try:
        folds = tuple(tuple(_integer(item, "fold year") for item in pair) for pair in root["folds"])
        candidates = tuple(root["candidates"])
        confirm_seeds = tuple(_integer(item, "confirmation seed") for item in root["confirm_seeds"])
    except TypeError as error:
        raise PrivilegedContractError("campaign grid differs") from error
    _sealed(folds, _FOLDS, "folds")
    _sealed(candidates, _CANDIDATES, "candidates")
    _sealed((_integer(root["screen_seed"], "screen seed"), confirm_seeds), (3407, (42, 2026)), "seeds")

    teacher = _object(root["teacher"], {
        "folds", "lambdas", "minimum_total_coverage", "minimum_latest_coverage",
    }, "teacher")
    teacher_lambdas = tuple(_decimal(value, "teacher lambda") for value in teacher["lambdas"])
    _sealed((_integer(teacher["folds"], "teacher folds"), teacher_lambdas,
             _decimal(teacher["minimum_total_coverage"], "total coverage"),
             _decimal(teacher["minimum_latest_coverage"], "latest coverage")),
            (5, (Decimal("0.15"), Decimal("0.35")), Decimal("0.30"), Decimal("0.20")),
            "teacher grid")

    profiles = _object(root["profiles"], {
        "identity_strengths", "interaction_strengths", "matchup_strengths", "minimum_rows",
    }, "profiles")
    minimum_rows = _object(profiles["minimum_rows"], {"identity", "interaction", "matchup"}, "minimum rows")
    profile_tuple = (
        tuple(_integer(value, "identity strength") for value in profiles["identity_strengths"]),
        tuple(_integer(value, "interaction strength") for value in profiles["interaction_strengths"]),
        tuple(_integer(value, "matchup strength") for value in profiles["matchup_strengths"]),
        {key: _integer(value, f"minimum {key}") for key, value in minimum_rows.items()},
    )
    _sealed(profile_tuple, ((25, 75, 200), (50, 150, 400), (100, 300, 800),
                            {"identity": 20, "interaction": 50, "matchup": 100}), "profile grid")

    rf_alphas = tuple(_decimal(value, "R/F alpha") for value in root["rf_alphas"])
    _sealed(rf_alphas, tuple(map(Decimal, ("0.35", "0.60", "0.80", "1.00"))), "R/F alphas")

    catboost = _object(root["catboost"], {
        "iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength",
        "bagging_temperature", "border_count", "max_ctr_complexity", "od_wait",
    }, "catboost")
    catboost_values = (
        _integer(catboost["iterations"], "iterations"), _integer(catboost["depth"], "depth"),
        _decimal(catboost["learning_rate"], "learning rate"), _decimal(catboost["l2_leaf_reg"], "l2"),
        _decimal(catboost["random_strength"], "random strength"),
        _decimal(catboost["bagging_temperature"], "bagging temperature"),
        _integer(catboost["border_count"], "border count"),
        _integer(catboost["max_ctr_complexity"], "CTR complexity"), _integer(catboost["od_wait"], "od wait"),
    )
    _sealed(catboost_values, (1200, 8, Decimal("0.04"), Decimal("5.0"), Decimal("0.5"),
                              Decimal("0.5"), 128, 2, 60), "catboost grid")

    gates = _object(root["gates"], {
        "weighted_gain", "latest_min_gain", "maximum_segment_regression", "minimum_improved_folds",
        "minimum_non_worse_seeds", "maximum_screen_correlation", "ensemble_incremental_gain",
        "probability_tolerance",
    }, "gates")
    gate_values = (
        _decimal(gates["weighted_gain"], "weighted gain"),
        _decimal(gates["latest_min_gain"], "latest gain"),
        _decimal(gates["maximum_segment_regression"], "segment regression"),
        _integer(gates["minimum_improved_folds"], "improved folds"),
        _integer(gates["minimum_non_worse_seeds"], "non-worse seeds"),
        _decimal(gates["maximum_screen_correlation"], "screen correlation"),
        _decimal(gates["ensemble_incremental_gain"], "ensemble gain"),
        _decimal(gates["probability_tolerance"], "probability tolerance"),
    )
    expected_gates = (Decimal("0.00005"), Decimal("-0.00002"), Decimal("0.00050"), 2, 2,
                      Decimal("0.998"), Decimal("0.00002"), Decimal("0.000001"))
    _sealed(gate_values, expected_gates, "gates")

    runtime = _object(root["runtime"], {
        "wall_seconds", "new_job_guard_seconds", "artifact_reserve_seconds",
        "snapshot_interval_seconds", "maximum_confirmed_candidates", "gpu_count",
    }, "runtime")
    runtime_values = tuple(_integer(runtime[key], key) for key in (
        "wall_seconds", "new_job_guard_seconds", "artifact_reserve_seconds",
        "snapshot_interval_seconds", "maximum_confirmed_candidates", "gpu_count",
    ))
    _sealed(runtime_values, (37800, 7200, 4500, 600, 2, 2), "runtime")

    return PrivilegedContract(
        campaign_id=str(root["campaign_id"]), review_only=True, submission_package=False,
        inputs=MappingProxyType({str(key): str(value) for key, value in inputs.items()}),
        folds=folds, candidates=candidates, screen_seed=3407, confirm_seeds=confirm_seeds,
        teacher=TeacherContract(5, teacher_lambdas, Decimal("0.30"), Decimal("0.20")),
        profiles=ProfileContract(profile_tuple[0], profile_tuple[1], profile_tuple[2],
                                 MappingProxyType(profile_tuple[3])),
        rf_alphas=rf_alphas,
        catboost=CatBoostContract(*catboost_values), gates=GateContract(*gate_values),
        runtime=RuntimeContract(*runtime_values),
    )


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()
