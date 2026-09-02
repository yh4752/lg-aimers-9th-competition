from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class E3ContractError(ValueError):
    pass


@dataclass(frozen=True)
class RoleSpec:
    role_id: str
    target: str
    decay: float | None
    recent_seasons: int | None
    game_type: str | None
    profile: str


@dataclass(frozen=True)
class E3Contract:
    schema_version: int
    campaign_id: str
    policy_version: str
    folds: tuple[tuple[int, int], ...]
    selection_years: tuple[int, ...]
    confirmation_year: int
    seeds: tuple[int, ...]
    roles: tuple[RoleSpec, ...]
    success_parameters: Mapping[str, int | float]
    subtype_parameters: Mapping[str, int | float]
    gate_parameters: Mapping[str, int | float]
    label_gate: Mapping[str, int | float]
    e2_weights: tuple[float, ...]
    gate_strengths: tuple[float, ...]
    gates: Mapping[str, int | float]
    runtime: Mapping[str, int]
    versions: Mapping[str, str]
    data_policy: Mapping[str, object]

    @property
    def inference_max_seconds(self) -> int:
        return self.runtime["inference_max_seconds"]


_ROOT = {
    "schema_version", "campaign_id", "policy_version", "data", "folds",
    "selection_years", "confirmation_year", "seeds", "roles", "model",
    "label_gate", "blend", "gates", "runtime", "versions",
}
_DATA = {"allowed_sources", "training_rows_only", "evaluation_scope", "pre_pitch_only", "external_api"}
_ROLE = {"role_id", "target", "decay", "recent_seasons", "game_type", "profile"}
_MODEL = {"success", "subtype", "gate"}
_TREE_MODEL = {
    "iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength",
    "max_ctr_complexity", "border_count", "od_wait",
}
_GATE_MODEL = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength"}
_LABEL_GATE = {
    "delta_tolerance", "minimum_coverage", "minimum_binary_fraction",
    "minimum_success_agreement", "minimum_positive_rows",
}
_BLEND = {"e2_weights", "gate_strengths"}
_GATES = {
    "weighted_gain", "latest_gain", "minimum_fold_gain", "bootstrap_lower",
    "maximum_segment_regression", "minimum_segment_rows", "minimum_non_worse_seeds",
}
_RUNTIME = {
    "wall_seconds", "new_job_guard_seconds", "artifact_reserve_seconds",
    "snapshot_interval_seconds", "minimum_free_bytes", "maximum_deployed_models",
    "inference_max_seconds",
}
_VERSIONS = {"catboost", "pandas", "numpy"}


def _mapping(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise E3ContractError(f"{label} keys differ")
    return value


def _numbers(value: object, keys: set[str], label: str) -> Mapping[str, int | float]:
    source = _mapping(value, keys, label)
    if any(type(item) not in {int, float} or not math.isfinite(float(item)) for item in source.values()):
        raise E3ContractError(f"{label} values differ")
    return MappingProxyType(dict(source))


def _integer_list(value: object, label: str) -> tuple[int, ...]:
    if type(value) is not list or not value or any(type(item) is not int for item in value):
        raise E3ContractError(f"{label} differs")
    return tuple(value)


def _float_list(value: object, label: str) -> tuple[float, ...]:
    if type(value) is not list or not value or any(type(item) not in {int, float} for item in value):
        raise E3ContractError(f"{label} differs")
    return tuple(float(item) for item in value)


def _validate_registered(contract: E3Contract) -> None:
    expected_roles = (
        ("S_GLOBAL", "success", 0.55, None),
        ("S_FAST", "success", 0.30, None),
        ("S_R", "success", 0.55, "R"),
        ("S_F", "success", 0.55, "F"),
        ("MIDDLE", "middle", 0.55, None),
        ("WILD", "wild", 0.55, None),
        ("REVERSE", "reverse", 0.55, None),
    )
    observed_roles = tuple((role.role_id, role.target, role.decay, role.game_type) for role in contract.roles)
    if (
        contract.schema_version != 1
        or contract.campaign_id != "failure_regime_e3_v1"
        or contract.policy_version != "dacon-236743-2026-08-15"
        or contract.folds != ((2021, 2022), (2022, 2023), (2023, 2024))
        or contract.selection_years != (2022, 2023)
        or contract.confirmation_year != 2024
        or contract.seeds != (42, 2026, 3407)
        or observed_roles != expected_roles
        or contract.success_parameters["iterations"] != 2400
        or contract.success_parameters["depth"] != 10
        or contract.subtype_parameters["iterations"] != 1800
        or contract.subtype_parameters["depth"] != 9
        or contract.gate_parameters["depth"] != 3
        or contract.e2_weights != (0.5, 0.6, 0.7, 0.8)
        or contract.gate_strengths != (0.5, 0.75, 1.0)
        or contract.runtime["maximum_deployed_models"] != 24
        or contract.runtime["inference_max_seconds"] != 480
        or contract.runtime["wall_seconds"] != 36000
    ):
        raise E3ContractError("registered contract differs")
    policy = contract.data_policy
    if (
        policy["allowed_sources"] != ("official_train", "official_trackman")
        or policy["training_rows_only"] is not True
        or policy["evaluation_scope"] != "current_row_only"
        or policy["pre_pitch_only"] is not True
        or policy["external_api"] is not False
    ):
        raise E3ContractError("registered data policy differs")


def load_contract(path: Path | None = None) -> E3Contract:
    source = Path(path) if path is not None else Path(__file__).with_name("contract.json")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E3ContractError("contract is unreadable") from error
    root = _mapping(payload, _ROOT, "contract")
    data = _mapping(root["data"], _DATA, "data")
    allowed = data["allowed_sources"]
    if type(allowed) is not list or any(type(item) is not str for item in allowed):
        raise E3ContractError("allowed sources differ")
    data_policy = MappingProxyType({**data, "allowed_sources": tuple(allowed)})
    folds_raw = root["folds"]
    if type(folds_raw) is not list:
        raise E3ContractError("folds differ")
    folds: list[tuple[int, int]] = []
    for fold in folds_raw:
        if type(fold) is not list or len(fold) != 2 or any(type(item) is not int for item in fold):
            raise E3ContractError("folds differ")
        folds.append((fold[0], fold[1]))
    roles_raw = root["roles"]
    if type(roles_raw) is not list or not roles_raw:
        raise E3ContractError("roles differ")
    roles: list[RoleSpec] = []
    for raw in roles_raw:
        role = _mapping(raw, _ROLE, "role")
        if (
            type(role["role_id"]) is not str
            or type(role["target"]) is not str
            or role["profile"] not in {"standard", "high_ctr"}
            or role["game_type"] not in {None, "R", "F"}
            or role["recent_seasons"] is not None
            or type(role["decay"]) not in {int, float}
        ):
            raise E3ContractError("role values differ")
        roles.append(RoleSpec(
            role_id=role["role_id"], target=role["target"], decay=float(role["decay"]),
            recent_seasons=None, game_type=role["game_type"], profile=role["profile"],
        ))
    model = _mapping(root["model"], _MODEL, "model")
    blend = _mapping(root["blend"], _BLEND, "blend")
    versions = _mapping(root["versions"], _VERSIONS, "versions")
    if any(type(value) is not str or not value for value in versions.values()):
        raise E3ContractError("versions differ")
    runtime_raw = _mapping(root["runtime"], _RUNTIME, "runtime")
    if any(type(value) is not int or value <= 0 for value in runtime_raw.values()):
        raise E3ContractError("runtime differs")
    contract = E3Contract(
        schema_version=root["schema_version"] if type(root["schema_version"]) is int else -1,
        campaign_id=root["campaign_id"] if type(root["campaign_id"]) is str else "",
        policy_version=root["policy_version"] if type(root["policy_version"]) is str else "",
        folds=tuple(folds),
        selection_years=_integer_list(root["selection_years"], "selection years"),
        confirmation_year=root["confirmation_year"] if type(root["confirmation_year"]) is int else -1,
        seeds=_integer_list(root["seeds"], "seeds"),
        roles=tuple(roles),
        success_parameters=_numbers(model["success"], _TREE_MODEL, "success model"),
        subtype_parameters=_numbers(model["subtype"], _TREE_MODEL, "subtype model"),
        gate_parameters=_numbers(model["gate"], _GATE_MODEL, "gate model"),
        label_gate=_numbers(root["label_gate"], _LABEL_GATE, "label gate"),
        e2_weights=_float_list(blend["e2_weights"], "E2 weights"),
        gate_strengths=_float_list(blend["gate_strengths"], "gate strengths"),
        gates=_numbers(root["gates"], _GATES, "gates"),
        runtime=MappingProxyType(dict(runtime_raw)),
        versions=MappingProxyType(dict(versions)),
        data_policy=data_policy,
    )
    _validate_registered(contract)
    return contract


def role_spec(role_id: str, contract: E3Contract | None = None) -> RoleSpec:
    active = contract or load_contract()
    for role in active.roles:
        if role.role_id == role_id:
            return role
    raise E3ContractError(f"unknown role: {role_id}")
