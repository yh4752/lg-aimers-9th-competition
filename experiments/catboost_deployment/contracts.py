from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


class DeploymentContractError(ValueError):
    """Raised when the preregistered deployment contract differs."""


DEFAULT_CONTRACT = Path(__file__).with_name("contract.json")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_PREFIXES = (4, 32, 64, 128, 192, 296, 400)
_CATBOOST_PARAMETERS: dict[str, object] = {
    "iterations": 400,
    "depth": 7,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "border_count": 128,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "l2_leaf_reg": 3.0,
    "model_size_reg": 0.5,
    "max_ctr_complexity": 1,
    "random_seed": 42,
    "task_type": "GPU",
}
_TOP_KEYS = {
    "schema_version",
    "campaign_id",
    "source_blend_delivery_sha256",
    "source_stage_c_delivery_sha256",
    "tabm_weight",
    "tree_prefixes",
    "catboost_parameters",
    "gates",
    "budget",
}


@dataclass(frozen=True)
class DeploymentJob:
    job_id: str
    kind: str
    train_end_year: int
    valid_year: int | None
    seed: int


@dataclass(frozen=True)
class DeploymentContract:
    schema_version: int
    campaign_id: str
    source_blend_delivery_sha256: str
    source_stage_c_delivery_sha256: str
    tabm_weight: float
    tree_prefixes: tuple[int, ...]
    minimum_weighted_gain: float
    maximum_fold_regression: float
    catboost_parameters: Mapping[str, object]
    snapshot_interval_seconds: int
    emergency_interval_seconds: int
    session_seconds: int
    new_job_guard_seconds: int


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise DeploymentContractError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DeploymentContractError(f"{label} keys differ")


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise DeploymentContractError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise DeploymentContractError(f"{label} must be an integer")
    return value


def _strict_json_object(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise DeploymentContractError(f"cannot read deployment contract: {error}") from error
    return _object(value, "contract")


def _validate_contract(root: dict[str, object]) -> DeploymentContract:
    _exact_keys(root, _TOP_KEYS, "contract")
    if _integer(root["schema_version"], "schema version") != 1:
        raise DeploymentContractError("schema version differs")
    if root["campaign_id"] != "catboost_deployment_v1":
        raise DeploymentContractError("campaign identity differs")

    blend_sha = _sha(root["source_blend_delivery_sha256"], "blend delivery SHA-256")
    stage_c_sha = _sha(root["source_stage_c_delivery_sha256"], "Stage C delivery SHA-256")
    if blend_sha != "ab7ca41e98e2b94b997369d7c777f8293c64acf61114f314110d17bf743edcfa":
        raise DeploymentContractError("blend delivery identity differs")
    if stage_c_sha != "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a":
        raise DeploymentContractError("Stage C delivery identity differs")

    if type(root["tabm_weight"]) is not float or root["tabm_weight"] != 0.7:
        raise DeploymentContractError("TabM weight differs")
    raw_prefixes = root["tree_prefixes"]
    if type(raw_prefixes) is not list or any(type(value) is not int for value in raw_prefixes):
        raise DeploymentContractError("tree prefixes must be integer list")
    prefixes = tuple(raw_prefixes)
    if prefixes != _PREFIXES or any(right <= left for left, right in zip(prefixes, prefixes[1:])):
        raise DeploymentContractError("tree prefixes differ")

    parameters = _object(root["catboost_parameters"], "CatBoost parameters")
    if parameters != _CATBOOST_PARAMETERS or any(type(value) is bool for value in parameters.values()):
        raise DeploymentContractError("CatBoost parameters differ")

    gates = _object(root["gates"], "gates")
    _exact_keys(gates, {"minimum_weighted_gain", "maximum_fold_regression"}, "gates")
    if gates != {"minimum_weighted_gain": 0.00003, "maximum_fold_regression": 0.00003}:
        raise DeploymentContractError("deployment gates differ")

    budget = _object(root["budget"], "budget")
    _exact_keys(
        budget,
        {
            "snapshot_interval_seconds",
            "emergency_interval_seconds",
            "session_seconds",
            "new_job_guard_seconds",
        },
        "budget",
    )
    expected_budget = {
        "snapshot_interval_seconds": 300,
        "emergency_interval_seconds": 1200,
        "session_seconds": 10800,
        "new_job_guard_seconds": 900,
    }
    if budget != expected_budget or any(type(value) is not int for value in budget.values()):
        raise DeploymentContractError("budget differs")

    return DeploymentContract(
        schema_version=1,
        campaign_id="catboost_deployment_v1",
        source_blend_delivery_sha256=blend_sha,
        source_stage_c_delivery_sha256=stage_c_sha,
        tabm_weight=0.7,
        tree_prefixes=_PREFIXES,
        minimum_weighted_gain=0.00003,
        maximum_fold_regression=0.00003,
        catboost_parameters=MappingProxyType(dict(_CATBOOST_PARAMETERS)),
        snapshot_interval_seconds=300,
        emergency_interval_seconds=1200,
        session_seconds=10800,
        new_job_guard_seconds=900,
    )


def load_contract(path: Path = DEFAULT_CONTRACT) -> DeploymentContract:
    return _validate_contract(_strict_json_object(Path(path).read_bytes()))


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    source = Path(path)
    load_contract(source)
    return sha256(source.read_bytes()).hexdigest()


def build_jobs(contract: DeploymentContract) -> tuple[DeploymentJob, ...]:
    if contract.campaign_id != "catboost_deployment_v1":
        raise DeploymentContractError("campaign identity differs")
    return (
        DeploymentJob("align_2022_2023", "alignment", 2022, 2023, 42),
        DeploymentJob("align_2023_2024", "alignment", 2023, 2024, 42),
        DeploymentJob("full_2024", "full_fit", 2024, None, 42),
    )
