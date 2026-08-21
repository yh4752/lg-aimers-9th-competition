from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class RealignContractError(ValueError):
    """Raised when the preregistered realignment contract differs."""


DEFAULT_CONTRACT = Path(__file__).with_name("contract.json")
_SEALED_CONTRACT_SHA256 = (
    "3b58eb4a142b4a54aeb7e0106fca42123212c59ec2009136ae5ee9688a2f9548"
)


@dataclass(frozen=True)
class RealignFold:
    train_end_year: int
    valid_year: int


@dataclass(frozen=True)
class RealignJob:
    job_id: str
    kind: str
    train_end_year: int
    valid_year: int | None
    seed: int


@dataclass(frozen=True)
class RealignContract:
    schema_version: int
    campaign_id: str
    review_only: bool
    source_sha256: Mapping[str, str]
    folds: tuple[RealignFold, ...]
    tabm_weight: Decimal
    tree_prefixes: tuple[int, ...]
    tabm: Mapping[str, object]
    catboost_parameters: Mapping[str, object]
    minimum_weighted_gain: Decimal
    latest_bootstrap_lower_minimum: Decimal
    maximum_segment_regression: Decimal
    selection_tolerance: Decimal
    bootstrap_repeats: int
    bootstrap_seed: int
    session_seconds: int
    new_job_guard_seconds: int
    snapshot_interval_seconds: int


def _object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RealignContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise RealignContractError(f"non-finite JSON number: {value}")


def _load_json(path: Path) -> dict[str, object]:
    source = Path(path)
    try:
        payload = source.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except RealignContractError:
        raise
    except Exception as error:
        raise RealignContractError(f"cannot read realignment contract: {error}") from error
    if type(value) is not dict:
        raise RealignContractError("contract must be an object")
    if sha256(payload).hexdigest() != _SEALED_CONTRACT_SHA256:
        raise RealignContractError("contract bytes differ")
    return value


def load_contract(path: Path = DEFAULT_CONTRACT) -> RealignContract:
    root = _load_json(path)
    folds = tuple(RealignFold(int(left), int(right)) for left, right in root["folds"])
    gates = root["gates"]
    bootstrap = root["bootstrap"]
    budget = root["budget"]
    return RealignContract(
        schema_version=int(root["schema_version"]),
        campaign_id=str(root["campaign_id"]),
        review_only=root["review_only"] is True,
        source_sha256=MappingProxyType(dict(root["source_sha256"])),
        folds=folds,
        tabm_weight=Decimal(root["tabm_weight"]),
        tree_prefixes=tuple(root["tree_prefixes"]),
        tabm=MappingProxyType(dict(root["tabm"])),
        catboost_parameters=MappingProxyType(dict(root["catboost"])),
        minimum_weighted_gain=Decimal(gates["minimum_weighted_gain"]),
        latest_bootstrap_lower_minimum=Decimal(gates["latest_bootstrap_lower_minimum"]),
        maximum_segment_regression=Decimal(gates["maximum_segment_regression"]),
        selection_tolerance=Decimal(root["selection_tolerance"]),
        bootstrap_repeats=int(bootstrap["repeats"]),
        bootstrap_seed=int(bootstrap["seed"]),
        session_seconds=int(budget["session_seconds"]),
        new_job_guard_seconds=int(budget["new_job_guard_seconds"]),
        snapshot_interval_seconds=int(budget["snapshot_interval_seconds"]),
    )


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    source = Path(path)
    load_contract(source)
    return sha256(source.read_bytes()).hexdigest()


def build_jobs(contract: RealignContract) -> tuple[RealignJob, ...]:
    if contract.campaign_id != "catboost_50_50_realign_v2":
        raise RealignContractError("campaign identity differs")
    return (
        RealignJob("tabm_f1_2022", "tabm_alignment", 2021, 2022, 3407),
        RealignJob("catboost_f1_2022", "catboost_alignment", 2021, 2022, 42),
        RealignJob("catboost_full_2024", "full_fit", 2024, None, 42),
    )
