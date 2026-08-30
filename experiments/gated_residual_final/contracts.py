from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class FinalContractError(ValueError):
    pass


@dataclass(frozen=True)
class FinalContract:
    version: str
    alpha_grid: tuple[float, ...]
    k_grid: tuple[int, ...]
    beta_grid: tuple[float, ...]
    lambda_grid: tuple[int, ...]
    archetypes: tuple[str, ...]
    gates: Mapping[str, float | int]
    expected_hashes: Mapping[str, str]
    maximum_runtime_seconds: int
    full_fit_guard_seconds: int
    full_fit_seeds: tuple[int, ...]
    maximum_deployed_models: int
    depth: int
    maximum_iterations: int


_ROOT_KEYS = {"version", "search", "gates", "expected_hashes", "runtime"}
_SEARCH_KEYS = {"alpha", "k", "beta", "lambda", "archetypes"}
_GATE_KEYS = {
    "weighted_gain", "latest_gain", "minimum_fold_gain", "bootstrap_lower",
    "minimum_non_worse_seeds", "minimum_latest_non_worse_seeds",
    "maximum_segment_regression", "minimum_segment_rows",
}
_HASH_KEYS = {"e2_submission", "stage_a_handoff", "stage_b_handoff", "train_csv", "trackman_history"}
_RUNTIME_KEYS = {
    "maximum_seconds", "full_fit_guard_seconds", "full_fit_seeds",
    "maximum_deployed_models", "depth", "maximum_iterations",
}


def _exact_dict(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise FinalContractError(f"{label} keys differ")
    return value


def _ints(values: object, label: str) -> tuple[int, ...]:
    if type(values) is not list or not values or any(type(item) is not int for item in values):
        raise FinalContractError(f"{label} differs")
    return tuple(values)


def _floats(values: object, label: str) -> tuple[float, ...]:
    if type(values) is not list or not values or any(type(item) not in (int, float) for item in values):
        raise FinalContractError(f"{label} differs")
    return tuple(float(item) for item in values)


def load_contract(path: Path | None = None) -> FinalContract:
    source = Path(path) if path is not None else Path(__file__).with_name("contract.json")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalContractError("contract is unreadable") from error
    root = _exact_dict(payload, _ROOT_KEYS, "contract")
    search = _exact_dict(root["search"], _SEARCH_KEYS, "search")
    gates = _exact_dict(root["gates"], _GATE_KEYS, "gate")
    hashes = _exact_dict(root["expected_hashes"], _HASH_KEYS, "expected hash")
    runtime = _exact_dict(root["runtime"], _RUNTIME_KEYS, "runtime")
    if type(root["version"]) is not str or not root["version"]:
        raise FinalContractError("version differs")
    if type(search["archetypes"]) is not list or tuple(search["archetypes"]) != ("G0", "G1", "G2", "G3"):
        raise FinalContractError("archetypes differ")
    if any(type(value) not in (int, float) for value in gates.values()):
        raise FinalContractError("gate values differ")
    if any(type(value) is not str or len(value) != 64 for value in hashes.values()):
        raise FinalContractError("expected hashes differ")
    integer_runtime = {key: value for key, value in runtime.items() if key != "full_fit_seeds"}
    if any(type(value) is not int or value <= 0 for value in integer_runtime.values()):
        raise FinalContractError("runtime values differ")
    return FinalContract(
        version=root["version"],
        alpha_grid=_floats(search["alpha"], "alpha grid"),
        k_grid=_ints(search["k"], "k grid"),
        beta_grid=_floats(search["beta"], "beta grid"),
        lambda_grid=_ints(search["lambda"], "lambda grid"),
        archetypes=tuple(search["archetypes"]),
        gates=MappingProxyType(dict(gates)),
        expected_hashes=MappingProxyType(dict(hashes)),
        maximum_runtime_seconds=runtime["maximum_seconds"],
        full_fit_guard_seconds=runtime["full_fit_guard_seconds"],
        full_fit_seeds=_ints(runtime["full_fit_seeds"], "full fit seeds"),
        maximum_deployed_models=runtime["maximum_deployed_models"],
        depth=runtime["depth"],
        maximum_iterations=runtime["maximum_iterations"],
    )
