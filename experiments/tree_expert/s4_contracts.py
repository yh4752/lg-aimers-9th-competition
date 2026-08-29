from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class S4ContractError(ValueError):
    pass


DEFAULT_S4_CONTRACT = Path(__file__).with_name("s4_contract.json")
_FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
_WEIGHTS = (0.60, 0.75, 0.90)
_DECAYS = (0.30, 0.55, 0.75)
_FAMILIES = ("catboost", "catboost_rf", "xgboost", "lightgbm", "dual_temporal")
_PROFILES = ("global_game", "pitcher", "batter", "matchup", "rf_matchup")


@dataclass(frozen=True)
class S4Gates:
    weighted_gain: float
    recent_gain: float
    maximum_fold_regression: float
    maximum_segment_regression: float
    minimum_segment_rows: int
    bootstrap_repeats: int
    bootstrap_seed: int
    minimum_non_worse_seed_count: int
    probability_tolerance: float
    catastrophic_structure_regression: float


@dataclass(frozen=True)
class S4Runtime:
    wall_seconds: int
    new_job_guard_seconds: int
    artifact_reserve_seconds: int
    snapshot_interval_seconds: int
    rss_max_bytes: int


@dataclass(frozen=True)
class S4Contract:
    campaign_id: str
    inputs: Mapping[str, str]
    folds: tuple[tuple[int, int], ...]
    recent_weights: tuple[float, ...]
    decays: tuple[float, ...]
    residual_families: tuple[str, ...]
    residual_alphas: tuple[float, ...]
    calibration_profiles: tuple[str, ...]
    calibration_betas: tuple[float, ...]
    structure_seed: int
    confirmation_seeds: tuple[int, ...]
    minimum_full_chains: int
    maximum_anchor_coverage: int
    maximum_confirmation_candidates: int
    versions: Mapping[str, str]
    parameters: Mapping[str, Mapping[str, object]]
    gates: S4Gates
    runtime: S4Runtime


@dataclass(frozen=True)
class AnchorSpec:
    candidate_id: str
    recent_weight: float | None
    decay: float | None
    route_by_game_type: bool
    mandatory_role: str | None


@dataclass(frozen=True)
class ResidualSpec:
    family: str
    route_by_game_type: bool


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise S4ContractError(f"{label} keys differ")
    return dict(value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise S4ContractError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise S4ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise S4ContractError(f"{label} must be finite")
    return result


def _numbers(value: object, expected: tuple[float, ...], label: str) -> tuple[float, ...]:
    if type(value) is not list:
        raise S4ContractError(f"{label} must be a list")
    result = tuple(_number(item, label) for item in value)
    if result != expected:
        raise S4ContractError(f"{label} differs")
    return result


def load_s4_contract(path: Path = DEFAULT_S4_CONTRACT) -> S4Contract:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S4ContractError("S4 contract cannot be loaded") from error
    root = _object(raw, {
        "schema_version", "campaign_id", "inputs", "folds", "recent_weights", "decays",
        "residual_families", "residual_alphas", "calibration_profiles", "calibration_betas",
        "structure_seed", "confirmation_seeds", "minimum_full_chains",
        "maximum_anchor_coverage", "maximum_confirmation_candidates", "versions", "models",
        "gates", "runtime",
    }, "S4 contract")
    if root["schema_version"] != 1 or root["campaign_id"] != "tree_s4_full_chain_v1":
        raise S4ContractError("S4 contract identity differs")
    inputs = _object(root["inputs"], {
        "official_train_sha256", "official_history_sha256", "e2_handoff_sha256",
    }, "S4 inputs")
    if any(type(value) is not str or len(value) != 64 for value in inputs.values()):
        raise S4ContractError("S4 input hashes differ")
    folds = tuple(tuple(_integer(item, "fold year") for item in pair) for pair in root["folds"])
    if folds != _FOLDS:
        raise S4ContractError("folds differ")
    families = tuple(root["residual_families"])
    profiles = tuple(root["calibration_profiles"])
    if families != _FAMILIES or profiles != _PROFILES:
        raise S4ContractError("registered families differ")
    alphas = _numbers(root["residual_alphas"], (0.25, 0.50, 0.75, 1.00), "residual alphas")
    betas = _numbers(root["calibration_betas"], (0.10, 0.25, 0.50, 0.75), "calibration betas")
    confirmation = tuple(_integer(item, "confirmation seed") for item in root["confirmation_seeds"])
    if confirmation != (42, 2026) or root["structure_seed"] != 3407:
        raise S4ContractError("seed registry differs")
    minimum_full_chains = _integer(root["minimum_full_chains"], "minimum_full_chains")
    if minimum_full_chains < 12:
        raise S4ContractError("minimum_full_chains must be at least twelve")
    versions = _object(root["versions"], {"catboost", "xgboost", "lightgbm"}, "versions")
    if versions != {"catboost": "1.2.10", "xgboost": "3.0.2", "lightgbm": "4.6.0"}:
        raise S4ContractError("dependency versions differ")
    models = _object(root["models"], {"catboost", "xgboost", "lightgbm"}, "models")
    gates = _object(root["gates"], {
        "weighted_gain", "recent_gain", "maximum_fold_regression", "maximum_segment_regression",
        "minimum_segment_rows", "bootstrap_repeats", "bootstrap_seed",
        "minimum_non_worse_seed_count", "probability_tolerance",
        "catastrophic_structure_regression",
    }, "gates")
    runtime = _object(root["runtime"], {
        "wall_seconds", "new_job_guard_seconds", "artifact_reserve_seconds",
        "snapshot_interval_seconds", "rss_max_bytes",
    }, "runtime")
    return S4Contract(
        campaign_id=str(root["campaign_id"]),
        inputs=MappingProxyType({str(key): str(value) for key, value in inputs.items()}),
        folds=folds,
        recent_weights=_numbers(root["recent_weights"], _WEIGHTS, "recent weights"),
        decays=_numbers(root["decays"], _DECAYS, "decays"),
        residual_families=families,
        residual_alphas=alphas,
        calibration_profiles=profiles,
        calibration_betas=betas,
        structure_seed=_integer(root["structure_seed"], "structure seed"),
        confirmation_seeds=confirmation,
        minimum_full_chains=minimum_full_chains,
        maximum_anchor_coverage=_integer(root["maximum_anchor_coverage"], "anchor coverage"),
        maximum_confirmation_candidates=_integer(root["maximum_confirmation_candidates"], "confirmation candidates"),
        versions=MappingProxyType({str(key): str(value) for key, value in versions.items()}),
        parameters=MappingProxyType({
            str(key): MappingProxyType(dict(value))
            for key, value in models.items() if type(value) is dict
        }),
        gates=S4Gates(
            weighted_gain=_number(gates["weighted_gain"], "weighted gain"),
            recent_gain=_number(gates["recent_gain"], "recent gain"),
            maximum_fold_regression=_number(gates["maximum_fold_regression"], "fold regression"),
            maximum_segment_regression=_number(gates["maximum_segment_regression"], "segment regression"),
            minimum_segment_rows=_integer(gates["minimum_segment_rows"], "minimum segment rows"),
            bootstrap_repeats=_integer(gates["bootstrap_repeats"], "bootstrap repeats"),
            bootstrap_seed=_integer(gates["bootstrap_seed"], "bootstrap seed"),
            minimum_non_worse_seed_count=_integer(gates["minimum_non_worse_seed_count"], "minimum seed count"),
            probability_tolerance=_number(gates["probability_tolerance"], "probability tolerance"),
            catastrophic_structure_regression=_number(gates["catastrophic_structure_regression"], "catastrophic regression"),
        ),
        runtime=S4Runtime(
            wall_seconds=_integer(runtime["wall_seconds"], "wall seconds"),
            new_job_guard_seconds=_integer(runtime["new_job_guard_seconds"], "job guard"),
            artifact_reserve_seconds=_integer(runtime["artifact_reserve_seconds"], "artifact reserve"),
            snapshot_interval_seconds=_integer(runtime["snapshot_interval_seconds"], "snapshot interval"),
            rss_max_bytes=_integer(runtime["rss_max_bytes"], "RSS bytes"),
        ),
    )


def contract_sha256(path: Path = DEFAULT_S4_CONTRACT) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def anchor_specs(contract: S4Contract) -> tuple[AnchorSpec, ...]:
    if contract.minimum_full_chains < 12:
        raise S4ContractError("minimum_full_chains must be at least twelve")
    output = [AnchorSpec("s4__anchor__e2", None, None, False, "e2_control")]
    for weight in contract.recent_weights:
        for decay in contract.decays:
            role = "external_template" if (weight, decay) == (0.75, 0.55) else None
            output.append(AnchorSpec(
                f"s4__anchor__w{int(weight * 100):02d}__d{int(decay * 100):02d}",
                weight, decay, False, role,
            ))
    if len(output) != 10 or len({item.candidate_id for item in output}) != 10:
        raise S4ContractError("anchor registry differs")
    return tuple(output)


def residual_specs(contract: S4Contract) -> tuple[ResidualSpec, ...]:
    output = tuple(ResidualSpec(family, family == "catboost_rf") for family in contract.residual_families)
    if len({item.family for item in output}) != len(output):
        raise S4ContractError("residual registry differs")
    return output
