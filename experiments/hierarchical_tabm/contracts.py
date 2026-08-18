from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class HierarchicalContractError(ValueError):
    """Raised when the preregistered campaign contract differs."""


@dataclass(frozen=True)
class Fold:
    train_end_year: int
    valid_year: int


@dataclass(frozen=True)
class ModelPolicy:
    k: int
    width: int
    blocks: int
    dropout: float
    num_embedding: str
    loss: str
    scheduler: str
    learning_rate: float
    weight_decay: float
    effective_batch_size: int
    micro_batch_size: int
    seed: int


@dataclass(frozen=True)
class HierarchicalJob:
    job_id: str
    kind: str
    train_end_year: int
    valid_year: int | None


@dataclass(frozen=True)
class HierarchicalContract:
    schema_version: int
    campaign_id: str
    review_only: bool
    submission_package: bool
    official_train_sha256: str
    official_history_sha256: str
    source_stage_c_delivery_sha256: str
    development_fold: Fold
    oof_folds: tuple[Fold, ...]
    candidate_ids: tuple[str, ...]
    k_candidates: tuple[float, ...]
    k_tie_tolerance: float
    pitcher_reliability_k: float
    batter_reliability_k: float
    hierarchy_columns: tuple[str, ...]
    model: ModelPolicy
    max_epochs: int
    min_epochs: int
    patience: int
    final_min_epochs: int
    final_max_epochs: int
    calibration_grid: tuple[float, ...]
    probability_clip: float
    segment_min_rows: int
    gates: Mapping[str, Mapping[str, float]]
    inference_limits: Mapping[str, int | str]
    session_seconds: int
    new_job_guard_seconds: int
    snapshot_interval_seconds: int
    download_interval_seconds: int


DEFAULT_CONTRACT = Path(__file__).with_name("contract.json")

_EXPECTED: dict[str, object] = {
    "schema_version": 1,
    "campaign_id": "hierarchical_tabm_score_push_v1",
    "review_only": True,
    "submission_package": False,
    "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
    "source_stage_c_delivery_sha256": "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a",
    "development_fold": {"train_end_year": 2021, "valid_year": 2022},
    "oof_folds": [
        {"train_end_year": 2022, "valid_year": 2023},
        {"train_end_year": 2023, "valid_year": 2024},
    ],
    "candidate_ids": ["H1", "H2", "H3"],
    "k_candidates": [32.0, 128.0, 512.0],
    "k_tie_tolerance": 0.000001,
    "reliability": {"pitcher_k": 100.0, "batter_k": 250.0},
    "hierarchy_columns": [
        "hier_context_rate",
        "hier_context_logit",
        "hier_pitcher_reliability",
        "hier_batter_reliability",
        "hier_pitcher_context_gap",
        "hier_batter_context_gap",
        "hier_pitcher_weighted_gap",
        "hier_batter_weighted_gap",
    ],
    "model": {
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.1,
        "num_embedding": "piecewise_linear",
        "loss": "bce",
        "scheduler": "plateau",
        "learning_rate": 0.0006,
        "weight_decay": 0.0001,
        "effective_batch_size": 4096,
        "micro_batch_size": 512,
        "seed": 3407,
    },
    "training": {
        "max_epochs": 40,
        "min_epochs": 3,
        "patience": 10,
        "final_min_epochs": 2,
        "final_max_epochs": 8,
    },
    "calibration": {
        "grid": [0.0001, 0.001, 0.01, 0.1],
        "probability_clip": 0.000001,
    },
    "segments": {"minimum_rows": 5000},
    "gates": {
        "H1_strong": {
            "minimum_weighted_gain": 0.0001,
            "minimum_latest_gain": 0.00005,
            "maximum_old_fold_regression": 0.00015,
            "maximum_segment_regression": 0.00075,
        },
        "H1_frontier": {"maximum_old_fold_regression": 0.00025},
        "H2": {
            "minimum_latest_gain_vs_h1": 0.00003,
            "minimum_latest_gain_vs_anchor": 0.00008,
            "maximum_segment_regression_vs_h1": 0.0002,
        },
        "H3": {
            "minimum_latest_gain_vs_h1": 0.00003,
            "minimum_latest_gain_vs_anchor": 0.00008,
            "maximum_segment_regression_vs_h1": 0.0002,
            "minimum_latest_gain_vs_h2": 0.00002,
        },
    },
    "inference_limits": {
        "python_version": "3.11.15",
        "inference_seconds": 480,
        "gpu_bytes": 21474836480,
        "rss_bytes": 22000000000,
        "artifact_bytes": 2000000000,
    },
    "budget": {
        "session_seconds": 10800,
        "new_job_guard_seconds": 900,
        "snapshot_interval_seconds": 300,
        "download_interval_seconds": 1200,
    },
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HierarchicalContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise HierarchicalContractError(f"non-finite JSON number: {value}")


def _same_exact(observed: object, expected: object, path: str = "contract") -> None:
    if type(observed) is not type(expected):
        raise HierarchicalContractError(f"{path} type differs")
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        if tuple(observed) != tuple(expected):
            raise HierarchicalContractError(f"{path} keys or order differ")
        for key in expected:
            _same_exact(observed[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(observed, list)
        if len(observed) != len(expected):
            raise HierarchicalContractError(f"{path} length differs")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _same_exact(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        assert isinstance(observed, float)
        if not math.isfinite(observed) or observed != expected:
            raise HierarchicalContractError(f"{path} differs")
        return
    if observed != expected:
        raise HierarchicalContractError(f"{path} differs")


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except HierarchicalContractError:
        raise
    except Exception as error:
        raise HierarchicalContractError(f"cannot read contract: {error}") from error
    if type(value) is not dict:
        raise HierarchicalContractError("contract must be an object")
    _same_exact(value, _EXPECTED)
    return value


def _fold(value: Mapping[str, object]) -> Fold:
    return Fold(int(value["train_end_year"]), int(value["valid_year"]))


def _freeze_nested(
    value: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    return MappingProxyType(
        {key: MappingProxyType(dict(nested)) for key, nested in value.items()}
    )


def load_contract(path: Path = DEFAULT_CONTRACT) -> HierarchicalContract:
    raw = _read(Path(path))
    model = raw["model"]
    training = raw["training"]
    calibration = raw["calibration"]
    reliability = raw["reliability"]
    segments = raw["segments"]
    budget = raw["budget"]
    assert isinstance(model, dict)
    assert isinstance(training, dict)
    assert isinstance(calibration, dict)
    assert isinstance(reliability, dict)
    assert isinstance(segments, dict)
    assert isinstance(budget, dict)
    return HierarchicalContract(
        schema_version=1,
        campaign_id="hierarchical_tabm_score_push_v1",
        review_only=True,
        submission_package=False,
        official_train_sha256=str(raw["official_train_sha256"]),
        official_history_sha256=str(raw["official_history_sha256"]),
        source_stage_c_delivery_sha256=str(raw["source_stage_c_delivery_sha256"]),
        development_fold=_fold(raw["development_fold"]),  # type: ignore[arg-type]
        oof_folds=tuple(_fold(value) for value in raw["oof_folds"]),  # type: ignore[arg-type]
        candidate_ids=tuple(raw["candidate_ids"]),  # type: ignore[arg-type]
        k_candidates=tuple(raw["k_candidates"]),  # type: ignore[arg-type]
        k_tie_tolerance=float(raw["k_tie_tolerance"]),
        pitcher_reliability_k=float(reliability["pitcher_k"]),
        batter_reliability_k=float(reliability["batter_k"]),
        hierarchy_columns=tuple(raw["hierarchy_columns"]),  # type: ignore[arg-type]
        model=ModelPolicy(**model),  # type: ignore[arg-type]
        max_epochs=int(training["max_epochs"]),
        min_epochs=int(training["min_epochs"]),
        patience=int(training["patience"]),
        final_min_epochs=int(training["final_min_epochs"]),
        final_max_epochs=int(training["final_max_epochs"]),
        calibration_grid=tuple(calibration["grid"]),  # type: ignore[arg-type]
        probability_clip=float(calibration["probability_clip"]),
        segment_min_rows=int(segments["minimum_rows"]),
        gates=_freeze_nested(raw["gates"]),  # type: ignore[arg-type]
        inference_limits=MappingProxyType(dict(raw["inference_limits"])),  # type: ignore[arg-type]
        session_seconds=int(budget["session_seconds"]),
        new_job_guard_seconds=int(budget["new_job_guard_seconds"]),
        snapshot_interval_seconds=int(budget["snapshot_interval_seconds"]),
        download_interval_seconds=int(budget["download_interval_seconds"]),
    )


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    source = Path(path)
    load_contract(source)
    return sha256(source.read_bytes()).hexdigest()


def build_jobs(contract: HierarchicalContract) -> tuple[HierarchicalJob, ...]:
    if contract.campaign_id != "hierarchical_tabm_score_push_v1":
        raise HierarchicalContractError("campaign identity differs")
    return (
        HierarchicalJob("h1__tr2022__va2023__s3407", "oof", 2022, 2023),
        HierarchicalJob("h1__tr2023__va2024__s3407", "oof", 2023, 2024),
        HierarchicalJob("h1__full__tr2024__s3407", "full_fit", 2024, None),
    )

