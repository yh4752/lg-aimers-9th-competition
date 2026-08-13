from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class CampaignContractError(ValueError):
    """Raised when a campaign configuration violates its sealed contract."""


@dataclass(frozen=True)
class FixedPreprocessing:
    profile: str
    components: tuple[str, ...]
    fit_scope: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    capacity: str
    k: int
    width: int
    blocks: int
    dropout: float
    num_embedding: str
    loss: str
    scheduler: str
    learning_rate: float
    seed: int


@dataclass(frozen=True)
class Refinement:
    learning_rate: float
    dropout_offset: float


@dataclass(frozen=True)
class Fold:
    train_end_year: int
    valid_year: int


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    preprocessing: FixedPreprocessing
    wall_seconds: Mapping[str, int]
    finalization_reserve_seconds: int
    proxy_max_rows: int
    seeds: tuple[int, ...]
    max_final_weights: int
    version_a_candidates: tuple[Candidate, ...]
    refinements: tuple[Refinement, ...]
    folds: tuple[Fold, ...]
    optimization: Mapping[str, Any]
    stage_epochs: Mapping[str, Mapping[str, int]]
    gates: Mapping[str, float]


_TOP_KEYS = {
    "campaign_id",
    "preprocessing",
    "wall_seconds",
    "finalization_reserve_seconds",
    "proxy_max_rows",
    "seeds",
    "max_final_weights",
    "candidates",
    "refinements",
    "folds",
    "optimization",
    "stage_epochs",
    "gates",
}
_CANDIDATE_KEYS = {
    "candidate_id",
    "family",
    "capacity",
    "k",
    "width",
    "blocks",
    "dropout",
    "num_embedding",
    "loss",
    "scheduler",
    "learning_rate",
    "seed",
}
_SEALED_WALL_SECONDS = {"A": 7200, "B": 10800, "C": 10800, "D": 7200}
_CAPACITIES = {
    "p2": (32, 512, 4, 0.10),
    "p3_lite": (32, 768, 6, 0.15),
    "p3_full": (64, 768, 6, 0.15),
}


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CampaignContractError(f"non-finite JSON number: {value}")


def _require_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise CampaignContractError(f"{context} unknown keys: {sorted(unknown)}")
    if missing:
        raise CampaignContractError(f"{context} missing keys: {sorted(missing)}")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignContractError(f"{name} must be finite")
    return result


def _load_candidate(raw: Mapping[str, Any]) -> Candidate:
    _require_keys(raw, _CANDIDATE_KEYS, "candidate")
    candidate = Candidate(
        candidate_id=str(raw["candidate_id"]),
        family=str(raw["family"]),
        capacity=str(raw["capacity"]),
        k=int(raw["k"]),
        width=int(raw["width"]),
        blocks=int(raw["blocks"]),
        dropout=_finite_float(raw["dropout"], "candidate.dropout"),
        num_embedding=str(raw["num_embedding"]),
        loss=str(raw["loss"]),
        scheduler=str(raw["scheduler"]),
        learning_rate=_finite_float(raw["learning_rate"], "candidate.learning_rate"),
        seed=int(raw["seed"]),
    )
    if candidate.family != "tabm":
        raise CampaignContractError("candidate family must be tabm")
    if candidate.capacity not in _CAPACITIES:
        raise CampaignContractError(f"unknown capacity: {candidate.capacity}")
    expected_shape = _CAPACITIES[candidate.capacity]
    if (candidate.k, candidate.width, candidate.blocks, candidate.dropout) != expected_shape:
        raise CampaignContractError(f"capacity parameters changed: {candidate.capacity}")
    if candidate.num_embedding not in {"piecewise_linear", "periodic"}:
        raise CampaignContractError("invalid numerical embedding")
    if candidate.loss not in {"bce", "brier"}:
        raise CampaignContractError("invalid loss")
    if candidate.scheduler not in {"plateau", "one_cycle"}:
        raise CampaignContractError("invalid scheduler")
    expected_id = (
        f"a__{candidate.capacity}__{candidate.num_embedding}__"
        f"{candidate.loss}__{candidate.scheduler}__s{candidate.seed}"
    )
    if candidate.candidate_id != expected_id:
        raise CampaignContractError(f"non-canonical candidate_id: {candidate.candidate_id}")
    if candidate.learning_rate != 0.0006 or candidate.seed != 42:
        raise CampaignContractError("Version A learning rate and seed are sealed")
    return candidate


def load_campaign(path: str | Path) -> Campaign:
    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignContractError(f"invalid campaign JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CampaignContractError("campaign root must be an object")
    _require_keys(raw, _TOP_KEYS, "campaign")

    prep = raw["preprocessing"]
    _require_keys(prep, {"profile", "components", "fit_scope"}, "preprocessing")
    preprocessing = FixedPreprocessing(
        profile=str(prep["profile"]),
        components=tuple(str(item) for item in prep["components"]),
        fit_scope=str(prep["fit_scope"]),
    )
    if (preprocessing.profile, preprocessing.components) != ("dl_standard", ("hand_matchup",)):
        raise CampaignContractError("preprocessing must be dl_standard + hand_matchup")
    if preprocessing.fit_scope != "official_train_only":
        raise CampaignContractError("fit_scope must be official_train_only")

    wall_seconds = {str(key): int(value) for key, value in raw["wall_seconds"].items()}
    if wall_seconds != _SEALED_WALL_SECONDS:
        raise CampaignContractError(f"wall_seconds must equal {_SEALED_WALL_SECONDS}")

    candidates = tuple(_load_candidate(item) for item in raw["candidates"])
    if len(candidates) != 24 or len({item.candidate_id for item in candidates}) != 24:
        raise CampaignContractError("Version A must contain 24 unique candidate IDs")
    actual_cross_product = {
        (item.capacity, item.num_embedding, item.loss, item.scheduler) for item in candidates
    }
    expected_cross_product = {
        (capacity, embedding, loss, scheduler)
        for capacity in _CAPACITIES
        for embedding in ("piecewise_linear", "periodic")
        for loss in ("bce", "brier")
        for scheduler in ("plateau", "one_cycle")
    }
    if actual_cross_product != expected_cross_product:
        raise CampaignContractError("Version A candidate cross-product must contain 24 combinations")

    refinements = tuple(
        Refinement(
            learning_rate=_finite_float(item["learning_rate"], "refinement.learning_rate"),
            dropout_offset=_finite_float(item["dropout_offset"], "refinement.dropout_offset"),
        )
        for item in raw["refinements"]
    )
    expected_refinements = {(lr, offset) for lr in (0.0003, 0.0006, 0.0009) for offset in (-0.05, 0.0, 0.05)}
    if len(refinements) != 9 or {(x.learning_rate, x.dropout_offset) for x in refinements} != expected_refinements:
        raise CampaignContractError("refinements must be the sealed 3x3 grid")

    folds = tuple(Fold(int(item["train_end_year"]), int(item["valid_year"])) for item in raw["folds"])
    if folds != (Fold(2023, 2024), Fold(2022, 2023)):
        raise CampaignContractError("folds must be 2023->2024 and 2022->2023")
    if tuple(raw["seeds"]) != (42, 2026, 3407):
        raise CampaignContractError("seeds must be [42, 2026, 3407]")
    if int(raw["max_final_weights"]) != 3:
        raise CampaignContractError("max_final_weights must be 3")
    if int(raw["proxy_max_rows"]) != 400_000:
        raise CampaignContractError("proxy_max_rows must be 400000")
    if int(raw["finalization_reserve_seconds"]) != 600:
        raise CampaignContractError("finalization_reserve_seconds must be 600")

    optimization = dict(raw["optimization"])
    expected_optimization = {
        "optimizer": "adamw",
        "weight_decay": 0.0001,
        "effective_batch_size": 4096,
        "micro_batch_size": 512,
        "amp": True,
    }
    if optimization != expected_optimization:
        raise CampaignContractError("optimization settings differ from the sealed contract")
    stage_epochs = raw["stage_epochs"]
    if stage_epochs != {
        "A": {"max_epochs": 8, "min_epochs": 3, "patience": 3},
        "B": {"max_epochs": 40, "min_epochs": 3, "patience": 10},
        "C": {"max_epochs": 40, "min_epochs": 3, "patience": 10},
    }:
        raise CampaignContractError("stage_epochs differ from the sealed contract")

    expected_gate_keys = {
        "primary_max_degrade",
        "older_max_degrade",
        "primary_weight",
        "older_weight",
        "ensemble_min_primary_gain",
        "ensemble_max_older_degrade",
        "tie_tolerance",
        "segment_min_rows",
        "segment_max_degrade",
        "row_probability_tolerance",
    }
    gates = {str(key): _finite_float(value, f"gates.{key}") for key, value in raw["gates"].items()}
    if set(gates) != expected_gate_keys:
        raise CampaignContractError("gates have unknown or missing keys")

    return Campaign(
        campaign_id=str(raw["campaign_id"]),
        preprocessing=preprocessing,
        wall_seconds=MappingProxyType(wall_seconds),
        finalization_reserve_seconds=600,
        proxy_max_rows=400_000,
        seeds=(42, 2026, 3407),
        max_final_weights=3,
        version_a_candidates=candidates,
        refinements=refinements,
        folds=folds,
        optimization=MappingProxyType(optimization),
        stage_epochs=MappingProxyType({key: MappingProxyType(dict(value)) for key, value in stage_epochs.items()}),
        gates=MappingProxyType(gates),
    )
