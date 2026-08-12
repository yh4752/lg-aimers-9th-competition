"""Strict, dependency-light contract for the independent DL campaign."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class CampaignContractError(ValueError):
    """Raised when a campaign configuration is incomplete or ambiguous."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    feature_view: str
    seed: int
    epochs: int
    model: Mapping[str, object]
    training: Mapping[str, object]
    train_end_year: int | None = None
    valid_year: int | None = None
    stage: str = "exploration"
    parent_candidate_id: str | None = None
    parent_model: Mapping[str, object] | None = None


@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str
    protocol: str
    exploration_fold: tuple[int, int]
    oof_folds: tuple[tuple[int, int], ...]
    feature_views: tuple[str, ...]
    confirmation_seeds: tuple[int, ...]
    blend_weights: tuple[float, ...]
    boundary_expansion: Mapping[str, Mapping[str, tuple[int, ...]]]
    survivor_policy: Mapping[str, object]
    candidates: tuple[CandidateSpec, ...]


_TOP_LEVEL_KEYS = {
    "schema_version",
    "campaign_id",
    "protocol",
    "exploration_fold",
    "oof_folds",
    "feature_views",
    "execution_waves",
    "frontier_candidates",
    "exploration_seed",
    "confirmation_seeds",
    "blend_weights",
    "training_profiles",
    "capacity_profiles",
    "boundary_expansion",
    "survivor_policy",
}
_FAMILIES = ("tabm", "mlp_resnet", "ft_transformer", "tabr")
_FEATURE_VIEWS = (
    "raw_typed",
    "engineered",
    "entity_context",
    "trackman_augmented",
)
_TABICL_V2_MODEL = {
    "architecture": "tabicl_v2",
    "n_estimators": 32,
    "kv_cache": True,
    "offload_mode": "auto",
    "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise CampaignContractError(f"non-finite JSON number: {value}")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CampaignContractError(f"{label} must be an object")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignContractError(f"{label} must be an integer >= {minimum}")
    return value


def _year_pair(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise CampaignContractError(f"{label} must contain two years")
    train_end = _integer(value[0], f"{label}[0]", minimum=1900)
    valid = _integer(value[1], f"{label}[1]", minimum=1900)
    if valid != train_end + 1:
        raise CampaignContractError(f"{label} must be a yearly transition")
    return train_end, valid


def _number_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise CampaignContractError(f"{label} must be a non-empty list")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CampaignContractError(f"{label} must contain numbers")
        number = float(item)
        if not isfinite(number):
            raise CampaignContractError(f"{label} must contain finite numbers")
        numbers.append(number)
    return tuple(numbers)


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _boundary_expansion(value: object) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
    payload = _mapping(value, "boundary_expansion")
    if tuple(payload) != _FAMILIES:
        raise CampaignContractError("boundary_expansion family order is invalid")
    result: dict[str, Mapping[str, tuple[int, ...]]] = {}
    for family, axes_value in payload.items():
        axes = _mapping(axes_value, f"boundary_expansion.{family}")
        frozen_axes: dict[str, tuple[int, ...]] = {}
        for axis, values in axes.items():
            if not isinstance(values, list) or not values:
                raise CampaignContractError(
                    f"boundary_expansion.{family}.{axis} must be a non-empty list"
                )
            frozen_axes[axis] = tuple(
                _integer(item, f"boundary_expansion.{family}.{axis}", minimum=1)
                for item in values
            )
        result[family] = MappingProxyType(frozen_axes)
    return MappingProxyType(result)


def _expand_candidates(payload: dict[str, object]) -> tuple[CandidateSpec, ...]:
    seed = _integer(payload["exploration_seed"], "exploration_seed", minimum=0)
    train_end_year, valid_year = _year_pair(
        payload["exploration_fold"], "exploration_fold"
    )
    training_profiles = _mapping(payload["training_profiles"], "training_profiles")
    capacity_profiles = _mapping(payload["capacity_profiles"], "capacity_profiles")
    if tuple(capacity_profiles) != _FAMILIES:
        raise CampaignContractError("capacity profile family order is invalid")

    candidates: list[CandidateSpec] = []
    for family in _FAMILIES:
        profiles = capacity_profiles[family]
        if not isinstance(profiles, list) or len(profiles) != 4:
            raise CampaignContractError(f"{family} must define four capacity profiles")
        for feature_view in _FEATURE_VIEWS:
            for profile_value in profiles:
                profile = _mapping(profile_value, f"capacity_profiles.{family}")
                if set(profile) != {"profile_id", "training_profile", "epochs", "model"}:
                    raise CampaignContractError(f"{family} capacity profile keys are invalid")
                profile_id = profile["profile_id"]
                training_id = profile["training_profile"]
                if not isinstance(profile_id, str) or not profile_id:
                    raise CampaignContractError(f"{family} profile_id is invalid")
                if not isinstance(training_id, str) or training_id not in training_profiles:
                    raise CampaignContractError(f"{family} training profile is invalid")
                training = _mapping(
                    training_profiles[training_id], f"training_profiles.{training_id}"
                )
                if set(training) != {
                    "optimizer",
                    "scheduler",
                    "learning_rate",
                    "weight_decay",
                    "effective_batch_size",
                    "micro_batch_size",
                    "amp",
                    "patience",
                }:
                    raise CampaignContractError(f"{training_id} keys are invalid")
                model = _mapping(profile["model"], f"{family}.{profile_id}.model")
                candidate_id = f"{family}__{feature_view}__{profile_id}__s{seed}"
                candidates.append(
                    CandidateSpec(
                        candidate_id=candidate_id,
                        family=family,
                        feature_view=feature_view,
                        seed=seed,
                        epochs=_integer(profile["epochs"], "epochs", minimum=100),
                        model=_freeze_mapping(model),
                        training=_freeze_mapping(training),
                        train_end_year=train_end_year,
                        valid_year=valid_year,
                    )
                )
    if len({candidate.candidate_id for candidate in candidates}) != 64:
        raise CampaignContractError("candidate IDs are not unique")
    frontier_value = payload["frontier_candidates"]
    if not isinstance(frontier_value, list) or not frontier_value:
        raise CampaignContractError("frontier_candidates must be a non-empty list")
    for index, candidate_value in enumerate(frontier_value):
        candidate = _mapping(candidate_value, f"frontier_candidates[{index}]")
        if set(candidate) != {
            "candidate_id",
            "family",
            "feature_view",
            "seed",
            "epochs",
            "stage",
            "model",
            "training",
        }:
            raise CampaignContractError("frontier candidate keys are invalid")
        if candidate["family"] != "tabicl_v2":
            raise CampaignContractError("frontier candidate family is invalid")
        if candidate["feature_view"] not in _FEATURE_VIEWS:
            raise CampaignContractError("frontier candidate feature view is invalid")
        if candidate["stage"] != "research_only":
            raise CampaignContractError("frontier candidate must be research_only")
        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id:
            raise CampaignContractError("frontier candidate_id is invalid")
        model = _mapping(candidate["model"], "frontier candidate model")
        training = _mapping(candidate["training"], "frontier candidate training")
        if model != _TABICL_V2_MODEL:
            raise CampaignContractError("frontier candidate model is invalid")
        if training:
            raise CampaignContractError("frontier candidate training must be empty")
        candidates.append(
            CandidateSpec(
                candidate_id=candidate_id,
                family="tabicl_v2",
                feature_view=str(candidate["feature_view"]),
                seed=_integer(candidate["seed"], "frontier seed", minimum=0),
                epochs=_integer(candidate["epochs"], "frontier epochs", minimum=1),
                model=_freeze_mapping(model),
                training=_freeze_mapping(training),
                train_end_year=train_end_year,
                valid_year=valid_year,
                stage="research_only",
            )
        )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise CampaignContractError("candidate IDs are not unique")
    waves_value = payload["execution_waves"]
    if not isinstance(waves_value, list) or not waves_value:
        raise CampaignContractError("execution_waves must be a non-empty list")
    ordered_ids: list[str] = []
    saw_remaining = False
    for index, wave_value in enumerate(waves_value):
        wave = _mapping(wave_value, f"execution_waves[{index}]")
        if set(wave) != {"wave_id", "candidate_ids"}:
            raise CampaignContractError("execution wave keys are invalid")
        wave_id = wave["wave_id"]
        if not isinstance(wave_id, str) or not wave_id:
            raise CampaignContractError("execution wave_id is invalid")
        ids = wave["candidate_ids"]
        if ids == "remaining_grid":
            if saw_remaining or index != len(waves_value) - 1:
                raise CampaignContractError("remaining_grid must be the final wave")
            saw_remaining = True
            ordered_ids.extend(
                candidate.candidate_id
                for candidate in candidates
                if candidate.candidate_id not in ordered_ids
            )
            continue
        if not isinstance(ids, list) or not ids:
            raise CampaignContractError("execution candidate_ids are invalid")
        for candidate_id in ids:
            if not isinstance(candidate_id, str) or candidate_id not in by_id:
                raise CampaignContractError("execution wave contains unknown candidate")
            if candidate_id in ordered_ids:
                raise CampaignContractError("execution wave contains duplicate candidate")
            ordered_ids.append(candidate_id)
    if not saw_remaining or len(ordered_ids) != len(candidates):
        raise CampaignContractError("execution waves do not cover every candidate")
    return tuple(by_id[candidate_id] for candidate_id in ordered_ids)


def load_campaign(path: str | Path) -> CampaignSpec:
    """Load and validate the sealed independent-DL campaign JSON."""

    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except CampaignContractError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignContractError(f"cannot read campaign config: {error}") from error
    payload = _mapping(value, "campaign")
    if set(payload) != _TOP_LEVEL_KEYS:
        raise CampaignContractError("campaign top-level keys are invalid")
    if payload["schema_version"] != 2:
        raise CampaignContractError("unsupported campaign schema_version")
    if payload["campaign_id"] != "independent_dl_campaign_v1":
        raise CampaignContractError("campaign_id is invalid")
    if payload["protocol"] != "yearly_transition_independent_dl_v1":
        raise CampaignContractError("protocol is invalid")
    if tuple(payload["feature_views"]) != _FEATURE_VIEWS:
        raise CampaignContractError("feature view order is invalid")

    confirmation = payload["confirmation_seeds"]
    if not isinstance(confirmation, list) or not confirmation:
        raise CampaignContractError("confirmation_seeds must be a non-empty list")
    confirmation_seeds = tuple(
        _integer(item, "confirmation_seeds", minimum=0) for item in confirmation
    )
    oof_value = payload["oof_folds"]
    if not isinstance(oof_value, list) or not oof_value:
        raise CampaignContractError("oof_folds must be a non-empty list")
    oof_folds = tuple(
        _year_pair(item, f"oof_folds[{index}]")
        for index, item in enumerate(oof_value)
    )
    survivor_policy = _mapping(payload["survivor_policy"], "survivor_policy")
    return CampaignSpec(
        campaign_id=str(payload["campaign_id"]),
        protocol=str(payload["protocol"]),
        exploration_fold=_year_pair(payload["exploration_fold"], "exploration_fold"),
        oof_folds=oof_folds,
        feature_views=_FEATURE_VIEWS,
        confirmation_seeds=confirmation_seeds,
        blend_weights=_number_tuple(payload["blend_weights"], "blend_weights"),
        boundary_expansion=_boundary_expansion(payload["boundary_expansion"]),
        survivor_policy=_freeze_mapping(survivor_policy),
        candidates=_expand_candidates(payload),
    )
