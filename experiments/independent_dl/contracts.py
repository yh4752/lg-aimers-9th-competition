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
                    )
                )
    if len({candidate.candidate_id for candidate in candidates}) != 64:
        raise CampaignContractError("candidate IDs are not unique")
    return tuple(candidates)


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
    if payload["schema_version"] != 1:
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

