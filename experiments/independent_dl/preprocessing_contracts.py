"""Strict contract for the EDA-informed preprocessing campaign."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .contracts import CandidateSpec, load_campaign


class PreprocessingContractError(ValueError):
    """Raised when the preprocessing campaign cannot be reproduced exactly."""


@dataclass(frozen=True)
class PreprocessingSetting:
    setting_id: str
    profile: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessingJob:
    job_id: str
    wave: str
    anchor_id: str
    family: str
    profile_id: str
    model: Mapping[str, object]
    training: Mapping[str, object]
    feature_view: str
    setting: PreprocessingSetting
    train_end_year: int
    valid_year: int
    seed: int


@dataclass(frozen=True)
class PreprocessingCampaignSpec:
    campaign_id: str
    protocol: str
    folds: tuple[tuple[int, int], ...]
    seeds: tuple[int, ...]
    anchors: Mapping[str, CandidateSpec]
    anchor_profiles: Mapping[str, str]
    dl_settings: tuple[PreprocessingSetting, ...]
    pitcher_smoothing_k: tuple[int, ...]
    batter_smoothing_k: tuple[int, ...]
    catboost_structures: Mapping[str, Mapping[str, object]]
    catboost_settings: tuple[PreprocessingSetting, ...]
    promotion: Mapping[str, object]
    wave_a_jobs: tuple[PreprocessingJob, ...]


_TOP_LEVEL_KEYS = {
    "schema_version",
    "campaign_id",
    "protocol",
    "source_campaign_config",
    "source_campaign_sha256",
    "folds",
    "seeds",
    "anchors",
    "pitcher_smoothing_k",
    "batter_smoothing_k",
    "dl_settings",
    "promotion",
    "catboost_structures",
    "catboost_settings",
}
_PROFILES = {"tree_native", "dl_standard", "dl_selective_transform"}
_ANCHOR_KEYS = {"anchor_id", "source_candidate_id", "profile_id"}
_SETTING_KEYS = {"setting_id", "profile", "components"}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PreprocessingContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise PreprocessingContractError(f"non-finite JSON number: {value}")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PreprocessingContractError(f"{label} must be an object")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PreprocessingContractError(f"{label} must be an integer >= {minimum}")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or not all(
        character.isalnum() or character in {"_", "-"} for character in value
    ):
        raise PreprocessingContractError(f"{label} must be a non-empty identifier")
    return value


def _integer_tuple(value: object, label: str, *, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PreprocessingContractError(f"{label} must be a non-empty list")
    result = tuple(_integer(item, label, minimum=minimum) for item in value)
    if len(set(result)) != len(result):
        raise PreprocessingContractError(f"{label} must contain unique values")
    return result


def _folds(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise PreprocessingContractError("folds must be a non-empty list")
    result: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise PreprocessingContractError(f"folds[{index}] must contain two years")
        train_end = _integer(item[0], f"folds[{index}][0]", minimum=1900)
        valid = _integer(item[1], f"folds[{index}][1]", minimum=1900)
        if valid != train_end + 1:
            raise PreprocessingContractError("each fold must be a yearly transition")
        result.append((train_end, valid))
    if len(set(result)) != len(result):
        raise PreprocessingContractError("folds must be unique")
    return tuple(result)


def _settings(value: object, label: str) -> tuple[PreprocessingSetting, ...]:
    if not isinstance(value, list) or not value:
        raise PreprocessingContractError(f"{label} must be a non-empty list")
    result: list[PreprocessingSetting] = []
    for index, item in enumerate(value):
        payload = _mapping(item, f"{label}[{index}]")
        if set(payload) != _SETTING_KEYS:
            raise PreprocessingContractError(f"{label}[{index}] keys are invalid")
        profile = _identifier(payload["profile"], f"{label}[{index}].profile")
        if profile not in _PROFILES:
            raise PreprocessingContractError(f"{label}[{index}].profile is invalid")
        raw_components = payload["components"]
        if not isinstance(raw_components, list):
            raise PreprocessingContractError(f"{label}[{index}].components must be a list")
        components = tuple(
            _identifier(item, f"{label}[{index}].components")
            for item in raw_components
        )
        if len(set(components)) != len(components):
            raise PreprocessingContractError(f"{label}[{index}] components repeat")
        result.append(
            PreprocessingSetting(
                setting_id=_identifier(
                    payload["setting_id"], f"{label}[{index}].setting_id"
                ),
                profile=profile,
                components=components,
            )
        )
    if len({setting.setting_id for setting in result}) != len(result):
        raise PreprocessingContractError(f"{label} setting IDs must be unique")
    return tuple(result)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _finite_mapping(value: object, label: str) -> Mapping[str, object]:
    payload = _mapping(value, label)
    result: dict[str, object] = {}
    for key, item in payload.items():
        if not isinstance(key, str) or not key:
            raise PreprocessingContractError(f"{label} keys must be non-empty strings")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PreprocessingContractError(f"{label}.{key} must be numeric")
        if not isfinite(float(item)):
            raise PreprocessingContractError(f"{label}.{key} must be finite")
        result[key] = item
    return MappingProxyType(result)


def _load_payload(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except PreprocessingContractError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise PreprocessingContractError(f"cannot read preprocessing config: {error}") from error
    return _mapping(value, "preprocessing campaign")


def load_preprocessing_campaign(path: str | Path) -> PreprocessingCampaignSpec:
    """Load the sealed preprocessing campaign and bind its existing DL anchors."""

    config_path = Path(path).resolve()
    payload = _load_payload(config_path)
    if set(payload) != _TOP_LEVEL_KEYS:
        raise PreprocessingContractError("preprocessing campaign top-level keys are invalid")
    if payload["schema_version"] != 1:
        raise PreprocessingContractError("unsupported preprocessing schema_version")
    campaign_id = _identifier(payload["campaign_id"], "campaign_id")
    if campaign_id != "preprocessing_campaign_v1":
        raise PreprocessingContractError("campaign_id is invalid")
    if payload["protocol"] != "five_fold_eda_informed_preprocessing_v1":
        raise PreprocessingContractError("protocol is invalid")

    source_name = payload["source_campaign_config"]
    if not isinstance(source_name, str) or Path(source_name).name != source_name:
        raise PreprocessingContractError("source_campaign_config must be a sibling filename")
    source_path = config_path.parent / source_name
    expected_source_hash = payload["source_campaign_sha256"]
    if not isinstance(expected_source_hash, str) or len(expected_source_hash) != 64:
        raise PreprocessingContractError("source campaign SHA-256 is invalid")
    try:
        actual_source_hash = _file_sha256(source_path)
    except OSError as error:
        raise PreprocessingContractError(f"cannot read source campaign config: {error}") from error
    if actual_source_hash != expected_source_hash:
        raise PreprocessingContractError(
            "source campaign SHA-256 differs: "
            f"expected={expected_source_hash} actual={actual_source_hash}"
        )
    source_campaign = load_campaign(source_path)
    source_candidates = {
        candidate.candidate_id: candidate for candidate in source_campaign.candidates
    }

    raw_anchors = payload["anchors"]
    if not isinstance(raw_anchors, list) or len(raw_anchors) != 8:
        raise PreprocessingContractError("anchors must contain exactly eight entries")
    anchors: dict[str, CandidateSpec] = {}
    anchor_profiles: dict[str, str] = {}
    for index, item in enumerate(raw_anchors):
        anchor_payload = _mapping(item, f"anchors[{index}]")
        if set(anchor_payload) != _ANCHOR_KEYS:
            raise PreprocessingContractError(f"anchors[{index}] keys are invalid")
        anchor_id = _identifier(anchor_payload["anchor_id"], f"anchors[{index}].anchor_id")
        source_id = _identifier(
            anchor_payload["source_candidate_id"],
            f"anchors[{index}].source_candidate_id",
        )
        profile_id = _identifier(
            anchor_payload["profile_id"], f"anchors[{index}].profile_id"
        )
        if profile_id not in {"p3", "p4"}:
            raise PreprocessingContractError("only p3 and p4 anchors are allowed")
        candidate = source_candidates.get(source_id)
        if candidate is None or candidate.feature_view != "raw_typed":
            raise PreprocessingContractError(f"source anchor is missing: {source_id}")
        if f"__{profile_id}__" not in candidate.candidate_id:
            raise PreprocessingContractError(f"anchor profile differs: {anchor_id}")
        if anchor_id in anchors:
            raise PreprocessingContractError(f"anchor ID repeats: {anchor_id}")
        anchors[anchor_id] = candidate
        anchor_profiles[anchor_id] = profile_id

    folds = _folds(payload["folds"])
    seeds = _integer_tuple(payload["seeds"], "seeds")
    if seeds != (42, 2026, 3407):
        raise PreprocessingContractError("seeds must equal 42, 2026, 3407")
    pitcher_k = _integer_tuple(
        payload["pitcher_smoothing_k"], "pitcher_smoothing_k", minimum=1
    )
    batter_k = _integer_tuple(
        payload["batter_smoothing_k"], "batter_smoothing_k", minimum=1
    )
    dl_settings = _settings(payload["dl_settings"], "dl_settings")
    catboost_settings = _settings(payload["catboost_settings"], "catboost_settings")
    if len(dl_settings) != 19 or len(catboost_settings) != 17:
        raise PreprocessingContractError("preprocessing setting counts are invalid")

    raw_structures = _mapping(payload["catboost_structures"], "catboost_structures")
    if tuple(raw_structures) != ("champion", "depth5", "depth8", "lr003"):
        raise PreprocessingContractError("catboost structure order is invalid")
    structures = MappingProxyType(
        {
            name: _finite_mapping(value, f"catboost_structures.{name}")
            for name, value in raw_structures.items()
        }
    )
    promotion = MappingProxyType(dict(_mapping(payload["promotion"], "promotion")))

    jobs: list[PreprocessingJob] = []
    for anchor_id, candidate in anchors.items():
        profile_id = anchor_profiles[anchor_id]
        for setting in dl_settings:
            for train_end_year, valid_year in folds:
                job_id = (
                    f"a__{anchor_id}__{setting.setting_id}__"
                    f"tr{train_end_year}__va{valid_year}__s42"
                )
                jobs.append(
                    PreprocessingJob(
                        job_id=job_id,
                        wave="a",
                        anchor_id=anchor_id,
                        family=candidate.family,
                        profile_id=profile_id,
                        model=candidate.model,
                        training=candidate.training,
                        feature_view="raw_typed",
                        setting=setting,
                        train_end_year=train_end_year,
                        valid_year=valid_year,
                        seed=42,
                    )
                )
    if len(jobs) != 760 or len({job.job_id for job in jobs}) != len(jobs):
        raise PreprocessingContractError("Wave A expansion is invalid")

    return PreprocessingCampaignSpec(
        campaign_id=campaign_id,
        protocol=str(payload["protocol"]),
        folds=folds,
        seeds=seeds,
        anchors=MappingProxyType(anchors),
        anchor_profiles=MappingProxyType(anchor_profiles),
        dl_settings=dl_settings,
        pitcher_smoothing_k=pitcher_k,
        batter_smoothing_k=batter_k,
        catboost_structures=structures,
        catboost_settings=catboost_settings,
        promotion=promotion,
        wave_a_jobs=tuple(jobs),
    )
