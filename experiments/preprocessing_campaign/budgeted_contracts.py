"""Sealed contracts for the five-stage, time-budgeted preprocessing campaign."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pandas as pd


class BudgetedContractError(ValueError):
    """Raised before a malformed campaign can start expensive work."""


@dataclass(frozen=True)
class BudgetedJob:
    job_id: str
    stage_id: int
    family: str
    profile_id: str
    setting_id: str
    preprocessing_profile: str
    components: tuple[str, ...]
    model: Mapping[str, object]
    training: Mapping[str, object]
    train_end_year: int
    valid_year: int
    seed: int
    max_seconds: int
    sample_mode: str = "proxy"


@dataclass(frozen=True)
class BudgetedCampaign:
    campaign_id: str
    session_seconds: int
    stop_new_jobs_seconds: int
    archive_reserve_seconds: int
    proxy_max_rows: int
    seed: int
    blend_weights: tuple[float, ...]
    primary_fold: tuple[int, int]
    stability_fold: tuple[int, int]
    stages: Mapping[int, tuple[BudgetedJob, ...]]

    def stage_jobs(self, stage_id: int) -> tuple[BudgetedJob, ...]:
        if stage_id not in range(1, 6):
            raise BudgetedContractError("stage_id must be between 1 and 5")
        return self.stages.get(stage_id, ())


_TOP_LEVEL_KEYS = {
    "schema_version",
    "campaign_id",
    "session_seconds",
    "stop_new_jobs_seconds",
    "archive_reserve_seconds",
    "proxy_max_rows",
    "seed",
    "blend_weights",
    "primary_fold",
    "stability_fold",
    "stages",
}
_JOB_KEYS = {
    "job_id",
    "family",
    "profile_id",
    "setting_id",
    "preprocessing_profile",
    "components",
    "model",
    "training",
    "train_end_year",
    "valid_year",
    "seed",
    "max_seconds",
    "sample_mode",
}
_FAMILIES = {"tabm", "ft_transformer", "tabnet", "catboost"}
_PROFILES = {"dl_standard", "dl_selective_transform", "tree_native"}


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BudgetedContractError(f"{label} must be an object")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BudgetedContractError(f"{label} must be a positive integer")
    return value


def _fold(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise BudgetedContractError(f"{label} must contain two years")
    left = _positive_int(value[0], f"{label}[0]")
    right = _positive_int(value[1], f"{label}[1]")
    if right != left + 1:
        raise BudgetedContractError(f"{label} must be an adjacent temporal fold")
    return left, right


def _job(raw: object, stage_id: int) -> BudgetedJob:
    payload = _object(raw, f"stage {stage_id} job")
    if set(payload) != _JOB_KEYS:
        raise BudgetedContractError(f"stage {stage_id} job keys are invalid")
    family = str(payload["family"])
    profile = str(payload["preprocessing_profile"])
    if family not in _FAMILIES:
        raise BudgetedContractError(f"unknown family: {family}")
    if profile not in _PROFILES:
        raise BudgetedContractError(f"unknown preprocessing profile: {profile}")
    components = payload["components"]
    if not isinstance(components, list) or any(
        not isinstance(item, str) or not item for item in components
    ):
        raise BudgetedContractError("components must be a string list")
    if len(set(components)) != len(components):
        raise BudgetedContractError("components must be unique")
    model = _object(payload["model"], "model")
    training = _object(payload["training"], "training")
    sample_mode = str(payload["sample_mode"])
    if sample_mode not in {"proxy", "full"}:
        raise BudgetedContractError("sample_mode must be proxy or full")
    return BudgetedJob(
        job_id=str(payload["job_id"]),
        stage_id=stage_id,
        family=family,
        profile_id=str(payload["profile_id"]),
        setting_id=str(payload["setting_id"]),
        preprocessing_profile=profile,
        components=tuple(components),
        model=MappingProxyType(dict(model)),
        training=MappingProxyType(dict(training)),
        train_end_year=_positive_int(payload["train_end_year"], "train_end_year"),
        valid_year=_positive_int(payload["valid_year"], "valid_year"),
        seed=_positive_int(payload["seed"], "seed"),
        max_seconds=_positive_int(payload["max_seconds"], "max_seconds"),
        sample_mode=sample_mode,
    )


def load_budgeted_campaign(path: str | Path) -> BudgetedCampaign:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BudgetedContractError(f"cannot read campaign config: {error}") from error
    raw = _object(payload, "campaign")
    if set(raw) != _TOP_LEVEL_KEYS or raw["schema_version"] != 1:
        raise BudgetedContractError("campaign top-level contract is invalid")
    session = _positive_int(raw["session_seconds"], "session_seconds")
    stop = _positive_int(raw["stop_new_jobs_seconds"], "stop_new_jobs_seconds")
    reserve = _positive_int(raw["archive_reserve_seconds"], "archive_reserve_seconds")
    if not reserve < stop < session:
        raise BudgetedContractError("session guards must satisfy reserve < stop < session")
    seed = _positive_int(raw["seed"], "seed")
    weights = raw["blend_weights"]
    if not isinstance(weights, list) or not weights:
        raise BudgetedContractError("blend_weights must be non-empty")
    blend_weights = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) or not 0 < value < 1 for value in blend_weights):
        raise BudgetedContractError("blend weights must be finite and between zero and one")
    stages_raw = _object(raw["stages"], "stages")
    if set(stages_raw) != {str(index) for index in range(1, 6)}:
        raise BudgetedContractError("stages 1 through 5 must be declared")
    stages: dict[int, tuple[BudgetedJob, ...]] = {}
    all_ids: set[str] = set()
    for stage_id in range(1, 6):
        values = stages_raw[str(stage_id)]
        if not isinstance(values, list):
            raise BudgetedContractError(f"stage {stage_id} must be a job list")
        jobs = tuple(_job(value, stage_id) for value in values)
        for job in jobs:
            if not job.job_id or job.job_id in all_ids:
                raise BudgetedContractError(f"duplicate or empty job ID: {job.job_id}")
            if job.seed != seed or job.valid_year != job.train_end_year + 1:
                raise BudgetedContractError(f"job temporal contract is invalid: {job.job_id}")
            all_ids.add(job.job_id)
        stages[stage_id] = jobs
    return BudgetedCampaign(
        campaign_id=str(raw["campaign_id"]),
        session_seconds=session,
        stop_new_jobs_seconds=stop,
        archive_reserve_seconds=reserve,
        proxy_max_rows=_positive_int(raw["proxy_max_rows"], "proxy_max_rows"),
        seed=seed,
        blend_weights=blend_weights,
        primary_fold=_fold(raw["primary_fold"], "primary_fold"),
        stability_fold=_fold(raw["stability_fold"], "stability_fold"),
        stages=MappingProxyType(stages),
    )


def _row_hash(seed: int, row_id: object) -> str:
    return sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()


def deterministic_temporal_sample(
    frame: pd.DataFrame,
    *,
    train_end_year: int,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Return a deterministic, season-proportional sample from training years only."""

    _positive_int(train_end_year, "train_end_year")
    _positive_int(max_rows, "max_rows")
    _positive_int(seed, "seed")
    missing = {"row_id", "season"} - set(frame.columns)
    if missing:
        raise BudgetedContractError(f"sample frame is missing columns: {sorted(missing)}")
    seasons = pd.to_numeric(frame["season"], errors="raise").astype("int64")
    eligible = frame.loc[seasons.le(train_end_year)].copy()
    eligible["season"] = seasons.loc[eligible.index]
    if eligible.empty:
        raise BudgetedContractError("temporal sample has no eligible training rows")
    if eligible["row_id"].astype(str).duplicated().any():
        raise BudgetedContractError("row_id must be unique")
    eligible["__sample_hash"] = [
        _row_hash(seed, value) for value in eligible["row_id"].astype(str)
    ]
    if len(eligible) <= max_rows:
        return (
            eligible.sort_values("__sample_hash", kind="stable")
            .drop(columns="__sample_hash")
            .reset_index(drop=True)
        )
    counts = eligible.groupby("season", sort=True).size()
    quotas = counts.astype(float) * (max_rows / len(eligible))
    allocation = quotas.map(math.floor).astype(int)
    remaining = max_rows - int(allocation.sum())
    order = sorted(
        counts.index,
        key=lambda year: (-(quotas.loc[year] - allocation.loc[year]), int(year)),
    )
    for year in order[:remaining]:
        allocation.loc[year] += 1
    pieces = []
    for year, count in allocation.items():
        part = (
            eligible.loc[eligible["season"].eq(year)]
            .sort_values("__sample_hash", kind="stable")
            .head(int(count))
        )
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True).sort_values(
        "__sample_hash", kind="stable"
    )
    return result.drop(columns="__sample_hash").reset_index(drop=True)
