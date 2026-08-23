"""Deterministic physical-stage planning for the temporal campaign."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping

from .contracts import JobSpec, build_stage_jobs, load_contract
from .identity import TrainingIdentity


class PlannerError(ValueError):
    """Raised when prior evidence cannot authorize a stage plan."""


PHYSICAL_STAGES = ("T1", "T2A", "T2B", "T3A", "T3BT4", "T5A", "T5B")
_STAGE_IDS = {stage: index + 1 for index, stage in enumerate(PHYSICAL_STAGES)}
_CAPS = {"T1": 32, "T2A": 10, "T2B": 10, "T3A": 10, "T3BT4": 10, "T5A": 12, "T5B": 12}


@dataclass(frozen=True)
class PlannedJob:
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
    sample_mode: str
    identity: TrainingIdentity


@dataclass(frozen=True)
class PriorReview:
    input_rows_sha256: str
    authorized_jobs: Mapping[str, tuple[PlannedJob, ...]]
    parent_manifest_sha256: str | None = None
    sequence: int = 0

    def __post_init__(self) -> None:
        if not _is_sha256(self.input_rows_sha256):
            raise PlannerError("prior review input rows SHA-256 is invalid")
        if not isinstance(self.authorized_jobs, Mapping):
            raise PlannerError("prior review jobs must be a mapping")
        if type(self.sequence) is not int or self.sequence < 0:
            raise PlannerError("prior review sequence is invalid")
        if (self.sequence == 0) != (self.parent_manifest_sha256 is None):
            raise PlannerError("prior review lineage is incoherent")
        if self.parent_manifest_sha256 is not None and not _is_sha256(
            self.parent_manifest_sha256
        ):
            raise PlannerError("prior review parent manifest SHA-256 is invalid")
        snapshot: dict[str, tuple[PlannedJob, ...]] = {}
        for stage, jobs in self.authorized_jobs.items():
            if stage not in PHYSICAL_STAGES or type(jobs) is not tuple or any(type(job) is not PlannedJob for job in jobs):
                raise PlannerError("prior review contains invalid authorized jobs")
            if len({job.job_id for job in jobs}) != len(jobs):
                raise PlannerError("prior review job IDs must be unique")
            snapshot[stage] = jobs
        object.__setattr__(self, "authorized_jobs", MappingProxyType(snapshot))


@dataclass(frozen=True)
class StagePlan:
    stage: str
    jobs: tuple[PlannedJob, ...]
    scheduler_identity: Mapping[str, str]
    gpu_free_decision: bool


def plan_stage(
    stage: str,
    *,
    prior_review: PriorReview,
    completed: Mapping[str, str],
) -> StagePlan:
    if stage not in PHYSICAL_STAGES:
        raise PlannerError("physical stage is not authorized")
    if type(prior_review) is not PriorReview:
        raise PlannerError("prior_review must be sealed PriorReview")
    if stage != "T1" and prior_review.sequence == 0:
        raise PlannerError("later stage requires verified prior lineage")
    completed_snapshot = _completed(completed)
    if stage == "T1":
        jobs = tuple(_t1_job(spec, prior_review.input_rows_sha256) for spec in build_stage_jobs(load_contract(), "T1"))
    else:
        jobs = prior_review.authorized_jobs.get(stage, ())
        if not jobs and stage != "T3BT4":
            raise PlannerError("prior review authorized no jobs for this stage")
    if stage == "T5A" and any(job.seed != load_contract().confirm_seed for job in jobs):
        raise PlannerError("T5A accepts only confirmation-seed jobs")
    pending = tuple(job for job in jobs if job.identity.sha256 not in completed_snapshot)
    if len(pending) > _CAPS[stage]:
        raise PlannerError("planned jobs exceed the physical-stage cap")
    if len({job.identity.sha256 for job in jobs}) != len(jobs):
        raise PlannerError("stage contains duplicate semantic identities")
    payload = {
        "stage": stage,
        "input_rows_sha256": prior_review.input_rows_sha256,
        "jobs": [job.identity.sha256 for job in pending],
    }
    plan_sha = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return StagePlan(
        stage,
        pending,
        MappingProxyType({"campaign_id": "temporal_portfolio_v1", "stage": stage, "plan_sha256": plan_sha}),
        stage == "T3BT4",
    )


def _t1_job(spec: JobSpec, data_rows: str) -> PlannedJob:
    decay = None if spec.decay is None else format(spec.decay, "f")
    model = {"job_id": spec.job_id, "expert": spec.expert, "profile": "p2"}
    identity = TrainingIdentity.from_payload(
        {
            "data_rows": data_rows,
            "train_seasons": list(range(spec.fold.multi_start, spec.fold.multi_end + 1)) if spec.expert == "multi" else [spec.fold.recent_year],
            "valid_year": spec.fold.valid_year,
            "decay": decay,
            "features": ["base"],
            "model": model,
            "loss": "bce",
            "seed": spec.seed,
        }
    )
    maximum = max(900, load_contract().physical_stage_seconds["T1"] // max(1, len(build_stage_jobs(load_contract(), "T1"))))
    return PlannedJob(
        spec.job_id,
        _STAGE_IDS["T1"],
        "tabm",
        "temporal",
        spec.expert,
        "dl_standard",
        ("base",),
        MappingProxyType(model),
        MappingProxyType({"training_identity_sha256": identity.sha256}),
        spec.fold.multi_end if spec.expert == "multi" else spec.fold.recent_year,
        spec.fold.valid_year,
        spec.seed,
        maximum,
        "full",
        identity,
    )


def _completed(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise PlannerError("completed catalog must be a mapping")
    snapshot = dict(value)
    if any(not _is_sha256(digest) or type(path) is not str or not path for digest, path in snapshot.items()):
        raise PlannerError("completed catalog is invalid")
    return snapshot


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
