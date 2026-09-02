from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class E3StateError(ValueError):
    pass


@dataclass(frozen=True)
class CampaignState:
    phase: str
    status: str
    completed_jobs: tuple[str, ...]
    failed_jobs: Mapping[str, str]
    recipe_id: str | None


_PHASES = ("screening", "confirmation", "extra_seeds", "decision", "full_fit", "audit", "completed")


def initial_state() -> CampaignState:
    return CampaignState("screening", "running", (), MappingProxyType({}), None)


def complete_job(state: CampaignState, job_id: str) -> CampaignState:
    if type(job_id) is not str or not job_id:
        raise E3StateError("job identity differs")
    completed = tuple(sorted({*state.completed_jobs, job_id}))
    failed = {name: reason for name, reason in state.failed_jobs.items() if name != job_id}
    return CampaignState(state.phase, state.status, completed, MappingProxyType(failed), state.recipe_id)


def fail_job(state: CampaignState, job_id: str, reason: str) -> CampaignState:
    if job_id in state.completed_jobs:
        raise E3StateError("completed job cannot fail")
    if type(job_id) is not str or not job_id or type(reason) is not str or not reason:
        raise E3StateError("failed job evidence differs")
    failed = dict(state.failed_jobs)
    failed[job_id] = reason
    return CampaignState(state.phase, state.status, state.completed_jobs, MappingProxyType(dict(sorted(failed.items()))), state.recipe_id)


def transition(state: CampaignState, phase: str, *, recipe_id: str | None = None) -> CampaignState:
    try:
        current = _PHASES.index(state.phase)
    except ValueError as error:
        raise E3StateError("current phase differs") from error
    if current + 1 >= len(_PHASES) or _PHASES[current + 1] != phase:
        raise E3StateError("phase transition differs")
    active_recipe = recipe_id if recipe_id is not None else state.recipe_id
    if active_recipe is not None and (type(active_recipe) is not str or not active_recipe):
        raise E3StateError("recipe identity differs")
    status = "completed" if phase == "completed" else "running"
    return CampaignState(phase, status, state.completed_jobs, state.failed_jobs, active_recipe)


def _payload(state: CampaignState) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": state.phase,
        "status": state.status,
        "completed_jobs": list(state.completed_jobs),
        "failed_jobs": dict(state.failed_jobs),
        "recipe_id": state.recipe_id,
    }


def save_state(path: Path, state: CampaignState) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.dumps(_payload(state), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise E3StateError("state is not canonicalizable") from error
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def load_state(path: Path) -> CampaignState:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E3StateError("state is unreadable") from error
    if type(payload) is not dict or set(payload) != {
        "schema_version", "phase", "status", "completed_jobs", "failed_jobs", "recipe_id",
    }:
        raise E3StateError("state keys differ")
    if payload["schema_version"] != 1 or payload["phase"] not in _PHASES:
        raise E3StateError("state identity differs")
    completed = payload["completed_jobs"]
    failed = payload["failed_jobs"]
    if (
        type(completed) is not list
        or any(type(item) is not str or not item for item in completed)
        or len(completed) != len(set(completed))
        or type(failed) is not dict
        or any(type(key) is not str or type(value) is not str for key, value in failed.items())
        or set(completed).intersection(failed)
    ):
        raise E3StateError("state job evidence differs")
    return CampaignState(
        payload["phase"], payload["status"], tuple(completed),
        MappingProxyType(dict(sorted(failed.items()))), payload["recipe_id"],
    )

