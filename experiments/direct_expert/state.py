from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path

from .inputs import canonical_json


class DirectExpertStateError(ValueError):
    pass


_PHASES = {
    "stage_a": ("screening", "selection", "completed"),
    "stage_b": ("confirmation", "extra_seeds", "stacking", "decision", "full_fit", "audit", "completed"),
}


@dataclass(frozen=True)
class CampaignState:
    schema_version: int
    stage: str
    phase: str
    completed_jobs: tuple[str, ...]
    failed_jobs: tuple[tuple[str, str], ...]
    decisions: tuple[tuple[str, str], ...]


def initial_state(stage: str) -> CampaignState:
    if stage not in _PHASES:
        raise DirectExpertStateError("campaign stage differs")
    return CampaignState(1, stage, _PHASES[stage][0], (), (), ())


def _validate(state: CampaignState) -> None:
    if type(state) is not CampaignState or state.schema_version != 1 or state.stage not in _PHASES:
        raise DirectExpertStateError("campaign state differs")
    if state.phase not in _PHASES[state.stage]:
        raise DirectExpertStateError("campaign phase differs")
    completed = set(state.completed_jobs)
    failed = {name for name, _ in state.failed_jobs}
    if len(completed) != len(state.completed_jobs) or completed & failed:
        raise DirectExpertStateError("job state overlaps")


def complete_job(state: CampaignState, job_id: str) -> CampaignState:
    _validate(state)
    failed = tuple(item for item in state.failed_jobs if item[0] != job_id)
    completed = tuple(sorted({*state.completed_jobs, job_id}))
    result = replace(state, completed_jobs=completed, failed_jobs=failed)
    _validate(result)
    return result


def fail_job(state: CampaignState, job_id: str, reason: str) -> CampaignState:
    _validate(state)
    if job_id in state.completed_jobs or not reason:
        raise DirectExpertStateError("failed job differs")
    failed = dict(state.failed_jobs)
    failed[job_id] = reason
    result = replace(state, failed_jobs=tuple(sorted(failed.items())))
    _validate(result)
    return result


def retry_job(state: CampaignState, job_id: str) -> CampaignState:
    _validate(state)
    if job_id not in dict(state.failed_jobs):
        raise DirectExpertStateError("retry job is not failed")
    return replace(state, failed_jobs=tuple(item for item in state.failed_jobs if item[0] != job_id))


def transition(state: CampaignState, phase: str) -> CampaignState:
    _validate(state)
    phases = _PHASES[state.stage]
    index = phases.index(state.phase)
    if index + 1 >= len(phases) or phases[index + 1] != phase:
        raise DirectExpertStateError("phase transition differs")
    return replace(state, phase=phase)


def save_state(path: Path, state: CampaignState) -> Path:
    _validate(state)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(canonical_json(asdict(state)))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def load_state(path: Path) -> CampaignState:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        state = CampaignState(
            schema_version=payload["schema_version"],
            stage=payload["stage"],
            phase=payload["phase"],
            completed_jobs=tuple(payload["completed_jobs"]),
            failed_jobs=tuple(tuple(item) for item in payload["failed_jobs"]),
            decisions=tuple(tuple(item) for item in payload["decisions"]),
        )
    except Exception as error:
        raise DirectExpertStateError("campaign state is unreadable") from error
    _validate(state)
    return state
