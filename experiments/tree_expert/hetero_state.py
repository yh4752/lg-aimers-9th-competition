from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class HeteroStateError(ValueError):
    pass


@dataclass(frozen=True)
class HeteroState:
    phase: str
    completed_jobs: tuple[str, ...]
    failed_jobs: tuple[str, ...]
    decisions: Mapping[str, str]


def initial_state() -> HeteroState:
    return HeteroState("structure", (), (), MappingProxyType({}))


def mark_job(state: HeteroState, job_id: str, status: str) -> HeteroState:
    if type(state) is not HeteroState or type(job_id) is not str or not job_id:
        raise HeteroStateError("job state identity differs")
    if status not in {"completed", "failed"}:
        raise HeteroStateError("job status differs")
    completed = set(state.completed_jobs)
    failed = set(state.failed_jobs)
    completed.discard(job_id)
    failed.discard(job_id)
    (completed if status == "completed" else failed).add(job_id)
    return replace(state, completed_jobs=tuple(sorted(completed)), failed_jobs=tuple(sorted(failed)))


def record_decision(state: HeteroState, family: str, status: str) -> HeteroState:
    if status not in {"passed", "accepted", "rejected"}:
        raise HeteroStateError("decision status differs")
    decisions = dict(state.decisions)
    decisions[family] = status
    return replace(state, decisions=MappingProxyType(dict(sorted(decisions.items()))))


def advance_phase(state: HeteroState, phase: str) -> HeteroState:
    transitions = {"structure": "confirmation", "confirmation": "completed"}
    if transitions.get(state.phase) != phase:
        raise HeteroStateError("campaign phase transition differs")
    return replace(state, phase=phase)


def state_payload(state: HeteroState) -> dict[str, object]:
    return {
        "phase": state.phase, "completed_jobs": list(state.completed_jobs),
        "failed_jobs": list(state.failed_jobs), "decisions": dict(state.decisions),
    }


def save_state(state: HeteroState, path: Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(state_payload(state), sort_keys=True, separators=(",", ":")))
    os.replace(temporary, output)
    return output


def load_state(path: Path) -> HeteroState:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise HeteroStateError("campaign state cannot be loaded") from error
    if type(payload) is not dict or set(payload) != {"phase", "completed_jobs", "failed_jobs", "decisions"}:
        raise HeteroStateError("campaign state schema differs")
    state = HeteroState(
        str(payload["phase"]), tuple(payload["completed_jobs"]), tuple(payload["failed_jobs"]),
        MappingProxyType(dict(payload["decisions"])),
    )
    if state.phase not in {"structure", "confirmation", "completed"}:
        raise HeteroStateError("campaign phase differs")
    return state
