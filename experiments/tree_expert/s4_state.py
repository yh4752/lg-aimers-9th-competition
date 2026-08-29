from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class S4StateError(ValueError):
    pass


_PHASES = ("anchors", "residuals", "full_chains", "confirmation", "full_fit", "completed")


@dataclass(frozen=True)
class S4State:
    phase: str
    completed_jobs: tuple[str, ...]
    failed_jobs: tuple[str, ...]
    decisions: Mapping[str, str]


def initial_s4_state() -> S4State:
    return S4State("anchors", (), (), MappingProxyType({}))


def _mark(state: S4State, job_id: str, status: str) -> S4State:
    if type(state) is not S4State or type(job_id) is not str or not job_id or status not in {"completed", "failed"}:
        raise S4StateError("job state differs")
    completed, failed = set(state.completed_jobs), set(state.failed_jobs)
    completed.discard(job_id)
    failed.discard(job_id)
    (completed if status == "completed" else failed).add(job_id)
    return replace(state, completed_jobs=tuple(sorted(completed)), failed_jobs=tuple(sorted(failed)))


def mark_completed(state: S4State, job_id: str) -> S4State:
    return _mark(state, job_id, "completed")


def mark_failed(state: S4State, job_id: str) -> S4State:
    return _mark(state, job_id, "failed")


def record_s4_decision(state: S4State, candidate_id: str, status: str) -> S4State:
    if type(candidate_id) is not str or not candidate_id or status not in {"research_only", "accepted", "rejected"}:
        raise S4StateError("decision state differs")
    decisions = dict(state.decisions)
    decisions[candidate_id] = status
    return replace(state, decisions=MappingProxyType(dict(sorted(decisions.items()))))


def advance_s4_phase(state: S4State, phase: str) -> S4State:
    try:
        index = _PHASES.index(state.phase)
    except ValueError as error:
        raise S4StateError("campaign phase differs") from error
    if index + 1 >= len(_PHASES) or _PHASES[index + 1] != phase:
        raise S4StateError("campaign phase transition differs")
    return replace(state, phase=phase)


def s4_state_payload(state: S4State) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": state.phase,
        "completed_jobs": list(state.completed_jobs),
        "failed_jobs": list(state.failed_jobs),
        "decisions": dict(state.decisions),
    }


def _from_payload(payload: object) -> S4State:
    if type(payload) is not dict or set(payload) != {"schema_version", "phase", "completed_jobs", "failed_jobs", "decisions"}:
        raise S4StateError("campaign state schema differs")
    if payload["schema_version"] != 1 or payload["phase"] not in _PHASES:
        raise S4StateError("campaign state identity differs")
    if type(payload["completed_jobs"]) is not list or type(payload["failed_jobs"]) is not list or type(payload["decisions"]) is not dict:
        raise S4StateError("campaign state values differ")
    completed = tuple(sorted(str(value) for value in payload["completed_jobs"]))
    failed = tuple(sorted(str(value) for value in payload["failed_jobs"]))
    decisions = {str(key): str(value) for key, value in payload["decisions"].items()}
    if set(completed) & set(failed) or any(value not in {"research_only", "accepted", "rejected"} for value in decisions.values()):
        raise S4StateError("campaign terminal state differs")
    return S4State(str(payload["phase"]), completed, failed, MappingProxyType(dict(sorted(decisions.items()))))


def save_s4_state(state: S4State, path: Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(s4_state_payload(state), sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(temporary, output)
    return output


def load_s4_state(path: Path) -> S4State:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S4StateError("campaign state cannot be loaded") from error
    return _from_payload(payload)
