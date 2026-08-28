from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping


class HCStateError(ValueError):
    """Raised when campaign state transitions are invalid."""


@dataclass(frozen=True)
class HCBindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str
    e2_handoff_sha256: str


@dataclass(frozen=True)
class CampaignState:
    schema_version: int
    campaign_id: str
    stage: str
    status: str
    bindings: HCBindings
    completed_jobs: tuple[str, ...]
    skipped_jobs: tuple[tuple[str, str], ...]
    failed_jobs: tuple[str, ...]
    active_job: str | None
    decisions: Mapping[str, Mapping[str, object]]
    artifact_paths: Mapping[str, str]


def _validate_sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HCStateError(f"{label} differs")
    return value


def _validate_bindings(bindings: HCBindings) -> HCBindings:
    if type(bindings) is not HCBindings:
        raise HCStateError("bindings differ")
    for name, value in bindings.__dict__.items():
        _validate_sha(value, name)
    return bindings


def initial_state(bindings: HCBindings) -> CampaignState:
    return CampaignState(
        schema_version=1,
        campaign_id="tree_hierarchical_residual_v1",
        stage="H1",
        status="running",
        bindings=_validate_bindings(bindings),
        completed_jobs=(),
        skipped_jobs=(),
        failed_jobs=(),
        active_job=None,
        decisions=MappingProxyType({}),
        artifact_paths=MappingProxyType({}),
    )


def _terminal_jobs(state: CampaignState) -> set[str]:
    return set(state.completed_jobs) | {job for job, _ in state.skipped_jobs}


def start_job(state: CampaignState, job_id: str) -> CampaignState:
    if type(job_id) is not str or not job_id:
        raise HCStateError("job identity differs")
    if state.active_job is not None:
        raise HCStateError("another job is active")
    if job_id in _terminal_jobs(state):
        raise HCStateError("completed or skipped job is immutable")
    failed = tuple(job for job in state.failed_jobs if job != job_id)
    return replace(state, active_job=job_id, failed_jobs=failed)


def complete_job(state: CampaignState, job_id: str) -> CampaignState:
    if state.active_job != job_id or job_id in _terminal_jobs(state):
        raise HCStateError("active job differs")
    return replace(
        state,
        active_job=None,
        completed_jobs=(*state.completed_jobs, job_id),
    )


def fail_job(state: CampaignState, job_id: str) -> CampaignState:
    if state.active_job != job_id or job_id in _terminal_jobs(state):
        raise HCStateError("active job differs")
    failed = state.failed_jobs if job_id in state.failed_jobs else (*state.failed_jobs, job_id)
    return replace(state, active_job=None, failed_jobs=failed)


def skip_job(state: CampaignState, job_id: str, reason: str) -> CampaignState:
    if (
        type(reason) is not str
        or not reason
        or state.active_job is not None
        or job_id in _terminal_jobs(state)
    ):
        raise HCStateError("completed or skipped job is immutable")
    return replace(state, skipped_jobs=(*state.skipped_jobs, (job_id, reason)))


def record_decision(
    state: CampaignState, name: str, payload: Mapping[str, object]
) -> CampaignState:
    if type(name) is not str or not name or not isinstance(payload, Mapping):
        raise HCStateError("decision differs")
    decisions = {key: MappingProxyType(dict(value)) for key, value in state.decisions.items()}
    if name in decisions and dict(decisions[name]) != dict(payload):
        raise HCStateError("decision is immutable")
    decisions[name] = MappingProxyType(dict(payload))
    return replace(state, decisions=MappingProxyType(decisions))


def record_artifact(state: CampaignState, name: str, path: str) -> CampaignState:
    if type(name) is not str or not name or type(path) is not str or not path:
        raise HCStateError("artifact path differs")
    artifacts = dict(state.artifact_paths)
    if name in artifacts and artifacts[name] != path:
        raise HCStateError("artifact path is immutable")
    artifacts[name] = path
    return replace(state, artifact_paths=MappingProxyType(artifacts))


def advance_stage(state: CampaignState, next_stage: str) -> CampaignState:
    expected = {"H1": "H2", "H2": "H3"}.get(state.stage)
    if next_stage != expected or state.active_job is not None:
        raise HCStateError("stage transition differs")
    if next_stage == "H3":
        winner = state.decisions.get("winner")
        if winner is None or winner.get("status") != "accepted" or winner.get("candidate") not in {"C1", "C2"}:
            raise HCStateError("H3 requires an accepted winner")
    return replace(state, stage=next_stage)


def state_payload(state: CampaignState) -> dict[str, object]:
    return {
        "schema_version": state.schema_version,
        "campaign_id": state.campaign_id,
        "stage": state.stage,
        "status": state.status,
        "bindings": dict(state.bindings.__dict__),
        "completed_jobs": list(state.completed_jobs),
        "skipped_jobs": [list(item) for item in state.skipped_jobs],
        "failed_jobs": list(state.failed_jobs),
        "active_job": state.active_job,
        "decisions": {key: dict(value) for key, value in state.decisions.items()},
        "artifact_paths": dict(state.artifact_paths),
    }


def state_from_payload(payload: Mapping[str, object]) -> CampaignState:
    expected = {
        "schema_version",
        "campaign_id",
        "stage",
        "status",
        "bindings",
        "completed_jobs",
        "skipped_jobs",
        "failed_jobs",
        "active_job",
        "decisions",
        "artifact_paths",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise HCStateError("state payload differs")
    bindings_raw = payload["bindings"]
    if type(bindings_raw) is not dict or set(bindings_raw) != set(HCBindings.__dataclass_fields__):
        raise HCStateError("state bindings differ")
    bindings = _validate_bindings(HCBindings(**bindings_raw))
    decisions_raw = payload["decisions"]
    artifacts_raw = payload["artifact_paths"]
    if type(decisions_raw) is not dict or type(artifacts_raw) is not dict:
        raise HCStateError("state maps differ")
    state = CampaignState(
        schema_version=int(payload["schema_version"]),
        campaign_id=str(payload["campaign_id"]),
        stage=str(payload["stage"]),
        status=str(payload["status"]),
        bindings=bindings,
        completed_jobs=tuple(str(value) for value in payload["completed_jobs"]),
        skipped_jobs=tuple((str(item[0]), str(item[1])) for item in payload["skipped_jobs"]),
        failed_jobs=tuple(str(value) for value in payload["failed_jobs"]),
        active_job=None if payload["active_job"] is None else str(payload["active_job"]),
        decisions=MappingProxyType(
            {key: MappingProxyType(dict(value)) for key, value in decisions_raw.items()}
        ),
        artifact_paths=MappingProxyType(
            {str(key): str(value) for key, value in artifacts_raw.items()}
        ),
    )
    if state.schema_version != 1 or state.campaign_id != "tree_hierarchical_residual_v1":
        raise HCStateError("state identity differs")
    return state
