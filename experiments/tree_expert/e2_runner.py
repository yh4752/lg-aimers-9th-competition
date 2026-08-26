from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence


class E2RunnerError(ValueError):
    """Raised when E2 state or a phase transition differs from the contract."""


PHASES = ("B0", "B1", "B2", "B3", "ACCEPTANCE", "FULL_FIT", "AUDIT", "TERMINAL")
TERMINAL_STATUSES = {
    "accepted",
    "rejected_structure",
    "rejected_seed_instability",
    "rejected_acceptance",
    "budget_inconclusive",
    "failed",
}
_BINDING_KEYS = {
    "contract_sha256",
    "code_sha256",
    "input_sha256",
    "train_sha256",
    "history_sha256",
}
_NEXT = {
    "B0": "B1",
    "B1": "B2",
    "B2": "B3",
    "B3": "ACCEPTANCE",
    "ACCEPTANCE": "FULL_FIT",
    "FULL_FIT": "AUDIT",
    "AUDIT": "TERMINAL",
}


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _bindings(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _BINDING_KEYS:
        raise E2RunnerError("campaign binding keys differ")
    output = dict(value)
    if any(not _valid_sha(item) for item in output.values()):
        raise E2RunnerError("campaign binding SHA-256 differs")
    return output


def _strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(type(item) is not str or not item for item in result) or len(set(result)) != len(result):
        raise E2RunnerError(f"campaign {label} differs")
    return result


def _merge(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*left, *right)))


@dataclass(frozen=True)
class CampaignState:
    schema_version: int
    campaign_id: str
    status: str
    phase: str
    bindings: Mapping[str, str]
    completed: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[str, ...]
    active: Mapping[str, object]
    decisions: Mapping[str, object]
    artifact_paths: Mapping[str, str]

    @classmethod
    def initial(cls, bindings: Mapping[str, str]) -> "CampaignState":
        return cls(
            schema_version=1,
            campaign_id="tree_expert_e2_v1",
            status="running",
            phase="B0",
            bindings=MappingProxyType(_bindings(bindings)),
            completed=(),
            skipped=(),
            failed=(),
            active=MappingProxyType({}),
            decisions=MappingProxyType({}),
            artifact_paths=MappingProxyType({}),
        )

    def with_update(self, **changes: object) -> "CampaignState":
        allowed = {
            "status",
            "phase",
            "completed",
            "skipped",
            "failed",
            "active",
            "decisions",
            "artifact_paths",
        }
        if not set(changes).issubset(allowed):
            raise E2RunnerError("campaign state update keys differ")
        normalized = dict(changes)
        for name in ("completed", "skipped", "failed"):
            if name in normalized:
                normalized[name] = _strings(normalized[name], name)
        for name in ("active", "decisions", "artifact_paths"):
            if name in normalized:
                if not isinstance(normalized[name], Mapping):
                    raise E2RunnerError(f"campaign {name} differs")
                normalized[name] = MappingProxyType(dict(normalized[name]))
        updated = replace(self, **normalized)
        _validate_state(updated)
        return updated

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "status": self.status,
            "phase": self.phase,
            "bindings": dict(self.bindings),
            "completed": list(self.completed),
            "skipped": list(self.skipped),
            "failed": list(self.failed),
            "active": dict(self.active),
            "decisions": dict(self.decisions),
            "artifact_paths": dict(self.artifact_paths),
        }

    def write(self, path: Path) -> Path:
        _validate_state(self)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, output)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        return output


def _validate_state(state: CampaignState) -> None:
    if (
        type(state) is not CampaignState
        or state.schema_version != 1
        or state.campaign_id != "tree_expert_e2_v1"
        or state.phase not in PHASES
        or (state.phase == "TERMINAL") != (state.status in TERMINAL_STATUSES)
        or (state.phase != "TERMINAL" and state.status != "running")
        or dict(state.bindings) != _bindings(state.bindings)
    ):
        raise E2RunnerError("campaign state identity differs")
    completed = _strings(state.completed, "completed jobs")
    skipped = _strings(state.skipped, "skipped jobs")
    failed = _strings(state.failed, "failed jobs")
    if set(completed) & set(skipped) or set(completed) & set(failed) or set(skipped) & set(failed):
        raise E2RunnerError("campaign job status sets overlap")
    if not isinstance(state.active, Mapping) or not isinstance(state.decisions, Mapping):
        raise E2RunnerError("campaign state mappings differ")
    if not isinstance(state.artifact_paths, Mapping) or any(
        type(key) is not str or type(value) is not str
        for key, value in state.artifact_paths.items()
    ):
        raise E2RunnerError("campaign artifact paths differ")


def load_campaign_state(path: Path, expected_bindings: Mapping[str, str]) -> CampaignState:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise E2RunnerError(f"cannot read campaign state: {error}") from error
    expected = {
        "schema_version",
        "campaign_id",
        "status",
        "phase",
        "bindings",
        "completed",
        "skipped",
        "failed",
        "active",
        "decisions",
        "artifact_paths",
    }
    if type(value) is not dict or set(value) != expected:
        raise E2RunnerError("campaign state keys differ")
    if value["bindings"] != _bindings(expected_bindings):
        raise E2RunnerError("campaign state binding differs")
    state = CampaignState(
        schema_version=value["schema_version"],
        campaign_id=value["campaign_id"],
        status=value["status"],
        phase=value["phase"],
        bindings=MappingProxyType(dict(value["bindings"])),
        completed=_strings(value["completed"], "completed jobs"),
        skipped=_strings(value["skipped"], "skipped jobs"),
        failed=_strings(value["failed"], "failed jobs"),
        active=MappingProxyType(dict(value["active"])),
        decisions=MappingProxyType(dict(value["decisions"])),
        artifact_paths=MappingProxyType(dict(value["artifact_paths"])),
    )
    _validate_state(state)
    return state


@dataclass(frozen=True)
class PhaseOutcome:
    status: str
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    decisions: Mapping[str, object]
    artifact_paths: Mapping[str, str]
    skipped: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"passed", "rejected", "failed", "budget_inconclusive"}:
            raise E2RunnerError("phase outcome status differs")
        _strings(self.completed, "phase completed jobs")
        _strings(self.failed, "phase failed jobs")
        _strings(self.skipped, "phase skipped jobs")
        if not isinstance(self.decisions, Mapping) or not isinstance(self.artifact_paths, Mapping):
            raise E2RunnerError("phase outcome mappings differ")


class CampaignRuntime(Protocol):
    def run_phase(
        self,
        phase: str,
        state: CampaignState,
        *,
        gpu_ids: tuple[int, int],
        wall_deadline: float,
    ) -> PhaseOutcome: ...

    def publish(self, state: CampaignState, output_dir: Path) -> object: ...


@dataclass(frozen=True)
class CampaignResult:
    status: str
    state: CampaignState
    bundles: object


@dataclass(frozen=True)
class GpuJob:
    job_id: str
    payload: object


def _run_gpu_worker(worker: Callable[[object], object], payload: object, gpu_id: int) -> object:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return worker(payload)


def run_two_gpu_jobs(
    worker: Callable[[object], object],
    jobs: Sequence[GpuJob],
) -> Mapping[str, object]:
    pending = tuple(jobs)
    if any(type(job) is not GpuJob for job in pending) or len({job.job_id for job in pending}) != len(pending):
        raise E2RunnerError("GPU jobs differ")
    context = multiprocessing.get_context("spawn")
    results: dict[str, object] = {}
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        active: dict[object, str] = {}
        for index in range(0, len(pending), 2):
            batch = pending[index : index + 2]
            for gpu_id, job in enumerate(batch):
                future = pool.submit(_run_gpu_worker, worker, job.payload, gpu_id)
                active[future] = job.job_id
            for future in tuple(active):
                results[active.pop(future)] = future.result()
    return MappingProxyType(results)


def _terminal_status(phase: str, outcome: str) -> str | None:
    if outcome == "failed":
        return "failed"
    if outcome == "budget_inconclusive":
        return "budget_inconclusive"
    if outcome == "rejected":
        if phase == "B1":
            return "rejected_structure"
        if phase == "B2":
            return "rejected_seed_instability"
        if phase == "ACCEPTANCE":
            return "rejected_acceptance"
        if phase != "B3":
            return "failed"
    return None


def run_e2_campaign(
    *,
    runtime: CampaignRuntime,
    output_dir: Path,
    bindings: Mapping[str, str],
    wall_deadline: float,
    state_path: Path | None = None,
    clock: Callable[[], float] = time.time,
    new_job_guard_seconds: int = 600,
) -> CampaignResult:
    if type(new_job_guard_seconds) is not int or new_job_guard_seconds < 0:
        raise E2RunnerError("new-job guard differs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_file = output / "stage_state.json" if state_path is None else Path(state_path)
    state = (
        load_campaign_state(state_file, bindings)
        if state_file.is_file()
        else CampaignState.initial(bindings)
    )
    if state.phase == "TERMINAL":
        return CampaignResult(state.status, state, runtime.publish(state, output / "bundles"))

    while state.phase != "TERMINAL":
        if wall_deadline - clock() < new_job_guard_seconds:
            state = state.with_update(status="budget_inconclusive", phase="TERMINAL", active={})
            state.write(state_file)
            break
        phase = state.phase
        try:
            outcome = runtime.run_phase(
                phase,
                state,
                gpu_ids=(0, 1),
                wall_deadline=wall_deadline,
            )
        except Exception as error:
            decisions = dict(state.decisions)
            decisions[phase] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            state = state.with_update(
                status="failed",
                phase="TERMINAL",
                active={},
                decisions=decisions,
            )
            state.write(state_file)
            break

        decisions = dict(state.decisions)
        decisions[phase] = dict(outcome.decisions)
        artifacts = dict(state.artifact_paths)
        artifacts.update(outcome.artifact_paths)
        completed = _merge(state.completed, outcome.completed)
        skipped = _merge(state.skipped, outcome.skipped)
        failed = _merge(state.failed, outcome.failed)
        terminal = _terminal_status(phase, outcome.status)
        if terminal is not None:
            state = state.with_update(
                status=terminal,
                phase="TERMINAL",
                completed=completed,
                skipped=skipped,
                failed=failed,
                active={},
                decisions=decisions,
                artifact_paths=artifacts,
            )
        elif phase == "AUDIT":
            state = state.with_update(
                status="accepted",
                phase="TERMINAL",
                completed=completed,
                skipped=skipped,
                failed=failed,
                active={},
                decisions=decisions,
                artifact_paths=artifacts,
            )
        else:
            state = state.with_update(
                phase=_NEXT[phase],
                completed=completed,
                skipped=skipped,
                failed=failed,
                active={},
                decisions=decisions,
                artifact_paths=artifacts,
            )
        state.write(state_file)

    bundles = runtime.publish(state, output / "bundles")
    return CampaignResult(state.status, state, bundles)
