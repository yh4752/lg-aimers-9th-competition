from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import errno
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Mapping, Protocol

from .s4_artifacts import (
    S4ArtifactError,
    S4Bindings,
    create_s4_handoff,
    estimate_s4_handoff_peak_bytes,
    restore_s4_resume,
)
from .s4_contracts import load_s4_contract
from .s4_state import (
    S4State,
    advance_s4_phase,
    initial_s4_state,
    load_s4_state,
    mark_completed,
    mark_failed,
    save_s4_state,
)


class S4RunnerError(RuntimeError):
    pass


_DISK_RESERVE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class S4Job:
    job_id: str
    phase: str
    payload: Mapping[str, object]


class S4Runtime(Protocol):
    def jobs_for_phase(
        self, phase: str, state: S4State, root: Path
    ) -> tuple[S4Job, ...]: ...

    def run_job(
        self, job: S4Job, job_dir: Path, gpu_id: int, deadline: float
    ) -> str: ...

    def finalize_phase(self, phase: str, state: S4State, root: Path) -> S4State: ...


@dataclass(frozen=True)
class S4CampaignResult:
    status: str
    state: S4State
    handoff: Path
    started_jobs: tuple[str, ...]
    maximum_concurrent_gpu_jobs: int
    completed_full_chain_count: int
    optional_confirmation_started: bool


def _append(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
        handle.flush()
    print(message, flush=True)


def _initialize(root: Path, resume: Path | None, bindings: S4Bindings) -> S4State:
    if resume is None:
        root.mkdir(parents=True, exist_ok=False)
        state = initial_s4_state()
        save_s4_state(state, root / "state/state.json")
        diagnostics = root / "diagnostics"
        diagnostics.mkdir(parents=True)
        (diagnostics / "campaign.json").write_text(
            json.dumps({"campaign": "tree_s4_full_chain_v1"}, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return state
    return load_s4_state(restore_s4_resume(resume, root, bindings) / "state/state.json")


def _next_phase(state: S4State) -> str:
    return {
        "anchors": "residuals",
        "residuals": "full_chains",
        "full_chains": "confirmation",
        "confirmation": "full_fit",
        "full_fit": "completed",
    }[state.phase]


def run_s4_campaign(
    bindings: S4Bindings,
    output_dir: Path,
    *,
    runtime: S4Runtime,
    wall_deadline: float,
    resume_bundle: Path | None = None,
    gpu_ids: tuple[int, int] = (0, 1),
    clock=time.monotonic,
    disk_usage=shutil.disk_usage,
) -> S4CampaignResult:
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise S4RunnerError("S4 requires two distinct GPU workers")
    contract = load_s4_contract()
    root = Path(output_dir)
    state = _initialize(root, resume_bundle, bindings)
    log_path = root / "s4_campaign.log"
    handoff = root.parent / "anchor_residual_hierarchical_handoff.zip"
    started: list[str] = []
    maximum_active = 0
    active = 0
    lock = threading.Lock()
    last_snapshot = clock()
    caught: BaseException | None = None
    final_snapshot_ready = False

    def publish_snapshot(*, final: bool) -> bool:
        include_delivery = (
            (root / "accepted/token.json").is_file()
            and (root / "models").is_dir()
        )
        estimated = estimate_s4_handoff_peak_bytes(
            root, include_delivery=include_delivery, log_path=log_path
        )
        free = int(disk_usage(handoff.parent).free)
        _append(
            log_path,
            f"S4_DISK_STATUS free_bytes={free} estimated_peak_bytes={estimated}",
        )
        if free < estimated + _DISK_RESERVE_BYTES:
            _append(log_path, "S4_SNAPSHOT_SKIPPED reason=insufficient_space")
            if final and not handoff.is_file():
                raise S4ArtifactError("insufficient space for final S4 handoff")
            return False
        try:
            create_s4_handoff(
                root,
                handoff,
                bindings,
                include_delivery=include_delivery,
                log_path=log_path,
            )
        except OSError as error:
            if error.errno not in {errno.ENOSPC, errno.EDQUOT}:
                raise
            _append(log_path, "S4_SNAPSHOT_SKIPPED reason=insufficient_space")
            if final and not handoff.is_file():
                raise S4ArtifactError("insufficient space for final S4 handoff") from error
            return False
        _append(
            log_path,
            f"S4_SNAPSHOT_READY path={handoff} size_bytes={handoff.stat().st_size}",
        )
        return True

    def execute(job: S4Job, gpu_id: int) -> str:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            _append(log_path, f"S4_JOB_START job={job.job_id} phase={job.phase} gpu={gpu_id}")
            status = runtime.run_job(job, root / "jobs" / job.job_id, gpu_id, wall_deadline)
            if status not in {"completed", "failed"}:
                raise S4RunnerError(f"job returned invalid status: {job.job_id}")
            _append(log_path, f"S4_JOB_END job={job.job_id} status={status}")
            return status
        finally:
            with lock:
                active -= 1

    try:
        while state.phase != "completed":
            phase = state.phase
            jobs = runtime.jobs_for_phase(phase, state, root)
            if any(type(job) is not S4Job or job.phase != phase for job in jobs):
                raise S4RunnerError("phase job registry differs")
            if len({job.job_id for job in jobs}) != len(jobs):
                raise S4RunnerError("duplicate phase job identity")
            pending = [
                job for job in jobs
                if job.job_id not in state.completed_jobs and job.job_id not in state.failed_jobs
            ]
            stopped = False
            for offset in range(0, len(pending), 2):
                if clock() + contract.runtime.new_job_guard_seconds >= wall_deadline:
                    stopped = True
                    break
                batch = pending[offset:offset + 2]
                first_error: BaseException | None = None
                with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="s4_gpu") as pool:
                    futures = {}
                    for index, job in enumerate(batch):
                        started.append(job.job_id)
                        futures[pool.submit(execute, job, gpu_ids[index])] = job
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            status = future.result()
                        except BaseException as error:
                            status = "failed"
                            if first_error is None:
                                first_error = error
                        state = mark_completed(state, job.job_id) if status == "completed" else mark_failed(state, job.job_id)
                        save_s4_state(state, root / "state/state.json")
                if clock() - last_snapshot >= contract.runtime.snapshot_interval_seconds:
                    publish_snapshot(final=False)
                    last_snapshot = clock()
                if first_error is not None:
                    raise first_error
            terminal = set(state.completed_jobs) | set(state.failed_jobs)
            if stopped or any(job.job_id not in terminal for job in jobs):
                break
            state = runtime.finalize_phase(phase, state, root)
            if state.phase != phase:
                raise S4RunnerError("runtime changed the campaign phase")
            state = advance_s4_phase(state, _next_phase(state))
            save_s4_state(state, root / "state/state.json")
            _append(log_path, f"S4_PHASE_COMPLETE phase={phase} next={state.phase}")
    except BaseException as error:
        caught = error
        _append(log_path, f"S4_ERROR type={type(error).__name__} message={str(error).replace(' ', '_')}")
    finally:
        final_snapshot_ready = publish_snapshot(final=True)
        if handoff.is_file():
            status = "current" if final_snapshot_ready else "previous"
            _append(log_path, f"S4_HANDOFF_READY path={handoff} status={status}")
    if caught is not None:
        raise caught
    status = "completed" if state.phase == "completed" and final_snapshot_ready else "incomplete"
    if status == "completed":
        _append(log_path, f"S4_SUCCESS status={status} handoff={handoff}")
    return S4CampaignResult(
        status=status,
        state=state,
        handoff=handoff,
        started_jobs=tuple(started),
        maximum_concurrent_gpu_jobs=maximum_active,
        completed_full_chain_count=sum(name.startswith("full_chains") for name in state.completed_jobs),
        optional_confirmation_started=any(name.startswith("confirmation") for name in started),
    )
