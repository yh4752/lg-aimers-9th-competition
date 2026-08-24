"""Restartable two-GPU execution for the sealed T1 temporal jobs."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Callable, Mapping, Protocol

import pandas as pd

from .contracts import build_stage_jobs, load_contract
from .inputs import VerifiedOfficialData
from .job_materialization import MaterializedJob, materialize_t1_job
from .t1_artifacts import verify_compact_result
from .worker import WorkerBudgetIncomplete, run_worker, verify_worker_result


class T1RunnerError(RuntimeError):
    """Raised when T1 cannot continue without losing artifact integrity."""


_STOP_NEW_SECONDS = 900
_HANDOFF_RESERVE_SECONDS = 600
_WORKER_SHUTDOWN_GRACE_SECONDS = 120
_POLL_SECONDS = 0.25


class ProcessHandle(Protocol):
    @property
    def returncode(self) -> int | None: ...

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


class Launcher(Protocol):
    def start(
        self,
        materialized: MaterializedJob,
        output_dir: Path,
        *,
        gpu: int,
        deadline: float,
    ) -> ProcessHandle: ...


@dataclass(frozen=True)
class T1StageResult:
    status: str
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    output_root: Path


@dataclass
class _Active:
    job: MaterializedJob
    gpu: int
    output_dir: Path
    hard_deadline: float
    process: ProcessHandle


class _MultiprocessingHandle:
    def __init__(self, process: multiprocessing.Process) -> None:
        self._process = process

    @property
    def returncode(self) -> int | None:
        return self._process.exitcode

    def poll(self) -> int | None:
        return self._process.exitcode

    def terminate(self) -> None:
        self._process.terminate()

    def wait(self, timeout: float | None = None) -> int:
        self._process.join(timeout)
        if self._process.exitcode is None:
            raise TimeoutError("worker did not stop within its shutdown grace")
        return int(self._process.exitcode)

    def kill(self) -> None:
        self._process.kill()


class ForkLauncher:
    """Launch a materialized job with copy-on-write arrays on one visible GPU."""

    def __init__(self, worker_entry=None) -> None:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise T1RunnerError("T1 GPU workers require a Linux fork runtime")
        self._context = multiprocessing.get_context("fork")
        self._worker_entry = _worker_entry if worker_entry is None else worker_entry

    def start(
        self,
        materialized: MaterializedJob,
        output_dir: Path,
        *,
        gpu: int,
        deadline: float,
    ) -> ProcessHandle:
        process = self._context.Process(
            target=self._worker_entry,
            args=(materialized, output_dir, gpu, deadline),
            daemon=False,
        )
        process.start()
        return _MultiprocessingHandle(process)


def run_t1_stage(
    *,
    verified: VerifiedOfficialData,
    output_root: str | Path,
    deadline: float,
    launcher: Launcher | None = None,
    materializer: Callable[..., MaterializedJob] = materialize_t1_job,
    result_verifier: Callable[[str | Path], Mapping[str, object]] = verify_worker_result,
    frame_loader: Callable[[Path], pd.DataFrame] = pd.read_csv,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> T1StageResult:
    """Run at most two T1 workers, retaining resumable partial checkpoints."""

    if type(verified) is not VerifiedOfficialData:
        raise T1RunnerError("verified official data identity is invalid")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise T1RunnerError("deadline must be a Unix timestamp")
    started_at = float(clock())
    if float(deadline) <= started_at:
        raise T1RunnerError("deadline has already passed")
    root = Path(output_root).resolve()
    jobs_root = root / "jobs"
    cache_root = root / "feature_cache"
    jobs_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    train = frame_loader(verified.train)
    history = frame_loader(verified.history)
    specs = list(build_stage_jobs(load_contract(), "T1"))
    data_rows_sha256 = _input_identity(verified)
    stage_deadline = min(
        float(deadline),
        started_at + load_contract().physical_stage_seconds["T1"],
    )
    if launcher is None:
        require_two_t4_gpus()
        runtime: Launcher = ForkLauncher()
    else:
        runtime = launcher
    completed: list[str] = []
    failed: list[str] = []
    pending: list[str] = []
    active: dict[int, _Active] = {}
    last_heartbeat = started_at
    print(
        f"T1_STAGE_START jobs={len(specs)} deadline_unix={stage_deadline:.0f}",
        flush=True,
    )

    try:
        while specs or active:
            now = float(clock())
            for gpu in range(2):
                if gpu in active or not specs:
                    continue
                if stage_deadline - now <= _STOP_NEW_SECONDS + _HANDOFF_RESERVE_SECONDS:
                    break
                spec = specs.pop(0)
                job = materializer(
                    spec,
                    data_rows_sha256=data_rows_sha256,
                    train=train,
                    history=history,
                    cache_root=cache_root,
                )
                output_dir = jobs_root / spec.job_id
                if _reuse_completed(output_dir, job, result_verifier):
                    completed.append(spec.job_id)
                    print(f"T1_JOB_REUSED job={spec.job_id}", flush=True)
                    continue
                hard_deadline = min(
                    now + job.plan.max_seconds,
                    stage_deadline - _HANDOFF_RESERVE_SECONDS,
                )
                worker_deadline = max(
                    now, hard_deadline - _WORKER_SHUTDOWN_GRACE_SECONDS
                )
                print(
                    f"T1_JOB_START job={spec.job_id} gpu={gpu} "
                    f"deadline_unix={worker_deadline:.0f}",
                    flush=True,
                )
                process = runtime.start(
                    job, output_dir, gpu=gpu, deadline=worker_deadline
                )
                active[gpu] = _Active(
                    job, gpu, output_dir, hard_deadline, process
                )

            now = float(clock())
            for gpu, worker in list(active.items()):
                returncode = worker.process.poll()
                deadline_reached = returncode is None and now >= worker.hard_deadline
                if deadline_reached:
                    _stop_worker(worker.process)
                    returncode = worker.process.returncode
                if returncode is None:
                    continue
                job_id = worker.job.training.job_id
                if returncode == 0:
                    _require_completed(worker.output_dir, worker.job, result_verifier)
                    completed.append(job_id)
                    print(f"T1_JOB_COMPLETE job={job_id} gpu={gpu}", flush=True)
                elif returncode == 75 or deadline_reached:
                    pending.append(job_id)
                    print(f"T1_JOB_PENDING job={job_id} gpu={gpu}", flush=True)
                else:
                    failed.append(job_id)
                    print(
                        f"T1_JOB_FAILED job={job_id} gpu={gpu} returncode={returncode}",
                        flush=True,
                    )
                del active[gpu]

            now = float(clock())
            if now - last_heartbeat >= 60:
                print(
                    f"T1_HEARTBEAT completed={len(completed)} active={len(active)} "
                    f"queued={len(specs)} remaining_seconds={max(0, int(stage_deadline - now))}",
                    flush=True,
                )
                last_heartbeat = now
            if specs and not active and (
                stage_deadline - now <= _STOP_NEW_SECONDS + _HANDOFF_RESERVE_SECONDS
            ):
                pending.extend(spec.job_id for spec in specs)
                specs.clear()
            if active:
                sleeper(_POLL_SECONDS)
    except BaseException:
        for worker in active.values():
            if worker.process.poll() is None:
                _stop_worker(worker.process)
        raise

    status = "completed" if not pending and not failed else "budget_inconclusive"
    result = T1StageResult(
        status,
        tuple(completed),
        tuple(pending),
        tuple(failed),
        root,
    )
    _write_result(root / "t1_stage_result.json", result, data_rows_sha256)
    print(
        f"T1_STAGE_RESULT status={status} completed={len(completed)} "
        f"pending={len(pending)} failed={len(failed)}",
        flush=True,
    )
    return result


def _worker_entry(
    materialized: MaterializedJob,
    output_dir: Path,
    gpu: int,
    deadline: float,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PREPROCESSING_SESSION_DEADLINE_UNIX"] = str(deadline)
    try:
        run_worker(materialized.training, output_dir, backend=None)
    except WorkerBudgetIncomplete as error:
        print(f"T1_WORKER_BUDGET_INCOMPLETE job={materialized.training.job_id} {error}", flush=True)
        raise SystemExit(75) from error
    except BaseException:
        traceback.print_exc()
        raise


def _reuse_completed(
    output_dir: Path,
    materialized: MaterializedJob,
    verifier: Callable[[str | Path], Mapping[str, object]],
) -> bool:
    manifest = output_dir / "worker_result.json"
    compact = output_dir / "compact_result.json"
    if manifest.exists() or manifest.is_symlink():
        _require_completed(output_dir, materialized, verifier)
        return True
    if not compact.exists() and not compact.is_symlink():
        return False
    try:
        payload = verify_compact_result(output_dir)
    except Exception as error:
        raise T1RunnerError(
            f"compact worker verification failed: {materialized.training.job_id}"
        ) from error
    _require_identity(payload, materialized)
    return True


def _require_completed(
    output_dir: Path,
    materialized: MaterializedJob,
    verifier: Callable[[str | Path], Mapping[str, object]],
) -> None:
    try:
        payload = verifier(output_dir)
    except Exception as error:
        raise T1RunnerError(
            f"completed worker verification failed: {materialized.training.job_id}"
        ) from error
    _require_identity(payload, materialized)


def _require_identity(
    payload: Mapping[str, object], materialized: MaterializedJob
) -> None:
    if (
        payload.get("status") != "completed"
        or payload.get("job_id") != materialized.training.job_id
        or payload.get("training_identity_sha256")
        != materialized.training.identity.sha256
    ):
        raise T1RunnerError(
            f"completed worker identity differs: {materialized.training.job_id}"
        )


def _stop_worker(process: ProcessHandle) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except (TimeoutError, OSError):
        process.kill()
        process.wait(timeout=10)


def _input_identity(verified: VerifiedOfficialData) -> str:
    payload = {
        "members": dict(sorted(verified.member_sha256.items())),
        "train_rows": verified.train_rows,
        "test_rows": verified.test_rows,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def require_two_t4_gpus(
    probe: Callable[[], tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    names = (probe or _nvidia_gpu_names)()
    if len(names) != 2 or any(name.casefold() != "tesla t4" for name in names):
        observed = ", ".join(names) if names else "none"
        raise T1RunnerError(
            f"exactly two Tesla T4 devices are required; observed={observed}"
        )
    print(f"T1_GPU_READY count=2 names={' | '.join(names)}", flush=True)
    return names


def _nvidia_gpu_names() -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise T1RunnerError("cannot inspect CUDA devices with nvidia-smi") from error
    if result.returncode != 0:
        raise T1RunnerError("cannot inspect CUDA devices with nvidia-smi")
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _write_result(path: Path, result: T1StageResult, data_sha256: str) -> None:
    payload = {
        "schema_version": 1,
        "stage": "T1",
        "status": result.status,
        "data_rows_sha256": data_sha256,
        "completed": list(result.completed),
        "pending": list(result.pending),
        "failed": list(result.failed),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)
