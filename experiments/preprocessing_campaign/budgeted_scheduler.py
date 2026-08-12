"""Two-GPU worker scheduler with a parent-owned, hash-validated manifest."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, Protocol, Sequence

from .budgeted_contracts import BudgetedJob


class ArtifactValidationError(RuntimeError):
    """Raised when a worker attempts to publish an invalid result."""


@dataclass(frozen=True)
class SchedulerSummary:
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]


class WorkerProcess(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...

    def read_available(self) -> tuple[str, ...]: ...

    def terminate(self) -> None: ...


class _SubprocessHandle:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def _read_stdout(self) -> None:
        if self._process.stdout is None:
            return
        for line in self._process.stdout:
            self._lines.put(line.rstrip("\n"))

    def poll(self) -> int | None:
        return self._process.poll()

    def read_available(self) -> tuple[str, ...]:
        lines = []
        while True:
            try:
                lines.append(self._lines.get_nowait())
            except queue.Empty:
                return tuple(lines)

    def terminate(self) -> None:
        self._process.terminate()


ProcessFactory = Callable[
    [BudgetedJob, Path, dict[str, str], float], WorkerProcess
]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _job_payload(job: BudgetedJob) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "stage_id": job.stage_id,
        "family": job.family,
        "profile_id": job.profile_id,
        "setting_id": job.setting_id,
        "preprocessing_profile": job.preprocessing_profile,
        "components": list(job.components),
        "model": dict(job.model),
        "training": dict(job.training),
        "train_end_year": job.train_end_year,
        "valid_year": job.valid_year,
        "seed": job.seed,
        "max_seconds": job.max_seconds,
        "sample_mode": job.sample_mode,
    }


def _default_gpu_probe() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return int(torch.cuda.device_count())


def _default_gpu_status() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return " | ".join(lines) if completed.returncode == 0 and lines else "unavailable"


def _default_process_factory(
    job: BudgetedJob,
    temporary_dir: Path,
    env: dict[str, str],
    deadline: float,
) -> WorkerProcess:
    temporary_dir.mkdir(parents=True, exist_ok=False)
    job_path = temporary_dir / "job.json"
    _atomic_json(job_path, _job_payload(job))
    command = [
        sys.executable,
        "-m",
        "experiments.preprocessing_campaign.run_budgeted_campaign",
        "worker",
        "--job-json",
        str(job_path),
        "--output-dir",
        str(temporary_dir),
        "--deadline-unix",
        str(deadline),
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return _SubprocessHandle(process)


@dataclass
class _ActiveWorker:
    job: BudgetedJob
    gpu: int
    temporary_dir: Path
    deadline: float
    process: WorkerProcess


class BudgetedScheduler:
    """Assign independent jobs to exactly two GPUs and validate every result."""

    def __init__(
        self,
        *,
        process_factory: ProcessFactory | None = None,
        gpu_probe: Callable[[], int] | None = None,
        gpu_status_probe: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        stop_new_jobs_seconds: int = 900,
        archive_reserve_seconds: int = 600,
        heartbeat_seconds: int = 60,
        poll_seconds: float = 0.25,
    ) -> None:
        self.process_factory = process_factory or _default_process_factory
        self.gpu_probe = gpu_probe or _default_gpu_probe
        self.gpu_status_probe = gpu_status_probe or _default_gpu_status
        self.clock = clock
        self.sleeper = sleeper
        self.stop_new_jobs_seconds = stop_new_jobs_seconds
        self.archive_reserve_seconds = archive_reserve_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.poll_seconds = poll_seconds

    def run(
        self,
        jobs: Sequence[BudgetedJob],
        output_root: str | Path,
        *,
        deadline: float,
    ) -> SchedulerSummary:
        gpu_count = self.gpu_probe()
        if gpu_count != 2:
            raise RuntimeError("exactly two CUDA devices are required")
        print(
            f"GPU_READY device_count={gpu_count} status={self.gpu_status_probe()}",
            flush=True,
        )
        root = Path(output_root).resolve()
        workers_root = root / "workers"
        jobs_root = root / "jobs"
        workers_root.mkdir(parents=True, exist_ok=True)
        jobs_root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "campaign_manifest.json"
        manifest = self._read_manifest(manifest_path)
        pending_queue = list(jobs)
        completed: list[str] = []
        failed: list[str] = []
        active: dict[int, _ActiveWorker] = {}
        last_heartbeat = self.clock()

        try:
            while pending_queue or active:
                now = self.clock()
                for gpu in range(2):
                    if gpu in active or not pending_queue:
                        continue
                    if deadline - now <= self.stop_new_jobs_seconds:
                        break
                    job = pending_queue.pop(0)
                    temporary_dir = workers_root / job.job_id
                    if temporary_dir.exists():
                        shutil.rmtree(temporary_dir)
                    worker_deadline = min(
                        now + job.max_seconds,
                        deadline - self.archive_reserve_seconds,
                    )
                    env = dict(os.environ)
                    env.update(
                        {
                            "CUDA_VISIBLE_DEVICES": str(gpu),
                            "PYTHONUNBUFFERED": "1",
                            "PREPROCESSING_SESSION_DEADLINE_UNIX": str(
                                worker_deadline
                            ),
                        }
                    )
                    print(
                        f"JOB_START job={job.job_id} gpu={gpu} "
                        f"deadline_unix={worker_deadline:.0f}",
                        flush=True,
                    )
                    process = self.process_factory(
                        job, temporary_dir, env, worker_deadline
                    )
                    active[gpu] = _ActiveWorker(
                        job, gpu, temporary_dir, worker_deadline, process
                    )

                for gpu, worker in list(active.items()):
                    for line in worker.process.read_available():
                        print(f"WORKER[{gpu}:{worker.job.job_id}] {line}", flush=True)
                    returncode = worker.process.poll()
                    if returncode is None and now >= worker.deadline:
                        worker.process.terminate()
                        returncode = worker.process.poll()
                    if returncode is None:
                        continue
                    for line in worker.process.read_available():
                        print(f"WORKER[{gpu}:{worker.job.job_id}] {line}", flush=True)
                    if returncode == 0:
                        entry = self._validate_and_publish(worker, jobs_root)
                        manifest["jobs"][worker.job.job_id] = entry
                        _atomic_json(manifest_path, manifest)
                        completed.append(worker.job.job_id)
                        print(
                            f"JOB_COMPLETE job={worker.job.job_id} gpu={gpu}",
                            flush=True,
                        )
                    else:
                        failed.append(worker.job.job_id)
                        manifest["jobs"][worker.job.job_id] = {
                            "state": "pending",
                            "returncode": int(returncode),
                        }
                        _atomic_json(manifest_path, manifest)
                    del active[gpu]

                now = self.clock()
                if now - last_heartbeat >= self.heartbeat_seconds:
                    print(
                        f"HEARTBEAT active={len(active)} queued={len(pending_queue)} "
                        f"remaining_seconds={max(0, int(deadline - now))}",
                        flush=True,
                    )
                    print(f"GPU_STATUS {self.gpu_status_probe()}", flush=True)
                    last_heartbeat = now
                if active and self.poll_seconds:
                    self.sleeper(self.poll_seconds)
                if pending_queue and not active and (
                    deadline - self.clock() <= self.stop_new_jobs_seconds
                ):
                    break
        finally:
            for worker in active.values():
                if worker.process.poll() is None:
                    worker.process.terminate()

        return SchedulerSummary(
            tuple(completed),
            tuple(job.job_id for job in pending_queue),
            tuple(failed),
        )

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, object]:
        if not path.is_file():
            return {"schema_version": 1, "jobs": {}}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("jobs"), dict
        ):
            raise ArtifactValidationError("campaign manifest is invalid")
        return payload

    @staticmethod
    def _validate_and_publish(
        worker: _ActiveWorker, jobs_root: Path
    ) -> dict[str, object]:
        result_path = worker.temporary_dir / "worker_result.json"
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactValidationError(
                f"worker result is missing or invalid: {worker.job.job_id}"
            ) from error
        if (
            payload.get("schema_version") != 1
            or payload.get("job_id") != worker.job.job_id
            or payload.get("state") != "completed"
            or not isinstance(payload.get("artifacts"), list)
        ):
            raise ArtifactValidationError("worker result contract is invalid")
        artifacts: list[dict[str, object]] = []
        temporary_root = worker.temporary_dir.resolve()
        for raw in payload["artifacts"]:
            if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
                raise ArtifactValidationError("artifact record is invalid")
            relative = Path(str(raw["path"]))
            candidate = (temporary_root / relative).resolve()
            if relative.is_absolute() or not candidate.is_relative_to(temporary_root):
                raise ArtifactValidationError("artifact path escapes worker root")
            if not candidate.is_file():
                raise ArtifactValidationError(f"artifact is missing: {relative}")
            observed = sha256(candidate.read_bytes()).hexdigest()
            if observed != raw["sha256"]:
                raise ArtifactValidationError(f"artifact sha256 mismatch: {relative}")
            artifacts.append(
                {
                    "path": str(Path("jobs") / worker.job.job_id / relative),
                    "sha256": observed,
                    "size_bytes": candidate.stat().st_size,
                }
            )
        names = {Path(str(item["path"])).name for item in artifacts}
        if not {"metrics.json", "predictions.csv"}.issubset(names):
            raise ArtifactValidationError("metrics and predictions are required")
        destination = jobs_root / worker.job.job_id
        if destination.exists():
            raise ArtifactValidationError(
                f"published job already exists: {worker.job.job_id}"
            )
        os.replace(worker.temporary_dir, destination)
        return {
            "state": "completed",
            "gpu": worker.gpu,
            "artifacts": artifacts,
            "job": _job_payload(worker.job),
        }
