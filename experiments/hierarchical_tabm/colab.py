from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

import pandas as pd

from .artifacts import (
    CampaignEvidence,
    verify_resume_bundle,
    write_campaign_bundles,
)
from .contracts import DEFAULT_CONTRACT
from .calibration import apply_calibration, canonical_state_json, select_calibration
from .context_features import select_smoothing_k
from .contracts import HierarchicalJob, load_contract
from .metrics import decide_calibrated, decide_h1, paired_fold_metrics
from .runner import CampaignRun, _restored_result, run_campaign


class HierarchicalColabError(ValueError):
    """Raised when a Colab recovery artifact cannot be trusted."""


@dataclass
class SnapshotCadence:
    snapshot_interval_seconds: float
    download_interval_seconds: float
    started_at: float
    last_snapshot_at: float | None = None
    last_download_at: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.snapshot_interval_seconds,
            self.download_interval_seconds,
            self.started_at,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ) or self.snapshot_interval_seconds <= 0 or self.download_interval_seconds <= 0:
            raise HierarchicalColabError("snapshot cadence is invalid")

    def snapshot_due(self, now: float) -> bool:
        origin = self.started_at if self.last_snapshot_at is None else self.last_snapshot_at
        return float(now) >= origin + self.snapshot_interval_seconds

    def download_due(self, now: float) -> bool:
        origin = self.started_at if self.last_download_at is None else self.last_download_at
        return float(now) >= origin + self.download_interval_seconds

    def mark_snapshot(self, now: float) -> None:
        self.last_snapshot_at = float(now)

    def mark_download(self, now: float) -> None:
        self.last_download_at = float(now)


@dataclass(frozen=True)
class ActiveCheckpoint:
    job_id: str
    job_directory: Path
    checkpoint: Path
    checkpoint_meta: Path
    completed_epochs: int


class Clock(Protocol):
    def time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class _SystemClock:
    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


SYSTEM_CLOCK = _SystemClock()


class SubprocessCampaignRuntime:
    """Production runtime that keeps each GPU training job in one child process."""

    def __init__(
        self,
        *,
        data_dir: Path,
        bindings: Mapping[str, str],
        log_path: Path,
        python_executable: Path | str = sys.executable,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.bindings = MappingProxyType(dict(bindings))
        self.log_path = Path(log_path)
        self.python_executable = str(python_executable)
        self._process: subprocess.Popen[str] | None = None
        self._active_job: HierarchicalJob | None = None
        self._active_directory: Path | None = None
        self._metrics: dict[str, Mapping[str, object]] = {}
        self._contract = load_contract()

    def select_k(self, fit_rows, valid_rows):
        selection = select_smoothing_k(
            fit_rows,
            valid_rows,
            self._contract.k_candidates,
            tie_tolerance=self._contract.k_tie_tolerance,
        )
        print(f"HIER_CONTEXT_SELECTED k={selection.selected_k:g}", flush=True)
        return selection

    def _run_job(self, job: HierarchicalJob, **kwargs: object):
        output = Path(kwargs["output_dir"])
        output.parent.mkdir(parents=True, exist_ok=True)
        job_path = output.parent / f".{job.job_id}.job.json"
        job_path.write_text(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "train_end_year": job.train_end_year,
                    "valid_year": job.valid_year,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        command = [
            self.python_executable,
            "-m", "experiments.hierarchical_tabm.training",
            "--job", str(job_path),
            "--data-dir", str(self.data_dir),
            "--output-dir", str(output),
            "--selected-k", str(kwargs["selected_k"]),
            "--absolute-deadline", str(kwargs["absolute_deadline"]),
            "--identity", json.dumps(dict(self.bindings), sort_keys=True, separators=(",", ":")),
        ]
        if kwargs.get("final_epochs") is not None:
            command.extend(["--final-epochs", str(kwargs["final_epochs"])])
        if kwargs.get("resume_checkpoint") is not None:
            command.extend(["--resume-checkpoint", str(kwargs["resume_checkpoint"])])
        environment = dict(os.environ)
        repository = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = repository + os.pathsep + environment.get("PYTHONPATH", "")
        self._active_job = job
        self._active_directory = output
        fold = (
            f"{job.train_end_year}->{job.valid_year}"
            if job.valid_year is not None else f"2019->{job.train_end_year}"
        )
        if job.kind == "full_fit":
            print(f"HIER_FULL_TRAIN_START fold={fold}", flush=True)
        print(f"HIER_JOB_START fold={fold} job={job.job_id}", flush=True)
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            shell=False,
        )
        try:
            assert self._process.stdout is not None
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log:
                for line in self._process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    if "TRAINING_PROGRESS" in line:
                        print(f"HIER_TRAINING_PROGRESS fold={fold}", flush=True)
            return_code = self._process.wait()
            if return_code != 0 and not (output / "worker_result.json").is_file():
                raise HierarchicalColabError(
                    f"training subprocess failed without result: {job.job_id}"
                )
            result = _restored_result(output, job)
            print(
                f"HIER_JOB_END fold={fold} job={job.job_id} status={result.status}",
                flush=True,
            )
            return result
        finally:
            self._process = None
            self._active_job = None
            self._active_directory = None
            job_path.unlink(missing_ok=True)

    def run_oof(self, job: HierarchicalJob, **kwargs: object):
        return self._run_job(job, **kwargs)

    def run_full_fit(self, job: HierarchicalJob, **kwargs: object):
        return self._run_job(job, **kwargs)

    def calibrate(self, kind: str, **kwargs: object):
        frame = kwargs["oof_by_fold"]["2022->2023"].rename(
            columns={"target": "control_success"}
        )
        selection = select_calibration(
            frame,
            kind=kind,
            grid=self._contract.calibration_grid,
            fit_month_max=7,
            validation_month_min=8,
        )
        Path(kwargs["output_path"]).write_bytes(canonical_state_json(selection.state))
        return selection

    def decide(self, candidate_id: str, **kwargs: object):
        oof = kwargs["oof_by_fold"]
        anchors = {
            fold: pd.read_csv(path) if isinstance(path, (str, Path)) else path
            for fold, path in kwargs["anchor_predictions"].items()
        }
        if candidate_id == "H1":
            metrics = paired_fold_metrics(
                anchors, oof, segment_min_rows=self._contract.segment_min_rows
            )
            self._metrics["H1"] = metrics
            decision = decide_h1(metrics, self._contract)
            print(
                f"HIER_DECISION candidate=H1 status={decision.status}", flush=True
            )
            return decision
        selection = kwargs["selections"][candidate_id]
        calibrated = {}
        for fold, frame in oof.items():
            copy = frame.copy()
            copy["probability"] = apply_calibration(
                copy["probability"].to_numpy(dtype="float64"), copy, selection.state
            )
            calibrated[fold] = copy
        metrics = paired_fold_metrics(
            anchors, calibrated, segment_min_rows=self._contract.segment_min_rows
        )
        self._metrics[candidate_id] = metrics
        decision = decide_calibrated(
            candidate_id,
            self._metrics["H1"],
            metrics,
            self._contract,
            h2_metrics=self._metrics.get("H2"),
        )
        print(
            f"HIER_DECISION candidate={candidate_id} status={decision.status}",
            flush=True,
        )
        return decision

    def active_checkpoint(self) -> ActiveCheckpoint | None:
        job = self._active_job
        directory = self._active_directory
        if job is None or directory is None:
            return None
        checkpoint = directory / "checkpoint.pt"
        metadata = directory / "checkpoint_meta.json"
        if not checkpoint.is_file() or not metadata.is_file():
            return None
        payload = _read_metadata(metadata)
        epoch = payload.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            return None
        return ActiveCheckpoint(job.job_id, directory, checkpoint, metadata, epoch + 1)

    def terminate(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _file_sha256(path: Path, check_deadline: Callable[[], None]) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check_deadline()
            digest.update(chunk)
    return digest.hexdigest()


def _read_metadata(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HierarchicalColabError("active checkpoint metadata is unreadable") from error
    if type(value) is not dict:
        raise HierarchicalColabError("active checkpoint metadata is invalid")
    return value


def _checkpoint_child(connection, checkpoint: str, job_id: str, epoch: int) -> None:
    try:
        import numpy as np
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "candidate_id", "epoch", "best_epoch", "best_brier",
            "validation_curve", "validation_time_curve", "elapsed_seconds",
            "model", "optimizer", "scheduler", "scaler", "python_rng",
            "numpy_rng", "torch_rng", "cuda_rng", "adapter_state",
        }
        if (
            type(payload) is not dict
            or set(payload) != expected
            or payload["candidate_id"] != job_id
            or payload["epoch"] != epoch
            or not isinstance(payload["model"], Mapping)
            or not payload["model"]
            or "model.output.weight" not in payload["model"]
        ):
            raise HierarchicalColabError("active checkpoint payload identity differs")
        for name, tensor in payload["model"].items():
            if not isinstance(name, str) or not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                raise HierarchicalColabError("active checkpoint model state is invalid")
        optimizer = payload["optimizer"]
        if (
            type(optimizer) is not dict
            or set(optimizer) != {"state", "param_groups"}
            or not optimizer["state"]
            or not optimizer["param_groups"]
        ):
            raise HierarchicalColabError("active checkpoint optimizer state is invalid")
        if type(payload["scheduler"]) is not dict or type(payload["scaler"]) is not dict:
            raise HierarchicalColabError("active checkpoint scheduler state is invalid")
        if not isinstance(payload["python_rng"], tuple) or not isinstance(payload["numpy_rng"], tuple):
            raise HierarchicalColabError("active checkpoint RNG state is invalid")
        if not torch.is_tensor(payload["torch_rng"]) or payload["torch_rng"].dtype != torch.uint8:
            raise HierarchicalColabError("active checkpoint torch RNG state is invalid")
        if not isinstance(payload["cuda_rng"], list) or any(
            not torch.is_tensor(value) or value.dtype != torch.uint8
            for value in payload["cuda_rng"]
        ):
            raise HierarchicalColabError("active checkpoint CUDA RNG state is invalid")
        np.random.RandomState().set_state(payload["numpy_rng"])
        random_state = payload["python_rng"]
        import random
        random.Random().setstate(random_state)
        connection.send((True, None))
    except BaseException as error:
        connection.send((False, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _validate_checkpoint_isolated(
    checkpoint: Path,
    *,
    job_id: str,
    epoch: int,
    check_deadline: Callable[[], None],
) -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_checkpoint_child,
        args=(child, str(checkpoint), job_id, epoch),
    )
    process.start()
    child.close()
    try:
        while process.is_alive():
            check_deadline()
            process.join(0.1)
        if process.exitcode != 0 or not parent.poll():
            raise HierarchicalColabError("active checkpoint validation child failed")
        accepted, detail = parent.recv()
        if accepted is not True:
            raise HierarchicalColabError(f"active checkpoint payload is invalid: {detail}")
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join(0.5)
            if process.is_alive():
                process.kill()
                process.join(0.5)
        raise
    finally:
        parent.close()


def _stable_copy(
    source: Path, destination: Path, check_deadline: Callable[[], None]
) -> str:
    before = source.stat()
    expected = _file_sha256(source, check_deadline)
    digest = sha256()
    with source.open("rb") as read, destination.open("xb") as write:
        while chunk := read.read(1024 * 1024):
            check_deadline()
            write.write(chunk)
            digest.update(chunk)
    after = source.stat()
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or digest.hexdigest() != expected
    ):
        destination.unlink(missing_ok=True)
        raise HierarchicalColabError("active checkpoint changed during copy")
    return expected


def promote_active_checkpoint(
    active: ActiveCheckpoint,
    evidence: CampaignEvidence,
    snapshot_dir: Path,
    *,
    check_deadline: Callable[[], None],
) -> Path:
    check_deadline()
    if (
        not isinstance(active.job_id, str)
        or not active.job_id
        or isinstance(active.completed_epochs, bool)
        or not isinstance(active.completed_epochs, int)
        or active.completed_epochs <= 0
    ):
        raise HierarchicalColabError("active checkpoint identity is invalid")
    job_root = Path(active.job_directory)
    checkpoint = Path(active.checkpoint)
    metadata_path = Path(active.checkpoint_meta)
    for path in (checkpoint, metadata_path):
        if (
            path.is_symlink()
            or not path.is_file()
            or not stat.S_ISREG(path.stat().st_mode)
            or job_root not in path.resolve().parents
        ):
            raise HierarchicalColabError("active checkpoint source is invalid")
    metadata = _read_metadata(metadata_path)
    expected_meta = {
        "candidate_id", "epoch", "checkpoint", "adapter_state",
        "checkpoint_binding",
    }
    epoch = active.completed_epochs - 1
    if (
        set(metadata) != expected_meta
        or metadata["candidate_id"] != active.job_id
        or metadata["epoch"] != epoch
        or metadata["checkpoint"] != checkpoint.name
        or metadata["checkpoint_binding"] != dict(evidence.bindings)
    ):
        raise HierarchicalColabError("active checkpoint binding differs")
    _validate_checkpoint_isolated(
        checkpoint,
        job_id=active.job_id,
        epoch=epoch,
        check_deadline=check_deadline,
    )
    root = Path(snapshot_dir)
    root.mkdir(parents=True, exist_ok=True)
    sequence = len(tuple(root.glob("snapshot_*")))
    target_root = root / f"snapshot_{sequence:04d}"
    if target_root.exists() or target_root.is_symlink():
        raise HierarchicalColabError("snapshot destination already exists")
    staging = root / f".snapshot_{sequence:04d}.tmp"
    if staging.exists() or staging.is_symlink():
        raise HierarchicalColabError("snapshot staging destination already exists")
    staging.mkdir()
    try:
        active_copy = staging / "active"
        active_copy.mkdir()
        copied_checkpoint = active_copy / "checkpoint.pt"
        checkpoint_sha = _stable_copy(checkpoint, copied_checkpoint, check_deadline)
        copied_meta = active_copy / "checkpoint_meta.json"
        _stable_copy(metadata_path, copied_meta, check_deadline)
        (active_copy / "checkpoint.pt.meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": active.job_id,
                    "identity": dict(evidence.bindings),
                    "checkpoint_sha256": checkpoint_sha,
                    "completed_epochs": active.completed_epochs,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(staging, target_root)
        promoted = CampaignEvidence(
            evidence.bindings,
            evidence.contract_path,
            evidence.log_path,
            evidence.state_path,
            evidence.k_selection_path,
            evidence.completed_job_directories,
            evidence.calibration_paths,
            evidence.decision_paths,
            target_root / "active",
        )
        bundles = write_campaign_bundles(
            promoted, target_root / "bundles", check_deadline=check_deadline
        )
        return bundles.resume
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(target_root, ignore_errors=True)
        raise


def _campaign_evidence(
    output_dir: Path,
    bindings: Mapping[str, str],
) -> CampaignEvidence:
    output = Path(output_dir)
    state_path = output / "stage_state.json"
    state = _read_metadata(state_path)
    completed = state.get("completed_job_ids")
    if type(completed) is not list or any(not isinstance(value, str) for value in completed):
        raise HierarchicalColabError("campaign progress state is invalid")
    jobs = {
        job_id: output / "jobs" / job_id
        for job_id in completed
    }
    calibration = {
        path.stem: path
        for path in sorted((output / "calibration").glob("*.json"))
    } if (output / "calibration").is_dir() else {}
    decisions = {
        path.stem: path
        for path in sorted((output / "decisions").glob("*.json"))
    } if (output / "decisions").is_dir() else {}
    return CampaignEvidence(
        MappingProxyType(dict(bindings)),
        DEFAULT_CONTRACT,
        output / "campaign.log",
        state_path,
        output / "k_selection.json",
        MappingProxyType(jobs),
        MappingProxyType(calibration),
        MappingProxyType(decisions),
        None,
    )


def run_supervised_campaign(
    verified: object,
    *,
    output_dir: Path,
    snapshot_dir: Path,
    resume_bundle: Path | None,
    wall_deadline: float,
    on_verified_resume: Callable[[Path], None],
    log_path: Path,
    runtime: object,
    snapshot_interval_seconds: float = 300.0,
    download_interval_seconds: float = 1200.0,
    clock: Clock = SYSTEM_CLOCK,
) -> CampaignRun:
    """Run one campaign and expose only recursively verified recovery ZIPs."""
    output = Path(output_dir)
    snapshots = Path(snapshot_dir)
    external_log = Path(log_path)
    if snapshots.exists() or snapshots.is_symlink():
        raise HierarchicalColabError("snapshot directory already exists")
    if external_log.exists() or external_log.is_symlink():
        raise HierarchicalColabError("supervisor log already exists")
    snapshots.mkdir(parents=True)
    external_log.parent.mkdir(parents=True, exist_ok=True)
    external_log.write_text("HIERARCHICAL_SUPERVISOR_START\n", encoding="utf-8")
    cadence = SnapshotCadence(
        snapshot_interval_seconds,
        download_interval_seconds,
        started_at=clock.time(),
    )
    lock = threading.Lock()
    latest: Path | None = None
    sequence = 0
    observed_checkpoint: tuple[str, int] | None = None
    monitor_error: BaseException | None = None
    stop = threading.Event()

    def check_deadline() -> None:
        if clock.time() >= wall_deadline:
            raise HierarchicalColabError("campaign wall deadline reached")

    def deliver(path: Path) -> None:
        nonlocal latest
        verify_resume_bundle(path, expected_bindings=runtime.bindings)
        on_verified_resume(path)
        latest = path
        cadence.mark_download(clock.time())

    def publish_stable(_: Path | None = None) -> Path:
        nonlocal latest, sequence
        with lock:
            check_deadline()
            evidence = _campaign_evidence(output, runtime.bindings)
            state = _read_metadata(evidence.state_path)
            if state.get("active_job_id") is not None:
                if latest is None:
                    raise HierarchicalColabError("active job has no verified recovery snapshot")
                return latest
            target = snapshots / f"stable_{sequence:04d}"
            sequence += 1
            bundles = write_campaign_bundles(
                evidence, target, check_deadline=check_deadline
            )
            verify_resume_bundle(
                bundles.resume,
                expected_bindings=runtime.bindings,
            )
            deliver(bundles.resume)
            cadence.mark_snapshot(clock.time())
            return bundles.resume

    def accept_uploaded(path: Path) -> None:
        check_deadline()
        verify_resume_bundle(
            Path(path),
            expected_bindings=runtime.bindings,
        )
        on_verified_resume(Path(path))

    def monitor() -> None:
        nonlocal latest, observed_checkpoint, monitor_error, sequence
        try:
            while not stop.is_set():
                check_deadline()
                active_method = getattr(runtime, "active_checkpoint", None)
                active = active_method() if callable(active_method) else None
                if active is not None:
                    identity = (active.job_id, active.completed_epochs)
                    if identity != observed_checkpoint:
                        with lock:
                            evidence = _campaign_evidence(output, runtime.bindings)
                            target = snapshots / f"active_{sequence:04d}"
                            sequence += 1
                            resume = promote_active_checkpoint(
                                active, evidence, target,
                                check_deadline=check_deadline,
                            )
                            deliver(resume)
                            cadence.mark_snapshot(clock.time())
                            observed_checkpoint = identity
                now = clock.time()
                if cadence.snapshot_due(now) or cadence.download_due(now):
                    with lock:
                        if latest is not None:
                            verify_resume_bundle(
                                latest, expected_bindings=runtime.bindings
                            )
                            on_verified_resume(latest)
                            cadence.mark_snapshot(now)
                            cadence.mark_download(now)
                delay = min(5.0, max(0.05, wall_deadline - clock.time()))
                if clock is SYSTEM_CLOCK:
                    stop.wait(delay)
                else:
                    clock.sleep(delay)
        except BaseException as error:
            monitor_error = error
            terminate = getattr(runtime, "terminate", None)
            if callable(terminate):
                terminate()

    try:
        monitor_thread = threading.Thread(
            target=monitor, name="hierarchical-tabm-monitor", daemon=True
        )
        monitor_thread.start()
        result = run_campaign(
            verified,
            output,
            resume_bundle=resume_bundle,
            absolute_deadline=wall_deadline,
            runtime=runtime,
            on_verified_resume=accept_uploaded,
            on_progress_state=publish_stable,
        )
        stop.set()
        monitor_thread.join(5.0)
        if monitor_thread.is_alive():
            raise HierarchicalColabError("campaign monitor did not stop")
        if monitor_error is not None:
            raise monitor_error
        publish_stable()
        external_log.write_text(
            external_log.read_text(encoding="utf-8")
            + f"HIERARCHICAL_SUPERVISOR_SUCCESS status={result.state.status}\n",
            encoding="utf-8",
        )
        return result
    except Exception as error:
        stop.set()
        thread = locals().get("monitor_thread")
        if isinstance(thread, threading.Thread):
            thread.join(1.0)
        terminate = getattr(runtime, "terminate", None)
        if callable(terminate):
            terminate()
        with external_log.open("a", encoding="utf-8") as stream:
            stream.write(
                f"HIERARCHICAL_SUPERVISOR_ERROR type={type(error).__name__} "
                f"message={str(error).replace(' ', '_')}\n"
            )
        raise
