from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Mapping

from .artifacts import verify_deployment_resume, write_deployment_bundles
from .contracts import DEFAULT_CONTRACT, build_jobs, load_contract
from .inputs import VerifiedDeploymentInputs, deployment_bindings
from .runner import DeploymentRun, code_sha256, run_deployment_campaign
from .training import DeploymentJobResult


class DeploymentColabError(RuntimeError):
    """Raised when the direct-upload deployment run cannot continue safely."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class EmergencyCadence:
    interval_seconds: float
    started_at: float
    last_published_at: float | None = None
    last_snapshot_sha256: str | None = None

    def should_publish(self, *, now: float, snapshot_sha256: str) -> bool:
        boundary = self.started_at if self.last_published_at is None else self.last_published_at
        return now - boundary >= self.interval_seconds and snapshot_sha256 != self.last_snapshot_sha256

    def mark_published(self, *, now: float, snapshot_sha256: str) -> None:
        self.last_published_at = now
        self.last_snapshot_sha256 = snapshot_sha256


class _SubprocessRuntime:
    def __init__(
        self,
        *,
        verified: VerifiedDeploymentInputs,
        output_dir: Path,
        snapshot_dir: Path,
        bindings: Mapping[str, str],
        wall_deadline: float,
        on_verified_resume: Callable[[Path], None],
        log_path: Path,
    ) -> None:
        self.verified = verified
        self.output_dir = Path(output_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.bindings = dict(bindings)
        self.wall_deadline = wall_deadline
        self.on_verified_resume = on_verified_resume
        self.log_path = Path(log_path)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.touch(exist_ok=True)
        self.cadence = EmergencyCadence(1200.0, time.time())
        self.latest: Path | None = None
        self.sequence = 0

    def publish(self, path: Path) -> None:
        digest = file_sha256(path)
        if self.latest is not None and file_sha256(self.latest) == digest:
            return
        target = self.snapshot_dir / f"catboost_deployment_emergency_{digest[:12]}.zip"
        shutil.copyfile(path, target)
        verify_deployment_resume(target, expected_bindings=self.bindings)
        self.latest = target
        print(f"DEPLOY_SNAPSHOT_READY path={target} sha256={digest}", flush=True)
        self.on_verified_resume(target)

    def _completed_directories(self, jobs_root: Path) -> tuple[list[str], dict[str, Path]]:
        completed: list[str] = []
        directories: dict[str, Path] = {}
        for job in build_jobs(load_contract()):
            directory = jobs_root / job.job_id
            try:
                result = json.loads((directory / "worker_result.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            if result.get("job_id") == job.job_id and result.get("status") == "completed":
                completed.append(job.job_id)
                directories[job.job_id] = directory
        return completed, directories

    def _stable_active_copy(self, source: Path, destination: Path) -> str:
        destination.mkdir(parents=True, exist_ok=False)
        for name in ("job.json", "worker.log", "experiment.cbsnapshot"):
            path = source / name
            if path.is_symlink() or not path.is_file():
                raise DeploymentColabError(f"active snapshot source is missing: {name}")
            before = file_sha256(path)
            shutil.copyfile(path, destination / name)
            after = file_sha256(path)
            if before != after or file_sha256(destination / name) != before:
                raise DeploymentColabError(f"active snapshot changed while copying: {name}")
        return file_sha256(destination / "experiment.cbsnapshot")

    def _seal_active(self, job_directory: Path, *, force: bool = False) -> None:
        snapshot = job_directory / "experiment.cbsnapshot"
        if not snapshot.is_file() or snapshot.is_symlink():
            return
        observed = file_sha256(snapshot)
        now = time.time()
        if not force and not self.cadence.should_publish(now=now, snapshot_sha256=observed):
            return
        publish_root = self.snapshot_dir / f"publish_{self.sequence:04d}"
        self.sequence += 1
        active_copy = publish_root / "active" / job_directory.name
        snapshot_sha = self._stable_active_copy(job_directory, active_copy)
        completed, directories = self._completed_directories(job_directory.parent)
        directories[job_directory.name] = active_copy
        decision_path = self.output_dir / "alignment_decision.json"
        decision = None
        selected = None
        decision_sha = None
        if decision_path.is_file():
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            selected = decision.get("selected_tree_count")
            decision_sha = file_sha256(decision_path)
        kind = json.loads((active_copy / "job.json").read_text(encoding="utf-8")).get("kind")
        status = "full_fit_active" if kind == "full_fit" else "alignment_active"
        state_path = publish_root / "stage_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_id": "catboost_deployment_v1",
                    "status": status,
                    "completed_job_ids": completed,
                    "active_job_id": job_directory.name,
                    "decision_sha256": decision_sha,
                    "selected_tree_count": selected,
                    "bindings": self.bindings,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        bundles = write_deployment_bundles(
            output_dir=publish_root / "bundles",
            bindings=self.bindings,
            contract_path=DEFAULT_CONTRACT,
            stage_state_path=state_path,
            campaign_log_path=self.log_path,
            job_directories=directories,
            decision_path=decision_path if decision is not None else None,
        )
        self.publish(bundles.resume)
        self.cadence.mark_published(now=now, snapshot_sha256=snapshot_sha)

    def _result(
        self,
        job_directory: Path,
        job_id: str,
        *,
        kind: str,
        killed_for_deadline: bool,
    ) -> DeploymentJobResult:
        try:
            value = json.loads((job_directory / "worker_result.json").read_text(encoding="utf-8"))
        except Exception as error:
            snapshot = job_directory / "experiment.cbsnapshot"
            if killed_for_deadline:
                return DeploymentJobResult(
                    job_id=job_id,
                    kind=kind,
                    status="budget_inconclusive",
                    train_rows=0,
                    valid_rows=None if kind == "full_fit" else 0,
                    predictions_path=None,
                    model_path=None,
                    preprocessing_path=None,
                    snapshot_path=snapshot if snapshot.is_file() else None,
                    elapsed_seconds=0.0,
                    failure="absolute session deadline reached",
                )
            raise DeploymentColabError("worker exited without a valid result") from error
        if value.get("job_id") != job_id:
            raise DeploymentColabError("worker result identity differs")
        return DeploymentJobResult(
            job_id=job_id,
            kind=value["kind"],
            status=value["status"],
            train_rows=int(value["train_rows"]),
            valid_rows=None if value["valid_rows"] is None else int(value["valid_rows"]),
            predictions_path=None if value["predictions"] is None else job_directory / value["predictions"],
            model_path=None if value["model"] is None else job_directory / value["model"],
            preprocessing_path=None if value["preprocessing"] is None else job_directory / value["preprocessing"],
            snapshot_path=None if value["snapshot"] is None else job_directory / value["snapshot"],
            elapsed_seconds=float(value["elapsed_seconds"]),
            failure=value["failure"],
        )

    def _run(self, *, full_fit: bool, **kwargs: object) -> DeploymentJobResult:
        job = kwargs["job"]
        job_directory = Path(kwargs["output_dir"])
        command = [
            sys.executable,
            "-m",
            "experiments.catboost_deployment.training",
            "--job-id", job.job_id,
            "--data-dir", str(kwargs["data_dir"]),
            "--output-dir", str(job_directory),
            "--contract-sha256", str(kwargs["contract_sha256"]),
            "--input-manifest-sha256", str(kwargs["input_manifest_sha256"]),
            "--code-sha256", str(kwargs["code_sha256"]),
            "--absolute-deadline", str(kwargs["absolute_deadline"]),
        ]
        if full_fit:
            command.extend(
                [
                    "--selected-tree-count", str(kwargs["selected_tree_count"]),
                    "--alignment-decision-sha256", str(kwargs["alignment_decision_sha256"]),
                ]
            )
        environment = dict(os.environ)
        root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=environment,
        )

        def relay() -> None:
            assert process.stdout is not None
            with self.log_path.open("a", encoding="utf-8") as log:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()

        thread = threading.Thread(target=relay, name="catboost-deploy-log", daemon=True)
        thread.start()
        killed = False
        while process.poll() is None:
            time.sleep(5.0)
            try:
                self._seal_active(job_directory)
            except (DeploymentColabError, OSError):
                pass
            if time.time() >= self.wall_deadline:
                killed = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                break
        thread.join(timeout=5)
        if killed:
            try:
                self._seal_active(job_directory, force=True)
            except (DeploymentColabError, OSError):
                pass
        return self._result(
            job_directory,
            job.job_id,
            kind="full_fit" if full_fit else "alignment",
            killed_for_deadline=killed,
        )

    def run_alignment(self, **kwargs: object) -> DeploymentJobResult:
        return self._run(full_fit=False, **kwargs)

    def run_full_fit(self, **kwargs: object) -> DeploymentJobResult:
        return self._run(full_fit=True, **kwargs)


def run_supervised_deployment(
    *,
    verified: VerifiedDeploymentInputs,
    output_dir: Path,
    snapshot_dir: Path,
    resume_bundle: Path | None,
    wall_deadline: float,
    on_verified_resume: Callable[[Path], None],
    log_path: Path,
) -> DeploymentRun:
    bindings = deployment_bindings(verified, expected_code_sha256=code_sha256())
    runtime = _SubprocessRuntime(
        verified=verified,
        output_dir=output_dir,
        snapshot_dir=snapshot_dir,
        bindings=bindings,
        wall_deadline=wall_deadline,
        on_verified_resume=on_verified_resume,
        log_path=log_path,
    )
    return run_deployment_campaign(
        verified=verified,
        output_dir=output_dir,
        resume_bundle=resume_bundle,
        absolute_deadline=wall_deadline,
        on_verified_resume=runtime.publish,
        alignment_runtime=runtime.run_alignment,
        full_runtime=runtime.run_full_fit,
    )
