from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable, Protocol

import pandas as pd

from .artifacts import (
    E1BundlePaths,
    extract_e1_resume,
    file_sha256,
    verify_e1_resume,
    write_e1_bundles,
)
from .contracts import DEFAULT_CONTRACT, E1Contract, E1Job, build_e1_jobs, load_e1_contract
from .inputs import VerifiedE1Input, VerifiedOfficialData
from .kaggle import runtime_member_names
from .metrics import CandidateMetric, decide_e1, evaluate_e1_candidate, skipped_metric
from .training import FoldResult, run_e1_job


class TreeRunnerError(RuntimeError):
    """Raised when the E1 scheduler cannot preserve its campaign contract."""


class Runtime(Protocol):
    def run_job(self, **kwargs: object) -> FoldResult: ...


@dataclass(frozen=True)
class E1CampaignResult:
    status: str
    completed: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[str, ...]
    reused: tuple[str, ...]
    executed: tuple[str, ...]
    bundles: E1BundlePaths


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()


def code_identity_member_names() -> tuple[str, ...]:
    return runtime_member_names()


def _code_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = sha256()
    for name in code_identity_member_names():
        path = root / name
        if not path.is_file():
            raise TreeRunnerError(f"runtime source is absent: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _bindings(
    data: VerifiedOfficialData,
    evidence: VerifiedE1Input,
) -> dict[str, str]:
    contract = load_e1_contract()
    if (
        evidence.stage_c_delivery_sha256 != contract.stage_c_delivery_sha256
        or evidence.stage_c_review_sha256 != contract.stage_c_review_sha256
        or evidence.baseline_fold != "2023->2024"
    ):
        raise TreeRunnerError("verified E1 input identity differs")
    return {
        "contract_sha256": file_sha256(DEFAULT_CONTRACT),
        "code_sha256": _code_sha256(),
        "input_manifest_sha256": evidence.manifest_sha256,
        "train_sha256": data.train_sha256,
        "history_sha256": data.history_sha256,
        "stage_c_delivery_sha256": evidence.stage_c_delivery_sha256,
        "stage_c_review_sha256": evidence.stage_c_review_sha256,
        "baseline_predictions_sha256": file_sha256(evidence.baseline_predictions),
    }


def _default_job(
    *,
    job: E1Job,
    data: VerifiedOfficialData,
    baseline: pd.DataFrame,
    output_dir: Path,
    absolute_deadline: float,
    gpu_id: int,
) -> FoldResult:
    return run_e1_job(
        job=job,
        data=data,
        baseline=baseline,
        output_dir=output_dir,
        absolute_deadline=absolute_deadline,
        gpu_id=gpu_id,
    )


def _runtime_job(runtime: Runtime, **kwargs: object) -> FoldResult:
    return runtime.run_job(**kwargs)


def _terminal_directories(jobs_root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(jobs_root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
        and (
            (path / "worker_result.json").is_file()
            or (path / "experiment.cbsnapshot").is_file()
        )
    ) if jobs_root.is_dir() else ()


def _audits(jobs_root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(jobs_root.glob("*/failure_label_audit.json"))
        if path.is_file()
    )


def _publish(
    *,
    output: Path,
    bindings: dict[str, str],
    state: dict[str, object],
    log_path: Path,
    decision_path: Path | None,
) -> E1BundlePaths:
    state_path = output / "stage_state.json"
    _atomic_bytes(state_path, _canonical_json(state))
    return write_e1_bundles(
        output_dir=output / "bundles",
        contract_path=DEFAULT_CONTRACT,
        bindings=bindings,
        state_path=state_path,
        log_path=log_path,
        job_directories=_terminal_directories(output / "jobs"),
        decision_path=decision_path,
        audit_paths=_audits(output / "jobs"),
    )


def _restore(
    resume: Path,
    output: Path,
    bindings: dict[str, str],
) -> tuple[set[str], set[str]]:
    verified = verify_e1_resume(resume, bindings)
    restored = extract_e1_resume(resume, output / "restored_resume", bindings)
    restored_jobs = restored / "jobs"
    jobs_root = output / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    if restored_jobs.is_dir():
        for source in restored_jobs.iterdir():
            destination = jobs_root / source.name
            if not destination.exists():
                shutil.copytree(source, destination)
    return set(verified.completed), set(verified.skipped)


def _metric_payload(metric: CandidateMetric) -> dict[str, object]:
    return asdict(metric)


def run_e1_campaign(
    *,
    verified_data: VerifiedOfficialData,
    verified_input: VerifiedE1Input,
    output_dir: Path,
    resume_bundle: Path | None,
    absolute_deadline: float,
    gpu_ids: tuple[int, int],
    runtime: Runtime | None = None,
    clock: Callable[[], float] = time.time,
) -> E1CampaignResult:
    if type(gpu_ids) is not tuple or len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise TreeRunnerError("E1 requires two distinct GPU IDs")
    contract = load_e1_contract()
    bindings = _bindings(verified_data, verified_input)
    output = Path(output_dir)
    jobs_root = output / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    log_path = output / "tree_expert_e1.log"
    _append_log(log_path, "TREE_E1_INPUTS_VERIFIED")
    _append_log(log_path, "TREE_E1_GPU_READY device_count=2")
    baseline = pd.read_csv(verified_input.baseline_predictions)
    jobs = build_e1_jobs(contract)
    completed: set[str] = set()
    skipped: set[str] = set()
    reused: set[str] = set()
    executed: set[str] = set()
    failed: set[str] = set()
    if resume_bundle is not None:
        completed, skipped = _restore(Path(resume_bundle), output, bindings)
        reused = completed | skipped

    state: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": "tree_expert_e1_v1",
        "status": "running",
        "completed": sorted(completed),
        "skipped": sorted(skipped),
        "failed": [],
        "active": {},
        "bindings": bindings,
    }
    pending = [job for job in jobs if job.job_id not in completed | skipped]
    active: dict[object, tuple[E1Job, int]] = {}
    free_gpus = list(gpu_ids)
    stopping = False
    executor_class = ThreadPoolExecutor if runtime is not None else ProcessPoolExecutor
    executor_kwargs: dict[str, object] = {"max_workers": 2}
    if runtime is None:
        executor_kwargs["mp_context"] = multiprocessing.get_context("spawn")
    with executor_class(**executor_kwargs) as executor:
        while pending or active:
            while pending and free_gpus and not stopping:
                if clock() + contract.new_job_guard_seconds >= absolute_deadline:
                    stopping = True
                    break
                job = pending.pop(0)
                gpu_id = free_gpus.pop(0)
                job_output = jobs_root / job.job_id
                kwargs = {
                    "job": job,
                    "data": verified_data,
                    "baseline": baseline,
                    "output_dir": job_output,
                    "absolute_deadline": absolute_deadline,
                    "gpu_id": gpu_id,
                }
                future = executor.submit(
                    _runtime_job if runtime is not None else _default_job,
                    runtime,
                    **kwargs,
                ) if runtime is not None else executor.submit(_default_job, **kwargs)
                active[future] = (job, gpu_id)
                state["active"][job.job_id] = gpu_id
                executed.add(job.job_id)
                _append_log(
                    log_path,
                    f"TREE_E1_JOB_START job={job.job_id} gpu={gpu_id}",
                )
            if not active:
                break
            done, _ = wait(
                active,
                timeout=min(
                    float(contract.snapshot_interval_seconds),
                    max(0.0, absolute_deadline - clock()),
                ),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                try:
                    _publish(
                        output=output,
                        bindings=bindings,
                        state=state,
                        log_path=log_path,
                        decision_path=None,
                    )
                except Exception:
                    pass
                continue
            for future in done:
                job, gpu_id = active.pop(future)
                free_gpus.append(gpu_id)
                free_gpus.sort()
                del state["active"][job.job_id]
                try:
                    result = future.result()
                except Exception as error:
                    result = FoldResult(
                        job.job_id,
                        job.candidate_id,
                        "failed",
                        None,
                        None,
                        None,
                        None,
                        f"{type(error).__name__}: {error}",
                    )
                if result.status == "completed":
                    completed.add(job.job_id)
                elif result.status == "skipped":
                    skipped.add(job.job_id)
                else:
                    failed.add(job.job_id)
                    stopping = True
                _append_log(
                    log_path,
                    f"TREE_E1_JOB_END job={job.job_id} gpu={gpu_id} status={result.status}",
                )
                state["completed"] = sorted(completed)
                state["skipped"] = sorted(skipped)
                state["failed"] = sorted(failed)
                _publish(
                    output=output,
                    bindings=bindings,
                    state=state,
                    log_path=log_path,
                    decision_path=None,
                )

    decision_path: Path | None = None
    if failed:
        status = "failed"
    elif pending or len(completed | skipped) < len(jobs):
        status = "budget_inconclusive"
    else:
        metrics: list[CandidateMetric] = []
        valid_rows = pd.read_csv(verified_data.train)
        valid_rows = valid_rows.loc[valid_rows["season"].eq(contract.fold[1])]
        groups = valid_rows["pitcher_id"].tolist()
        for job in jobs:
            if job.job_id in skipped:
                reason_path = jobs_root / job.job_id / "worker_result.json"
                reason = json.loads(reason_path.read_text()).get("failure", "skipped")
                metrics.append(skipped_metric(job.candidate_id, str(reason)))
            else:
                candidate = pd.read_csv(jobs_root / job.job_id / "predictions.csv")
                metrics.append(
                    evaluate_e1_candidate(
                        baseline,
                        candidate,
                        contract,
                        candidate_id=job.candidate_id,
                        bootstrap_groups=groups,
                    )
                )
        decision = decide_e1(metrics, contract)
        decision_path = output / "decision.json"
        _atomic_bytes(
            decision_path,
            _canonical_json(
                {
                    "status": decision.status,
                    "promoted": list(decision.promoted),
                    "reason": decision.reason,
                    "metrics": [_metric_payload(metric) for metric in metrics],
                }
            ),
        )
        status = decision.status
        _append_log(
            log_path,
            f"TREE_E1_DECISION status={status} promoted={','.join(decision.promoted) or 'none'}",
        )
    state["status"] = status
    state["active"] = {}
    bundles = _publish(
        output=output,
        bindings=bindings,
        state=state,
        log_path=log_path,
        decision_path=decision_path,
    )
    print(
        f"TREE_E1_HANDOFF_READY path={bundles.handoff} "
        f"sha256={bundles.handoff_sha256}",
        flush=True,
    )
    return E1CampaignResult(
        status=status,
        completed=tuple(sorted(completed)),
        skipped=tuple(sorted(skipped)),
        failed=tuple(sorted(failed)),
        reused=tuple(sorted(reused)),
        executed=tuple(sorted(executed)),
        bundles=bundles,
    )
