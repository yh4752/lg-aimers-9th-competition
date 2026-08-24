"""Restartable two-phase T2-A feature screening on two Tesla T4 GPUs."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import time
import traceback
from typing import Callable, Mapping
from zipfile import ZipFile

import pandas as pd

from .inputs import VerifiedOfficialData
from .t1_review import VerifiedT2AInput, verify_t2a_input
from .t1_runner import (
    ForkLauncher,
    Launcher,
    _Active,
    _input_identity,
    _require_completed,
    _reuse_completed,
    _stop_worker,
    require_two_t4_gpus,
)
from .t2a import (
    T2AError,
    T2AJobSpec,
    build_phase_m_specs,
    build_phase_r_specs,
    evaluate_feature_candidate,
    materialize_t2a_job,
    select_phase_m_bundles,
)
from .worker import WorkerBudgetIncomplete, run_worker, verify_worker_result


class T2ARunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class T2AStageResult:
    status: str
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    skipped: Mapping[str, str]
    evidence: Mapping[str, object]
    output_root: Path


_STOP_NEW_SECONDS = 900
_HANDOFF_RESERVE_SECONDS = 600
_WORKER_GRACE_SECONDS = 120


def run_t2a_stage(
    *,
    verified: VerifiedOfficialData,
    t2a_input: str | Path,
    output_root: str | Path,
    deadline: float,
    launcher: Launcher | None = None,
    frame_loader: Callable[[Path], pd.DataFrame] = pd.read_csv,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> T2AStageResult:
    if type(verified) is not VerifiedOfficialData:
        raise T2ARunnerError("verified official data identity is invalid")
    prepared = verify_t2a_input(t2a_input)
    if _input_identity(verified) != prepared.data_rows_sha256:
        raise T2ARunnerError("T2-A input is bound to different official data")
    started = float(clock())
    stage_deadline = min(float(deadline), started + 14_400)
    if stage_deadline <= started:
        raise T2ARunnerError("deadline has already passed")
    root = Path(output_root).resolve()
    jobs_root = root / "jobs"
    cache_root = root / "feature_cache"
    jobs_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    train = frame_loader(verified.train)
    history = frame_loader(verified.history)
    anchor, fixed_multi = _reference_frames(prepared)
    runtime = launcher
    if runtime is None:
        require_two_t4_gpus()
        runtime = ForkLauncher(_worker_entry)

    completed: list[str] = []
    pending: list[str] = []
    failed: list[str] = []
    skipped: dict[str, str] = {}
    print("T2A_STAGE_START phase=R jobs=7", flush=True)
    _run_jobs(
        list(build_phase_r_specs()),
        verified=verified,
        prepared=prepared,
        train=train,
        history=history,
        jobs_root=jobs_root,
        cache_root=cache_root,
        stage_deadline=stage_deadline,
        launcher=runtime,
        completed=completed,
        pending=pending,
        failed=failed,
        skipped=skipped,
        clock=clock,
        sleeper=sleeper,
    )
    phase_r = _phase_r_evidence(
        jobs_root, anchor, fixed_multi, completed, failed, skipped
    )
    selected: tuple[str, ...] = ()
    phase_m: dict[str, Mapping[str, object]] = {}
    if not pending:
        selected = select_phase_m_bundles(phase_r)
        print(
            f"T2A_PHASE_R_COMPLETE selected={','.join(selected) if selected else 'none'}",
            flush=True,
        )
        if selected:
            print(f"T2A_STAGE_START phase=M jobs={len(selected)}", flush=True)
            _run_jobs(
                list(build_phase_m_specs(selected)),
                verified=verified,
                prepared=prepared,
                train=train,
                history=history,
                jobs_root=jobs_root,
                cache_root=cache_root,
                stage_deadline=stage_deadline,
                launcher=runtime,
                completed=completed,
                pending=pending,
                failed=failed,
                skipped=skipped,
                clock=clock,
                sleeper=sleeper,
            )
            phase_m = _phase_m_evidence(
                jobs_root, anchor, selected, completed, failed, skipped
            )
    evidence = {
        "phase_r": phase_r,
        "phase_m_selected": list(selected),
        "phase_m": phase_m,
        "promoted": _promoted(phase_r, phase_m),
    }
    status = "completed" if not pending else "budget_inconclusive"
    result = T2AStageResult(
        status,
        tuple(completed),
        tuple(pending),
        tuple(failed),
        dict(sorted(skipped.items())),
        evidence,
        root,
    )
    _write_state(root / "t2a_stage_result.json", result, prepared)
    print(
        f"T2A_STAGE_RESULT status={status} completed={len(completed)} "
        f"pending={len(pending)} failed={len(failed)} skipped={len(skipped)}",
        flush=True,
    )
    return result


def _run_jobs(
    specs: list[T2AJobSpec],
    *,
    verified: VerifiedOfficialData,
    prepared: VerifiedT2AInput,
    train: pd.DataFrame,
    history: pd.DataFrame,
    jobs_root: Path,
    cache_root: Path,
    stage_deadline: float,
    launcher: Launcher,
    completed: list[str],
    pending: list[str],
    failed: list[str],
    skipped: dict[str, str],
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    active: dict[int, _Active] = {}
    last_heartbeat = float(clock())
    try:
        while specs or active:
            now = float(clock())
            for gpu in range(2):
                if gpu in active or not specs:
                    continue
                if stage_deadline - now <= _STOP_NEW_SECONDS + _HANDOFF_RESERVE_SECONDS:
                    break
                spec = specs.pop(0)
                try:
                    job = materialize_t2a_job(
                        spec,
                        data_rows_sha256=_input_identity(verified),
                        t1_decision_sha256=prepared.decision_sha256,
                        train=train,
                        history=history,
                        cache_root=cache_root,
                    )
                except T2AError as error:
                    if str(error) != "insufficient_mapping":
                        raise
                    skipped[spec.job_id] = "insufficient_mapping"
                    print(f"T2A_JOB_SKIPPED job={spec.job_id} reason=insufficient_mapping", flush=True)
                    continue
                output = jobs_root / spec.job_id
                if _reuse_completed(output, job, verify_worker_result):
                    completed.append(spec.job_id)
                    print(f"T2A_JOB_REUSED job={spec.job_id}", flush=True)
                    continue
                hard_deadline = min(
                    now + job.plan.max_seconds,
                    stage_deadline - _HANDOFF_RESERVE_SECONDS,
                )
                worker_deadline = max(now, hard_deadline - _WORKER_GRACE_SECONDS)
                print(
                    f"T2A_JOB_START job={spec.job_id} gpu={gpu} deadline_unix={worker_deadline:.0f}",
                    flush=True,
                )
                process = launcher.start(job, output, gpu=gpu, deadline=worker_deadline)
                active[gpu] = _Active(job, gpu, output, hard_deadline, process)
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
                    _require_completed(worker.output_dir, worker.job, verify_worker_result)
                    completed.append(job_id)
                    print(f"T2A_JOB_COMPLETE job={job_id} gpu={gpu}", flush=True)
                elif returncode == 75 or deadline_reached:
                    pending.append(job_id)
                    print(f"T2A_JOB_PENDING job={job_id} gpu={gpu}", flush=True)
                else:
                    failed.append(job_id)
                    print(f"T2A_JOB_FAILED job={job_id} gpu={gpu} returncode={returncode}", flush=True)
                del active[gpu]
            now = float(clock())
            if now - last_heartbeat >= 60:
                print(
                    f"T2A_HEARTBEAT completed={len(completed)} active={len(active)} "
                    f"queued={len(specs)} remaining_seconds={max(0, int(stage_deadline - now))}",
                    flush=True,
                )
                last_heartbeat = now
            if specs and not active and stage_deadline - now <= _STOP_NEW_SECONDS + _HANDOFF_RESERVE_SECONDS:
                pending.extend(spec.job_id for spec in specs)
                specs.clear()
            if active:
                sleeper(0.25)
    except BaseException:
        for worker in active.values():
            if worker.process.poll() is None:
                _stop_worker(worker.process)
        raise


def _phase_r_evidence(jobs_root, anchor, fixed_multi, completed, failed, skipped):
    evidence = {}
    completed_set = set(completed)
    failed_set = set(failed)
    for spec in build_phase_r_specs():
        if spec.job_id in completed_set:
            item = evaluate_feature_candidate(
                anchor, fixed_multi, _prediction(jobs_root / spec.job_id)
            )
            item.update(_mapping_evidence(jobs_root / spec.job_id))
            evidence[spec.bundle] = item
        elif spec.job_id in skipped:
            evidence[spec.bundle] = {"status": skipped[spec.job_id]}
        elif spec.job_id in failed_set:
            evidence[spec.bundle] = {"status": "failed"}
        else:
            evidence[spec.bundle] = {"status": "pending"}
    return evidence


def _phase_m_evidence(jobs_root, anchor, selected, completed, failed, skipped):
    evidence = {}
    for bundle in selected:
        recent = jobs_root / f"t2a__r__{bundle.casefold()}__va2024__s3407"
        multi_id = f"t2a__m__{bundle.casefold()}__va2024__s3407"
        if multi_id in completed:
            item = evaluate_feature_candidate(
                anchor, _prediction(jobs_root / multi_id), _prediction(recent)
            )
            item.update(_mapping_evidence(jobs_root / multi_id))
            evidence[bundle] = item
        elif multi_id in skipped:
            evidence[bundle] = {"status": skipped[multi_id]}
        elif multi_id in failed:
            evidence[bundle] = {"status": "failed"}
        else:
            evidence[bundle] = {"status": "pending"}
    return evidence


def _promoted(phase_r, phase_m):
    ranked = []
    for bundle, recent in phase_r.items():
        if recent.get("status") != "completed":
            continue
        multi = phase_m.get(bundle)
        variants = [("recent_only", recent)]
        if multi is not None and multi.get("status") == "completed":
            variants.append(("both_experts", multi))
        variant, evidence = max(variants, key=lambda item: float(item[1]["gain"]))
        if (
            float(evidence["gain"]) >= 0.00003
            and float(evidence["bootstrap_lower"]) > 0
            and float(evidence["max_segment_regression"]) <= 0.00050
        ):
            mapping = evidence.get("mapping_status", "not_applicable")
            ranked.append((bundle, variant, float(evidence["gain"]), mapping))
    ranked.sort(key=lambda item: (-item[2], item[0]))
    return [
        {
            "bundle": bundle,
            "variant": variant,
            "gain": gain,
            "mapping_gate": "exploratory_only" if mapping == "exploratory" else "eligible",
        }
        for bundle, variant, gain, mapping in ranked[:3]
    ]


def _reference_frames(prepared: VerifiedT2AInput):
    with ZipFile(prepared.path) as archive:
        return (
            pd.read_csv(io.BytesIO(archive.read("t1_anchor_2024.csv"))),
            pd.read_csv(io.BytesIO(archive.read("t1_multi_2024.csv"))),
        )


def _prediction(root: Path) -> pd.DataFrame:
    verify_worker_result(root)
    return pd.read_csv(root / "predictions.csv")


def _mapping_evidence(root: Path) -> dict[str, object]:
    payload = json.loads((root / "checkpoint_meta.json").read_text(encoding="utf-8"))
    binding = payload.get("checkpoint_binding", {})
    status = binding.get("mapping_status", "not_applicable")
    coverage = binding.get("mapping_coverage", "not_applicable")
    if status not in {"not_applicable", "exploratory", "accepted"}:
        raise T2ARunnerError(f"mapping status differs: {root.name}")
    return {
        "mapping_status": status,
        "mapping_coverage": coverage,
    }


def _write_state(path: Path, result: T2AStageResult, prepared: VerifiedT2AInput) -> None:
    payload = {
        "schema_version": 1,
        "stage": "T2A",
        "status": result.status,
        "data_rows_sha256": prepared.data_rows_sha256,
        "t1_decision_sha256": prepared.decision_sha256,
        "completed": list(result.completed),
        "pending": list(result.pending),
        "failed": list(result.failed),
        "skipped": dict(result.skipped),
        "evidence": result.evidence,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def _worker_entry(materialized, output_dir, gpu, deadline):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PREPROCESSING_SESSION_DEADLINE_UNIX"] = str(deadline)
    try:
        run_worker(materialized.training, output_dir, backend=None)
    except WorkerBudgetIncomplete as error:
        print(f"T2A_WORKER_BUDGET_INCOMPLETE job={materialized.training.job_id} {error}", flush=True)
        raise SystemExit(75) from error
    except BaseException:
        traceback.print_exc()
        raise
