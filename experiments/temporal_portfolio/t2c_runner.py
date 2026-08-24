"""Restartable six-job T2-C confirmation on two Tesla T4 GPUs."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import traceback
from typing import Callable, Mapping

import pandas as pd

from .inputs import VerifiedOfficialData
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
from .t2a_runner import _prediction
from .t2c import (
    MAXIMUM_JOB_COUNT,
    NEW_SEEDS,
    VALID_YEARS,
    build_gated_s1_oof,
    build_t2c_specs,
    decide_t2c,
    evaluate_gated_s1_oof,
    materialize_t2c_job,
)
from .t2c_input import (
    VerifiedT2CInput,
    load_t2c_references,
    verify_t2c_input,
)
from .worker import WorkerBudgetIncomplete, run_worker, verify_worker_result


class T2CRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class T2CStageResult:
    status: str
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    evidence: Mapping[str, object]
    output_root: Path


STAGE_CAP_SECONDS = 7_200
_STOP_NEW_SECONDS = 900
_HANDOFF_RESERVE_SECONDS = 600
_WORKER_GRACE_SECONDS = 120


def run_t2c_stage(
    *,
    verified: VerifiedOfficialData,
    t2c_input: str | Path,
    output_root: str | Path,
    deadline: float,
    launcher: Launcher | None = None,
    frame_loader: Callable[[Path], pd.DataFrame] = pd.read_csv,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> T2CStageResult:
    if type(verified) is not VerifiedOfficialData:
        raise T2CRunnerError("verified official data identity is invalid")
    prepared = verify_t2c_input(t2c_input)
    if _input_identity(verified) != prepared.data_rows_sha256:
        raise T2CRunnerError("T2-C input is bound to different official data")
    started = float(clock())
    stage_deadline = min(float(deadline), started + STAGE_CAP_SECONDS)
    if stage_deadline <= started:
        raise T2CRunnerError("deadline has already passed")
    root = Path(output_root).resolve()
    jobs_root = root / "jobs"
    cache_root = root / "feature_cache"
    jobs_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    train = frame_loader(verified.train)
    history = frame_loader(verified.history)
    references = load_t2c_references(prepared)
    runtime = launcher
    if runtime is None:
        require_two_t4_gpus(log_prefix="T2C")
        runtime = ForkLauncher(_worker_entry)
    specs = list(build_t2c_specs())
    if len(specs) != MAXIMUM_JOB_COUNT:
        raise T2CRunnerError("T2-C schedule must contain exactly six jobs")
    completed: list[str] = []
    pending: list[str] = []
    failed: list[str] = []
    active: dict[int, _Active] = {}
    last_heartbeat = started
    print(
        f"T2C_STAGE_START jobs={len(specs)} deadline_unix={stage_deadline:.0f}",
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
                job = materialize_t2c_job(
                    spec,
                    data_rows_sha256=_input_identity(verified),
                    parent_sha256=prepared.decision_sha256,
                    train=train,
                    history=history,
                    cache_root=cache_root,
                )
                output = jobs_root / spec.job_id
                if _reuse_completed(output, job, verify_worker_result):
                    completed.append(spec.job_id)
                    print(f"T2C_JOB_REUSED job={spec.job_id}", flush=True)
                    continue
                hard_deadline = min(
                    now + job.plan.max_seconds,
                    stage_deadline - _HANDOFF_RESERVE_SECONDS,
                )
                worker_deadline = max(now, hard_deadline - _WORKER_GRACE_SECONDS)
                print(
                    f"T2C_JOB_START job={spec.job_id} gpu={gpu} "
                    f"deadline_unix={worker_deadline:.0f}",
                    flush=True,
                )
                process = runtime.start(job, output, gpu=gpu, deadline=worker_deadline)
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
                    print(f"T2C_JOB_COMPLETE job={job_id} gpu={gpu}", flush=True)
                elif returncode == 75 or deadline_reached:
                    pending.append(job_id)
                    print(f"T2C_JOB_PENDING job={job_id} gpu={gpu}", flush=True)
                else:
                    failed.append(job_id)
                    print(
                        f"T2C_JOB_FAILED job={job_id} gpu={gpu} returncode={returncode}",
                        flush=True,
                    )
                del active[gpu]
            now = float(clock())
            if now - last_heartbeat >= 60:
                print(
                    f"T2C_HEARTBEAT completed={len(completed)} active={len(active)} "
                    f"queued={len(specs)} remaining_seconds={max(0, int(stage_deadline - now))}",
                    flush=True,
                )
                last_heartbeat = now
            if (
                specs
                and not active
                and stage_deadline - now <= _STOP_NEW_SECONDS + _HANDOFF_RESERVE_SECONDS
            ):
                pending.extend(spec.job_id for spec in specs)
                specs.clear()
            if active:
                sleeper(0.25)
    except BaseException:
        for worker in active.values():
            if worker.process.poll() is None:
                _stop_worker(worker.process)
        raise
    evidence = _collect_evidence(
        jobs_root,
        references,
        completed=tuple(completed),
        pending=tuple(pending),
        failed=tuple(failed),
    )
    decision_status = str(evidence.get("decision", {}).get("status", "rejected"))
    status = "completed" if not pending and not failed else "budget_inconclusive"
    if status == "completed" and decision_status not in {"promoted", "rejected"}:
        raise T2CRunnerError("complete T2-C jobs produced an incomplete decision")
    result = T2CStageResult(
        status,
        tuple(dict.fromkeys(completed)),
        tuple(dict.fromkeys(pending)),
        tuple(dict.fromkeys(failed)),
        evidence,
        root,
    )
    _write_state(root / "t2c_stage_result.json", result, prepared)
    print(
        f"T2C_DECISION candidate={prepared.candidate_id} status={decision_status}",
        flush=True,
    )
    print(
        f"T2C_STAGE_RESULT status={status} completed={len(result.completed)} "
        f"pending={len(result.pending)} failed={len(result.failed)}",
        flush=True,
    )
    return result


def _collect_evidence(
    jobs_root: Path,
    references,
    *,
    completed: tuple[str, ...],
    pending: tuple[str, ...],
    failed: tuple[str, ...],
) -> dict[str, object]:
    anchors, fixed_multi, seed_3407, _decision = references
    seed_evidence: dict[int, dict[int, Mapping[str, object]]] = {}
    prediction_maps: dict[int, dict[int, pd.DataFrame]] = {3407: dict(seed_3407)}
    for seed in NEW_SEEDS:
        folds = {}
        predictions = {}
        for year in VALID_YEARS:
            job_id = f"t2c__s1__va{year}__s{seed}"
            if job_id in completed:
                prediction = _prediction(jobs_root / job_id)
                predictions[year] = prediction
                oof = build_gated_s1_oof(
                    anchors[year], fixed_multi[year], (prediction,), valid_year=year
                )
                folds[year] = dict(evaluate_gated_s1_oof(oof))
            elif job_id in failed:
                folds[year] = {"status": "failed"}
            else:
                folds[year] = {"status": "pending"}
        seed_evidence[seed] = folds
        prediction_maps[seed] = predictions
    ensemble: dict[str, object] = {}
    if all(
        year in prediction_maps[seed]
        for seed in (3407, *NEW_SEEDS)
        for year in VALID_YEARS
    ):
        folds = {}
        fold_oof = []
        for year in VALID_YEARS:
            oof = build_gated_s1_oof(
                anchors[year],
                fixed_multi[year],
                tuple(prediction_maps[seed][year] for seed in (3407, *NEW_SEEDS)),
                valid_year=year,
            )
            folds[year] = dict(evaluate_gated_s1_oof(oof))
            fold_oof.append(oof)
        combined_oof = pd.concat(fold_oof, ignore_index=True)
        ensemble = {
            "folds": folds,
            "combined": dict(evaluate_gated_s1_oof(combined_oof)),
        }
    decision = decide_t2c(seed_evidence=seed_evidence, ensemble=ensemble)
    payload = {
        "candidate_id": decision.candidate_id,
        "status": decision.status,
        "reason": decision.reason,
        "new_seed_weighted_gains": {
        str(key): value for key, value in decision.new_seed_weighted_gains.items()
        },
        "ensemble_weighted_gain": decision.ensemble_weighted_gain,
        "ensemble_latest_gain": decision.ensemble_latest_gain,
        "ensemble_bootstrap_lower": decision.ensemble_bootstrap_lower,
        "ensemble_max_segment_regression": decision.ensemble_max_segment_regression,
    }
    return {
        "seed_evidence": seed_evidence,
        "ensemble": ensemble,
        "decision": payload,
    }


def _write_state(path: Path, result: T2CStageResult, prepared: VerifiedT2CInput) -> None:
    payload = {
        "schema_version": 1,
        "stage": "T2C",
        "status": result.status,
        "data_rows_sha256": prepared.data_rows_sha256,
        "parent_sha256": prepared.decision_sha256,
        "completed": list(result.completed),
        "pending": list(result.pending),
        "failed": list(result.failed),
        "evidence": result.evidence,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _worker_entry(materialized, output_dir, gpu, deadline):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PREPROCESSING_SESSION_DEADLINE_UNIX"] = str(deadline)
    try:
        run_worker(materialized.training, output_dir, backend=None)
    except WorkerBudgetIncomplete as error:
        print(
            f"T2C_WORKER_BUDGET_INCOMPLETE job={materialized.training.job_id} {error}",
            flush=True,
        )
        raise SystemExit(75) from error
    except BaseException:
        traceback.print_exc()
        raise
