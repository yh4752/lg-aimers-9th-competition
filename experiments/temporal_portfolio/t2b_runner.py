"""Restartable three-phase T2-B campaign on two Tesla T4 GPUs."""
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
from .t2b import (
    T2BJobSpec,
    build_phase_c_specs,
    build_phase_f_specs,
    build_phase_h_specs,
    build_t2b_oof,
    decide_t2b_candidates,
    evaluate_t2b_fold,
    evaluate_t2b_oof,
    materialize_t2b_job,
    select_combination,
)
from .t2b_input import (
    VerifiedT2BInput,
    load_t2b_references,
    verify_t2b_input,
)
from .worker import WorkerBudgetIncomplete, run_worker, verify_worker_result


class T2BRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class T2BStageResult:
    status: str
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    evidence: Mapping[str, object]
    output_root: Path


MAXIMUM_JOB_COUNT = 10
_STOP_NEW_SECONDS = 900
_HANDOFF_RESERVE_SECONDS = 600
_WORKER_GRACE_SECONDS = 120


def run_t2b_stage(
    *,
    verified: VerifiedOfficialData,
    t2b_input: str | Path,
    output_root: str | Path,
    deadline: float,
    launcher: Launcher | None = None,
    frame_loader: Callable[[Path], pd.DataFrame] = pd.read_csv,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> T2BStageResult:
    if type(verified) is not VerifiedOfficialData:
        raise T2BRunnerError("verified official data identity is invalid")
    prepared = verify_t2b_input(t2b_input)
    if _input_identity(verified) != prepared.data_rows_sha256:
        raise T2BRunnerError("T2-B input is bound to different official data")
    started = float(clock())
    stage_deadline = min(float(deadline), started + 14_400)
    if stage_deadline <= started:
        raise T2BRunnerError("deadline has already passed")
    root = Path(output_root).resolve()
    jobs_root = root / "jobs"
    cache_root = root / "feature_cache"
    jobs_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    train = frame_loader(verified.train)
    history = frame_loader(verified.history)
    anchors, fixed_multi, latest_predictions, t2a_evidence = load_t2b_references(
        prepared
    )
    runtime = launcher
    if runtime is None:
        require_two_t4_gpus(log_prefix="T2B")
        runtime = ForkLauncher(_worker_entry)

    completed: list[str] = []
    pending: list[str] = []
    failed: list[str] = []
    phase_f_specs = build_phase_f_specs()
    phase_c_specs = build_phase_c_specs()
    if len(phase_f_specs) + len(phase_c_specs) + 2 > MAXIMUM_JOB_COUNT:
        raise T2BRunnerError("T2-B schedule exceeds ten jobs")

    print("T2B_PHASE_START phase=F jobs=6", flush=True)
    _run_jobs(
        list(phase_f_specs),
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
        clock=clock,
        sleeper=sleeper,
    )
    phase_f = _phase_f_evidence(
        jobs_root,
        anchors,
        fixed_multi,
        latest_predictions,
        t2a_evidence,
        completed,
        failed,
    )

    phase_c: Mapping[str, Mapping[str, object]] = {}
    selected = None
    if not pending:
        print("T2B_PHASE_START phase=C jobs=2", flush=True)
        _run_jobs(
            list(phase_c_specs),
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
            clock=clock,
            sleeper=sleeper,
        )
        phase_c = _phase_c_evidence(
            jobs_root, anchors, fixed_multi, completed, failed
        )
        if not pending:
            selected = select_combination(phase_c)
            print(
                f"T2B_COMBINATION_SELECTED candidate={selected or 'none'}",
                flush=True,
            )

    if selected is not None and not pending:
        print("T2B_PHASE_START phase=H jobs=2", flush=True)
        _run_jobs(
            list(build_phase_h_specs(selected)),
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
            clock=clock,
            sleeper=sleeper,
        )

    final, decisions = _final_evidence(
        jobs_root,
        anchors,
        fixed_multi,
        latest_predictions,
        phase_f,
        phase_c,
        selected,
        completed,
        failed,
    )
    for decision in decisions:
        print(
            f"T2B_DECISION candidate={decision.candidate_id} status={decision.status}",
            flush=True,
        )
    evidence = {
        "phase_f": phase_f,
        "phase_c": phase_c,
        "selected_combination": selected,
        **final,
    }
    status = "completed" if not pending else "budget_inconclusive"
    result = T2BStageResult(
        status,
        tuple(dict.fromkeys(completed)),
        tuple(dict.fromkeys(pending)),
        tuple(dict.fromkeys(failed)),
        evidence,
        root,
    )
    _write_state(root / "t2b_stage_result.json", result, prepared)
    print(
        f"T2B_STAGE_RESULT status={status} completed={len(result.completed)} "
        f"pending={len(result.pending)} failed={len(result.failed)}",
        flush=True,
    )
    return result


def _run_jobs(
    specs: list[T2BJobSpec],
    *,
    verified: VerifiedOfficialData,
    prepared: VerifiedT2BInput,
    train: pd.DataFrame,
    history: pd.DataFrame,
    jobs_root: Path,
    cache_root: Path,
    stage_deadline: float,
    launcher: Launcher,
    completed: list[str],
    pending: list[str],
    failed: list[str],
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
                job = materialize_t2b_job(
                    spec,
                    data_rows_sha256=_input_identity(verified),
                    parent_sha256=prepared.t2a_decision_sha256,
                    train=train,
                    history=history,
                    cache_root=cache_root,
                )
                output = jobs_root / spec.job_id
                if _reuse_completed(output, job, verify_worker_result):
                    completed.append(spec.job_id)
                    print(f"T2B_JOB_REUSED job={spec.job_id}", flush=True)
                    continue
                hard_deadline = min(
                    now + job.plan.max_seconds,
                    stage_deadline - _HANDOFF_RESERVE_SECONDS,
                )
                worker_deadline = max(now, hard_deadline - _WORKER_GRACE_SECONDS)
                print(
                    f"T2B_JOB_START job={spec.job_id} gpu={gpu} "
                    f"deadline_unix={worker_deadline:.0f}",
                    flush=True,
                )
                process = launcher.start(
                    job, output, gpu=gpu, deadline=worker_deadline
                )
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
                    _require_completed(
                        worker.output_dir, worker.job, verify_worker_result
                    )
                    completed.append(job_id)
                    print(f"T2B_JOB_COMPLETE job={job_id} gpu={gpu}", flush=True)
                elif returncode == 75 or deadline_reached:
                    pending.append(job_id)
                    print(f"T2B_JOB_PENDING job={job_id} gpu={gpu}", flush=True)
                else:
                    failed.append(job_id)
                    print(
                        f"T2B_JOB_FAILED job={job_id} gpu={gpu} "
                        f"returncode={returncode}",
                        flush=True,
                    )
                del active[gpu]
            now = float(clock())
            if now - last_heartbeat >= 60:
                print(
                    f"T2B_HEARTBEAT completed={len(completed)} active={len(active)} "
                    f"queued={len(specs)} "
                    f"remaining_seconds={max(0, int(stage_deadline - now))}",
                    flush=True,
                )
                last_heartbeat = now
            if (
                specs
                and not active
                and stage_deadline - now
                <= _STOP_NEW_SECONDS + _HANDOFF_RESERVE_SECONDS
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


def _phase_f_evidence(
    jobs_root,
    anchors,
    fixed_multi,
    latest_predictions,
    _t2a_evidence,
    completed,
    failed,
):
    evidence = {}
    for bundle in ("S1", "P3", "P2"):
        folds = {}
        for year in (2022, 2023):
            job_id = f"t2b__f__{bundle.casefold()}__va{year}__s3407"
            if job_id in completed:
                folds[year] = evaluate_t2b_fold(
                    anchors[year],
                    fixed_multi[year],
                    _prediction(jobs_root / job_id),
                    valid_year=year,
                )
            elif job_id in failed:
                folds[year] = {"status": "failed"}
            else:
                folds[year] = {"status": "pending"}
        folds[2024] = evaluate_t2b_fold(
            anchors[2024],
            fixed_multi[2024],
            latest_predictions[bundle],
            valid_year=2024,
        )
        evidence[bundle] = folds
    return evidence


def _phase_c_evidence(jobs_root, anchors, fixed_multi, completed, failed):
    evidence = {}
    for bundles in (("S1", "P3"), ("S1", "P2")):
        name = "+".join(bundles)
        job_id = f"t2b__c__{'_'.join(item.casefold() for item in bundles)}__va2024__s3407"
        if job_id in completed:
            evidence[name] = evaluate_t2b_fold(
                anchors[2024],
                fixed_multi[2024],
                _prediction(jobs_root / job_id),
                valid_year=2024,
            )
        elif job_id in failed:
            evidence[name] = {"status": "failed"}
        else:
            evidence[name] = {"status": "pending"}
    return evidence


def _final_evidence(
    jobs_root,
    anchors,
    fixed_multi,
    latest_predictions,
    phase_f,
    phase_c,
    selected,
    completed,
    failed,
):
    folds = {bundle: dict(phase_f[bundle]) for bundle in ("S1", "P3", "P2")}
    predictions = {
        bundle: {
            2022: _prediction(
                jobs_root / f"t2b__f__{bundle.casefold()}__va2022__s3407"
            ),
            2023: _prediction(
                jobs_root / f"t2b__f__{bundle.casefold()}__va2023__s3407"
            ),
            2024: latest_predictions[bundle],
        }
        for bundle in ("S1", "P3", "P2")
        if all(folds[bundle][year].get("status") == "completed" for year in (2022, 2023, 2024))
    }
    if selected is not None:
        bundles = tuple(selected.split("+"))
        combo_folds = {2024: phase_c[selected]}
        combo_predictions = {}
        for year in (2022, 2023):
            job_id = f"t2b__h__{'_'.join(item.casefold() for item in bundles)}__va{year}__s3407"
            if job_id in completed:
                combo_folds[year] = evaluate_t2b_fold(
                    anchors[year],
                    fixed_multi[year],
                    _prediction(jobs_root / job_id),
                    valid_year=year,
                )
                combo_predictions[year] = _prediction(jobs_root / job_id)
            elif job_id in failed:
                combo_folds[year] = {"status": "failed"}
            else:
                combo_folds[year] = {"status": "pending"}
        folds[selected] = combo_folds
        latest_id = f"t2b__c__{'_'.join(item.casefold() for item in bundles)}__va2024__s3407"
        if latest_id in completed and all(
            combo_folds[year].get("status") == "completed" for year in (2022, 2023, 2024)
        ):
            combo_predictions[2024] = _prediction(jobs_root / latest_id)
            predictions[selected] = combo_predictions

    complete_folds = {name: folds[name] for name in predictions}
    combined = {}
    for name, by_year in predictions.items():
        oof = pd.concat(
            [
                build_t2b_oof(
                    anchors[year],
                    fixed_multi[year],
                    by_year[year],
                    valid_year=year,
                )
                for year in (2022, 2023, 2024)
            ],
            ignore_index=True,
        )
        combined[name] = evaluate_t2b_oof(oof)
    decisions = (
        decide_t2b_candidates(complete_folds, combined) if complete_folds else ()
    )
    decision_payload = [
        {
            "candidate": item.candidate_id,
            "status": item.status,
            "tier": item.tier,
        }
        for item in decisions
    ]
    eligible = [
        item for item in decision_payload if item["status"] in ("champion", "exploratory")
    ]
    eligible.sort(
        key=lambda item: (
            0 if item["status"] == "champion" else 1,
            -float(combined[item["candidate"]]["gain"]),
            item["candidate"],
        )
    )
    return (
        {
            "candidate_folds": folds,
            "combined": combined,
            "decisions": decision_payload,
            "promoted": eligible[:3],
        },
        decisions,
    )


def _write_state(path: Path, result: T2BStageResult, prepared: VerifiedT2BInput) -> None:
    payload = {
        "schema_version": 1,
        "stage": "T2B",
        "status": result.status,
        "data_rows_sha256": prepared.data_rows_sha256,
        "parent_sha256": prepared.t2a_decision_sha256,
        "completed": list(result.completed),
        "pending": list(result.pending),
        "failed": list(result.failed),
        "evidence": result.evidence,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
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
            f"T2B_WORKER_BUDGET_INCOMPLETE job={materialized.training.job_id} {error}",
            flush=True,
        )
        raise SystemExit(75) from error
    except BaseException:
        traceback.print_exc()
        raise
