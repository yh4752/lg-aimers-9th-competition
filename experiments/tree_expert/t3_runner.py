from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from types import MappingProxyType
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .t3_artifacts import (
    T3Bindings,
    create_handoff_bundle,
    create_model_delivery,
    create_review_bundle,
    create_resume_bundle,
    publish_stable_resume,
    restore_resume_bundle,
)
from .t3_contracts import T3Job, confirmation_jobs, contract_sha256, load_t3_contract, structure_jobs
from .t3_decisions import (
    T3AcceptanceEvidence,
    T3StructureEvidence,
    acceptance_payload,
    accept_t3,
    blended_probability,
    select_structure,
    structure_payload,
)
from .t3_diagnostics import calibration_diagnostics, residual_correlation, segment_diagnostics
from .t3_full_fit import full_fit_t3
from .t3_inputs import VerifiedOfficialData, VerifiedT3Input
from .t3_training import T3JobResult, load_t3_job_result, run_t3_job


class T3RunnerError(RuntimeError):
    pass


JobExecutor = Callable[[T3Job, Path, int], object]
Clock = Callable[[], float]


@dataclass
class T3CampaignState:
    root: Path
    completed_jobs: set[str]
    failed_jobs: set[str]


@dataclass(frozen=True)
class T3CampaignResult:
    status: str
    acceptance_status: str
    review_bundle: Path
    resume_bundle: Path
    handoff_bundle: Path
    model_delivery: Path | None


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_state(state: T3CampaignState) -> None:
    path = state.root / "state/stage_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_canonical({
        "completed_jobs": sorted(state.completed_jobs),
        "failed_jobs": sorted(state.failed_jobs),
    }))
    temporary.replace(path)


def _load_state(root: Path) -> T3CampaignState:
    path = root / "state/stage_state.json"
    if not path.is_file():
        state = T3CampaignState(root, set(), set())
        _write_state(state)
        return state
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise T3RunnerError("campaign state cannot be loaded") from error
    if type(payload) is not dict or set(payload) != {"completed_jobs", "failed_jobs"}:
        raise T3RunnerError("campaign state schema differs")
    return T3CampaignState(root, set(payload["completed_jobs"]), set(payload["failed_jobs"]))


def _status(value: object) -> str:
    if isinstance(value, T3JobResult):
        return value.status
    if type(value) is str:
        return value
    return str(getattr(value, "status", "failed"))


def run_pending_jobs(
    state: T3CampaignState,
    jobs: Iterable[T3Job],
    *,
    executor: JobExecutor,
    gpu_ids: tuple[int, int],
    wall_deadline: float | None = None,
    clock: Clock = time.monotonic,
    new_job_guard_seconds: int = 600,
    on_progress: Callable[[], None] | None = None,
) -> None:
    if type(state) is not T3CampaignState or len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise T3RunnerError("job scheduler configuration differs")
    pending = [job for job in jobs if job.job_id not in state.completed_jobs]
    for offset in range(0, len(pending), len(gpu_ids)):
        if wall_deadline is not None and wall_deadline - clock() < new_job_guard_seconds:
            break
        batch = pending[offset:offset + len(gpu_ids)]
        with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="t3-gpu") as pool:
            futures = {
                pool.submit(executor, job, state.root / "jobs" / job.job_id, gpu_ids[index]): job
                for index, job in enumerate(batch)
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    status = _status(future.result())
                except Exception:
                    status = "failed"
                if status == "completed":
                    state.completed_jobs.add(job.job_id)
                    state.failed_jobs.discard(job.job_id)
                else:
                    state.failed_jobs.add(job.job_id)
                _write_state(state)
        if on_progress is not None:
            on_progress()


def require_full_fit_window(
    *,
    wall_deadline: float,
    clock: Clock = time.monotonic,
    guard_seconds: int,
) -> None:
    if type(guard_seconds) is not int or guard_seconds <= 0:
        raise T3RunnerError("full-fit guard differs")
    if wall_deadline - clock() < guard_seconds:
        raise T3RunnerError("full fit deferred; resume required")


_CODE_MEMBERS = (
    "experiments/tree_expert/contracts.py",
    "experiments/tree_expert/features.py",
    "experiments/tree_expert/t3_artifacts.py",
    "experiments/tree_expert/t3_contract.json",
    "experiments/tree_expert/t3_contracts.py",
    "experiments/tree_expert/t3_decisions.py",
    "experiments/tree_expert/t3_diagnostics.py",
    "experiments/tree_expert/t3_full_fit.py",
    "experiments/tree_expert/t3_inference.py",
    "experiments/tree_expert/t3_inputs.py",
    "experiments/tree_expert/t3_runner.py",
    "experiments/tree_expert/t3_state.py",
    "experiments/tree_expert/t3_temporal.py",
    "experiments/tree_expert/t3_training.py",
    "experiments/temporal_portfolio/seasonal_features.py",
    "experiments/independent_dl/feature_sources/seasonal.py",
)

_COMPATIBLE_RESUME_CODE_SHA256S = frozenset({
    "87c731d13abe14bc191f2b8a6b768b59c7e6aeedad071e11e5158e73d87052d3",
})


def code_sha256(root: Path | None = None) -> str:
    project = Path(__file__).resolve().parents[2] if root is None else Path(root)
    digest = sha256()
    for name in _CODE_MEMBERS:
        path = project / name
        if not path.is_file():
            raise T3RunnerError(f"runtime member is missing: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def campaign_bindings(verified: VerifiedT3Input, data: VerifiedOfficialData) -> T3Bindings:
    return T3Bindings(
        contract_sha256(), code_sha256(), verified.manifest_sha256,
        data.train_sha256, data.history_sha256, verified.e2_handoff_sha256,
    )


def _job_result(state: T3CampaignState, job: T3Job) -> T3JobResult:
    if job.job_id not in state.completed_jobs:
        raise T3RunnerError(f"required job is incomplete: {job.job_id}")
    return load_t3_job_result(state.root / "jobs" / job.job_id, job.job_id)


def _structure_evidence(state: T3CampaignState, verified: VerifiedT3Input):
    contract = load_t3_contract()
    targets: dict[tuple[int, int], np.ndarray] = {}
    baselines: dict[tuple[int, int], np.ndarray] = {}
    recent: dict[tuple[int, int], np.ndarray] = {}
    multi = {decay: {} for decay in contract.decays}
    job_map = {job.job_id: job for job in structure_jobs(contract)}
    for fold, path in verified.fold_predictions.items():
        baseline = pd.read_csv(path)
        targets[fold] = baseline["target"].to_numpy(dtype="float64")
        baselines[fold] = baseline["probability"].to_numpy(dtype="float64")
        for job in job_map.values():
            if (job.train_end_year, job.valid_year) != fold:
                continue
            prediction = _job_result(state, job).predictions["probability"].to_numpy(dtype="float64")
            if job.head == "recent":
                recent[fold] = prediction
            else:
                multi[job.decay][fold] = prediction
    return T3StructureEvidence(
        target=MappingProxyType(targets), baseline=MappingProxyType(baselines),
        recent=MappingProxyType(recent),
        multi=MappingProxyType({key: MappingProxyType(value) for key, value in multi.items()}),
        maximum_segment_regression=0.0,
    )


def _diagnose_structure(
    evidence: T3StructureEvidence,
    decision,
    train: pd.DataFrame,
    verified: VerifiedT3Input,
    output: Path,
) -> float:
    segments: list[pd.DataFrame] = []
    calibrations: list[pd.DataFrame] = []
    correlations: list[pd.DataFrame] = []
    for fold in load_t3_contract().folds:
        baseline_frame = pd.read_csv(verified.fold_predictions[fold])
        candidate = blended_probability(
            evidence.recent[fold], evidence.multi[decision.decay][fold], decision.recent_weight,
        )
        rows = train.loc[train["row_id"].astype(str).isin(baseline_frame["row_id"].astype(str))].copy()
        rows = baseline_frame[["row_id"]].merge(rows, on="row_id", how="left", validate="one_to_one")
        target = evidence.target[fold]
        report = segment_diagnostics(
            rows, target, candidate, evidence.baseline[fold],
            minimum_rows=load_t3_contract().gates.minimum_segment_rows,
        )
        report.insert(0, "fold", f"{fold[0]}->{fold[1]}")
        segments.append(report)
        calibration = calibration_diagnostics(target, candidate)
        calibration.insert(0, "fold", f"{fold[0]}->{fold[1]}")
        calibrations.append(calibration)
        correlation = residual_correlation(target, {"baseline": evidence.baseline[fold], "candidate": candidate})
        correlation.insert(0, "model", correlation.index)
        correlation.insert(0, "fold", f"{fold[0]}->{fold[1]}")
        correlations.append(correlation.reset_index(drop=True))
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    segment_frame = pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()
    segment_frame.to_csv(diagnostics / "segments.csv", index=False)
    pd.concat(calibrations, ignore_index=True).to_csv(diagnostics / "calibration.csv", index=False)
    pd.concat(correlations, ignore_index=True).to_csv(diagnostics / "residual_correlation.csv", index=False)
    if segment_frame.empty:
        return 0.0
    return max(0.0, float(-segment_frame["gain"].min()))


def _seed_fold_gains(state: T3CampaignState, verified: VerifiedT3Input, decision):
    contract = load_t3_contract()
    jobs = (*structure_jobs(contract), *confirmation_jobs(contract, decision.decay))
    by_key = {(job.head, job.seed, job.train_end_year, job.valid_year): job for job in jobs}
    result: dict[int, dict[tuple[int, int], float]] = {}
    for seed in (contract.structure_seed, *contract.confirmation_seeds):
        fold_gains: dict[tuple[int, int], float] = {}
        for fold, baseline_path in verified.fold_predictions.items():
            recent_job = by_key[("recent", seed, *fold)]
            multi_job = by_key[("multi", seed, *fold)]
            recent = _job_result(state, recent_job).predictions["probability"].to_numpy(dtype="float64")
            multi = _job_result(state, multi_job).predictions["probability"].to_numpy(dtype="float64")
            candidate = blended_probability(recent, multi, decision.recent_weight)
            baseline = pd.read_csv(baseline_path)
            target = baseline["target"].to_numpy(dtype="float64")
            base_probability = baseline["probability"].to_numpy(dtype="float64")
            fold_gains[fold] = float(np.mean((base_probability - target) ** 2) - np.mean((candidate - target) ** 2))
        result[seed] = fold_gains
    return MappingProxyType({seed: MappingProxyType(value) for seed, value in result.items()})


def _iteration_evidence(state: T3CampaignState, decision):
    contract = load_t3_contract()
    jobs = (*structure_jobs(contract), *confirmation_jobs(contract, decision.decay))
    result = {head: {seed: [] for seed in (42, 2026, 3407)} for head in ("recent", "multi")}
    for job in jobs:
        if job.head == "multi" and job.decay != decision.decay:
            continue
        item = _job_result(state, job)
        result[job.head][job.seed].append(int(item.best_iteration))
    return {head: {seed: tuple(values) for seed, values in by_seed.items()} for head, by_seed in result.items()}


def run_t3_campaign(
    verified: VerifiedT3Input,
    data: VerifiedOfficialData,
    output_dir: Path,
    *,
    resume_bundle: Path | None = None,
    wall_deadline: float | None = None,
    clock: Clock = time.monotonic,
) -> T3CampaignResult:
    contract = load_t3_contract()
    bindings = campaign_bindings(verified, data)
    output = Path(output_dir)
    if resume_bundle is not None:
        restore_resume_bundle(
            resume_bundle,
            output,
            bindings,
            compatible_code_sha256s=_COMPATIBLE_RESUME_CODE_SHA256S,
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
    state = _load_state(output)
    train = pd.read_csv(data.train)
    baseline_by_year = {
        fold[1]: pd.read_csv(path) for fold, path in verified.fold_predictions.items()
    }
    log_path = output / "tree_expert_t3.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def execute(job: T3Job, job_dir: Path, gpu_id: int):
        log(f"TREE_T3_JOB_START job={job.job_id} gpu={gpu_id}")
        result = run_t3_job(
            job=job, train=train, valid=train,
            baseline=baseline_by_year[job.valid_year], output_dir=job_dir, gpu_id=gpu_id,
        )
        log(f"TREE_T3_JOB_END job={job.job_id} status={result.status}")
        return result

    snapshot_root = output.parent / "snapshots"
    on_progress = lambda: publish_stable_resume(output, snapshot_root, bindings)
    deadline = wall_deadline if wall_deadline is not None else clock() + contract.wall_seconds
    run_pending_jobs(
        state, structure_jobs(contract), executor=execute, gpu_ids=(0, 1),
        wall_deadline=deadline, clock=clock,
        new_job_guard_seconds=contract.new_job_guard_seconds, on_progress=on_progress,
    )
    if any(job.job_id not in state.completed_jobs for job in structure_jobs(contract)):
        raise T3RunnerError("structure jobs are incomplete; resume required")
    structure_evidence = _structure_evidence(state, verified)
    structure = select_structure(structure_evidence, contract)
    maximum_segment_regression = _diagnose_structure(structure_evidence, structure, train, verified, output)
    structure_evidence = T3StructureEvidence(
        structure_evidence.target, structure_evidence.baseline, structure_evidence.recent,
        structure_evidence.multi, maximum_segment_regression,
    )
    structure = select_structure(structure_evidence, contract)
    decisions = output / "decisions"
    decisions.mkdir(parents=True, exist_ok=True)
    (decisions / "structure.json").write_text(_canonical(structure_payload(structure)))
    acceptance_status = "rejected"
    model_delivery = None
    if structure.status == "passed":
        run_pending_jobs(
            state, confirmation_jobs(contract, structure.decay), executor=execute, gpu_ids=(0, 1),
            wall_deadline=deadline, clock=clock,
            new_job_guard_seconds=contract.new_job_guard_seconds, on_progress=on_progress,
        )
        required = confirmation_jobs(contract, structure.decay)
        if any(job.job_id not in state.completed_jobs for job in required):
            raise T3RunnerError("confirmation jobs are incomplete; resume required")
        acceptance = accept_t3(
            T3AcceptanceEvidence(
                structure,
                _seed_fold_gains(state, verified, structure),
                MappingProxyType({
                    fold: len(pd.read_csv(path, usecols=["row_id"]))
                    for fold, path in verified.fold_predictions.items()
                }),
            ),
            contract,
        )
        acceptance_status = acceptance.status
        (decisions / "acceptance.json").write_text(_canonical(acceptance_payload(acceptance)))
        if acceptance.status == "accepted":
            _write_state(state)
            publish_stable_resume(output, snapshot_root, bindings)
            require_full_fit_window(
                wall_deadline=deadline,
                clock=clock,
                guard_seconds=contract.full_fit_guard_seconds,
            )
            full_fit = full_fit_t3(
                decision=acceptance, train=train, output_dir=output / "full_fit",
                iteration_evidence=_iteration_evidence(state, structure), gpu_ids=(0, 1),
            )
            model_delivery = create_model_delivery(
                full_fit.root, output / "bundles/tree_expert_t3_model_delivery.zip", bindings,
            )
    else:
        (decisions / "acceptance.json").write_text(_canonical({
            "status": "rejected", "reason": "structure_rejected",
        }))
    _write_state(state)
    bundles = output / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    review = create_review_bundle(output, bundles / "tree_expert_t3_review.zip", bindings)
    resume = create_resume_bundle(output, bundles / "tree_expert_t3_resume.zip", bindings)
    handoff = create_handoff_bundle(
        review=review, resume=resume, model_delivery=model_delivery,
        log=log_path, destination=bundles / "tree_expert_t3_handoff.zip", bindings=bindings,
    )
    log(f"TREE_T3_SUCCESS handoff={handoff}")
    return T3CampaignResult("completed", acceptance_status, review, resume, handoff, model_delivery)
