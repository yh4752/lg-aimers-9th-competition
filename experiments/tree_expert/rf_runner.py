from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .rf_artifacts import (
    RFBindings,
    create_rf_delivery,
    create_rf_resume,
    create_rf_review,
    restore_rf_resume,
)
from .rf_contracts import (
    RFContract,
    RFJob,
    confirmation_jobs,
    contract_sha256,
    load_rf_contract,
    structure_jobs,
)
from .rf_decisions import (
    RFAcceptanceEvidence,
    RFStructureEvidence,
    acceptance_payload,
    accept_rf,
    route_probability,
    screen_structure_heads,
    select_rf_structure,
    structure_payload,
)
from .rf_diagnostics import build_rf_diagnostics
from .rf_full_fit import full_fit_rf
from .rf_inference import audit_row_independence, load_rf_inference_runtime
from .rf_inputs import VerifiedRFInput
from .rf_training import RFJobResult, load_rf_job_result, run_rf_job


class RFRunnerError(RuntimeError):
    pass


JobExecutor = Callable[[RFJob, Path, int], object]
Clock = Callable[[], float]


@dataclass
class RFCampaignState:
    root: Path
    phase: str
    status: str
    completed_jobs: set[str]
    failed_jobs: set[str]


@dataclass(frozen=True)
class RFCampaignResult:
    status: str
    acceptance_status: str
    review_bundle: Path
    resume_bundle: Path
    delivery_bundle: Path | None


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_campaign_state(state: RFCampaignState) -> None:
    if type(state) is not RFCampaignState:
        raise RFRunnerError("campaign state type differs")
    path = state.root / "campaign_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        _canonical(
            {
                "schema_version": 1,
                "phase": state.phase,
                "status": state.status,
                "completed_jobs": sorted(state.completed_jobs),
                "failed_jobs": sorted(state.failed_jobs),
            }
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_campaign_state(root: Path) -> RFCampaignState:
    source = Path(root)
    path = source / "campaign_state.json"
    if not path.is_file():
        state = RFCampaignState(source, "structure", "running", set(), set())
        write_campaign_state(state)
        return state
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RFRunnerError("campaign state cannot be loaded") from error
    if (
        type(payload) is not dict
        or set(payload) != {"schema_version", "phase", "status", "completed_jobs", "failed_jobs"}
        or payload["schema_version"] != 1
        or payload["phase"] not in {"structure", "confirmation", "full_fit", "completed", "rejected", "paused"}
        or payload["status"] not in {"running", "accepted", "rejected", "paused", "failed"}
        or type(payload["completed_jobs"]) is not list
        or type(payload["failed_jobs"]) is not list
    ):
        raise RFRunnerError("campaign state schema differs")
    return RFCampaignState(
        source,
        str(payload["phase"]),
        str(payload["status"]),
        set(payload["completed_jobs"]),
        set(payload["failed_jobs"]),
    )


def _status(value: object) -> str:
    if type(value) is str:
        return value
    return str(getattr(value, "status", "failed"))


def run_pending_jobs(
    state: RFCampaignState,
    jobs: Iterable[RFJob],
    *,
    executor: JobExecutor,
    gpu_ids: tuple[int, int],
    wall_deadline: float | None = None,
    clock: Clock = time.monotonic,
    new_job_guard_seconds: int = 600,
    on_progress: Callable[[], None] | None = None,
) -> None:
    if (
        type(state) is not RFCampaignState
        or len(gpu_ids) != 2
        or len(set(gpu_ids)) != 2
        or any(type(item) is not int or item < 0 for item in gpu_ids)
    ):
        raise RFRunnerError("job scheduler configuration differs")
    pending = [job for job in jobs if job.job_id not in state.completed_jobs]
    for offset in range(0, len(pending), len(gpu_ids)):
        if wall_deadline is not None and wall_deadline - clock() < new_job_guard_seconds:
            break
        batch = pending[offset:offset + len(gpu_ids)]
        with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="rf-gpu") as pool:
            futures = {
                pool.submit(
                    executor,
                    job,
                    state.root / "jobs" / job.job_id,
                    gpu_ids[index],
                ): job
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
                write_campaign_state(state)
        if on_progress is not None:
            on_progress()


def require_full_fit_window(
    *,
    wall_deadline: float,
    clock: Clock = time.monotonic,
    guard_seconds: int,
) -> None:
    if type(guard_seconds) is not int or guard_seconds <= 0:
        raise RFRunnerError("full-fit guard differs")
    if wall_deadline - clock() < guard_seconds:
        raise RFRunnerError("full fit deferred; resume required")


def _prepare_full_fit_output(campaign_root: Path) -> Path:
    root = Path(campaign_root)
    destination = root / "full_fit"
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise RFRunnerError("full-fit output path differs")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    return destination


_CODE_MEMBERS = (
    "experiments/tree_expert/e2_full_fit.py",
    "experiments/tree_expert/e2_inference.py",
    "experiments/tree_expert/features.py",
    "experiments/tree_expert/rf_artifacts.py",
    "experiments/tree_expert/rf_contract.json",
    "experiments/tree_expert/rf_contracts.py",
    "experiments/tree_expert/rf_decisions.py",
    "experiments/tree_expert/rf_diagnostics.py",
    "experiments/tree_expert/rf_full_fit.py",
    "experiments/tree_expert/rf_inference.py",
    "experiments/tree_expert/rf_inputs.py",
    "experiments/tree_expert/rf_runner.py",
    "experiments/tree_expert/rf_training.py",
    "experiments/temporal_portfolio/seasonal_features.py",
    "experiments/independent_dl/feature_sources/seasonal.py",
)


def code_sha256(root: Path | None = None) -> str:
    project = Path(__file__).resolve().parents[2] if root is None else Path(root)
    digest = sha256()
    for name in _CODE_MEMBERS:
        path = project / name
        if not path.is_file():
            raise RFRunnerError(f"runtime member is missing: {name}")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def campaign_bindings(verified: VerifiedRFInput, data: object) -> RFBindings:
    return RFBindings(
        contract_sha256=contract_sha256(),
        code_sha256=code_sha256(),
        input_manifest_sha256=verified.manifest_sha256,
        official_train_sha256=str(getattr(data, "train_sha256")),
        official_history_sha256=str(getattr(data, "history_sha256")),
        e2_handoff_sha256=verified.e2_handoff_sha256,
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_canonical(value), encoding="utf-8")
    os.replace(temporary, path)


def _job_result(state: RFCampaignState, job: RFJob) -> RFJobResult:
    if job.job_id not in state.completed_jobs:
        raise RFRunnerError(f"required job is incomplete: {job.job_id}")
    return load_rf_job_result(state.root / "jobs" / job.job_id, job.job_id)


def _evidence(
    state: RFCampaignState,
    verified: VerifiedRFInput,
    jobs: Iterable[RFJob],
    folds: tuple[tuple[int, int], ...],
    *,
    fill_missing_with_baseline: bool,
) -> RFStructureEvidence:
    job_map = {(job.head, job.train_end_year, job.valid_year): job for job in jobs}
    target: dict[tuple[int, int], np.ndarray] = {}
    baseline: dict[tuple[int, int], np.ndarray] = {}
    game_type: dict[tuple[int, int], np.ndarray] = {}
    expert = {head: {} for head in ("f_small", "f_wide", "r_expert")}
    for fold in folds:
        frame = pd.read_csv(verified.fold_predictions[fold])
        target[fold] = frame["target"].to_numpy(dtype="float64")
        baseline[fold] = frame["probability"].to_numpy(dtype="float64")
        game_type[fold] = frame["game_type"].astype(str).to_numpy()
        for head in expert:
            job = job_map.get((head, *fold))
            if job is not None and job.job_id in state.completed_jobs:
                expert[head][fold] = _job_result(state, job).predictions["probability"].to_numpy(dtype="float64")
            elif fill_missing_with_baseline:
                expert[head][fold] = baseline[fold].copy()
            else:
                raise RFRunnerError(f"required structure prediction is absent: {head} {fold}")
    return RFStructureEvidence(
        target=MappingProxyType(target),
        baseline=MappingProxyType(baseline),
        game_type=MappingProxyType(game_type),
        expert=MappingProxyType({head: MappingProxyType(value) for head, value in expert.items()}),
    )


def _iteration_evidence(
    state: RFCampaignState,
    jobs: Iterable[RFJob],
    heads: tuple[str, ...],
) -> dict[str, dict[int, tuple[int, ...]]]:
    result: dict[str, dict[int, list[int]]] = {
        head: {seed: [] for seed in (42, 2026, 3407)} for head in heads
    }
    for job in jobs:
        if job.head not in result or job.job_id not in state.completed_jobs:
            continue
        item = _job_result(state, job)
        result[job.head][job.seed].append(int(item.best_iteration))
    output = {
        head: {seed: tuple(values) for seed, values in by_seed.items()}
        for head, by_seed in result.items()
    }
    if any(len(values) != 3 for by_seed in output.values() for values in by_seed.values()):
        raise RFRunnerError("iteration evidence differs")
    return output


def _seed_candidates(
    state: RFCampaignState,
    verified: VerifiedRFInput,
    jobs: Iterable[RFJob],
    *,
    f_head: str,
    include_r: bool,
    alpha_r: float,
    alpha_f: float,
    contract: RFContract,
) -> Mapping[int, Mapping[tuple[int, int], np.ndarray]]:
    by_key = {(job.head, job.seed, job.train_end_year, job.valid_year): job for job in jobs}
    result: dict[int, Mapping[tuple[int, int], np.ndarray]] = {}
    for seed in (42, 2026, 3407):
        folds: dict[tuple[int, int], np.ndarray] = {}
        for fold in contract.folds:
            baseline_frame = pd.read_csv(verified.fold_predictions[fold])
            baseline = baseline_frame["probability"].to_numpy(dtype="float64")
            f_job = by_key[(f_head, seed, *fold)]
            f_probability = _job_result(state, f_job).predictions["probability"].to_numpy(dtype="float64")
            r_probability = baseline
            if include_r:
                r_job = by_key[("r_expert", seed, *fold)]
                r_probability = _job_result(state, r_job).predictions["probability"].to_numpy(dtype="float64")
            folds[fold] = route_probability(
                baseline,
                r_probability,
                f_probability,
                baseline_frame["game_type"].astype(str).to_numpy(),
                alpha_r,
                alpha_f,
            )
        result[seed] = MappingProxyType(folds)
    return MappingProxyType(result)


def _publish_terminal(
    state: RFCampaignState,
    bindings: RFBindings,
    *,
    acceptance_status: str,
    delivery: Path | None,
) -> RFCampaignResult:
    bundles = state.root / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    review = create_rf_review(state.root, bundles / "tree_expert_rf_review.zip", bindings)
    resume = create_rf_resume(state.root, bundles / "tree_expert_rf_resume.zip", bindings)
    return RFCampaignResult(state.status, acceptance_status, review, resume, delivery)


def run_rf_campaign(
    verified: VerifiedRFInput,
    data: object,
    output_dir: Path,
    *,
    resume_bundle: Path | None = None,
    wall_deadline: float | None = None,
    clock: Clock = time.monotonic,
) -> RFCampaignResult:
    contract = load_rf_contract()
    bindings = campaign_bindings(verified, data)
    output = Path(output_dir)
    if resume_bundle is not None:
        restored = restore_rf_resume(resume_bundle, output, bindings)
        state = load_campaign_state(restored.root)
    else:
        output.mkdir(parents=True, exist_ok=False)
        state = load_campaign_state(output)
    train = pd.read_csv(Path(getattr(data, "train")))
    baseline_by_year = {
        fold[1]: pd.read_csv(path) for fold, path in verified.fold_predictions.items()
    }
    log_path = output / "tree_expert_rf.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def execute(job: RFJob, job_dir: Path, gpu_id: int) -> RFJobResult:
        if job.job_id in state.completed_jobs:
            log(f"TREE_RF_JOB_REUSED job={job.job_id}")
            return _job_result(state, job)
        log(f"TREE_RF_JOB_START job={job.job_id} gpu={gpu_id}")
        result = run_rf_job(
            job=job,
            train=train,
            valid=train,
            baseline=baseline_by_year[job.valid_year],
            output_dir=job_dir,
            gpu_id=gpu_id,
            contract=contract,
        )
        log(f"TREE_RF_JOB_END job={job.job_id} status={result.status}")
        return result

    deadline = wall_deadline if wall_deadline is not None else clock() + contract.wall_seconds
    snapshot_path = output.parent / "snapshots/tree_expert_rf_resume.zip"

    def snapshot() -> None:
        create_rf_resume(output, snapshot_path, bindings)

    potential = structure_jobs(contract)
    selection_folds = set(contract.folds[:2])
    selection_jobs = tuple(
        job for job in potential if (job.train_end_year, job.valid_year) in selection_folds
    )
    log("TREE_RF_STAGE_SELECTED stage=structure_selection")
    run_pending_jobs(
        state,
        selection_jobs,
        executor=execute,
        gpu_ids=(0, 1),
        wall_deadline=deadline,
        clock=clock,
        new_job_guard_seconds=contract.new_job_guard_seconds,
        on_progress=snapshot,
    )
    if any(job.job_id not in state.completed_jobs for job in selection_jobs):
        state.phase, state.status = "paused", "paused"
        write_campaign_state(state)
        log("TREE_RF_DECISION status=paused reason=selection_incomplete")
        return _publish_terminal(state, bindings, acceptance_status="paused", delivery=None)

    partial = _evidence(
        state,
        verified,
        selection_jobs,
        contract.folds[:2],
        fill_missing_with_baseline=False,
    )
    survivors = screen_structure_heads(partial, contract)
    if not {"f_small", "f_wide"}.intersection(survivors):
        state.phase, state.status = "rejected", "rejected"
        _atomic_json(output / "decisions/acceptance.json", {"status": "rejected", "reason": "no_F_head_survived"})
        write_campaign_state(state)
        log("TREE_RF_DECISION status=rejected reason=no_F_head_survived")
        return _publish_terminal(state, bindings, acceptance_status="rejected", delivery=None)

    holdout_fold = contract.folds[-1]
    holdout_jobs = tuple(
        job for job in potential
        if (job.train_end_year, job.valid_year) == holdout_fold and job.head in survivors
    )
    log("TREE_RF_STAGE_SELECTED stage=structure_holdout")
    run_pending_jobs(
        state,
        holdout_jobs,
        executor=execute,
        gpu_ids=(0, 1),
        wall_deadline=deadline,
        clock=clock,
        new_job_guard_seconds=contract.new_job_guard_seconds,
        on_progress=snapshot,
    )
    if any(job.job_id not in state.completed_jobs for job in holdout_jobs):
        state.phase, state.status = "paused", "paused"
        write_campaign_state(state)
        log("TREE_RF_DECISION status=paused reason=holdout_incomplete")
        return _publish_terminal(state, bindings, acceptance_status="paused", delivery=None)

    structure_evidence = _evidence(
        state,
        verified,
        potential,
        contract.folds,
        fill_missing_with_baseline=True,
    )
    structure = select_rf_structure(structure_evidence, contract, allowed_heads=survivors)
    _atomic_json(output / "decisions/structure.json", structure_payload(structure))
    log(f"TREE_RF_DECISION status={structure.status} reason={structure.reason}")
    if structure.status != "passed":
        state.phase, state.status = "rejected", "rejected"
        _atomic_json(output / "decisions/acceptance.json", {"status": "rejected", "reason": "structure_rejected"})
        write_campaign_state(state)
        return _publish_terminal(state, bindings, acceptance_status="rejected", delivery=None)

    state.phase = "confirmation"
    write_campaign_state(state)
    confirmation = confirmation_jobs(
        contract,
        f_head=structure.f_head,
        include_r=structure.include_r,
    )
    run_pending_jobs(
        state,
        confirmation,
        executor=execute,
        gpu_ids=(0, 1),
        wall_deadline=deadline,
        clock=clock,
        new_job_guard_seconds=contract.new_job_guard_seconds,
        on_progress=snapshot,
    )
    if any(job.job_id not in state.completed_jobs for job in confirmation):
        state.phase, state.status = "paused", "paused"
        write_campaign_state(state)
        log("TREE_RF_DECISION status=paused reason=confirmation_incomplete")
        return _publish_terminal(state, bindings, acceptance_status="paused", delivery=None)

    all_jobs = (*potential, *confirmation)
    seed_candidate = _seed_candidates(
        state,
        verified,
        all_jobs,
        f_head=structure.f_head,
        include_r=structure.include_r,
        alpha_r=structure.alpha_r,
        alpha_f=structure.alpha_f,
        contract=contract,
    )
    acceptance = accept_rf(
        RFAcceptanceEvidence(
            structure=structure,
            target=structure_evidence.target,
            baseline=structure_evidence.baseline,
            game_type=structure_evidence.game_type,
            seed_candidate=seed_candidate,
        ),
        contract,
    )
    _atomic_json(output / "decisions/acceptance.json", acceptance_payload(acceptance))
    joined_frame = pd.concat(
        [
            pd.read_csv(verified.fold_predictions[fold])[["row_id", "target", "game_type"]]
            for fold in contract.folds
        ],
        ignore_index=True,
    )
    joined_base = np.concatenate([structure_evidence.baseline[fold] for fold in contract.folds])
    mean_candidate = np.concatenate([
        np.mean([seed_candidate[seed][fold] for seed in (42, 2026, 3407)], axis=0)
        for fold in contract.folds
    ])
    _atomic_json(
        output / "diagnostics/rf_diagnostics.json",
        build_rf_diagnostics(
            joined_frame,
            joined_base,
            mean_candidate,
            minimum_rows=contract.gates.minimum_segment_rows,
        ),
    )
    log(f"TREE_RF_DECISION status={acceptance.status} reason={acceptance.reason}")
    if acceptance.status != "accepted":
        state.phase, state.status = "rejected", "rejected"
        write_campaign_state(state)
        return _publish_terminal(state, bindings, acceptance_status="rejected", delivery=None)

    try:
        require_full_fit_window(
            wall_deadline=deadline,
            clock=clock,
            guard_seconds=contract.full_fit_guard_seconds,
        )
    except RFRunnerError:
        state.phase, state.status = "paused", "paused"
        write_campaign_state(state)
        return _publish_terminal(state, bindings, acceptance_status="accepted", delivery=None)

    state.phase = "full_fit"
    write_campaign_state(state)
    heads = (structure.f_head, "r_expert") if structure.include_r else (structure.f_head,)
    full_fit = full_fit_rf(
        decision=acceptance,
        train=train,
        output_dir=_prepare_full_fit_output(output),
        iteration_evidence=_iteration_evidence(state, all_jobs, heads),
        gpu_ids=(0, 1),
        contract=contract,
    )
    inference = load_rf_inference_runtime(
        baseline_delivery_root=verified.full_fit_root,
        rf_full_fit_root=full_fit.root,
    )
    audit_rows = train.drop(columns="control_success").head(32).copy()
    audit = audit_row_independence(
        audit_rows,
        inference,
        tolerance=contract.probability_tolerance,
    )
    _atomic_json(output / "audits/independence.json", audit)
    if audit["status"] != "passed":
        state.phase, state.status = "rejected", "rejected"
        write_campaign_state(state)
        return _publish_terminal(state, bindings, acceptance_status="rejected", delivery=None)
    state.phase, state.status = "completed", "accepted"
    write_campaign_state(state)
    bundles = output / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    delivery = create_rf_delivery(
        output,
        bundles / "tree_expert_rf_delivery.zip",
        acceptance,
        bindings,
    )
    log("TREE_RF_CAMPAIGN_SUCCESS status=accepted")
    return _publish_terminal(state, bindings, acceptance_status="accepted", delivery=delivery)
