from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .hetero_artifacts import (
    HeteroBindings,
    create_handoff_bundle,
    create_resume_bundle,
    create_review_bundle,
    restore_resume_bundle,
)
from .hetero_contracts import (
    HeteroJob,
    confirmation_jobs,
    contract_sha256,
    load_hetero_contract,
    structure_jobs,
)
from .hetero_decisions import (
    FamilyDecision,
    FamilyEvidence,
    confirm_family,
    decision_payload,
    decide_equal_blend,
    select_family_structure,
)
from .hetero_state import (
    HeteroState,
    advance_phase,
    initial_state,
    load_state,
    mark_job,
    record_decision,
    save_state,
)
from .hetero_training import HeteroJobResult, run_hetero_job
from .t3_inputs import VerifiedOfficialData, VerifiedT3Input


class HeteroRunnerError(RuntimeError):
    pass


JobExecutor = Callable[[HeteroJob, Path, int], object]
Clock = Callable[[], float]
Fold = tuple[int, int]


@dataclass(frozen=True)
class HeteroCampaignResult:
    status: str
    decisions: Mapping[str, str]
    review_bundle: Path
    resume_bundle: Path
    handoff_bundle: Path


def _status(value: object) -> str:
    if isinstance(value, HeteroJobResult):
        return value.status
    if type(value) is str:
        return value
    return str(getattr(value, "status", "failed"))


def run_pending_jobs(
    state: HeteroState,
    jobs: Iterable[HeteroJob],
    *,
    root: Path,
    executor: JobExecutor,
    gpu_ids: tuple[int, int],
    wall_deadline: float,
    clock: Clock = time.monotonic,
    new_job_guard_seconds: int = 600,
    on_progress: Callable[[HeteroState], None] | None = None,
) -> HeteroState:
    if type(state) is not HeteroState or len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise HeteroRunnerError("job scheduler configuration differs")
    pending = [job for job in jobs if job.job_id not in state.completed_jobs]
    current = state
    for offset in range(0, len(pending), 2):
        if clock() + new_job_guard_seconds >= wall_deadline:
            break
        batch = pending[offset:offset + 2]
        with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="hetero") as pool:
            futures = {
                pool.submit(executor, job, Path(root) / "jobs" / job.job_id, gpu_ids[index]): job
                for index, job in enumerate(batch)
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    status = _status(future.result())
                except Exception:
                    status = "failed"
                current = mark_job(current, job.job_id, "completed" if status == "completed" else "failed")
                save_state(current, Path(root) / "state/state.json")
        if on_progress:
            on_progress(current)
    return current


_CODE_MEMBERS = (
    "experiments/tree_expert/contracts.py",
    "experiments/tree_expert/features.py",
    "experiments/tree_expert/hetero_artifacts.py",
    "experiments/tree_expert/hetero_contract.json",
    "experiments/tree_expert/hetero_contracts.py",
    "experiments/tree_expert/hetero_decisions.py",
    "experiments/tree_expert/hetero_features.py",
    "experiments/tree_expert/hetero_runner.py",
    "experiments/tree_expert/hetero_state.py",
    "experiments/tree_expert/hetero_training.py",
    "experiments/tree_expert/t3_contract.json",
    "experiments/tree_expert/t3_contracts.py",
    "experiments/tree_expert/t3_inputs.py",
    "experiments/temporal_portfolio/seasonal_features.py",
    "experiments/independent_dl/feature_sources/seasonal.py",
)

_COMPATIBLE_RESUME_CODE_SHA256S = frozenset({
    "1804f34d5f48b6568e17ec86f6265e12961d10c1cd1c720c5d2063061ba6bc53",
})


def code_sha256(root: Path | None = None) -> str:
    project = Path(__file__).resolve().parents[2] if root is None else Path(root)
    digest = sha256()
    for name in _CODE_MEMBERS:
        path = project / name
        if not path.is_file():
            raise HeteroRunnerError(f"runtime member is missing: {name}")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def campaign_bindings(verified: VerifiedT3Input, data: VerifiedOfficialData) -> HeteroBindings:
    return HeteroBindings(
        contract_sha256(), code_sha256(), verified.manifest_sha256,
        data.train_sha256, data.history_sha256, verified.e2_handoff_sha256,
    )


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _prediction(root: Path, job: HeteroJob) -> pd.DataFrame:
    path = Path(root) / "jobs" / job.job_id / "predictions.csv"
    try:
        return pd.read_csv(path)
    except Exception as error:
        raise HeteroRunnerError(f"job prediction cannot be read: {job.job_id}") from error


def _family_frames(root: Path, family: str, seed: int) -> Mapping[Fold, pd.DataFrame]:
    contract = load_hetero_contract()
    jobs = (
        [job for job in structure_jobs(contract) if job.family == family]
        if seed == contract.structure_seed
        else list(confirmation_jobs(contract, family))
    )
    selected = [job for job in jobs if job.seed == seed]
    return MappingProxyType({
        (job.train_end_year, job.valid_year): _prediction(root, job)
        for job in selected
    })


def _average_seed_frames(seed_frames: Mapping[int, Mapping[Fold, pd.DataFrame]]) -> Mapping[Fold, pd.DataFrame]:
    contract = load_hetero_contract()
    output: dict[Fold, pd.DataFrame] = {}
    for fold in contract.folds:
        frames = [seed_frames[seed][fold] for seed in sorted(seed_frames)]
        reference = frames[0].copy(deep=True)
        if any(not reference["row_id"].astype(str).equals(frame["row_id"].astype(str)) for frame in frames[1:]):
            raise HeteroRunnerError("seed row alignment differs")
        reference["model_probability"] = np.mean(
            [frame["model_probability"].to_numpy(dtype="float64") for frame in frames], axis=0,
        )
        output[fold] = reference
    return MappingProxyType(output)


def _write_decision(root: Path, name: str, payload: dict[str, object]) -> None:
    path = Path(root) / "decisions" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload))


def _publish(root: Path, bindings: HeteroBindings, log_path: Path) -> tuple[Path, Path, Path]:
    bundles = Path(root) / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    review = create_review_bundle(root, bundles / "tree_hetero_review.zip", bindings)
    resume = create_resume_bundle(root, bundles / "tree_hetero_resume.zip", bindings)
    handoff = create_handoff_bundle(
        review, resume, bundles / "tree_hetero_handoff.zip", bindings,
        log_path if log_path.is_file() else None,
    )
    return review, resume, handoff


def run_hetero_campaign(
    verified: VerifiedT3Input,
    data: VerifiedOfficialData,
    output_dir: Path,
    *,
    resume_bundle: Path | None = None,
    wall_deadline: float | None = None,
    clock: Clock = time.monotonic,
) -> HeteroCampaignResult:
    contract = load_hetero_contract()
    bindings = campaign_bindings(verified, data)
    root = Path(output_dir)
    if resume_bundle is None:
        root.mkdir(parents=True, exist_ok=False)
        state = initial_state()
        save_state(state, root / "state/state.json")
    else:
        restore_resume_bundle(
            resume_bundle,
            root,
            bindings,
            compatible_code_sha256s=_COMPATIBLE_RESUME_CODE_SHA256S,
        )
        state = load_state(root / "state/state.json")
    train = pd.read_csv(data.train)
    baselines = {fold[1]: pd.read_csv(path) for fold, path in verified.fold_predictions.items()}
    log_path = root / "tree_hetero.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def execute(job: HeteroJob, job_dir: Path, gpu_id: int) -> HeteroJobResult:
        log(f"TREE_HETERO_JOB_START job={job.job_id} gpu={gpu_id}")
        result = run_hetero_job(
            job=job, train=train, baseline=baselines[job.valid_year],
            output_dir=job_dir, gpu_id=gpu_id,
        )
        log(f"TREE_HETERO_JOB_END job={job.job_id} status={result.status}")
        return result

    deadline = wall_deadline if wall_deadline is not None else clock() + contract.wall_seconds
    latest = {"state": state}

    def snapshot(current: HeteroState) -> None:
        latest["state"] = current
        create_resume_bundle(root, root.parent / "tree_hetero_stable_resume.zip", bindings)

    structures: dict[str, FamilyDecision] = {}
    if state.phase == "structure":
        state = run_pending_jobs(
            state, structure_jobs(contract), root=root, executor=execute, gpu_ids=(0, 1),
            wall_deadline=deadline, clock=clock,
            new_job_guard_seconds=contract.new_job_guard_seconds, on_progress=snapshot,
        )
        for family in contract.families:
            required = [job for job in structure_jobs(contract) if job.family == family]
            if all(job.job_id in state.completed_jobs for job in required):
                decision = select_family_structure(
                    FamilyEvidence(_family_frames(root, family, contract.structure_seed)), contract, family,
                )
                structures[family] = decision
                state = record_decision(state, family, decision.status)
                _write_decision(root, f"structure_{family}", decision_payload(decision))
                log(f"TREE_HETERO_DECISION family={family} stage=structure status={decision.status}")
            elif any(job.job_id in state.failed_jobs for job in required):
                state = record_decision(state, family, "rejected")
                _write_decision(root, f"structure_{family}", {"family": family, "status": "rejected", "reason": "family_job_failed"})
        if any(job.job_id not in state.completed_jobs and job.job_id not in state.failed_jobs for job in structure_jobs(contract)):
            save_state(state, root / "state/state.json")
            review, resume, handoff = _publish(root, bindings, log_path)
            return HeteroCampaignResult("incomplete", state.decisions, review, resume, handoff)
        state = advance_phase(state, "confirmation")
        save_state(state, root / "state/state.json")
        snapshot(state)

    if state.phase == "confirmation":
        for family in contract.families:
            if state.decisions.get(family) != "passed":
                continue
            structure = structures.get(family) or select_family_structure(
                FamilyEvidence(_family_frames(root, family, contract.structure_seed)), contract, family,
            )
            state = run_pending_jobs(
                state, confirmation_jobs(contract, family), root=root, executor=execute, gpu_ids=(0, 1),
                wall_deadline=deadline, clock=clock,
                new_job_guard_seconds=contract.new_job_guard_seconds, on_progress=snapshot,
            )
            required = confirmation_jobs(contract, family)
            if any(job.job_id not in state.completed_jobs for job in required):
                if any(job.job_id in state.failed_jobs for job in required):
                    state = record_decision(state, family, "rejected")
                    _write_decision(root, f"acceptance_{family}", {"family": family, "status": "rejected", "reason": "family_job_failed"})
                    continue
                save_state(state, root / "state/state.json")
                review, resume, handoff = _publish(root, bindings, log_path)
                return HeteroCampaignResult("incomplete", state.decisions, review, resume, handoff)
            seed_frames = MappingProxyType({
                seed: _family_frames(root, family, seed)
                for seed in (contract.structure_seed, *contract.confirmation_seeds)
            })
            decision = confirm_family(structure, seed_frames, contract)
            structures[family] = decision
            state = record_decision(state, family, decision.status)
            _write_decision(root, f"acceptance_{family}", decision_payload(decision))
            log(f"TREE_HETERO_DECISION family={family} stage=confirmation status={decision.status}")
        accepted = [decision for decision in structures.values() if decision.status == "accepted"]
        if len(accepted) == 2:
            family_frames = {}
            for decision in accepted:
                seeds = {
                    seed: _family_frames(root, decision.family, seed)
                    for seed in (contract.structure_seed, *contract.confirmation_seeds)
                }
                family_frames[decision.family] = _average_seed_frames(MappingProxyType(seeds))
            blend = decide_equal_blend(
                accepted[0], accepted[1], family_frames[accepted[0].family],
                family_frames[accepted[1].family], contract,
            )
            state = record_decision(state, "equal_blend", blend.status)
            _write_decision(root, "acceptance_equal_blend", decision_payload(blend))
            log(f"TREE_HETERO_DECISION family=equal_blend stage=confirmation status={blend.status}")
        state = advance_phase(state, "completed")
        save_state(state, root / "state/state.json")

    review, resume, handoff = _publish(root, bindings, log_path)
    log(f"TREE_HETERO_SUCCESS handoff={handoff}")
    return HeteroCampaignResult("completed", state.decisions, review, resume, handoff)
