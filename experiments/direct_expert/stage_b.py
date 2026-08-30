from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from .artifacts import DirectExpertBindings, create_bundle, extract_bundle
from .contracts import ExpertJob, load_contract
from .inputs import canonical_json
from .state import complete_job, fail_job, initial_state, save_state, transition


class StageBRuntime(Protocol):
    bindings: DirectExpertBindings
    def now(self) -> float: ...
    def locked_experts(self, handoff: Path) -> tuple[str, ...]: ...
    def run_job(self, job: ExpertJob, gpu: int, output: Path) -> Path: ...
    def decide(self, jobs: Mapping[str, Path]) -> Mapping[str, object]: ...
    def full_fit(self, decision: Mapping[str, object], output: Path) -> Mapping[str, Path]: ...
    def audit(self, payloads: Mapping[str, Path]) -> bool: ...


@dataclass(frozen=True)
class StageBResult:
    review: Path
    handoff: Path
    delivery: Path | None
    status: str


def _jobs(experts: tuple[str, ...]) -> tuple[ExpertJob, ...]:
    if len(experts) != 4 or len(set(experts)) != 4:
        raise ValueError("locked expert count differs")
    confirmation = tuple(
        ExpertJob(f"confirm__{expert}__2023_2024__s3407", expert, (2023, 2024), 3407, "confirmation")
        for expert in experts
    )
    extra = tuple(
        ExpertJob(
            f"extra__{expert}__{train}_{valid}__s{seed}",
            expert,
            (train, valid),
            seed,
            "extra_seeds",
        )
        for train, valid in ((2021, 2022), (2022, 2023), (2023, 2024))
        for expert in experts
        for seed in (42, 2026)
    )
    jobs = (*confirmation, *extra)
    if len(jobs) != 28 or 16 + len(jobs) > 44:
        raise ValueError("OOF job budget differs")
    return jobs


def run_stage_b(
    runtime: StageBRuntime,
    stage_a_handoff: Path,
    output_dir: Path,
    *,
    absolute_deadline: float,
    previous_handoff: Path | None = None,
) -> StageBResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experts = runtime.locked_experts(Path(stage_a_handoff))
    jobs = _jobs(experts)
    state = initial_state("stage_b")
    completed: dict[str, Path] = {}
    failures = {}
    guard = int(load_contract().runtime["new_job_guard_seconds"])
    if previous_handoff is not None:
        restored = output / "restored_stage_b"
        extract_bundle(previous_handoff, restored, "handoff", runtime.bindings)
        if hasattr(runtime, "restore_job"):
            for job in jobs:
                folder = restored / "jobs" / job.job_id
                if (folder / "predictions.csv").is_file() and (folder / "metrics.json").is_file():
                    completed[job.job_id] = Path(runtime.restore_job(job, folder))
                    state = complete_job(state, job.job_id)
    for phase in ("confirmation", "extra_seeds"):
        phase_jobs = [job for job in jobs if job.phase == phase and job.job_id not in completed]
        if phase_jobs and runtime.now() + guard >= absolute_deadline:
            break
        def execute_gpu(gpu: int, assigned: list[ExpertJob]):
            observed = []
            for job in assigned:
                if runtime.now() + guard >= absolute_deadline:
                    break
                try:
                    observed.append((job, Path(runtime.run_job(job, gpu, output / "jobs" / job.job_id)), None))
                except Exception as error:
                    observed.append((job, None, error))
            return observed
        if phase_jobs:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(execute_gpu, gpu, phase_jobs[gpu::2]) for gpu in (0, 1)]
                for future in futures:
                    for job, path, error in future.result():
                        if error is None:
                            if not path.is_file():
                                raise ValueError("worker result is absent")
                            completed[job.job_id] = path
                            state = complete_job(state, job.job_id)
                        else:
                            failures[job.job_id] = f"{type(error).__name__}: {error}"
                            state = fail_job(state, job.job_id, failures[job.job_id])
        if any(job.job_id not in completed for job in jobs if job.phase == phase):
            break
        if phase == "confirmation":
            state = transition(state, "extra_seeds")
    decision: Mapping[str, object] = {"candidate_id": None, "status": "rejected", "reason": "incomplete"}
    delivery = None
    if len(completed) == len(jobs):
        state = transition(state, "stacking")
        state = transition(state, "decision")
        try:
            decision = runtime.decide(completed)
        except Exception as error:
            decision = {
                "candidate_id": None,
                "status": "rejected",
                "reason": f"decision_failed:{type(error).__name__}:{error}",
            }
        if decision.get("status") in {"accepted_stable", "accepted_aggressive"}:
            full_fit_guard = int(load_contract().runtime.get("full_fit_guard_seconds", guard))
            if runtime.now() + full_fit_guard < absolute_deadline:
                state = transition(state, "full_fit")
                try:
                    payloads = runtime.full_fit(decision, output / "full_fit")
                    model_members = [name for name in payloads if name.startswith("models/")]
                    if len(model_members) > 9:
                        raise ValueError("full-fit model budget exceeded")
                except Exception as error:
                    payloads = None
                    decision = {
                        **decision,
                        "status": "accepted_full_fit_failed",
                        "reason": f"full_fit_failed:{type(error).__name__}:{error}",
                    }
                state = transition(state, "audit")
                if payloads is not None:
                    try:
                        if runtime.audit(payloads) is not True:
                            raise ValueError("audit returned false")
                        delivery = create_bundle(
                            "delivery",
                            payloads,
                            output / "direct_expert_delivery.zip",
                            runtime.bindings,
                        )
                    except Exception as error:
                        decision = {
                            **decision,
                            "status": "rejected",
                            "reason": f"audit_or_delivery_failed:{type(error).__name__}:{error}",
                        }
                state = transition(state, "completed")
            else:
                decision = {**decision, "status": "accepted_pending_full_fit"}
        else:
            state = transition(state, "full_fit")
            state = transition(state, "audit")
            state = transition(state, "completed")
    state_path = save_state(output / "stage_b_state.json", state)
    decision_path = output / "decision.json"
    decision_path.write_bytes(canonical_json(dict(decision)))
    review = create_bundle(
        "review",
        {"state/state.json": state_path, "decision.json": decision_path},
        output / "direct_expert_review.zip",
        runtime.bindings,
    )
    handoff_payloads = {
        "state/state.json": state_path,
        "decision.json": decision_path,
    }
    for name, path in completed.items():
        for member in ("predictions.csv", "metrics.json", "job_identity.json", "worker.log"):
            source = path.parent / member
            if source.is_file():
                handoff_payloads[f"jobs/{name}/{member}"] = source
    handoff = create_bundle(
        "handoff",
        handoff_payloads,
        output / "direct_expert_handoff.zip",
        runtime.bindings,
    )
    return StageBResult(review, handoff, delivery, str(decision.get("status")))
