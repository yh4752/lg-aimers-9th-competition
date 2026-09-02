from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Protocol

from .artifacts import E3Bindings, create_bundle, extract_bundle
from .contracts import load_contract
from .state import CampaignState, complete_job, fail_job, initial_state, load_state, save_state, transition
from .training import E3Job


class E3RunnerError(RuntimeError):
    pass


class CampaignRuntime(Protocol):
    bindings: E3Bindings

    def now(self) -> float: ...
    def run_oof_job(self, job: E3Job, gpu: int, output: Path) -> None: ...
    def select_recipe(self, output: Path) -> str: ...
    def decide(self, output: Path, recipe_id: str) -> str: ...
    def run_full_fit(self, role_id: str, seed: int, gpu: int, output: Path) -> None: ...
    def audit(self, output: Path) -> bool: ...


@dataclass(frozen=True)
class CampaignResult:
    status: str
    decision: str | None
    review: Path
    handoff: Path
    state: CampaignState


def _oof_jobs(phase: str) -> tuple[E3Job, ...]:
    contract = load_contract()
    roles = tuple(role.role_id for role in contract.roles)
    if phase == "screening":
        folds = tuple(fold for fold in contract.folds if fold[1] in contract.selection_years)
        seeds = (3407,)
    elif phase == "confirmation":
        folds = tuple(fold for fold in contract.folds if fold[1] == contract.confirmation_year)
        seeds = (3407,)
    elif phase == "extra_seeds":
        folds = contract.folds
        seeds = tuple(seed for seed in contract.seeds if seed != 3407)
    else:
        raise E3RunnerError("OOF phase differs")
    return tuple(
        E3Job(
            f"{phase}__{role}__tr{fold[0]}__va{fold[1]}__s{seed}",
            role,
            fold,
            seed,
            phase,
        )
        for fold in folds
        for seed in seeds
        for role in roles
    )


def _full_fit_jobs() -> tuple[tuple[str, int, str], ...]:
    contract = load_contract()
    experts = tuple(
        (role.role_id, seed, f"full_fit__{role.role_id}__s{seed}")
        for seed in contract.seeds
        for role in contract.roles
    )
    gates = tuple(("GATE", seed, f"full_fit__GATE__s{seed}") for seed in contract.seeds)
    return experts + gates


def _restore(previous: Path, root: Path, bindings: E3Bindings) -> None:
    restored = extract_bundle(previous, root / ".restored_handoff", "handoff", bindings)
    for source in sorted(restored.rglob("*")):
        if not source.is_file() or source.name == "manifest.json":
            continue
        relative = source.relative_to(restored)
        target = root / relative
        if target.exists() or target.is_symlink():
            raise E3RunnerError(f"resume target already exists: {relative.as_posix()}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _payloads(root: Path, *, review: bool) -> dict[str, Path]:
    payloads: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".restored_handoff" in path.parts or "bundles" in path.parts:
            continue
        if "catboost_info" in path.parts or path.suffix == ".snapshot":
            continue
        if "jobs" in path.parts and path.suffix == ".cbm":
            # Completed OOF models are never used after their bound predictions
            # are materialized. Omitting them keeps a restart handoff practical.
            continue
        relative = path.relative_to(root).as_posix()
        if review and (
            path.suffix in {".cbm", ".bin", ".snapshot"}
        ):
            continue
        payloads[relative] = path
    if "campaign/state.json" not in payloads:
        raise E3RunnerError("campaign state is absent")
    return payloads


def _bundles(root: Path, bindings: E3Bindings) -> tuple[Path, Path]:
    bundles = root / "bundles"
    review = create_bundle("review", _payloads(root, review=True), bundles / "failure_regime_e3_review.zip", bindings)
    handoff = create_bundle("handoff", _payloads(root, review=False), bundles / "failure_regime_e3_handoff.zip", bindings)
    return review, handoff


def _halted(
    root: Path,
    runtime: CampaignRuntime,
    state: CampaignState,
    decision: str | None,
    status: str,
) -> CampaignResult:
    save_state(root / "campaign" / "state.json", state)
    review, handoff = _bundles(root, runtime.bindings)
    return CampaignResult(status, decision, review, handoff, state)


def _deadline(runtime: CampaignRuntime, absolute_deadline: float, guard: int) -> bool:
    capacity = getattr(runtime, "can_start_job", None)
    return runtime.now() + guard >= absolute_deadline or (callable(capacity) and not capacity())


def _run_oof_phase(
    runtime: CampaignRuntime,
    root: Path,
    state: CampaignState,
    absolute_deadline: float,
    guard: int,
) -> tuple[CampaignState, str]:
    pending = [job for job in _oof_jobs(state.phase) if job.job_id not in state.completed_jobs]
    while pending:
        if _deadline(runtime, absolute_deadline, guard):
            return state, "paused"
        batch, pending = pending[:2], pending[2:]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(runtime.run_oof_job, job, gpu, root / "jobs" / job.job_id)
                for gpu, job in enumerate(batch)
            ]
            completed: list[E3Job] = []
            failed = False
            for job, future in zip(batch, futures):
                try:
                    future.result()
                except Exception as error:
                    failed = True
                    state = fail_job(state, job.job_id, f"{type(error).__name__}:{error}")
                else:
                    completed.append(job)
                    state = complete_job(state, job.job_id)
        save_state(root / "campaign" / "state.json", state)
        compact = getattr(runtime, "compact_oof_job", None)
        if callable(compact):
            for job in completed:
                compact(job, root / "jobs" / job.job_id)
        if failed:
            return state, "failed"
    return state, "completed"


def _run_full_fit_phase(
    runtime: CampaignRuntime,
    root: Path,
    state: CampaignState,
    absolute_deadline: float,
    guard: int,
) -> tuple[CampaignState, str]:
    pending = [job for job in _full_fit_jobs() if job[2] not in state.completed_jobs]
    while pending:
        if _deadline(runtime, absolute_deadline, guard):
            return state, "paused"
        batch, pending = pending[:2], pending[2:]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(runtime.run_full_fit, role, seed, gpu, root / "full_fit" / job_id)
                for gpu, (role, seed, job_id) in enumerate(batch)
            ]
            failed = False
            for (_, _, job_id), future in zip(batch, futures):
                try:
                    future.result()
                except Exception as error:
                    failed = True
                    state = fail_job(state, job_id, f"{type(error).__name__}:{error}")
                else:
                    state = complete_job(state, job_id)
        save_state(root / "campaign" / "state.json", state)
        if failed:
            return state, "failed"
    return state, "completed"


def run_campaign(
    runtime: CampaignRuntime,
    output_dir: Path,
    *,
    absolute_deadline: float,
    previous_handoff: Path | None = None,
    new_job_guard_seconds: int | None = None,
) -> CampaignResult:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "campaign" / "state.json"
    if previous_handoff is not None:
        _restore(Path(previous_handoff), root, runtime.bindings)
    state = load_state(state_path) if state_path.is_file() else initial_state()
    save_state(state_path, state)
    guard = int(new_job_guard_seconds or load_contract().runtime["new_job_guard_seconds"])
    decision: str | None = None

    for phase in ("screening", "confirmation", "extra_seeds"):
        if state.phase != phase:
            continue
        state, outcome = _run_oof_phase(runtime, root, state, absolute_deadline, guard)
        if outcome != "completed":
            return _halted(root, runtime, state, decision, outcome)
        if phase == "screening":
            recipe = runtime.select_recipe(root / "selection" / "recipe.json")
            state = transition(state, "confirmation", recipe_id=recipe)
        elif phase == "confirmation":
            state = transition(state, "extra_seeds")
        else:
            state = transition(state, "decision")
        save_state(state_path, state)

    if state.phase == "decision":
        if state.recipe_id is None:
            raise E3RunnerError("selected recipe is absent")
        decision = runtime.decide(root / "selection" / "decision.json", state.recipe_id)
        if decision not in {"accepted", "rejected"}:
            raise E3RunnerError("candidate decision differs")
        if decision == "rejected":
            review, handoff = _bundles(root, runtime.bindings)
            return CampaignResult("rejected", decision, review, handoff, state)
        state = transition(state, "full_fit")
        save_state(state_path, state)

    if state.phase == "full_fit":
        decision = "accepted"
        state, outcome = _run_full_fit_phase(runtime, root, state, absolute_deadline, guard)
        if outcome != "completed":
            return _halted(root, runtime, state, decision, outcome)
        state = transition(state, "audit")
        save_state(state_path, state)

    if state.phase == "audit":
        decision = "accepted"
        if not runtime.audit(root / "audit" / "inference_audit.json"):
            review, handoff = _bundles(root, runtime.bindings)
            return CampaignResult("audit_failed", decision, review, handoff, state)
        state = transition(state, "completed")
        save_state(state_path, state)

    if state.phase != "completed":
        raise E3RunnerError("campaign phase is unresolved")
    review, handoff = _bundles(root, runtime.bindings)
    return CampaignResult("completed", decision or "accepted", review, handoff, state)
