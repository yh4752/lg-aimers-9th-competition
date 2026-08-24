"""Restartable physical-stage runner with a reserved handoff window."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Mapping, Protocol

from experiments.preprocessing_campaign.budgeted_scheduler import SchedulerSummary

from .artifacts import HandoffPath, StageEvidence, canonical_json, write_handoff
from .compatibility import STATE_SCHEMA_VERSION
from .contracts import DEFAULT_CONTRACT, load_contract
from .inputs import VerifiedOfficialData
from .planner import PlannedJob, PriorReview, StagePlan, plan_stage
from .state import Bindings, Lineage
from .worker import verify_worker_result


class PortfolioRunnerError(RuntimeError):
    """Raised when a stage cannot be safely resumed or published."""


_STOP_NEW_SECONDS = 900
_HANDOFF_RESERVE_SECONDS = 600


class Scheduler(Protocol):
    def run(self, jobs: tuple[PlannedJob, ...], output_root: Path, *, deadline: float) -> SchedulerSummary: ...


@dataclass(frozen=True)
class CompletedJob:
    root: Path
    status: str
    identity_sha256: str

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"


@dataclass(frozen=True)
class StageRun:
    status: str
    handoff: HandoffPath
    decision: str
    plan: StagePlan


def verify_completed_job(root: str | Path) -> CompletedJob:
    path = Path(root)
    try:
        payload = verify_worker_result(path)
    except Exception as error:
        raise PortfolioRunnerError("completed job artifact verification failed") from error
    if payload.get("status") != "completed":
        raise PortfolioRunnerError("completed job status differs")
    identity = payload.get("training_identity_sha256")
    if type(identity) is not str:
        raise PortfolioRunnerError("completed job identity is missing")
    return CompletedJob(path, "completed", identity)


def run_stage(
    *,
    stage: str,
    verified: VerifiedOfficialData,
    output_root: str | Path,
    deadline: float,
    scheduler: Scheduler | None = None,
    prior_review: PriorReview | None = None,
    completed: Mapping[str, str] | None = None,
) -> StageRun:
    if type(verified) is not VerifiedOfficialData:
        raise PortfolioRunnerError("verified data identity is invalid")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not float(deadline) > 0:
        raise PortfolioRunnerError("deadline must be a positive Unix timestamp")
    if scheduler is None:
        raise PortfolioRunnerError("a temporal platform scheduler is required")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    input_sha = _input_manifest_sha(verified)
    review = prior_review or PriorReview(input_sha, {})
    if review.input_rows_sha256 != input_sha:
        raise PortfolioRunnerError("prior review input identity differs")
    catalog = _verified_completed_catalog(completed or {})
    plan = plan_stage(stage, prior_review=review, completed=catalog)
    (root / "stage_plan.json").write_bytes(
        canonical_json({"stage": stage, "jobs": [job.job_id for job in plan.jobs], "identities": [job.identity.sha256 for job in plan.jobs]})
    )

    runtime = scheduler
    now = _clock(runtime)
    stage_limit = load_contract().physical_stage_seconds[stage]
    effective_deadline = min(float(deadline), now + stage_limit)
    if effective_deadline - now <= _STOP_NEW_SECONDS + _HANDOFF_RESERVE_SECONDS:
        summary = SchedulerSummary((), tuple(job.job_id for job in plan.jobs), ())
    else:
        summary = runtime.run(plan.jobs, root / "campaign", deadline=effective_deadline)
    status = "completed" if not summary.pending and not summary.failed else "budget_inconclusive"
    decision = "gpu_free_decision_pending" if plan.gpu_free_decision else status
    handoff = _write_stage_handoff(
        root, verified, plan, summary, status, decision, review
    )
    return StageRun(status, handoff, decision, plan)


def _write_stage_handoff(
    root: Path,
    verified: VerifiedOfficialData,
    plan: StagePlan,
    summary: SchedulerSummary,
    status: str,
    decision: str,
    prior_review: PriorReview,
) -> HandoffPath:
    input_sha = _input_manifest_sha(verified)
    contract_sha = sha256(Path(DEFAULT_CONTRACT).read_bytes()).hexdigest()
    bindings = Bindings("temporal_portfolio_v1", contract_sha, input_sha)
    runtime_sha = _runtime_sha()
    lineage = Lineage(
        "temporal_portfolio_v1",
        plan.stage,
        prior_review.sequence + 1,
        prior_review.parent_manifest_sha256 or input_sha,
        plan.scheduler_identity["plan_sha256"],
        STATE_SCHEMA_VERSION,
        runtime_sha,
    )
    summary_payload = {
        "stage": plan.stage,
        "status": status,
        "decision": decision,
        "completed": list(summary.completed),
        "pending": list(summary.pending),
        "failed": list(summary.failed),
    }
    evidence = StageEvidence(
        plan.stage,
        bindings,
        lineage,
        {"review.json": canonical_json(summary_payload)},
        {
            "resume.json": canonical_json(
                {
                    **summary_payload,
                    "planned_identity_sha256": [job.identity.sha256 for job in plan.jobs],
                }
            )
        },
        ("STAGE_RESULT " + json.dumps(summary_payload, sort_keys=True) + "\n").encode(),
        summary_payload,
    )
    return write_handoff(root / "handoff", evidence)


def _input_manifest_sha(verified: VerifiedOfficialData) -> str:
    payload = {
        "members": dict(sorted(verified.member_sha256.items())),
        "train_rows": verified.train_rows,
        "test_rows": verified.test_rows,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verified_completed_catalog(value: Mapping[str, str]) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for claimed_identity, raw_path in value.items():
        completed = verify_completed_job(raw_path)
        if completed.identity_sha256 != claimed_identity:
            raise PortfolioRunnerError("completed job identity differs")
        catalog[claimed_identity] = str(completed.root)
    return catalog


def _runtime_sha() -> str:
    digest = sha256()
    root = Path(__file__).resolve().parent
    for name in ("planner.py", "runner.py", "run_campaign.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _clock(scheduler: object) -> float:
    clock = getattr(scheduler, "clock", None)
    return float(clock()) if callable(clock) else time.time()
