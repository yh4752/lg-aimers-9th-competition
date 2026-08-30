from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .artifacts import (
    ArtifactBindings,
    create_delivery,
    create_handoff,
    create_review,
    extract_bundle,
)
from .contracts import load_contract
from .inputs import canonical_json
from .selection import CandidateDecision
from .state import CampaignState, load_state, save_state


class FinalRunnerError(ValueError):
    pass


@dataclass(frozen=True)
class SelectionOutcome:
    decision: CandidateDecision
    evidence_path: Path


@dataclass(frozen=True)
class FullFitOutcome:
    completed: bool
    completed_jobs: tuple[str, ...]
    payloads: Mapping[str, Path]


@dataclass(frozen=True)
class AuditOutcome:
    passed: bool
    report_path: Path


@dataclass(frozen=True)
class CampaignResult:
    status: str
    review: Path
    handoff: Path
    delivery: Path | None


class CampaignRuntime(Protocol):
    bindings: ArtifactBindings

    def now(self) -> float: ...
    def select(self, output: Path) -> SelectionOutcome: ...
    def full_fit(
        self,
        decision: CandidateDecision,
        output: Path,
        *,
        completed_jobs: tuple[str, ...],
        restored_root: Path | None,
        on_job_complete: Callable[[str], None],
        absolute_deadline: float | None,
    ) -> FullFitOutcome: ...
    def audit(self, full_fit: FullFitOutcome, output: Path) -> AuditOutcome: ...


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))
    return path


def _finalize(
    *,
    root: Path,
    runtime: CampaignRuntime,
    state: CampaignState,
    decision_path: Path,
    evidence_path: Path,
    production_payloads: Mapping[str, Path],
    audit: AuditOutcome | None,
    allow_delivery: bool,
) -> CampaignResult:
    save_state(root / "campaign_state.json", state)
    common = {
        "campaign_state.json": root / "campaign_state.json",
        "decision.json": decision_path,
        "evidence.json": evidence_path,
    }
    if audit is not None:
        common["audit/report.json"] = audit.report_path
    bundles = root / "bundles"
    review = create_review(
        bundles / "gated_residual_final_review.zip",
        bindings=runtime.bindings,
        payloads=common,
    )
    handoff = create_handoff(
        bundles / "gated_residual_final_handoff.zip",
        bindings=runtime.bindings,
        payloads={**common, **dict(production_payloads)},
    )
    delivery = None
    if allow_delivery:
        delivery = create_delivery(
            bundles / "gated_residual_final_delivery.zip",
            bindings=runtime.bindings,
            payloads={**common, **dict(production_payloads)},
            acceptance_evidence={"status": "accepted", "candidate_id": state.candidate_id},
        )
    return CampaignResult(state.status, review, handoff, delivery)


def run_campaign(
    *,
    runtime: CampaignRuntime,
    output_dir: Path,
    previous_handoff: Path | None = None,
    absolute_deadline: float | None = None,
) -> CampaignResult:
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        raise FinalRunnerError("campaign output already exists")
    root.mkdir(parents=True)
    restored = None
    if previous_handoff is not None:
        restored = extract_bundle(
            Path(previous_handoff), root / "restored_handoff",
            kind="handoff", expected_bindings=runtime.bindings,
        )
        state = load_state(restored / "campaign_state.json")
    else:
        state = CampaignState("P1", "running", (), None, None)
    print("FINAL_CANDIDATE_STAGE phase=P1", flush=True)
    selection = runtime.select(root / "selection")
    decision = selection.decision
    decision_path = _write_json(root / "decision.json", asdict(decision))
    evidence_path = selection.evidence_path
    state = CampaignState("P2", "running", state.completed_jobs, decision.status, decision.candidate_id)
    save_state(root / "campaign_state.json", state)
    print(
        f"FINAL_CANDIDATE_DECISION status={decision.status} candidate={decision.candidate_id}",
        flush=True,
    )
    if decision.status != "accepted":
        state = CampaignState("P5", "completed_no_candidate", state.completed_jobs, decision.status, decision.candidate_id)
        return _finalize(
            root=root, runtime=runtime, state=state, decision_path=decision_path,
            evidence_path=evidence_path, production_payloads={}, audit=None, allow_delivery=False,
        )
    if absolute_deadline is not None and runtime.now() + load_contract().full_fit_guard_seconds >= absolute_deadline:
        state = CampaignState("P3", "paused", state.completed_jobs, decision.status, decision.candidate_id)
        return _finalize(
            root=root, runtime=runtime, state=state, decision_path=decision_path,
            evidence_path=evidence_path, production_payloads={}, audit=None, allow_delivery=False,
        )
    print("FINAL_CANDIDATE_STAGE phase=P3", flush=True)
    completed = list(state.completed_jobs)

    def on_job_complete(job_id: str) -> None:
        if job_id not in completed:
            completed.append(job_id)
        save_state(
            root / "campaign_state.json",
            CampaignState("P3", "running", tuple(completed), decision.status, decision.candidate_id),
        )
        print(f"FINAL_CANDIDATE_JOB_END job={job_id} status=completed", flush=True)

    full_fit = runtime.full_fit(
        decision, root / "full_fit", completed_jobs=tuple(completed),
        restored_root=restored, on_job_complete=on_job_complete,
        absolute_deadline=absolute_deadline,
    )
    if not full_fit.completed:
        state = CampaignState("P3", "paused", full_fit.completed_jobs, decision.status, decision.candidate_id)
        return _finalize(
            root=root, runtime=runtime, state=state, decision_path=decision_path,
            evidence_path=evidence_path, production_payloads=full_fit.payloads,
            audit=None, allow_delivery=False,
        )
    print("FINAL_CANDIDATE_STAGE phase=P4", flush=True)
    audit = runtime.audit(full_fit, root / "audit")
    if not audit.passed:
        state = CampaignState("P4", "audit_failed", full_fit.completed_jobs, decision.status, decision.candidate_id)
        return _finalize(
            root=root, runtime=runtime, state=state, decision_path=decision_path,
            evidence_path=evidence_path, production_payloads=full_fit.payloads,
            audit=audit, allow_delivery=False,
        )
    print("FINAL_CANDIDATE_STAGE phase=P5", flush=True)
    state = CampaignState("P5", "completed", full_fit.completed_jobs, decision.status, decision.candidate_id)
    return _finalize(
        root=root, runtime=runtime, state=state, decision_path=decision_path,
        evidence_path=evidence_path, production_payloads=full_fit.payloads,
        audit=audit, allow_delivery=True,
    )
