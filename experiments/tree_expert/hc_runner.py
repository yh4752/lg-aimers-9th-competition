from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Mapping, Protocol

from .hc_artifacts import (
    create_handoff,
    create_review_bundle,
    create_resume_bundle,
)
from .hc_state import CampaignState


class HCRunnerError(ValueError):
    """Raised when an HC stage cannot run or publish safely."""


@dataclass(frozen=True)
class HCStageOutcome:
    state: CampaignState
    status: str
    review_sources: Mapping[str, Path]
    acceptance: Path | None
    delivery: Path | None


@dataclass(frozen=True)
class CampaignResult:
    status: str
    completed_stage: str
    next_stage: str
    review: Path
    resume: Path
    handoff: Path
    delivery: Path | None


class HCRuntime(Protocol):
    def run_stage(
        self,
        stage: str,
        state: CampaignState,
        deadline: float,
        gpu_count: int,
    ) -> HCStageOutcome: ...


def _append(path: Path, message: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
        handle.flush()


def _validate_h3_state(state: CampaignState) -> None:
    winner = state.decisions.get("winner")
    if winner is None or winner.get("status") != "accepted" or winner.get("candidate") not in {"C1", "C2"}:
        raise HCRunnerError("H3 requires an accepted winner")


def run_campaign(
    *,
    runtime: HCRuntime,
    state: CampaignState,
    campaign_root: Path,
    bundle_root: Path,
    log_path: Path,
    wall_deadline: float,
    gpu_count: int,
    clock: Callable[[], float] = time.time,
) -> CampaignResult:
    if gpu_count not in {1, 2}:
        raise HCRunnerError("GPU count must be one or two")
    if state.stage not in {"H1", "H2", "H3"}:
        raise HCRunnerError("campaign stage differs")
    if state.stage == "H3":
        _validate_h3_state(state)
    if clock() >= wall_deadline:
        raise TimeoutError("campaign stage deadline reached")
    completed_stage = state.stage
    _append(log_path, f"TREE_HC_STAGE_SELECTED stage={completed_stage}")
    outcome = runtime.run_stage(completed_stage, state, wall_deadline, gpu_count)
    if outcome.state.bindings != state.bindings:
        raise HCRunnerError("stage outcome bindings differ")
    if completed_stage == "H1" and outcome.state.stage != "H2":
        raise HCRunnerError("H1 outcome did not advance to H2")
    if completed_stage == "H2" and outcome.state.stage not in {"H2", "H3"}:
        raise HCRunnerError("H2 outcome stage differs")
    if completed_stage == "H3" and outcome.delivery is None:
        raise HCRunnerError("H3 outcome is missing model delivery")
    root = Path(campaign_root)
    bundles = Path(bundle_root)
    bundles.mkdir(parents=True, exist_ok=True)
    review = create_review_bundle(
        sources=outcome.review_sources,
        state=outcome.state,
        output=bundles / "tree_hierarchical_review.zip",
    )
    resume = create_resume_bundle(
        root,
        bundles / "tree_hierarchical_resume.zip",
        outcome.state.bindings,
        outcome.state,
    )
    handoff = create_handoff(
        review=review,
        resume=resume,
        log=Path(log_path),
        output=bundles / "tree_hierarchical_handoff.zip",
        status=outcome.status,
        acceptance=outcome.acceptance,
        delivery=outcome.delivery,
    )
    _append(
        log_path,
        f"TREE_HC_ARTIFACT_READY path={handoff}",
    )
    return CampaignResult(
        status=outcome.status,
        completed_stage=completed_stage,
        next_stage=outcome.state.stage,
        review=review,
        resume=resume,
        handoff=handoff,
        delivery=outcome.delivery,
    )
