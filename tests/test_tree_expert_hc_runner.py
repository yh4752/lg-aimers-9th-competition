from pathlib import Path

import pytest

from experiments.tree_expert.hc_runner import (
    CampaignResult,
    HCStageOutcome,
    HCRunnerError,
    run_campaign,
)
from experiments.tree_expert.hc_state import (
    HCBindings,
    advance_stage,
    initial_state,
    record_decision,
)


def _bindings():
    return HCBindings(*tuple(str(index) * 64 for index in range(1, 7)))


class _Runtime:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.called = []

    def run_stage(self, stage, state, deadline, gpu_count):
        self.called.append((stage, gpu_count))
        evidence = self.tmp_path / f"{stage}.json"
        evidence.write_text('{"status":"completed"}')
        if stage == "H1":
            next_state = advance_stage(state, "H2")
        elif stage == "H2":
            accepted = record_decision(
                state, "winner", {"status": "accepted", "candidate": "C1"}
            )
            next_state = advance_stage(accepted, "H3")
        else:
            next_state = state
        return HCStageOutcome(
            state=next_state,
            status="completed",
            review_sources={f"evidence/{stage}.json": evidence},
            acceptance=None,
            delivery=None,
        )


def test_fresh_campaign_runs_only_h1_and_publishes_handoff(tmp_path: Path):
    runtime = _Runtime(tmp_path)
    result = run_campaign(
        runtime=runtime,
        state=initial_state(_bindings()),
        campaign_root=tmp_path / "campaign",
        bundle_root=tmp_path / "bundles",
        log_path=tmp_path / "campaign.log",
        wall_deadline=10**12,
        gpu_count=2,
    )
    assert isinstance(result, CampaignResult)
    assert runtime.called == [("H1", 2)]
    assert result.completed_stage == "H1"
    assert result.next_stage == "H2"
    assert result.handoff.name == "tree_hierarchical_handoff.zip"


def test_resumed_h2_runs_only_h2(tmp_path: Path):
    runtime = _Runtime(tmp_path)
    state = advance_stage(initial_state(_bindings()), "H2")
    result = run_campaign(
        runtime=runtime,
        state=state,
        campaign_root=tmp_path / "campaign",
        bundle_root=tmp_path / "bundles",
        log_path=tmp_path / "campaign.log",
        wall_deadline=10**12,
        gpu_count=2,
    )
    assert runtime.called == [("H2", 2)]
    assert result.next_stage == "H3"


def test_h3_refuses_state_without_accepted_winner(tmp_path: Path):
    state = advance_stage(initial_state(_bindings()), "H2")
    object.__setattr__(state, "stage", "H3")
    with pytest.raises(HCRunnerError, match="accepted winner"):
        run_campaign(
            runtime=_Runtime(tmp_path),
            state=state,
            campaign_root=tmp_path / "campaign",
            bundle_root=tmp_path / "bundles",
            log_path=tmp_path / "campaign.log",
            wall_deadline=10**12,
            gpu_count=2,
        )


def test_deadline_before_stage_does_not_call_runtime(tmp_path: Path):
    runtime = _Runtime(tmp_path)
    with pytest.raises(TimeoutError, match="deadline"):
        run_campaign(
            runtime=runtime,
            state=initial_state(_bindings()),
            campaign_root=tmp_path / "campaign",
            bundle_root=tmp_path / "bundles",
            log_path=tmp_path / "campaign.log",
            wall_deadline=0,
            gpu_count=2,
            clock=lambda: 1.0,
        )
    assert runtime.called == []


def test_gpu_count_must_be_one_or_two(tmp_path: Path):
    with pytest.raises(HCRunnerError, match="GPU count"):
        run_campaign(
            runtime=_Runtime(tmp_path),
            state=initial_state(_bindings()),
            campaign_root=tmp_path / "campaign",
            bundle_root=tmp_path / "bundles",
            log_path=tmp_path / "campaign.log",
            wall_deadline=10**12,
            gpu_count=0,
        )
