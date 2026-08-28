from dataclasses import FrozenInstanceError

import pytest

from experiments.tree_expert.hc_state import (
    HCBindings,
    HCStateError,
    advance_stage,
    complete_job,
    fail_job,
    initial_state,
    record_decision,
    skip_job,
    start_job,
)


def _bindings():
    return HCBindings(*tuple(str(index) * 64 for index in range(1, 7)))


def test_state_is_immutable_and_completed_jobs_cannot_restart():
    state = initial_state(_bindings())
    running = start_job(state, "job_a")
    completed = complete_job(running, "job_a")
    with pytest.raises(HCStateError, match="immutable"):
        start_job(completed, "job_a")
    with pytest.raises(FrozenInstanceError):
        completed.stage = "H2"


def test_failed_job_can_retry_without_removing_completed_sibling():
    state = complete_job(start_job(initial_state(_bindings()), "job_a"), "job_a")
    failed = fail_job(start_job(state, "job_b"), "job_b")
    retried = start_job(failed, "job_b")
    assert retried.active_job == "job_b"
    assert retried.completed_jobs == ("job_a",)
    assert "job_b" not in retried.failed_jobs


def test_h3_transition_requires_accepted_winner():
    h2 = advance_stage(initial_state(_bindings()), "H2")
    with pytest.raises(HCStateError, match="accepted winner"):
        advance_stage(h2, "H3")
    rejected = record_decision(h2, "winner", {"status": "fallback", "candidate": "C0"})
    with pytest.raises(HCStateError, match="accepted winner"):
        advance_stage(rejected, "H3")
    accepted = record_decision(h2, "winner", {"status": "accepted", "candidate": "C1"})
    assert advance_stage(accepted, "H3").stage == "H3"


def test_skip_is_immutable_and_distinct_from_completion():
    skipped = skip_job(initial_state(_bindings()), "conditional_job", "not_selected")
    assert skipped.skipped_jobs == (("conditional_job", "not_selected"),)
    assert skipped.completed_jobs == ()
    with pytest.raises(HCStateError, match="immutable"):
        start_job(skipped, "conditional_job")
