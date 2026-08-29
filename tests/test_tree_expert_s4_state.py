import pytest

from experiments.tree_expert.s4_state import (
    S4StateError,
    advance_s4_phase,
    initial_s4_state,
    load_s4_state,
    mark_completed,
    mark_failed,
    record_s4_decision,
    save_s4_state,
)


def test_state_round_trip_preserves_phase_jobs_and_decisions(tmp_path):
    state = mark_completed(initial_s4_state(), "anchor__a0")
    state = record_s4_decision(state, "full__c1", "research_only")
    path = save_s4_state(state, tmp_path / "state.json")
    assert load_s4_state(path) == state


def test_job_terminal_sets_are_disjoint_and_retry_replaces_failure():
    state = mark_failed(initial_s4_state(), "a")
    state = mark_completed(state, "a")
    assert state.completed_jobs == ("a",)
    assert state.failed_jobs == ()


def test_phase_transition_must_follow_registry():
    with pytest.raises(S4StateError):
        advance_s4_phase(initial_s4_state(), "confirmation")

