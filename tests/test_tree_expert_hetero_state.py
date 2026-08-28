import pytest

from experiments.tree_expert.hetero_state import (
    HeteroStateError,
    advance_phase,
    initial_state,
    mark_job,
)


def test_state_transitions_are_immutable_and_ordered():
    first = initial_state()
    second = mark_job(first, "job", "completed")
    assert first.completed_jobs == ()
    assert second.completed_jobs == ("job",)
    third = advance_phase(second, "confirmation")
    assert third.phase == "confirmation"
    assert advance_phase(third, "completed").phase == "completed"
    with pytest.raises(HeteroStateError):
        advance_phase(first, "completed")
