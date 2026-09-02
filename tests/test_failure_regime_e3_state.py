from __future__ import annotations

from pathlib import Path

import pytest

from experiments.failure_regime_e3.state import (
    E3StateError,
    complete_job,
    fail_job,
    initial_state,
    load_state,
    save_state,
    transition,
)


def test_failed_job_can_be_retried_and_completed(tmp_path: Path) -> None:
    state = initial_state()
    state = fail_job(state, "job-a", "temporary OOM")
    state = complete_job(state, "job-a")
    state = transition(state, "confirmation")
    path = save_state(tmp_path / "state.json", state)

    restored = load_state(path)

    assert restored.completed_jobs == ("job-a",)
    assert dict(restored.failed_jobs) == {}
    assert restored.phase == "confirmation"


def test_completed_job_is_immutable() -> None:
    state = complete_job(initial_state(), "job-a")

    with pytest.raises(E3StateError, match="completed job cannot fail"):
        fail_job(state, "job-a", "late failure")


def test_invalid_transition_is_rejected() -> None:
    with pytest.raises(E3StateError, match="phase transition differs"):
        transition(initial_state(), "full_fit")
