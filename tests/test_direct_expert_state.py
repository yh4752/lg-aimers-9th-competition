from pathlib import Path

import pytest

from experiments.direct_expert.state import (
    DirectExpertStateError,
    complete_job,
    fail_job,
    initial_state,
    load_state,
    retry_job,
    save_state,
    transition,
)


def test_phase_order_and_retry_state_round_trip(tmp_path: Path) -> None:
    state = initial_state("stage_a")
    state = complete_job(state, "screen__D0__2021_2022__s3407")
    state = fail_job(state, "screen__D1__2021_2022__s3407", "oom")
    state = retry_job(state, "screen__D1__2021_2022__s3407")
    save_state(tmp_path / "state.json", state)
    assert load_state(tmp_path / "state.json") == state


def test_phase_cannot_skip_forward() -> None:
    with pytest.raises(DirectExpertStateError, match="phase transition differs"):
        transition(initial_state("stage_b"), "stacking")
