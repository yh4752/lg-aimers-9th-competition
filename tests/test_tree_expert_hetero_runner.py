import threading

from experiments.tree_expert.hetero_contracts import load_hetero_contract, structure_jobs
from experiments.tree_expert.hetero_runner import run_pending_jobs
from experiments.tree_expert.hetero_state import initial_state


def test_scheduler_uses_at_most_two_workers_and_stops_before_guard(tmp_path):
    lock = threading.Lock()
    active = 0
    maximum = 0

    def execute(job, output, gpu):
        nonlocal active, maximum
        del job, output, gpu
        with lock:
            active += 1
            maximum = max(maximum, active)
            active -= 1
        return "completed"

    contract = load_hetero_contract()
    state = run_pending_jobs(
        initial_state(), structure_jobs(contract), root=tmp_path, executor=execute,
        gpu_ids=(0, 1), wall_deadline=10_000, clock=lambda: 0,
        new_job_guard_seconds=600,
    )
    assert len(state.completed_jobs) == 6
    assert maximum <= 2
    stopped = run_pending_jobs(
        initial_state(), structure_jobs(contract), root=tmp_path / "stopped", executor=execute,
        gpu_ids=(0, 1), wall_deadline=500, clock=lambda: 0,
        new_job_guard_seconds=600,
    )
    assert stopped.completed_jobs == ()
