from pathlib import Path
import threading
import time

import pandas as pd

from experiments.direct_expert.artifacts import DirectExpertBindings
from experiments.direct_expert.stage_b import run_stage_b


class FakeRuntime:
    def __init__(self, *, accepted: bool):
        self.accepted = accepted
        self.events = []
        self.full_fit_jobs = []
        self.active_by_gpu = {0: 0, 1: 0}
        self.maximum_by_gpu = {0: 0, 1: 0}
        self.lock = threading.Lock()
        self.clock = 0.0
        self.bindings = DirectExpertBindings(*[str(i) * 64 for i in range(1, 7)])

    def now(self): return self.clock
    def locked_experts(self, _handoff): return ("D0", "D1", "D2", "D7")
    def run_job(self, job, gpu, output):
        with self.lock:
            self.active_by_gpu[gpu] += 1
            self.maximum_by_gpu[gpu] = max(self.maximum_by_gpu[gpu], self.active_by_gpu[gpu])
            self.events.append(job.phase)
        try:
            time.sleep(0.002)
            output.mkdir(parents=True)
            pd.DataFrame({"row_id": [job.job_id], "probability": [0.5]}).to_csv(output / "predictions.csv", index=False)
            (output / "metrics.json").write_text('{"brier":0.25,"best_iteration":1}', encoding="utf-8")
            return output / "metrics.json"
        finally:
            with self.lock:
                self.active_by_gpu[gpu] -= 1
    def restore_job(self, job, folder):
        self.events.append(f"restored:{job.job_id}")
        return folder / "metrics.json"
    def decide(self, _jobs):
        self.events.append("stacking")
        self.events.append("decision")
        return {"candidate_id": "champion", "status": "accepted_stable" if self.accepted else "rejected"}
    def full_fit(self, decision, output):
        self.events.append("full_fit")
        token = output / "token.json"; token.parent.mkdir(parents=True, exist_ok=True); token.write_text('{"status":"accepted"}', encoding="utf-8")
        model = output / "model.cbm"; model.write_text("model", encoding="utf-8")
        self.full_fit_jobs.extend(range(9))
        return {"accepted/token.json": token, "models/model.cbm": model}
    def audit(self, _payloads): self.events.append("audit"); return True


def test_stage_b_runs_locked_confirmation_before_extra_seeds(tmp_path: Path) -> None:
    runtime = FakeRuntime(accepted=False)
    result = run_stage_b(runtime, tmp_path / "stage_a.zip", tmp_path, absolute_deadline=20_000)
    assert runtime.events.index("confirmation") < runtime.events.index("extra_seeds")
    assert runtime.events.index("extra_seeds") < runtime.events.index("stacking")
    assert runtime.events.index("stacking") < runtime.events.index("decision")
    assert runtime.maximum_by_gpu == {0: 1, 1: 1}
    assert result.delivery is None


def test_accepted_result_full_fits_at_most_nine_models(tmp_path: Path) -> None:
    runtime = FakeRuntime(accepted=True)
    result = run_stage_b(runtime, tmp_path / "stage_a.zip", tmp_path, absolute_deadline=20_000)
    assert result.delivery.is_file()
    assert len(runtime.full_fit_jobs) <= 9


def test_stage_b_resume_reuses_completed_jobs(tmp_path: Path) -> None:
    first_runtime = FakeRuntime(accepted=False)
    first = run_stage_b(first_runtime, tmp_path / "stage_a.zip", tmp_path / "first", absolute_deadline=20_000)
    resumed_runtime = FakeRuntime(accepted=False)
    second = run_stage_b(
        resumed_runtime,
        tmp_path / "stage_a.zip",
        tmp_path / "second",
        absolute_deadline=20_000,
        previous_handoff=first.handoff,
    )
    assert second.status == "rejected"
    assert sum(str(event).startswith("restored:") for event in resumed_runtime.events) == 28
    assert "confirmation" not in resumed_runtime.events
    assert "extra_seeds" not in resumed_runtime.events


def test_stage_b_defers_full_fit_when_safe_time_is_absent(tmp_path: Path) -> None:
    runtime = FakeRuntime(accepted=True)
    result = run_stage_b(runtime, tmp_path / "stage_a.zip", tmp_path, absolute_deadline=7_000)
    assert result.status == "accepted_pending_full_fit"
    assert runtime.full_fit_jobs == []
    assert result.delivery is None
    assert result.handoff.is_file()


def test_full_fit_failure_still_publishes_review_and_handoff(tmp_path: Path) -> None:
    runtime = FakeRuntime(accepted=True)
    runtime.full_fit = lambda _decision, _output: (_ for _ in ()).throw(RuntimeError("synthetic oom"))
    result = run_stage_b(runtime, tmp_path / "stage_a.zip", tmp_path, absolute_deadline=20_000)
    assert result.status == "accepted_full_fit_failed"
    assert result.delivery is None
    assert result.review.is_file()
    assert result.handoff.is_file()
