from pathlib import Path

from experiments.direct_expert.artifacts import DirectExpertBindings
from experiments.direct_expert.stage_b import run_stage_b


class FakeRuntime:
    def __init__(self, *, accepted: bool):
        self.accepted = accepted
        self.events = []
        self.full_fit_jobs = []
        self.bindings = DirectExpertBindings(*[str(i) * 64 for i in range(1, 7)])

    def now(self): return 0.0
    def locked_experts(self, _handoff): return ("D0", "D1", "D2", "D7")
    def run_job(self, job, gpu, output):
        output.mkdir(parents=True)
        (output / "result.json").write_text("{}", encoding="utf-8")
        self.events.append(job.phase)
        return output / "result.json"
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
    assert result.delivery is None


def test_accepted_result_full_fits_at_most_nine_models(tmp_path: Path) -> None:
    runtime = FakeRuntime(accepted=True)
    result = run_stage_b(runtime, tmp_path / "stage_a.zip", tmp_path, absolute_deadline=20_000)
    assert result.delivery.is_file()
    assert len(runtime.full_fit_jobs) <= 9
