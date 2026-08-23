from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from experiments.temporal_portfolio.identity import TrainingIdentity
from experiments.temporal_portfolio.inputs import VerifiedOfficialData
from experiments.temporal_portfolio.t1_runner import (
    T1RunnerError,
    require_two_t4_gpus,
    run_t1_stage,
)
import pytest


def _verified(tmp_path: Path) -> VerifiedOfficialData:
    tmp_path.mkdir(parents=True)
    paths = []
    hashes = {}
    sizes = {}
    for name in (
        "train.csv",
        "test.csv",
        "trackman_history.csv",
        "sample_submission.csv",
    ):
        path = tmp_path / name
        path.write_text("fixture\n", encoding="utf-8")
        paths.append(path)
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        sizes[name] = path.stat().st_size
    return VerifiedOfficialData(
        tmp_path,
        *paths,
        MappingProxyType(hashes),
        24,
        4,
        MappingProxyType(sizes),
        MappingProxyType({}),
    )


def _materializer(spec, **_kwargs):
    identity = TrainingIdentity.from_payload(
        {
            "data_rows": "a" * 64,
            "train_seasons": [2023],
            "valid_year": 2024,
            "decay": None,
            "features": ["base"],
            "model": {"job_id": spec.job_id},
            "loss": "bce",
            "seed": 3407,
        }
    )
    training = SimpleNamespace(job_id=spec.job_id, identity=identity)
    return SimpleNamespace(training=training, plan=SimpleNamespace(max_seconds=1_440))


def _verify(path: str | Path):
    return json.loads((Path(path) / "worker_result.json").read_text(encoding="utf-8"))


class _FinishedHandle:
    returncode = 0

    def poll(self):
        return 0

    def terminate(self):
        raise AssertionError("completed fixture must not be terminated")

    def wait(self, timeout=None):
        del timeout
        return 0

    def kill(self):
        raise AssertionError("completed fixture must not be killed")


class _Launcher:
    def __init__(self) -> None:
        self.starts: list[tuple[str, int, float]] = []

    def start(self, materialized, output_dir, *, gpu, deadline):
        self.starts.append((materialized.training.job_id, gpu, deadline))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": materialized.training.job_id,
            "status": "completed",
            "training_identity_sha256": materialized.training.identity.sha256,
        }
        (output_dir / "worker_result.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return _FinishedHandle()


def test_t1_runner_fills_two_gpu_slots_and_completes_all_jobs(tmp_path: Path) -> None:
    launcher = _Launcher()
    result = run_t1_stage(
        verified=_verified(tmp_path / "data"),
        output_root=tmp_path / "output",
        deadline=30_000,
        launcher=launcher,
        materializer=_materializer,
        result_verifier=_verify,
        frame_loader=lambda _path: SimpleNamespace(),
        clock=lambda: 1_000.0,
        sleeper=lambda _seconds: None,
    )

    assert result.status == "completed"
    assert len(result.completed) == 15
    assert result.pending == ()
    assert result.failed == ()
    assert [gpu for _, gpu, _ in launcher.starts[:2]] == [0, 1]
    assert {gpu for _, gpu, _ in launcher.starts} == {0, 1}


def test_t1_runner_starts_nothing_inside_stop_and_handoff_window(tmp_path: Path) -> None:
    launcher = _Launcher()
    result = run_t1_stage(
        verified=_verified(tmp_path / "data"),
        output_root=tmp_path / "output",
        deadline=2_500,
        launcher=launcher,
        materializer=_materializer,
        result_verifier=_verify,
        frame_loader=lambda _path: SimpleNamespace(),
        clock=lambda: 1_000.0,
        sleeper=lambda _seconds: None,
    )

    assert result.status == "budget_inconclusive"
    assert len(result.pending) == 15
    assert launcher.starts == []


def test_t1_runner_reuses_exact_completed_workers_without_relaunch(tmp_path: Path) -> None:
    verified = _verified(tmp_path / "data")
    output = tmp_path / "output"
    first_launcher = _Launcher()
    first = run_t1_stage(
        verified=verified,
        output_root=output,
        deadline=30_000,
        launcher=first_launcher,
        materializer=_materializer,
        result_verifier=_verify,
        frame_loader=lambda _path: SimpleNamespace(),
        clock=lambda: 1_000.0,
        sleeper=lambda _seconds: None,
    )
    second_launcher = _Launcher()
    second = run_t1_stage(
        verified=verified,
        output_root=output,
        deadline=30_000,
        launcher=second_launcher,
        materializer=_materializer,
        result_verifier=_verify,
        frame_loader=lambda _path: SimpleNamespace(),
        clock=lambda: 1_000.0,
        sleeper=lambda _seconds: None,
    )

    assert first.completed == second.completed
    assert second_launcher.starts == []


def test_t1_runner_rejects_completed_worker_with_different_identity(tmp_path: Path) -> None:
    verified = _verified(tmp_path / "data")
    output = tmp_path / "output"
    launcher = _Launcher()
    run_t1_stage(
        verified=verified,
        output_root=output,
        deadline=30_000,
        launcher=launcher,
        materializer=_materializer,
        result_verifier=_verify,
        frame_loader=lambda _path: SimpleNamespace(),
        clock=lambda: 1_000.0,
        sleeper=lambda _seconds: None,
    )
    first = output / "jobs" / launcher.starts[0][0] / "worker_result.json"
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["training_identity_sha256"] = "f" * 64
    first.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(T1RunnerError, match="identity differs"):
        run_t1_stage(
            verified=verified,
            output_root=output,
            deadline=30_000,
            launcher=_Launcher(),
            materializer=_materializer,
            result_verifier=_verify,
            frame_loader=lambda _path: SimpleNamespace(),
            clock=lambda: 1_000.0,
            sleeper=lambda _seconds: None,
        )


def test_production_gpu_gate_requires_exactly_two_tesla_t4_devices() -> None:
    assert require_two_t4_gpus(lambda: ("Tesla T4", "Tesla T4")) == (
        "Tesla T4",
        "Tesla T4",
    )
    with pytest.raises(T1RunnerError, match="exactly two Tesla T4"):
        require_two_t4_gpus(lambda: ("Tesla T4",))
    with pytest.raises(T1RunnerError, match="exactly two Tesla T4"):
        require_two_t4_gpus(lambda: ("NVIDIA A100-SXM4-40GB", "Tesla T4"))
