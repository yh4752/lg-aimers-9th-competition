from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
from collections import OrderedDict
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from experiments.hierarchical_tabm.artifacts import (
    CampaignEvidence,
    restore_resume,
    verify_resume_bundle,
)
from experiments.hierarchical_tabm.colab import (
    ActiveCheckpoint,
    SnapshotCadence,
    SubprocessCampaignRuntime,
    promote_active_checkpoint,
    run_supervised_campaign,
)
from experiments.hierarchical_tabm.contracts import HierarchicalJob
from experiments.hierarchical_tabm.contracts import DEFAULT_CONTRACT
from experiments.hierarchical_tabm.inputs import EXPECTED_BINDING_KEYS
from experiments.hierarchical_tabm.metrics import CandidateDecision
from experiments.hierarchical_tabm.training import TrainingJobResult


def _bindings() -> MappingProxyType:
    return MappingProxyType(
        {key: sha256(key.encode()).hexdigest() for key in EXPECTED_BINDING_KEYS}
    )


def _evidence(tmp_path: Path, job_id: str) -> CampaignEvidence:
    root = tmp_path / "campaign"
    root.mkdir()
    log = root / "campaign.log"; log.write_text("CAMPAIGN_START\n")
    k_path = root / "k_selection.json"
    k_path.write_text(json.dumps({
        "fold": "2021->2022", "selected_k": 128.0,
        "scores": {"32.0": 0.2}, "row_count": 1,
    }))
    state = root / "stage_state.json"
    state.write_text(json.dumps({
        "schema_version": 1, "status": "running", "bindings": dict(_bindings()),
        "selected_k": 128.0, "completed_job_ids": [], "active_job_id": job_id,
        "decisions": {}, "delivery_candidate_ids": [], "final_epochs": None,
        "final_fit_completed": False,
    }))
    return CampaignEvidence(
        _bindings(), DEFAULT_CONTRACT, log, state, k_path,
        MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), None,
    )


def _active_checkpoint(tmp_path: Path, job_id: str, epoch: int = 3) -> ActiveCheckpoint:
    torch = __import__("torch")
    root = tmp_path / "live"; root.mkdir()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    (parameter.square().sum()).backward(); optimizer.step()
    checkpoint = root / "checkpoint.pt"
    torch.save(
        {
            "candidate_id": job_id,
            "epoch": epoch,
            "best_epoch": epoch,
            "best_brier": 0.24,
            "validation_curve": [(epoch, 0.24)],
            "validation_time_curve": [(epoch, 1.0, 0.24)],
            "elapsed_seconds": 1.0,
            "model": {"model.output.weight": torch.ones((1, 1))},
            "optimizer": optimizer.state_dict(),
            "scheduler": {"last_epoch": epoch + 1},
            "scaler": {},
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": [],
            "adapter_state": None,
        },
        checkpoint,
    )
    metadata = root / "checkpoint_meta.json"
    metadata.write_text(json.dumps({
        "candidate_id": job_id,
        "epoch": epoch,
        "checkpoint": "checkpoint.pt",
        "adapter_state": None,
        "checkpoint_binding": dict(_bindings()),
    }))
    return ActiveCheckpoint(job_id, root, checkpoint, metadata, epoch + 1)


def test_snapshot_cadence_starts_at_session_start_not_first_checkpoint() -> None:
    cadence = SnapshotCadence(300.0, 1200.0, started_at=100.0)
    assert cadence.snapshot_due(399.9) is False
    assert cadence.snapshot_due(400.0) is True
    assert cadence.download_due(1299.9) is False
    assert cadence.download_due(1300.0) is True
    cadence.mark_snapshot(400.0)
    cadence.mark_download(1300.0)
    assert cadence.snapshot_due(699.9) is False
    assert cadence.snapshot_due(700.0) is True


def test_active_checkpoint_download_is_immediate_then_interval_limited() -> None:
    cadence = SnapshotCadence(300.0, 1200.0, started_at=0.0)
    snapshots = []
    requested = []
    for epoch, now in enumerate(range(60, 24 * 60 + 1, 60), start=1):
        if cadence.active_snapshot_due(float(now), new_job=epoch == 1):
            snapshots.append(epoch)
            cadence.mark_snapshot(float(now))
            if cadence.active_download_due(float(now)):
                requested.append(epoch)
                cadence.mark_active_download(float(now))

    assert snapshots == [1, 6, 11, 16, 21]
    assert requested == [1, 21]


def test_active_checkpoint_becomes_verified_incomplete_resume(tmp_path: Path) -> None:
    job_id = "h1__tr2022__va2023__s3407"
    active = _active_checkpoint(tmp_path, job_id)
    evidence = _evidence(tmp_path, job_id)
    resume = promote_active_checkpoint(
        active,
        evidence,
        tmp_path / "snapshots",
        check_deadline=lambda: None,
    )
    verify_resume_bundle(resume, expected_bindings=evidence.bindings)
    restored = restore_resume(
        resume, tmp_path / "restored", expected_bindings=evidence.bindings
    )
    assert restored.active_job_directory is not None
    assert (restored.active_job_directory / "checkpoint.pt").read_bytes() == active.checkpoint.read_bytes()
    metadata = json.loads(
        (restored.active_job_directory / "checkpoint.pt.meta.json").read_text()
    )
    assert metadata["completed_epochs"] == 4


def test_active_checkpoint_binding_mismatch_is_rejected(tmp_path: Path) -> None:
    job_id = "h1__tr2022__va2023__s3407"
    active = _active_checkpoint(tmp_path, job_id)
    metadata = json.loads(active.checkpoint_meta.read_text())
    metadata["checkpoint_binding"]["code_sha256"] = "0" * 64
    active.checkpoint_meta.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="binding"):
        promote_active_checkpoint(
            active, _evidence(tmp_path, job_id), tmp_path / "snapshots",
            check_deadline=lambda: None,
        )


def test_active_checkpoint_invalid_optimizer_is_rejected(tmp_path: Path) -> None:
    torch = __import__("torch")
    job_id = "h1__tr2022__va2023__s3407"
    active = _active_checkpoint(tmp_path, job_id)
    payload = torch.load(active.checkpoint, map_location="cpu", weights_only=False)
    payload["optimizer"] = {"state": {}, "param_groups": []}
    torch.save(payload, active.checkpoint)
    with pytest.raises(ValueError, match="optimizer"):
        promote_active_checkpoint(
            active, _evidence(tmp_path, job_id), tmp_path / "snapshots",
            check_deadline=lambda: None,
        )


def test_active_checkpoint_accepts_torch_ordered_model_state(tmp_path: Path) -> None:
    torch = __import__("torch")
    job_id = "h1__tr2022__va2023__s3407"
    active = _active_checkpoint(tmp_path, job_id)
    payload = torch.load(active.checkpoint, map_location="cpu", weights_only=False)
    payload["model"] = OrderedDict(payload["model"])
    torch.save(payload, active.checkpoint)

    resume = promote_active_checkpoint(
        active, _evidence(tmp_path, job_id), tmp_path / "snapshots",
        check_deadline=lambda: None,
    )
    assert resume.is_file()


def test_subprocess_runtime_uses_argv_no_shell_and_explicit_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class Process:
        stdout = iter(())

        def wait(self, timeout=None):
            return 2

        def poll(self):
            return 2

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr("experiments.hierarchical_tabm.colab.subprocess.Popen", popen)
    runtime = SubprocessCampaignRuntime(
        data_dir=tmp_path / "data", bindings=_bindings(),
        log_path=tmp_path / "worker.log", python_executable="python3.11",
    )
    job = HierarchicalJob("h1__tr2022__va2023__s3407", "oof", 2022, 2023)
    with pytest.raises(ValueError, match="without result"):
        runtime.run_oof(
            job, output_dir=tmp_path / "job", selected_k=128.0,
            absolute_deadline=10_000.0, resume_checkpoint=None,
        )
    command, kwargs = calls[0]
    assert command[:3] == ["python3.11", "-m", "experiments.hierarchical_tabm.training"]
    assert kwargs["shell"] is False
    assert str(Path(__file__).resolve().parents[1]) in kwargs["env"]["PYTHONPATH"]


class _QuickRuntime:
    def __init__(self) -> None:
        self.bindings = _bindings()

    def select_k(self, fit_rows, valid_rows):
        return type("Selection", (), {
            "selected_k": 128.0, "fold": "2021->2022",
            "scores": {128.0: 0.2}, "row_count": len(valid_rows),
        })()

    def run_oof(self, job, **kwargs):
        output = kwargs["output_dir"]; output.mkdir(parents=True)
        predictions = output / "predictions.csv"
        pd.DataFrame({
            "row_id": [f"{job.valid_year}-r"], "target": [0],
            "probability": [.4], "season": [job.valid_year], "game_month": [4],
            "game_type": ["R"], "count_state": ["0_0"],
            "hand_matchup": ["R_L"], "base_out_state": ["000_0"],
            "pitcher_known": ["known"], "batter_known": ["known"],
        }).to_csv(predictions, index=False)
        checkpoint = output / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint")
        feature = output / "feature_state.json"; feature.write_text("{}")
        return TrainingJobResult(
            job.job_id, "oof", "completed", 10, 1, 0, .16, 1,
            predictions, checkpoint, None, feature, None,
        )

    def calibrate(self, kind, **kwargs):
        kwargs["output_path"].write_text("{}")
        return type("Calibration", (), {
            "kind": kind, "selected_regularization": .01, "state": None,
        })()

    def decide(self, candidate_id, **kwargs):
        return CandidateDecision(
            candidate_id, "rejected", False, None,
            {"2022->2023": .2, "2023->2024": .2},
            {"2022->2023": 0., "2023->2024": 0.}, 0., 0., "fixture",
        )

    def active_checkpoint(self):
        return None


def _verified(tmp_path: Path):
    data = tmp_path / "data"; data.mkdir()
    pd.DataFrame({
        "row_id": [f"{year}-r" for year in range(2019, 2025)],
        "season": list(range(2019, 2025)),
        "control_success": [0, 1, 0, 1, 0, 1],
    }).to_csv(data / "train.csv", index=False)
    return type("Verified", (), {
        "training": type("Training", (), {"data_dir": data})(),
        "anchor_predictions": {},
    })()


def test_supervisor_callbacks_receive_only_recursively_verified_resumes(tmp_path: Path) -> None:
    runtime = _QuickRuntime()
    downloads: list[Path] = []
    result = run_supervised_campaign(
        _verified(tmp_path),
        output_dir=tmp_path / "run",
        snapshot_dir=tmp_path / "snapshots",
        resume_bundle=None,
        wall_deadline=__import__("time").time() + 3600,
        on_verified_resume=downloads.append,
        log_path=tmp_path / "hierarchical_tabm.log",
        runtime=runtime,
    )
    assert result.state.status == "completed_no_candidate"
    assert len(downloads) == 2
    for path in downloads:
        verify_resume_bundle(path, expected_bindings=runtime.bindings)
