from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

import pandas as pd

from experiments.tree_expert.contracts import load_e1_contract
from experiments.tree_expert.inputs import (
    PREDICTION_COLUMNS,
    VerifiedE1Input,
    VerifiedOfficialData,
    file_sha256,
)
from experiments.tree_expert.runner import code_identity_member_names, run_e1_campaign
from experiments.tree_expert.training import FoldResult


class RecordingRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_concurrent = 0
        self.gpus_used: list[int] = []

    def run_job(self, *, job, baseline, output_dir, gpu_id, **_kwargs) -> FoldResult:
        with self._lock:
            self.active += 1
            self.maximum_concurrent = max(self.maximum_concurrent, self.active)
            self.gpus_used.append(gpu_id)
        try:
            time.sleep(0.02)
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            predictions = baseline.copy(deep=True)
            predictions["probability"] = [0.4, 0.6]
            predictions.to_csv(output / "predictions.csv", index=False)
            (output / "model.cbm").write_bytes(b"model")
            (output / "job.json").write_text(
                json.dumps({"job_id": job.job_id}), encoding="utf-8"
            )
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            (output / "worker.log").write_text("complete\n", encoding="utf-8")
            (output / "worker_result.json").write_text(
                json.dumps({"job_id": job.job_id, "status": "completed"}),
                encoding="utf-8",
            )
            return FoldResult(
                job_id=job.job_id,
                candidate_id=job.candidate_id,
                status="completed",
                brier=0.16,
                model_path=output / "model.cbm",
                predictions_path=output / "predictions.csv",
                snapshot_path=None,
                failure=None,
            )
        finally:
            with self._lock:
                self.active -= 1


@dataclass
class FakeClock:
    now: float

    def __call__(self) -> float:
        return self.now


def test_code_identity_excludes_generated_kaggle_cell() -> None:
    members = code_identity_member_names()

    assert "experiments/tree_expert/runner.py" in members
    assert "experiments/tree_expert/kaggle.py" in members
    assert "experiments/tree_expert/KAGGLE_E1_CELL.py" not in members


def _context(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    train = pd.DataFrame(
        {
            "row_id": ["t0", "t1", "v0", "v1"],
            "season": [2023, 2023, 2024, 2024],
            "pitcher_id": [1, 2, 1, 2],
            "control_success": [1, 0, 0, 1],
        }
    )
    train.to_csv(data_root / "train.csv", index=False)
    pd.DataFrame({"season": [2023]}).to_csv(
        data_root / "trackman_history.csv", index=False
    )
    baseline = pd.DataFrame(
        {
            "row_id": ["v0", "v1"],
            "target": [0, 1],
            "probability": [0.45, 0.55],
            "game_type": ["R", "F"],
            "game_month": [3, 4],
            "pitcher_id_known": ["known", "known"],
            "batter_id_known": ["known", "known"],
        }
    ).loc[:, PREDICTION_COLUMNS]
    baseline_path = tmp_path / "baseline.csv"
    baseline.to_csv(baseline_path, index=False)
    verified_data = VerifiedOfficialData(
        root=data_root,
        train=data_root / "train.csv",
        history=data_root / "trackman_history.csv",
        train_sha256=file_sha256(data_root / "train.csv"),
        history_sha256=file_sha256(data_root / "trackman_history.csv"),
    )
    verified_input = VerifiedE1Input(
        archive_sha256="1" * 64,
        manifest_sha256="2" * 64,
        stage_c_delivery_sha256=load_e1_contract().stage_c_delivery_sha256,
        stage_c_review_sha256=load_e1_contract().stage_c_review_sha256,
        baseline_fold="2023->2024",
        baseline_predictions=baseline_path,
    )
    return verified_data, verified_input


def test_runner_assigns_at_most_one_job_per_gpu(tmp_path: Path, capsys) -> None:
    verified_data, verified_input = _context(tmp_path)
    runtime = RecordingRuntime()

    result = run_e1_campaign(
        verified_data=verified_data,
        verified_input=verified_input,
        output_dir=tmp_path / "run",
        resume_bundle=None,
        absolute_deadline=time.time() + 3600,
        gpu_ids=(0, 1),
        runtime=runtime,
    )

    assert runtime.maximum_concurrent == 2
    assert set(runtime.gpus_used) == {0, 1}
    assert len(result.completed) + len(result.skipped) == 4
    assert result.bundles.handoff.is_file()
    assert result.bundles.handoff_sha256 == file_sha256(result.bundles.handoff)
    assert (
        f"TREE_E1_HANDOFF_READY path={result.bundles.handoff} "
        f"sha256={result.bundles.handoff_sha256}"
    ) in capsys.readouterr().out


def test_runner_stops_starting_jobs_inside_guard(tmp_path: Path) -> None:
    verified_data, verified_input = _context(tmp_path)
    clock = FakeClock(1000.0)

    result = run_e1_campaign(
        verified_data=verified_data,
        verified_input=verified_input,
        output_dir=tmp_path / "guard",
        resume_bundle=None,
        absolute_deadline=1599.0,
        gpu_ids=(0, 1),
        runtime=RecordingRuntime(),
        clock=clock,
    )

    assert result.status == "budget_inconclusive"
    assert result.bundles.resume.is_file()
