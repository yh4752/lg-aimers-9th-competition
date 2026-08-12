from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.campaign import (
    CandidateExecutionError,
    CandidateRunResult,
    OfficialCampaignRuntime,
    _new_manifest,
    run_campaign,
)
from experiments.independent_dl.contracts import CandidateSpec, CampaignSpec
from experiments.independent_dl.training import TrainResult


def _candidate(candidate_id: str, *, width: int) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=candidate_id,
        family="mlp_resnet",
        feature_view="raw_typed",
        seed=42,
        epochs=100,
        model={
            "architecture": "mlp",
            "width": width,
            "blocks": 4,
            "dropout": 0.1,
            "embedding_dim": 32,
        },
        training={
            "optimizer": "adamw",
            "scheduler": "cosine",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "effective_batch_size": 4096,
            "micro_batch_size": 512,
            "amp": True,
            "patience": 30,
        },
        train_end_year=2023,
        valid_year=2024,
    )


def _campaign(candidates: tuple[CandidateSpec, ...], *, expansion=()) -> CampaignSpec:
    return CampaignSpec(
        campaign_id="tiny",
        protocol="test",
        exploration_fold=(2023, 2024),
        oof_folds=(),
        feature_views=("raw_typed",),
        confirmation_seeds=(),
        blend_weights=(0.5,),
        boundary_expansion={
            "mlp_resnet": {"width": tuple(expansion)},
        },
        survivor_policy={
            "proximity_delta": 0.002,
            "max_error_correlation": 0.98,
            "top_k_per_family": 2,
        },
        candidates=candidates,
    )


class _FakeRuntime:
    def __init__(self, *, interrupt_on: str | None = None) -> None:
        self.interrupt_on = interrupt_on
        self.started: list[str] = []

    def run_candidate(
        self, candidate: CandidateSpec, output_dir: Path
    ) -> CandidateRunResult:
        self.started.append(candidate.candidate_id)
        if candidate.candidate_id == self.interrupt_on:
            raise RuntimeError("fake interruption")
        output_dir.mkdir(parents=True, exist_ok=True)
        metric = output_dir / "metrics.json"
        prediction = output_dir / "predictions.csv"
        brier = 0.24 if candidate.model["width"] == 512 else 0.25
        metric.write_text(json.dumps({"brier": brier}), encoding="utf-8")
        prediction.write_text("row_id,probability\na,0.5\n", encoding="utf-8")
        return CandidateRunResult(
            metrics_path=metric,
            predictions_path=prediction,
            best_brier=brier,
        )


def test_campaign_resumes_without_repeating_completed_candidate(tmp_path: Path) -> None:
    campaign = _campaign((_candidate("candidate_0001", width=512), _candidate("candidate_0002", width=256)))
    first_runtime = _FakeRuntime(interrupt_on="candidate_0002")

    with pytest.raises(RuntimeError, match="fake interruption"):
        run_campaign(campaign, tmp_path, first_runtime)

    second_runtime = _FakeRuntime()
    summary = run_campaign(campaign, tmp_path, second_runtime)

    assert "candidate_0001" not in second_runtime.started
    assert summary.completed == ("candidate_0001", "candidate_0002")


def test_existing_manifest_order_does_not_override_campaign_priority(
    tmp_path: Path,
) -> None:
    first = _candidate("priority_first", width=512)
    second = _candidate("priority_second", width=256)
    campaign = _campaign((first, second))
    manifest = _new_manifest(campaign)
    manifest["candidates"] = {
        "priority_second": manifest["candidates"]["priority_second"],
        "priority_first": manifest["candidates"]["priority_first"],
    }
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    runtime = _FakeRuntime(interrupt_on="priority_first")

    with pytest.raises(RuntimeError, match="fake interruption"):
        run_campaign(campaign, tmp_path, runtime)

    assert runtime.started == ["priority_first"]


def test_changed_completed_artifact_is_run_again(tmp_path: Path) -> None:
    campaign = _campaign((_candidate("candidate_0001", width=512),))
    run_campaign(campaign, tmp_path, _FakeRuntime())
    prediction = tmp_path / "candidates/candidate_0001/predictions.csv"
    prediction.write_text("changed", encoding="utf-8")
    runtime = _FakeRuntime()

    run_campaign(campaign, tmp_path, runtime)

    assert runtime.started == ["candidate_0001"]


def test_boundary_winner_creates_larger_candidate_not_smaller_model(
    tmp_path: Path,
) -> None:
    campaign = _campaign(
        (
            _candidate("small", width=256),
            _candidate("boundary", width=512),
        ),
        expansion=(768, 1024),
    )

    summary = run_campaign(campaign, tmp_path, _FakeRuntime())
    expanded = [item for item in summary.registered if item.parent_candidate_id]

    assert [item.model["width"] for item in expanded] == [768, 1024]
    assert all(item.model["width"] > item.parent_model["width"] for item in expanded)


def test_official_runtime_writes_fold_bound_prediction_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": ["tr-1", "tr-2", "va-1", "va-2"],
            "season": [2023, 2023, 2024, 2024],
            "game_type": ["R", "F", "R", "F"],
            "feature": [0.0, 1.0, 2.0, 3.0],
            "control_success": [0, 1, 0, 1],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"season": pd.Series(dtype="int64")}).to_csv(
        data_dir / "trackman_history.csv", index=False
    )

    def fake_fit(request, adapter, output_dir):
        checkpoint = Path(output_dir) / "best_checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return TrainResult(
            candidate_id=request.candidate_id,
            best_epoch=3,
            best_brier=0.25,
            checkpoint=checkpoint,
            predictions=np.array([0.4, 0.6], dtype="float64"),
            attempted_micro_batches=(512,),
            effective_batch_size=4096,
            model_config=request.model_config,
            started_epoch=0,
            hardware={
                "device_count": 1,
                "devices": ({"index": 0, "name": "fake", "vram_bytes": 1024},),
                "training_mode": "single_gpu",
                "training_device_indices": (0,),
            },
        )

    runtime = OfficialCampaignRuntime(
        data_dir,
        cache_root=tmp_path / "cache",
        fit_function=fake_fit,
        adapter_factory=lambda family: object(),
    )
    output_dir = tmp_path / "candidate"

    result = runtime.run_candidate(_candidate("candidate", width=512), output_dir)

    predictions = pd.read_csv(result.predictions_path)
    assert predictions.columns.tolist() == [
        "row_id",
        "fold",
        "season",
        "game_type",
        "target",
        "probability",
        "candidate_id",
    ]
    assert predictions["row_id"].tolist() == ["va-1", "va-2"]
    assert predictions["fold"].tolist() == ["valid_2024", "valid_2024"]
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["hardware"]["training_mode"] == "single_gpu"
    assert metrics["hardware"]["training_device_indices"] == [0]


def test_campaign_cli_exposes_run_status_and_summarize() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.independent_dl.run_campaign",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run" in completed.stdout
    assert "status" in completed.stdout
    assert "summarize" in completed.stdout


def test_official_runtime_preserves_inner_traceback(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "row_id": ["tr", "va"],
            "season": [2023, 2024],
            "game_type": ["R", "R"],
            "feature": [0.0, 1.0],
            "control_success": [0, 1],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"season": pd.Series(dtype="int64")}).to_csv(
        data_dir / "trackman_history.csv", index=False
    )

    def fail_fit(request, adapter, output_dir):
        raise ValueError("inner training failure")

    runtime = OfficialCampaignRuntime(
        data_dir,
        cache_root=tmp_path / "cache",
        fit_function=fail_fit,
        adapter_factory=lambda family: object(),
    )

    with pytest.raises(CandidateExecutionError) as captured:
        runtime.run_candidate(_candidate("candidate", width=512), tmp_path / "out")

    assert "Traceback (most recent call last)" in str(captured.value)
    assert "ValueError: inner training failure" in str(captured.value)
