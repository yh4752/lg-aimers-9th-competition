from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from experiments.independent_dl.training import TrainResult
from experiments.preprocessing_campaign.budgeted_contracts import BudgetedJob
from experiments.preprocessing_campaign.budgeted_runtime import BudgetedRuntime


def _job(*, sample_mode: str = "proxy") -> BudgetedJob:
    return BudgetedJob(
        job_id=f"runtime-{sample_mode}",
        stage_id=1,
        family="tabm",
        profile_id="p2",
        setting_id="dl_standard",
        preprocessing_profile="dl_standard",
        components=(),
        model=MappingProxyType(
            {
                "architecture": "tabm",
                "k": 4,
                "width": 8,
                "blocks": 2,
                "dropout": 0.1,
                "num_embedding": "linear_relu",
            }
        ),
        training=MappingProxyType(
            {
                "optimizer": "adamw",
                "scheduler": "plateau",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "effective_batch_size": 8,
                "micro_batch_size": 4,
                "amp": False,
                "patience": 5,
            }
        ),
        train_end_year=2023,
        valid_year=2024,
        seed=42,
        max_seconds=60,
        sample_mode=sample_mode,
    )


def _official_like(preprocessing_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in range(2019, 2025):
        for repeat in range(3):
            part = preprocessing_frame.copy()
            part["row_id"] = [f"{season}-{repeat}-{index}" for index in range(len(part))]
            part["season"] = season
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def test_proxy_runtime_uses_deterministic_train_sample_and_full_validation(
    tmp_path: Path,
    preprocessing_frame: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
) -> None:
    full = _official_like(preprocessing_frame)
    observed: dict[str, object] = {}

    def fake_fit(request, adapter, output_dir):
        del adapter
        observed["train_rows"] = len(request.train.row_id)
        checkpoint = Path(output_dir) / "best_checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"fixture")
        return TrainResult(
            candidate_id=request.candidate_id,
            best_epoch=11,
            best_brier=0.24,
            checkpoint=checkpoint,
            predictions=np.full(len(request.valid.row_id), 0.5),
            attempted_micro_batches=(4,),
            effective_batch_size=8,
            model_config=request.model_config,
            started_epoch=0,
            hardware=MappingProxyType({"device_count": 1}),
            completed_epochs=12,
            validation_curve=((0, 0.25), (11, 0.24)),
        )

    runtime = BudgetedRuntime(
        data_dir=tmp_path,
        cache_root=tmp_path / "cache",
        proxy_max_rows=20,
        frame_loader=lambda: (full, preprocessing_history),
        fit_function=fake_fit,
        adapter_factory=lambda family: object(),
    )

    result = runtime.run_job(_job(), tmp_path / "job")
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert observed["train_rows"] == 20
    assert metrics["train_rows"] == 20
    assert metrics["valid_rows"] == 15
    assert metrics["sample_mode"] == "proxy"
    assert len(metrics["sample_sha256"]) == 64
    assert metrics["completed_epochs"] == 12
    assert metrics["validation_curve"] == [[0, 0.25], [11, 0.24]]


def test_full_runtime_uses_all_training_rows(
    tmp_path: Path,
    preprocessing_frame: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
) -> None:
    full = _official_like(preprocessing_frame)

    def fake_fit(request, adapter, output_dir):
        del adapter
        checkpoint = Path(output_dir) / "best_checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"fixture")
        return TrainResult(
            request.candidate_id,
            9,
            0.245,
            checkpoint,
            np.full(len(request.valid.row_id), 0.5),
            (4,),
            8,
            request.model_config,
            0,
            MappingProxyType({}),
            10,
            ((9, 0.245),),
        )

    runtime = BudgetedRuntime(
        data_dir=tmp_path,
        cache_root=tmp_path / "cache",
        proxy_max_rows=20,
        frame_loader=lambda: (full, preprocessing_history),
        fit_function=fake_fit,
        adapter_factory=lambda family: object(),
    )

    result = runtime.run_job(_job(sample_mode="full"), tmp_path / "full-job")
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert metrics["train_rows"] == 75
    assert metrics["sample_mode"] == "full"
