"""Minimal shared job protocol for the sealed Stage P runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


STAGE_P_RUNTIME_PYTHON_MEMBERS = (
    "experiments/independent_dl/feature_sources/trackman.py",
    "experiments/independent_dl/features.py",
    "experiments/independent_dl/models/common.py",
    "experiments/independent_dl/models/tabm.py",
    "experiments/independent_dl/preprocessing.py",
    "experiments/independent_dl/progress.py",
    "experiments/independent_dl/row_features.py",
    "experiments/independent_dl/training.py",
    "experiments/tabm_campaign/artifacts.py",
    "experiments/tabm_campaign/cache.py",
    "experiments/tabm_campaign/row_feature_colab.py",
    "experiments/tabm_campaign/row_feature_contracts.py",
    "experiments/tabm_campaign/row_feature_decisions.py",
    "experiments/tabm_campaign/row_feature_proxy.py",
    "experiments/tabm_campaign/row_feature_runtime.py",
    "experiments/tabm_campaign/sampling.py",
    "experiments/tabm_campaign/training.py",
    "experiments/tabm_campaign/worker.py",
)


@dataclass(frozen=True)
class CampaignJob:
    candidate_id: str
    capacity: str
    k: int
    width: int
    blocks: int
    dropout: float
    num_embedding: str
    loss: str
    scheduler: str
    learning_rate: float
    seed: int
    train_end_year: int
    valid_year: int
    sample_mode: str
    max_epochs: int
    min_epochs: int
    patience: int
    feature_bundle: str | None = None


@dataclass(frozen=True)
class CampaignJobResult:
    candidate_id: str
    status: str
    brier: float | None
    best_epoch: int | None
    completed_epochs: int
    checkpoint: Path | None
    predictions_path: Path | None
    resource_evidence: Mapping[str, object]
    failure: str | None


class CampaignRuntime(Protocol):
    def run_jobs(
        self,
        version: str,
        jobs: tuple[CampaignJob, ...],
        output_dir: Path,
        *,
        gpu_count: int,
        job_deadline: float,
    ) -> tuple[CampaignJobResult, ...]: ...
