from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import zipfile

import pandas as pd

from experiments.preprocessing_campaign.review import write_review_bundle


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_review_bundle_contains_small_aggregate_evidence(tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    job_dir = output / "jobs/job-1"
    job_dir.mkdir(parents=True)
    predictions = job_dir / "predictions.csv"
    pd.DataFrame(
        {
            "row_id": ["a", "b"],
            "fold": ["valid_2024", "valid_2024"],
            "season": [2024, 2024],
            "game_type": ["R", "F"],
            "target": [1, 0],
            "probability": [0.8, 0.3],
            "anchor_id": ["tabm_p3", "tabm_p3"],
            "preprocessing_id": ["dl_standard", "dl_standard"],
            "seed": [42, 42],
            "pitcher_oov": [0, 1],
            "batter_oov": [1, 1],
        }
    ).to_csv(predictions, index=False)
    metrics = job_dir / "metrics.json"
    metrics.write_text(json.dumps({"brier": 0.065}), encoding="utf-8")
    manifest = {
        "campaign_id": "preprocessing_campaign_v1",
        "jobs": {
            "job-1": {
                "state": "completed",
                "job": {"wave": "a", "family": "tabm"},
                "metrics_path": "jobs/job-1/metrics.json",
                "predictions_path": "jobs/job-1/predictions.csv",
                "metrics_sha256": _hash(metrics),
                "predictions_sha256": _hash(predictions),
                "elapsed_seconds": 120.0,
                "peak_ram_gb": 2.0,
                "peak_gpu_gb": 3.0,
            },
            "job-2": {
                "state": "failed",
                "job": {"wave": "a", "family": "tabm"},
                "failure_reason": "RuntimeError: fixture failure",
            },
        },
    }
    (output / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    config = tmp_path / "config.json"
    config.write_text('{"campaign_id":"preprocessing_campaign_v1"}', encoding="utf-8")
    archive = tmp_path / "review.zip"

    summary = write_review_bundle(output, config, archive)

    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["hash_valid_completed"] == 1
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert {
            "review_summary.json",
            "fold_metrics.csv",
            "segment_metrics.csv",
            "resource_usage.csv",
            "failed_jobs.json",
            "campaign_manifest.json",
            "preprocessing_ablation_v1.json",
        } <= names
        assert not any(name.endswith("predictions.csv") for name in names)


def test_review_bundle_rejects_tampered_completed_artifact(tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    output.mkdir()
    predictions = output / "predictions.csv"
    predictions.write_text("changed", encoding="utf-8")
    (output / "metrics.json").write_text("{}", encoding="utf-8")
    (output / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": "x",
                "jobs": {
                    "job": {
                        "state": "completed",
                        "job": {"wave": "a", "family": "tabm"},
                        "metrics_path": "metrics.json",
                        "predictions_path": "predictions.csv",
                        "metrics_sha256": "0" * 64,
                        "predictions_sha256": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="hash-invalid"):
        write_review_bundle(output, config, tmp_path / "review.zip")
