from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from experiments.preprocessing_campaign.budgeted_artifacts import (
    REQUIRED_REVIEW_NAMES,
    write_stage_bundles,
)


def _fixture_campaign(root: Path, *, completed_stage: int = 1) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    job_id = "s1__tabm_fixture"
    job_root = root / "jobs" / job_id
    job_root.mkdir(parents=True)
    metrics = {
        "job_id": job_id,
        "family": "tabm",
        "setting_id": "dl_standard",
        "train_end_year": 2023,
        "valid_year": 2024,
        "seed": 42,
        "sample_mode": "proxy",
        "train_rows": 10,
        "valid_rows": 2,
        "brier": 0.25,
        "best_epoch": 1,
        "completed_epochs": 2,
        "validation_curve": [[0, 0.26], [1, 0.25]],
        "elapsed_seconds": 3.5,
        "hardware": {"device_count": 1},
    }
    (job_root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame(
        {
            "row_id": ["a", "b"],
            "fold": ["valid_2024", "valid_2024"],
            "season": [2024, 2024],
            "game_type": ["R", "R"],
            "target": [0, 1],
            "probability": [0.4, 0.6],
            "family": ["tabm", "tabm"],
            "setting_id": ["dl_standard", "dl_standard"],
            "seed": [42, 42],
            "pitcher_oov": [0, 1],
            "batter_oov": [1, 0],
        }
    ).to_csv(job_root / "predictions.csv", index=False)
    (job_root / "best_checkpoint.pt").write_bytes(b"fixture checkpoint")
    pending = root / "workers" / "pending_job"
    pending.mkdir(parents=True)
    (pending / "checkpoint.pt").write_bytes(b"resumable fixture checkpoint")
    manifest = {
        "schema_version": 1,
        "jobs": {
            job_id: {
                "state": "completed",
                "artifacts": [
                    {
                        "path": f"jobs/{job_id}/metrics.json",
                        "sha256": sha256(
                            (job_root / "metrics.json").read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        },
    }
    (root / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "stage_state.json").write_text(
        json.dumps(
            {
                "completed_stage": completed_stage,
                "selected_model": {"family": "tabm"},
                "preprocessing_status": "inconclusive",
            }
        ),
        encoding="utf-8",
    )
    (root / "run.log").write_text("STAGE_START\nJOB_COMPLETE\n", encoding="utf-8")
    return root


def test_stage_writes_separate_resume_and_small_review_bundles(
    tmp_path: Path,
) -> None:
    root = _fixture_campaign(tmp_path / "campaign")

    result = write_stage_bundles(
        campaign_root=root,
        stage_id=1,
        campaign_id="budgeted_preprocessing_campaign_v1",
    )

    assert result.resume.name == "preprocessing_stage_01_resume_bundle.zip"
    assert result.review.name == "preprocessing_stage_01_review_bundle.zip"
    with ZipFile(result.review) as archive:
        assert REQUIRED_REVIEW_NAMES <= set(archive.namelist())
        assert not any(
            name.endswith(".pt") or name.endswith(".cbm")
            for name in archive.namelist()
        )
        manifest = json.loads(archive.read("artifact_manifest.json"))
        assert manifest["schema_version"] == 1
        assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    with ZipFile(result.resume) as archive:
        assert "campaign_manifest.json" in archive.namelist()
        assert "stage_state.json" in archive.namelist()
        assert "resume_metadata.json" in archive.namelist()
        assert "workers/pending_job/checkpoint.pt" in archive.namelist()
        assert not any(
            name.startswith("jobs/") and name.endswith(".pt")
            for name in archive.namelist()
        )


def test_review_contains_wide_aligned_predictions(tmp_path: Path) -> None:
    root = _fixture_campaign(tmp_path / "campaign", completed_stage=5)

    result = write_stage_bundles(
        campaign_root=root,
        stage_id=5,
        campaign_id="budgeted_preprocessing_campaign_v1",
    )

    assert result.final_review is not None
    assert result.final_review.name == "preprocessing_campaign_final_review_bundle.zip"
    extracted = tmp_path / "extracted"
    with ZipFile(result.final_review) as archive:
        archive.extract("validation_predictions.csv.gz", extracted)
        assert "decision_table.csv" in archive.namelist()
    frame = pd.read_csv(extracted / "validation_predictions.csv.gz")
    assert list(frame[["row_id", "fold", "target"]].itertuples(index=False, name=None)) == [
        ("a", "valid_2024", 0),
        ("b", "valid_2024", 1),
    ]
    probability_columns = [
        column for column in frame if column.startswith("probability__")
    ]
    assert probability_columns == ["probability__s1__tabm_fixture"]
