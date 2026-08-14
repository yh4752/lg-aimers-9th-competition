from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from experiments.tabm_campaign.artifacts import (
    StageEvidence,
    verify_resume_bundle,
    write_stage_bundles,
)
from experiments.tabm_campaign.readjudication import readjudicate_version_b
from experiments.tabm_campaign.runner import VERSION_B_REFERENCE_CANDIDATE_ID


ONE_CYCLE = "a__p2__piecewise_linear__bce__one_cycle__s42"
PERIODIC = "a__p2__periodic__bce__plateau__s42"


def _candidate(candidate_id: str, scheduler: str, embedding: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "family": "tabm",
        "capacity": "p2",
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.1,
        "num_embedding": embedding,
        "loss": "bce",
        "scheduler": scheduler,
        "learning_rate": 0.0006,
        "seed": 42,
    }


def _job(candidate_id: str, fold: str, brier: float) -> dict[str, object]:
    return {
        "candidate_id": f"b__{candidate_id[3:]}__{fold}",
        "status": "completed",
        "brier": brier,
        "best_epoch": 2,
        "completed_epochs": 13,
        "checkpoint": None,
        "predictions": None,
        "resource_evidence": {},
        "failure": None,
    }


def _source_review(tmp_path: Path) -> Path:
    survivors = [
        _candidate(VERSION_B_REFERENCE_CANDIDATE_ID, "plateau", "piecewise_linear"),
        _candidate(ONE_CYCLE, "one_cycle", "piecewise_linear"),
        _candidate(PERIODIC, "plateau", "periodic"),
    ]
    results = [
        _job(VERSION_B_REFERENCE_CANDIDATE_ID, "tr2023__va2024", 0.2480974092),
        _job(ONE_CYCLE, "tr2023__va2024", 0.2482091607),
        _job(PERIODIC, "tr2023__va2024", 0.2482533824),
        _job(VERSION_B_REFERENCE_CANDIDATE_ID, "tr2022__va2023", 0.2525345588),
        _job(ONE_CYCLE, "tr2022__va2023", 0.2498542210),
        _job(PERIODIC, "tr2022__va2023", 0.2526993769),
    ]
    state = {
        "version": "B",
        "stage_complete": True,
        "survivors": survivors,
        "champion": survivors[1],
        "selection_delta": -0.0008,
        "champion_fold_briers": {"2024": 0.2482091607, "2023": 0.2498542210},
        "results": results,
        "resume_artifacts": {},
    }
    state_bytes = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    result_bytes = json.dumps(results, sort_keys=True, separators=(",", ":")).encode()
    return write_stage_bundles(
        tmp_path / "source",
        StageEvidence(
            "B",
            "a" * 64,
            "b" * 64,
            {
                "logs/stage.log": b"version=B completed=6\n",
                "metrics/job_results.json": result_bytes,
                "stage_state.json": state_bytes,
            },
            {"stage_state.json": state_bytes},
        ),
    ).review


def test_version_b_review_is_readjudicated_without_retraining(tmp_path: Path) -> None:
    source = _source_review(tmp_path)

    corrected = readjudicate_version_b(source, tmp_path / "corrected")

    assert corrected.bundles.resume is not None
    verified = verify_resume_bundle(corrected.bundles.resume)
    assert verified.version == "B"
    assert corrected.champion_id == VERSION_B_REFERENCE_CANDIDATE_ID
    with ZipFile(corrected.bundles.review) as archive:
        state = json.loads(archive.read("stage_state.json"))
        metrics = json.loads(archive.read("metrics/job_results.json"))
    assert state["champion"]["candidate_id"] == VERSION_B_REFERENCE_CANDIDATE_ID
    assert state["selection_delta"] == 0.0
    assert state["champion_fold_briers"] == {
        "2024": 0.2480974092,
        "2023": 0.2525345588,
    }
    assert state["results"] == metrics
    assert state["adjudication"]["original_champion_candidate_id"] == ONE_CYCLE
    assert state["adjudication"]["source_review_manifest_sha256"]
