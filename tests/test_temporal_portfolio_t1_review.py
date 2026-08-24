from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from experiments.temporal_portfolio.contracts import build_stage_jobs, load_contract
from experiments.temporal_portfolio.t1_artifacts import write_t1_bundles
from experiments.temporal_portfolio.t1_review import (
    build_t2a_input,
    evaluate_t1_review,
    verify_t2a_input,
)
from experiments.temporal_portfolio.uncertainty import SEGMENT_COLUMNS


def _review(tmp_path: Path) -> Path:
    jobs = tmp_path / "jobs"
    payloads = {}
    target = [0, 1, 0, 1]
    recent = [0.40, 0.60, 0.40, 0.60]
    ordinary = [0.35, 0.65, 0.35, 0.65]
    best = [0.10, 0.90, 0.10, 0.90]
    for spec in build_stage_jobs(load_contract(), "T1"):
        root = jobs / spec.job_id
        root.mkdir(parents=True)
        probability = recent if spec.expert == "recent" else (
            best if str(spec.decay) == "0.55" else ordinary
        )
        frame = pd.DataFrame(
            {
                "row_id": [f"{spec.fold.valid_year}_{index}" for index in range(4)],
                "target": target,
                "probability": probability,
                "pitcher_id": [1, 2, 3, 4],
                "batter_id": [11, 12, 13, 14],
                "game_type": ["R"] * 4,
                "pitcher_id_known": ["known"] * 4,
                "batter_id_known": ["known"] * 4,
                "trackman_available": ["available"] * 4,
                "hand_matchup": ["R_R"] * 4,
                "history_count_bucket": ["high"] * 4,
                "runner_state": ["empty"] * 4,
                "leverage_bucket": ["medium"] * 4,
            }
        )
        if spec.expert == "multi":
            frame["pitcher_id_known"] = "oov"
            frame["batter_id_known"] = "oov"
        frame.to_csv(root / "predictions.csv", index=False)
        (root / "metrics.json").write_text(
            json.dumps(
                {
                    "best_epoch": 2,
                    "best_brier": float(((frame.probability - frame.target) ** 2).mean()),
                    "train_target_rate": 0.5,
                    "weighted_train_target_rate": 0.5,
                    "train_request_sha256": "c" * 64,
                }
            ),
            encoding="utf-8",
        )
        identity = (hex(abs(hash(spec.job_id)))[2:] * 64)[:64].replace("x", "a")
        identity = (identity + "a" * 64)[:64]
        (root / "checkpoint_meta.json").write_text(
            json.dumps(
                {
                    "training_identity_sha256": identity,
                    "train_request_sha256": "c" * 64,
                    "checkpoint_binding": {"data_rows_sha256": "d" * 64},
                }
            ),
            encoding="utf-8",
        )
        payloads[spec.job_id] = {
            "job_id": spec.job_id,
            "status": "completed",
            "training_identity_sha256": identity,
        }
    completed = tuple(spec.job_id for spec in build_stage_jobs(load_contract(), "T1"))
    return write_t1_bundles(
        tmp_path / "bundles",
        jobs_root=jobs,
        completed=completed,
        pending=(),
        failed=(),
        verifier=lambda path: payloads[Path(path).name],
    ).review


def test_t1_review_selects_verified_champion_and_builds_small_t2a_input(
    tmp_path: Path,
) -> None:
    review = _review(tmp_path)
    decision = evaluate_t1_review(review)

    assert str(decision.decay) == "0.55"
    assert str(decision.recent_weight) == "0.50"
    assert decision.mode == "logit"
    assert str(decision.beta) == "0"
    assert decision.status == "champion"
    assert decision.bootstrap_lower > 0

    prepared = build_t2a_input(review, decision, tmp_path / "t2a_input.zip")
    verified = verify_t2a_input(prepared)
    assert verified.decision_sha256 == decision.sha256
    assert verified.data_rows_sha256 == "d" * 64
    with ZipFile(prepared) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "decision.json",
            "t1_anchor_2024.csv",
            "t1_multi_2024.csv",
        }
        anchor = pd.read_csv(archive.open("t1_anchor_2024.csv"))
    assert set(SEGMENT_COLUMNS).issubset(anchor.columns)
    assert len(anchor) == 4
