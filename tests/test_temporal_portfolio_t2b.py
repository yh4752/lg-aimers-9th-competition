from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from experiments.temporal_portfolio.contracts import build_stage_jobs, load_contract
from experiments.temporal_portfolio.t1_artifacts import write_t1_bundles
from experiments.temporal_portfolio.t2a_artifacts import write_t2a_bundles
from experiments.temporal_portfolio.t2b_input import (
    T2BInputError,
    prepare_t2b_input,
    verify_t2b_input,
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _t1_review(tmp_path: Path) -> Path:
    jobs = tmp_path / "t1_jobs"
    payloads = {}
    for spec in build_stage_jobs(load_contract(), "T1"):
        root = jobs / spec.job_id
        root.mkdir(parents=True)
        target = [0, 1, 0, 1]
        if spec.expert == "recent":
            probability = [0.40, 0.60, 0.40, 0.60]
        elif str(spec.decay) == "0.55":
            probability = [0.10, 0.90, 0.10, 0.90]
        else:
            probability = [0.35, 0.65, 0.35, 0.65]
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
        frame.to_csv(root / "predictions.csv", index=False)
        (root / "metrics.json").write_text(
            json.dumps(
                {
                    "weighted_train_target_rate": 0.5,
                    "train_request_sha256": "c" * 64,
                }
            ),
            encoding="utf-8",
        )
        identity = sha256(spec.job_id.encode()).hexdigest()
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
    return write_t1_bundles(
        tmp_path / "t1_bundles",
        jobs_root=jobs,
        completed=tuple(spec.job_id for spec in build_stage_jobs(load_contract(), "T1")),
        pending=(),
        failed=(),
        verifier=lambda path: payloads[Path(path).name],
    ).review


def _t2a_handoff(tmp_path: Path, t1_decision_sha256: str) -> Path:
    completed = (
        "t2a__r__s1__va2024__s3407",
        "t2a__r__p0__va2024__s3407",
        "t2a__r__p1__va2024__s3407",
        "t2a__r__p2__va2024__s3407",
        "t2a__r__p3__va2024__s3407",
        "t2a__m__p3__va2024__s3407",
        "t2a__m__s1__va2024__s3407",
        "t2a__m__p2__va2024__s3407",
    )
    jobs = tmp_path / "t2a_jobs"
    for job_id in completed:
        root = jobs / job_id
        root.mkdir(parents=True)
        records = {}
        for name, data in {
            "predictions.csv": b"row_id,target,probability\na,1,0.8\n",
            "metrics.json": b"{}",
            "checkpoint_meta.json": b"{}",
        }.items():
            (root / name).write_bytes(data)
            records[name] = {
                "size_bytes": len(data),
                "sha256": sha256(data).hexdigest(),
            }
        (root / "compact_result.json").write_bytes(
            _json_bytes(
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "status": "completed",
                    "training_identity_sha256": sha256(job_id.encode()).hexdigest(),
                    "members": records,
                }
            )
        )
    recent_evidence = {
        "S1": {
            "status": "completed",
            "gain": 0.00016,
            "bootstrap_lower": 0.00012,
            "bootstrap_median": 0.00016,
            "bootstrap_upper": 0.00020,
            "max_segment_regression": 0.0,
            "mapping_status": "not_applicable",
            "mapping_coverage": "not_applicable",
        },
        "P3": {
            "status": "completed",
            "gain": 0.00017,
            "bootstrap_lower": 0.00004,
            "bootstrap_median": 0.00017,
            "bootstrap_upper": 0.00029,
            "max_segment_regression": 0.00020,
            "mapping_status": "not_applicable",
            "mapping_coverage": "not_applicable",
        },
        "P2": {
            "status": "completed",
            "gain": 0.00012,
            "bootstrap_lower": 0.00003,
            "bootstrap_median": 0.00013,
            "bootstrap_upper": 0.00022,
            "max_segment_regression": 0.00020,
            "mapping_status": "not_applicable",
            "mapping_coverage": "not_applicable",
        },
    }
    evidence = {
        "phase_r": recent_evidence,
        "phase_m_selected": ["P3", "S1", "P2"],
        "phase_m": {},
        "promoted": [
            {
                "bundle": "S1",
                "variant": "both_experts",
                "gain": 0.00054,
                "mapping_gate": "eligible",
            },
            {
                "bundle": "P3",
                "variant": "recent_only",
                "gain": 0.00017,
                "mapping_gate": "eligible",
            },
            {
                "bundle": "P2",
                "variant": "recent_only",
                "gain": 0.00012,
                "mapping_gate": "eligible",
            },
        ],
    }
    skipped = {
        "t2a__r__b1__va2024__s3407": "insufficient_mapping",
        "t2a__r__m1__va2024__s3407": "insufficient_mapping",
    }
    bundles = write_t2a_bundles(
        tmp_path / "t2a_bundles",
        jobs_root=jobs,
        completed=completed,
        pending=(),
        failed=(),
        skipped=skipped,
        evidence=evidence,
        t1_decision_sha256=t1_decision_sha256,
    )
    stage = {
        "schema_version": 1,
        "stage": "T2A",
        "status": "completed",
        "completed": list(completed),
        "pending": [],
        "failed": [],
        "skipped": skipped,
        "evidence": evidence,
        "data_rows_sha256": "d" * 64,
        "t1_decision_sha256": t1_decision_sha256,
    }
    members = {
        "review.zip": bundles.review.read_bytes(),
        "resume.zip": bundles.resume.read_bytes(),
        "run.log": b"T2A_STAGE_RESULT status=completed\n",
        "stage_summary.json": _json_bytes(stage),
    }
    manifest = {
        "artifact_kind": "temporal_t2a_handoff_v1",
        "members": {
            name: {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
            for name, data in sorted(members.items())
        },
    }
    output = tmp_path / "temporal_t2a_handoff.zip"
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("handoff_manifest.json", _json_bytes(manifest))
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    return output


def _rewrite_member(path: Path, name: str, data: bytes) -> None:
    with ZipFile(path) as archive:
        members = {item: archive.read(item) for item in archive.namelist()}
    members[name] = data
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for member, payload in sorted(members.items()):
            archive.writestr(member, payload)


def test_t2b_input_contains_three_fixed_t1_folds_and_t2a_lineage(
    tmp_path: Path,
) -> None:
    review = _t1_review(tmp_path)
    from experiments.temporal_portfolio.t1_review import evaluate_t1_review

    decision = evaluate_t1_review(review)
    handoff = _t2a_handoff(tmp_path, decision.sha256)

    prepared = prepare_t2b_input(review, handoff, tmp_path / "t2b_input.zip")
    verified = verify_t2b_input(prepared)

    assert verified.promoted == ("S1", "P3", "P2")
    assert verified.t1_decision_sha256 == decision.sha256
    with ZipFile(prepared) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "t1_decision.json",
            "t2a_decision.json",
            "t1_anchor_2022.csv",
            "t1_anchor_2023.csv",
            "t1_anchor_2024.csv",
            "t1_multi_2022.csv",
            "t1_multi_2023.csv",
            "t1_multi_2024.csv",
        }


def test_t2b_input_rejects_tampered_member(tmp_path: Path) -> None:
    review = _t1_review(tmp_path)
    from experiments.temporal_portfolio.t1_review import evaluate_t1_review

    decision = evaluate_t1_review(review)
    prepared = prepare_t2b_input(
        review,
        _t2a_handoff(tmp_path, decision.sha256),
        tmp_path / "t2b_input.zip",
    )
    _rewrite_member(prepared, "t1_anchor_2022.csv", b"tampered")

    with pytest.raises(T2BInputError, match="member differs"):
        verify_t2b_input(prepared)
