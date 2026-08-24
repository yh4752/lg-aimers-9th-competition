from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from experiments.temporal_portfolio.t2c_input import (
    EXPECTED_T2C_MEMBERS,
    T2CInputError,
    prepare_t2c_input,
    verify_t2c_input,
)


YEARS = (2022, 2023, 2024)
T2B_COMPLETED = (
    "t2b__f__s1__va2022__s3407",
    "t2b__f__p3__va2022__s3407",
    "t2b__f__p2__va2022__s3407",
    "t2b__f__s1__va2023__s3407",
    "t2b__f__p3__va2023__s3407",
    "t2b__f__p2__va2023__s3407",
    "t2b__c__s1_p3__va2024__s3407",
    "t2b__c__s1_p2__va2024__s3407",
    "t2b__h__s1_p2__va2023__s3407",
    "t2b__h__s1_p2__va2022__s3407",
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _oof(year: int, probabilities: tuple[float, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"{year}_{index}" for index in range(4)],
            "valid_year": [year] * 4,
            "target": [0, 1, 0, 1],
            "probability": probabilities,
            "pitcher_id": [1, 2, 3, 4],
            "batter_id": [11, 12, 13, 14],
            "game_type": ["R", "F", "R", "F"],
            "hand_matchup": ["R_R"] * 4,
            "pitcher_id_known": ["known"] * 4,
            "batter_id_known": ["known"] * 4,
            "trackman_available": ["available"] * 4,
            "history_count_bucket": ["high"] * 4,
            "runner_state": ["empty"] * 4,
            "leverage_bucket": ["medium"] * 4,
        }
    )


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)
    return path


def _t2b_input(tmp_path: Path) -> Path:
    data_rows_sha256 = "d" * 64
    t1_decision = {
        "status": "champion",
        "decay": "0.55",
        "recent_weight": "0.50",
        "mode": "logit",
        "beta": "0",
        "data_rows_sha256": data_rows_sha256,
        "review_sha256": "1" * 64,
    }
    t1_raw = _json_bytes(t1_decision)
    t1_sha = sha256(t1_raw).hexdigest()
    t2a_decision = {
        "status": "completed",
        "data_rows_sha256": data_rows_sha256,
        "t1_decision_sha256": t1_sha,
        "handoff_sha256": "2" * 64,
        "review_sha256": "3" * 64,
        "promoted": ["S1", "P3", "P2"],
        "recent_evidence": {
            name: {"status": "completed"} for name in ("S1", "P3", "P2")
        },
    }
    t2a_raw = _json_bytes(t2a_decision)
    members = {
        "t1_decision.json": t1_raw,
        "t2a_decision.json": t2a_raw,
    }
    for year in YEARS:
        members[f"t1_anchor_{year}.csv"] = _oof(
            year, (0.40, 0.60, 0.40, 0.60)
        ).to_csv(index=False).encode()
        members[f"t1_multi_{year}.csv"] = _oof(
            year, (0.30, 0.70, 0.30, 0.70)
        ).to_csv(index=False).encode()
    for bundle in ("s1", "p3", "p2"):
        members[f"t2a_recent_{bundle}_2024.csv"] = _oof(
            2024, (0.20, 0.80, 0.20, 0.80)
        ).to_csv(index=False).encode()
    records = {
        name: {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
        for name, data in sorted(members.items())
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "temporal_t2b_input_v1",
        "data_rows_sha256": data_rows_sha256,
        "t1_review_sha256": "1" * 64,
        "t1_decision_sha256": t1_sha,
        "t2a_handoff_sha256": "2" * 64,
        "t2a_review_sha256": "3" * 64,
        "t2a_decision_sha256": sha256(t2a_raw).hexdigest(),
        "members": records,
    }
    members["manifest.json"] = _json_bytes(manifest)
    return _write_zip(tmp_path / "t2b_input.zip", members)


def _t2b_handoff(tmp_path: Path, t2b_input: Path) -> Path:
    from experiments.temporal_portfolio.t2b_input import verify_t2b_input

    verified = verify_t2b_input(t2b_input)
    review_members: dict[str, bytes] = {}
    for year in (2022, 2023):
        job_id = f"t2b__f__s1__va{year}__s3407"
        review_members[f"jobs/{job_id}/predictions.csv"] = _oof(
            year, (0.20, 0.80, 0.20, 0.80)
        ).drop(columns="valid_year").to_csv(index=False).encode()
    review_members["t2b_evidence.json"] = _json_bytes(
        {"evidence": {"decisions": [{"candidate": "S1", "status": "rejected"}]}}
    )
    review_manifest = {
        "schema_version": 1,
        "artifact_kind": "temporal_t2b_review_v1",
        "stage": "T2B",
        "parent_sha256": verified.t2a_decision_sha256,
        "completed": list(T2B_COMPLETED),
        "pending": [],
        "failed": [],
        "training_identities": {name: sha256(name.encode()).hexdigest() for name in T2B_COMPLETED},
        "members": {
            name: {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
            for name, data in sorted(review_members.items())
        },
    }
    review_members["manifest.json"] = _json_bytes(review_manifest)
    review = io.BytesIO()
    with ZipFile(review, "w", ZIP_DEFLATED) as archive:
        for name, data in sorted(review_members.items()):
            archive.writestr(name, data)
    evidence = {"decisions": [{"candidate": "S1", "status": "rejected"}]}
    stage = _json_bytes(
        {
            "schema_version": 1,
            "stage": "T2B",
            "status": "completed",
            "completed": list(T2B_COMPLETED),
            "pending": [],
            "failed": [],
            "data_rows_sha256": verified.data_rows_sha256,
            "parent_sha256": verified.t2a_decision_sha256,
            "evidence": evidence,
        }
    )
    outer_members = {
        "review.zip": review.getvalue(),
        "resume.zip": b"fixture",
        "run.log": b"T2B_HANDOFF_READY\n",
        "stage_summary.json": stage,
    }
    outer_manifest = {
        "artifact_kind": "temporal_t2b_handoff_v1",
        "members": {
            name: {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
            for name, data in sorted(outer_members.items())
        },
    }
    outer_members["handoff_manifest.json"] = _json_bytes(outer_manifest)
    return _write_zip(tmp_path / "t2b_handoff.zip", outer_members)


def _rewrite_member(path: Path, name: str, data: bytes) -> None:
    with ZipFile(path) as archive:
        members = {member: archive.read(member) for member in archive.namelist()}
    members[name] = data
    _write_zip(path, members)


def test_t2c_input_keeps_only_s1_references_and_t2b_lineage(tmp_path: Path) -> None:
    t2b_input = _t2b_input(tmp_path)
    t2b_handoff = _t2b_handoff(tmp_path, t2b_input)

    path = prepare_t2c_input(t2b_input, t2b_handoff, tmp_path / "t2c_input.zip")
    verified = verify_t2c_input(path)

    assert verified.candidate_id == "s1_game_type_f_fallback_v1"
    assert verified.t2b_handoff_sha256 == sha256(t2b_handoff.read_bytes()).hexdigest()
    with ZipFile(path) as archive:
        assert set(archive.namelist()) == EXPECTED_T2C_MEMBERS


def test_t2c_input_rejects_tampered_seed_3407_prediction(tmp_path: Path) -> None:
    t2b_input = _t2b_input(tmp_path)
    path = prepare_t2c_input(
        t2b_input,
        _t2b_handoff(tmp_path, t2b_input),
        tmp_path / "t2c_input.zip",
    )
    _rewrite_member(path, "s1_recent_s3407_2023.csv", b"tampered")

    with pytest.raises(T2CInputError, match="member differs"):
        verify_t2c_input(path)
