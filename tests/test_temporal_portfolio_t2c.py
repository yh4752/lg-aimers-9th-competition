from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.features import FeatureBatch

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


def _training_rows() -> pd.DataFrame:
    rows = []
    for year in (2019, 2020, 2021, 2022, 2023, 2024):
        for index in range(4):
            rows.append(
                {
                    "row_id": f"{year}_{index}",
                    "season": year,
                    "game_type": "R",
                    "pitcher_id": 10 + index,
                    "batter_id": 20 + index,
                    "pitcher_hand": "R",
                    "batter_hand": "L" if index % 2 else "R",
                    "asof_pitcher_n": 10 + index,
                    "asof_batter_n": 10 + index,
                    "asof_pitcher_success_rate": 0.5,
                    "asof_batter_success_rate": 0.5,
                    "base_state": "empty",
                    "li": 1.0,
                    "numeric": float(index),
                    "control_success": index % 2,
                }
            )
    return pd.DataFrame(rows)


def test_t2c_schedule_is_exactly_two_new_seeds_by_three_folds() -> None:
    from experiments.temporal_portfolio.t2c import build_t2c_specs

    specs = build_t2c_specs()

    assert [(item.seed, item.valid_year) for item in specs] == [
        (42, 2022),
        (2026, 2022),
        (42, 2023),
        (2026, 2023),
        (42, 2024),
        (2026, 2024),
    ]


def test_t2c_materialization_uses_previous_season_and_cutoff_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.temporal_portfolio.t2c import T2CJobSpec, materialize_t2c_job

    observed: dict[str, object] = {}

    def materialize(cache_root, *, train, valid, feature_fit_rows, spec, **_kwargs):
        observed.update(
            train_seasons=tuple(sorted(train["season"].unique())),
            context_seasons=tuple(sorted(feature_fit_rows["season"].unique())),
            valid_seasons=tuple(sorted(valid["season"].unique())),
            features=spec.bundles,
        )

        def batch(frame: pd.DataFrame, target: bool) -> FeatureBatch:
            return FeatureBatch(
                frame["row_id"].to_numpy(),
                frame["season"].to_numpy(),
                frame["game_type"].to_numpy(),
                np.zeros((len(frame), 2), dtype="float32"),
                np.zeros((len(frame), 1), dtype="int64"),
                frame["control_success"].to_numpy(dtype="int8") if target else None,
            )

        return SimpleNamespace(
            root=Path(cache_root),
            train=batch(train, True),
            valid=batch(valid, True),
            identity_sha256="e" * 64,
            state=SimpleNamespace(fitted_sources={}),
        )

    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2c.materialize_fold_cache", materialize
    )
    job = materialize_t2c_job(
        T2CJobSpec("t2c__s1__va2022__s42", 42, 2022),
        data_rows_sha256="a" * 64,
        parent_sha256="b" * 64,
        train=_training_rows(),
        history=pd.DataFrame(),
        cache_root=tmp_path / "cache",
    )

    assert observed == {
        "train_seasons": (2021,),
        "context_seasons": (2019, 2020, 2021),
        "valid_seasons": (2022,),
        "features": ("base", "S1"),
    }
    assert tuple(job.training.identity.payload["train_seasons"]) == (2021,)
    assert tuple(job.training.identity.payload["features"]) == ("base", "S1")
    job.training.validate_seals()


def test_safety_gate_uses_anchor_only_for_exact_game_type_f() -> None:
    from experiments.temporal_portfolio.t2c import build_gated_s1_oof

    anchor = _oof(2024, (0.40, 0.60, 0.40, 0.60))
    anchor.loc[2, "game_type"] = "f"
    multi = anchor.copy(deep=True)
    multi["probability"] = (0.30, 0.70, 0.30, 0.70)
    recent = anchor.copy(deep=True)
    recent["probability"] = (0.10, 0.90, 0.10, 0.90)

    oof = build_gated_s1_oof(anchor, multi, (recent,), valid_year=2024)

    exact_f = oof["game_type"].eq("F")
    assert oof.loc[exact_f, "candidate"].equals(
        oof.loc[exact_f, "baseline"]
    )
    assert oof.loc[~exact_f, "candidate"].ne(
        oof.loc[~exact_f, "baseline"]
    ).any()


def test_recent_seed_ensemble_averages_logits_before_multi_blend() -> None:
    from experiments.temporal_portfolio.t2c import build_gated_s1_oof

    anchor = _oof(2024, (0.45, 0.55, 0.45, 0.55))
    anchor["game_type"] = ("R", "F", "R", "F")
    multi = anchor.copy(deep=True)
    multi["probability"] = (0.25, 0.75, 0.35, 0.65)
    recent_a = anchor.copy(deep=True)
    recent_a["probability"] = (0.10, 0.90, 0.20, 0.80)
    recent_b = anchor.copy(deep=True)
    recent_b["probability"] = (0.30, 0.70, 0.40, 0.60)

    actual = build_gated_s1_oof(
        anchor, multi, (recent_a, recent_b), valid_year=2024
    )

    left = np.asarray(recent_a["probability"], dtype="float64")
    right = np.asarray(recent_b["probability"], dtype="float64")
    recent_logit = (
        np.log(left / (1.0 - left)) + np.log(right / (1.0 - right))
    ) / 2.0
    recent_mean = 1.0 / (1.0 + np.exp(-recent_logit))
    fixed = np.asarray(multi["probability"], dtype="float64")
    expected_logit = (recent_logit + np.log(fixed / (1.0 - fixed))) / 2.0
    expected = 1.0 / (1.0 + np.exp(-expected_logit))
    non_f = ~actual["game_type"].eq("F").to_numpy()
    np.testing.assert_allclose(actual.loc[non_f, "candidate"], expected[non_f])


def test_gated_oof_evaluation_returns_paired_brier_evidence() -> None:
    from experiments.temporal_portfolio.t2c import (
        build_gated_s1_oof,
        evaluate_gated_s1_oof,
    )

    anchor = _oof(2024, (0.40, 0.60, 0.40, 0.60))
    multi = anchor.copy(deep=True)
    multi["probability"] = (0.30, 0.70, 0.30, 0.70)
    recent = anchor.copy(deep=True)
    recent["probability"] = (0.05, 0.95, 0.05, 0.95)
    oof = build_gated_s1_oof(anchor, multi, (recent,), valid_year=2024)

    evidence = evaluate_gated_s1_oof(oof, bootstrap_repeats=50)

    assert evidence["status"] == "completed"
    assert evidence["rows"] == 4
    assert evidence["gain"] > 0
    assert evidence["bootstrap_lower"] >= 0
