from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
from types import MappingProxyType
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


def _passing_seed_evidence() -> dict[int, dict[int, dict[str, object]]]:
    return {
        seed: {
            year: {
                "status": "completed",
                "rows": 100 + year - 2022,
                "gain": 0.00010,
                "max_segment_regression": 0.00010,
            }
            for year in YEARS
        }
        for seed in (42, 2026)
    }


def _passing_ensemble() -> dict[str, object]:
    folds = {
        year: {
            "status": "completed",
            "rows": 100 + year - 2022,
            "gain": 0.00010,
            "max_segment_regression": 0.00010,
        }
        for year in YEARS
    }
    return {
        "folds": folds,
        "combined": {
            "status": "completed",
            "rows": sum(item["rows"] for item in folds.values()),
            "gain": 0.00010,
            "bootstrap_lower": 0.00002,
            "max_segment_regression": 0.00010,
        },
    }


def test_t2c_promotes_only_complete_reproducible_candidate() -> None:
    from experiments.temporal_portfolio.t2c import decide_t2c

    decision = decide_t2c(
        seed_evidence=_passing_seed_evidence(), ensemble=_passing_ensemble()
    )

    assert decision.status == "promoted"
    assert decision.candidate_id == "s1_game_type_f_fallback_v1"


def test_t2c_rejects_when_one_new_seed_has_nonpositive_weighted_gain() -> None:
    from experiments.temporal_portfolio.t2c import decide_t2c

    evidence = _passing_seed_evidence()
    for fold in evidence[42].values():
        fold["gain"] = 0.0

    assert (
        decide_t2c(seed_evidence=evidence, ensemble=_passing_ensemble()).status
        == "rejected"
    )


def test_t2c_marks_incomplete_jobs_budget_inconclusive() -> None:
    from experiments.temporal_portfolio.t2c import decide_t2c

    evidence = _passing_seed_evidence()
    evidence[2026][2023] = {"status": "pending"}

    decision = decide_t2c(seed_evidence=evidence, ensemble={})

    assert decision.status == "budget_inconclusive"


def test_t2c_rejects_extra_or_contradictory_evidence() -> None:
    from experiments.temporal_portfolio.t2c import decide_t2c

    evidence = _passing_seed_evidence()
    evidence[3407] = evidence[42]
    assert decide_t2c(seed_evidence=evidence, ensemble=_passing_ensemble()).status == "rejected"

    evidence = _passing_seed_evidence()
    ensemble = _passing_ensemble()
    ensemble["combined"]["rows"] = 1
    assert decide_t2c(seed_evidence=evidence, ensemble=ensemble).status == "rejected"


def _verified_official(tmp_path: Path):
    from experiments.temporal_portfolio.inputs import VerifiedOfficialData

    data = tmp_path / "official"
    data.mkdir(exist_ok=True)
    paths, hashes, sizes = [], {}, {}
    for name in ("train.csv", "test.csv", "trackman_history.csv", "sample_submission.csv"):
        path = data / name
        path.write_text("fixture\n", encoding="utf-8")
        paths.append(path)
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        sizes[name] = path.stat().st_size
    return VerifiedOfficialData(
        data,
        *paths,
        MappingProxyType(hashes),
        10,
        2,
        MappingProxyType(sizes),
        MappingProxyType({}),
    )


def _runner_prepared(tmp_path: Path, verified):
    from experiments.temporal_portfolio.t1_runner import _input_identity
    from experiments.temporal_portfolio.t2c_input import VerifiedT2CInput

    path = tmp_path / "t2c_input.zip"
    path.write_bytes(b"fixture")
    return VerifiedT2CInput(
        path,
        "a" * 64,
        _input_identity(verified),
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "s1_game_type_f_fallback_v1",
    )


def _materialized_for(spec, prepared):
    from experiments.temporal_portfolio.identity import TrainingIdentity

    identity = TrainingIdentity.from_payload(
        {
            "data_rows": prepared.data_rows_sha256,
            "train_seasons": [spec.valid_year - 1],
            "valid_year": spec.valid_year,
            "decay": None,
            "features": ["base", "S1"],
            "model": {"job_id": spec.job_id},
            "loss": "bce",
            "seed": spec.seed,
        }
    )
    return SimpleNamespace(
        training=SimpleNamespace(job_id=spec.job_id, identity=identity),
        plan=SimpleNamespace(max_seconds=2_400),
    )


def _write_compact_job(root: Path, job_id: str, identity: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    members = {
        "predictions.csv": b"row_id,target,probability\na,1,0.8\n",
        "metrics.json": b"{}",
        "checkpoint_meta.json": b"{}",
    }
    records = {}
    for name, data in members.items():
        (root / name).write_bytes(data)
        records[name] = {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
    (root / "compact_result.json").write_bytes(
        _json_bytes(
            {
                "schema_version": 1,
                "job_id": job_id,
                "status": "completed",
                "training_identity_sha256": identity,
                "members": records,
            }
        )
    )


def _patch_runner_core(monkeypatch: pytest.MonkeyPatch, prepared) -> None:
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2c_runner.verify_t2c_input",
        lambda _path: prepared,
    )
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2c_runner.load_t2c_references",
        lambda _prepared: ({}, {}, {}, {}),
    )
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2c_runner.materialize_t2c_job",
        lambda spec, **_kwargs: _materialized_for(spec, prepared),
    )
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2c_runner.verify_worker_result",
        lambda path: json.loads((Path(path) / "worker_result.json").read_text()),
    )
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2c_runner._collect_evidence",
        lambda *_args, **_kwargs: {
            "seed_evidence": _passing_seed_evidence(),
            "ensemble": _passing_ensemble(),
            "decision": {"status": "promoted"},
        },
    )


def test_t2c_runner_starts_exactly_six_authorized_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.temporal_portfolio.t2c_runner import run_t2c_stage

    verified = _verified_official(tmp_path)
    prepared = _runner_prepared(tmp_path, verified)
    _patch_runner_core(monkeypatch, prepared)

    class Handle:
        returncode = 0

        def poll(self):
            return 0

        def terminate(self):
            raise AssertionError

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError

    class FakeLauncher:
        def __init__(self):
            self.jobs = []

        def start(self, job, output, *, gpu, deadline):
            self.jobs.append((job.training.job_id, gpu))
            output.mkdir(parents=True, exist_ok=True)
            (output / "worker_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "job_id": job.training.job_id,
                        "training_identity_sha256": job.training.identity.sha256,
                    }
                ),
                encoding="utf-8",
            )
            return Handle()

    launcher = FakeLauncher()
    result = run_t2c_stage(
        verified=verified,
        t2c_input=prepared.path,
        output_root=tmp_path / "output",
        deadline=20_000,
        launcher=launcher,
        frame_loader=lambda _path: pd.DataFrame(),
        clock=lambda: 1_000.0,
        sleeper=lambda _seconds: None,
    )

    assert len(launcher.jobs) == 6
    assert len({job_id for job_id, _gpu in launcher.jobs}) == 6
    assert result.status == "completed"


def test_t2c_artifacts_bind_parent_and_compact_identities(tmp_path: Path) -> None:
    from experiments.temporal_portfolio.t1_artifacts import _read_bundle
    from experiments.temporal_portfolio.t2c import build_t2c_specs
    from experiments.temporal_portfolio.t2c_artifacts import (
        restore_t2c_resume_source,
        write_t2c_bundles,
    )

    jobs = tmp_path / "jobs"
    identities = {}
    for spec in build_t2c_specs():
        identity = sha256(spec.job_id.encode()).hexdigest()
        identities[spec.job_id] = identity
        _write_compact_job(jobs / spec.job_id, spec.job_id, identity)
    completed = tuple(spec.job_id for spec in build_t2c_specs())
    bundles = write_t2c_bundles(
        tmp_path / "bundles",
        jobs_root=jobs,
        completed=completed,
        pending=(),
        failed=(),
        evidence={"decision": {"status": "promoted"}},
        parent_sha256="a" * 64,
    )

    manifest, _ = _read_bundle(
        bundles.review, expected_kind="temporal_t2c_review_v1"
    )
    assert manifest["parent_sha256"] == "a" * 64
    assert manifest["training_identities"] == identities
    restored = restore_t2c_resume_source(bundles.resume, tmp_path / "restored")
    assert (restored / "jobs" / completed[0] / "compact_result.json").is_file()


def test_t2c_runner_reuses_compact_completed_job_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.temporal_portfolio.t2c import build_t2c_specs
    from experiments.temporal_portfolio.t2c_artifacts import (
        restore_t2c_resume_source,
        write_t2c_bundles,
    )
    from experiments.temporal_portfolio.t2c_runner import run_t2c_stage

    verified = _verified_official(tmp_path)
    prepared = _runner_prepared(tmp_path, verified)
    _patch_runner_core(monkeypatch, prepared)
    source = tmp_path / "source_jobs"
    for spec in build_t2c_specs():
        job = _materialized_for(spec, prepared)
        _write_compact_job(source / spec.job_id, spec.job_id, job.training.identity.sha256)
    completed = tuple(spec.job_id for spec in build_t2c_specs())
    bundles = write_t2c_bundles(
        tmp_path / "bundles",
        jobs_root=source,
        completed=completed,
        pending=(),
        failed=(),
        evidence={"decision": {"status": "promoted"}},
        parent_sha256=prepared.decision_sha256,
    )
    output = restore_t2c_resume_source(bundles.resume, tmp_path / "output")

    class MustNotLaunch:
        def start(self, *_args, **_kwargs):
            raise AssertionError("completed T2-C job was relaunched")

    result = run_t2c_stage(
        verified=verified,
        t2c_input=prepared.path,
        output_root=output,
        deadline=20_000,
        launcher=MustNotLaunch(),
        frame_loader=lambda _path: pd.DataFrame(),
        clock=lambda: 1_000.0,
        sleeper=lambda _seconds: None,
    )

    assert result.completed == completed


def test_t2c_collected_evidence_is_canonical_json_serializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.temporal_portfolio.t2c import build_t2c_specs
    from experiments.temporal_portfolio.t2c_runner import _collect_evidence

    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2c_runner._prediction",
        lambda root: pd.read_csv(Path(root) / "predictions.csv"),
    )

    jobs = tmp_path / "jobs"
    anchors, multi, seed_3407 = {}, {}, {}
    for year in YEARS:
        anchors[year] = _oof(year, (0.40, 0.60, 0.40, 0.60))
        multi[year] = _oof(year, (0.30, 0.70, 0.30, 0.70))
        seed_3407[year] = _oof(year, (0.10, 0.90, 0.10, 0.90)).drop(
            columns="valid_year"
        )
    for spec in build_t2c_specs():
        root = jobs / spec.job_id
        root.mkdir(parents=True)
        _oof(spec.valid_year, (0.10, 0.90, 0.10, 0.90)).drop(
            columns="valid_year"
        ).to_csv(root / "predictions.csv", index=False)

    evidence = _collect_evidence(
        jobs,
        (anchors, multi, seed_3407, {}),
        completed=tuple(spec.job_id for spec in build_t2c_specs()),
        pending=(),
        failed=(),
    )

    encoded = json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert '"candidate_id":"s1_game_type_f_fallback_v1"' in encoded
    assert '"status":"rejected"' in encoded


def test_t2c_kaggle_cell_is_deterministic_small_and_single_handoff(
    tmp_path: Path,
) -> None:
    from experiments.temporal_portfolio.t2c_platform import build_t2c_kaggle_cell

    first = build_t2c_kaggle_cell(tmp_path / "first.py")
    second = build_t2c_kaggle_cell(tmp_path / "second.py")
    text = first.read_text(encoding="utf-8")

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 1_000_000
    assert "files.download" not in text
    assert "temporal_t2c_handoff.zip" in text
    assert "temporal_t2c_emergency_handoff.zip" in text
    compile(text, str(first), "exec")


def test_gpu_probe_accepts_t2c_log_prefix() -> None:
    from experiments.temporal_portfolio.t1_runner import require_two_t4_gpus

    assert require_two_t4_gpus(
        lambda: ("Tesla T4", "Tesla T4"), log_prefix="T2C"
    ) == ("Tesla T4", "Tesla T4")
