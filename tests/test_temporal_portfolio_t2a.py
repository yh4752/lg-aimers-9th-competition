from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
import pandas as pd
import numpy as np
from types import SimpleNamespace

from experiments.independent_dl.features import FeatureBatch

from experiments.temporal_portfolio.t2a import (
    T2AJobSpec,
    build_phase_r_specs,
    evaluate_feature_candidate,
    materialize_t2a_job,
    select_phase_m_bundles,
)
from experiments.temporal_portfolio.t2a_platform import build_t2a_kaggle_cell
from experiments.temporal_portfolio.t1_artifacts import _read_bundle, verify_compact_result
from experiments.temporal_portfolio.t2a_artifacts import restore_t2a_resume_source, write_t2a_bundles
from experiments.temporal_portfolio.identity import TrainingIdentity
from experiments.temporal_portfolio.inputs import VerifiedOfficialData
from experiments.temporal_portfolio.t1_review import VerifiedT2AInput
from experiments.temporal_portfolio.t1_runner import _input_identity
from experiments.temporal_portfolio.t2a_runner import _prediction, run_t2a_stage


def _train() -> pd.DataFrame:
    rows = []
    for year in (2019, 2020, 2021, 2022, 2023, 2024):
        for index in range(4):
            rows.append(
                {
                    "row_id": f"{year}_{index}",
                    "season": year,
                    "game_type": "regular",
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


def test_t2a_phase_r_is_exactly_the_seven_single_bundle_jobs() -> None:
    specs = build_phase_r_specs()

    assert len(specs) == 7
    assert [spec.bundle for spec in specs] == ["S1", "P0", "P1", "P2", "P3", "B1", "M1"]
    assert all(spec.expert == "recent" and spec.valid_year == 2024 for spec in specs)


def _cache(monkeypatch, observed: dict[str, int]) -> None:
    def materialize(cache_root, *, train, valid, feature_fit_rows, **_kwargs):
        observed["train"] = len(train)
        observed["context"] = len(feature_fit_rows)
        observed["valid"] = len(valid)

        def batch(frame, target):
            return FeatureBatch(
                frame["row_id"].to_numpy(),
                frame["season"].to_numpy(),
                frame["game_type"].to_numpy(),
                np.zeros((len(frame), 2), dtype="float32"),
                np.zeros((len(frame), 1), dtype="int64"),
                None if target is None else frame["control_success"].to_numpy(dtype="int8"),
            )

        return SimpleNamespace(
            root=Path(cache_root),
            train=batch(train, True),
            valid=batch(valid, True),
            identity_sha256="e" * 64,
            state=SimpleNamespace(fitted_sources={}),
        )

    monkeypatch.setattr("experiments.temporal_portfolio.t2a.materialize_fold_cache", materialize)


def test_t2a_materialization_uses_recent_rows_but_cutoff_safe_prefix_context(tmp_path: Path, monkeypatch) -> None:
    observed = {}
    _cache(monkeypatch, observed)
    spec = T2AJobSpec("t2a__r__s1__va2024__s3407", "recent", "S1", 2024, 3407)
    materialized = materialize_t2a_job(
        spec,
        data_rows_sha256=sha256(b"rows").hexdigest(),
        t1_decision_sha256=sha256(b"decision").hexdigest(),
        train=_train(),
        history=pd.DataFrame(),
        cache_root=tmp_path / "cache",
    )

    assert len(materialized.training.train_request.train.row_id) == 4
    assert observed == {"train": 4, "context": 16, "valid": 4}
    assert materialized.training.valid_rows == 4
    assert materialized.training.train_request.epochs == 12
    assert materialized.training.train_request.training_config["patience"] == 3
    assert materialized.training.sample_weight.tolist() == [1.0] * 4
    assert tuple(materialized.training.identity.payload["train_seasons"]) == (2023,)
    assert tuple(materialized.training.identity.payload["features"]) == ("base", "S1")
    assert materialized.training.identity.payload["model"]["t1_decision_sha256"] == sha256(b"decision").hexdigest()
    materialized.training.validate_seals()


def test_t2a_multi_materialization_uses_fixed_t1_decay(tmp_path: Path, monkeypatch) -> None:
    observed = {}
    _cache(monkeypatch, observed)
    spec = T2AJobSpec("t2a__m__p1__va2024__s3407", "multi", "P1", 2024, 3407)
    materialized = materialize_t2a_job(
        spec,
        data_rows_sha256=sha256(b"rows").hexdigest(),
        t1_decision_sha256=sha256(b"decision").hexdigest(),
        train=_train(),
        history=pd.DataFrame(),
        cache_root=tmp_path / "cache",
    )

    assert len(materialized.training.train_request.train.row_id) == 16
    assert materialized.training.sample_weight.tolist() == [0.55**3] * 4 + [0.55**2] * 4 + [0.55] * 4 + [1.0] * 4
    assert tuple(materialized.training.identity.payload["train_seasons"]) == (2020, 2021, 2022, 2023)


def test_phase_m_selects_top_two_and_a_structurally_different_wildcard() -> None:
    evidence = {
        "P1": {"status": "completed", "gain": 0.0010, "bootstrap_lower": 0.0008, "max_segment_regression": 0.0},
        "P2": {"status": "completed", "gain": 0.0009, "bootstrap_lower": 0.0007, "max_segment_regression": 0.0},
        "S1": {"status": "completed", "gain": 0.0007, "bootstrap_lower": 0.0005, "max_segment_regression": 0.0},
        "B1": {"status": "completed", "gain": 0.0004, "bootstrap_lower": 0.0002, "max_segment_regression": 0.0},
        "M1": {"status": "insufficient_mapping"},
        "P0": {"status": "completed", "gain": -0.0001, "bootstrap_lower": -0.0002, "max_segment_regression": 0.0001},
        "P3": {"status": "completed", "gain": 0.0002, "bootstrap_lower": -0.0001, "max_segment_regression": 0.0},
    }

    assert select_phase_m_bundles(evidence) == ("P1", "P2", "S1")


def test_t2a_feature_evaluation_blends_with_fixed_t1_multi_and_beats_anchor() -> None:
    target = [0, 1, 0, 1]
    base = {
        "row_id": ["a", "b", "c", "d"],
        "valid_year": [2024] * 4,
        "target": target,
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
    anchor = pd.DataFrame({**base, "probability": [0.4, 0.6, 0.4, 0.6]})
    multi = pd.DataFrame({**base, "probability": [0.3, 0.7, 0.3, 0.7]})
    feature = pd.DataFrame({**base, "probability": [0.1, 0.9, 0.1, 0.9]}).drop(columns="valid_year")

    evidence = evaluate_feature_candidate(anchor, multi, feature, bootstrap_repeats=100)

    assert evidence["gain"] > 0
    assert evidence["bootstrap_lower"] > 0
    assert evidence["max_segment_regression"] == 0


def test_t2a_feature_evaluation_accepts_worker_multi_prediction_without_valid_year() -> None:
    target = [0, 1, 0, 1]
    base = {
        "row_id": ["a", "b", "c", "d"],
        "valid_year": [2024] * 4,
        "target": target,
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
    anchor = pd.DataFrame({**base, "probability": [0.4, 0.6, 0.4, 0.6]})
    multi_worker = pd.DataFrame({**base, "probability": [0.3, 0.7, 0.3, 0.7]}).drop(columns="valid_year")
    recent_worker = pd.DataFrame({**base, "probability": [0.1, 0.9, 0.1, 0.9]}).drop(columns="valid_year")

    evidence = evaluate_feature_candidate(
        anchor, multi_worker, recent_worker, bootstrap_repeats=100
    )

    assert evidence["gain"] > 0


def test_t2a_kaggle_cell_is_deterministic_small_and_single_handoff(tmp_path: Path) -> None:
    first = build_t2a_kaggle_cell(tmp_path / "first.py")
    second = build_t2a_kaggle_cell(tmp_path / "second.py")
    text = first.read_text(encoding="utf-8")

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 1_000_000
    assert "T2A_HANDOFF_READY" in text
    assert "temporal_t2a_handoff.zip" in text
    assert "from experiments.temporal_portfolio.t1_artifacts import verify_compact_result" in text
    assert "verify_compact_result(job_root)" in text
    compile(text, str(first), "exec")


def test_t2a_runner_executes_seven_recent_then_three_selected_multi_jobs(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    paths, hashes, sizes = [], {}, {}
    for name in ("train.csv", "test.csv", "trackman_history.csv", "sample_submission.csv"):
        path = data / name
        path.write_text("fixture\n", encoding="utf-8")
        paths.append(path)
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        sizes[name] = path.stat().st_size
    verified = VerifiedOfficialData(
        data, *paths, MappingProxyType(hashes), 10, 2,
        MappingProxyType(sizes), MappingProxyType({}),
    )
    prepared = VerifiedT2AInput(
        tmp_path / "input.zip", "a" * 64, "b" * 64,
        _input_identity(verified), "c" * 64,
    )
    monkeypatch.setattr("experiments.temporal_portfolio.t2a_runner.verify_t2a_input", lambda _path: prepared)
    monkeypatch.setattr("experiments.temporal_portfolio.t2a_runner._reference_frames", lambda _prepared: (pd.DataFrame(), pd.DataFrame()))

    def materialize(spec, **_kwargs):
        identity = TrainingIdentity.from_payload(
            {"data_rows": prepared.data_rows_sha256, "train_seasons": [2023], "valid_year": 2024,
             "decay": None, "features": ["base", spec.bundle], "model": {"job_id": spec.job_id},
             "loss": "bce", "seed": 3407}
        )
        return SimpleNamespace(training=SimpleNamespace(job_id=spec.job_id, identity=identity), plan=SimpleNamespace(max_seconds=900))

    monkeypatch.setattr("experiments.temporal_portfolio.t2a_runner.materialize_t2a_job", materialize)
    monkeypatch.setattr("experiments.temporal_portfolio.t2a_runner.verify_worker_result", lambda path: json.loads((Path(path) / "worker_result.json").read_text()))
    phase_r = {
        bundle: {"status": "completed", "gain": 0.001 - index * 0.00005, "bootstrap_lower": 0.0001,
                 "max_segment_regression": 0.0}
        for index, bundle in enumerate(("S1", "P0", "P1", "P2", "P3", "B1", "M1"))
    }
    monkeypatch.setattr("experiments.temporal_portfolio.t2a_runner._phase_r_evidence", lambda *_args: phase_r)
    monkeypatch.setattr("experiments.temporal_portfolio.t2a_runner._phase_m_evidence", lambda *_args: {})

    class Handle:
        returncode = 0
        def poll(self): return 0
        def terminate(self): raise AssertionError
        def wait(self, timeout=None): return 0
        def kill(self): raise AssertionError

    class Launcher:
        def __init__(self): self.jobs = []
        def start(self, job, output, *, gpu, deadline):
            self.jobs.append((job.training.job_id, gpu))
            output.mkdir(parents=True, exist_ok=True)
            (output / "worker_result.json").write_text(json.dumps({"status": "completed", "job_id": job.training.job_id, "training_identity_sha256": job.training.identity.sha256}))
            return Handle()

    launcher = Launcher()
    result = run_t2a_stage(
        verified=verified, t2a_input=prepared.path, output_root=tmp_path / "output",
        deadline=20_000, launcher=launcher, frame_loader=lambda _path: pd.DataFrame(),
        clock=lambda: 1_000.0, sleeper=lambda _seconds: None,
    )

    assert result.status == "completed"
    assert len(launcher.jobs) == 10
    assert sum(job_id.startswith("t2a__r__") for job_id, _ in launcher.jobs) == 7
    assert sum(job_id.startswith("t2a__m__") for job_id, _ in launcher.jobs) == 3


def test_t2a_review_and_resume_are_compact_verified_bundles(tmp_path: Path) -> None:
    job_id = "t2a__r__s1__va2024__s3407"
    root = tmp_path / "jobs" / job_id
    root.mkdir(parents=True)
    records = {}
    for name, data in {
        "predictions.csv": b"row_id,target,probability\na,1,0.8\n",
        "metrics.json": b"{}",
        "checkpoint_meta.json": b"{}",
    }.items():
        (root / name).write_bytes(data)
        records[name] = {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
    identity = "d" * 64
    (root / "compact_result.json").write_text(
        json.dumps(
            {"schema_version": 1, "job_id": job_id, "status": "completed",
             "training_identity_sha256": identity, "members": records},
            sort_keys=True, separators=(",", ":"),
        ), encoding="utf-8",
    )
    bundles = write_t2a_bundles(
        tmp_path / "bundles", jobs_root=tmp_path / "jobs",
        completed=(job_id,), pending=(), failed=(), skipped={},
        evidence={"promoted": []}, t1_decision_sha256="e" * 64,
    )

    manifest, _ = _read_bundle(bundles.review, expected_kind="temporal_t2a_review_v1")
    assert manifest["completed"] == [job_id]
    restored = restore_t2a_resume_source(bundles.resume, tmp_path / "restored")
    assert verify_compact_result(restored / "jobs" / job_id)["training_identity_sha256"] == identity


def test_t2a_prediction_accepts_restored_compact_result(tmp_path: Path) -> None:
    root = tmp_path / "restored" / "jobs" / "t2a__r__s1__va2024__s3407"
    root.mkdir(parents=True)
    records = {}
    for name, data in {
        "predictions.csv": b"row_id,target,probability\na,1,0.8\n",
        "metrics.json": b"{}",
        "checkpoint_meta.json": b"{}",
    }.items():
        (root / name).write_bytes(data)
        records[name] = {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
    (root / "compact_result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": root.name,
                "status": "completed",
                "training_identity_sha256": "d" * 64,
                "members": records,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    prediction = _prediction(root)

    assert prediction.loc[0, "probability"] == 0.8
