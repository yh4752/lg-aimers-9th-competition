from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.features import FeatureBatch
from experiments.temporal_portfolio.contracts import build_stage_jobs, load_contract
from experiments.temporal_portfolio.t1_artifacts import write_t1_bundles
from experiments.temporal_portfolio.t2a_artifacts import write_t2a_bundles
from experiments.temporal_portfolio.t2b_input import (
    T2BInputError,
    prepare_t2b_input,
    verify_t2b_input,
)


def _training_rows() -> pd.DataFrame:
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
        prediction = pd.DataFrame(
            {
                "row_id": [f"2024_{index}" for index in range(4)],
                "target": [0, 1, 0, 1],
                "probability": [0.2, 0.8, 0.2, 0.8],
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
        ).to_csv(index=False).encode("utf-8")
        records = {}
        for name, data in {
            "predictions.csv": prediction,
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
            "t2a_recent_s1_2024.csv",
            "t2a_recent_p3_2024.csv",
            "t2a_recent_p2_2024.csv",
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


def test_t2b_schedule_is_six_single_two_latest_combo_then_two_history() -> None:
    from experiments.temporal_portfolio.t2b import (
        build_phase_c_specs,
        build_phase_f_specs,
        build_phase_h_specs,
    )

    phase_f = build_phase_f_specs()
    phase_c = build_phase_c_specs()
    phase_h = build_phase_h_specs("S1+P3")

    assert len(phase_f) == 6
    assert [(item.bundles, item.valid_year) for item in phase_f] == [
        (("S1",), 2022),
        (("P3",), 2022),
        (("P2",), 2022),
        (("S1",), 2023),
        (("P3",), 2023),
        (("P2",), 2023),
    ]
    assert [(item.bundles, item.valid_year) for item in phase_c] == [
        (("S1", "P3"), 2024),
        (("S1", "P2"), 2024),
    ]
    assert [(item.bundles, item.valid_year) for item in phase_h] == [
        (("S1", "P3"), 2022),
        (("S1", "P3"), 2023),
    ]
    assert len(phase_f + phase_c + phase_h) == 10


def test_t2b_materialization_uses_previous_season_and_cutoff_safe_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.temporal_portfolio.t2b import T2BJobSpec, materialize_t2b_job

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
        "experiments.temporal_portfolio.t2b.materialize_fold_cache", materialize
    )
    job = materialize_t2b_job(
        T2BJobSpec(
            "t2b__f__s1__va2022__s3407", "F", ("S1",), 2022, 3407
        ),
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
    assert job.training.identity.payload["valid_year"] == 2022
    assert job.training.identity.payload["model"]["parent_sha256"] == "b" * 64
    job.training.validate_seals()


def test_t2b_combination_selection_requires_all_latest_fold_gates() -> None:
    from experiments.temporal_portfolio.t2b import select_combination

    good = {
        "S1+P3": {
            "status": "completed",
            "gain": 0.00010,
            "bootstrap_lower": 0.00001,
            "max_segment_regression": 0.00020,
        },
        "S1+P2": {
            "status": "completed",
            "gain": 0.00009,
            "bootstrap_lower": 0.00002,
            "max_segment_regression": 0.00010,
        },
    }
    assert select_combination(good) == "S1+P3"

    tied = {name: {**item, "gain": 0.00010} for name, item in good.items()}
    assert select_combination(tied) == "S1+P3"

    failed = {
        "S1+P3": {**good["S1+P3"], "bootstrap_lower": 0.0},
        "S1+P2": {**good["S1+P2"], "max_segment_regression": 0.00051},
    }
    assert select_combination(failed) is None


def test_t2b_candidate_decision_uses_all_three_folds() -> None:
    from experiments.temporal_portfolio.t2b import decide_t2b_candidates

    folds = {
        "S1": {
            year: {
                "status": "completed",
                "gain": 0.00010,
                "rows": 100,
                "max_segment_regression": 0.0,
            }
            for year in (2022, 2023, 2024)
        }
    }
    combined = {
        "S1": {
            "bootstrap_lower": 0.00005,
            "max_segment_regression": 0.0,
        }
    }

    decisions = decide_t2b_candidates(folds, combined)

    assert len(decisions) == 1
    assert decisions[0].candidate_id == "S1"
    assert decisions[0].status == "champion"


def test_t2b_fold_evaluation_accepts_worker_prediction_without_year() -> None:
    from experiments.temporal_portfolio.t2b import evaluate_t2b_fold

    base = {
        "row_id": ["a", "b", "c", "d"],
        "valid_year": [2023] * 4,
        "target": [0, 1, 0, 1],
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
    recent = pd.DataFrame({**base, "probability": [0.1, 0.9, 0.1, 0.9]}).drop(
        columns="valid_year"
    )

    evidence = evaluate_t2b_fold(
        anchor, multi, recent, valid_year=2023, bootstrap_repeats=100
    )

    assert evidence["rows"] == 4
    assert evidence["gain"] > 0
    assert evidence["bootstrap_lower"] > 0


def test_t2b_review_and_resume_are_compact_verified_bundles(tmp_path: Path) -> None:
    from experiments.temporal_portfolio.t1_artifacts import (
        _read_bundle,
        verify_compact_result,
    )
    from experiments.temporal_portfolio.t2b_artifacts import (
        restore_t2b_resume_source,
        write_t2b_bundles,
    )

    job_id = "t2b__f__s1__va2022__s3407"
    root = tmp_path / "jobs" / job_id
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
    identity = "d" * 64
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
    bundles = write_t2b_bundles(
        tmp_path / "bundles",
        jobs_root=tmp_path / "jobs",
        completed=(job_id,),
        pending=(),
        failed=(),
        evidence={"promoted": []},
        parent_sha256="e" * 64,
    )

    manifest, _ = _read_bundle(
        bundles.review, expected_kind="temporal_t2b_review_v1"
    )
    assert manifest["completed"] == [job_id]
    restored = restore_t2b_resume_source(bundles.resume, tmp_path / "restored")
    payload = verify_compact_result(restored / "jobs" / job_id)
    assert payload["training_identity_sha256"] == identity


def test_t2b_runner_never_starts_more_than_ten_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import MappingProxyType

    from experiments.temporal_portfolio.identity import TrainingIdentity
    from experiments.temporal_portfolio.inputs import VerifiedOfficialData
    from experiments.temporal_portfolio.t1_runner import _input_identity
    from experiments.temporal_portfolio.t2b_input import VerifiedT2BInput
    from experiments.temporal_portfolio.t2b_runner import run_t2b_stage

    data = tmp_path / "data"
    data.mkdir()
    paths, hashes, sizes = [], {}, {}
    for name in (
        "train.csv",
        "test.csv",
        "trackman_history.csv",
        "sample_submission.csv",
    ):
        path = data / name
        path.write_text("fixture\n", encoding="utf-8")
        paths.append(path)
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        sizes[name] = path.stat().st_size
    verified = VerifiedOfficialData(
        data,
        *paths,
        MappingProxyType(hashes),
        10,
        2,
        MappingProxyType(sizes),
        MappingProxyType({}),
    )
    prepared = VerifiedT2BInput(
        tmp_path / "input.zip",
        "a" * 64,
        _input_identity(verified),
        "b" * 64,
        "c" * 64,
        "d" * 64,
        ("S1", "P3", "P2"),
    )
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2b_runner.verify_t2b_input",
        lambda _path: prepared,
    )
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2b_runner.load_t2b_references",
        lambda _prepared: ({}, {}, {}, {}),
    )

    def materialize(spec, **_kwargs):
        identity = TrainingIdentity.from_payload(
            {
                "data_rows": prepared.data_rows_sha256,
                "train_seasons": [spec.valid_year - 1],
                "valid_year": spec.valid_year,
                "decay": None,
                "features": ["base", *spec.bundles],
                "model": {"job_id": spec.job_id},
                "loss": "bce",
                "seed": 3407,
            }
        )
        return SimpleNamespace(
            training=SimpleNamespace(job_id=spec.job_id, identity=identity),
            plan=SimpleNamespace(max_seconds=900),
        )

    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2b_runner.materialize_t2b_job",
        materialize,
    )
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2b_runner.verify_worker_result",
        lambda path: json.loads((Path(path) / "worker_result.json").read_text()),
    )
    phase_f = {
        bundle: {
            year: {
                "status": "completed",
                "gain": 0.0001,
                "rows": 100,
                "max_segment_regression": 0.0,
            }
            for year in (2022, 2023, 2024)
        }
        for bundle in ("S1", "P3", "P2")
    }
    phase_c = {
        name: {
            "status": "completed",
            "gain": 0.0001,
            "bootstrap_lower": 0.00001,
            "max_segment_regression": 0.0,
        }
        for name in ("S1+P3", "S1+P2")
    }
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2b_runner._phase_f_evidence",
        lambda *_args: phase_f,
    )
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2b_runner._phase_c_evidence",
        lambda *_args: phase_c,
    )
    monkeypatch.setattr(
        "experiments.temporal_portfolio.t2b_runner._final_evidence",
        lambda *_args: ({"promoted": []}, ()),
    )

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

    class Launcher:
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

    launcher = Launcher()
    result = run_t2b_stage(
        verified=verified,
        t2b_input=prepared.path,
        output_root=tmp_path / "output",
        deadline=20_000,
        launcher=launcher,
        frame_loader=lambda _path: pd.DataFrame(),
        clock=lambda: 1_000.0,
        sleeper=lambda _seconds: None,
    )

    assert result.status == "completed"
    assert len(launcher.jobs) == 10
    assert len({job_id for job_id, _ in launcher.jobs}) == 10


def test_t2b_gpu_probe_accepts_exactly_two_t4_devices() -> None:
    from experiments.temporal_portfolio.t1_runner import require_two_t4_gpus

    assert require_two_t4_gpus(
        lambda: ("Tesla T4", "Tesla T4"), log_prefix="T2B"
    ) == ("Tesla T4", "Tesla T4")


def test_t2b_stops_new_jobs_twenty_five_minutes_before_deadline() -> None:
    from experiments.temporal_portfolio.t2b_runner import (
        _HANDOFF_RESERVE_SECONDS,
        _STOP_NEW_SECONDS,
    )

    assert _STOP_NEW_SECONDS + _HANDOFF_RESERVE_SECONDS == 25 * 60


def test_t2b_kaggle_cell_is_deterministic_small_and_single_handoff(
    tmp_path: Path,
) -> None:
    from experiments.temporal_portfolio.t2b_platform import build_t2b_kaggle_cell

    first = build_t2b_kaggle_cell(tmp_path / "first.py")
    second = build_t2b_kaggle_cell(tmp_path / "second.py")
    text = first.read_text(encoding="utf-8")

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 1_000_000
    assert "files.download" not in text
    assert "T2B_HANDOFF_READY" in text
    assert "T2B_EMERGENCY_HANDOFF_READY" in text
    assert "temporal_t2b_handoff.zip" in text
    assert "temporal_t2b_emergency_handoff.zip" in text
    compile(text, str(first), "exec")
