from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.preprocessing import PreprocessingSpec
from experiments.tabm_campaign import worker as worker_module
from experiments.tabm_campaign.cache import CacheIdentity
from experiments.tabm_campaign.runner import CampaignJob, CampaignJobResult


def _job(**changes: object) -> CampaignJob:
    values = dict(
        candidate_id="candidate",
        capacity="p2",
        k=32,
        width=512,
        blocks=4,
        dropout=0.1,
        num_embedding="piecewise_linear",
        loss="bce",
        scheduler="plateau",
        learning_rate=0.0006,
        seed=42,
        train_end_year=2023,
        valid_year=2024,
        sample_mode="proxy",
        max_epochs=8,
        min_epochs=3,
        patience=3,
    )
    values.update(changes)
    return CampaignJob(**values)


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = pd.DataFrame(
        {
            "row_id": ["f1", "f2"],
            "pitcher_id": [10, 11],
            "batter_id": [20, 21],
        }
    )
    valid = pd.DataFrame(
        {
            "row_id": ["v1", "v2"],
            "control_success": [0, 1],
            "game_type": ["R", "F"],
            "game_month": [4, 5],
            "pitcher_hand": ["R", "L"],
            "batter_hand": ["L", "L"],
            "pitcher_id": [10, 99],
            "batter_id": [98, 21],
            "li": [1.6, 0.8],
            "inning": [7, 6],
            "runner_on_2b": [1, 0],
            "runner_on_3b": [0, 0],
        }
    )
    return fit, valid


def _valid_batch(**changes: object) -> SimpleNamespace:
    values = dict(
        row_id=np.asarray(["v1", "v2"]),
        y=np.asarray([0.0, 1.0]),
        game_type=np.asarray(["R", "F"]),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_preprocessing_spec_supports_baseline_and_one_bundle() -> None:
    assert worker_module._preprocessing_spec(_job()) == PreprocessingSpec(
        "dl_standard", ("hand_matchup",)
    )
    assert worker_module._preprocessing_spec(
        _job(feature_bundle="count_context")
    ) == PreprocessingSpec("dl_standard", ("hand_matchup", "count_context"))


def test_preprocessing_spec_rejects_unknown_bundle() -> None:
    with pytest.raises(ValueError, match="unknown row feature bundle"):
        worker_module._preprocessing_spec(_job(feature_bundle="unknown"))


def test_run_worker_rejects_unknown_bundle_before_data_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_lookup(*args: object, **kwargs: object) -> None:
        raise AssertionError("data lookup must not run")

    monkeypatch.setattr(worker_module, "_find_one", unexpected_lookup)
    with pytest.raises(ValueError, match="unknown row feature bundle"):
        worker_module.run_worker(
            _job(feature_bundle="unknown"),
            tmp_path / "missing-data",
            tmp_path / "output",
            tmp_path / "cache",
            123.0,
        )


def test_feature_bundle_changes_job_and_cache_provenance() -> None:
    baseline = _job()
    bundled = replace(baseline, feature_bundle="count_context")
    assert worker_module._job_sha(baseline) != worker_module._job_sha(bundled)

    empty = pd.DataFrame()
    cache_kwargs = dict(
        train=empty,
        valid=empty,
        history=empty,
        train_end_year=2023,
        valid_year=2024,
        sample_ids=("f1",),
    )
    baseline_cache = CacheIdentity.from_frames(
        **cache_kwargs, spec=worker_module._preprocessing_spec(baseline)
    )
    bundled_cache = CacheIdentity.from_frames(
        **cache_kwargs, spec=worker_module._preprocessing_spec(bundled)
    )
    assert baseline_cache.digest() != bundled_cache.digest()


def test_prediction_evidence_uses_fit_only_ids_and_preserves_alignment() -> None:
    fit, valid = _rows()

    result = worker_module._prediction_evidence_frame(
        fit_rows=fit,
        valid_rows=valid,
        valid_batch=_valid_batch(),
        probability=np.asarray([0.25, 0.75]),
        category_maps={"pitcher_id": {"10": 0}, "batter_id": {"21": 0}},
    )

    assert result.loc[:, ["row_id", "target", "probability"]].to_dict("list") == {
        "row_id": ["v1", "v2"],
        "target": [0, 1],
        "probability": [0.25, 0.75],
    }
    assert result["pitcher_id_segment"].tolist() == ["known", "oov"]
    assert result["batter_id_segment"].tolist() == ["oov", "known"]
    assert result["pitcher_id_known"].tolist() == ["known", "oov"]
    assert result["batter_id_known"].tolist() == ["oov", "known"]
    assert list(result.columns[-7:]) == [
        "game_type_segment",
        "hand_matchup_segment",
        "pitcher_id_segment",
        "batter_id_segment",
        "li_high",
        "late_inning",
        "runner_in_scoring_position",
    ]


@pytest.mark.parametrize(
    ("valid_change", "batch_change", "probability", "message"),
    [
        ({"row_id": ["v1", "v1"]}, {}, [0.25, 0.75], "unique"),
        ({"control_success": [1, 0]}, {}, [0.25, 0.75], "target"),
        ({}, {}, [np.nan, 0.75], "finite"),
        ({}, {}, [-0.01, 0.75], r"\[0, 1\]"),
        ({}, {}, [0.25, 1.01], r"\[0, 1\]"),
    ],
)
def test_prediction_evidence_rejects_invalid_alignment_before_csv(
    valid_change: dict[str, object],
    batch_change: dict[str, object],
    probability: list[float],
    message: str,
) -> None:
    fit, valid = _rows()
    for column, values in valid_change.items():
        valid[column] = values

    with pytest.raises(RuntimeError, match=message):
        worker_module._prediction_evidence_frame(
            fit_rows=fit,
            valid_rows=valid,
            valid_batch=_valid_batch(**batch_change),
            probability=np.asarray(probability),
            category_maps={},
        )


def test_prediction_evidence_rejects_unequal_row_count() -> None:
    fit, valid = _rows()

    with pytest.raises(RuntimeError, match="equal lengths"):
        worker_module._prediction_evidence_frame(
            fit_rows=fit,
            valid_rows=valid.iloc[:1].copy(),
            valid_batch=_valid_batch(),
            probability=np.asarray([0.25, 0.75]),
            category_maps={},
        )


def test_prediction_evidence_rejects_row_order_mismatch() -> None:
    fit, valid = _rows()

    with pytest.raises(RuntimeError, match="row_id order differs"):
        worker_module._prediction_evidence_frame(
            fit_rows=fit,
            valid_rows=valid.iloc[::-1].reset_index(drop=True),
            valid_batch=_valid_batch(),
            probability=np.asarray([0.25, 0.75]),
            category_maps={},
        )


def test_training_budget_status_is_inconclusive_only_when_reached() -> None:
    assert worker_module._training_status(True) == "inconclusive"
    assert worker_module._training_status(False) == "completed"


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [("completed", 0), ("inconclusive", 0), ("failed", 1)],
)
def test_main_atomically_writes_each_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, expected_exit: int
) -> None:
    job = _job()
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(asdict(job)), encoding="utf-8")
    output_dir = tmp_path / status
    result = CampaignJobResult(
        job.candidate_id, status, 0.2 if status != "failed" else None, 1, 2,
        None, None, {}, None if status != "failed" else "failure",
    )
    monkeypatch.setattr(worker_module, "run_worker", lambda *args: result)

    exit_code = worker_module._main(
        [
            "--job", str(job_path),
            "--data-dir", str(tmp_path / "data"),
            "--output-dir", str(output_dir),
            "--cache-root", str(tmp_path / "cache"),
            "--deadline", "123",
        ]
    )

    payload = json.loads((output_dir / "worker_result.json").read_text())
    assert exit_code == expected_exit
    assert payload["status"] == status
    assert payload["job_sha256"] == worker_module._job_sha(job)
    assert not (output_dir / "worker_result.json.tmp").exists()


def test_main_persists_unexpected_exception_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job()
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(asdict(job)), encoding="utf-8")

    def fail(*args: object) -> CampaignJobResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(worker_module, "run_worker", fail)
    output_dir = tmp_path / "failed"
    exit_code = worker_module._main(
        [
            "--job", str(job_path),
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
            "--cache-root", str(tmp_path / "cache"),
            "--deadline", "123",
        ]
    )

    payload = json.loads((output_dir / "worker_result.json").read_text())
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["job_sha256"] == worker_module._job_sha(job)
    assert "RuntimeError: boom" in payload["failure"]


class _HangingProcess:
    returncode = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stdout: list[str] = []

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.returncode = -15

    def wait(self, timeout: float) -> int:
        return int(self.returncode or 0)

    def kill(self) -> None:
        self.returncode = -9


class _FinishedProcess(_HangingProcess):
    returncode = 1

    def poll(self) -> int:
        return 1


def test_deadline_grace_result_is_durable_but_resumed_next_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job()
    output_dir = tmp_path / "jobs"
    job_dir = output_dir / job.candidate_id
    job_dir.mkdir(parents=True)
    (job_dir / "best_checkpoint.pt").write_bytes(b"best")
    (job_dir / "checkpoint_meta.json").write_text(
        json.dumps({"epoch": 3, "checkpoint": "checkpoint.pt"}),
        encoding="utf-8",
    )
    calls: list[type[_HangingProcess]] = []

    def hanging_popen(*args: object, **kwargs: object) -> _HangingProcess:
        calls.append(_HangingProcess)
        return _HangingProcess()

    clock_values = iter([0.0, 0.0, 0.0, 221.0])
    monkeypatch.setattr(worker_module.subprocess, "Popen", hanging_popen)
    monkeypatch.setattr(worker_module.time, "time", lambda: next(clock_values))
    monkeypatch.setattr(worker_module.time, "sleep", lambda _: None)
    runtime = worker_module.SubprocessCampaignRuntime(tmp_path)

    first = runtime.run_jobs(
        "P", (job,), output_dir, gpu_count=1, job_deadline=100.0
    )

    payload = json.loads((job_dir / "worker_result.json").read_text())
    assert first[0].status == "inconclusive"
    assert payload["status"] == "inconclusive"
    assert payload["job_sha256"] == worker_module._job_sha(job)
    assert payload["checkpoint"] == str(job_dir / "best_checkpoint.pt")
    assert payload["completed_epochs"] == 4
    assert payload["best_epoch"] is None

    def finished_popen(*args: object, **kwargs: object) -> _FinishedProcess:
        calls.append(_FinishedProcess)
        return _FinishedProcess()

    monkeypatch.setattr(worker_module.subprocess, "Popen", finished_popen)
    monkeypatch.setattr(worker_module.time, "time", lambda: 0.0)
    second = runtime.run_jobs(
        "P", (job,), output_dir, gpu_count=1, job_deadline=100.0
    )

    assert len(calls) == 2
    assert second[0].status == "failed"
    assert (job_dir / "best_checkpoint.pt").is_file()


def test_completed_result_with_checkpoint_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job()
    job_dir = tmp_path / "jobs" / job.candidate_id
    job_dir.mkdir(parents=True)
    checkpoint = job_dir / "best_checkpoint.pt"
    checkpoint.write_bytes(b"best")
    predictions = job_dir / "predictions.csv"
    predictions.write_text("row_id,target,probability\nv1,0,0.25\n", encoding="utf-8")
    completed = CampaignJobResult(
        job.candidate_id, "completed", 0.2, 1, 2, checkpoint, predictions, {}, None
    )
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(completed, worker_module._job_sha(job)),
    )

    def unexpected_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("completed job should not restart")

    monkeypatch.setattr(worker_module.subprocess, "Popen", unexpected_popen)
    monkeypatch.setattr(worker_module.time, "time", lambda: 0.0)
    result = worker_module.SubprocessCampaignRuntime(tmp_path).run_jobs(
        "P", (job,), tmp_path / "jobs", gpu_count=1, job_deadline=100.0
    )

    assert result[0].status == "completed"
    assert result[0].checkpoint == checkpoint


@pytest.mark.parametrize("prediction_state", ["none", "missing"])
def test_completed_result_without_prediction_evidence_is_not_reused(
    tmp_path: Path, prediction_state: str
) -> None:
    job = _job()
    job_dir = tmp_path / job.candidate_id
    job_dir.mkdir()
    checkpoint = job_dir / "best_checkpoint.pt"
    checkpoint.write_bytes(b"best")
    predictions = None if prediction_state == "none" else job_dir / "missing.csv"
    completed = CampaignJobResult(
        job.candidate_id, "completed", 0.2, 1, 2, checkpoint, predictions, {}, None
    )
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(completed, worker_module._job_sha(job)),
    )

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


def test_jobs_not_started_before_deadline_have_no_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job()

    def unexpected_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("worker must not start after the deadline")

    monkeypatch.setattr(worker_module.subprocess, "Popen", unexpected_popen)
    monkeypatch.setattr(worker_module.time, "time", lambda: 100.0)
    monkeypatch.setattr(worker_module.time, "sleep", lambda _: None)
    output_dir = tmp_path / "jobs"

    result = worker_module.SubprocessCampaignRuntime(tmp_path).run_jobs(
        "P", (job,), output_dir, gpu_count=1, job_deadline=100.0
    )

    assert result[0].status == "inconclusive"
    assert result[0].failure == "not_started_before_stage_deadline"
    assert not (output_dir / job.candidate_id).exists()
    assert not (output_dir / job.candidate_id / "worker_result.json").exists()
