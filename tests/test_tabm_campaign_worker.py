from __future__ import annotations

import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.preprocessing import PreprocessingSpec
from experiments.tabm_campaign import worker as worker_module
from experiments.tabm_campaign.cache import CacheIdentity
from experiments.tabm_campaign.runner import CampaignJob, CampaignJobResult


_INDEPENDENT_DL_ROOT = Path(worker_module.__file__).resolve().parents[1] / "independent_dl"
_MODERN_EVIDENCE_KEYS = (
    "preprocessing_code_sha256",
    "row_feature_code_sha256",
    "feature_code_sha256",
    "training_source_sha256",
    "checkpoint_sha256",
    "predictions_sha256",
)


def _expected_training_source_sha256() -> str:
    return sha256(
        (_INDEPENDENT_DL_ROOT / "training.py").read_bytes()
        + (_INDEPENDENT_DL_ROOT / "models" / "tabm.py").read_bytes()
    ).hexdigest()


def _modern_evidence(job_dir: Path, cache_digest: str) -> dict[str, object]:
    return {
        "cache_digest": cache_digest,
        "preprocessing_code_sha256": sha256(
            (_INDEPENDENT_DL_ROOT / "preprocessing.py").read_bytes()
        ).hexdigest(),
        "row_feature_code_sha256": sha256(
            (_INDEPENDENT_DL_ROOT / "row_features.py").read_bytes()
        ).hexdigest(),
        "feature_code_sha256": sha256(
            (_INDEPENDENT_DL_ROOT / "features.py").read_bytes()
        ).hexdigest(),
        "training_source_sha256": _expected_training_source_sha256(),
        "checkpoint_sha256": sha256(
            (job_dir / "best_checkpoint.pt").read_bytes()
        ).hexdigest(),
        "predictions_sha256": sha256(
            (job_dir / "predictions.csv").read_bytes()
        ).hexdigest(),
    }


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


def _write_reusable_completed(
    job_dir: Path,
    *,
    job: CampaignJob | None = None,
    candidate_id: str | None = None,
    status: str = "completed",
    resource_evidence: dict[str, object] | None = None,
    brier: float | None = 0.0625,
) -> tuple[CampaignJob, CampaignJobResult]:
    job = _job() if job is None else job
    job_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = job_dir / "best_checkpoint.pt"
    checkpoint.write_bytes(b"best")
    predictions = job_dir / "predictions.csv"
    predictions.write_text(
        "row_id,target,probability,game_type_segment\n"
        "v1,0,0.25,R\n"
        "v2,1,0.75,F\n",
        encoding="utf-8",
    )
    cache_digest = "c" * 64
    worker_module._atomic_json(
        job_dir / "checkpoint_meta.json",
        {
            "candidate_id": job.candidate_id,
            "epoch": 1,
            "checkpoint": "checkpoint.pt",
            "checkpoint_binding": {
                "config_sha256": worker_module._job_sha(job),
                "cache_sha256": cache_digest,
                "training_source_sha256": _expected_training_source_sha256(),
            },
        },
    )
    result = CampaignJobResult(
        candidate_id or job.candidate_id,
        status,
        brier,
        1,
        2,
        checkpoint,
        predictions,
        (
            {"cache_digest": cache_digest}
            if resource_evidence is None
            else resource_evidence
        ),
        None,
    )
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(result, worker_module._job_sha(job)),
    )
    return job, result


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


def test_baseline_job_payload_and_sha_match_legacy_schema_exactly() -> None:
    baseline = _job()
    legacy_payload = {
        "candidate_id": "candidate",
        "capacity": "p2",
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.1,
        "num_embedding": "piecewise_linear",
        "loss": "bce",
        "scheduler": "plateau",
        "learning_rate": 0.0006,
        "seed": 42,
        "train_end_year": 2023,
        "valid_year": 2024,
        "sample_mode": "proxy",
        "max_epochs": 8,
        "min_epochs": 3,
        "patience": 3,
    }

    assert worker_module._job_payload(baseline) == legacy_payload
    assert worker_module._job_sha(baseline) == (
        "1d9139cb6ebb44148da20b1e209161344c964dfe6b41bd2dbae89d232a2d6fd3"
    )

    bundled = replace(baseline, feature_bundle="count_context")
    assert worker_module._job_payload(bundled) == {
        **legacy_payload,
        "feature_bundle": "count_context",
    }
    assert worker_module._job_sha(bundled) != worker_module._job_sha(baseline)


def test_resource_identity_evidence_contains_all_code_provenance() -> None:
    identity = SimpleNamespace(
        preprocessing_code_sha256="1" * 64,
        row_feature_code_sha256="2" * 64,
        feature_code_sha256="3" * 64,
        digest=lambda: "4" * 64,
    )

    evidence = worker_module._resource_identity_evidence(
        _job(feature_bundle="count_context"),
        PreprocessingSpec("dl_standard", ("hand_matchup", "count_context")),
        identity,
        cache_reused=True,
        training_source_sha256="5" * 64,
    )

    assert evidence == {
        "cache_digest": "4" * 64,
        "cache_reused": True,
        "feature_bundle": "count_context",
        "preprocessing_spec": {
            "profile": "dl_standard",
            "components": ("hand_matchup", "count_context"),
        },
        "preprocessing_code_sha256": "1" * 64,
        "row_feature_code_sha256": "2" * 64,
        "feature_code_sha256": "3" * 64,
        "training_source_sha256": "5" * 64,
    }


def test_training_source_sha_uses_current_training_and_tabm_sources() -> None:
    assert worker_module._training_source_sha256() == (
        _expected_training_source_sha256()
    )


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

    written_job = json.loads((job_dir / "job.json").read_text())
    payload = json.loads((job_dir / "worker_result.json").read_text())
    assert written_job == worker_module._job_payload(job)
    assert "feature_bundle" not in written_job
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
    job_dir = tmp_path / "jobs" / "candidate"
    job, completed = _write_reusable_completed(job_dir)

    def unexpected_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("completed job should not restart")

    monkeypatch.setattr(worker_module.subprocess, "Popen", unexpected_popen)
    monkeypatch.setattr(worker_module.time, "time", lambda: 0.0)
    result = worker_module.SubprocessCampaignRuntime(tmp_path).run_jobs(
        "P", (job,), tmp_path / "jobs", gpu_count=1, job_deadline=100.0
    )

    assert result[0].status == "completed"
    assert result[0].checkpoint == completed.checkpoint


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


def test_completed_result_rejects_external_checkpoint(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job, result = _write_reusable_completed(job_dir)
    external = tmp_path / "external.pt"
    external.write_bytes(b"best")
    foreign = replace(result, checkpoint=external)
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(foreign, worker_module._job_sha(job)),
    )

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


@pytest.mark.parametrize(
    "csv_text",
    [
        "",
        "row_id,target\nv1,0\n",
        "row_id,target,probability\nv1,2,0.5\n",
        "row_id,target,probability\nv1,0,nan\n",
        "row_id,target,probability\nv1,0,1.5\n",
        "row_id,target,probability\nv1,0,0.2\nv1,1,0.8\n",
    ],
)
def test_completed_result_rejects_malformed_prediction_csv(
    tmp_path: Path, csv_text: str
) -> None:
    job_dir = tmp_path / "job"
    job, result = _write_reusable_completed(job_dir)
    assert result.predictions_path is not None
    result.predictions_path.write_text(csv_text, encoding="utf-8")

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


def test_worker_result_rejects_candidate_mismatch_and_invalid_status(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    job, result = _write_reusable_completed(job_dir)
    mismatch = replace(result, candidate_id="foreign")
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(mismatch, worker_module._job_sha(job)),
    )
    assert (
        worker_module.SubprocessCampaignRuntime._read_worker_result(job_dir, job)
        is None
    )

    invalid = replace(result, status="running")
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(invalid, worker_module._job_sha(job)),
    )
    assert (
        worker_module.SubprocessCampaignRuntime._read_worker_result(job_dir, job)
        is None
    )


@pytest.mark.parametrize("meta_defect", ["candidate", "binding"])
def test_completed_result_rejects_bad_checkpoint_metadata(
    tmp_path: Path, meta_defect: str
) -> None:
    job_dir = tmp_path / "job"
    job, _ = _write_reusable_completed(job_dir)
    meta = json.loads((job_dir / "checkpoint_meta.json").read_text())
    if meta_defect == "candidate":
        meta["candidate_id"] = "foreign"
    else:
        meta["checkpoint_binding"]["config_sha256"] = "0" * 64
    worker_module._atomic_json(job_dir / "checkpoint_meta.json", meta)

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


def test_completed_result_rejects_symlink_artifact(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job, result = _write_reusable_completed(job_dir)
    assert result.checkpoint is not None
    external = tmp_path / "external.pt"
    external.write_bytes(b"best")
    result.checkpoint.unlink()
    result.checkpoint.symlink_to(external)

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


def test_completed_result_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job, result = _write_reusable_completed(
        job_dir,
        resource_evidence={
            "checkpoint_sha256": "0" * 64,
            "predictions_sha256": "1" * 64,
        },
    )
    assert result.checkpoint is not None and result.predictions_path is not None

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


@pytest.mark.parametrize("missing_key", ("cache_digest", *_MODERN_EVIDENCE_KEYS))
def test_feature_result_requires_every_modern_evidence_key(
    tmp_path: Path, missing_key: str
) -> None:
    job_dir = tmp_path / "job"
    job = _job(feature_bundle="count_context")
    job, result = _write_reusable_completed(job_dir, job=job)
    evidence = _modern_evidence(job_dir, "c" * 64)
    evidence.pop(missing_key)
    result = replace(result, resource_evidence=evidence)
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(result, worker_module._job_sha(job)),
    )

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


def test_feature_result_accepts_complete_current_modern_evidence(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    job = _job(feature_bundle="count_context")
    job, result = _write_reusable_completed(job_dir, job=job)
    result = replace(
        result, resource_evidence=_modern_evidence(job_dir, "c" * 64)
    )
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(result, worker_module._job_sha(job)),
    )

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is not None


@pytest.mark.parametrize("binding_key", ["cache_sha256", "training_source_sha256"])
def test_completed_result_rejects_wrong_cache_or_training_binding(
    tmp_path: Path, binding_key: str
) -> None:
    job_dir = tmp_path / "job"
    job, _ = _write_reusable_completed(job_dir)
    meta = json.loads((job_dir / "checkpoint_meta.json").read_text())
    meta["checkpoint_binding"][binding_key] = "0" * 64
    worker_module._atomic_json(job_dir / "checkpoint_meta.json", meta)

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


def test_legacy_baseline_requires_cache_digest_but_not_modern_hashes(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    job, result = _write_reusable_completed(job_dir)

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is not None

    result = replace(result, resource_evidence={})
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(result, worker_module._job_sha(job)),
    )
    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


@pytest.mark.parametrize("artifact_name", ["best_checkpoint.pt", "predictions.csv"])
def test_modern_result_rejects_artifact_changed_after_hashing(
    tmp_path: Path, artifact_name: str
) -> None:
    job_dir = tmp_path / "job"
    job = _job(feature_bundle="count_context")
    job, result = _write_reusable_completed(job_dir, job=job)
    evidence = _modern_evidence(job_dir, "c" * 64)
    result = replace(result, resource_evidence=evidence)
    worker_module._atomic_json(
        job_dir / "worker_result.json",
        worker_module._serialize_result(result, worker_module._job_sha(job)),
    )
    artifact = job_dir / artifact_name
    if artifact_name == "predictions.csv":
        artifact.write_text(
            "row_id,target,probability\nv1,0,0.20\nv2,1,0.80\n",
            encoding="utf-8",
        )
    else:
        artifact.write_bytes(b"changed")

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


@pytest.mark.parametrize("stored_brier", [None, 0.2, float("nan"), float("inf")])
def test_completed_result_rejects_missing_nonfinite_or_mismatched_brier(
    tmp_path: Path, stored_brier: float | None
) -> None:
    job_dir = tmp_path / "job"
    job, result = _write_reusable_completed(job_dir)
    result = replace(result, brier=stored_brier)
    payload = worker_module._serialize_result(result, worker_module._job_sha(job))
    (job_dir / "worker_result.json").write_text(json.dumps(payload), encoding="utf-8")

    assert worker_module.SubprocessCampaignRuntime._read_result(job_dir, job) is None


def test_completed_result_accepts_exact_recomputed_brier(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job, _ = _write_reusable_completed(job_dir, brier=0.0625)

    result = worker_module.SubprocessCampaignRuntime._read_result(job_dir, job)

    assert result is not None
    assert result.brier == 0.0625


def test_artifact_hash_evidence_records_checkpoint_and_predictions(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_checkpoint.pt"
    predictions = tmp_path / "predictions.csv"
    checkpoint.write_bytes(b"best")
    predictions.write_bytes(b"row_id,target,probability\nv1,0,0.25\n")

    evidence = worker_module._artifact_hash_evidence(checkpoint, predictions)

    assert evidence == {
        "checkpoint_sha256": sha256(b"best").hexdigest(),
        "predictions_sha256": sha256(predictions.read_bytes()).hexdigest(),
    }


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
