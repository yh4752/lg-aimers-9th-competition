from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.tabm_campaign.runner import CampaignJobResult
from experiments.tabm_campaign.row_feature_contracts import (
    load_row_feature_proxy_contract,
)
from experiments.tabm_campaign.row_feature_proxy import (
    RowFeatureProxyError,
    build_proxy_jobs,
    run_row_feature_proxy,
)


CONTRACT = load_row_feature_proxy_contract()


class _Runtime:
    def __init__(self, statuses: dict[str, str] | None = None) -> None:
        self.statuses = statuses or {}
        self.calls: list[tuple[str, tuple[object, ...], int, float]] = []
        self.state_snapshots: list[bytes] = []
        self.preexisting_checkpoints: list[str] = []

    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        self.calls.append((version, jobs, gpu_count, job_deadline))
        state_path = output_dir.parent / "stage_state.json"
        if state_path.is_file():
            self.state_snapshots.append(state_path.read_bytes())
        job = jobs[0]
        job_dir = output_dir / job.candidate_id
        job_dir.mkdir(parents=True, exist_ok=True)
        if (job_dir / "checkpoint.pt").is_file():
            self.preexisting_checkpoints.append(job.candidate_id)
        status = self.statuses.get(job.candidate_id, "completed")
        checkpoint = None
        predictions = None
        brier = None
        if status in {"completed", "inconclusive"}:
            (job_dir / "checkpoint.pt").write_bytes(f"checkpoint:{job.candidate_id}".encode())
            (job_dir / "checkpoint_meta.json").write_text("{}", encoding="utf-8")
            (job_dir / "best_checkpoint.pt").write_bytes(f"best:{job.candidate_id}".encode())
            checkpoint = job_dir / "best_checkpoint.pt"
        if status == "completed":
            predictions = job_dir / "predictions.csv"
            predictions.write_text("row_id,target,probability\nr1,0,0.2\nr2,1,0.8\n", encoding="utf-8")
            brier = 0.2
        (job_dir / "job.json").write_text(
            json.dumps(asdict(job), sort_keys=True), encoding="utf-8"
        )
        (job_dir / "worker_result.json").write_text("{}", encoding="utf-8")
        (job_dir / "worker.log").write_text(f"{status}\n", encoding="utf-8")
        return (
            CampaignJobResult(
                job.candidate_id,
                status,
                brier,
                2 if checkpoint else None,
                3 if checkpoint else 0,
                checkpoint,
                predictions,
                {"gpu": 0},
                None if status == "completed" else status,
            ),
        )


class _BudgetRuntime(_Runtime):
    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        completed = super().run_jobs(
            version,
            jobs,
            output_dir,
            gpu_count=gpu_count,
            job_deadline=job_deadline,
        )[0]
        return (
            CampaignJobResult(
                completed.candidate_id,
                "inconclusive",
                completed.brier,
                completed.best_epoch,
                completed.completed_epochs,
                completed.checkpoint,
                completed.predictions_path,
                completed.resource_evidence,
                "budget_reached",
            ),
        )


def _official_data(tmp_path: Path, monkeypatch) -> Path:
    data = tmp_path / "official"
    data.mkdir()
    train = data / "train.csv"
    train.write_text("tiny fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        "experiments.tabm_campaign.row_feature_proxy._official_train_file_sha256",
        lambda path: CONTRACT.official_train_sha256,
    )
    return data


def test_build_proxy_jobs_has_exact_paired_contract_order_and_fields() -> None:
    jobs = build_proxy_jobs(CONTRACT)

    assert [job.candidate_id for job in jobs] == [
        "rfp__baseline__s42",
        "rfp__baseline__s3407",
        *(
            f"rfp__{bundle}__s{seed}"
            for bundle in CONTRACT.feature_bundles
            for seed in CONTRACT.seeds
        ),
    ]
    assert len(jobs) == 14
    for job in jobs:
        assert (
            job.capacity,
            job.k,
            job.width,
            job.blocks,
            job.dropout,
            job.num_embedding,
            job.loss,
            job.scheduler,
            job.learning_rate,
        ) == (
            CONTRACT.baseline.capacity,
            CONTRACT.baseline.k,
            CONTRACT.baseline.width,
            CONTRACT.baseline.blocks,
            CONTRACT.baseline.dropout,
            CONTRACT.baseline.num_embedding,
            CONTRACT.baseline.loss,
            CONTRACT.baseline.scheduler,
            CONTRACT.baseline.learning_rate,
        )
        assert (job.train_end_year, job.valid_year, job.sample_mode) == (2023, 2024, "proxy")
        assert (job.max_epochs, job.min_epochs, job.patience) == (8, 3, 3)


def test_run_is_sequential_guarded_and_publishes_exact_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    runtime = _Runtime()
    output = tmp_path / "out"

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=runtime,
        gpu_count=1,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    assert [call[1][0].candidate_id for call in runtime.calls] == [
        job.candidate_id for job in build_proxy_jobs(CONTRACT)
    ]
    assert all(call[0] == "P" and call[2] == 1 and call[3] == 9_100.0 for call in runtime.calls)
    assert [
        sum(row["status"] == "completed" for row in json.loads(value)["results"])
        for value in runtime.state_snapshots
    ] == list(range(14))
    assert run.completed == tuple(job.candidate_id for job in build_proxy_jobs(CONTRACT))
    assert run.failed == () and run.inconclusive == ()
    assert run.decision.status == "complete"
    assert run.state_path == output / "stage_state.json"
    assert run.state_path.is_file()
    assert run.bundles.review.name == "tabm_row_feature_stage_P_review_bundle.zip"
    assert run.bundles.resume is not None
    assert run.bundles.resume.name == "tabm_row_feature_stage_P_resume_bundle.zip"

    with ZipFile(run.bundles.review) as archive:
        review_names = set(archive.namelist())
    assert review_names == {
        "manifest.json",
        "config/row_feature_proxy_v1.json",
        "metrics/job_results.json",
        "decisions/proxy_decision.json",
        "logs/stage.log",
        "state/stage_state.json",
        *(f"predictions/{job.candidate_id}.csv" for job in build_proxy_jobs(CONTRACT)),
    }
    assert not any(
        forbidden in name.lower()
        for name in review_names
        for forbidden in ("submission", "test_predictions", "evaluation")
    )


def test_inconclusive_stops_session_and_marks_remainder_not_started(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    runtime = _Runtime({jobs[2].candidate_id: "inconclusive"})

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "out",
        runtime=runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    assert [call[1][0].candidate_id for call in runtime.calls] == [
        job.candidate_id for job in jobs[:3]
    ]
    assert run.inconclusive == tuple(job.candidate_id for job in jobs[2:])
    assert run.decision.status == "incomplete"
    with ZipFile(run.bundles.review) as archive:
        assert "decisions/proxy_decision.json" not in archive.namelist()
        state = json.loads(archive.read("state/stage_state.json"))
    assert state["results"][2]["disposition"] == "inconclusive"
    assert all(row["disposition"] == "not_started" for row in state["results"][3:])


def test_budget_inconclusive_keeps_metric_but_bundles_only_training_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "out",
        runtime=_BudgetRuntime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    with ZipFile(run.bundles.resume) as archive:
        names = set(archive.namelist())
        state = json.loads(archive.read("state/stage_state.json"))
    assert state["results"][0]["brier"] == 0.2
    assert f"predictions/{jobs[0].candidate_id}.csv" not in names
    assert f"jobs/{jobs[0].candidate_id}/predictions.csv" not in names
    assert f"jobs/{jobs[0].candidate_id}/checkpoint.pt" in names


def test_same_directory_reuses_completed_and_failed_but_reruns_inconclusive(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    first_runtime = _Runtime(
        {
            jobs[1].candidate_id: "failed",
        }
    )
    output = tmp_path / "out"
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=first_runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    assert first.decision.status == "blocked"

    retry_runtime = _Runtime()
    retry = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=retry_runtime,
        wall_deadline=20_000.0,
        now=lambda: 2_000.0,
    )
    assert retry_runtime.calls == []
    assert retry.completed == (jobs[0].candidate_id,)
    assert retry.failed == (jobs[1].candidate_id,)


def test_resume_reuses_completed_and_chains_manifest(tmp_path: Path, monkeypatch) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    first_runtime = _Runtime({jobs[2].candidate_id: "inconclusive"})
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=first_runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    second_runtime = _Runtime()
    second = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "second",
        resume_bundle=first.bundles.resume,
        runtime=second_runtime,
        wall_deadline=20_000.0,
        now=lambda: 2_000.0,
    )

    assert [call[1][0].candidate_id for call in second_runtime.calls] == [
        job.candidate_id for job in jobs[2:]
    ]
    assert (tmp_path / "second" / "jobs" / jobs[2].candidate_id / "checkpoint.pt").is_file()
    with ZipFile(second.bundles.resume) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["prior_manifest_sha256"] == first.bundles.manifest_sha256


def test_failed_feature_is_terminal_but_independent_later_jobs_continue(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    runtime = _Runtime({jobs[2].candidate_id: "failed"})

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "out",
        runtime=runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    assert len(runtime.calls) == 14
    assert run.failed == (jobs[2].candidate_id,)
    assert run.decision.status == "complete"
    assert run.decision.failed_bundles == ("count_context",)


def test_same_directory_reruns_inconclusive_with_checkpoint_and_keeps_failed_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    output = tmp_path / "out"
    first_runtime = _Runtime(
        {
            jobs[2].candidate_id: "failed",
            jobs[3].candidate_id: "inconclusive",
        }
    )
    run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=first_runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    retry_runtime = _Runtime()
    retry = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=retry_runtime,
        wall_deadline=20_000.0,
        now=lambda: 2_000.0,
    )

    assert [call[1][0].candidate_id for call in retry_runtime.calls] == [
        job.candidate_id for job in jobs[3:]
    ]
    assert retry_runtime.preexisting_checkpoints == [jobs[3].candidate_id]
    assert retry.failed == (jobs[2].candidate_id,)


def test_job_guard_stops_before_starting_a_new_job_and_persists_progress(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    times = iter((1_000.0, 1_000.0, 9_100.0))
    runtime = _Runtime()

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "out",
        runtime=runtime,
        wall_deadline=10_000.0,
        now=lambda: next(times),
    )

    assert [call[1][0].candidate_id for call in runtime.calls] == [jobs[0].candidate_id]
    state = json.loads(run.state_path.read_bytes())
    assert state["results"][0]["disposition"] == "completed"
    assert all(row["disposition"] == "not_started" for row in state["results"][1:])


def test_resume_rejects_changed_code_binding(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=_Runtime({"rfp__count_context__s42": "inconclusive"}),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    monkeypatch.setattr(
        "experiments.tabm_campaign.row_feature_proxy._code_sha256",
        lambda: "f" * 64,
    )
    with pytest.raises(RowFeatureProxyError, match="code_sha256"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "wrong-code",
            resume_bundle=first.bundles.resume,
            runtime=_Runtime(),
            wall_deadline=20_000.0,
            now=lambda: 2_000.0,
        )


def test_local_recovery_rejects_noncanonical_not_started_state(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    output = tmp_path / "out"
    times = iter((1_000.0, 9_100.0))
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=_Runtime(),
        wall_deadline=10_000.0,
        now=lambda: next(times),
    )
    state = json.loads(first.state_path.read_bytes())
    state["results"][0]["completed_epochs"] = "forged"
    first.state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RowFeatureProxyError, match="completed_epochs"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=output,
            runtime=_Runtime(),
            wall_deadline=20_000.0,
            now=lambda: 2_000.0,
        )


def test_rejects_expired_deadline_and_ambiguous_or_symlink_train(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    with pytest.raises(RowFeatureProxyError, match="deadline"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "expired",
            runtime=_Runtime(),
            wall_deadline=1_000.0,
            now=lambda: 1_000.0,
        )

    nested = data / "nested"
    nested.mkdir()
    (nested / "train.csv").write_text("second\n", encoding="utf-8")
    with pytest.raises(RowFeatureProxyError, match="exactly one"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "ambiguous",
            runtime=_Runtime(),
            wall_deadline=10_000.0,
            now=lambda: 1_000.0,
        )

    (nested / "train.csv").unlink()
    (nested / "train.csv").symlink_to(data / "train.csv")
    run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "symlink-ignored",
        runtime=_Runtime({"rfp__baseline__s42": "failed"}),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )


def test_equivalent_runs_produce_identical_bundle_bytes(tmp_path: Path, monkeypatch) -> None:
    data = _official_data(tmp_path, monkeypatch)
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=_Runtime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    second = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "second",
        runtime=_Runtime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    assert sha256(first.bundles.review.read_bytes()).digest() == sha256(
        second.bundles.review.read_bytes()
    ).digest()
    assert sha256(first.bundles.resume.read_bytes()).digest() == sha256(
        second.bundles.resume.read_bytes()
    ).digest()
