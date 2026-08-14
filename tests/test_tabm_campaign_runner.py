from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from competition_rules import contract as rules_contract
from experiments.tabm_campaign import worker as worker_module
from experiments.tabm_campaign.contracts import Candidate
from experiments.tabm_campaign.runner import (
    CampaignJobResult,
    VERSION_C_TEMPORAL_SENTINEL_CANDIDATE_ID,
    _version_c_confirmation_jobs,
    run_one_version,
)


@pytest.fixture(autouse=True)
def _isolate_runner_mechanics_from_the_retired_rules_contract(monkeypatch) -> None:
    """These tests cover scheduling; the retired campaign is not runnable now."""

    monkeypatch.setattr(rules_contract, "assert_experiment_runnable", lambda **_: {})


class _Runtime:
    def __init__(self) -> None:
        self.started_versions: list[str] = []
        self.assignments: list[int] = []
        self.shared_model_processes = 0
        self.deadlines: list[float] = []
        self.preexisting_training_snapshots: list[str] = []
        self.started_candidate_ids: list[str] = []

    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        self.started_versions.append(version)
        self.deadlines.append(job_deadline)
        self.started_candidate_ids.extend(job.candidate_id for job in jobs)
        results = []
        for index, job in enumerate(jobs):
            if (output_dir / job.candidate_id / "checkpoint.pt").is_file():
                self.preexisting_training_snapshots.append(job.candidate_id)
            gpu = index % gpu_count
            self.assignments.append(gpu)
            checkpoint = output_dir / job.candidate_id / "best.bin"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(job.candidate_id.encode())
            predictions = checkpoint.parent / "predictions.csv"
            pd.DataFrame(
                {
                    "row_id": ["r1", "r2", "r3", "r4"],
                    "target": [0, 1, 0, 1],
                    "probability": [0.2 + index / 10000, 0.8, 0.3, 0.7],
                    "game_type": ["R", "R", "F", "F"],
                }
            ).to_csv(predictions, index=False)
            results.append(
                CampaignJobResult(
                    candidate_id=job.candidate_id,
                    status="completed",
                    brier=0.20 + index / 10000,
                    best_epoch=2,
                    completed_epochs=3,
                    checkpoint=checkpoint,
                    predictions_path=predictions,
                    resource_evidence={"gpu": gpu},
                    failure=None,
                )
            )
        return tuple(results)


def test_version_c_confirms_one_cycle_sentinel_on_both_temporal_folds() -> None:
    finalist = Candidate(
        "c__p2__piecewise_linear__bce__plateau__lr0p0006__d0p10__s42",
        "tabm",
        "p2",
        32,
        512,
        4,
        0.10,
        "piecewise_linear",
        "bce",
        "plateau",
        0.0006,
        42,
    )
    sentinel = Candidate(
        VERSION_C_TEMPORAL_SENTINEL_CANDIDATE_ID,
        "tabm",
        "p2",
        32,
        512,
        4,
        0.10,
        "piecewise_linear",
        "bce",
        "one_cycle",
        0.0006,
        42,
    )

    jobs = _version_c_confirmation_jobs((finalist,), sentinel)

    assert {job.candidate_id for job in jobs} == {
        f"{finalist.candidate_id}__tr2023__va2024",
        f"{finalist.candidate_id}__tr2022__va2023",
        f"{sentinel.candidate_id}__tr2023__va2024",
        f"{sentinel.candidate_id}__tr2022__va2023",
    }
    assert all(job.sample_mode == "full" and job.max_epochs == 40 for job in jobs)


def test_worker_environment_imports_embedded_runtime_outside_project_cwd(
    tmp_path: Path,
) -> None:
    environment = worker_module._worker_environment(0)

    completed = subprocess.run(
        [sys.executable, "-c", "import experiments.tabm_campaign.worker"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


class _IncompleteThenCompleteRuntime(_Runtime):
    def __init__(self, complete_limit: int | None) -> None:
        super().__init__()
        self.complete_limit = complete_limit

    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        results = list(super().run_jobs(version, jobs, output_dir, gpu_count=gpu_count, job_deadline=job_deadline))
        if self.complete_limit is not None:
            for index in range(self.complete_limit, len(results)):
                results[index] = CampaignJobResult(
                    results[index].candidate_id,
                    "inconclusive",
                    results[index].brier,
                    results[index].best_epoch,
                    results[index].completed_epochs,
                    results[index].checkpoint,
                    None,
                    results[index].resource_evidence,
                    "deadline",
                )
                root = results[index].checkpoint.parent
                (root / "checkpoint.pt").write_bytes(b"resume-state")
                (root / "checkpoint_meta.json").write_text("{}", encoding="utf-8")
                (root / "best_checkpoint.pt").write_bytes(b"best-state")
        return tuple(results)


def test_runner_advances_exactly_one_version(tmp_path: Path) -> None:
    runtime = _Runtime()
    result = run_one_version(tmp_path / "official", tmp_path / "out", runtime=runtime, now=lambda: 1000.0)
    assert result.version == "A"
    assert runtime.started_versions == ["A"]
    assert result.bundles.review.is_file()
    assert result.bundles.resume is not None and result.bundles.resume.is_file()


def test_two_gpus_run_one_independent_job_each(tmp_path: Path) -> None:
    runtime = _Runtime()
    run_one_version(tmp_path / "official", tmp_path / "out", runtime=runtime, now=lambda: 1000.0)
    assert set(runtime.assignments) == {0, 1}
    assert runtime.shared_model_processes == 0


def test_explicit_one_gpu_never_assigns_gpu_one(tmp_path: Path) -> None:
    runtime = _Runtime()

    run_one_version(
        tmp_path / "official",
        tmp_path / "out",
        runtime=runtime,
        gpu_count=1,
        now=lambda: 1000.0,
    )

    assert runtime.assignments
    assert set(runtime.assignments) == {0}


def test_finalization_reserve_is_removed_from_job_deadline(tmp_path: Path) -> None:
    runtime = _Runtime()
    run_one_version(tmp_path / "official", tmp_path / "out", runtime=runtime, now=lambda: 1000.0)
    assert runtime.deadlines == [1000.0 + 7200 - 600]


def test_incomplete_stage_writes_resume_and_restarts_same_version(tmp_path: Path) -> None:
    first = run_one_version(
        tmp_path / "official",
        tmp_path / "first",
        runtime=_IncompleteThenCompleteRuntime(complete_limit=3),
        now=lambda: 1000.0,
    )
    assert first.version == "A"
    assert first.bundles.resume is not None

    runtime = _IncompleteThenCompleteRuntime(complete_limit=None)
    second = run_one_version(
        tmp_path / "official",
        tmp_path / "second",
        resume_bundle=first.bundles.resume,
        runtime=runtime,
        now=lambda: 2000.0,
    )
    assert second.version == "A"
    assert len(runtime.assignments) == 21
    assert len(runtime.preexisting_training_snapshots) == 21


def test_incomplete_version_b_is_bundled_instead_of_crashing(tmp_path: Path) -> None:
    stage_a = run_one_version(
        tmp_path / "official",
        tmp_path / "a",
        runtime=_Runtime(),
        now=lambda: 1000.0,
    )
    stage_b = run_one_version(
        tmp_path / "official",
        tmp_path / "b",
        resume_bundle=stage_a.bundles.resume,
        runtime=_IncompleteThenCompleteRuntime(complete_limit=1),
        now=lambda: 2000.0,
    )
    assert stage_b.version == "B"
    assert stage_b.bundles.resume is not None

    restarted = run_one_version(
        tmp_path / "official",
        tmp_path / "b2",
        resume_bundle=stage_b.bundles.resume,
        runtime=_Runtime(),
        now=lambda: 3000.0,
    )
    assert restarted.version == "B"


def test_incomplete_version_c_restarts_c_not_final_review(tmp_path: Path) -> None:
    stage_a = run_one_version(tmp_path / "official", tmp_path / "a", runtime=_Runtime(), now=lambda: 1000.0)
    stage_b = run_one_version(
        tmp_path / "official", tmp_path / "b", resume_bundle=stage_a.bundles.resume, runtime=_Runtime(), now=lambda: 2000.0
    )
    stage_c = run_one_version(
        tmp_path / "official",
        tmp_path / "c",
        resume_bundle=stage_b.bundles.resume,
        runtime=_IncompleteThenCompleteRuntime(complete_limit=1),
        now=lambda: 3000.0,
    )
    assert stage_c.version == "C"
    restarted = run_one_version(
        tmp_path / "official",
        tmp_path / "c2",
        resume_bundle=stage_c.bundles.resume,
        runtime=_Runtime(),
        now=lambda: 4000.0,
    )
    assert restarted.version == "C"


def test_completed_version_c_records_temporal_one_cycle_sentinel(
    tmp_path: Path,
) -> None:
    stage_a = run_one_version(
        tmp_path / "official",
        tmp_path / "a",
        runtime=_Runtime(),
        now=lambda: 1000.0,
    )
    stage_b = run_one_version(
        tmp_path / "official",
        tmp_path / "b",
        resume_bundle=stage_a.bundles.resume,
        runtime=_Runtime(),
        now=lambda: 2000.0,
    )
    runtime = _Runtime()

    stage_c = run_one_version(
        tmp_path / "official",
        tmp_path / "c",
        resume_bundle=stage_b.bundles.resume,
        runtime=runtime,
        now=lambda: 3000.0,
    )

    assert stage_c.version == "C"
    assert {
        f"{VERSION_C_TEMPORAL_SENTINEL_CANDIDATE_ID}__tr2023__va2024",
        f"{VERSION_C_TEMPORAL_SENTINEL_CANDIDATE_ID}__tr2022__va2023",
    }.issubset(runtime.started_candidate_ids)
    with ZipFile(stage_c.bundles.review) as archive:
        state = json.loads(archive.read("stage_state.json"))
    assert state["stage_complete"] is True
    assert state["temporal_sentinel"]["candidate"]["candidate_id"] == (
        VERSION_C_TEMPORAL_SENTINEL_CANDIDATE_ID
    )
    assert set(state["temporal_sentinel"]["fold_briers"]) == {"2024", "2023"}
