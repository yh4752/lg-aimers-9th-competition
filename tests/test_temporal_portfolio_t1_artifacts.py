from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from experiments.temporal_portfolio.t1_artifacts import (
    restore_t1_resume,
    restore_t1_resume_source,
    verify_compact_result,
    write_t1_bundles,
)


def _completed(root: Path, job_id: str, identity: str) -> dict[str, object]:
    root.mkdir(parents=True)
    (root / "predictions.csv").write_text(
        "row_id,target,probability\nr1,1,0.7\n", encoding="utf-8"
    )
    (root / "metrics.json").write_text("{}", encoding="utf-8")
    (root / "checkpoint_meta.json").write_text("{}", encoding="utf-8")
    return {
        "job_id": job_id,
        "status": "completed",
        "training_identity_sha256": identity,
    }


def test_t1_resume_keeps_compact_completed_evidence_and_only_partial_checkpoint(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    completed_payload = _completed(jobs / "done", "done", "a" * 64)
    partial = jobs / "partial"
    partial.mkdir(parents=True)
    (partial / "checkpoint.pt").write_bytes(b"restart")
    (partial / "checkpoint_meta.json").write_text("{}", encoding="utf-8")
    (partial / "progress.jsonl").write_text("{}\n", encoding="utf-8")

    bundles = write_t1_bundles(
        tmp_path / "bundles",
        jobs_root=jobs,
        completed=("done",),
        pending=("partial",),
        failed=(),
        verifier=lambda _path: completed_payload,
    )

    with ZipFile(bundles.resume) as archive:
        names = set(archive.namelist())
    assert "jobs/done/predictions.csv" in names
    assert "jobs/done/checkpoint.pt" not in names
    assert "jobs/partial/checkpoint.pt" in names
    assert "jobs/partial/progress.jsonl" in names
    assert "jobs/partial/predictions.csv" not in names

    restored = restore_t1_resume(bundles.resume, tmp_path / "restored")
    payload = verify_compact_result(restored / "jobs" / "done")
    assert payload["training_identity_sha256"] == "a" * 64
    assert (restored / "jobs" / "partial" / "checkpoint.pt").read_bytes() == b"restart"

    extracted = tmp_path / "extracted"
    with ZipFile(bundles.resume) as archive:
        archive.extractall(extracted)
    restored_directory = restore_t1_resume_source(
        extracted, tmp_path / "restored_directory"
    )
    assert verify_compact_result(restored_directory / "jobs" / "done")["job_id"] == "done"


def test_t1_review_manifest_records_terminal_counts(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    payload = _completed(jobs / "done", "done", "b" * 64)
    bundles = write_t1_bundles(
        tmp_path / "bundles",
        jobs_root=jobs,
        completed=("done",),
        pending=("waiting",),
        failed=("broken",),
        verifier=lambda _path: payload,
    )
    with ZipFile(bundles.review) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["artifact_kind"] == "temporal_t1_review_v1"
    assert manifest["completed"] == ["done"]
    assert manifest["pending"] == ["waiting"]
    assert manifest["failed"] == ["broken"]
