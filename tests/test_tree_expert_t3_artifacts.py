from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.tree_expert.t3_artifacts import (
    T3ArtifactError,
    T3Bindings,
    create_resume_bundle,
    publish_stable_resume,
    restore_resume_bundle,
    verify_resume_bundle,
)


def bindings():
    return T3Bindings(*("1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64, "6" * 64))


def campaign(tmp_path):
    root = tmp_path / "campaign"
    (root / "state").mkdir(parents=True)
    (root / "jobs/job_a").mkdir(parents=True)
    (root / "state/stage_state.json").write_text(
        '{"completed_jobs":["job_a"],"failed_jobs":[]}'
    )
    (root / "jobs/job_a/metrics.json").write_text('{"status":"completed"}')
    (root / "jobs/job_a/checkpoint.cbm").write_bytes(b"model")
    return root


def test_resume_round_trip_preserves_completed_jobs(tmp_path):
    bundle = create_resume_bundle(campaign(tmp_path), tmp_path / "resume.zip", bindings())
    restored = restore_resume_bundle(bundle, tmp_path / "restored", bindings())
    assert json.loads((restored / "state/stage_state.json").read_text())["completed_jobs"] == ["job_a"]


def test_resume_accepts_only_an_explicit_compatible_code_binding(tmp_path):
    legacy = replace(bindings(), code_sha256="7" * 64)
    bundle = create_resume_bundle(campaign(tmp_path), tmp_path / "resume.zip", legacy)
    with pytest.raises(T3ArtifactError, match="bindings differ"):
        verify_resume_bundle(bundle, bindings())
    verify_resume_bundle(
        bundle,
        bindings(),
        compatible_code_sha256s=frozenset({legacy.code_sha256}),
    )
    restored = restore_resume_bundle(
        bundle,
        tmp_path / "restored",
        bindings(),
        compatible_code_sha256s=frozenset({legacy.code_sha256}),
    )
    assert (restored / "jobs/job_a/checkpoint.cbm").read_bytes() == b"model"


def test_snapshot_directory_contains_only_latest_verified_resume(tmp_path):
    root = campaign(tmp_path)
    snapshot_root = tmp_path / "snapshots"
    first = publish_stable_resume(root, snapshot_root, bindings())
    first_hash = first.read_bytes()
    (root / "state/stage_state.json").write_text(
        '{"completed_jobs":["job_a","job_b"],"failed_jobs":[]}'
    )
    second = publish_stable_resume(root, snapshot_root, bindings())
    assert first == second
    assert first_hash != second.read_bytes()
    assert [path.name for path in snapshot_root.glob("*.zip")] == ["tree_expert_t3_resume.zip"]


def test_resume_excludes_files_from_an_active_job(tmp_path):
    root = campaign(tmp_path)
    active = root / "jobs/job_b"
    active.mkdir(parents=True)
    (active / ".predictions.csv.tmp").write_text("in progress")
    (active / "checkpoint.cbm").write_bytes(b"partial")
    bundle = create_resume_bundle(root, tmp_path / "resume.zip", bindings())
    with ZipFile(bundle) as archive:
        assert not any(name.startswith("jobs/job_b/") for name in archive.namelist())


def test_resume_rejects_changed_member_and_wrong_binding(tmp_path):
    bundle = create_resume_bundle(campaign(tmp_path), tmp_path / "resume.zip", bindings())
    with pytest.raises(T3ArtifactError, match="bindings differ"):
        verify_resume_bundle(bundle, replace(bindings(), e2_handoff_sha256="0" * 64))
    changed = tmp_path / "changed.zip"
    with ZipFile(bundle) as source, ZipFile(changed, "w") as target:
        for info in source.infolist():
            payload = source.read(info)
            if info.filename == "state/stage_state.json":
                payload = b"{}"
            clone = ZipInfo(info.filename, (2026, 1, 1, 0, 0, 0))
            clone.compress_type = ZIP_DEFLATED
            target.writestr(clone, payload)
    with pytest.raises(T3ArtifactError, match="member SHA-256 differs"):
        verify_resume_bundle(changed, bindings())


def test_resume_rejects_parent_path(tmp_path):
    unsafe = tmp_path / "unsafe.zip"
    with ZipFile(unsafe, "w") as archive:
        archive.writestr("../state.json", b"{}")
    with pytest.raises(T3ArtifactError, match="unsafe"):
        verify_resume_bundle(unsafe, bindings())
