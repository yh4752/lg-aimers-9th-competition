from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.tree_expert.hc_artifacts import (
    HCArtifactError,
    create_handoff,
    create_model_delivery,
    create_resume_bundle,
    restore_resume_bundle,
    verify_handoff,
    verify_resume_bundle,
)
from experiments.tree_expert.hc_state import (
    HCBindings,
    complete_job,
    initial_state,
    record_decision,
    start_job,
)


def _bindings():
    return HCBindings(*tuple(str(index) * 64 for index in range(1, 7)))


def _campaign(tmp_path: Path):
    root = tmp_path / "campaign"
    job = root / "jobs/job_a"
    job.mkdir(parents=True)
    (job / "model.cbm").write_bytes(b"model")
    (job / "metrics.json").write_text('{"status":"completed"}')
    state = complete_job(start_job(initial_state(_bindings()), "job_a"), "job_a")
    return root, state


def test_resume_is_deterministic_and_contains_only_completed_jobs(tmp_path: Path):
    root, state = _campaign(tmp_path)
    active = root / "jobs/job_b"
    active.mkdir(parents=True)
    (active / "partial.tmp").write_bytes(b"partial")
    first = create_resume_bundle(root, tmp_path / "first.zip", _bindings(), state)
    second = create_resume_bundle(root, tmp_path / "second.zip", _bindings(), state)
    assert first.read_bytes() == second.read_bytes()
    verified = verify_resume_bundle(first, _bindings())
    assert verified.status == "running"
    with ZipFile(first) as archive:
        assert any(name.startswith("jobs/job_a/") for name in archive.namelist())
        assert not any(name.startswith("jobs/job_b/") for name in archive.namelist())


def test_model_delivery_requires_accepted_hash_bound_evidence(tmp_path: Path):
    _, state = _campaign(tmp_path)
    source = tmp_path / "model.cbm"
    source.write_bytes(b"model")
    with pytest.raises(HCArtifactError, match="accepted evidence"):
        create_model_delivery(
            state=state,
            sources={"models/c1_seed_3407.cbm": source},
            inference_audit={"status": "passed", "max_probability_error": 0.0},
            output=tmp_path / "delivery.zip",
        )


def test_handoff_binds_review_resume_and_optional_delivery(tmp_path: Path):
    root, state = _campaign(tmp_path)
    state = record_decision(state, "winner", {"status": "accepted", "candidate": "C1"})
    resume = create_resume_bundle(root, tmp_path / "resume.zip", _bindings(), state)
    review = tmp_path / "review.zip"
    review.write_bytes(b"review")
    log = tmp_path / "campaign.log"
    log.write_text("ok\n")
    handoff = create_handoff(
        review=review,
        resume=resume,
        log=log,
        output=tmp_path / "tree_hierarchical_handoff.zip",
        status="accepted",
    )
    verified = verify_handoff(handoff)
    assert verified.status == "accepted"
    assert verified.delivery is False


def test_resume_rejects_binding_change(tmp_path: Path):
    root, state = _campaign(tmp_path)
    resume = create_resume_bundle(root, tmp_path / "resume.zip", _bindings(), state)
    changed = HCBindings("0" * 64, *tuple(str(index) * 64 for index in range(2, 7)))
    with pytest.raises(HCArtifactError, match="bindings differ"):
        verify_resume_bundle(resume, changed)


def test_resume_restores_completed_jobs_and_shared_evidence(tmp_path: Path):
    root, state = _campaign(tmp_path)
    evidence = root / "evidence/source_e2_2021.csv"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("row_id,probability\nr1,0.5\n")
    decision = root / "decisions/profile.json"
    decision.parent.mkdir(parents=True)
    decision.write_text('{"profile":"hc_balanced"}')
    resume = create_resume_bundle(root, tmp_path / "resume.zip", _bindings(), state)
    restored = restore_resume_bundle(resume, tmp_path / "restored", _bindings())
    assert (restored / "jobs/job_a/model.cbm").read_bytes() == b"model"
    assert (restored / "evidence/source_e2_2021.csv").is_file()
    assert (restored / "decisions/profile.json").is_file()
