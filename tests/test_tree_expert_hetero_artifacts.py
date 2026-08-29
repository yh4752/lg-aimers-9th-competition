from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.tree_expert.hetero_artifacts import (
    HeteroArtifactError,
    HeteroBindings,
    create_handoff_bundle,
    create_resume_bundle,
    create_review_bundle,
    verify_handoff_bundle,
    verify_resume_bundle,
)


def _bindings():
    return HeteroBindings(*("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, "f" * 64))


def _campaign(root: Path):
    (root / "state").mkdir(parents=True)
    (root / "state/state.json").write_text(json.dumps({
        "phase": "completed", "completed_jobs": ["job"], "failed_jobs": [], "decisions": {},
    }))
    (root / "jobs/job").mkdir(parents=True)
    (root / "jobs/job/model.json").write_text("model")
    (root / "jobs/job/predictions.csv").write_text("row_id,target\na,1\n")


def test_bundles_are_deterministic_review_only_and_bound(tmp_path):
    root = tmp_path / "campaign"
    _campaign(root)
    resume1 = create_resume_bundle(root, tmp_path / "r1.zip", _bindings())
    resume2 = create_resume_bundle(root, tmp_path / "r2.zip", _bindings())
    assert resume1.read_bytes() == resume2.read_bytes()
    verify_resume_bundle(resume1, _bindings())
    with pytest.raises(HeteroArtifactError, match="bindings"):
        verify_resume_bundle(resume1, replace(_bindings(), code_sha256="0" * 64))
    verify_resume_bundle(
        resume1,
        replace(_bindings(), code_sha256="0" * 64),
        compatible_code_sha256s=frozenset({_bindings().code_sha256}),
    )
    review = create_review_bundle(root, tmp_path / "review.zip", _bindings())
    with ZipFile(review) as archive:
        assert not any(name.endswith((".json", ".txt")) and name.startswith("jobs/job/model") for name in archive.namelist())
    handoff = create_handoff_bundle(review, resume1, tmp_path / "handoff.zip", _bindings())
    manifest = verify_handoff_bundle(handoff, _bindings())
    assert manifest["delivery"] is False
    assert manifest["review_only"] is True
    assert manifest["submission_package"] is False
