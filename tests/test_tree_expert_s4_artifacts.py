from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.tree_expert.s4_artifacts import (
    S4ArtifactError,
    S4Bindings,
    create_s4_handoff,
    create_s4_resume,
    verify_s4_resume,
)
from experiments.tree_expert.s4_state import initial_s4_state, save_s4_state


def _bindings():
    return S4Bindings(*("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, "f" * 64))


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "campaign"
    (root / "state").mkdir(parents=True)
    (root / "jobs" / "done").mkdir(parents=True)
    (root / "jobs" / "done" / "result.json").write_text("{}")
    (root / "diagnostics").mkdir()
    (root / "diagnostics" / "summary.json").write_text("{}")
    save_s4_state(initial_s4_state(), root / "state/state.json")
    return root


def test_rejected_handoff_has_no_model_or_submission(tmp_path):
    handoff = create_s4_handoff(_root(tmp_path), tmp_path / "handoff.zip", _bindings())
    with ZipFile(handoff) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    assert {"review.zip", "resume.zip"}.issubset(names)
    assert "model_delivery.zip" not in names
    assert all("submission" not in name for name in names)
    assert manifest["status"] == "review_ready"


def test_binding_change_rejects_resume(tmp_path):
    resume = create_s4_resume(_root(tmp_path), tmp_path / "resume.zip", _bindings())
    with pytest.raises(S4ArtifactError, match="bindings differ"):
        verify_s4_resume(resume, replace(_bindings(), contract_sha256="0" * 64))


def test_handoff_delivery_requires_accepted_token(tmp_path):
    root = _root(tmp_path)
    (root / "models").mkdir()
    (root / "models" / "model.bin").write_bytes(b"model")
    with pytest.raises(S4ArtifactError, match="accepted token"):
        create_s4_handoff(root, tmp_path / "handoff.zip", _bindings(), include_delivery=True)
