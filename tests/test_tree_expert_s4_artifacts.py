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
from experiments.tree_expert.s4_state import S4State, initial_s4_state, save_s4_state
from types import MappingProxyType


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


def test_resume_excludes_oof_models_and_reproducible_e2(tmp_path):
    root = _root(tmp_path)
    save_s4_state(
        S4State("full_chains", ("full_chains__00",), (), MappingProxyType({})),
        root / "state/state.json",
    )
    (root / "jobs/full_chains__00").mkdir(parents=True)
    (root / "jobs/full_chains__00/model.cbm").write_bytes(b"large-model")
    (root / "jobs/full_chains__00/result.json").write_text("{}")
    (root / "verified_e2").mkdir()
    (root / "verified_e2/fold.csv").write_bytes(b"reproducible")
    (root / "verified_e2_input.zip").write_bytes(b"reproducible")
    resume = create_s4_resume(root, tmp_path / "resume.zip", _bindings())
    with ZipFile(resume) as archive:
        names = set(archive.namelist())
    assert "jobs/full_chains__00/result.json" in names
    assert "jobs/full_chains__00/model.cbm" not in names
    assert not any(name.startswith("verified_e2/") for name in names)
    assert "verified_e2_input.zip" not in names


def test_review_keeps_decisions_and_confirmation_but_drops_raw_search_tables(tmp_path):
    root = _root(tmp_path)
    for name in (
        "decisions/acceptance_c02.json",
        "full_chains/c02/config.json",
        "full_chains/c02/2024.csv",
        "residual_predictions/c02/s3407/2024.csv",
        "confirmation/c02/2024.csv",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if name.endswith(".json") else "row_id,p\n1,0.5\n")
    handoff = create_s4_handoff(root, tmp_path / "handoff.zip", _bindings())
    with ZipFile(handoff) as outer:
        review_bytes = outer.read("review.zip")
    review = tmp_path / "review.zip"
    review.write_bytes(review_bytes)
    with ZipFile(review) as archive:
        names = set(archive.namelist())
    assert "decisions/acceptance_c02.json" in names
    assert "full_chains/c02/config.json" in names
    assert "confirmation/c02/2024.csv" in names
    assert "full_chains/c02/2024.csv" not in names
    assert not any(name.startswith("residual_predictions/") for name in names)
