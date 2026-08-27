import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.tree_expert.t3_kaggle import (
    T3KaggleError,
    build_t3_kaggle_cell,
    discover_t3_inputs,
    runtime_member_names,
)


def test_t3_runtime_archive_contains_every_direct_dependency():
    members = runtime_member_names()
    assert "experiments/tree_expert/t3_runner.py" in members
    assert "experiments/tree_expert/t3_state.py" in members
    assert "experiments/tree_expert/features.py" in members
    assert "experiments/temporal_portfolio/seasonal_features.py" in members
    assert "experiments/independent_dl/feature_sources/trackman.py" in members


def test_discovery_accepts_unpacked_input_and_at_most_one_resume(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "manifest.json").write_text(json.dumps({"artifact_kind": "tree_expert_t3_input_v1"}))
    found = discover_t3_inputs(tmp_path)
    assert found.t3_input == input_root
    assert found.resume is None
    resume_root = tmp_path / "resume"
    resume_root.mkdir()
    (resume_root / "manifest.json").write_text(json.dumps({"artifact_kind": "tree_expert_t3_resume_v1"}))
    assert discover_t3_inputs(tmp_path).resume == resume_root
    second = tmp_path / "resume2"
    second.mkdir()
    (second / "manifest.json").write_text(json.dumps({"artifact_kind": "tree_expert_t3_resume_v1"}))
    with pytest.raises(T3KaggleError, match="resume count"):
        discover_t3_inputs(tmp_path)


def test_generated_cell_is_deterministic_and_below_kaggle_limit(tmp_path):
    first = build_t3_kaggle_cell(tmp_path / "first.py")
    second = build_t3_kaggle_cell(tmp_path / "second.py")
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 1_000_000
    compile(first.read_text(), str(first), "exec")
    text = first.read_text()
    assert "TREE_T3_SUCCESS" in text
    assert "TREE_T3_ERROR" in text
