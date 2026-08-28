import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from experiments.tree_expert.hetero_kaggle import (
    HeteroKaggleError,
    build_hetero_kaggle_cell,
    discover_hetero_inputs,
    runtime_archive,
    runtime_member_names,
)


def test_runtime_is_importable_in_isolation(tmp_path):
    members = runtime_member_names()
    assert "experiments/tree_expert/hetero_runner.py" in members
    assert "experiments/tree_expert/t3_inputs.py" in members
    with tarfile.open(fileobj=io.BytesIO(runtime_archive()), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    result = subprocess.run(
        [sys.executable, "-c", "import experiments.tree_expert.hetero_kaggle; import experiments.tree_expert.hetero_runner"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_discovery_accepts_unpacked_input_and_one_resume(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"artifact_kind": "tree_expert_t3_input_v1"}))
    assert discover_hetero_inputs(tmp_path).resume is None
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "manifest.json").write_text(json.dumps({"artifact_kind": "tree_hetero_residual_resume_v1"}))
    assert discover_hetero_inputs(tmp_path).resume == resume
    duplicate = tmp_path / "resume2"
    duplicate.mkdir()
    (duplicate / "manifest.json").write_text(json.dumps({"artifact_kind": "tree_hetero_residual_resume_v1"}))
    with pytest.raises(HeteroKaggleError, match="resume count"):
        discover_hetero_inputs(tmp_path)


def test_cell_is_deterministic_compilable_and_below_limit(tmp_path):
    first = build_hetero_kaggle_cell(tmp_path / "first.py")
    second = build_hetero_kaggle_cell(tmp_path / "second.py")
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 1_000_000
    compile(first.read_text(), str(first), "exec")
    text = first.read_text()
    assert "TREE_HETERO_SUCCESS" in text
    assert 'Path("/kaggle/working/tree_hetero_handoff.zip")' in text


def test_committed_cell_matches_renderer(tmp_path):
    rendered = build_hetero_kaggle_cell(tmp_path / "rendered.py")
    committed = Path(__file__).parents[1] / "experiments/tree_expert/KAGGLE_HETERO_CELL.py"
    assert committed.read_bytes() == rendered.read_bytes()
