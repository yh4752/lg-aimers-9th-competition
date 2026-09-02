from __future__ import annotations

import ast
import io
from pathlib import Path
import subprocess
import sys
import tarfile

from experiments.failure_regime_e3.kaggle import _runtime_archive, build_kaggle_cell
from experiments.failure_regime_e3.runtime_inventory import runtime_members


def test_embedded_runtime_imports_in_isolated_python(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(_runtime_archive(Path.cwd())), mode="r:gz") as archive:
        archive.extractall(runtime_root)
    modules = tuple(
        name[:-3].replace("/", ".")
        for name in runtime_members(Path.cwd())
        if name.endswith(".py") and not name.endswith("__init__.py")
    )
    script = (
        f"import importlib,sys;sys.path.insert(0,{str(runtime_root)!r});"
        f"mods={modules!r};[importlib.import_module(name) for name in mods]"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_cell_is_deterministic_parseable_small_and_has_no_delivery(tmp_path: Path) -> None:
    left = build_kaggle_cell(tmp_path / "left.py")
    right = build_kaggle_cell(tmp_path / "right.py")
    assert left.read_bytes() == right.read_bytes()
    ast.parse(left.read_text(encoding="utf-8"))
    assert left.stat().st_size < 1_000_000
    source = left.read_text(encoding="utf-8")
    assert "E3_GPU_SMOKE_SUCCESS" in source
    assert "E3_PREFLIGHT_SUCCESS" in source
    assert "E3_ERROR" in source
    assert "E3_REVIEW_READY" in source and "E3_HANDOFF_READY" in source
    assert "delivery" not in source.lower() and "submission.zip" not in source


def test_checked_in_cell_and_runtime_inventory_are_exact(tmp_path: Path) -> None:
    rendered = build_kaggle_cell(tmp_path / "cell.py")
    checked = Path("experiments/failure_regime_e3/KAGGLE_CELL.py")
    assert checked.read_bytes() == rendered.read_bytes()
    members = runtime_members(Path.cwd())
    assert "experiments/failure_regime_e3/runtime.py" in members
    assert "experiments/direct_expert/features.py" in members
    assert "competition_rules/code_gate.py" in members
