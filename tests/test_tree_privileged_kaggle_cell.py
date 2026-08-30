from __future__ import annotations

from pathlib import Path

from experiments.tree_privileged.kaggle import build_kaggle_cell, runtime_member_names


def test_generated_cell_is_deterministic_under_one_megabyte_and_has_one_download(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    first = build_kaggle_cell(tmp_path / "first.py", root=root)
    second = build_kaggle_cell(tmp_path / "second.py", root=root)
    assert first.read_bytes() == second.read_bytes()
    source = first.read_text()
    assert len(source.encode()) < 1_000_000
    assert source.count("files.download(") == 1
    assert "TREE_PRIV_CODE_READY" in source
    assert "TREE_PRIV_CAMPAIGN_SUCCESS" in source


def test_generated_cell_installs_dependencies_before_importing_runtime_modules(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = build_kaggle_cell(tmp_path / "cell.py", root=root).read_text()
    dependency_ready = source.index("TREE_PRIV_DEPENDENCIES_READY")
    assert "from experiments." not in source[:dependency_ready]


def test_generated_cell_replaces_partial_runtime_and_isolates_campaign_by_code(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = build_kaggle_cell(tmp_path / "cell.py", root=root).read_text()
    assert "if CODE_ROOT.exists():\n        shutil.rmtree(CODE_ROOT)" in source
    assert 'CAMPAIGN_ROOT = Path("/kaggle/working/tree_privileged") / CODE_SHA256[:12]' in source


def test_runtime_inventory_contains_required_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    members = set(runtime_member_names(root))
    assert "experiments/tree_privileged/runner.py" in members
    assert "experiments/tree_privileged/contract.json" in members
    assert "experiments/temporal_portfolio/lupi_teacher.py" in members
    assert "experiments/tree_expert/e2_inference.py" in members
    assert not any(name.endswith("KAGGLE_CELL.py") for name in members)
