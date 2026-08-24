from __future__ import annotations

from pathlib import Path

from experiments.temporal_portfolio.platform import build_kaggle_cell


def test_generated_kaggle_cell_is_under_limit_offline_and_compilable(
    tmp_path: Path,
) -> None:
    output = build_kaggle_cell(tmp_path / "KAGGLE_CELL.py")
    source = output.read_text(encoding="utf-8")

    assert output.stat().st_size < 1_000_000
    assert "github.com" not in source
    assert "requests.get" not in source
    assert "git clone" not in source
    assert "T1_HANDOFF_READY" in source
    assert "T1_PORTFOLIO_ERROR" in source
    compile(source, str(output), "exec")


def test_generated_kaggle_cell_is_deterministic(tmp_path: Path) -> None:
    first = build_kaggle_cell(tmp_path / "first.py")
    second = build_kaggle_cell(tmp_path / "second.py")
    assert first.read_bytes() == second.read_bytes()
