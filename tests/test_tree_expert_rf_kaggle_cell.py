from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from experiments.tree_expert.rf_kaggle import build_rf_kaggle_cell


def test_generated_rf_cell_is_deterministic_and_below_kernel_limit(tmp_path: Path) -> None:
    first = build_rf_kaggle_cell(tmp_path / "first.py")
    second = build_rf_kaggle_cell(tmp_path / "second.py")

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 900_000
    compile(first.read_text(encoding="utf-8"), str(first), "exec")
    assert len(sha256(first.read_bytes()).hexdigest()) == 64


def test_generated_rf_cell_exposes_required_operator_markers(tmp_path: Path) -> None:
    cell = build_rf_kaggle_cell(tmp_path / "cell.py").read_text(encoding="utf-8")

    for marker in (
        "TREE_RF_CODE_READY",
        "TREE_RF_DEPENDENCIES_READY",
        "TREE_RF_GPU_READY",
        "TREE_RF_INPUTS_VERIFIED",
        "TREE_RF_CAMPAIGN_SUCCESS",
        "TREE_RF_ERROR",
    ):
        assert marker in cell
    assert "submission.zip" not in cell
    assert "time.monotonic() + 19_800" in cell


def test_checked_in_rf_cell_matches_generator(tmp_path: Path) -> None:
    generated = build_rf_kaggle_cell(tmp_path / "generated.py")
    checked_in = Path(__file__).parents[1] / "experiments/tree_expert/KAGGLE_RF_CELL.py"

    assert checked_in.read_bytes() == generated.read_bytes()
