from pathlib import Path

from experiments.direct_expert.kaggle import build_kaggle_cell


def test_stage_a_cell_is_deterministic_small_and_has_no_submission(tmp_path: Path) -> None:
    left = build_kaggle_cell("A", tmp_path / "left.py")
    right = build_kaggle_cell("A", tmp_path / "right.py")
    assert left.read_bytes() == right.read_bytes()
    assert left.stat().st_size < 1_000_000
    assert b"submission.zip" not in left.read_bytes()


def test_checked_in_stage_a_cell_matches_renderer(tmp_path: Path) -> None:
    rendered = build_kaggle_cell("A", tmp_path / "cell.py")
    checked = Path("experiments/direct_expert/KAGGLE_STAGE_A_CELL.py")
    assert checked.read_bytes() == rendered.read_bytes()
