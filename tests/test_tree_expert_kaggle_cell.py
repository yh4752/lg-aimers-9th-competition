from pathlib import Path

from experiments.tree_expert.kaggle import build_e1_kaggle_cell


def test_generated_kaggle_cell_is_small_and_has_one_final_handoff(tmp_path: Path) -> None:
    output = build_e1_kaggle_cell(tmp_path / "cell.py")
    text = output.read_text(encoding="utf-8")

    assert output.stat().st_size < 1_000_000
    assert "TREE_E1_HANDOFF_READY" in text
    assert "tree_expert_e1_handoff.zip" in text
    assert "files.download" not in text
