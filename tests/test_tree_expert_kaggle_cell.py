from pathlib import Path

from experiments.tree_expert.kaggle import build_e1_kaggle_cell


def test_generated_kaggle_cell_is_small_and_has_one_final_handoff(tmp_path: Path) -> None:
    output = build_e1_kaggle_cell(tmp_path / "cell.py")
    text = output.read_text(encoding="utf-8")

    assert output.stat().st_size < 1_000_000
    assert "TREE_E1_HANDOFF_READY" in text
    assert "tree_expert_e1_handoff.zip" in text
    assert "files.download" not in text


def test_e1_runbook_names_inputs_outputs_runtime_and_return_logs() -> None:
    text = Path("docs/TREE_EXPERT_E1_KAGGLE.md").read_text(encoding="utf-8")
    for required in (
        "lg-aimers-9th-data",
        "tree_expert_e1_input",
        "tree_expert_e1_handoff.zip",
        "3~4시간",
        "재실행",
        "TREE_E1_HANDOFF_READY",
        "TREE_EXPERT_ERROR",
    ):
        assert required in text
