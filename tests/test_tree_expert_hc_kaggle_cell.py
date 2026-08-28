from pathlib import Path

from experiments.tree_expert.hc_kaggle import build_hc_kaggle_cell


def test_generated_hc_cell_is_deterministic_small_and_offline(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    first = build_hc_kaggle_cell(tmp_path / "first.py", root).read_bytes()
    second = build_hc_kaggle_cell(tmp_path / "second.py", root).read_bytes()
    assert first == second
    assert len(first) < 1_000_000
    text = first.decode()
    for marker in (
        "TREE_HC_CODE_READY",
        "TREE_HC_STAGE_SELECTED",
        "TREE_HC_JOB_START",
        "TREE_HC_TRAINING_PROGRESS",
        "TREE_HC_JOB_END",
        "TREE_HC_DECISION",
        "TREE_HC_ARTIFACT_READY",
        "TREE_HC_SUCCESS",
        "TREE_HC_ERROR",
    ):
        assert marker in text
    assert "github.com" not in text.lower()
    assert "drive.mount" not in text
    assert "files.download" not in text
    assert "submission.zip" not in text
