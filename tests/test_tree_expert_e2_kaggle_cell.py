from pathlib import Path

from experiments.tree_expert.e2_kaggle import build_e2_kaggle_cell


def test_generated_cell_is_small_has_markers_and_no_auto_download(tmp_path: Path) -> None:
    path = build_e2_kaggle_cell(tmp_path / "cell.py")
    source = path.read_text(encoding="utf-8")
    assert path.stat().st_size < 1_000_000
    assert "TREE_E2_HANDOFF_READY" in source
    assert "TREE_EXPERT_ERROR stage=" in source
    assert "materialize_resume_source" in source
    assert "files.download" not in source
    assert "submission.csv" not in source
    compile(source, str(path), "exec")


def test_e2_runbook_names_runtime_and_return_artifact() -> None:
    source = Path("docs/TREE_EXPERT_E2_KAGGLE.md").read_text(encoding="utf-8")
    for required in (
        "45–90",
        "6시간",
        "T4 x2",
        "TREE_E2_HANDOFF_READY",
        "tree_expert_e2_handoff.zip",
        "제출 파일을 만들지 않는다",
    ):
        assert required in source
