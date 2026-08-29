from pathlib import Path


def test_runbook_names_exact_inputs_runtime_and_return_file():
    text = Path("docs/TREE_S4_KAGGLE.md").read_text()
    for phrase in (
        "T4 x2", "Save Version", "10~12시간", "KAGGLE_S4_CELL.py",
        "anchor_residual_hierarchical_handoff.zip", "S4_SUCCESS",
    ):
        assert phrase in text


def test_runbook_does_not_claim_submission_is_created():
    text = Path("docs/TREE_S4_KAGGLE.md").read_text()
    assert "DACON 제출 ZIP을 만들지 않습니다" in text
    assert "/Users/" not in text
