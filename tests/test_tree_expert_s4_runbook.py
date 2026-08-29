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


def test_recovery_runbook_explains_one_cell_rerun_and_return_artifact():
    text = Path("docs/TREE_S4_RECOVERY.md").read_text()
    for phrase in (
        "prepare_tree_s4_recovery_input.py",
        "10 GB",
        "T4 x2",
        "Save Version",
        "S4_RECOVERY_READY",
        "S4_DISK_STATUS",
        "full_chains__14",
        "anchor_residual_hierarchical_handoff.zip",
    ):
        assert phrase in text
    assert "DACON 제출 ZIP을 만들지 않습니다" in text
    assert "/Users/" not in text
