from __future__ import annotations

from pathlib import Path

from experiments.tree_expert.failure_audit_kaggle import build_failure_audit_cell


def test_generated_cell_is_deterministic_small_and_review_only(tmp_path: Path) -> None:
    first = build_failure_audit_cell(tmp_path / "first.py")
    second = build_failure_audit_cell(tmp_path / "second.py")

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 900_000
    compile(first.read_text(encoding="utf-8"), str(first), "exec")
    text = first.read_text(encoding="utf-8")
    assert "FAIL_AUDIT_SUCCESS" in text
    assert "FAIL_AUDIT_ERROR" in text
    assert "FAIL_AUDIT_CUTOFF_START" in text
    assert "sample_submission" not in text.lower()
    assert "submission.zip" not in text.lower()
    assert "nvidia-smi" not in text.lower()
    assert "pip install" not in text.lower()
    assert "memory limit exceeded before audit" in text
    assert "memory limit exceeded after audit" in text


def test_checked_in_cell_matches_generator(tmp_path: Path) -> None:
    generated = build_failure_audit_cell(tmp_path / "generated.py")
    checked_in = (
        Path(__file__).parents[1]
        / "experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py"
    )

    assert checked_in.read_bytes() == generated.read_bytes()
