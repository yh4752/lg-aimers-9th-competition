from pathlib import Path


def test_runbook_names_inputs_time_and_return_files() -> None:
    text = Path("docs/DIRECT_EXPERT_KAGGLE.md").read_text(encoding="utf-8")
    for required in (
        "10~11시간",
        "8~9시간",
        "T4 x2",
        "direct_expert_stage_A_handoff.zip",
        "direct_expert_review.zip",
    ):
        assert required in text
