from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "experiments/independent_dl/COLAB.md"
REQUIREMENTS = ROOT / "experiments/independent_dl/requirements-colab.txt"


def test_colab_handoff_is_one_cell_and_does_not_package_submission() -> None:
    text = HANDOFF.read_text(encoding="utf-8")

    assert text.count("```python") == 1
    assert text.count("```") == 2
    for phrase in (
        "목적",
        "필수 입력",
        "예상 시간",
        "재실행",
        "성공 시",
        "오류 시",
        "T4",
        "INDEPENDENT_DL_CAMPAIGN_CHECKPOINTED",
        "campaign_manifest.json",
        "campaign_summary.json",
    ):
        assert phrase in text
    for forbidden in (
        "submit.zip",
        "submission.zip",
        "build-submission",
        "git push",
        "colab_pipeline.ipynb",
    ):
        assert forbidden not in text


def test_runtime_requirements_pin_only_the_approved_packages() -> None:
    lines = [
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert lines == [
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.16.3",
        "scikit-learn==1.8.0",
        "tabm==0.0.3",
        "rtdl-revisiting-models==0.0.2",
        "rtdl-num-embeddings==0.0.12",
    ]
    assert all(not line.startswith("torch") for line in lines)


def test_handoff_uses_child_runtime_and_existing_colab_secret() -> None:
    text = HANDOFF.read_text(encoding="utf-8")

    assert 'userdata.get("GITHUB_TOKEN")' in text
    assert 'REQUIRED_CODE_COMMIT = "c36811956e632a00f775525f83bcb666ff2ec9d6"' in text
    assert 'PYTHONPATH' in text
    assert 'subprocess.Popen' in text
    assert 'pip", "install", "--target"' in text
    assert '"run",' in text
    assert 'force_remount=True' not in text


def test_roadmap_marks_dl_code_ready_and_waiting_for_user_run() -> None:
    text = (ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8")

    assert "독립 DL 캠페인 코드: `code_ready`" in text
    assert "공식 데이터 T4 실행: `waiting_for_user_run`" in text
    assert "전체 데이터 실행 결과는 아직 없음" in text
