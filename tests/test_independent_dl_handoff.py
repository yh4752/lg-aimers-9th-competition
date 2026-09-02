from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "experiments/independent_dl/COLAB.md"
REQUIREMENTS = ROOT / "experiments/independent_dl/requirements-colab.txt"


def test_colab_handoff_explains_split_family_execution_and_does_not_package() -> None:
    text = HANDOFF.read_text(encoding="utf-8")

    assert text.count("```python") >= 8
    for phrase in (
        "목적",
        "필수 입력",
        "예상 시간",
        "재실행",
        "정상 완료",
        "오류 전달",
        "VRAM",
        "다음 후보",
        "epoch checkpoint",
        "후보 하나",
        "--max-candidates",
        "P3",
        "P4",
        "boundary expansion",
        "confirmation",
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
        "tabicl==2.1.1",
        "faiss-gpu-cu12==1.14.1.post1",
    ]
    assert all(not line.startswith("torch") for line in lines)


def test_handoff_uses_child_runtime_and_existing_colab_secret() -> None:
    text = HANDOFF.read_text(encoding="utf-8")

    assert 'userdata.get("GITHUB_TOKEN")' in text
    assert 'REQUIRED_CODE_COMMIT = "a8f0525' in text
    assert 'PYTHONPATH' in text
    assert 'subprocess.Popen' in text
    assert 'pip", "install", "--target"' in text
    for family in ("tabm", "mlp_resnet", "ft_transformer", "tabr", "tabicl_v2"):
        assert f'run_one_family("{family}")' in text
    assert 'force_remount=True' not in text
    assert "assert 'T4' in name" not in text
    assert 'torch.cuda.device_count()' in text
    assert 'total_memory' in text
    assert "import tabicl" in text


def test_roadmap_records_completed_dl_evidence_and_next_research_boundary() -> None:
    text = (ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8")

    assert "규칙 준수 TabM 단일 모델: Public `872.3920184667`" in text
    assert "TabM은 유효한 독립 축" in text
    assert "더 긴 학습과 단순 seed 평균은 이미 이득이" in text
    assert "TabICLv2" in text
    assert "대회 사용 가능성" in text
