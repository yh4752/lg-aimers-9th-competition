from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "experiments/preprocessing_campaign/requirements-colab.txt"
CONFIG = ROOT / "experiments/independent_dl/configs/preprocessing_ablation_v1.json"


def test_cli_exposes_run_promote_status_and_summarize() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.preprocessing_campaign.run_campaign", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert all(
        name in completed.stdout for name in ("run", "promote", "status", "summarize")
    )


def test_status_dry_contract_reports_sealed_counts() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.preprocessing_campaign.run_campaign",
            "status",
            "--config",
            str(CONFIG),
            "--dry-contract",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"wave_a_jobs": 760' in completed.stdout
    assert '"catboost_wave_e_jobs": 1020' in completed.stdout


def test_colab_handoff_is_one_complete_python_block() -> None:
    text = (ROOT / "experiments/preprocessing_campaign/COLAB.md").read_text(
        encoding="utf-8"
    )
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)

    assert len(blocks) == 1
    compile(blocks[0], "<preprocessing-colab>", "exec")
    assert "PREPROCESSING_CAMPAIGN_CHECKPOINTED" in blocks[0]
    assert "PREPROCESSING_CAMPAIGN_ERROR" in blocks[0]
    assert "GITHUB_TOKEN" in blocks[0]
    assert "submission" not in blocks[0].lower()


def test_requirements_pin_catboost_and_keep_existing_dl_runtime() -> None:
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()

    assert "catboost==1.2.10" in lines
    assert "tabm==0.0.3" in lines
    assert "scikit-learn==1.8.0" in lines
