from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from tools import render_tabm_campaign_kaggle_cell as renderer


CELL = Path("experiments/tabm_campaign/KAGGLE_CELL.py")


def test_generated_cell_is_self_contained_and_under_kaggle_limit() -> None:
    assert CELL.is_file()
    assert CELL.stat().st_size < 950_000
    compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec")


def test_generated_cell_requires_only_official_data_and_optional_resume() -> None:
    text = CELL.read_text(encoding="utf-8")
    assert "lg-aimers-9th-data" in text
    assert 'WORK_ROOT / "normalized_resume"' in text
    assert "git clone" not in text
    assert "github.com" not in text
    assert "submit.zip" not in text
    assert "BUNDLE_SUCCESS" in text
    assert "TABM_CAMPAIGN_ERROR" in text
    assert "normalize_resume_input" in text
    assert '"RESUME_FOUND "' in text
    assert "normalized_resume.source" in text


def test_renderer_is_byte_deterministic() -> None:
    assert renderer.render() == renderer.render()


def test_embedded_runtime_imports_training_from_an_isolated_directory(
    tmp_path: Path,
) -> None:
    with tarfile.open(fileobj=io.BytesIO(renderer._archive_bytes()), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experiments.independent_dl.training; import experiments.tabm_campaign.worker",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
