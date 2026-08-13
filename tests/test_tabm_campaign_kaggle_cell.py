from __future__ import annotations

import subprocess
import sys
from hashlib import sha256
from pathlib import Path


CELL = Path("experiments/tabm_campaign/KAGGLE_CELL.py")
RENDERER = Path("tools/render_tabm_campaign_kaggle_cell.py")


def test_generated_cell_is_self_contained_and_under_kaggle_limit() -> None:
    assert CELL.is_file()
    assert CELL.stat().st_size < 950_000
    compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec")


def test_generated_cell_requires_only_official_data_and_optional_resume() -> None:
    text = CELL.read_text(encoding="utf-8")
    assert "lg-aimers-9th-data" in text
    assert "tabm_search_stage_" in text
    assert "git clone" not in text
    assert "github.com" not in text
    assert "submit.zip" not in text
    assert "BUNDLE_SUCCESS" in text
    assert "TABM_CAMPAIGN_ERROR" in text


def test_renderer_is_byte_deterministic() -> None:
    subprocess.run([sys.executable, str(RENDERER)], check=True)
    first = sha256(CELL.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(RENDERER)], check=True)
    assert sha256(CELL.read_bytes()).hexdigest() == first
