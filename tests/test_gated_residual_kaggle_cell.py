from __future__ import annotations

import base64
import gzip
import io
from pathlib import Path
import subprocess
import sys
import tarfile

from tools.build_gated_residual_final_kaggle_cell import build_cell


CELL = Path("experiments/gated_residual_final/KAGGLE_CELL.py")


def _payload(source: str) -> bytes:
    marker = 'RUNTIME_B64 = "'
    start = source.index(marker) + len(marker)
    end = source.index('"', start)
    return base64.b64decode(source[start:end])


def test_checked_in_cell_matches_generator(tmp_path: Path) -> None:
    generated = build_cell(tmp_path / "cell.py")

    assert generated.read_bytes() == CELL.read_bytes()


def test_cell_is_below_kaggle_source_limit_and_has_terminal_markers() -> None:
    source = CELL.read_text()

    assert CELL.stat().st_size < 1_000_000
    assert "FINAL_CANDIDATE_REVIEW_READY" in source
    assert "FINAL_CANDIDATE_HANDOFF_READY" in source
    assert "FINAL_CANDIDATE_DELIVERY_READY" in source
    assert "github.com" not in source


def test_embedded_runtime_imports_with_isolated_python(tmp_path: Path) -> None:
    payload = _payload(CELL.read_text())
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r:") as archive:
            archive.extractall(tmp_path)
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import experiments.gated_residual_final.kaggle; "
        "import experiments.gated_residual_final.runtime; print('IMPORT_OK')"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "IMPORT_OK"


def test_embedded_runtime_contains_every_declared_member() -> None:
    payload = _payload(CELL.read_text())
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r:") as archive:
            names = set(archive.getnames())

    assert "experiments/gated_residual_final/runtime.py" in names
    assert "experiments/direct_expert/kaggle.py" in names
    assert "experiments/tree_expert/features.py" in names
