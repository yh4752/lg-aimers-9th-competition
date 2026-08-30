import io
import subprocess
import sys
import tarfile
from pathlib import Path

from experiments.direct_expert.kaggle import _runtime_archive, build_kaggle_cell


def test_embedded_runtime_imports_in_an_isolated_python(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(_runtime_archive(Path.cwd())), mode="r:gz") as archive:
        archive.extractall(runtime_root)

    script = (
        "import sys; "
        f"sys.path.insert(0, {str(runtime_root)!r}); "
        "import experiments.direct_expert.kaggle; "
        "import experiments.direct_expert.stage_b_runtime"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_stage_a_cell_is_deterministic_small_and_has_no_submission(tmp_path: Path) -> None:
    left = build_kaggle_cell("A", tmp_path / "left.py")
    right = build_kaggle_cell("A", tmp_path / "right.py")
    assert left.read_bytes() == right.read_bytes()
    assert left.stat().st_size < 1_000_000
    assert b"submission.zip" not in left.read_bytes()


def test_checked_in_stage_a_cell_matches_renderer(tmp_path: Path) -> None:
    rendered = build_kaggle_cell("A", tmp_path / "cell.py")
    checked = Path("experiments/direct_expert/KAGGLE_STAGE_A_CELL.py")
    assert checked.read_bytes() == rendered.read_bytes()


def test_stage_b_cell_prints_all_terminal_artifacts(tmp_path: Path) -> None:
    cell = build_kaggle_cell("B", tmp_path / "cell.py").read_text()
    assert "DIRECT_EXPERT_REVIEW_READY" in cell
    assert "DIRECT_EXPERT_HANDOFF_READY" in cell
    assert "DIRECT_EXPERT_DELIVERY_READY" in cell
    assert "submission.zip" not in cell


def test_checked_in_stage_b_cell_matches_renderer(tmp_path: Path) -> None:
    rendered = build_kaggle_cell("B", tmp_path / "cell.py")
    checked = Path("experiments/direct_expert/KAGGLE_STAGE_B_CELL.py")
    assert checked.read_bytes() == rendered.read_bytes()
