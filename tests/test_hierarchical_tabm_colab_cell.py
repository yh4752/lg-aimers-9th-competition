from __future__ import annotations

from hashlib import sha256
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py"


def test_runtime_archive_has_exact_import_closure() -> None:
    from experiments.hierarchical_tabm.runtime_inventory import (
        RUNTIME_MEMBERS, runtime_archive,
    )

    with tarfile.open(fileobj=io.BytesIO(runtime_archive(ROOT)), mode="r:gz") as archive:
        names = tuple(member.name for member in archive.getmembers())
    assert names == RUNTIME_MEMBERS
    assert "experiments/hierarchical_tabm/requirements-colab.txt" in names
    assert not any("submission" in name.lower() or "test.csv" in name.lower() for name in names)


def test_dependency_change_changes_code_identity(tmp_path: Path) -> None:
    from experiments.hierarchical_tabm.runtime_inventory import (
        RUNTIME_MEMBERS, code_identity_sha256,
    )

    copied = tmp_path / "runtime"
    for name in RUNTIME_MEMBERS:
        target = copied / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    before = code_identity_sha256(copied)
    (copied / "experiments/hierarchical_tabm/requirements-colab.txt").write_text(
        "tabm==0.0.4\nrtdl-num-embeddings==0.0.12\n", encoding="utf-8"
    )
    assert code_identity_sha256(copied) != before


def test_embedded_runtime_imports_without_repository_pythonpath(tmp_path: Path) -> None:
    from experiments.hierarchical_tabm.runtime_inventory import runtime_archive

    with tarfile.open(fileobj=io.BytesIO(runtime_archive(ROOT)), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [
            "/Users/yonghyun/Documents/lg-aimers-9th-competition/artifacts/"
            "tabm_submission_python311/bin/python",
            "-c",
            "from experiments.hierarchical_tabm.contracts import build_jobs,load_contract;"
            "from experiments.hierarchical_tabm.colab import SnapshotCadence;"
            "assert len(build_jobs(load_contract()))==3;assert SnapshotCadence",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_checked_in_cell_matches_renderer_and_static_safety() -> None:
    from experiments.hierarchical_tabm.runtime_inventory import render_cell

    first = render_cell(ROOT)
    second = render_cell(ROOT)
    assert first == second
    assert CELL.read_bytes() == first
    assert len(first) < 1_000_000
    text = first.decode("utf-8")
    assert text.index("SESSION_DEADLINE = time.time() + 10800") < text.index("files.upload()")
    lowered = text.lower()
    assert "drive.mount" not in lowered
    assert "github" not in lowered
    assert "HIER_ERROR" in text
    assert "HIER_INPUTS_VERIFIED" in text
    assert sha256(first).hexdigest()
