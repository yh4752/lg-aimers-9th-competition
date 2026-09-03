from __future__ import annotations

from hashlib import sha256
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


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


def test_cached_upload_discovery_keeps_latest_file_per_content_kind(
    tmp_path: Path,
) -> None:
    from experiments.hierarchical_tabm.runtime_inventory import discover_cached_uploads

    old = tmp_path / "old.zip"; old.write_bytes(b"training")
    new = tmp_path / "new.zip"; new.write_bytes(b"training")
    stage = tmp_path / "stage.zip"; stage.write_bytes(b"stage")
    unknown = tmp_path / "unknown.zip"; unknown.write_bytes(b"unknown")
    now = time.time()
    os.utime(old, (now - 10, now - 10))

    kinds = {b"training": "training_input", b"stage": "stage_c_delivery"}
    result = discover_cached_uploads(
        (unknown, old, stage, new), classifier=lambda path: kinds[path.read_bytes()]
    )
    assert result == (new, stage)


def test_embedded_runtime_imports_without_repository_pythonpath(tmp_path: Path) -> None:
    from experiments.hierarchical_tabm.runtime_inventory import runtime_archive

    with tarfile.open(fileobj=io.BytesIO(runtime_archive(ROOT)), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [
            str(ROOT / "artifacts/tabm_submission_python311/bin/python"),
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
    assert "HIER_UPLOAD_CACHE_REUSED" in text
    assert 'RUN_BASE.rglob("hierarchical_tabm_resume.zip")' in text
    purge = 'if module_name == "experiments" or module_name.startswith("experiments."):'
    assert purge in text
    assert text.index(purge) < text.index(
        "from experiments.hierarchical_tabm.runtime_inventory import code_identity_sha256"
    )
    assert sha256(first).hexdigest()
