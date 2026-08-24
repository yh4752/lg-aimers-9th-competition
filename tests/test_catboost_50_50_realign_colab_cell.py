from __future__ import annotations

from hashlib import sha256
import io
import os
from pathlib import Path
import subprocess
import tarfile

from experiments.catboost_50_50_realign.runtime_inventory import (
    RUNTIME_MEMBERS,
    code_identity_sha256,
    render_colab_cell,
    runtime_archive,
)


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "experiments/catboost_50_50_realign/COLAB_CATBOOST_50_50_REALIGN_CELL.py"
PYTHON = (
    "/Users/yonghyun/Documents/lg-aimers-9th-competition/artifacts/"
    "tabm_submission_python311/bin/python"
)


def test_runtime_inventory_is_explicit_and_rule_safe() -> None:
    assert "experiments/catboost_50_50_realign/runner.py" in RUNTIME_MEMBERS
    assert "experiments/catboost_50_50_realign/requirements-colab.txt" in RUNTIME_MEMBERS
    assert len(RUNTIME_MEMBERS) == len(set(RUNTIME_MEMBERS))
    assert all("submission" not in name.lower() for name in RUNTIME_MEMBERS)
    assert len(runtime_archive(ROOT)) < 900_000


def test_checked_in_cell_matches_deterministic_renderer() -> None:
    first = render_colab_cell(ROOT)
    second = render_colab_cell(ROOT)
    assert first == second
    assert CELL.read_bytes() == first
    assert len(first) < 1_000_000
    text = first.decode("utf-8")
    assert text.index("SESSION_DEADLINE = time.time() + 10800") < text.index(
        "files.upload()"
    )
    assert "drive.mount" not in text.lower()
    assert "github" not in text.lower()
    assert "http://" not in text and "https://" not in text
    assert "REALIGN_CAMPAIGN_SUCCESS" in text
    assert "REALIGN_ERROR" in text
    assert "REALIGN_RESUME_CACHE_READY" in text
    purge = 'if module_name == "experiments" or module_name.startswith("experiments.")'
    assert purge in text
    assert text.index(purge) < text.index(
        "from experiments.catboost_50_50_realign.colab import"
    )
    assert sha256(first).hexdigest()


def test_embedded_runtime_imports_without_repository(tmp_path: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(runtime_archive(ROOT)), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [
            PYTHON,
            "-c",
            "from experiments.catboost_50_50_realign.contracts import build_jobs,load_contract;"
            "from experiments.catboost_50_50_realign.colab import DownloadEvent;"
            "from experiments.catboost_50_50_realign.runner import _code_sha256;"
            "assert len(build_jobs(load_contract()))==3;assert DownloadEvent;"
            "assert len(_code_sha256())==64",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_requirements_are_in_code_identity(tmp_path: Path) -> None:
    copied = tmp_path / "repo"
    for name in RUNTIME_MEMBERS:
        target = copied / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    before = code_identity_sha256(copied)
    requirements = copied / "experiments/catboost_50_50_realign/requirements-colab.txt"
    requirements.write_text("catboost==1.2.9\n")
    assert code_identity_sha256(copied) != before
