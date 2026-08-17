from __future__ import annotations

from hashlib import sha256
import io
import os
from pathlib import Path
import subprocess
import tarfile

from experiments.catboost_deployment.runtime_inventory import (
    RUNTIME_MEMBERS,
    code_identity_sha256,
    render_colab_cell,
    runtime_archive,
)
from experiments.catboost_deployment.runner import code_sha256


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "experiments/catboost_deployment/COLAB_CATBOOST_DEPLOYMENT_CELL.py"
PYTHON = (
    "/Users/yonghyun/Documents/lg-aimers-9th-competition/artifacts/"
    "tabm_submission_python311/bin/python"
)


def test_runtime_inventory_is_explicit_and_rule_safe() -> None:
    assert "experiments/catboost_deployment/runner.py" in RUNTIME_MEMBERS
    assert "experiments/catboost_deployment/requirements-colab.txt" in RUNTIME_MEMBERS
    assert all("submission" not in name.lower() for name in RUNTIME_MEMBERS)
    assert all("inference_runtime" not in name.lower() for name in RUNTIME_MEMBERS)
    assert len(RUNTIME_MEMBERS) == len(set(RUNTIME_MEMBERS))
    assert len(runtime_archive(ROOT)) < 900_000


def test_checked_in_cell_matches_deterministic_renderer() -> None:
    first = render_colab_cell(ROOT)
    second = render_colab_cell(ROOT)

    assert first == second
    assert CELL.read_bytes() == first
    assert len(first) < 1_000_000
    text = first.decode("utf-8")
    assert text.index("SESSION_DEADLINE = time.time() + 10800") < text.index(
        "google.colab.files.upload()"
    )
    assert "drive.mount" not in text.lower()
    assert "github.com" not in text.lower()
    assert "DEPLOY_DELIVERY_READY" in text
    assert "DEPLOY_ERROR" in text
    assert sha256(first).hexdigest()


def test_embedded_runtime_imports_without_repository_on_pythonpath(
    tmp_path: Path,
) -> None:
    with tarfile.open(fileobj=io.BytesIO(runtime_archive(ROOT)), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [
            PYTHON,
            "-c",
            "from experiments.catboost_deployment.contracts import build_jobs,load_contract;"
            "from experiments.catboost_deployment.colab import EmergencyCadence;"
            "assert len(build_jobs(load_contract())) == 3; assert EmergencyCadence",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_requirements_are_part_of_code_identity(tmp_path: Path) -> None:
    copied = tmp_path / "repo"
    for name in RUNTIME_MEMBERS:
        target = copied / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    before = code_identity_sha256(copied)
    requirements = copied / "experiments/catboost_deployment/requirements-colab.txt"
    requirements.write_text("catboost==1.2.9\n")

    assert code_identity_sha256(copied) != before


def test_runner_uses_the_embedded_runtime_identity() -> None:
    assert code_sha256() == code_identity_sha256(ROOT)
