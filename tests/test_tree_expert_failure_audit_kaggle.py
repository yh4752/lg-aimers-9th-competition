from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from experiments.tree_expert.failure_audit_kaggle import (
    FailureAuditKaggleError,
    discover_official_data,
    runtime_archive,
    runtime_member_names,
)


def _official_fixture(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "train.csv").write_text("row_id,control_success\nTRAIN_1,1\n", encoding="utf-8")
    (root / "trackman_history.csv").write_text("pitcher_id\nP1\n", encoding="utf-8")


def test_runtime_archive_contains_only_audit_dependencies() -> None:
    members = runtime_member_names()

    assert "experiments/tree_expert/failure_audit.py" in members
    assert "experiments/tree_expert/failure_audit_artifacts.py" in members
    assert "experiments/tree_expert/failure_audit_contract.json" in members
    assert not any("training.py" in name or "inference.py" in name for name in members)


def test_runtime_archive_imports_in_isolation(tmp_path: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(runtime_archive()), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experiments.tree_expert.failure_audit_kaggle; "
            "import experiments.tree_expert.failure_audit",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_official_discovery_requires_one_exact_root(tmp_path: Path) -> None:
    _official_fixture(tmp_path / "official")
    assert discover_official_data(tmp_path, testing=True).name == "official"
    _official_fixture(tmp_path / "duplicate")

    with pytest.raises(FailureAuditKaggleError, match="official data count must be one"):
        discover_official_data(tmp_path, testing=True)
