from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PYTHON = Path(sys.executable)
TOOL = Path("tools/build_experiment_audit.py")


def test_cli_writes_deterministic_report(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "schema_version": 1,
        "experiments": [],
        "evidence_gaps": [],
    }), encoding="utf-8")
    output = tmp_path / "audit.md"
    command = [
        str(PYTHON),
        str(TOOL),
        "--registry",
        str(registry),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "EXPERIMENT_AUDIT_SUCCESS" in result.stdout
    first = output.read_bytes()
    subprocess.run(command, text=True, capture_output=True, check=True)
    assert output.read_bytes() == first


def test_cli_rejects_wrong_declared_artifact_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "review.zip"
    artifact.write_bytes(b"fixture")
    result = subprocess.run(
        [
            str(PYTHON),
            str(TOOL),
            "--registry",
            "reports/experiment_registry.json",
            "--output",
            str(tmp_path / "audit.md"),
            "--artifact",
            f"tree_privileged_profile_p_only_v1={'0' * 64}:{artifact}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "EXPERIMENT_AUDIT_ERROR" in result.stdout
    assert "artifact_sha256_differs" in result.stdout


def test_cli_rejects_artifact_for_unknown_experiment(tmp_path: Path) -> None:
    artifact = tmp_path / "review.zip"
    artifact.write_bytes(b"fixture")
    result = subprocess.run(
        [
            str(PYTHON),
            str(TOOL),
            "--registry",
            "reports/experiment_registry.json",
            "--output",
            str(tmp_path / "audit.md"),
            "--artifact",
            f"unknown={'0' * 64}:{artifact}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "unknown_experiment_id" in result.stdout
