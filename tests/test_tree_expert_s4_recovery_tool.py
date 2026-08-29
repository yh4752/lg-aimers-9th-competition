from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

from experiments.tree_expert.s4_recovery import RecoveryResult


def _load_tool():
    script = Path(__file__).resolve().parents[1] / "tools/prepare_tree_s4_recovery_input.py"
    spec = importlib.util.spec_from_file_location("prepare_tree_s4_recovery_input", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_tool_imports_repository_from_another_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/prepare_tree_s4_recovery_input.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--source-handoff" in completed.stdout
    assert "--output" in completed.stdout


def test_recovery_tool_reports_verified_source_and_compaction_sizes(
    tmp_path, monkeypatch, capsys,
):
    module = _load_tool()
    source = tmp_path / "source.zip"
    output = tmp_path / "recovery.zip"
    source.write_bytes(b"source")
    result = RecoveryResult(output, "a" * 64, "b" * 64, 123, 456)
    monkeypatch.setattr(module, "runtime_identity_sha256", lambda _root: "c" * 64)
    monkeypatch.setattr(module, "compact_recovery_handoff", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        sys, "argv",
        ["prepare_tree_s4_recovery_input.py", "--source-handoff", str(source), "--output", str(output)],
    )

    assert module.main() == 0
    text = capsys.readouterr().out
    assert f"TREE_S4_RECOVERY_SOURCE_VERIFIED sha256={'b' * 64}" in text
    assert "TREE_S4_RECOVERY_INPUT_READY" in text
    assert "retained_bytes=123 dropped_bytes=456" in text
    assert "submission" not in text.lower()
