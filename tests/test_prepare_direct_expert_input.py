from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_cli_is_import_safe_outside_repository(tmp_path: Path) -> None:
    script = Path("tools/prepare_direct_expert_input.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--s4-handoff" in result.stdout
    assert "--e2-submission" in result.stdout
    assert "--e2-receipt" in result.stdout
