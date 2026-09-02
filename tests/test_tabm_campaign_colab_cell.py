from __future__ import annotations

from pathlib import Path
import subprocess
import sys


CELL = Path("experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py")
HANDOFF_BUILDER = Path("tools/prepare_tabm_colab_stage_c_handoff.py")


def test_colab_cell_is_self_contained_and_small() -> None:
    text = CELL.read_text(encoding="utf-8")
    compile(text, str(CELL), "exec")
    assert CELL.stat().st_size < 950_000
    assert "google.colab" in text
    assert "files.upload()" in text
    assert "files.download(" in text
    assert "/content/tabm_stage_c_recovery" in text
    assert '"--gpu-count", "1"' in text
    assert "UPLOAD_EMERGENCY_SNAPSHOT" in text


def test_colab_cell_has_no_drive_github_or_package_entrypoint() -> None:
    text = CELL.read_text(encoding="utf-8").lower()
    assert "drive.mount" not in text
    assert "git clone" not in text
    assert "github.com" not in text
    assert "submit.zip" not in text
    assert "submission package" not in text


def test_colab_cell_exposes_required_logs_and_download_contract() -> None:
    text = CELL.read_text(encoding="utf-8")
    for marker in (
        "COLAB_CODE_READY",
        "COLAB_INPUTS_VERIFIED",
        "COLAB_GPU_READY",
        "EMERGENCY_SNAPSHOT_READY",
        "EMERGENCY_DOWNLOAD_REQUESTED",
        "COLAB_DELIVERY_READY",
        "COLAB_DOWNLOAD_REQUESTED",
        "COLAB_STAGE_C_ERROR",
    ):
        assert marker in text


def test_colab_renderer_is_byte_deterministic() -> None:
    from tools import render_tabm_colab_stage_c_recovery_cell as renderer

    assert renderer.render() == renderer.render()


def test_colab_handoff_builder_supports_direct_script_execution() -> None:
    completed = subprocess.run(
        [sys.executable, str(HANDOFF_BUILDER), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--data-dir" in completed.stdout
