from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import subprocess
import sys

import pytest

from experiments.tabm_campaign.version_d import RecoverySelection


CELL = Path("experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py")
RENDERER = Path("tools/render_tabm_colab_version_d_cell.py")


def test_frozen_recovery_skips_training(tmp_path: Path, monkeypatch) -> None:
    import experiments.tabm_campaign.colab_version_d as runtime

    calls: list[str] = []
    frozen = tmp_path / "frozen.zip"
    frozen.write_bytes(b"frozen")
    review = tmp_path / "review.zip"
    review.write_bytes(b"review")
    delivery = tmp_path / "delivery.zip"
    delivery.write_bytes(b"delivery")
    monkeypatch.setattr(runtime, "verify_inputs", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(
        runtime,
        "restore_recovery",
        lambda *args, **kwargs: RecoverySelection("frozen", frozen, 3),
    )
    monkeypatch.setattr(
        runtime,
        "fit_final",
        lambda *args, **kwargs: calls.append("fit"),
    )
    monkeypatch.setattr(runtime, "_frozen_fit_report", lambda *args: {"status": "completed"})
    monkeypatch.setattr(
        runtime,
        "review_final",
        lambda *args, **kwargs: calls.append("review") or review,
    )
    monkeypatch.setattr(
        runtime,
        "publish_delivery",
        lambda *args, **kwargs: delivery,
    )

    result = runtime.run_version_d(
        data_archive=tmp_path / "data.zip",
        stage_c_delivery=tmp_path / "stage_c.zip",
        work_root=tmp_path / "work",
        recovery_archive=frozen,
        absolute_deadline=4_000_000_000.0,
        runtime_sha256="1" * 64,
        runtime_versions={"python": "3.12"},
        log_path=tmp_path / "run.log",
        on_download=lambda path, kind: calls.append(kind),
    )
    assert result == delivery
    assert "fit" not in calls
    assert calls == ["frozen", "review", "delivery"]


def test_review_failure_never_publishes_delivery(tmp_path: Path, monkeypatch) -> None:
    import experiments.tabm_campaign.colab_version_d as runtime

    published: list[bool] = []
    frozen = tmp_path / "frozen.zip"
    frozen.write_bytes(b"frozen")
    monkeypatch.setattr(runtime, "verify_inputs", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(
        runtime,
        "restore_recovery",
        lambda *args, **kwargs: RecoverySelection("frozen", frozen, 3),
    )
    monkeypatch.setattr(
        runtime,
        "review_final",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("review failed")),
    )
    monkeypatch.setattr(runtime, "_frozen_fit_report", lambda *args: {"status": "completed"})
    monkeypatch.setattr(
        runtime,
        "publish_delivery",
        lambda *args, **kwargs: published.append(True),
    )
    with pytest.raises(RuntimeError, match="review failed"):
        runtime.run_version_d(
            data_archive=tmp_path / "data.zip",
            stage_c_delivery=tmp_path / "stage_c.zip",
            work_root=tmp_path / "work",
            recovery_archive=frozen,
            absolute_deadline=4_000_000_000.0,
            runtime_sha256="1" * 64,
            runtime_versions={"python": "3.12"},
            log_path=tmp_path / "run.log",
            on_download=lambda path, kind: None,
        )
    assert published == []


def test_expired_deadline_stops_before_input_verification(tmp_path: Path, monkeypatch) -> None:
    import experiments.tabm_campaign.colab_version_d as runtime

    verified: list[bool] = []
    monkeypatch.setattr(
        runtime,
        "verify_inputs",
        lambda *args, **kwargs: verified.append(True),
    )
    with pytest.raises(runtime.ColabVersionDError, match="deadline"):
        runtime.run_version_d(
            data_archive=tmp_path / "data.zip",
            stage_c_delivery=tmp_path / "stage_c.zip",
            work_root=tmp_path / "work",
            recovery_archive=None,
            absolute_deadline=0.0,
            runtime_sha256="1" * 64,
            runtime_versions={"python": "3.12"},
            log_path=tmp_path / "run.log",
            on_download=lambda path, kind: None,
        )
    assert verified == []


def test_version_d_cell_is_self_contained_small_and_review_only() -> None:
    text = CELL.read_text(encoding="utf-8")
    compile(text, str(CELL), "exec")
    assert CELL.stat().st_size < 950_000
    assert "files.upload()" in text
    assert "files.download(" in text
    assert "RECOVERY_MODE" in text
    assert "VERSION_D_DELIVERY_READY" in text
    lowered = text.lower()
    assert "drive.mount" not in lowered
    assert "git clone" not in lowered
    assert "submit.zip" not in lowered
    assert "script.py" not in lowered


def test_version_d_log_starts_before_code_ready_marker() -> None:
    text = CELL.read_text(encoding="utf-8")
    assert text.index('LOG_PATH.open("w"') < text.index("VERSION_D_CODE_READY")


def test_version_d_renderer_is_deterministic() -> None:
    subprocess.run([sys.executable, str(RENDERER)], check=True)
    first = sha256(CELL.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(RENDERER)], check=True)
    assert sha256(CELL.read_bytes()).hexdigest() == first


def test_version_d_runtime_excludes_stage_c_working_files() -> None:
    from tools import render_tabm_colab_version_d_cell as renderer

    names = {path.name for path in renderer._source_paths()}
    assert "colab_recovery.py" not in names
    assert "colab_stage_c_contract.json" not in names
    assert "COLAB_STAGE_C_RECOVERY_CELL.py" not in names


def test_version_d_cell_repairs_colab_venv_without_changing_runtime_identity() -> None:
    from tools import render_tabm_colab_version_d_cell as renderer

    text = renderer.render().decode("utf-8")
    assert "def ensure_venv_ready():" in text
    assert 'stage = "venv"' in text
    assert "python3.12-venv" in text
    assert '\n    ensure_venv_ready()\n\n    stage = "run"' in text
    assert sha256(renderer._archive_bytes()).hexdigest() == (
        "75500c32988d94f671cff16195d0633a3f3f2d1fb648c8d6e28ddc4a396ad1f2"
    )
