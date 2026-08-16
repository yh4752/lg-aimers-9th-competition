from __future__ import annotations

import io
import tarfile
from pathlib import Path

from tools import build_tabm_row_feature_colab_cell as builder


CELL = Path("experiments/tabm_campaign/COLAB_ROW_FEATURE_PROXY_CELL.py")


def test_embedded_runtime_inventory_is_minimal_complete_and_deterministic() -> None:
    paths = builder._source_paths()
    names = {path.relative_to(builder.ROOT).as_posix() for path in paths}
    assert "experiments/independent_dl/row_features.py" in names
    assert "experiments/tabm_campaign/row_feature_colab.py" in names
    assert "experiments/tabm_campaign/configs/row_feature_proxy_v1.json" in names
    assert "experiments/tabm_campaign/requirements-kaggle.txt" in names
    assert all("COLAB_" not in path.name and "KAGGLE_" not in path.name for path in paths)
    assert builder._archive_bytes() == builder._archive_bytes()
    with tarfile.open(fileobj=io.BytesIO(builder._archive_bytes()), mode="r:gz") as archive:
        assert set(archive.getnames()) == names


def test_generated_cell_is_deterministic_compilable_and_under_limit() -> None:
    first = builder.render()
    second = builder.render()
    assert first == second
    assert len(first) < 1_000_000
    compile(first.decode("utf-8"), str(CELL), "exec")


def test_generated_cell_has_direct_upload_deadline_and_recovery_contract() -> None:
    text = builder.render().decode("utf-8")
    for marker in (
        "ROW_FEATURE_CODE_READY",
        "ROW_FEATURE_INPUTS_VERIFIED",
        "ROW_FEATURE_DEPENDENCIES_READY",
        "ROW_FEATURE_GPU_READY",
        "ROW_FEATURE_STAGE_SELECTED version=P",
        "JOB_START",
        "TRAINING_PROGRESS",
        "EPOCH_CHECKPOINTED",
        "ROW_FEATURE_BUNDLE_SUCCESS",
        "ROW_FEATURE_DELIVERY_READY",
        "ROW_FEATURE_ERROR stage={stage} type={type(error).__name__} message={message}",
    ):
        assert marker in text
    assert "google.colab.files.upload()" in text
    assert text.count("google.colab.files.upload()") == 1
    assert "len(uploaded) not in {1, 2}" in text
    assert "SESSION_DEADLINE = time.time() + 10800" in text
    assert "NEW_JOB_GUARD_SECONDS = 900" in text
    assert text.index("SESSION_DEADLINE = time.time() + 10800") < text.index(
        "google.colab.files.upload()"
    )
    assert text.index("SESSION_DEADLINE = time.time() + 10800") < text.index(
        '"pip", "install"'
    )
    assert "snapshot_interval_seconds=600" in text
    assert "tabm_row_feature_stage_P_delivery.zip" in text
    assert "publish_latest_verified_resume" in text
    assert "raise" in text[text.index("publish_latest_verified_resume") :]


def test_generated_cell_has_no_remote_or_disallowed_data_paths() -> None:
    text = builder.render().decode("utf-8").casefold()
    for forbidden in (
        "drive.mount",
        "google drive",
        "github",
        "git clone",
        "http://",
        "https://",
        "test.csv",
        "sample_submission.csv",
        "submit.zip",
        "create_submission",
        "run_inference",
    ):
        assert forbidden not in text


def test_checked_in_cell_matches_renderer() -> None:
    assert CELL.is_file()
    assert CELL.read_bytes() == builder.render()
