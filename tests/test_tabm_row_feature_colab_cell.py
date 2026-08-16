from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from tools import build_tabm_row_feature_colab_cell as builder


CELL = Path("experiments/tabm_campaign/COLAB_ROW_FEATURE_PROXY_CELL.py")
EXPECTED_RUNTIME_MEMBERS = {
    "experiments/independent_dl/feature_sources/trackman.py",
    "experiments/independent_dl/features.py",
    "experiments/independent_dl/models/common.py",
    "experiments/independent_dl/models/tabm.py",
    "experiments/independent_dl/preprocessing.py",
    "experiments/independent_dl/progress.py",
    "experiments/independent_dl/row_features.py",
    "experiments/independent_dl/training.py",
    "experiments/tabm_campaign/artifacts.py",
    "experiments/tabm_campaign/cache.py",
    "experiments/tabm_campaign/configs/row_feature_proxy_v1.json",
    "experiments/tabm_campaign/requirements-kaggle.txt",
    "experiments/tabm_campaign/row_feature_colab.py",
    "experiments/tabm_campaign/row_feature_contracts.py",
    "experiments/tabm_campaign/row_feature_decisions.py",
    "experiments/tabm_campaign/row_feature_proxy.py",
    "experiments/tabm_campaign/row_feature_runtime.py",
    "experiments/tabm_campaign/sampling.py",
    "experiments/tabm_campaign/training.py",
    "experiments/tabm_campaign/worker.py",
}


def test_embedded_runtime_inventory_is_minimal_complete_and_deterministic() -> None:
    paths = builder._source_paths()
    names = {path.relative_to(builder.ROOT).as_posix() for path in paths}
    assert names == EXPECTED_RUNTIME_MEMBERS
    assert builder._archive_bytes() == builder._archive_bytes()
    with tarfile.open(fileobj=io.BytesIO(builder._archive_bytes()), mode="r:gz") as archive:
        assert set(archive.getnames()) == names
        decoded = b"\n".join(
            archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        )
    for forbidden in (
        b"inference_runtime.py",
        b"final_training.py",
        b"handoff.py",
        b"KAGGLE_SUBMISSION",
        b"test.csv",
        b"sample_submission.csv",
    ):
        assert forbidden not in decoded
    assert b"checkpoint_payload_validator=(" in decoded
    assert b"_validate_checkpoint_payload_isolated" in decoded


def test_embedded_runtime_imports_and_builds_stage_p_grid_in_isolation(
    tmp_path: Path,
) -> None:
    with tarfile.open(fileobj=io.BytesIO(builder._archive_bytes()), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "from experiments.tabm_campaign.row_feature_contracts import "
                "load_row_feature_proxy_contract; "
                "from experiments.tabm_campaign.row_feature_proxy import build_proxy_jobs; "
                "import experiments.tabm_campaign.row_feature_colab; "
                "from experiments.tabm_campaign import worker; "
                "jobs=build_proxy_jobs(load_row_feature_proxy_contract()); "
                "assert len(jobs) == 14; "
                "caught=False; "
                "\ntry: worker.run_worker(jobs[0], Path('missing-data'), Path('out'), Path('cache'), 10**12)"
                "\nexcept RuntimeError as error: caught='train.csv' in str(error)"
                "\nassert caught"
                "\nassert all(name in sys.modules for name in ("
                "'experiments.tabm_campaign.cache', "
                "'experiments.tabm_campaign.sampling', "
                "'experiments.tabm_campaign.training'))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


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
    registration = text.index(
        "        register_verified_uploaded_resume(\n            resume_path,"
    )
    assert registration < text.index('stage = "dependencies"')
    assert registration < text.index('stage = "gpu"')
    assert "latest_verified[0] = path" in text
    assert text.index("publish_latest_verified_resume()", registration) > registration
    assert "def request_download(path: Path, *, enforce_deadline: bool = True)" in text
    assert "request_download(Path(path), enforce_deadline=False)" in text
    assert text.count("check_deadline=remaining_seconds") >= 2


def test_generated_cell_uses_an_exclusive_run_root_and_reuses_only_verified_code(
    tmp_path: Path,
) -> None:
    text = builder.render().decode("utf-8")
    tree = ast.parse(text)
    create_run_root = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_run_root"
    )
    namespace = {"Path": Path}
    exec(compile(ast.Module([create_run_root], []), "<run-root>", "exec"), namespace)
    first = namespace["create_run_root"](tmp_path, "run-one")
    (first / "verified-resume.zip").write_bytes(b"preserve")
    second = namespace["create_run_root"](tmp_path, "run-two")

    assert first != second
    assert (first / "verified-resume.zip").read_bytes() == b"preserve"
    assert "RUN_ROOT = create_run_root(" in text
    for name in (
        "INPUT_ROOT",
        "DATA_ROOT",
        "OUTPUT_ROOT",
        "SNAPSHOT_ROOT",
        "LOG_PATH",
        "DELIVERY_PATH",
    ):
        assert f"{name} = RUN_ROOT /" in text
    assert "shutil.rmtree(CODE_ROOT)" not in text
    assert "verify_extracted_runtime(archive_bytes)" in text
    assert "temporary_dir=RUN_ROOT" in text


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
