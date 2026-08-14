from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic

from IPython.display import FileLink, display


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")


def fail(stage: str, error: Exception) -> None:
    message = str(error).replace(" ", "_").replace("\n", "_")
    print(
        f"VALIDATION_ERROR stage={stage} type={type(error).__name__} message={message}",
        flush=True,
    )


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def find_handoff() -> Path:
    matches = []
    for path in INPUT_ROOT.rglob("handoff_manifest.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if value.get("artifact_kind") == "tabm_submission_validation_handoff_v1":
            matches.append((path, value))
    if len(matches) != 1:
        raise RuntimeError(f"handoff_manifest_count_must_be_1_found={len(matches)}")
    manifest_path, manifest = matches[0]
    if manifest.get("submission_package") is not False:
        raise RuntimeError("handoff_submission_package_flag_differs")
    root = manifest_path.parent
    declared = manifest.get("members")
    if not isinstance(declared, dict) or not declared:
        raise RuntimeError("handoff_member_manifest_is_invalid")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(declared):
        raise RuntimeError("handoff_member_set_differs")
    for name, expected in declared.items():
        path = root / name
        if path.is_symlink() or file_sha256(path) != expected:
            raise RuntimeError(f"handoff_member_sha256_differs_{name}")
    return root


def find_official_data() -> Path:
    matches = {
        path.parent
        for path in INPUT_ROOT.rglob("test.csv")
        if (path.parent / "sample_submission.csv").is_file()
    }
    if len(matches) != 1:
        raise RuntimeError(f"official_data_count_must_be_1_found={len(matches)}")
    return next(iter(matches))


def run(command: list[str], *, log_path: Path, env: dict[str, str] | None = None) -> float:
    print("COMMAND_START " + " ".join(command), flush=True)
    started = monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    elapsed = monotonic() - started
    print(f"COMMAND_END return_code={return_code} elapsed_seconds={elapsed:.3f}", flush=True)
    if return_code:
        raise RuntimeError(f"command_failed_return_code_{return_code}")
    return elapsed


stage = "setup"
try:
    handoff = find_handoff()
    official_data = find_official_data()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = WORKING_ROOT / f"tabm_submission_validation_{run_id}"
    work.mkdir(parents=True, exist_ok=False)
    print(f"VALIDATION_HANDOFF_FOUND path={handoff}", flush=True)
    print(f"VALIDATION_DATA_FOUND path={official_data}", flush=True)

    stage = "host_dependencies"
    install_log = work / "submitted_requirements_install.log"
    install_seconds = run(
        [sys.executable, "-m", "pip", "install", "-r", str(handoff / "requirements.txt")],
        log_path=install_log,
    )

    stage = "exact_environment"
    setup_log = work / "exact_environment_setup.log"
    run(
        [sys.executable, "-m", "pip", "install", "uv"],
        log_path=setup_log,
    )
    run(
        [sys.executable, "-m", "uv", "python", "install", "3.11.15"],
        log_path=setup_log,
    )
    exact_env = work / "python311_exact"
    run(
        [sys.executable, "-m", "uv", "venv", "--python", "3.11.15", str(exact_env)],
        log_path=setup_log,
    )
    exact_python = exact_env / "bin/python"
    run(
        [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            str(exact_python),
            "torch==2.7.1",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        log_path=setup_log,
    )
    run(
        [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            str(exact_python),
            "pandas==2.0.3",
            "numpy==1.26.4",
        ],
        log_path=setup_log,
    )
    run(
        [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            str(exact_python),
            "--no-deps",
            "tabm==0.0.3",
            "rtdl-num-embeddings==0.0.12",
        ],
        log_path=setup_log,
    )
    print(f"EXACT_ENV_READY python={exact_python}", flush=True)

    stage = "validation"
    output_dir = work / "result"
    environment = os.environ.copy()
    runtime_path = str(handoff / "runtime")
    environment["PYTHONPATH"] = runtime_path + os.pathsep + environment.get("PYTHONPATH", "")
    run(
        [
            sys.executable,
            "-m",
            "submission.tabm_validation",
            "--candidate-root",
            str(handoff / "candidate"),
            "--data-dir",
            str(official_data),
            "--output-dir",
            str(output_dir),
            "--compat-python",
            str(exact_python),
            "--requirements",
            str(handoff / "requirements.txt"),
            "--install-log",
            str(install_log),
            "--install-seconds",
            f"{install_seconds:.9f}",
        ],
        log_path=work / "validation_process.log",
        env=environment,
    )

    stage = "publish"
    bundle_dir = output_dir / "bundles"
    review_source = bundle_dir / "tabm_submission_validation_review_bundle.zip"
    resume_source = bundle_dir / "tabm_submission_validation_resume_bundle.zip"
    review = WORKING_ROOT / f"tabm_submission_validation_review_{run_id}.zip"
    resume = WORKING_ROOT / f"tabm_submission_validation_resume_{run_id}.zip"
    shutil.copy2(review_source, review)
    shutil.copy2(resume_source, resume)
    print(
        f"VALIDATION_SUCCESS review={review} resume={resume} "
        f"review_sha256={file_sha256(review)} resume_sha256={file_sha256(resume)}",
        flush=True,
    )
    display(FileLink(str(review)))
    display(FileLink(str(resume)))
except Exception as error:
    fail(stage, error)
    raise
