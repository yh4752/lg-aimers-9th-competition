"""Render the self-contained Kaggle preprocessing cell from a sealed commit."""

from __future__ import annotations

import base64
import gzip
from hashlib import sha256
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_COMMIT = "9a88d2e2ad482a637a72179bb2fc65df72e79704"
OUTPUT = ROOT / "experiments/preprocessing_campaign/KAGGLE_CELL.py"


def _archive() -> bytes:
    completed = subprocess.run(
        ["git", "archive", "--format=tar", RUNTIME_COMMIT, "experiments"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return gzip.compress(completed.stdout, compresslevel=9, mtime=0)


def main() -> None:
    archive = _archive()
    encoded = base64.b64encode(archive).decode("ascii")
    digest = sha256(archive).hexdigest()
    template = '''from __future__ import annotations

from pathlib import Path
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback


# 사용자 설정: Save Version 한 번에는 fold 작업 하나만 완료·재개합니다.
MAX_JOBS_PER_SESSION = 1
# 약 8시간 20분 뒤에는 마지막 완료 epoch 체크포인트에서 정상 종료합니다.
MAX_SESSION_SECONDS = 30000
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
REPO_DIR = WORKING_ROOT / "preprocessing_embedded_code_9a88d2e"
REQUIRED_CODE_COMMIT = "__COMMIT__"
RUNTIME_DIR = WORKING_ROOT / "preprocessing_runtime_v1"
CAMPAIGN_OUTPUT_DIR = WORKING_ROOT / "preprocessing_campaign_v1"
REVIEW_ZIP = WORKING_ROOT / "preprocessing_campaign_review_bundle.zip"


def run_checked(command, *, cwd=None, env=None):
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command_failed returncode={completed.returncode} program={command[0]}"
        )
    return completed


def find_unique_file(filename):
    matches = sorted(path for path in INPUT_ROOT.rglob(filename) if path.is_file())
    if len(matches) != 1:
        detail = "\\n".join(f"- {path}" for path in matches) or "- not_found"
        raise RuntimeError(
            f"{filename} must exist exactly once; found={len(matches)}\\n{detail}"
        )
    return matches[0]


def find_checkpoint_source():
    matches = sorted(
        path
        for path in INPUT_ROOT.rglob("preprocessing_campaign_v1")
        if path.is_dir()
    )
    if len(matches) > 1:
        detail = "\\n".join(f"- {path}" for path in matches)
        raise RuntimeError(
            "Attach only one preprocessing_campaign_v1 checkpoint directory:\\n"
            + detail
        )
    return matches[0] if matches else None


def write_review_if_possible(config):
    manifest = CAMPAIGN_OUTPUT_DIR / "campaign_manifest.json"
    if not manifest.is_file() or not config.is_file():
        return None
    command = (
        "from experiments.preprocessing_campaign.review import write_review_bundle; "
        f"print(write_review_bundle({str(CAMPAIGN_OUTPUT_DIR)!r}, "
        f"{str(config)!r}, {str(REVIEW_ZIP)!r}))"
    )
    run_checked([sys.executable, "-c", command], cwd=REPO_DIR, env=child_env)
    return REVIEW_ZIP


try:
    if not INPUT_ROOT.is_dir() or not WORKING_ROOT.is_dir():
        raise RuntimeError("This cell must run in a Kaggle Notebook")
    if isinstance(MAX_JOBS_PER_SESSION, bool) or MAX_JOBS_PER_SESSION < 1:
        raise RuntimeError("MAX_JOBS_PER_SESSION must be a positive integer")
    if isinstance(MAX_SESSION_SECONDS, bool) or MAX_SESSION_SECONDS < 1:
        raise RuntimeError("MAX_SESSION_SECONDS must be a positive integer")

    train_path = find_unique_file("train.csv")
    trackman_path = find_unique_file("trackman_history.csv")
    if train_path.parent != trackman_path.parent:
        raise RuntimeError(
            "train.csv and trackman_history.csv must share one Dataset directory\\n"
            f"train={train_path}\\ntrackman={trackman_path}"
        )
    DATA_DIR = train_path.parent
    print("official_data=", DATA_DIR)

    checkpoint_source = find_checkpoint_source()
    if CAMPAIGN_OUTPUT_DIR.exists():
        if not CAMPAIGN_OUTPUT_DIR.is_dir():
            raise RuntimeError(f"output path is not a directory: {CAMPAIGN_OUTPUT_DIR}")
        print("resume_source=/kaggle/working")
    elif checkpoint_source is not None:
        shutil.copytree(checkpoint_source, CAMPAIGN_OUTPUT_DIR, dirs_exist_ok=False)
        print("resume_source=", checkpoint_source)
    else:
        CAMPAIGN_OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
        print("resume_source=new_campaign")

    EMBEDDED_RUNTIME_B64 = "__ARCHIVE__"
    EMBEDDED_RUNTIME_SHA256 = "__SHA256__"
    archive_bytes = base64.b64decode(EMBEDDED_RUNTIME_B64, validate=True)
    if hashlib.sha256(archive_bytes).hexdigest() != EMBEDDED_RUNTIME_SHA256:
        raise RuntimeError("runtime archive SHA-256 mismatch")
    runtime_marker = REPO_DIR / ".embedded_runtime_sha256"
    if REPO_DIR.exists():
        if not runtime_marker.is_file():
            raise RuntimeError(f"unverified existing code directory: {REPO_DIR}")
        if runtime_marker.read_text(encoding="utf-8").strip() != EMBEDDED_RUNTIME_SHA256:
            raise RuntimeError("existing embedded runtime SHA-256 differs")
    else:
        REPO_DIR.mkdir(parents=False, exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"unsafe runtime archive path: {member.name}")
                if member.issym() or member.islnk():
                    raise RuntimeError(f"runtime archive links are forbidden: {member.name}")
            archive.extractall(REPO_DIR)
        runtime_marker.write_text(EMBEDDED_RUNTIME_SHA256 + "\\n", encoding="utf-8")
    print("embedded_code_commit=", REQUIRED_CODE_COMMIT)

    requirements = REPO_DIR / "experiments/independent_dl/requirements-colab.txt"
    requirements_sha = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = RUNTIME_DIR / ".requirements_sha256"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != requirements_sha:
        if RUNTIME_DIR.exists():
            shutil.rmtree(RUNTIME_DIR)
        RUNTIME_DIR.mkdir(parents=True)
        run_checked(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(RUNTIME_DIR),
                "--upgrade",
                "--no-deps",
                "-r",
                str(requirements),
            ]
        )
        marker.write_text(requirements_sha + "\\n", encoding="utf-8")

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(RUNTIME_DIR), str(REPO_DIR), child_env.get("PYTHONPATH", "")]
    )
    child_env["PYTHONUNBUFFERED"] = "1"
    gpu_check = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA GPU is unavailable'; "
        "print('GPU:', torch.cuda.get_device_name(0))"
    )
    run_checked([sys.executable, "-c", gpu_check], cwd=REPO_DIR, env=child_env)

    config = REPO_DIR / "experiments/independent_dl/configs/preprocessing_ablation_v1.json"
    command = [
        sys.executable,
        "-m",
        "experiments.preprocessing_campaign.run_campaign",
        "run",
        "--wave",
        "a",
        "--config",
        str(config),
        "--data-dir",
        str(DATA_DIR),
        "--output-dir",
        str(CAMPAIGN_OUTPUT_DIR),
        "--max-jobs",
        str(MAX_JOBS_PER_SESSION),
        "--max-session-seconds",
        str(MAX_SESSION_SECONDS),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_DIR,
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"campaign process failed returncode={returncode}")

    review = write_review_if_possible(config)
    manifest = CAMPAIGN_OUTPUT_DIR / "campaign_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    states = {}
    for entry in payload.get("jobs", {}).values():
        state = str(entry.get("state", "unknown"))
        states[state] = states.get(state, 0) + 1
    print("PREPROCESSING_KAGGLE_SUCCESS")
    print("campaign_output=", CAMPAIGN_OUTPUT_DIR)
    print("review_zip=", review)
    print("state_counts=", json.dumps(states, sort_keys=True))
    print("next_step=Save Version, then attach the review ZIP in this chat")
except BaseException as error:
    traceback.print_exc()
    try:
        if "config" in globals() and "child_env" in globals():
            partial_review = write_review_if_possible(config)
            print("partial_review_zip=", partial_review)
    except BaseException:
        traceback.print_exc()
    print(
        "PREPROCESSING_KAGGLE_ERROR "
        f"type={type(error).__name__} message={str(error).replace(' ', '_')}"
    )
    raise
'''
    rendered = (
        template.replace("__COMMIT__", RUNTIME_COMMIT)
        .replace("__ARCHIVE__", encoded)
        .replace("__SHA256__", digest)
    )
    OUTPUT.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
