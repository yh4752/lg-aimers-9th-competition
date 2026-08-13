"""Render one offline-source Kaggle cell for the budgeted preprocessing campaign."""

from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_COMMIT = "1e0241c3c275cc0aa91cfdd395a19f695ab7ab1f"
OUTPUT = ROOT / "experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py"


def _archive() -> bytes:
    completed = subprocess.run(
        ["git", "archive", "--format=tar", RUNTIME_COMMIT, "experiments"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    filtered = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as source:
        with tarfile.open(fileobj=filtered, mode="w:") as target:
            for member in source.getmembers():
                name = Path(member.name).name
                if name.startswith("KAGGLE_") and name.endswith(".py"):
                    continue
                payload = source.extractfile(member) if member.isfile() else None
                target.addfile(member, payload)
    return gzip.compress(filtered.getvalue(), compresslevel=9, mtime=0)


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
import time
import traceback


SESSION_STARTED_UNIX = time.time()
MAX_SESSION_SECONDS = 6300
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
CODE_ROOT = WORKING_ROOT / "budgeted_preprocessing_embedded_code_1e0241c"
RUNTIME_ROOT = WORKING_ROOT / "budgeted_preprocessing_runtime"
CAMPAIGN_ROOT = WORKING_ROOT / "budgeted_preprocessing_campaign_v1"
CONFIG_PATH = CODE_ROOT / "experiments/preprocessing_campaign/configs/budgeted_campaign_v1.json"
REQUIRED_CODE_COMMIT = "__COMMIT__"


class StageNeedsReview(RuntimeError):
    pass


def remaining_seconds():
    return max(0.0, SESSION_STARTED_UNIX + MAX_SESSION_SECONDS - time.time())


def run_checked(command, *, cwd=None, env=None, timeout=None):
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command_failed returncode={completed.returncode} program={command[0]}"
        )
    return completed


def find_official_data():
    train_paths = sorted(path for path in INPUT_ROOT.rglob("train.csv") if path.is_file())
    history_paths = sorted(
        path for path in INPUT_ROOT.rglob("trackman_history.csv") if path.is_file()
    )
    pairs = [
        train.parent
        for train in train_paths
        if any(history.parent == train.parent for history in history_paths)
    ]
    if len(pairs) != 1:
        raise RuntimeError(
            "official train.csv and trackman_history.csv must form exactly one co-located pair; "
            f"found={len(pairs)}"
        )
    return pairs[0]


def extract_embedded_runtime():
    archive_bytes = base64.b64decode(EMBEDDED_RUNTIME_B64, validate=True)
    if hashlib.sha256(archive_bytes).hexdigest() != EMBEDDED_RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 mismatch")
    marker = CODE_ROOT / ".embedded_runtime_sha256"
    if CODE_ROOT.exists():
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != EMBEDDED_RUNTIME_SHA256:
            raise RuntimeError("existing embedded runtime is not hash-valid")
        return
    CODE_ROOT.mkdir(parents=False, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe embedded path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"embedded links are forbidden: {member.name}")
        archive.extractall(CODE_ROOT)
    marker.write_text(EMBEDDED_RUNTIME_SHA256 + "\\n", encoding="utf-8")


def install_runtime():
    requirements = CODE_ROOT / "experiments/preprocessing_campaign/requirements-kaggle-budgeted.txt"
    expected = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = RUNTIME_ROOT / ".requirements_sha256"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == expected:
        return
    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True)
    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(RUNTIME_ROOT),
            "--upgrade",
            "--no-deps",
            "-r",
            str(requirements),
        ],
        timeout=max(1.0, remaining_seconds()),
    )
    marker.write_text(expected + "\\n", encoding="utf-8")


def write_partial_review(env):
    state_path = CAMPAIGN_ROOT / "stage_state.json"
    completed_stage = 0
    if state_path.is_file():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        completed_stage = int(payload.get("completed_stage", 0))
    code = (
        "from experiments.preprocessing_campaign.budgeted_artifacts import write_stage_bundles; "
        f"print(write_stage_bundles(campaign_root={str(CAMPAIGN_ROOT)!r}, "
        f"stage_id={completed_stage}, campaign_id='budgeted_preprocessing_campaign_v1').review)"
    )
    return run_checked([sys.executable, "-c", code], cwd=CODE_ROOT, env=env)


EMBEDDED_RUNTIME_B64 = "__ARCHIVE__"
EMBEDDED_RUNTIME_SHA256 = "__SHA256__"


try:
    if not INPUT_ROOT.is_dir() or not WORKING_ROOT.is_dir():
        raise RuntimeError("This cell must run inside a Kaggle Notebook")
    DATA_DIR = find_official_data()
    print(f"DATA_FOUND path={DATA_DIR}", flush=True)
    extract_embedded_runtime()
    print(f"CODE_READY commit={REQUIRED_CODE_COMMIT}", flush=True)
    install_runtime()

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(RUNTIME_ROOT), str(CODE_ROOT), child_env.get("PYTHONPATH", "")]
    )
    child_env["PYTHONUNBUFFERED"] = "1"
    gpu_probe = (
        "import json, torch; "
        "print(json.dumps([torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]))"
    )
    gpu_result = run_checked([sys.executable, "-c", gpu_probe], env=child_env)
    gpu_names = json.loads(gpu_result.stdout)
    if len(gpu_names) != 2 or any("t4" not in str(name).casefold() for name in gpu_names):
        raise StageNeedsReview(
            f"exactly two T4 devices are required; found={gpu_names}"
        )
    print(f"T4_X2_READY devices={gpu_names}", flush=True)

    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = CAMPAIGN_ROOT / "run.log"
    command = [
        sys.executable,
        "-m",
        "experiments.preprocessing_campaign.run_budgeted_campaign",
        "auto",
        "--config",
        str(CONFIG_PATH),
        "--input-root",
        str(INPUT_ROOT),
        "--data-dir",
        str(DATA_DIR),
        "--output-root",
        str(CAMPAIGN_ROOT),
        "--session-started-unix",
        str(SESSION_STARTED_UNIX),
    ]
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=CODE_ROOT,
            env=child_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode == 2:
        raise StageNeedsReview("campaign requested manual review; inspect the streamed reason")
    if returncode != 0:
        raise RuntimeError(f"budgeted campaign failed returncode={returncode}")
    write_partial_review(child_env)

    final_review = WORKING_ROOT / "preprocessing_campaign_final_review_bundle.zip"
    stage_archives = sorted(WORKING_ROOT.glob("preprocessing_stage_*_resume_bundle.zip"))
    print(f"KAGGLE_STAGE_SUCCESS resume_bundle={stage_archives[-1] if stage_archives else None}", flush=True)
    if final_review.is_file():
        print(f"FINAL_REVIEW_READY path={final_review}", flush=True)
        print("NEXT_ACTION=send only the final review ZIP to Codex", flush=True)
    else:
        print("NEXT_ACTION=Save Version, attach the newest resume bundle as a Dataset, then run this same cell", flush=True)
except StageNeedsReview as error:
    traceback.print_exc()
    try:
        if "child_env" in globals() and CAMPAIGN_ROOT.is_dir():
            write_partial_review(child_env)
    except BaseException:
        traceback.print_exc()
    print(f"STAGE_NEEDS_REVIEW message={str(error).replace(' ', '_')}", flush=True)
    raise
except BaseException as error:
    traceback.print_exc()
    try:
        if "child_env" in globals() and CAMPAIGN_ROOT.is_dir():
            write_partial_review(child_env)
    except BaseException:
        traceback.print_exc()
    print(
        "STAGE_ERROR "
        f"type={type(error).__name__} message={str(error).replace(' ', '_')}",
        flush=True,
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
