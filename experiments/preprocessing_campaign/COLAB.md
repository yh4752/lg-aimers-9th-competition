# 전처리 성능 캠페인 Colab 실행

- 목적: 8개 중대형 DL 기준점에서 19개 전처리를 다섯 시간 fold로 비교하는 파동 A를
  생성·재개한다. 이후 결과 검토를 거쳐 seed 확인과 조합 실험을 등록한다.
- 필수 입력: Colab 보안 비밀 `GITHUB_TOKEN`, Drive의 `train.csv`와
  `trackman_history.csv`, CUDA GPU 런타임.
- 기본 입력 경로: `/content/drive/MyDrive/LG_AIMERS_2026/competition/data`
- 결과 경로: `/content/drive/MyDrive/LG_AIMERS_2026/outputs/preprocessing_campaign_v1`
- 예상 시간: 첫 기준선 다섯 fold가 끝나기 전에는 신뢰할 수 있는 시간이 없다. 이후
  manifest의 실측 중앙값으로 계산한다. 전체 760회이므로 여러 Colab 세션이 필요할 수 있다.
- 재실행 안전성: 아래 셀 전체를 다시 실행하면 같은 Drive manifest와 파일 해시를 확인해
  완료 작업은 건너뛰고 미완료 작업부터 진행한다.
- 반환할 결과: 성공 시 `campaign_manifest.json`과 결과 폴더의 ZIP, 오류 시 첫 traceback부터
  `PREPROCESSING_CAMPAIGN_ERROR` 줄까지 그대로 보내면 된다.

아래 코드 블록 전체가 복사용 단일 Colab 셀이다.

```python
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback

from google.colab import drive, userdata


DATA_DIR = Path("/content/drive/MyDrive/LG_AIMERS_2026/competition/data")
CAMPAIGN_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/LG_AIMERS_2026/outputs/preprocessing_campaign_v1"
)
REPO_URL = "https://github.com/yh4752/lg-aimers-9th-competition.git"
REPO_DIR = Path("/content/lg-aimers-9th-competition")
REQUIRED_CODE_COMMIT = "0000000000000000000000000000000000000000"
RUNTIME_DIR = Path("/content/preprocessing_runtime_v1")


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
            f"command failed(returncode={completed.returncode}): {command[0]}"
        )
    return completed


try:
    drive.mount("/content/drive")
    drive_root = Path("/content/drive").resolve()
    for label, path in (
        ("DATA_DIR", DATA_DIR),
        ("CAMPAIGN_OUTPUT_DIR", CAMPAIGN_OUTPUT_DIR),
    ):
        resolved = path.resolve()
        if drive_root not in resolved.parents:
            raise RuntimeError(f"{label} must be inside Drive: {resolved}")
    for filename in ("train.csv", "trackman_history.csv"):
        path = DATA_DIR / filename
        if not path.is_file():
            raise RuntimeError(f"required data file is missing: {path}")
    CAMPAIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = userdata.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Colab secret GITHUB_TOKEN is missing")
    with tempfile.TemporaryDirectory(prefix="preprocessing_git_") as temporary_value:
        temporary = Path(temporary_value)
        askpass = temporary / "askpass.sh"
        askpass.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
            "  *Password*) printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        git_env = os.environ.copy()
        git_env.update(
            {
                "GITHUB_TOKEN": token,
                "GIT_ASKPASS": str(askpass),
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        if REPO_DIR.exists():
            if not (REPO_DIR / ".git").is_dir():
                raise RuntimeError(f"existing path is not a Git repository: {REPO_DIR}")
            run_checked(["git", "fetch", "--prune", "origin"], cwd=REPO_DIR, env=git_env)
        else:
            run_checked(["git", "clone", "--no-checkout", REPO_URL, REPO_DIR], env=git_env)
            run_checked(["git", "fetch", "--prune", "origin"], cwd=REPO_DIR, env=git_env)
        run_checked(
            ["git", "checkout", "--detach", REQUIRED_CODE_COMMIT],
            cwd=REPO_DIR,
            env=git_env,
        )
    del token

    head = run_checked(["git", "rev-parse", "HEAD"], cwd=REPO_DIR).stdout.strip()
    if head != REQUIRED_CODE_COMMIT:
        raise RuntimeError(f"code commit differs: {head}")

    requirements = REPO_DIR / "experiments/preprocessing_campaign/requirements-colab.txt"
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
        marker.write_text(requirements_sha + "\n", encoding="utf-8")

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
        raise RuntimeError(f"campaign process failed(returncode={returncode})")

    manifest = CAMPAIGN_OUTPUT_DIR / "campaign_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"campaign manifest was not created: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    completed = sum(
        entry.get("state") == "completed" for entry in payload.get("jobs", {}).values()
    )
    failed = sum(
        entry.get("state") == "failed" for entry in payload.get("jobs", {}).values()
    )
    print("PREPROCESSING_CAMPAIGN_CHECKPOINTED")
    print("output_root=", CAMPAIGN_OUTPUT_DIR)
    print("manifest=", manifest)
    print("completed=", completed, "failed=", failed)
except BaseException as error:
    traceback.print_exc()
    print(
        "PREPROCESSING_CAMPAIGN_ERROR "
        f"type={type(error).__name__} message={str(error).replace(' ', '_')}"
    )
    raise
```
