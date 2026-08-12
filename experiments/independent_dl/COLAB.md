# 독립 DL 캠페인 Colab 실행

## 실행 안내

- 목적: 검증된 시간 cutoff를 유지하면서 TabM/TabMmini, MLP/ResNet,
  FT-Transformer, TabR의 64개 초기 후보와 이후 확장 후보를 T4에서 생성·재개한다.
- 필수 입력: Colab 보안 비밀 `GITHUB_TOKEN`, Drive의 `train.csv`와
  `trackman_history.csv`, GPU 런타임(T4).
- 예상 시간: 패키지 준비는 보통 수 분, 첫 64개 후보는 여러 Colab 세션과 수일이
  걸릴 수 있다. 오래 걸리는 것은 후보 기각 사유가 아니다.
- 재실행: 같은 `CAMPAIGN_OUTPUT_DIR`로 아래 셀 전체를 다시 실행하면 완료 artifact의
  해시를 확인해 건너뛰고, 중단 후보는 마지막 유효 checkpoint부터 이어간다.
- 성공 시: `INDEPENDENT_DL_CAMPAIGN_CHECKPOINTED`와 Drive 결과 경로가 출력된다.
  `campaign_manifest.json` 및 존재할 때 `campaign_summary.json`을 보내면 된다.
- 오류 시: 셀에 출력된 첫 traceback부터 마지막 줄까지 생략하지 말고 보내면 된다.

아래는 하나의 완결된 셀이다. 저장소 코드나 노트북을 수정하지 않는다.

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

from google.colab import drive, userdata


# 사용자 설정: 기존 Drive 구조가 다르면 이 두 경로만 바꾸세요.
DATA_DIR = Path("/content/drive/MyDrive/LG_AIMERS_2026/competition/data")
CAMPAIGN_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/LG_AIMERS_2026/outputs/independent_dl_campaign_v1"
)

REPO_URL = "https://github.com/yh4752/lg-aimers-9th-competition.git"
REPO_DIR = Path("/content/lg-aimers-9th-competition")
REQUIRED_CODE_COMMIT = "c36811956e632a00f775525f83bcb666ff2ec9d6"
RUNTIME_DIR = Path("/content/independent_dl_runtime_v1")


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
            f"명령 실패(returncode={completed.returncode}): {command[0]}"
        )
    return completed


drive.mount("/content/drive")
drive_root = Path("/content/drive").resolve()
for label, path in (("DATA_DIR", DATA_DIR), ("CAMPAIGN_OUTPUT_DIR", CAMPAIGN_OUTPUT_DIR)):
    resolved = path.resolve()
    if drive_root not in resolved.parents:
        raise RuntimeError(f"{label}은 Drive 안에 있어야 합니다: {resolved}")
for filename in ("train.csv", "trackman_history.csv"):
    path = DATA_DIR / filename
    if not path.is_file():
        raise RuntimeError(f"필수 데이터 파일이 없습니다: {path}")
CAMPAIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

token = userdata.get("GITHUB_TOKEN")
if not token:
    raise RuntimeError("Colab 보안 비밀 GITHUB_TOKEN이 없습니다.")

with tempfile.TemporaryDirectory(prefix="independent_dl_git_") as temporary_value:
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
            raise RuntimeError(f"기존 경로가 Git 저장소가 아닙니다: {REPO_DIR}")
        run_checked(["git", "fetch", "--prune", "origin"], cwd=REPO_DIR, env=git_env)
    else:
        run_checked(["git", "clone", "--no-checkout", REPO_URL, REPO_DIR], env=git_env)
        run_checked(["git", "fetch", "--prune", "origin"], cwd=REPO_DIR, env=git_env)
    run_checked(["git", "checkout", "--detach", REQUIRED_CODE_COMMIT], cwd=REPO_DIR, env=git_env)
del token

head = run_checked(
    ["git", "rev-parse", "HEAD"], cwd=REPO_DIR
).stdout.strip()
if head != REQUIRED_CODE_COMMIT:
    raise RuntimeError(f"코드 커밋 불일치: {head}")

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
            "pip", "install", "--target", str(RUNTIME_DIR),
            "--upgrade", "--no-deps", "-r", str(requirements),
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
    "assert torch.cuda.is_available(), 'CUDA GPU가 없습니다'; "
    "name=torch.cuda.get_device_name(0); "
    "print('GPU:', name); "
    "assert 'T4' in name, f'T4 런타임이 아닙니다: {name}'"
)
run_checked([sys.executable, "-c", gpu_check], cwd=REPO_DIR, env=child_env)

config = REPO_DIR / "experiments/independent_dl/configs/campaign_v1.json"
command = [
    sys.executable,
    "-m", "experiments.independent_dl.run_campaign",
    "run",
    "--config", str(config),
    "--data-dir", str(DATA_DIR),
    "--output-dir", str(CAMPAIGN_OUTPUT_DIR),
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
    raise RuntimeError(
        f"독립 DL 캠페인 자식 프로세스가 실패했습니다(returncode={returncode})."
    )

manifest = CAMPAIGN_OUTPUT_DIR / "campaign_manifest.json"
summary = CAMPAIGN_OUTPUT_DIR / "campaign_summary.json"
if not manifest.is_file():
    raise RuntimeError(f"campaign_manifest.json이 생성되지 않았습니다: {manifest}")
payload = json.loads(manifest.read_text(encoding="utf-8"))
completed = [
    candidate_id
    for candidate_id, entry in payload["candidates"].items()
    if entry["state"] == "completed"
]
print("INDEPENDENT_DL_CAMPAIGN_CHECKPOINTED")
print("결과 루트:", CAMPAIGN_OUTPUT_DIR)
print("Manifest:", manifest)
print("Summary:", summary if summary.is_file() else "아직 없음")
print("완료 후보 수:", len(completed))
print("최근 완료 후보:", completed[-1] if completed else "아직 없음")
print(
    "상태 확인 명령:",
    f"{sys.executable} -m experiments.independent_dl.run_campaign status "
    f"--output-dir {CAMPAIGN_OUTPUT_DIR}",
)
```
