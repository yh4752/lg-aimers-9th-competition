# 독립 DL 프런티어 캠페인 Colab Pro 실행

## 실행 안내

- 목적: 전체 `2023→2024` 검증에서 TabM 입력 표현, 대형 ResNet·FT-Transformer,
  TabICLv2와 수정된 TabR를 명시된 순서로 생성·재개한다. Smoke 결과는 성능 근거가
  아니다.
- 필수 입력: Colab 보안 비밀 `GITHUB_TOKEN`, Drive의 `train.csv`와
  `trackman_history.csv`, CUDA GPU 런타임이다.
- 예상 시간: 환경 준비는 수 분, 각 본 후보는 GPU와 모델에 따라 수십 분에서 여러
  시간이 걸릴 수 있으며 전체 캠페인은 여러 세션이 필요하다.
- Colab Pro: 더 빠른 GPU와 고용량 메모리는 가용성에 따라 달라진다. 아래 셀은
  특정 GPU를 요구하지 않고 실제 장치 수, 이름과 VRAM을 출력한다.
- 재실행: 같은 `CAMPAIGN_OUTPUT_DIR`로 셀 전체를 다시 실행하면 기존 checkpoint 재개
  여부와 다음 후보를 먼저 표시하고, 해시가 유효한 완료 후보를 건너뛴다.
- 성공 시: `INDEPENDENT_DL_CAMPAIGN_CHECKPOINTED`, Drive 결과 루트,
  `campaign_manifest.json`, `candidate_results.jsonl`과 존재하는
  `campaign_summary.json` 경로가 출력된다.
- 오류 시: 첫 번째 원본 traceback부터 마지막 오류까지 생략하지 않고 보내면 된다.
- 제출: 이 셀은 제출 CSV나 ZIP을 만들지 않는다. TabICLv2는 대회 사용 가능성이
  별도로 확인될 때까지 `research_only`다.

아래는 하나의 완결된 셀이다. 사용자가 바꿀 값은 `DATA_DIR`와
`CAMPAIGN_OUTPUT_DIR`뿐이다.

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


DATA_DIR = Path("/content/drive/MyDrive/LG_AIMERS_2026/competition/data")
CAMPAIGN_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/LG_AIMERS_2026/outputs/independent_dl_campaign_v1"
)

REPO_URL = "https://github.com/yh4752/lg-aimers-9th-competition.git"
REPO_DIR = Path("/content/lg-aimers-9th-competition")
REQUIRED_CODE_COMMIT = "a38333cd97a965e6a1a49b411f9f17fcffc6c0f0"
RUNTIME_DIR = Path("/content/independent_dl_runtime_v2")


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
    run_checked(
        ["git", "checkout", "--detach", REQUIRED_CODE_COMMIT],
        cwd=REPO_DIR,
        env=git_env,
    )
del token

head = run_checked(["git", "rev-parse", "HEAD"], cwd=REPO_DIR).stdout.strip()
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
            sys.executable, "-m", "pip", "install", "--target",
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

runtime_probe = r'''
import json
import sys
import numpy
import pandas
import tabicl
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU가 없습니다. Colab 런타임 유형에서 GPU를 선택하세요.")
devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    devices.append(
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "vram_gib": round(properties.total_memory / 1024**3, 2),
        }
    )
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("TabICL:", getattr(tabicl, "__version__", "version unavailable"))
print("GPU_COUNT:", torch.cuda.device_count())
print("GPU/VRAM:", json.dumps(devices, ensure_ascii=False))
print("실제 학습 모드: single_gpu, cuda:0")
'''
print("저장소 커밋:", head)
run_checked([sys.executable, "-c", runtime_probe], cwd=REPO_DIR, env=child_env)

config = REPO_DIR / "experiments/independent_dl/configs/campaign_v1.json"
manifest = CAMPAIGN_OUTPUT_DIR / "campaign_manifest.json"
results_jsonl = CAMPAIGN_OUTPUT_DIR / "candidate_results.jsonl"
summary = CAMPAIGN_OUTPUT_DIR / "campaign_summary.json"

if manifest.is_file():
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("candidates", {})
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    priority_ids = [
        candidate_id
        for wave in config_payload["execution_waves"]
        if isinstance(wave["candidate_ids"], list)
        for candidate_id in wave["candidate_ids"]
    ]
    next_candidate = next(
        (
            candidate_id
            for candidate_id in priority_ids
            if entries.get(candidate_id, {}).get("state") == "pending"
        ),
        "우선 파동 완료 후 remaining_grid에서 결정",
    )
    print("실행 상태: 기존 checkpoint 재개")
    print("현재 등록 기준 다음 후보:", next_candidate)
else:
    print("실행 상태: 새 캠페인")
    print("다음 후보: 설정의 첫 미완료 후보")

command = [
    sys.executable,
    "-m",
    "experiments.independent_dl.run_campaign",
    "run",
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
    raise RuntimeError(
        f"독립 DL 캠페인이 실패했습니다(returncode={returncode}). "
        "위의 첫 원본 traceback부터 모두 보내주세요."
    )

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
print("Candidate results:", results_jsonl if results_jsonl.is_file() else "아직 없음")
print("Summary:", summary if summary.is_file() else "아직 없음")
print("완료 후보 수:", len(completed))
print("최근 완료 후보:", completed[-1] if completed else "아직 없음")
print("성공 시 위 경로와 manifest·results·summary 파일을 보내주세요.")
```
