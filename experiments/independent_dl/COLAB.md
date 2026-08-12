# 독립 DL 캠페인 — 무료 Colab 분할 실행

한 번에 모델 후보 하나만 실행하고 Drive checkpoint에 저장합니다. 코드 변경 범위만 작게 유지했으며 P1~P4, 네 입력 표현, boundary expansion, confirmation fold·seed의 넓고 깊은 탐색 범위는 그대로입니다. 원하는 모델 계열 셀을 골라 실행할 수 있습니다.

## 공통 준비

Drive 연결, 비공개 저장소의 고정 커밋 준비, 패키지 설치, 공식 데이터와 GPU 확인만 수행합니다. 학습은 시작하지 않습니다. 새 런타임에서는 이 셀을 먼저 한 번 실행하세요. 정상 완료 문구는 `INDEPENDENT_DL_SETUP_READY`입니다.

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
REQUIRED_CODE_COMMIT = "a8f0525f19010e0a0f3aff3b6d0051aa1679857a"
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


CONFIG = config
MANIFEST = manifest
RESULTS_JSONL = results_jsonl
SUMMARY = summary
FAMILIES = ("tabm", "mlp_resnet", "ft_transformer", "tabr", "tabicl_v2")


def _stream_campaign(command):
    process = subprocess.Popen(
        [str(item) for item in command],
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
            f"독립 DL 후보 실행이 실패했습니다(returncode={returncode}). "
            "위의 첫 원본 traceback부터 모두 보내주세요."
        )


def show_campaign_status():
    if not MANIFEST.is_file():
        print("아직 campaign_manifest.json이 없습니다.")
        print("원하는 모델 계열 셀을 실행하면 첫 후보와 manifest가 생성됩니다.")
        return None
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.independent_dl.run_campaign",
            "status",
            "--output-dir",
            str(CAMPAIGN_OUTPUT_DIR),
        ],
        cwd=REPO_DIR,
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        print(completed.stderr)
        raise RuntimeError("캠페인 상태 확인에 실패했습니다. 위 오류를 보내주세요.")
    payload = json.loads(completed.stdout)
    print("계열별 현재 상태")
    for family, state in payload["family_status"].items():
        print(
            f"- {family}: 완료={len(state['completed'])}, 실패={len(state['failed'])}, "
            f"남음={state['remaining_count']}, 다음={state['next_candidate']}"
        )
    print("P3·P4, boundary expansion, confirmation 후보도 남은 수에 포함됩니다.")
    return payload


def run_one_family(family: str):
    if family not in FAMILIES:
        raise ValueError(f"지원하지 않는 모델 계열입니다: {family}")
    print(f"{family} 계열에서 다음 미완료 후보 하나를 실행합니다.")
    _stream_campaign(
        [
            sys.executable,
            "-m",
            "experiments.independent_dl.run_campaign",
            "run",
            "--config",
            str(CONFIG),
            "--data-dir",
            str(DATA_DIR),
            "--output-dir",
            str(CAMPAIGN_OUTPUT_DIR),
            "--family",
            family,
            "--max-candidates",
            "1",
        ]
    )
    if not MANIFEST.is_file():
        raise RuntimeError(f"campaign_manifest.json이 생성되지 않았습니다: {MANIFEST}")
    print(f"INDEPENDENT_DL_CANDIDATE_CHECKPOINTED family={family}")
    print("결과 루트:", CAMPAIGN_OUTPUT_DIR)
    print("Manifest:", MANIFEST)
    print("Candidate results:", RESULTS_JSONL if RESULTS_JSONL.is_file() else "아직 없음")
    show_campaign_status()


print("INDEPENDENT_DL_SETUP_READY")
print("결과 루트:", CAMPAIGN_OUTPUT_DIR)
print("다음으로 '현재 상태 확인' 셀을 실행하세요.")
```

## 현재 상태 확인

GPU 학습 없이 기존 Drive manifest를 읽어 계열별 완료·실패·남은 후보와 다음 후보를 표시합니다. P3·P4, boundary expansion과 confirmation 후보도 전체 잔여 수에서 숨기지 않습니다. 새 캠페인이라 manifest가 없으면 원하는 모델 실행 셀로 이동하면 됩니다.

```python
show_campaign_status()
```

## TabM 후보 1개

- **목적:** `tabm` 계열에서 캠페인 우선순위상 다음 미완료 후보 하나만 전체 2023→2024 데이터로 실행합니다.
- **필수 입력:** 위의 `공통 준비` 셀이 `INDEPENDENT_DL_SETUP_READY`로 끝나야 하며 Drive 데이터와 결과 폴더가 연결되어 있어야 합니다.
- **예상 시간:** 대략 수십 분에서 여러 시간이 걸릴 수 있습니다. GPU 종류, feature view, P3·P4 용량과 checkpoint 위치에 따라 달라집니다.
- **재실행:** 완료 후보는 건너뛰며, 중단된 후보는 Drive의 epoch checkpoint부터 재개합니다. 같은 셀을 다시 실행하면 해당 계열의 다음 후보 하나로 이동합니다.
- **정상 완료:** `INDEPENDENT_DL_CANDIDATE_CHECKPOINTED family=tabm`와 결과 경로가 출력됩니다.
- **오류 전달:** 첫 번째 원본 traceback부터 마지막 오류 문구까지 생략하지 말고 보내주세요.
- **제출:** 이 셀은 성능 evidence만 만들며 제출 CSV나 ZIP을 생성하지 않습니다.

```python
run_one_family("tabm")
```

## MLP/ResNet 후보 1개

- **목적:** `mlp_resnet` 계열에서 캠페인 우선순위상 다음 미완료 후보 하나만 전체 2023→2024 데이터로 실행합니다.
- **필수 입력:** 위의 `공통 준비` 셀이 `INDEPENDENT_DL_SETUP_READY`로 끝나야 하며 Drive 데이터와 결과 폴더가 연결되어 있어야 합니다.
- **예상 시간:** 대략 1~수 시간이 걸릴 수 있으며 P4는 더 오래 걸릴 수 있습니다. GPU 종류, feature view, P3·P4 용량과 checkpoint 위치에 따라 달라집니다.
- **재실행:** 완료 후보는 건너뛰며, 중단된 후보는 Drive의 epoch checkpoint부터 재개합니다. 같은 셀을 다시 실행하면 해당 계열의 다음 후보 하나로 이동합니다.
- **정상 완료:** `INDEPENDENT_DL_CANDIDATE_CHECKPOINTED family=mlp_resnet`와 결과 경로가 출력됩니다.
- **오류 전달:** 첫 번째 원본 traceback부터 마지막 오류 문구까지 생략하지 말고 보내주세요.
- **제출:** 이 셀은 성능 evidence만 만들며 제출 CSV나 ZIP을 생성하지 않습니다.

```python
run_one_family("mlp_resnet")
```

## FT-Transformer 후보 1개

- **목적:** `ft_transformer` 계열에서 캠페인 우선순위상 다음 미완료 후보 하나만 전체 2023→2024 데이터로 실행합니다.
- **필수 입력:** 위의 `공통 준비` 셀이 `INDEPENDENT_DL_SETUP_READY`로 끝나야 하며 Drive 데이터와 결과 폴더가 연결되어 있어야 합니다.
- **예상 시간:** 대략 수 시간에서 한 세션 가까이 걸릴 수 있습니다. GPU 종류, feature view, P3·P4 용량과 checkpoint 위치에 따라 달라집니다.
- **재실행:** 완료 후보는 건너뛰며, 중단된 후보는 Drive의 epoch checkpoint부터 재개합니다. 같은 셀을 다시 실행하면 해당 계열의 다음 후보 하나로 이동합니다.
- **정상 완료:** `INDEPENDENT_DL_CANDIDATE_CHECKPOINTED family=ft_transformer`와 결과 경로가 출력됩니다.
- **오류 전달:** 첫 번째 원본 traceback부터 마지막 오류 문구까지 생략하지 말고 보내주세요.
- **제출:** 이 셀은 성능 evidence만 만들며 제출 CSV나 ZIP을 생성하지 않습니다.

```python
run_one_family("ft_transformer")
```

## TabR 후보 1개

- **목적:** `tabr` 계열에서 캠페인 우선순위상 다음 미완료 후보 하나만 전체 2023→2024 데이터로 실행합니다.
- **필수 입력:** 위의 `공통 준비` 셀이 `INDEPENDENT_DL_SETUP_READY`로 끝나야 하며 Drive 데이터와 결과 폴더가 연결되어 있어야 합니다.
- **예상 시간:** retrieval 비용 때문에 수 시간 이상 또는 세션을 넘길 수 있습니다. GPU 종류, feature view, P3·P4 용량과 checkpoint 위치에 따라 달라집니다.
- **재실행:** 완료 후보는 건너뛰며, 중단된 후보는 Drive의 epoch checkpoint부터 재개합니다. 같은 셀을 다시 실행하면 해당 계열의 다음 후보 하나로 이동합니다.
- **정상 완료:** `INDEPENDENT_DL_CANDIDATE_CHECKPOINTED family=tabr`와 결과 경로가 출력됩니다.
- **오류 전달:** 첫 번째 원본 traceback부터 마지막 오류 문구까지 생략하지 말고 보내주세요.
- **제출:** 이 셀은 성능 evidence만 만들며 제출 CSV나 ZIP을 생성하지 않습니다.

```python
run_one_family("tabr")
```

## TabICLv2 후보 1개

- **목적:** `tabicl_v2` 계열에서 캠페인 우선순위상 다음 미완료 후보 하나만 전체 2023→2024 데이터로 실행합니다.
- **필수 입력:** 위의 `공통 준비` 셀이 `INDEPENDENT_DL_SETUP_READY`로 끝나야 하며 Drive 데이터와 결과 폴더가 연결되어 있어야 합니다.
- **예상 시간:** T4 메모리나 세션 한도를 넘길 수 있습니다. GPU 종류, feature view, P3·P4 용량과 checkpoint 위치에 따라 달라집니다.
- **재실행:** 완료 후보는 건너뛰며, 중단된 후보는 Drive의 epoch checkpoint부터 재개합니다. 같은 셀을 다시 실행하면 해당 계열의 다음 후보 하나로 이동합니다.
- **정상 완료:** `INDEPENDENT_DL_CANDIDATE_CHECKPOINTED family=tabicl_v2`와 결과 경로가 출력됩니다.
- **오류 전달:** 첫 번째 원본 traceback부터 마지막 오류 문구까지 생략하지 말고 보내주세요.
- **제출:** 이 셀은 성능 evidence만 만들며 제출 CSV나 ZIP을 생성하지 않습니다.
- **연구 상태:** TabICLv2는 대회 사용 가능성 확인 전까지 `research_only`이며 자동 채택하지 않습니다.

```python
run_one_family("tabicl_v2")
```

## 결과 요약

GPU 학습 없이 현재 계열별 상태와 `campaign_summary.json`, `candidate_results.jsonl` 경로를 확인합니다. 아직 모든 후보가 끝나지 않았으면 P3·P4, boundary expansion, confirmation까지 남은 수가 표시됩니다. 이 셀도 제출 파일을 만들지 않습니다.

```python
status_payload = show_campaign_status()
print("Manifest:", MANIFEST if MANIFEST.is_file() else "아직 없음")
print("Candidate results:", RESULTS_JSONL if RESULTS_JSONL.is_file() else "아직 없음")
print("Summary:", SUMMARY if SUMMARY.is_file() else "아직 없음")
print("위 상태 출력과 생성된 경로를 보내주세요.")
```
