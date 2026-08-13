"""Render the GitHub-free, split independent-DL Kaggle notebook."""

from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb"


def _git(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _archive() -> tuple[str, bytes]:
    commit = _git("rev-parse", "HEAD").decode().strip()
    tar_bytes = _git(
        "archive", "--format=tar", commit,
        "competition_rules", "experiments/independent_dl"
    )
    return commit, gzip.compress(tar_bytes, compresslevel=9, mtime=0)


def _markdown(value: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(True)}


def _code(value: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": value.splitlines(True),
    }


def _family_markdown(title: str, family: str, runtime: str, extra: str = "") -> str:
    return f"""## {title}

- **목적:** `{family}` 계열에서 우선순위상 다음 미완료 후보 하나를 전체 2023→2024 데이터로 실행합니다.
- **필수 입력:** `공통 준비` 셀이 `KAGGLE_DL_SETUP_READY`로 끝나야 합니다.
- **예상 시간:** {runtime} P3·P4 용량과 GPU에 따라 더 길어질 수 있습니다.
- **재실행:** 완료 후보는 건너뛰고, 중단 후보는 보존된 epoch checkpoint부터 재개합니다. 같은 셀을 다시 누르면 다음 후보 하나를 실행합니다.
- **정상 완료:** `KAGGLE_DL_CANDIDATE_CHECKPOINTED family={family}`가 출력됩니다.
- **오류 전달:** 첫 원본 traceback부터 마지막 줄까지 생략하지 않고 보내주세요.
- **제출:** 이 셀은 OOF evidence만 만들며 제출 CSV나 ZIP을 생성하지 않습니다.
{extra}"""


def main() -> None:
    commit, archive = _archive()
    encoded = base64.b64encode(archive).decode("ascii")
    digest = sha256(archive).hexdigest()
    setup = r"""from __future__ import annotations

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


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
CAMPAIGN_OUTPUT_DIR = WORKING_ROOT / "independent_dl_campaign_v1"
HANDOFF_DIR = WORKING_ROOT / "codex_handoffs"
RUNTIME_DIR = WORKING_ROOT / "independent_dl_runtime_v2"
REQUIRED_FILES = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "trackman_history.csv",
)
SOURCE_COMMIT = "__COMMIT__"
EMBEDDED_RUNTIME_B64 = "__ARCHIVE__"
EMBEDDED_RUNTIME_SHA256 = "__DIGEST__"
REPO_DIR = WORKING_ROOT / f"independent_dl_embedded_code_{SOURCE_COMMIT[:7]}"
ENVIRONMENT_JSON = CAMPAIGN_OUTPUT_DIR / "environment.json"


def run_checked(command, *, cwd=None, env=None):
    completed = subprocess.run(
        [str(item) for item in command], cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"명령 실패(returncode={completed.returncode}): {command[0]}"
        )
    return completed


if not INPUT_ROOT.is_dir() or not WORKING_ROOT.is_dir():
    raise RuntimeError("이 셀은 Kaggle Notebook에서 실행해야 합니다.")

data_directories = sorted(
    {
        train_path.parent
        for train_path in INPUT_ROOT.rglob("train.csv")
        if train_path.is_file()
        and all((train_path.parent / filename).is_file() for filename in REQUIRED_FILES)
    }
)
if len(data_directories) != 1:
    detail = "\n".join(f"- {path}" for path in data_directories) or "- 발견 없음"
    raise RuntimeError(
        "공식 CSV 네 개가 함께 있는 Dataset 폴더는 정확히 하나여야 합니다.\n" + detail
    )
DATA_DIR = data_directories[0]
print("공식 데이터:", DATA_DIR)

checkpoint_sources = sorted(
    path for path in INPUT_ROOT.rglob("independent_dl_campaign_v1") if path.is_dir()
)
if len(checkpoint_sources) > 1:
    raise RuntimeError(
        "이전 campaign checkpoint Dataset은 하나만 연결하세요.\n"
        + "\n".join(f"- {path}" for path in checkpoint_sources)
    )
if CAMPAIGN_OUTPUT_DIR.exists():
    if not CAMPAIGN_OUTPUT_DIR.is_dir():
        raise RuntimeError(f"결과 경로가 폴더가 아닙니다: {CAMPAIGN_OUTPUT_DIR}")
    print("재개 원본: /kaggle/working")
elif checkpoint_sources:
    shutil.copytree(checkpoint_sources[0], CAMPAIGN_OUTPUT_DIR, dirs_exist_ok=False)
    print("재개 원본:", checkpoint_sources[0])
else:
    CAMPAIGN_OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
    print("재개 원본: 새 캠페인")

archive_bytes = base64.b64decode(EMBEDDED_RUNTIME_B64, validate=True)
if hashlib.sha256(archive_bytes).hexdigest() != EMBEDDED_RUNTIME_SHA256:
    raise RuntimeError("runtime archive SHA-256 mismatch")
runtime_marker = REPO_DIR / ".embedded_runtime_sha256"
if REPO_DIR.exists():
    if not runtime_marker.is_file() or runtime_marker.read_text().strip() != EMBEDDED_RUNTIME_SHA256:
        raise RuntimeError(f"검증되지 않은 임베디드 코드 폴더입니다: {REPO_DIR}")
else:
    REPO_DIR.mkdir(parents=False, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe runtime archive path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"runtime archive links are forbidden: {member.name}")
        bundle.extractall(REPO_DIR)
    runtime_marker.write_text(EMBEDDED_RUNTIME_SHA256 + "\n", encoding="utf-8")

# 첫 실행은 PyPI 설치를 위해 Kaggle Settings의 Internet 옵션이 켜져 있어야 합니다.
requirements = REPO_DIR / "experiments/independent_dl/requirements-colab.txt"
requirements_sha = hashlib.sha256(requirements.read_bytes()).hexdigest()
requirements_marker = RUNTIME_DIR / ".requirements_sha256"
if not requirements_marker.is_file() or requirements_marker.read_text().strip() != requirements_sha:
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    RUNTIME_DIR.mkdir(parents=True)
    run_checked([
        sys.executable, "-m", "pip", "install", "--target", str(RUNTIME_DIR),
        "--upgrade", "--no-deps", "-r", str(requirements),
    ])
    requirements_marker.write_text(requirements_sha + "\n", encoding="utf-8")

child_env = os.environ.copy()
child_env["PYTHONPATH"] = os.pathsep.join(
    [str(RUNTIME_DIR), str(REPO_DIR), child_env.get("PYTHONPATH", "")]
)
child_env["PYTHONUNBUFFERED"] = "1"
CONFIG = REPO_DIR / "experiments/independent_dl/configs/campaign_v1.json"
MANIFEST = CAMPAIGN_OUTPUT_DIR / "campaign_manifest.json"
RESULTS_JSONL = CAMPAIGN_OUTPUT_DIR / "candidate_results.jsonl"
SUMMARY = CAMPAIGN_OUTPUT_DIR / "campaign_summary.json"
FAMILIES = ("tabm", "mlp_resnet", "ft_transformer", "tabr", "tabicl_v2")

environment_code = r'''import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import torch

packages = {}
for name in ("numpy", "pandas", "scipy", "scikit-learn", "tabm", "tabicl"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "vram_gib": round(properties.total_memory / 1024**3, 2),
        })
payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "devices": devices,
    "packages": packages,
}
if not payload["cuda_available"]:
    raise RuntimeError("CUDA GPU가 없습니다. Kaggle Accelerator를 GPU로 바꾸세요.")
Path(__ENVIRONMENT__).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
'''.replace("__ENVIRONMENT__", repr(str(ENVIRONMENT_JSON)))
run_checked([sys.executable, "-c", environment_code], cwd=REPO_DIR, env=child_env)


def _stream(command):
    process = subprocess.Popen(
        [str(item) for item in command], cwd=REPO_DIR, env=child_env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"후보 실행 실패(returncode={returncode}). 첫 traceback부터 보내주세요."
        )


def show_campaign_status():
    if not MANIFEST.is_file():
        print("아직 manifest가 없습니다. 원하는 모델 계열 셀을 실행하세요.")
        return None
    completed = run_checked([
        sys.executable, "-m", "experiments.independent_dl.run_campaign", "status",
        "--output-dir", str(CAMPAIGN_OUTPUT_DIR),
    ], cwd=REPO_DIR, env=child_env)
    payload = json.loads(completed.stdout)
    for family, state in payload["family_status"].items():
        print(
            f"- {family}: 완료={len(state['completed'])}, 실패={len(state['failed'])}, "
            f"남음={state['remaining_count']}, 다음={state['next_candidate']}"
        )
    print("P3·P4, boundary expansion, confirmation 후보도 남은 수에 포함됩니다.")
    return payload


def run_one_family(family):
    if family not in FAMILIES:
        raise ValueError(f"지원하지 않는 모델 계열입니다: {family}")
    before = {}
    if MANIFEST.is_file():
        before_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        before = {
            candidate_id: int(entry.get("attempts", 0))
            for candidate_id, entry in before_payload.get("candidates", {}).items()
        }
    _stream([
        sys.executable, "-m", "experiments.independent_dl.run_campaign", "run",
        "--config", str(CONFIG), "--data-dir", str(DATA_DIR),
        "--output-dir", str(CAMPAIGN_OUTPUT_DIR),
        "--family", family, "--max-candidates", "1",
    ])
    after_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    attempted = [
        (candidate_id, entry)
        for candidate_id, entry in after_payload["candidates"].items()
        if int(entry.get("attempts", 0)) > before.get(candidate_id, 0)
    ]
    if len(attempted) != 1:
        raise RuntimeError(f"이번 셀의 시도 후보를 하나로 확인할 수 없습니다: {attempted}")
    candidate_id, entry = attempted[0]
    if entry.get("state") != "completed":
        raise RuntimeError(
            f"후보 실패: {candidate_id}\n{entry.get('failure_reason') or '원인 미기록'}"
        )
    print(f"KAGGLE_DL_CANDIDATE_CHECKPOINTED family={family}")
    print("완료 후보:", candidate_id)
    show_campaign_status()


def write_handoff(candidate_id):
    if not candidate_id or Path(candidate_id).name != candidate_id:
        raise ValueError("HANDOFF_CANDIDATE_ID에 완료 후보 ID 하나를 정확히 넣으세요.")
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    result = HANDOFF_DIR / f"{candidate_id}_handoff.zip"
    if result.exists():
        print("기존 handoff ZIP을 재사용합니다:", result)
    else:
        run_checked([
            sys.executable, "-m", "experiments.independent_dl.run_campaign", "handoff",
            "--output-dir", str(CAMPAIGN_OUTPUT_DIR),
            "--candidate-id", candidate_id,
            "--result", str(result),
            "--runtime-sha256", EMBEDDED_RUNTIME_SHA256,
            "--requirements", str(requirements),
            "--environment", str(ENVIRONMENT_JSON),
        ], cwd=REPO_DIR, env=child_env)
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    print("HANDOFF_READY")
    print("path:", result)
    print("size_bytes:", result.stat().st_size)
    print("sha256:", digest)
    print("Kaggle Output에서 이 ZIP을 다운로드해 대화에 첨부하세요.")
    return result


print("KAGGLE_DL_SETUP_READY")
print("공식 데이터:", DATA_DIR)
print("캠페인 결과:", CAMPAIGN_OUTPUT_DIR)
print("세션 종료 전에 Save Version으로 /kaggle/working 결과를 보존하세요.")
"""
    setup = (
        setup.replace("__COMMIT__", commit)
        .replace("__ARCHIVE__", encoded)
        .replace("__DIGEST__", digest)
    )
    cells = [
        _markdown("""# Kaggle 독립 DL 캠페인 — 후보별 실행·결과 전달

사진처럼 공식 CSV 네 개가 들어 있는 비공개 Dataset을 연결하고 Accelerator를 GPU로 설정합니다. 첫 실행은 패키지 설치를 위해 Kaggle **Internet 옵션을 켜야 하지만 GitHub 연결이나 토큰은 사용하지 않습니다.**

한 번에 후보 하나만 실행합니다. P1~P4, 네 입력 표현, 최대 400 epoch, boundary expansion과 다중 fold·seed confirmation은 그대로입니다. 세션 종료 전 **Save Version**으로 `/kaggle/working/independent_dl_campaign_v1`을 Output에 보존하세요. 새 세션에서는 이전 Notebook Output을 Input에 연결하면 checkpoint부터 이어갑니다.
"""),
        _markdown("""## 공통 준비

공식 Dataset·이전 checkpoint를 찾고, 임베디드 코드를 검증하며 패키지와 GPU/VRAM 환경을 준비합니다. 이 셀은 학습하지 않습니다. 정상 완료는 `KAGGLE_DL_SETUP_READY`입니다.
"""),
        _code(setup),
        _markdown("""## 현재 상태 확인

GPU를 사용하지 않고 계열별 완료·실패·남은 후보와 다음 후보를 표시합니다. 새 캠페인이면 manifest가 없다는 안내가 정상입니다.
"""),
        _code("show_campaign_status()\n"),
    ]
    sections = [
        ("TabM 후보 1개", "tabm", "수십 분에서 여러 시간입니다."),
        ("MLP/ResNet 후보 1개", "mlp_resnet", "1시간에서 여러 시간입니다."),
        ("FT-Transformer 후보 1개", "ft_transformer", "수 시간 또는 한 세션 가까이 걸릴 수 있습니다."),
        ("TabR 후보 1개", "tabr", "retrieval 비용으로 수 시간 이상 걸릴 수 있습니다."),
        ("TabICLv2 후보 1개", "tabicl_v2", "GPU 메모리와 세션 한도를 넘길 수 있습니다."),
    ]
    for title, family, runtime in sections:
        extra = (
            "- **연구 상태:** 대회 사용 가능성 확인 전까지 `research_only`이며 자동 채택하지 않습니다.\n"
            if family == "tabicl_v2" else ""
        )
        cells.append(_markdown(_family_markdown(title, family, runtime, extra)))
        cells.append(_code(f'run_one_family("{family}")\n'))
    cells.extend(
        [
            _markdown("""## 후보별 handoff ZIP

위 상태에서 `completed`인 후보 ID 하나를 `HANDOFF_CANDIDATE_ID`에 붙여넣고 실행합니다. OOF·metrics·환경·해시만 포함하며 checkpoint, feature cache, 원본 데이터와 제출 파일은 제외합니다. 정상 완료는 `HANDOFF_READY`이며 출력된 ZIP 하나를 이 대화에 첨부하면 됩니다.
"""),
            _code('HANDOFF_CANDIDATE_ID = ""\nHANDOFF_PATH = write_handoff(HANDOFF_CANDIDATE_ID)\n'),
            _markdown("""## 결과 요약

학습 없이 현재 상태와 전달 ZIP 목록을 표시합니다. 후보 비교 후 전체 학습·test 추론·제출 ZIP은 별도 승인된 단계에서만 추가합니다.
"""),
            _code('''show_campaign_status()
print("Manifest:", MANIFEST if MANIFEST.is_file() else "아직 없음")
print("Candidate results:", RESULTS_JSONL if RESULTS_JSONL.is_file() else "아직 없음")
print("Summary:", SUMMARY if SUMMARY.is_file() else "아직 없음")
print("Handoff ZIPs:", sorted(str(path) for path in HANDOFF_DIR.glob("*_handoff.zip")))
'''),
        ]
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"title": "Independent DL Candidate Handoff"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
