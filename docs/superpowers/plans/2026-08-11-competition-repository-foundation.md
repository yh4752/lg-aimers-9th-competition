# Competition Repository Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 새 대회 전용 저장소에 루트 가드레일, 현재 실험 계약, 사람이 읽는 라운드 기록과 검증 가능한 작은 판정 근거를 이전한다.

**Architecture:** 첫 단계는 문서와 작은 evidence만 이전하며 모델 코드나 Colab 경로를 바꾸지 않는다. 빠른 pytest 계약이 문서 링크, 필수 가드레일, JSON 상태·SHA와 비밀 정보 부재를 검사한다. R9, calibration과 XGBoost 코드 이전은 이 기반이 승인된 뒤 각각 별도 계획으로 수행한다.

**Tech Stack:** Markdown, JSON, Python 3.11+, pytest 8, Git

---

## 파일 책임

- `README.md`: 저장소 첫 화면과 현재 실험 상태 요약
- `AGENTS.md`: 저장소 전체에 적용되는 자동 실행 경계
- `.gitignore`: 데이터, 모델, OOF, ZIP, 비밀과 로컬 환경 제외
- `pyproject.toml`: 최소 Python/pytest 설정
- `docs/EXPERIMENT_CONTRACT.md`: 현재 운영 계약의 단일 진실원본
- `docs/ROADMAP.md`: 다음 실험 순서와 열린 모델 계열
- `docs/rounds/*.md`: 완료된 의사결정 단위의 사람용 기록
- `reports/EXPERIMENT_LEDGER.md`: 완료 실험의 시간순 색인
- `reports/acceptances/*.json`: 검증된 기반 또는 탐색 수용 근거
- `reports/rejections/*.json`: 후보 기각 근거
- `reports/diagnostics/*.json`: 패키지 수용이 아닌 종료·분석 근거
- `tests/test_repository_contract.py`: 구조, 문서, evidence와 보안 계약

## 범위 제한

이 계획은 저장소 기반과 기록만 구현한다. `src/`, `experiments/`, `notebooks/`의
모델 코드는 생성하거나 복사하지 않는다. 후속 작업은 아래 세 계획으로 분리한다.

1. R9 기반·공통 검증 코드 이전
2. calibration blending 코드 이전
3. XGBoost v3·original preprocessing rescue 코드와 사용자 Colab 이전

### Task 1: 최소 저장소 계약과 테스트 환경

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `tests/test_repository_contract.py`

- [ ] **Step 1: 저장소 구조 실패 테스트 작성**

`tests/test_repository_contract.py`에 다음 기반을 작성한다.

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_json(path: str) -> dict[str, object]:
    with (ROOT / path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_required_root_files_exist() -> None:
    for relative in (
        "README.md",
        "AGENTS.md",
        "docs/EXPERIMENT_CONTRACT.md",
        "docs/ROADMAP.md",
        "reports/EXPERIMENT_LEDGER.md",
    ):
        assert (ROOT / relative).is_file(), relative


def test_gitignore_blocks_large_and_secret_inputs() -> None:
    text = read_text(".gitignore")
    for pattern in (
        "data/",
        "artifacts/",
        "*.npy",
        "*.zip",
        ".env",
        "*.pem",
        ".venv/",
        "__pycache__/",
    ):
        assert pattern in text
```

- [ ] **Step 2: 테스트가 구조 부재로 실패하는지 확인**

Run: `python3 -m pytest tests/test_repository_contract.py -q`

Expected: `README.md` 또는 `.gitignore` 부재로 FAIL.

- [ ] **Step 3: 최소 설정 파일 작성**

`.gitignore`는 다음 내용으로 만든다.

```gitignore
.DS_Store
.env
.env.*
!.env.example
*.pem
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
data/
artifacts/
outputs/
submissions/
*.npy
*.npz
*.parquet
*.pkl
*.cbm
*.pt
*.pth
*.zip
```

`pyproject.toml`은 다음 최소 설정으로 만든다.

```toml
[project]
name = "lg-aimers-9th-competition"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
dev = ["pytest==8.4.1"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 4: 현재 예상 실패 경계를 확인**

Run: `python3 -m pytest tests/test_repository_contract.py -q`

Expected: gitignore 테스트는 PASS하고 아직 작성하지 않은 루트 문서 테스트만 FAIL.

- [ ] **Step 5: 커밋**

```bash
git add .gitignore pyproject.toml tests/test_repository_contract.py
git commit -m "chore: define competition repository contract"
```

### Task 2: 루트 가드레일과 현재 실험 계약

**Files:**
- Create: `AGENTS.md`
- Create: `docs/EXPERIMENT_CONTRACT.md`
- Modify: `tests/test_repository_contract.py`

- [ ] **Step 1: 필수 운영 경계 실패 테스트 추가**

```python
def test_agents_enforces_execution_ownership_and_mutation_boundaries() -> None:
    agents = read_text("AGENTS.md")
    required = (
        "Codex는 코드",
        "전체 데이터",
        "사용자가 수행",
        "비용만으로 후보를 제외하지 않는다",
        "사용자 요청 없이 push하지 않는다",
        "노트북",
        "패키징",
        "제출",
    )
    for phrase in required:
        assert phrase in agents


def test_experiment_contract_preserves_temporal_and_package_gates() -> None:
    contract = read_text("docs/EXPERIMENT_CONTRACT.md")
    required = (
        "planned → code_ready → waiting_for_user_run → passed → package_ready",
        "rejected",
        "failed",
        "시간 전이",
        "행 순서",
        "SHA-256",
        "acceptance",
        "제출 패키지를 만들지 않는다",
    )
    for phrase in required:
        assert phrase in contract
```

- [ ] **Step 2: 테스트가 두 문서 부재로 실패하는지 확인**

Run: `python3 -m pytest tests/test_repository_contract.py -k 'agents or experiment_contract' -q`

Expected: 두 테스트 모두 파일 부재로 FAIL.

- [ ] **Step 3: `AGENTS.md` 작성**

다음 규칙을 짧은 목록으로 정확히 포함한다.

```markdown
# Competition execution rules

- 최우선 목표는 누출 없는 검증으로 확인한 성능 향상이다.
- 비용은 실행 순서와 자원 안내에만 사용하며 비용만으로 후보를 제외하지 않는다.
- Codex는 코드 작성·검토, 정적 검사와 작은 합성 테스트를 담당한다.
- 공식 전체 데이터 전처리, 시간 전이 OOF, CPU·GPU 학습, Colab 장시간 실행과 제출 평가는 사용자가 수행한다.
- 사용자 요청 없이 push하지 않는다. 노트북 변경·재생성, 패키징과 제출도 각각 별도 요청이 필요하다.
- acceptance와 현재 산출물 해시가 통과하기 전에는 제출 패키지를 만들지 않는다.
- 완료된 실험과 판정은 `reports/EXPERIMENT_LEDGER.md`에 기록한다.
- 변경 전에 `docs/EXPERIMENT_CONTRACT.md`와 `docs/ROADMAP.md`를 읽는다.
```

- [ ] **Step 4: `docs/EXPERIMENT_CONTRACT.md` 작성**

설계 문서의 고정 원칙, 아래 상태 머신, 역할, evidence 스키마와 패키지 게이트를
현재 시제로 정리한다.

```text
planned → code_ready → waiting_for_user_run → passed → package_ready
                                         ↘ rejected
                                         ↘ failed
```

후보 evidence는 후보·실험·모델 ID, 전체/fold/필수 segment 지표, boolean gate와
`data_preflight`, `predictions`, `training_code`, `config` SHA-256을 가져야 한다.
검증 및 테스트 행은 같은 평가 배치의 다른 행과 행 순서에 의존하지 않으며,
수용 전에는 제출 패키지를 만들지 않는다고 명시한다.

- [ ] **Step 5: 집중 테스트 실행**

Run: `python3 -m pytest tests/test_repository_contract.py -k 'agents or experiment_contract' -q`

Expected: `2 passed`.

- [ ] **Step 6: 커밋**

```bash
git add AGENTS.md docs/EXPERIMENT_CONTRACT.md tests/test_repository_contract.py
git commit -m "docs: establish experiment execution contract"
```

### Task 3: 기존의 검증된 작은 evidence 이전

**Files:**
- Create: `reports/acceptances/core_acceptance.json`
- Create: `reports/acceptances/round9_temporal_oof_acceptance.json`
- Create: `reports/acceptances/anchor_brier_audit_acceptance.json`
- Create: `reports/rejections/fwfm_standalone_rejection.json`
- Create: `reports/rejections/r9_fwfm_game_type_f_blend_rejection.json`
- Create: `reports/rejections/r9_fwfm_game_type_f_blend_w080_rejection.json`
- Create: `reports/rejections/tabm_residual_rejection.json`
- Create: `reports/diagnostics/fwfm_failure_boundary_exit_audit.json`
- Modify: `tests/test_repository_contract.py`

- [ ] **Step 1: 고정 원본 해시와 상태 실패 테스트 추가**

```python
IMMUTABLE_REPORTS = {
    "reports/acceptances/core_acceptance.json": ("0dc7bce3a730523a3ce7ffd5f9f6c1c90d1694563b057c827d6aadf4d578c471", "passed"),
    "reports/acceptances/round9_temporal_oof_acceptance.json": ("e56f87dd8800e8a4d899e3d8cdb5830c1e961d74e1298a60505570e5d5e3621a", "passed"),
    "reports/acceptances/anchor_brier_audit_acceptance.json": ("9ccffd3fe983a0a125efed0822ba1722a2887a1505bb424ea02ed3e6a1283e0f", "passed"),
    "reports/rejections/fwfm_standalone_rejection.json": ("e9066396ca6db12dd69f823bc6610cb457653d5859317073aed88c3f43d52606", "rejected"),
    "reports/rejections/r9_fwfm_game_type_f_blend_rejection.json": ("0ae787f74a798077e136a43902d95b96f00b3dc67008eb00d138699ed3b68e66", "rejected"),
    "reports/rejections/r9_fwfm_game_type_f_blend_w080_rejection.json": ("43499ceb6b447e6f6acd5a1d5cf9c8bfe4aba46cd233be097e295c011b302332", "rejected"),
    "reports/rejections/tabm_residual_rejection.json": ("30839509418e0ae46d7e0acf426b98d25280ab1f0a246a2d77a8c7042eed427d", "rejected"),
    "reports/diagnostics/fwfm_failure_boundary_exit_audit.json": ("e0d8fdbefcba40ad77f1272aef819e83b992ee6bd4aea1b44ec8cfec3c32ca34", "read_only_diagnostic"),
}


def test_immutable_reports_match_original_bytes_and_status() -> None:
    for path, (expected_hash, expected_status) in IMMUTABLE_REPORTS.items():
        assert sha256(path) == expected_hash, path
        assert load_json(path)["status"] == expected_status, path
```

- [ ] **Step 2: 테스트가 모든 evidence 부재로 실패하는지 확인**

Run: `python3 -m pytest tests/test_repository_contract.py -k immutable_reports -q`

Expected: 첫 evidence 파일 부재로 FAIL.

- [ ] **Step 3: 원본 바이트를 변경하지 않고 선별 복사**

원본은 `/path/to/legacy-lg-aimers-workspace/reports/kyh/high_score/`이다.
각 원본 내용을 읽고 `apply_patch`로 위 대상 경로에 동일한 UTF-8 바이트를 만든다.
키 순서, 공백, 개행을 정리하지 않는다. 원본 저장소 파일은 수정하지 않는다.

- [ ] **Step 4: 해시·상태 테스트 실행**

Run: `python3 -m pytest tests/test_repository_contract.py -k immutable_reports -q`

Expected: `1 passed`.

- [ ] **Step 5: 커밋**

```bash
git add reports tests/test_repository_contract.py
git commit -m "docs: preserve verified experiment evidence"
```

### Task 4: Calibration과 XGBoost 실행 결과 정규화

**Files:**
- Create: `reports/rejections/calibration_blending_rejection.json`
- Create: `reports/acceptances/xgboost_v3_exploratory_acceptance.json`
- Create: `reports/acceptances/xgboost_original_preproc_rescue_acceptance.json`
- Modify: `tests/test_repository_contract.py`

- [ ] **Step 1: 핵심 실행 사실 실패 테스트 추가**

```python
def test_calibration_rejection_records_open_family() -> None:
    report = load_json("reports/rejections/calibration_blending_rejection.json")
    assert report["status"] == "rejected"
    assert report["candidate_id"] == "calibration_blending_exact_variants"
    assert report["family_closed"] is False
    assert report["best_candidate"] == "game_type_temperature"
    assert report["best_global_brier"] == 0.24789890676984772
    assert report["failed_gates"] == [
        "global_calibration_gap_not_higher",
        "max_fold_delta",
    ]


def test_xgboost_v3_exploratory_acceptance_is_not_public_score() -> None:
    report = load_json("reports/acceptances/xgboost_v3_exploratory_acceptance.json")
    assert report["status"] == "accepted_for_exploratory_submission"
    assert report["run_id"] == "f224a534242a41fea3b88218086e795c"
    assert report["mean_temporal_brier"] == 0.24701737756648098
    assert report["final_2024_brier"] == 0.248274358430683
    assert report["public_score"] is None


def test_xgboost_rescue_is_technically_verified_not_submitted() -> None:
    report = load_json("reports/acceptances/xgboost_original_preproc_rescue_acceptance.json")
    assert report["status"] == "verified_ready"
    assert report["run_id"] == "7c3820cb6b904193b9cf337ad602dff1"
    assert report["selected_candidate"] == "lossguide_l31"
    assert report["final_2024_brier"] == 0.24826687414041645
    assert report["archive_sha256"] == "075e38e355462953543da4532b658568898a6458c6d30e604d5fcbb6b4772006"
    assert report["public_score"] is None
```

- [ ] **Step 2: 세 보고서 부재로 실패하는지 확인**

Run: `python3 -m pytest tests/test_repository_contract.py -k 'calibration_rejection or xgboost' -q`

Expected: `3 failed` with missing report paths.

- [ ] **Step 3: Calibration 판정 보고서 작성**

`calibration_blending_audit.json`의 source artifact SHA를 기록하고 세 후보 모두
`eligible: false`임을 보존한다. 최선 후보의 전체 Brier, F Brier
`0.2474564827036286`, worst fold delta `0.00012750529970739777`와 두 실패 gate를
기록한다. 정확한 세 변형만 기각됐고 calibration family는 열려 있다고 명시한다.

- [ ] **Step 4: XGBoost v3 판정 보고서 작성**

Drive의 `decision_report.json`, `notebook_run_receipt.json`, `selected_config.json`,
`final_2024.json`을 근거로 선택 후보
`history_only__recent_medium__lossguide_l31__seed_mean__identity`, 4-fold 평균,
2024 holdout, run ID, prediction SHA와 receipt artifact SHA를 기록한다. 공식 Public
점수는 없으므로 `public_score`는 JSON null로 둔다.

- [ ] **Step 5: XGBoost rescue 판정 보고서 작성**

Drive의 `comparison_report.json`과 `notebook_run_receipt.json`을 근거로 선택 후보,
2024 Brier, best iteration 236, full fit 237 rounds, archive SHA와 7개 technical gate
PASS를 기록한다. `verified_ready`는 제출 완료 또는 Public 성능 수용으로 표현하지
않는다.

- [ ] **Step 6: 집중 테스트 실행**

Run: `python3 -m pytest tests/test_repository_contract.py -k 'calibration_rejection or xgboost' -q`

Expected: `3 passed`.

- [ ] **Step 7: 커밋**

```bash
git add reports tests/test_repository_contract.py
git commit -m "docs: record calibration and xgboost decisions"
```

### Task 5: 사람용 실험 장부와 라운드 기록

**Files:**
- Create: `reports/EXPERIMENT_LEDGER.md`
- Create: `docs/rounds/README.md`
- Create: `docs/rounds/01-r9-foundation.md`
- Create: `docs/rounds/02-fwfm.md`
- Create: `docs/rounds/03-tabm-residual.md`
- Create: `docs/rounds/04-calibration.md`
- Create: `docs/rounds/05-xgboost.md`
- Modify: `tests/test_repository_contract.py`

- [ ] **Step 1: 장부 완전성과 링크 실패 테스트 추가**

```python
def test_ledger_records_every_completed_family_and_protocol() -> None:
    ledger = read_text("reports/EXPERIMENT_LEDGER.md")
    for experiment in (
        "catboost_smooth_v1",
        "round9_temporal_oof",
        "fwfm_standalone",
        "tabm_residual",
        "calibration_blending",
        "xgboost_score_push_v3",
        "xgboost_original_preproc_rescue",
    ):
        assert experiment in ledger
    assert "검증 프로토콜" in ledger
    assert "Public" in ledger


def test_round_index_links_to_all_round_documents() -> None:
    index = read_text("docs/rounds/README.md")
    for name in (
        "01-r9-foundation.md",
        "02-fwfm.md",
        "03-tabm-residual.md",
        "04-calibration.md",
        "05-xgboost.md",
    ):
        assert f"]({name})" in index
        assert (ROOT / "docs/rounds" / name).is_file()
```

- [ ] **Step 2: 테스트가 문서 부재로 실패하는지 확인**

Run: `python3 -m pytest tests/test_repository_contract.py -k 'ledger or round_index' -q`

Expected: 두 테스트 모두 FAIL.

- [ ] **Step 3: 시간순 장부 작성**

기존 `reports/kyh/high_score/EXPERIMENT_LEDGER.md`의 10개 완료 기록을 보존하고
calibration, XGBoost v3, rescue를 추가한다. 각 행은 실험 ID, 모델 계열, 검증
프로토콜, 핵심 결과, 판정, Public 점수와 Git evidence 링크를 별도 열로 둔다.
프로토콜이 다른 R9 3-fold와 XGBoost v3 4-fold를 순위로 정렬하지 않는다.

- [ ] **Step 4: 라운드 기록 작성**

각 라운드 문서는 아래 동일한 제목만 사용한다.

```markdown
## 가설
## 검증 프로토콜
## 결과
## 판정
## 배운 점
## 다음 결정
## 근거
```

FwFM과 TabM은 기각 이유와 독립 후보를 막지 않는다는 점을 기록한다. Calibration은
정확한 세 변형은 기각하지만 family는 닫지 않는다. XGBoost 문서는 v3의 탐색 수용과
rescue의 technical ready를 Public 점수와 구분한다. 동료 R9 정보는
`castle9612/lg_aimers_9th`의 README 및 관련 round 보고서를 출처로 표시한다.

- [ ] **Step 5: 집중 테스트 실행**

Run: `python3 -m pytest tests/test_repository_contract.py -k 'ledger or round_index' -q`

Expected: `2 passed`.

- [ ] **Step 6: 커밋**

```bash
git add reports/EXPERIMENT_LEDGER.md docs/rounds tests/test_repository_contract.py
git commit -m "docs: publish competition experiment journal"
```

### Task 6: README 대시보드와 다음 로드맵

**Files:**
- Create: `README.md`
- Create: `docs/ROADMAP.md`
- Modify: `tests/test_repository_contract.py`

- [ ] **Step 1: 첫 화면과 로드맵 실패 테스트 추가**

```python
def test_readme_is_a_competition_dashboard() -> None:
    readme = read_text("README.md")
    for phrase in (
        "LG Aimers 9th",
        "현재 기준선",
        "실험 장부",
        "검증 프로토콜",
        "다음 후보",
        "Google Drive",
    ):
        assert phrase in readme
    assert "31개 LG Aimers VOD" not in readme


def test_roadmap_keeps_cost_and_independent_candidates_open() -> None:
    roadmap = read_text("docs/ROADMAP.md")
    assert "비용만으로" in roadmap
    assert "독립 후보" in roadmap
    assert "동료 저장소" in roadmap
    assert "calibration" in roadmap
    assert "XGBoost" in roadmap
```

- [ ] **Step 2: 테스트가 두 문서 부재로 실패하는지 확인**

Run: `python3 -m pytest tests/test_repository_contract.py -k 'readme or roadmap' -q`

Expected: 두 테스트 모두 FAIL.

- [ ] **Step 3: README 작성**

README는 문제·평가 지표, 현재 검증 기준 R9, 완료된 계열 판정, XGBoost 탐색 결과,
현재 열린 후보와 문서 링크를 한 화면에 제공한다. 표의 각 결과에는 검증 프로토콜을
같이 표시한다. Public 최고라고 확인되지 않은 로컬 Brier를 공식 순위처럼 표현하지
않는다. Drive는 대용량 산출물 저장소라고 설명하되 개인 마운트 절대경로는 넣지
않는다.

- [ ] **Step 4: ROADMAP 작성**

다음 순서를 기록한다.

1. 저장소 foundation 완료 및 검토
2. R9 기반·검증 코드 이전
3. calibration family의 다음 사전 고정 후보 설계
4. XGBoost v3/rescue 코드와 Colab 이전
5. 동료 저장소의 R25/R32 아이디어는 독립 실험으로만 검토

비용은 순서 안내일 뿐 배제 기준이 아니며, rejected 후보는 자기 패키지만 막는다고
명시한다.

- [ ] **Step 5: 전체 테스트 실행**

Run: `python3 -m pytest -q`

Expected: 모든 repository contract 테스트 PASS.

- [ ] **Step 6: 커밋**

```bash
git add README.md docs/ROADMAP.md tests/test_repository_contract.py
git commit -m "docs: add competition dashboard and roadmap"
```

### Task 7: 개인정보·링크·원본 무변경 최종 감사

**Files:**
- Modify: `tests/test_repository_contract.py`

- [ ] **Step 1: 비밀·절대 Drive 경로·끊어진 Markdown 링크 테스트 추가**

```python
import re


def test_tracked_text_contains_no_secret_or_private_mount_path() -> None:
    secret_markers = ("GITHUB_TOKEN=", "AIMERS_REPO_URL=")
    all_operational_paths = [ROOT / "README.md", ROOT / "AGENTS.md"]
    all_operational_paths.extend((ROOT / "docs/rounds").glob("*.md"))
    all_operational_paths.extend((ROOT / "reports").rglob("*.json"))
    all_operational_paths.extend(
        (ROOT / name)
        for name in ("docs/EXPERIMENT_CONTRACT.md", "docs/ROADMAP.md")
    )
    for path in all_operational_paths:
        text = path.read_text(encoding="utf-8")
        for value in secret_markers:
            assert value not in text, f"{value!r} in {path.relative_to(ROOT)}"

    human_docs = [ROOT / "README.md", ROOT / "AGENTS.md"]
    human_docs.extend((ROOT / "docs/rounds").glob("*.md"))
    human_docs.extend(
        (ROOT / name)
        for name in ("docs/EXPERIMENT_CONTRACT.md", "docs/ROADMAP.md")
    )
    for path in human_docs:
        assert "/content/drive/MyDrive/" not in path.read_text(encoding="utf-8")


def test_relative_markdown_links_resolve() -> None:
    pattern = re.compile(r"\[[^]]+\]\((?!https?://|#)([^)]+)\)")
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            clean = target.split("#", 1)[0]
            if clean:
                assert (path.parent / clean).resolve().exists(), (path, target)
```

- [ ] **Step 2: 전체 테스트 실행**

Run: `python3 -m pytest -q`

Expected: 모든 테스트 PASS. 실패한 링크나 금지 경로가 있으면 해당 문서만 수정한다.

- [ ] **Step 3: 정적 감사 실행**

Run: `git diff --check`

Expected: 출력 없이 exit 0.

Run: `find . -type f -size +5M -not -path './.git/*' -print`

Expected: 출력 없음.

Run: `git status --short`

Expected: 이 Task에서 의도한 테스트 및 필요한 문서 수정만 표시.

- [ ] **Step 4: 기존 저장소 무변경 확인**

Run: `git -C /path/to/legacy-lg-aimers-workspace status --short`

Expected: 이전 작업 전 존재하던 사용자 변경만 유지되고, 이 계획이 추가한 변경은 없음.

- [ ] **Step 5: 최종 커밋**

```bash
git add tests/test_repository_contract.py README.md AGENTS.md docs reports .gitignore pyproject.toml
git commit -m "test: audit competition repository foundation"
```

- [ ] **Step 6: push 전 사용자 검토 대기**

커밋 목록, 전체 테스트 결과, 새 파일 목록과 기존 저장소 무변경 사실을 사용자에게
보고한다. 사용자가 새 저장소 push를 명시적으로 요청하기 전에는 push하지 않는다.
