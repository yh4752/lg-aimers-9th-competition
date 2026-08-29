# Public Experiment Narrative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task.

**Goal:** 검증된 실험 기록을 비전공자도 이해할 수 있는 프로젝트 소개와 재현 가능한 기술 기록으로 정리한다.

**Architecture:** `reports/EXPERIMENT_LEDGER.md`를 수치와 판정의 단일 기준으로 삼고, `docs/EXPERIMENT_JOURNEY.md`가 실험의 문제·가설·결과·다음 선택을 설명한다. 루트 `README.md`는 두 문서의 핵심만 압축한 포트폴리오 진입점이며, `docs/rounds/README.md`는 세부 라운드 문서와 새 실험 여정을 연결한다.

**Tech Stack:** Markdown, Git, pytest 기반 문서 계약 검사, 로컬 실험 번들의 manifest/decision JSON

---

### Task 1: 근거와 공개 범위 확정

**Files:**
- Read: `reports/EXPERIMENT_LEDGER.md`
- Read: `docs/rounds/README.md`
- Read: 로컬 review/handoff ZIP의 manifest 및 decision 파일

**Step 1: 기존 공개 점수와 실험 판정을 대조한다**

CatBoost 828.9964, 격리된 XGBoost 820.9583, TabM 872.3920, Tree Expert E2 977.3810을 기존 원장과 대조한다.

**Step 2: 후기 실험의 판정을 번들에서 확인한다**

T3 구조 탐색, CatBoost+TabM 고정 블렌드, 배포 정렬 CatBoost, 계층 보정, 실패 유형 라벨 감사를 확인한다. 수치나 상태를 확인할 수 없는 실험은 성과로 기록하지 않는다.

**Step 3: 공개 금지 항목을 확인한다**

원본 데이터, 모델 파일, 대형 ZIP, 로컬 절대경로, 토큰과 비밀값은 문서에 포함하지 않는다.

### Task 2: 실험 원장을 최신화

**Files:**
- Modify: `reports/EXPERIMENT_LEDGER.md`

**Step 1: T3의 오래된 상태를 실제 판정으로 교체한다**

`implementation_ready`가 아니라 시간 구조 게이트 탈락으로 기록하고, weighted gain과 탈락 이유를 남긴다.

**Step 2: 후기 검증 행을 추가한다**

고정 블렌드의 OOF 통과, 배포 정렬 실패, 계층 보정 후보 없음, 실패 유형 라벨 부적합을 각각 독립된 행으로 추가한다.

**Step 3: 결론과 다음 방향을 갱신한다**

현재 최고 제출은 E2이며, 다음 독립 후보는 이종 트리 잔차 S3임을 명시한다. 코드 준비 상태를 완료된 실험과 혼동하지 않는다.

### Task 3: 실험 여정 문서 작성

**Files:**
- Create: `docs/EXPERIMENT_JOURNEY.md`

**Step 1: 비전공자용 배경을 설명한다**

대회의 예측 문제, Brier score, 시간 순서 검증, OOF, anchor, residual correction, ensemble을 일상적인 비유 뒤에 정확한 정의로 설명한다.

**Step 2: 실험 흐름을 문제 해결 이야기로 정리한다**

단일 트리 기준선 → 규칙 감사 → TabM → 전처리 탐색 → 시간 구조와 행 피처 → Tree Expert E2 → 후속 실패/보류 → 현재 S3 방향 순으로 정리한다.

**Step 3: 의사결정과 협업 방식을 설명한다**

실패한 실험을 폐기한 근거, 제출 게이트, 사용자와 AI의 역할 분담을 과장 없이 기록한다.

### Task 4: README를 포트폴리오 진입점으로 개편

**Files:**
- Modify: `README.md`

**Step 1: 30초 안에 이해할 요약을 앞에 배치한다**

문제, 제약, 최고 공개 점수, 핵심 개선, 현재 상태를 짧게 보여준다.

**Step 2: 점수 변화와 연구 전환점을 제시한다**

검증 가능한 네 개의 공개 점수만 시간순으로 보여주고, 872→977 개선이 무엇을 의미하는지 설명한다.

**Step 3: 기술·규칙·재현성·문서 링크를 정리한다**

대회 규칙 준수, 대형 산출물 제외, 주요 문서, 저장소 구조, 역할 분담을 연결한다.

### Task 5: 문서 탐색 경로 정리

**Files:**
- Modify: `docs/rounds/README.md`

**Step 1: 새 실험 여정과 원장 링크를 상단에 추가한다**

요약 독자는 실험 여정으로, 수치 검증 독자는 원장으로 이동하도록 안내한다.

**Step 2: 후기 캠페인의 위치를 설명한다**

기존 R01~R07은 초기 라운드 기록으로 유지하고, 이후 캠페인은 실험 여정과 원장을 기준으로 추적한다고 명시한다.

### Task 6: 자연스러운 표현과 문서 일관성 점검

**Files:**
- Review: `README.md`
- Review: `docs/EXPERIMENT_JOURNEY.md`
- Review: `reports/EXPERIMENT_LEDGER.md`
- Review: `docs/rounds/README.md`

**Step 1: 문장을 사람이 쓴 프로젝트 회고처럼 다듬는다**

과도한 나열, 반복되는 결론, 근거 없는 자신감, 번역투를 제거한다.

**Step 2: 용어와 숫자를 교차 확인한다**

동일한 실험의 이름, 점수, 상태가 네 문서에서 일치하는지 확인한다.

### Task 7: 문서 계약과 공개 안전성 검증

**Files:**
- Test: `tests/test_repository_contract.py`

**Step 1: 문서 계약 테스트를 실행한다**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_repository_contract.py -q`

Expected: PASS

**Step 2: 링크와 민감 문자열을 검사한다**

Run: `rg -n '/Users/|GHS[A-Z0-9]|TODO|TBD' README.md docs/EXPERIMENT_JOURNEY.md reports/EXPERIMENT_LEDGER.md docs/rounds/README.md`

Expected: 공개하면 안 되는 로컬 경로·토큰·미완성 표기가 없음

**Step 3: 패치 형식을 검사한다**

Run: `git diff --check`

Expected: 출력 없음

### Task 8: 문서 변경만 커밋하고 main에 반영

**Files:**
- Commit: 이 계획과 네 개의 문서만

**Step 1: 관련 파일만 명시적으로 stage한다**

기존 TabM 코드 수정과 미추적 노트북은 stage하지 않는다.

**Step 2: 커밋 전에 staged diff를 확인한다**

Run: `git diff --cached --stat && git diff --cached --check`

Expected: 문서 파일만 포함되고 형식 오류가 없음

**Step 3: 원격 변경을 확인하고 push한다**

Run: `git fetch origin && git rev-list --left-right --count origin/main...main`

Expected: 원격 전용 커밋 수가 0

Run: `git push origin main`

**Step 4: 원격 반영을 검증한다**

Run: `test "$(git rev-parse main)" = "$(git rev-parse origin/main)"`

Expected: exit code 0
