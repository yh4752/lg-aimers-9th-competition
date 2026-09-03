# LG Aimers Portfolio Finish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 공개 가능한 저장소 진입점, 대표 실험 흐름 그림, 포트폴리오 요약과 면접 문답을 완성한다.

**Architecture:** 기존 장부와 회고를 사실 기준으로 유지하고 새 문서는 독자의 목적에 따라 얇게 나눈다. README는 안내판, SVG는 흐름 요약, 포트폴리오 문서는 1~2분 소개, 면접 문답은 설명 연습용으로만 사용한다.

**Tech Stack:** Markdown, 정적 SVG, Git, Python 표준 라이브러리 기반 검사

---

### Task 1: 개인 경로 표기 정리

**Files:**
- Modify: `docs/**/*.md`
- Modify: `tests/test_catboost_50_50_realign_colab_cell.py`
- Modify: `tests/test_catboost_deployment_colab_cell.py`
- Modify: `tests/test_catboost_tabm_blend_colab_cell.py`
- Modify: `tests/test_hierarchical_tabm_colab_cell.py`

- [ ] **Step 1: 사용자 이름이 포함된 경로를 확인한다**

Run: `git grep -n 'yonghyun'`

Expected: 실행 안내와 과거 계획, 테스트 fixture의 위치가 출력된다.

- [ ] **Step 2: 경로를 공용 표기로 바꾼다**

다음 순서로 치환한다.

```text
/Users/yonghyun/Documents/lg-aimers-9th-competition → <REPO_ROOT>
/Users/yonghyun/Documents/LG_AIMERS_2026 → <LEGACY_WORKSPACE>
/Users/yonghyun/Documents/kaggle-lg-aimers-9th-data-upload → <OFFICIAL_DATA_DIR>
/Users/yonghyun/Downloads → <DOWNLOAD_DIR>
/Users/yonghyun → /Users/example
```

`assert "/Users/" not in text`처럼 절대 경로 금지를 검사하는 문자열은 바꾸지 않는다.

- [ ] **Step 3: 개인 경로가 사라졌는지 확인한다**

Run: `git grep -n 'yonghyun'`

Expected: 출력 없음.

- [ ] **Step 4: 경로 관련 테스트를 실행한다**

Run: `python3 -m pytest tests/test_catboost_50_50_realign_colab_cell.py tests/test_catboost_deployment_colab_cell.py tests/test_catboost_tabm_blend_colab_cell.py tests/test_hierarchical_tabm_colab_cell.py tests/test_tree_expert_s4_runbook.py -q`

Expected: 모든 테스트 통과.

### Task 2: 대표 실험 흐름 그림 제작

**Files:**
- Create: `docs/assets/lg-aimers-experiment-journey.svg`
- Modify: `README.md`

- [ ] **Step 1: 정적 SVG를 만든다**

SVG에는 다섯 상태를 순서대로 넣는다.

```text
시간 전이 OOF
→ 전처리 5단계
→ TabM 872.39
→ Tree Expert E2 977.38
→ E3 OOM / 성능 미판정
```

E2는 확인된 최고 제출로 강조하고 E3는 실패 원인을 텍스트로 표시한다. 외부 리소스,
스크립트, 애니메이션은 넣지 않는다.

- [ ] **Step 2: README에 그림과 대체 텍스트를 연결한다**

`처음 읽는다면` 앞에 `LG Aimers 9기 실험 흐름`이라는 대체 텍스트로
`docs/assets/lg-aimers-experiment-journey.svg`를 연결한다.

- [ ] **Step 3: SVG 구문을 확인한다**

Run: `python3 -c "import xml.etree.ElementTree as ET; ET.parse('docs/assets/lg-aimers-experiment-journey.svg')"`

Expected: 종료 코드 0.

### Task 3: 포트폴리오 요약 작성

**Files:**
- Create: `docs/PORTFOLIO_SUMMARY.md`
- Modify: `README.md`

- [ ] **Step 1: 1~2분 소개 문서를 작성한다**

다음 순서와 사실을 사용한다.

```text
문제: 투구 성공 확률 예측
제약: 시간 누출 금지, 평가 행 독립 예측
검증: 2021→2022, 2022→2023, 2023→2024
성과: TabM 872.3920 → Tree Expert E2 977.3810
실패: E3는 33,677초 뒤 OOM, 성능 미판정
역할: 사용자는 방향 승인·장시간 실행·산출물 전달·제출 결정,
      Codex는 코드·검토·결과 분석 보조
```

- [ ] **Step 2: 근거 문서 링크를 단다**

`PROJECT_RETROSPECTIVE.md`, `EXPERIMENT_LEDGER.md`, `experiment_registry.json`,
블로그 글 세 편으로 연결한다.

- [ ] **Step 3: README 저장소 안내에 요약 문서를 연결한다**

Expected: 처음 방문한 독자가 전체 회고를 읽기 전에 짧은 소개를 찾을 수 있다.

### Task 4: 면접 대비 문답 작성

**Files:**
- Create: `docs/INTERVIEW_QA.md`
- Modify: `README.md`

- [ ] **Step 1: 핵심 질문과 답변을 작성한다**

다음 주제를 포함한다.

```text
프로젝트 한 줄 설명 / 시간 전이 검증 / OOF / Brier / anchor / 잔차 보정 /
TabM에서 E2로 바꾼 이유 / seed 평균 / 규칙 준수 / E3 OOM 해석 /
사용자와 Codex의 역할 / 다음 대회 개선점
```

각 답변은 결론, 근거, 한계 순서로 쓰고 모르는 내용을 직접 구현했다고 표현하지 않는다.

- [ ] **Step 2: README 저장소 안내에 문답을 연결한다**

Expected: 포트폴리오 요약과 면접 문답의 목적이 겹치지 않는다.

### Task 5: 전체 공개·사실·회귀 검사

**Files:**
- Verify: all tracked files

- [ ] **Step 1: 민감정보와 대용량 산출물을 다시 검사한다**

대표 키 패턴, URL 토큰, 개인 경로, 추적된 `.zip/.pkl/.cbm/.pt/.parquet`와 5MB 초과
파일을 검사한다.

Expected: 비밀정보 0건, 개인 이름 경로 0건, 추적 산출물 0건, 5MB 초과 0건.

- [ ] **Step 2: 핵심 사실을 대조한다**

Run: `git grep -n '977.3809532715\|33,677' README.md docs/PORTFOLIO_SUMMARY.md docs/INTERVIEW_QA.md docs/PROJECT_RETROSPECTIVE.md reports/EXPERIMENT_LEDGER.md`

Expected: 최고 점수와 E3 실패 기록이 서로 모순되지 않는다.

- [ ] **Step 3: 전체 테스트를 실행한다**

Run: `python3 -m pytest -q`

Expected: 모든 테스트 통과.

- [ ] **Step 4: 사용자 파일을 제외해 커밋하고 푸시한다**

커밋 대상은 이번 계획에서 명시한 추적 문서·테스트와 새 SVG 두 문서뿐이다.
`notebooks/INDEPENDENT_DL_RECOVERY_AND_TABR.ipynb`, `uv.lock`은 포함하지 않는다.

```bash
git commit -m "docs: finish LG Aimers portfolio guide"
git push origin main
```
