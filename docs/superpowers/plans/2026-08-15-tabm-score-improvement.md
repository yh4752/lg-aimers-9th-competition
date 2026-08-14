# TabM Score Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 시간 전이 예측으로 seed 앙상블을 판정하고, 통과한 조합만 기존 단일 모델과 분리된 새 후보로 전체 학습한다.

**Architecture:** 읽기 전용 OOF 감사기를 첫 경계로 두고 Stage C review bundle과 공식 학습 정답을 `row_id`로 정렬한다. 감사 결과가 사전 등록 gate를 통과한 경우에만 새 Version E 학습기와 ensemble 전용 제출 후보를 연다. 기존 Version D 단일 모델과 제출물은 수정하거나 덮어쓰지 않는다.

**Tech Stack:** Python 3.11, pandas, NumPy, pytest, 기존 `experiments/tabm_campaign` 및 `submission` 모듈

---

## 파일 구조

- `experiments/tabm_campaign/ensemble_audit.py`: Stage C 예측 정렬, Brier와 gate 판정
- `experiments/tabm_campaign/score_improvement_contract.json`: 조합, fold와 승급 기준
- `tools/audit_tabm_seed_ensemble.py`: 사용자 실행용 진입점과 review bundle 생성
- `tests/test_tabm_ensemble_audit.py`: 정렬, 조합, gate와 실패 경로 fixture 테스트
- `experiments/tabm_campaign/version_e.py`: 승인된 seed의 전체 학습과 중단 복구
- `submission/tabm_ensemble_candidate.py`: 다중 member manifest와 artifact 검증
- `tests/test_tabm_submission_package.py`: 모델 member 해시와 패키지 gate 검사

행 단위 파생변수, CatBoost blend와 보정은 이 계획을 실행한 결과를 입력으로 받는
독립 작업이다. 각 단계는 점수 개선 설계의 공통 검증 계약을 유지하되 별도 구현 계획을
작성해 변경 원인을 섞지 않는다.

### Task 1: Seed 앙상블 계약 고정

**Files:**
- Create: `experiments/tabm_campaign/score_improvement_contract.json`
- Test: `tests/test_tabm_ensemble_audit.py`

- [ ] **Step 1: 계약 fixture 검사를 작성한다**

세 seed, 두 fold, 단일 세 후보와 `mean_all`, `mean_42_3407`, 최소 평균 개선
`0.00003`, 최대 fold 악화 `0.00003`이 정확히 선언됐는지 검사한다. 알 수 없는 seed,
중복 조합, 음수 허용치와 합이 1이 아닌 가중치를 각각 거부하는 테스트를 작성한다.

- [ ] **Step 2: 실패를 확인한다**

Run: `.venv/bin/pytest tests/test_tabm_ensemble_audit.py -q`

Expected: 계약 파일과 loader가 없어 실패한다.

- [ ] **Step 3: 최소 계약과 loader를 구현한다**

계약에는 다음 조합만 넣는다.

```json
{
  "folds": ["2022->2023", "2023->2024"],
  "seeds": [42, 2026, 3407],
  "ensembles": {
    "mean_all": {"42": 0.3333333333333333, "2026": 0.3333333333333333, "3407": 0.3333333333333333},
    "mean_42_3407": {"42": 0.5, "3407": 0.5}
  },
  "min_weighted_gain": 0.00003,
  "max_fold_degrade": 0.00003
}
```

loader는 unknown key와 유한하지 않은 수를 거부하고 정렬된 immutable dataclass를
반환한다.

- [ ] **Step 4: 계약 테스트를 통과시킨다**

Run: `.venv/bin/pytest tests/test_tabm_ensemble_audit.py -q`

Expected: contract 관련 테스트가 통과한다.

- [ ] **Step 5: 커밋한다**

```bash
git add experiments/tabm_campaign/score_improvement_contract.json tests/test_tabm_ensemble_audit.py
git commit -m "test: seal TabM ensemble audit contract"
```

### Task 2: 읽기 전용 OOF 감사기

**Files:**
- Create: `experiments/tabm_campaign/ensemble_audit.py`
- Modify: `tests/test_tabm_ensemble_audit.py`

- [ ] **Step 1: 정렬과 Brier 실패 테스트를 작성한다**

작은 두 fold fixture로 순서가 섞인 예측을 `row_id`에 맞춰 복구하고, 누락·중복 ID,
정답 불일치, `[0, 1]` 밖 확률과 NaN을 거부하는 테스트를 작성한다. 단일 seed와 두
평균 조합의 Brier를 직접 계산한 값과 비교한다.

- [ ] **Step 2: 실패를 확인한다**

Run: `.venv/bin/pytest tests/test_tabm_ensemble_audit.py -q`

Expected: `ensemble_audit` 모듈이 없어 실패한다.

- [ ] **Step 3: 최소 감사기를 구현한다**

`row_id`, `control_success`, `probability`만 읽고 merge 결과가 각 원본과 같은 행 수인지
검사한다. 각 조합의 fold Brier, 검증 행 수 가중 Brier, 최선 단일 seed 대비 gain과
worst fold delta를 계산한다. 계약 gate를 모두 통과한 단순 평균 중 가중 Brier가 가장
낮은 하나만 `promoted`로 기록한다.

- [ ] **Step 4: 전체 단위 테스트를 통과시킨다**

Run: `.venv/bin/pytest tests/test_tabm_ensemble_audit.py -q`

Expected: 모든 정렬, 수치와 gate 테스트가 통과한다.

- [ ] **Step 5: 커밋한다**

```bash
git add experiments/tabm_campaign/ensemble_audit.py tests/test_tabm_ensemble_audit.py
git commit -m "feat: audit TabM seed ensembles from OOF"
```

### Task 3: 사용자 실행 도구와 review bundle

**Files:**
- Create: `tools/audit_tabm_seed_ensemble.py`
- Modify: `tests/test_tabm_ensemble_audit.py`
- Modify: `docs/TABM_CHAMPION_KAGGLE.md`

- [ ] **Step 1: CLI fixture 테스트를 작성한다**

임시 Stage C ZIP과 작은 train CSV를 입력해 `ensemble_audit.json`, `manifest.json`과
로그만 든 review ZIP이 생성되는지 검사한다. 입력 ZIP과 예측 CSV의 SHA-256이
manifest에 남고 입력 경로, 원본 자료와 예측 행은 ZIP에 포함되지 않아야 한다.

- [ ] **Step 2: 실패를 확인한다**

Run: `.venv/bin/pytest tests/test_tabm_ensemble_audit.py -q`

Expected: CLI 진입점이 없어 실패한다.

- [ ] **Step 3: CLI를 구현한다**

감사 모드는 `--stage-c-review`, `--train-csv`, `--output-dir` 세 인자를 받고, 검증
모드는 이들과 상호 배타적인 `--verify-review` 하나만 받는다. 정상 종료는
`TABM_ENSEMBLE_AUDIT_SUCCESS decision=<promoted|keep_single> review=<path>`를,
오류는 `TABM_ENSEMBLE_AUDIT_ERROR stage=<stage> type=<type> message=<message>`를
출력한다. 어떤 경로에서도 모델 학습이나 제출 ZIP 생성을 호출하지 않는다.

- [ ] **Step 4: 문서와 테스트를 확인한다**

Run: `.venv/bin/pytest tests/test_tabm_ensemble_audit.py -q && git diff --check`

Expected: 테스트가 통과하고 whitespace 오류가 없다.

- [ ] **Step 5: 커밋한다**

```bash
git add tools/audit_tabm_seed_ensemble.py tests/test_tabm_ensemble_audit.py docs/TABM_CHAMPION_KAGGLE.md
git commit -m "feat: add TabM seed ensemble audit handoff"
```

### Task 4: 사용자 전체 OOF 감사와 판정 기록

**Files:**
- Create after user run: `reports/acceptances/tabm_seed_ensemble_acceptance.json` or `reports/rejections/tabm_seed_ensemble_rejection.json`
- Modify after user run: `reports/EXPERIMENT_LEDGER.md`

- [ ] **Step 1: 사용자에게 한 번에 실행 가능한 셀을 전달한다**

목적, Stage C review ZIP과 공식 train CSV 입력, CPU 예상 시간, 재실행 안전성,
성공·오류 로그와 반환할 review ZIP 이름을 셀 위 설명에 포함한다. 전체 OOF 계산은
사용자가 실행한다.

- [ ] **Step 2: review bundle의 해시와 행 정렬을 검증한다**

Run: `.venv/bin/python tools/audit_tabm_seed_ensemble.py --verify-review <returned-review.zip>`

Expected: `TABM_ENSEMBLE_REVIEW_VERIFIED`와 정확한 입력·출력 SHA-256이 출력된다.

- [ ] **Step 3: 사전 등록 gate만으로 판정한다**

`promoted`이면 acceptance, `keep_single`이면 rejection JSON을 기록한다. Public 점수는
판정 입력에 넣지 않는다.

- [ ] **Step 4: 기록을 검증하고 커밋한다**

Run: `.venv/bin/pytest -q && git diff --check`

Expected: 전체 테스트가 통과하고 문서와 JSON에 whitespace 오류가 없다.

### Task 5: 승급된 ensemble만 전체 학습 경로에 연결

**Files:**
- Create: `experiments/tabm_campaign/version_e.py`
- Create: `submission/tabm_ensemble_candidate.py`
- Create: `tests/test_tabm_campaign_version_e.py`
- Modify: `tests/test_tabm_submission_package.py`

- [ ] **Step 1: acceptance가 없으면 차단되는 테스트를 작성한다**

앙상블 acceptance JSON과 정확한 Stage C 예측 해시가 없거나 gate가 하나라도 false면
추가 seed 학습과 패키지 생성이 시작되기 전에 중단되는지 검사한다.

- [ ] **Step 2: 다중 member manifest 실패를 확인한다**

Run: `.venv/bin/pytest tests/test_tabm_campaign_version_e.py tests/test_tabm_submission_package.py -q`

Expected: 다중 seed 후보가 등록되지 않아 실패한다.

- [ ] **Step 3: 승인된 seed만 학습·평균하도록 구현한다**

Version E는 단일 모델과 같은 3 epoch, 전처리, 구조와 학습률을 사용하고 seed만
acceptance에 있는 값으로 바꾼다. 추론은 각 member의 sigmoid 확률을 acceptance의
고정 가중치로 합친다. 평가 자료로 가중치를 다시 계산하지 않는다. 출력 디렉터리와
후보 ID는 Version D와 겹치지 않게 고정한다.

- [ ] **Step 4: 모델 hash와 독립성 gate를 통과시킨다**

Run: `.venv/bin/pytest tests/test_tabm_campaign_version_e.py tests/test_tabm_submission_package.py -q`

Expected: acceptance 부재 차단, member 해시, 순서·batch 독립성 테스트가 모두 통과한다.

- [ ] **Step 5: 커밋한다**

```bash
git add experiments/tabm_campaign/version_e.py submission/tabm_ensemble_candidate.py tests/test_tabm_campaign_version_e.py tests/test_tabm_submission_package.py
git commit -m "feat: support accepted TabM seed ensemble"
```

### Task 6: 최종 검증과 제출 후보 분리

**Files:**
- Modify: `reports/EXPERIMENT_LEDGER.md`
- Modify: `docs/ROADMAP.md`
- Modify only after all gates pass: `submission/adapters.py`

- [ ] **Step 1: 새 후보의 provenance 검사를 실행한다**

입력 데이터, 전처리, 코드, 모델 member와 acceptance SHA-256을 하나의 identity에 묶는다.

- [ ] **Step 2: 사용자가 전체 규모 독립성·시간 검사를 실행한다**

L4 또는 T4에서 245,789행 이상, 단독 행, 역순, shuffle과 여러 batch 크기를 검사한다.
공식 제한의 80%인 설치 480초, 추론 480초 안에서만 통과시킨다.

- [ ] **Step 3: 모든 gate가 현재 해시와 일치할 때만 후보를 등록한다**

하나라도 누락되거나 이전 코드·모델 해시를 가리키면 registry 등록과 패키지 생성을
중단한다. 기존 단일 모델 후보의 acceptance에는 영향을 주지 않는다.

- [ ] **Step 4: 최종 검증을 실행한다**

Run: `.venv/bin/pytest -q && .venv/bin/python -m compileall submission experiments tools tests && git diff --check`

Expected: 전체 테스트 0 failures, compileall exit 0, whitespace 오류 없음.

- [ ] **Step 5: 사용자 승인 후 별도 제출 ZIP을 생성한다**

새 후보의 acceptance, 정책, 성능, 런타임, 독립성 및 현재 artifact 해시가 모두
통과한 경우에만 기존 hash-gated builder로 새 출력 디렉터리에 생성한다. 기존
`artifacts/tabm_submission_version_d/submit.zip`을 덮어쓰지 않는다.
