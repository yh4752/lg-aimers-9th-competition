# Preprocessing EDA Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 누출 없는 전처리 연구용 EDA를 사용자가 공식 데이터로 직접 실행할 수 있도록 순차 복사용 Colab 셀로 제공한다.

**Architecture:** 실행 가능한 코드는 `docs/PREPROCESSING_EDA_COLAB.md`의 8개 Python 셀에만 둔다. 셀은 설정·로딩, 순수 헬퍼, 무결성·프로파일, 시간 전이 probe, adversarial 진단, `asof_*`·수치 변환·상호작용, 그래프·요약, 게시·자체 검증 순으로 상태를 전달한다. 로컬 테스트는 Markdown의 Python 셀을 추출해 전부 컴파일하고, 외부 패키지가 필요 없는 핵심 헬퍼를 작은 합성 입력으로 실행한다.

**Tech Stack:** Python 3.11+, pandas, NumPy, SciPy, scikit-learn, Matplotlib, standard-library `unittest`

---

### Task 1: 과설계 검토 결과를 설계에 반영

**Files:**
- Modify: `docs/superpowers/specs/2026-08-12-preprocessing-eda-design.md`

- [ ] **Step 1: 의미 없는 수치 변환 성능 probe를 제거한다**

`raw`, `standard`, `robust`의 단변량 logistic 성능 비교를 삭제하고, 학습 fold에서
적합한 변환의 finite 비율, 왜도, 중앙 절대 편차, 1%·99% 분위수와 검증 범위
초과율을 기록하도록 바꾼다.

- [ ] **Step 2: `asof_*`의 타깃 의미를 제한한다**

`control_success` Brier와 calibration은 success rate에만 계산한다. K-grid
smoothing은 정확한 분모가 있는 아래 두 쌍에만 적용한다.

```python
SMOOTHING_PAIRS = {
    "asof_pitcher_success_rate": "asof_pitcher_n",
    "asof_batter_success_rate": "asof_batter_n",
}
```

- [ ] **Step 3: 설계 문서 검사를 실행한다**

Run: `git diff --check`

Expected: 출력 없이 종료 코드 0.

### Task 2: Markdown Colab 계약 테스트 작성

**Files:**
- Create: `tests/test_preprocessing_eda_colab.py`

- [ ] **Step 1: 존재하지 않는 구현 문서를 요구하는 테스트를 작성한다**

```python
from __future__ import annotations

import ast
import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "PREPROCESSING_EDA_COLAB.md"


def python_cells() -> list[str]:
    text = DOC.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)


class ColabContractTest(unittest.TestCase):
    def test_document_exists_and_has_eight_ordered_cells(self) -> None:
        self.assertTrue(DOC.is_file())
        cells = python_cells()
        self.assertEqual(len(cells), 8)
        ids = [re.search(r"^# CELL_ID: (\S+)", cell, re.MULTILINE).group(1) for cell in cells]
        self.assertEqual(ids, [f"{i:02d}" for i in range(1, 9)])
```

- [ ] **Step 2: RED를 확인한다**

Run: `python3 -m unittest tests.test_preprocessing_eda_colab -v`

Expected: `FileNotFoundError` 또는 문서 부재 assertion으로 FAIL.

- [ ] **Step 3: 나머지 정적·순수 헬퍼 계약을 추가한다**

테스트는 모든 셀의 `compile()` 성공, `pip install` 부재, 개인 Drive 하위 경로 부재,
13개 CSV와 2개 JSON 이름, `EDA_SUCCESS`·`EDA_ERROR`, `TEST_USAGE =
"schema_only"`를 확인한다. AST로 아래 세 함수를 추출해 합성 입력으로 실행한다.

```python
def load_pure_helpers(cells: list[str]) -> dict[str, object]:
    wanted = {"make_expanding_folds", "stable_sample_ids", "json_safe"}
    nodes = []
    for cell in cells:
        tree = ast.parse(cell)
        nodes.extend(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        )
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"hashlib": __import__("hashlib"), "math": math}
    exec(compile(module, str(DOC), "exec"), namespace)
    return namespace
```

### Task 3: 실행 안내·설정·순수 헬퍼 셀 작성

**Files:**
- Create: `docs/PREPROCESSING_EDA_COLAB.md`
- Test: `tests/test_preprocessing_eda_colab.py`

- [ ] **Step 1: 셀 01에 환경과 고정 설정을 작성한다**

셀 01은 Colab 여부를 확인하고 Drive를 `/content/drive`에 마운트한다. 개인 하위
경로를 코드에 넣지 않고 `ARCHIVE_PATH`가 비어 있으면 `official_open.zip`을
검색한다. 후보가 정확히 하나가 아니면 경로 설정 방법을 포함한 `EDA_ERROR`를
출력한다. `RUN_ID`, seed, K-grid, 표본 상한과 모든 출력 파일명을 고정한다.

- [ ] **Step 2: 셀 02에 외부 패키지 없는 순수 헬퍼를 먼저 작성한다**

```python
def make_expanding_folds(seasons):
    ordered = sorted({int(value) for value in seasons})
    if len(ordered) < 2:
        raise ValueError("at least two seasons are required")
    return [
        {"name": f"through_{valid - 1}_to_{valid}",
         "train_seasons": tuple(year for year in ordered if year < valid),
         "valid_season": valid}
        for valid in ordered[1:]
    ]
```

`stable_sample_ids`는 `row_id`와 seed를 SHA-256으로 점수화해 입력 행 순서와
무관한 작은 ID 집합을 반환한다. `json_safe`는 dict·list를 재귀 순회하며 비유한
실수를 `None`으로 바꾼다.

- [ ] **Step 3: GREEN 일부를 확인한다**

Run: `python3 -m unittest tests.test_preprocessing_eda_colab -v`

Expected: 문서·셀 수·순수 헬퍼 테스트 PASS, 아직 요구한 산출물 이름이 없으면 해당
계약만 FAIL.

### Task 4: 데이터 로딩·무결성·프로파일 셀 작성

**Files:**
- Modify: `docs/PREPROCESSING_EDA_COLAB.md`
- Test: `tests/test_preprocessing_eda_colab.py`

- [ ] **Step 1: 셀 03에 ZIP 로딩과 스키마 검사를 구현한다**

정확한 48개 입력 컬럼과 `control_success`를 선언한다. ZIP 멤버는 basename이
`train.csv`, `test.csv`인 항목을 각각 하나만 허용한다. ZIP과 두 멤버의 SHA-256,
행 수와 dtype을 기록한다. test는 스키마 검사 뒤 `fit_audit["test_rows_used"] = 0`을
고정하고 어떤 분석 함수에도 전달하지 않는다.

- [ ] **Step 2: 무결성 검사 함수를 구현한다**

`add_check(name, status, observed, expected, detail)`을 통해 스키마·ID·타깃 위반은
`fail`, 의미 관계는 `warn`으로 기록한다. `fail`이 하나라도 있으면
`EDA_ERROR stage=integrity`를 출력하고 중단한다. 원본값은 수정하지 않는다.

- [ ] **Step 3: 셀 04에 전체·시즌 프로파일과 drift를 구현한다**

수치형은 KS statistic과 IQR 정규화 Wasserstein, 범주형은 total variation과
Jensen-Shannon distance를 계산한다. 큰 연속형 표본은 `stable_sample_ids`로
시즌별 최대 200,000행만 선택하고 선택 ID 해시를 manifest에 남긴다.

### Task 5: 시간 전이 단변량·결측·ID probe 작성

**Files:**
- Modify: `docs/PREPROCESSING_EDA_COLAB.md`
- Test: `tests/test_preprocessing_eda_colab.py`

- [ ] **Step 1: 셀 05에 누출 없는 target-rate probe를 구현한다**

수치 피처는 학습 분위수 bin, 범주 피처는 학습 범주를 사용한다. 공통 식은 다음과
같다.

```python
prediction = (group_sum + SMOOTH_K * train_prior) / (group_count + SMOOTH_K)
```

미등록·결측 fallback은 학습 prior다. fold별 Brier와 검증 행 수 가중 aggregate를
저장하며, 검증 타깃은 평가에만 사용한다.

- [ ] **Step 2: 결측 indicator probe와 ID coverage를 구현한다**

결측이 존재하는 원본 피처마다 boolean category probe를 실행한다. 의미 범주형마다
학습 vocabulary, singleton·rare 비율, 다음 시즌 미등록률과 cold-start Brier를
계산한다.

### Task 6: 다변량 drift 진단 작성

**Files:**
- Modify: `docs/PREPROCESSING_EDA_COLAB.md`
- Test: `tests/test_preprocessing_eda_colab.py`

- [ ] **Step 1: 셀 06에 adversarial validation을 구현한다**

각 인접 시즌에서 최대 200,000행씩 선택하고 stratified 80:20으로 분리한다.
범주형 frequency와 수치 median은 판별 학습 부분에서만 적합한다.
`HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31,
random_state=SEED)`로 holdout ROC-AUC와 log loss를 계산한다.

- [ ] **Step 2: 상관 군집과 중요도를 구현한다**

수치형 Spearman 절댓값 0.95 이상을 union-find 연결 성분으로 묶는다. holdout 최대
50,000행에서 개별 permutation importance 5회와 군집 동시 permutation 5회를
계산한다. 시즌 판별 중요도임을 결과 `importance_scope`에 명시한다.

### Task 7: 전처리 특화 진단·상호작용 작성

**Files:**
- Modify: `docs/PREPROCESSING_EDA_COLAB.md`
- Test: `tests/test_preprocessing_eda_colab.py`

- [ ] **Step 1: 셀 07에 success-rate 신뢰도와 smoothing을 구현한다**

success rate의 표본 수 구간별 평균 예측, 관측률, Brier와 calibration gap을
계산한다. K-grid는 누적 투수·타자 성공률 두 쌍에만 적용한다. 다른 rate는 Brier
대상에 넣지 않는다.

- [ ] **Step 2: 수치 변환 분포 진단을 구현한다**

각 fold와 수치 피처에서 median·StandardScaler·RobustScaler·QuantileTransformer·
Yeo-Johnson을 학습 구간에만 적합한다. 성능 순위를 만들지 않고 finite 비율, 왜도,
중앙 절대 편차, 1%·99% 분위수와 검증 범위 초과율을 기록한다.

- [ ] **Step 3: 의미 기반 상호작용 probe를 구현한다**

count state, hand matchup, base state×outs, game type×count state, 점수 상태와 투수
팀 기대 승률만 만든다. 전수 조합 코드는 두지 않는다.

### Task 8: 그래프·요약·원자적 게시 셀 작성

**Files:**
- Modify: `docs/PREPROCESSING_EDA_COLAB.md`
- Test: `tests/test_preprocessing_eda_colab.py`

- [ ] **Step 1: 셀 08에 8개 고정 그래프를 작성한다**

시즌별 행 수·타깃률, 결측률, drift, 단변량 probe, adversarial 결과, ID 미등록률,
success-rate reliability, smoothing·수치 변환 요약만 PNG로 만든다.

- [ ] **Step 2: 구조화 산출물을 검증하고 게시한다**

13개 CSV와 `eda_summary.json`, `run_manifest.json`을 런타임 임시 디렉터리에 먼저
쓴다. 파일 존재·행 수·JSON 비유한값·fold 순서·test fit 0행을 검사한 뒤에만
Drive의 최종 `RUN_ID` 디렉터리로 rename한다. 기존 완성 디렉터리는 덮어쓰지 않는다.

- [ ] **Step 3: 자체 합성 검사를 셀 08 끝에서 실행한다**

작은 3시즌 배열로 fold, 안정 표본 추출과 JSON 안전 변환을 확인한다. 성공 시 아래
문자열을 정확히 출력한다.

```text
EDA_SUCCESS run_id=<RUN_ID> output_dir=<OUTPUT_DIR> summary=<SUMMARY_PATH>
```

### Task 9: 최종 검증과 커밋

**Files:**
- Modify: `docs/PREPROCESSING_EDA_COLAB.md`
- Modify: `docs/superpowers/specs/2026-08-12-preprocessing-eda-design.md`
- Create: `tests/test_preprocessing_eda_colab.py`

- [ ] **Step 1: 테스트와 문서 검사를 실행한다**

Run: `python3 -m unittest tests.test_preprocessing_eda_colab -v`

Expected: 모든 테스트 PASS.

Run: `git diff --check`

Expected: 출력 없이 종료 코드 0.

- [ ] **Step 2: 범위 이탈을 검사한다**

Run: `git diff --name-only`

Expected: 설계, 계획, Colab Markdown과 테스트 파일만 표시된다. `.ipynb`, 데이터,
ZIP, OOF, 모델 또는 submission 파일은 없어야 한다.

- [ ] **Step 3: 구현을 커밋한다**

```bash
git add docs/PREPROCESSING_EDA_COLAB.md \
  docs/superpowers/specs/2026-08-12-preprocessing-eda-design.md \
  docs/superpowers/plans/2026-08-12-preprocessing-eda-colab.md \
  tests/test_preprocessing_eda_colab.py
git commit -m "feat: add preprocessing eda colab cells"
```
