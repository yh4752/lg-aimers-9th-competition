# XGBoost Public Result Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 공격적 XGBoost 탐색의 검증된 구성과 Public `820.9583317093`을 Drive 원본에 연결해 기존 기록에 최소한으로 추가한다.

**Architecture:** 새 산출물은 작은 요약 JSON 하나뿐이다. 기존 README, 장부, XGBoost 라운드와 로드맵을 갱신하고, 루트 AGENTS에 넓고 깊은 탐색을 기본값으로 고정한다. Drive의 CSV, screenshot, ZIP과 모델은 복사하지 않는다.

**Tech Stack:** Markdown, JSON, Python 3.11, pytest 8

---

### Task 1: 결과 evidence와 탐색 기본값

**Files:**
- Create: `reports/acceptances/xgboost_aggressive_capacity_public_result.json`
- Modify: `AGENTS.md`
- Modify: `tests/test_repository_contract.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_repository_contract.py`에 다음을 추가한다.

```python
def test_xgboost_aggressive_capacity_public_result_is_bound() -> None:
    report = load_json(
        "reports/acceptances/xgboost_aggressive_capacity_public_result.json"
    )
    assert report["status"] == "public_scored"
    assert report["candidate_id"] == "xgboost_aggressive_capacity_v1"
    assert report["run_id"] == "a0b99dd0e7eb41fba2b5ff729b11aeb1"
    assert report["public_score"] == 820.9583317093
    assert report["local_brier"] == 0.2479270213638507
    assert report["local_score"] == 752.5433411090132
    assert report["archive_sha256"] == (
        "f81b5df770733898535d9a1d4a019b7c339aa72d7cbce0cb67a3e49ccb41f43a"
    )
    assert report["scale"] == 1.05
    assert report["mean_shift"] == "linear_extrapolated"
    assert report["members"] == [
        {"structure": "depthwise_d6", "seed": 42, "rounds": 119},
        {"structure": "depthwise_d6", "seed": 2026, "rounds": 134},
        {"structure": "lossguide_l63", "seed": 42, "rounds": 119},
        {"structure": "lossguide_l63", "seed": 2026, "rounds": 106},
    ]


def test_agents_default_to_broad_performance_exploration() -> None:
    agents = read_text("AGENTS.md")
    for phrase in (
        "모델 깊이",
        "GPU 사용량을 사전에 제한하지 않는다",
        "비용과 실행 시간은 안내와 실행 순서에만 사용한다",
        "탐색을 막지 않고",
        "새 폴더·문서·자동화",
    ):
        assert phrase in agents
```

- [ ] **Step 2: RED 확인**

Run: `uv run --no-project --python python3.11 --with pytest==8.4.1 python -m pytest tests/test_repository_contract.py -k 'aggressive_capacity or broad_performance' -q`

Expected: 새 JSON 부재와 AGENTS 문구 부재로 `2 failed`.

- [ ] **Step 3: 작은 결과 JSON 작성**

`reports/acceptances/xgboost_aggressive_capacity_public_result.json`에 아래 값을
strict JSON으로 기록한다.

```json
{
  "status": "public_scored",
  "candidate_id": "xgboost_aggressive_capacity_v1",
  "run_id": "a0b99dd0e7eb41fba2b5ff729b11aeb1",
  "submission_file": "submit_xgboost_v3.zip",
  "submitted_at": "2026-08-11T23:52:02+09:00",
  "public_score": 820.9583317093,
  "local_brier": 0.2479270213638507,
  "local_score": 752.5433411090132,
  "scale": 1.05,
  "mean_shift": "linear_extrapolated",
  "members": [
    {"structure": "depthwise_d6", "seed": 42, "rounds": 119},
    {"structure": "depthwise_d6", "seed": 2026, "rounds": 134},
    {"structure": "lossguide_l63", "seed": 42, "rounds": 119},
    {"structure": "lossguide_l63", "seed": 2026, "rounds": 106}
  ],
  "archive_sha256": "f81b5df770733898535d9a1d4a019b7c339aa72d7cbce0cb67a3e49ccb41f43a",
  "failures": [],
  "sources": {
    "receipt": "https://drive.google.com/file/d/1zoCNN3_odRw3SXLE1VtsxhLIS-nD_m2y/view",
    "candidate_scores": "https://drive.google.com/file/d/13gj77dDRxCoDyPfiy-WVveeY2252nc1E/view",
    "comparison_report": "https://drive.google.com/file/d/1UvRG_J0Sh6Tk0Muz57Qt2fYYb0Klb751/view",
    "submission_zip": "https://drive.google.com/file/d/1ZOll0MQZXIn3nvZruAAWpxh4U55OWtwD/view"
  },
  "public_evidence": "User-provided DACON submission history screenshot verified on 2026-08-12"
}
```

- [ ] **Step 4: AGENTS에 최소 원칙 추가**

기존 목록 끝에 다음 두 항목만 추가한다.

```markdown
- 대회 규정·시간 누출·데이터 손상 위험이 없다면 모델 깊이, leaves, seed 수, feature 수, 실행 시간과 GPU 사용량을 사전에 제한하지 않는다. 비용과 실행 시간은 안내와 실행 순서에만 사용한다. Fold·seed 편차, calibration과 과적합 가능성은 탐색을 막지 않고 최종 선택 자료로 기록한다.
- 기존 파일의 최소 수정을 우선한다. 새 폴더·문서·자동화와 Drive 산출물의 Git 중복은 기존 구조로 해결할 수 없을 때만 추가한다.
```

- [ ] **Step 5: GREEN 확인과 커밋**

Run: `uv run --no-project --python python3.11 --with pytest==8.4.1 python -m pytest tests/test_repository_contract.py -k 'aggressive_capacity or broad_performance' -q`

Expected: `2 passed`.

```bash
git add AGENTS.md reports/acceptances/xgboost_aggressive_capacity_public_result.json tests/test_repository_contract.py
git commit -m "docs: record xgboost public result"
```

### Task 2: 기존 사람용 문서 갱신

**Files:**
- Modify: `README.md`
- Modify: `reports/EXPERIMENT_LEDGER.md`
- Modify: `docs/rounds/05-xgboost.md`
- Modify: `docs/ROADMAP.md`
- Modify: `tests/test_repository_contract.py`

- [ ] **Step 1: 문서 실패 테스트 작성**

```python
def test_xgboost_public_result_is_visible_in_existing_documents() -> None:
    readme = read_text("README.md")
    ledger = read_text("reports/EXPERIMENT_LEDGER.md")
    round_doc = read_text("docs/rounds/05-xgboost.md")
    roadmap = read_text("docs/ROADMAP.md")

    assert "820.9583317093" in readme
    assert "xgboost_aggressive_capacity_v1" in ledger
    assert "820.9583317093" in ledger
    for phrase in (
        "depthwise_d6",
        "lossguide_l63",
        "depthwise_d8",
        "lossguide_l255",
        "714.8814792915847",
        "750.5344761643662",
        "752.5433411090132",
    ):
        assert phrase in round_doc
    assert "정렬된 OOF" in roadmap
    assert "CatBoost" in roadmap
```

- [ ] **Step 2: RED 확인**

Run: `uv run --no-project --python python3.11 --with pytest==8.4.1 python -m pytest tests/test_repository_contract.py -k public_result_is_visible -q`

Expected: 새 결과가 문서에 없어 FAIL.

- [ ] **Step 3: 기존 문서만 최소 수정**

- README의 한눈에 보기와 현재 결론에 확인된 XGBoost Public
  `820.9583317093`을 추가한다.
- 장부에 `xgboost_aggressive_capacity_v1` 한 행을 추가하고 새 JSON에 연결한다.
- 기존 `05-xgboost.md`에 구성, 단일 구조 표, 후처리 단계와 Public 결과를 한
  절로 추가한다. Depth 8과 127·255 leaves가 악화됐음을 명시한다.
- ROADMAP의 XGBoost 절에 동일한 행·프로토콜의 정렬된 CatBoost/XGBoost OOF를
  먼저 확인한 뒤 blend를 설계한다고 한 문단만 추가한다.
- 새 라운드, 이미지, CSV, ZIP, 코드 파일은 만들지 않는다.

- [ ] **Step 4: GREEN 확인과 커밋**

Run: `uv run --no-project --python python3.11 --with pytest==8.4.1 python -m pytest tests/test_repository_contract.py -k public_result_is_visible -q`

Expected: `1 passed`.

```bash
git add README.md reports/EXPERIMENT_LEDGER.md docs/rounds/05-xgboost.md docs/ROADMAP.md tests/test_repository_contract.py
git commit -m "docs: publish xgboost capacity findings"
```

### Task 3: 전체 계약 검증

**Files:**
- Verify only

- [ ] **Step 1: 전체 빠른 테스트**

Run: `uv run --no-project --python python3.11 --with pytest==8.4.1 python -m pytest -q`

Expected: 기존 16개와 새 3개를 합쳐 `19 passed`.

- [ ] **Step 2: 정적 검사**

Run: `git diff --check`

Expected: 출력 없이 exit 0.

Run: `find . -type f -size +5M -not -path './.git/*' -print`

Expected: 출력 없음.

Run: `git status --short`

Expected: clean.

- [ ] **Step 3: push 전 사용자 확인**

변경 파일, 테스트 결과와 로컬 커밋을 보고한다. 명시적 push 요청 전에는 push하지
않는다.
