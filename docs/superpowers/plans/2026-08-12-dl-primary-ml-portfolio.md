# DL Primary ML Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 독립 DL을 다음 주력 연구로 고정하면서 CatBoost와 XGBoost를 기준선·최종 앙상블 축으로 유지한다.

**Architecture:** 새 시스템을 만들지 않고 기존 저장소 계약을 확장한다. `AGENTS.md`는 모든 후속 작업에 적용할 불변 원칙을, `docs/ROADMAP.md`는 실제 우선순위를 담당하며 기존 저장소 계약 테스트 하나가 두 문서의 핵심 의미와 순서를 고정한다.

**Tech Stack:** Markdown, Python 3.11, pytest 8.4.1

---

### Task 1: DL 주력·ML 공존 가드레일 적용

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/ROADMAP.md`
- Test: `tests/test_repository_contract.py`

- [ ] **Step 1: 실패하는 저장소 계약 테스트 작성**

`tests/test_repository_contract.py`에 다음 테스트를 추가한다.

```python
def test_roadmap_keeps_dl_primary_without_discarding_ml() -> None:
    agents = read_text("AGENTS.md")
    roadmap = read_text("docs/ROADMAP.md")

    for phrase in (
        "다음 주력 연구는 독립 DL",
        "ML 단독 미세 조정",
        "독립 DL 준비나 실행을 지연시킬 수 없다",
        "TabM residual",
        "DL 계열 전체",
        "CatBoost와 XGBoost",
    ):
        assert phrase in agents

    stages = ("독립 DL 탐색", "DL 내부 앙상블", "ML+DL 앙상블")
    positions = [roadmap.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "보조 트랙" in roadmap
    assert "유망 후보" in roadmap
    assert "정렬된 OOF" in roadmap
```

- [ ] **Step 2: 새 테스트가 의도한 이유로 실패하는지 확인**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest \
  tests/test_repository_contract.py::test_roadmap_keeps_dl_primary_without_discarding_ml \
  -q
```

Expected: `AGENTS.md`에 `다음 주력 연구는 독립 DL`이 없어 FAIL.

- [ ] **Step 3: AGENTS에 최소 영구 원칙 추가**

`AGENTS.md` 끝에 다음 두 항목을 추가한다.

```markdown
- 다음 주력 연구는 독립 DL이다. 우선순위는 독립 DL 탐색, DL 내부 앙상블,
  ML+DL 앙상블 순서다. CatBoost와 XGBoost는 강한 기준선과 최종 앙상블 축으로
  유지하지만 ML 단독 미세 조정은 독립 DL 준비나 실행을 지연시킬 수 없다.
- 기존 `TabM residual` 기각은 그 잔차형 구성에만 적용한다. 독립 TabM이나 다른
  DL 구조와 DL 계열 전체의 실패로 확대하지 않는다.
```

기존의 모델 규모·GPU 비제한 원칙은 중복해서 다시 쓰지 않는다.

- [ ] **Step 4: ROADMAP의 실제 우선순위 교체**

`docs/ROADMAP.md`의 `## 다음 순서`를 다음 구조로 교체한다.

```markdown
## 다음 주력 순서

### 1. 독립 DL 탐색

트리 예측에 의존하지 않는 독립 DL 구조와 넓은 용량·최적화 범위를 먼저 탐색한다.
초기 후보 전체에 장기 실행 의무를 두지 않고 서로 다른 표현과 학습 설정을 폭넓게
확인한다.

### 2. 유망 DL 후보 심화와 DL 내부 앙상블

초기 결과가 유망한 후보에 충분한 학습 시간, 여러 seed와 필요한 체크포인트를
적용한다. 단독 성능과 fold 안정성을 확인하고 상위 구조·seed의 DL 내부 앙상블을
평가한다.

### 3. 정렬된 OOF 기반 ML+DL 앙상블

살아남은 DL 후보와 CatBoost·XGBoost의 동일 행·동일 검증 프로토콜 OOF를
정렬한다. 단독 점수뿐 아니라 오차 다양성과 앙상블 기여도를 함께 평가한다.

## 보조 트랙

- R9 preflight·시간 전이 OOF 검증·행 독립성·package gate 이전
- 열린 calibration family 재설계
- XGBoost 체크포인트·해시 계약과 Colab 이전
- 동료 저장소 연구의 출처·누출·행 독립성 검토 후 선별 이식

보조 트랙은 독립 DL 준비나 사용자 GPU 실행을 지연시키지 않는 범위에서 진행한다.
```

기존 `## 보류` 항목은 그대로 유지한다.

- [ ] **Step 5: 집중 테스트를 통과시키기**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest \
  tests/test_repository_contract.py::test_roadmap_keeps_dl_primary_without_discarding_ml \
  -q
```

Expected: `1 passed`.

- [ ] **Step 6: 전체 저장소 계약 검증**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest -q
git diff --check
git status --short
```

Expected: 전체 pytest PASS, `git diff --check` 출력 없음, 변경 파일은 승인된 세
파일뿐임.

- [ ] **Step 7: 커밋**

```bash
git add AGENTS.md docs/ROADMAP.md tests/test_repository_contract.py
git commit -m "docs: prioritize independent dl research"
```
