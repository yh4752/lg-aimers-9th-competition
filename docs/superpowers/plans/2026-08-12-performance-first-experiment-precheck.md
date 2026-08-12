# Performance-First Experiment Precheck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모든 새 ML·DL 실험이 smoke·비용·OOM·단일 실패 때문에 보수적으로 축소되지 않도록 필수 사전 확인을 저장소 계약으로 고정한다.

**Architecture:** 루트 `AGENTS.md`가 설계 전에 표시할 네 항목을 정의하고 기존 `docs/EXPERIMENT_CONTRACT.md`가 각 항목의 실행 의미를 정의한다. 기존 pytest 저장소 계약 하나가 두 문서의 핵심 의미를 함께 고정하며 별도 상태나 실행 시스템은 추가하지 않는다.

**Tech Stack:** Markdown, Python 3.11, pytest 8.4.1

---

### Task 1: 성능 우선 사전 확인 계약 적용

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/EXPERIMENT_CONTRACT.md`
- Test: `tests/test_repository_contract.py`

- [ ] **Step 1: 실패하는 계약 테스트 작성**

`tests/test_repository_contract.py`에 다음 테스트를 추가한다.

```python
def test_every_experiment_requires_performance_first_precheck() -> None:
    agents = read_text("AGENTS.md")
    contract = read_text("docs/EXPERIMENT_CONTRACT.md")

    for phrase in (
        "[성능 우선 확인]",
        "smoke 결과를 성능 근거로 사용하지 않았는가",
        "첫 본 실험에 큰 모델·긴 학습·넓은 탐색",
        "최고 설정이 탐색 경계에 있으면",
        "OOM·시간·비용 또는 단일 설정 실패",
        "실험 설계는 미완성",
    ):
        assert phrase in agents

    for phrase in (
        "기술 확인 전용",
        "성능을 판단할 수 있는 충분한",
        "다음 범위를 확장",
        "mixed precision",
        "gradient accumulation",
        "모델 계열 기각 근거가 아니다",
        "해당 설정만 종료",
    ):
        assert phrase in contract
```

- [ ] **Step 2: RED 확인**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest \
  tests/test_repository_contract.py::test_every_experiment_requires_performance_first_precheck \
  -q
```

Expected: `AGENTS.md`에 `[성능 우선 확인]`이 없어 FAIL.

- [ ] **Step 3: AGENTS에 네 항목 사전 확인 추가**

`AGENTS.md` 끝에 다음 항목을 추가한다.

````markdown
- 모든 새 ML·DL 실험 설계 전에 아래 블록을 사용자에게 표시한다. 하나라도
  충족하지 않으면 실험 설계는 미완성이다.

  ```text
  [성능 우선 확인]
  1. smoke 결과를 성능 근거로 사용하지 않았는가?
  2. 첫 본 실험에 큰 모델·긴 학습·넓은 탐색이 포함됐는가?
  3. 최고 설정이 탐색 경계에 있으면 다음 범위를 확장하는가?
  4. OOM·시간·비용 또는 단일 설정 실패를 후보 계열 종료 근거로 쓰지 않는가?
  ```
````

- [ ] **Step 4: 실행 계약에 의미 추가**

`docs/EXPERIMENT_CONTRACT.md`의 `## 고정 원칙` 다음에 아래 절을 추가한다.

```markdown
## 성능 우선 사전 확인

- smoke는 import, CUDA, 데이터 흐름과 출력 형태의 기술 확인 전용이며 성능을
  판단하지 않는다.
- 첫 본 실험은 성능을 판단할 수 있는 충분한 모델 규모, 학습 시간과 탐색 폭을
  포함한다.
- 최선 결과가 탐색 경계에 있으면 그 값을 상한으로 확정하지 않고 다음 범위를
  확장한다.
- OOM은 batch size, mixed precision, gradient accumulation과 checkpoint 재시작으로
  대응하며 모델 계열 기각 근거가 아니다.
- 비용과 실행 시간은 안내와 순서 정보로만 사용한다.
- 한 seed, 설정, residual 또는 feature view 실패는 해당 설정만 종료한다.
```

- [ ] **Step 5: 집중 GREEN 확인**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest \
  tests/test_repository_contract.py::test_every_experiment_requires_performance_first_precheck \
  -q
```

Expected: `1 passed`.

- [ ] **Step 6: 전체 범위 검증**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 python -m pytest -q
git diff --check
git diff --name-only
```

Expected: 전체 pytest PASS, 형식 오류 없음, 변경 파일은 `AGENTS.md`,
`docs/EXPERIMENT_CONTRACT.md`, `tests/test_repository_contract.py`뿐임.

- [ ] **Step 7: 커밋**

```bash
git add AGENTS.md docs/EXPERIMENT_CONTRACT.md tests/test_repository_contract.py
git commit -m "docs: require performance first experiment precheck"
```
