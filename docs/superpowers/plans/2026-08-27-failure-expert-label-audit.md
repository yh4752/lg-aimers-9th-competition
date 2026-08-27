# Failure Expert Label Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 공식 학습 데이터의 네 시간 cutoff에서 실패 유형 라벨 복원 신뢰도와 유형별 표본 수를 감사하는 Kaggle CPU 한 셀과 검증된 review ZIP을 만든다.

**Architecture:** 기존 `failure_labels.py`의 라벨 정의를 공용 상세 복원 결과로 확장하고 기존 E1 감사와 새 cutoff 감사가 같은 계산을 사용하게 한다. 별도 계약, 집계 실행기, deterministic review artifact와 임베디드 Kaggle 셀을 작은 모듈로 나눈다. 모델, 평가 데이터와 제출 코드는 포함하지 않는다.

**Tech Stack:** Python 3.11/3.12, pandas, NumPy, dataclasses, pytest, JSON, ZIP, tar.gz/base64

---

## File map

- Create `experiments/tree_expert/failure_audit_contract.json`: cutoff, tolerance, 품질·표본 gate와 공식 해시.
- Create `experiments/tree_expert/failure_audit_contracts.py`: 계약 strict parser와 SHA-256.
- Modify `experiments/tree_expert/failure_labels.py`: 기존 라벨 공식을 공유하는 상세 복원 API와 제외 사유 집계.
- Create `experiments/tree_expert/failure_audit.py`: A1~A4 실행, 집계표와 최종 eligibility 판정.
- Create `experiments/tree_expert/failure_audit_artifacts.py`: review-only deterministic ZIP 생성·검증.
- Create `experiments/tree_expert/failure_audit_kaggle.py`: 공식 데이터 탐색, 런타임 봉인과 셀 생성.
- Create `experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py`: Kaggle 복사용 생성 파일.
- Create `tools/build_failure_expert_label_audit_cell.py`: 저장소 루트에서 셀 재생성.
- Create focused tests under `tests/test_tree_expert_failure_audit_*.py`.
- Modify `README.md`: 감사 목적과 실행 상태를 한 단락으로 기록.

### Task 1: Fixed audit contract

**Files:**
- Create: `experiments/tree_expert/failure_audit_contract.json`
- Create: `experiments/tree_expert/failure_audit_contracts.py`
- Test: `tests/test_tree_expert_failure_audit_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from pathlib import Path
import pytest

from experiments.tree_expert.failure_audit_contracts import (
    FailureAuditContractError,
    contract_sha256,
    load_failure_audit_contract,
)


def test_contract_fixes_cutoffs_and_gates():
    contract = load_failure_audit_contract()
    assert contract.cutoffs == (("A1", 2021), ("A2", 2022), ("A3", 2023), ("A4", 2024))
    assert contract.delta_tolerance == 0.02
    assert contract.minimum_coverage == 0.98
    assert contract.minimum_binary_delta_fraction == 0.999
    assert contract.minimum_success_agreement == 0.999
    assert contract.maximum_middle_reverse_overlap == 0.001
    assert contract.minimum_positive_rows == 5_000
    assert contract.minimum_negative_rows == 5_000
    assert contract.types == ("middle", "reverse", "other_failure")


def test_contract_rejects_changed_tolerance(tmp_path: Path):
    source = Path(__file__).parents[1] / "experiments/tree_expert/failure_audit_contract.json"
    changed = source.read_text().replace('"delta_tolerance": 0.02', '"delta_tolerance": 0.03')
    path = tmp_path / "contract.json"
    path.write_text(changed)
    with pytest.raises(FailureAuditContractError, match="fixed audit contract differs"):
        load_failure_audit_contract(path)


def test_contract_hash_is_sha256():
    assert len(contract_sha256()) == 64
```

- [ ] **Step 2: Run the contract tests and confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_audit_contracts.py -q
```

Expected: import failure because `failure_audit_contracts.py` does not exist.

- [ ] **Step 3: Add the fixed JSON contract**

```json
{
  "schema_version": 1,
  "campaign_id": "failure_expert_label_audit_v1",
  "review_only": true,
  "submission_package": false,
  "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
  "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
  "cutoffs": [{"audit_id": "A1", "year": 2021}, {"audit_id": "A2", "year": 2022}, {"audit_id": "A3", "year": 2023}, {"audit_id": "A4", "year": 2024}],
  "types": ["middle", "reverse", "other_failure"],
  "delta_tolerance": 0.02,
  "quality_gates": {"minimum_coverage": 0.98, "minimum_binary_delta_fraction": 0.999, "minimum_success_agreement": 0.999, "maximum_middle_reverse_overlap": 0.001},
  "sample_gates": {"minimum_positive_rows": 5000, "minimum_negative_rows": 5000},
  "runtime": {"wall_seconds": 1800, "maximum_rss_bytes": 12884901888}
}
```

- [ ] **Step 4: Implement a strict immutable parser**

Define `FailureAuditContract` with only the fields used above. Reject extra or missing keys, non-finite values, reordered or changed cutoff/type identities, any changed gate, `review_only != true`, or `submission_package != false`. `contract_sha256()` hashes the exact checked-in bytes.

```python
@dataclass(frozen=True)
class FailureAuditContract:
    campaign_id: str
    official_train_sha256: str
    official_history_sha256: str
    cutoffs: tuple[tuple[str, int], ...]
    types: tuple[str, ...]
    delta_tolerance: float
    minimum_coverage: float
    minimum_binary_delta_fraction: float
    minimum_success_agreement: float
    maximum_middle_reverse_overlap: float
    minimum_positive_rows: int
    minimum_negative_rows: int
    wall_seconds: int
    maximum_rss_bytes: int
```

- [ ] **Step 5: Run tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_audit_contracts.py -q
git add experiments/tree_expert/failure_audit_contract.json experiments/tree_expert/failure_audit_contracts.py tests/test_tree_expert_failure_audit_contracts.py
git commit -m "feat(tree-expert): define failure label audit contract"
```

### Task 2: Shared detailed label recovery

**Files:**
- Modify: `experiments/tree_expert/failure_labels.py`
- Modify: `tests/test_tree_expert_failure_labels.py`

- [ ] **Step 1: Add failing tests for detailed recovery**

```python
from experiments.tree_expert.failure_labels import recover_failure_labels


def test_detailed_recovery_reports_labels_and_exclusions_by_source_position():
    result = recover_failure_labels(_failure_rows(), valid_year=2024, delta_tolerance=1e-9)
    assert result.rows.loc[result.rows["status"].eq("labeled"), "label"].tolist() == [
        "success", "middle", "reverse", "other_failure"
    ]
    assert result.rows["source_position"].is_unique
    assert result.exclusion_counts["no_successor"] == 1


def test_detailed_recovery_marks_duplicate_count():
    rows = pd.concat([_failure_rows(), _failure_rows().iloc[[2]]], ignore_index=True)
    result = recover_failure_labels(rows, valid_year=2024, delta_tolerance=1e-9)
    assert result.exclusion_counts["duplicate_count"] >= 1
```

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_labels.py -q
```

Expected: `recover_failure_labels` import failure.

- [ ] **Step 3: Extract the existing reconstruction into one public result**

Add immutable `FailureLabelRecovery` with a row table and scalar metrics. The table contains one row per input row in original order with only `source_position`, `season`, `status`, `label`, and `exclusion_reason`. Fixed exclusion reasons are:

```python
EXCLUSION_REASONS = (
    "no_successor", "duplicate_count", "skipped_count", "non_binary_delta",
    "success_disagreement", "middle_reverse_overlap", "subtype_on_success",
)
```

```python
@dataclass(frozen=True)
class FailureLabelRecovery:
    rows: pd.DataFrame
    linked_count: int
    labeled_count: int
    coverage: float
    binary_delta_fraction: float
    success_agreement: float
    middle_reverse_overlap: float
    class_counts: Mapping[str, int]
    exclusion_counts: Mapping[str, int]
```

Move the current sort/link/delta/snap logic into `recover_failure_labels`. Keep the same train-only cutoff check and exact label definitions. Rewrite `audit_failure_labels` as a compatibility wrapper that applies its existing gates to this result and returns the same `FailureLabelAudit` interface used by E1.

- [ ] **Step 4: Verify compatibility and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_labels.py tests/test_tree_expert_training.py -q
git add experiments/tree_expert/failure_labels.py tests/test_tree_expert_failure_labels.py
git commit -m "feat(tree-expert): expose detailed failure label recovery"
```

### Task 3: Four-cutoff audit and eligibility decision

**Files:**
- Create: `experiments/tree_expert/failure_audit.py`
- Test: `tests/test_tree_expert_failure_audit.py`

- [ ] **Step 1: Write failing cutoff and decision tests**

```python
from dataclasses import replace
from experiments.tree_expert.failure_audit import run_failure_label_audit


def test_audit_never_passes_rows_beyond_each_cutoff(audit_frame):
    result = run_failure_label_audit(audit_frame, contract=_small_contract())
    assert result.cutoffs["A1"].maximum_source_season == 2021
    assert result.cutoffs["A2"].maximum_source_season == 2022
    assert result.cutoffs["A3"].maximum_source_season == 2023
    assert result.cutoffs["A4"].maximum_source_season == 2024


def test_type_eligibility_is_independent(audit_frame):
    result = run_failure_label_audit(audit_frame, contract=_small_contract())
    assert result.type_decisions["middle"].status == "eligible"
    assert result.type_decisions["reverse"].status == "ineligible"
    assert "positive_rows" in result.type_decisions["reverse"].reason


def test_common_quality_failure_blocks_every_type(audit_frame):
    contract = replace(_small_contract(), minimum_coverage=1.0)
    result = run_failure_label_audit(audit_frame, contract=contract)
    assert {item.status for item in result.type_decisions.values()} == {"ineligible"}
```

- [ ] **Step 2: Run the audit tests and confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_audit.py -q
```

- [ ] **Step 3: Implement immutable cutoff and type results**

```python
@dataclass(frozen=True)
class CutoffAudit:
    audit_id: str
    cutoff_year: int
    status: str
    failed_gates: tuple[str, ...]
    maximum_source_season: int
    row_count: int
    linked_count: int
    labeled_count: int
    coverage: float
    binary_delta_fraction: float
    success_agreement: float
    middle_reverse_overlap: float
    class_counts: Mapping[str, int]
    exclusion_counts: Mapping[str, int]


@dataclass(frozen=True)
class TypeDecision:
    failure_type: str
    status: str
    reason: str
    positive_rows: Mapping[str, int]
    negative_rows: Mapping[str, int]
```

`run_failure_label_audit` filters `season <= cutoff`, calls `recover_failure_labels` with `valid_year=cutoff+1`, computes common gates, aggregates class and exclusions by original season, and decides each type across all four cutoffs. It returns `FailureAuditResult` containing `cutoffs`, `type_decisions`, `class_counts` and `exclusion_counts` DataFrames. It rejects missing columns, duplicate `row_id`, non-binary target, unknown seasons, or an empty prefix.

- [ ] **Step 4: Run tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_audit.py tests/test_tree_expert_failure_labels.py -q
git add experiments/tree_expert/failure_audit.py tests/test_tree_expert_failure_audit.py
git commit -m "feat(tree-expert): audit failure labels by cutoff"
```

### Task 4: Deterministic review-only artifact

**Files:**
- Create: `experiments/tree_expert/failure_audit_artifacts.py`
- Test: `tests/test_tree_expert_failure_audit_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

```python
def test_review_is_deterministic_and_contains_only_aggregate_evidence(tmp_path):
    first = create_failure_audit_review(RESULT, tmp_path / "first.zip", BINDINGS)
    second = create_failure_audit_review(RESULT, tmp_path / "second.zip", BINDINGS)
    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        assert set(archive.namelist()) == {
            "audit_summary.json", "class_counts.csv", "exclusion_counts.csv",
            "cutoffs/A1.json", "cutoffs/A2.json", "cutoffs/A3.json",
            "cutoffs/A4.json", "audit.log", "manifest.json",
        }
        assert "row_id" not in archive.read("class_counts.csv").decode()


def test_review_rejects_changed_member(tmp_path):
    review = create_failure_audit_review(RESULT, tmp_path / "review.zip", BINDINGS)
    tampered = rewrite_member(review, "cutoffs/A2.json", b"{}")
    with pytest.raises(FailureAuditArtifactError, match="member evidence differs"):
        verify_failure_audit_review(tampered, BINDINGS)
```

- [ ] **Step 2: Run tests and confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_audit_artifacts.py -q
```

- [ ] **Step 3: Implement bindings and atomic ZIP publication**

```python
@dataclass(frozen=True)
class FailureAuditBindings:
    contract_sha256: str
    code_sha256: str
    official_train_sha256: str
    official_history_sha256: str
```

Use canonical JSON, fixed ZIP timestamps `(2026, 1, 1, 0, 0, 0)`, sorted members, member size/SHA-256, a temporary sibling and `os.replace`. Manifest identity is `failure_expert_label_audit_review_v1`, `review_only=true`, `submission_package=false`. Verification rejects duplicates, encryption, symlinks, unsafe paths, undeclared or missing members and hash/size differences.

- [ ] **Step 4: Run tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_audit_artifacts.py -q
git add experiments/tree_expert/failure_audit_artifacts.py tests/test_tree_expert_failure_audit_artifacts.py
git commit -m "feat(tree-expert): package failure label audit"
```

### Task 5: Kaggle discovery and generated CPU cell

**Files:**
- Create: `experiments/tree_expert/failure_audit_kaggle.py`
- Create: `experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py`
- Create: `tools/build_failure_expert_label_audit_cell.py`
- Test: `tests/test_tree_expert_failure_audit_kaggle.py`
- Test: `tests/test_tree_expert_failure_audit_cell.py`

- [ ] **Step 1: Write failing runtime and cell tests**

```python
def test_runtime_archive_imports_in_isolation(tmp_path):
    extract(runtime_archive(), tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", "import experiments.tree_expert.failure_audit_kaggle; import experiments.tree_expert.failure_audit"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_official_discovery_requires_one_exact_root(tmp_path):
    official_fixture(tmp_path / "official")
    assert discover_official_data(tmp_path, testing=True).name == "official"
    official_fixture(tmp_path / "duplicate")
    with pytest.raises(FailureAuditKaggleError, match="official data count must be one"):
        discover_official_data(tmp_path, testing=True)


def test_generated_cell_is_deterministic_small_and_review_only(tmp_path):
    first = build_failure_audit_cell(tmp_path / "first.py")
    second = build_failure_audit_cell(tmp_path / "second.py")
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 900_000
    compile(first.read_text(), str(first), "exec")
    text = first.read_text()
    assert "FAIL_AUDIT_SUCCESS" in text
    assert "FAIL_AUDIT_ERROR" in text
    assert "sample_submission" not in text.lower()
    assert "submission.zip" not in text.lower()
```

- [ ] **Step 2: Run tests and confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_audit_kaggle.py tests/test_tree_expert_failure_audit_cell.py -q
```

- [ ] **Step 3: Implement deterministic runtime archive and official discovery**

Runtime members are only package init, `contracts.py` required by package init, `failure_labels.py`, the four new audit modules and contract JSON. `discover_official_data` searches for a directory containing exactly one root-level `train.csv` and `trackman_history.csv`; production verification requires the contract hashes. No RF input, model or resume is accepted.

- [ ] **Step 4: Implement the generated one-cell flow**

The generated cell must:

1. verify and safely extract the embedded tar.gz runtime;
2. print `FAIL_AUDIT_CODE_READY`;
3. discover and hash the single official dataset;
4. print `FAIL_AUDIT_DATA_VERIFIED`;
5. read only `train.csv` and run A1~A4 before a 30-minute deadline;
6. write and re-verify one review ZIP;
7. copy it to `/kaggle/working/failure_expert_label_audit_review.zip`;
8. print cutoff and type decisions followed by `FAIL_AUDIT_SUCCESS`;
9. on failure print one complete `FAIL_AUDIT_ERROR` and no success artifact.

Do not install packages, inspect GPUs, read evaluation data or create resume/model/submission outputs.

- [ ] **Step 5: Build twice and verify byte stability**

```bash
artifacts/tabm_submission_python311/bin/python tools/build_failure_expert_label_audit_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py
artifacts/tabm_submission_python311/bin/python tools/build_failure_expert_label_audit_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py
```

Expected: identical hashes and a file smaller than 900,000 bytes.

- [ ] **Step 6: Run tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_failure_audit_kaggle.py tests/test_tree_expert_failure_audit_cell.py -q
git add experiments/tree_expert/failure_audit_kaggle.py experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py tools/build_failure_expert_label_audit_cell.py tests/test_tree_expert_failure_audit_kaggle.py tests/test_tree_expert_failure_audit_cell.py
git commit -m "feat(tree-expert): add failure audit Kaggle cell"
```

### Task 6: Documentation, regression and user handoff

**Files:**
- Modify: `README.md`
- Test: all failure audit tests and existing tree expert regression tests

- [ ] **Step 1: Add a short README entry**

```markdown
### 실패 유형 라벨 감사

공식 학습 데이터의 누적 상태로 `middle`, `reverse`, `other_failure`를 시간 cutoff별로
복원할 수 있는지 확인하는 CPU 감사다. 최소 한 유형이 모든 고정 gate를 통과해야만
실패 유형 전문가 OOF 설계로 넘어간다.
```

Link the approved design and implementation plan. Do not claim any type is eligible before the user-run result exists.

- [ ] **Step 2: Run focused and regression tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_failure_labels.py \
  tests/test_tree_expert_failure_audit_*.py \
  tests/test_tree_expert_training.py \
  tests/test_tree_expert_e2_*.py \
  tests/test_tree_expert_rf_*.py -q
```

Expected: all selected tests pass. Existing user-owned TabM changes are not included or modified.

- [ ] **Step 3: Run static and boundary checks**

```bash
artifacts/tabm_submission_python311/bin/python -m compileall -q \
  experiments/tree_expert/failure_labels.py \
  experiments/tree_expert/failure_audit*.py \
  tools/build_failure_expert_label_audit_cell.py
git diff --check
rg -n "test|submission|predict|model|CatBoost|torch|cuda" \
  experiments/tree_expert/failure_audit*.py \
  experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py
```

Every search hit must be a prohibition, a fixed official-data label, or a test-mode parameter. Production code must not load evaluation data, instantiate a model or create a submission artifact.

- [ ] **Step 4: Commit docs and verify final state**

```bash
git add README.md
git commit -m "docs: explain failure label audit"
git status --short
git log --oneline -10
```

- [ ] **Step 5: Hand off one complete user-run operation**

Tell the user:

- purpose: label reliability and sample audit only;
- Kaggle input: official `lg-aimers-9th-data` only;
- accelerator: CPU, GPU disabled;
- Internet: off;
- cell: copy all of `KAGGLE_FAILURE_AUDIT_CELL.py` into one cell;
- runtime: 10~30 minutes;
- rerun safety: deterministic full rerun, no resume required;
- success line: `FAIL_AUDIT_SUCCESS`;
- error line: complete `FAIL_AUDIT_ERROR`;
- return file: `failure_expert_label_audit_review.zip`;
- impact of skipping: failure-type experts remain blocked because C3 previously lacked reliable labels.

Stop before executing the full-data audit locally. The user runs the Kaggle cell and returns the review ZIP. Only after verifying that artifact may a failure-expert OOF design begin.
