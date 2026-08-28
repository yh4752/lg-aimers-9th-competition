# Failure Label Audit Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and hand off the existing S0 Kaggle CPU audit cell so the user can determine which failure types are eligible before any GPU expert code is built.

**Architecture:** Reuse the already isolated, review-only failure audit implementation. Verify its contract, artifact integrity, runtime inventory, generated one-cell program, and fixture behavior without running official data. The user runs the full audit on Kaggle and returns one verified review ZIP; S1/S2 implementation then uses only eligible types declared by that artifact.

**Tech Stack:** Python 3.11, pandas, NumPy, pytest, deterministic ZIP artifacts, Kaggle CPU

---

### Task 1: Confirm the S0 contract matches the approved portfolio spec

**Files:**
- Inspect: `experiments/tree_expert/failure_audit_contract.json`
- Inspect: `experiments/tree_expert/failure_audit_contracts.py`
- Test: `tests/test_tree_expert_failure_audit_contracts.py`
- Test: `tests/test_tree_expert_failure_audit.py`

- [ ] **Step 1: Run the contract tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_failure_audit_contracts.py \
  tests/test_tree_expert_failure_audit.py -q
```

Expected: all tests pass and no official full-data work starts.

- [ ] **Step 2: Compare the fixed gates with the approved design**

```python
expected = {
    "minimum_coverage": 0.98,
    "minimum_binary_delta_fraction": 0.999,
    "minimum_success_agreement": 0.999,
    "maximum_middle_reverse_overlap": 0.001,
    "minimum_positive_rows": 5000,
    "minimum_negative_rows": 5000,
}
```

Expected: the JSON contract contains exactly these values and types are parsed without coercing booleans or strings.

### Task 2: Verify deterministic, review-only artifacts

**Files:**
- Inspect: `experiments/tree_expert/failure_audit_artifacts.py`
- Test: `tests/test_tree_expert_failure_audit_artifacts.py`

- [ ] **Step 1: Run artifact tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_failure_audit_artifacts.py -q
```

Expected: deterministic review creation, member hash verification, and tamper rejection pass.

- [ ] **Step 2: Confirm the artifact cannot be mistaken for a submission**

```python
assert manifest["artifact_kind"] == "failure_expert_label_audit_review_v1"
assert manifest["submission_package"] is False
assert "model/" not in members
assert "script.py" not in members
```

Expected: review ZIP only; no model or submission entry point.

### Task 3: Verify the Kaggle one-cell program

**Files:**
- Inspect: `experiments/tree_expert/failure_audit_kaggle.py`
- Inspect: `experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py`
- Test: `tests/test_tree_expert_failure_audit_kaggle.py`
- Test: `tests/test_tree_expert_failure_audit_cell.py`

- [ ] **Step 1: Run cell and runtime-inventory tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_failure_audit_kaggle.py \
  tests/test_tree_expert_failure_audit_cell.py -q
```

Expected: embedded runtime identity, official hash checks, syntax, deterministic rendering, and one-megabyte limit pass.

- [ ] **Step 2: Compile the generated cell**

```bash
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py
```

Expected: exit status 0.

- [ ] **Step 3: Record the exact handoff identity**

```python
from hashlib import sha256
from pathlib import Path

path = Path("experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py")
payload = path.read_bytes()
print(len(payload), sha256(payload).hexdigest())
assert len(payload) < 1_000_000
```

Expected: one stable size and SHA-256 are reported for the user.

### Task 4: Run the focused regression set

**Files:**
- Test: `tests/test_tree_expert_failure_labels.py`
- Test: `tests/test_tree_expert_failure_audit_contracts.py`
- Test: `tests/test_tree_expert_failure_audit.py`
- Test: `tests/test_tree_expert_failure_audit_artifacts.py`
- Test: `tests/test_tree_expert_failure_audit_kaggle.py`
- Test: `tests/test_tree_expert_failure_audit_cell.py`

- [ ] **Step 1: Run all S0 tests together**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_failure_labels.py \
  tests/test_tree_expert_failure_audit_contracts.py \
  tests/test_tree_expert_failure_audit.py \
  tests/test_tree_expert_failure_audit_artifacts.py \
  tests/test_tree_expert_failure_audit_kaggle.py \
  tests/test_tree_expert_failure_audit_cell.py -q
```

Expected: all focused tests pass with zero failures.

- [ ] **Step 2: Check the working diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; no production change is needed unless a focused test exposed a real defect.

### Task 5: Hand off the user-run S0 audit

**Files:**
- Deliver: `experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py`
- Receive later: `failure_expert_label_audit_review.zip`

- [ ] **Step 1: Give one complete Kaggle operation**

```text
Purpose: audit middle/reverse/other_failure label eligibility only
Inputs: official lg-aimers-9th-data dataset
Accelerator: None (CPU)
Internet: Off
Expected runtime: 10–30 minutes
Rerun safety: deterministic restart from the beginning; no GPU state is lost
Success text: FAIL_AUDIT_SUCCESS review=<path>
Return artifact: failure_expert_label_audit_review.zip
Error text to return: FAIL_AUDIT_ERROR stage=<stage> type=<type> message=<message>
```

Expected: the user receives the exact cell path, input configuration, success log, and one result filename.

- [ ] **Step 2: Stop before S1/S2 production implementation**

```python
if not verified_audit_review_available:
    raise RuntimeError("S1/S2 candidate set is not fixed")
```

Expected: no failure-expert GPU campaign or submission package is created before the audit identifies eligible types.
