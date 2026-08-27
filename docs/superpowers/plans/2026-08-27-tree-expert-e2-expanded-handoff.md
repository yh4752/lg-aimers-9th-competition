# Tree Expert E2 Expanded Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the E2 Kaggle cell consume a handoff whose nested review and resume ZIPs were automatically expanded by Kaggle without weakening artifact identity checks.

**Architecture:** Normalize artifact discovery so a resume nested below a handoff is not a second input. Materialize the selected resume directly from either an archive or an expanded directory using the repository's deterministic ZIP format; the production recovery layer still enforces the pinned full resume SHA-256 and manifest bindings.

**Tech Stack:** Python 3.11/3.12, `pathlib`, `zipfile`, pytest, generated single-cell Kaggle runtime.

---

### Task 1: Reproduce expanded handoff discovery

**Files:**
- Modify: `tests/test_tree_expert_e2_kaggle.py`
- Modify: `experiments/tree_expert/e2_kaggle.py`

- [ ] **Step 1: Write the failing discovery test**

Create an expanded `tree_expert_e2_handoff_v1` directory containing a nested expanded `tree_expert_e2_resume_v1` directory, call `discover_inputs`, and assert that the outer handoff is the single resume source.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q \
  tests/test_tree_expert_e2_kaggle.py::test_expanded_handoff_and_nested_resume_are_one_source
```

Expected: FAIL with `resume count must be zero or one; found=2`.

- [ ] **Step 3: Implement minimal candidate normalization**

In `_artifact_candidates`, discard only a `tree_expert_e2_resume_v1` directory that is a strict descendant of an already discovered `tree_expert_e2_handoff_v1` directory. Keep independent artifacts ambiguous.

- [ ] **Step 4: Run discovery tests and verify GREEN**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_e2_kaggle.py
```

Expected: all tests pass, including the existing independent-two-resume rejection.

### Task 2: Restore deterministic resume ZIP and regenerate the cell

**Files:**
- Modify: `tests/test_tree_expert_e2_kaggle.py`
- Modify: `tests/test_tree_expert_e2_kaggle_cell.py`
- Modify: `experiments/tree_expert/e2_kaggle.py`
- Regenerate: `experiments/tree_expert/KAGGLE_E2_CELL.py`
- Modify: `docs/TREE_EXPERT_E2_KAGGLE.md`

- [ ] **Step 1: Write the failing materialization test**

Build a deterministic resume ZIP fixture, expand it below a handoff directory, call `materialize_resume_source`, and assert that the rebuilt ZIP has the exact original SHA-256. Also assert that more than one nested resume is rejected.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q \
  tests/test_tree_expert_e2_kaggle.py -k 'materialize or nested'
```

Expected: FAIL because `materialize_resume_source` is absent.

- [ ] **Step 3: Implement deterministic materialization**

Add a ZIP writer using sorted member names, timestamp `(2026, 1, 1, 0, 0, 0)`, deflate level 6, and mode `0o100644`. For an expanded handoff, prefer its exact nested resume ZIP; otherwise rebuild exactly one nested expanded resume directory. Reject symlinks, missing resumes, and multiple nested resumes.

- [ ] **Step 4: Route the generated cell through the new function**

Replace handoff-wide re-zipping with `materialize_resume_source(found.resume, WORK / "materialized" / "tree_expert_e2_resume.zip")`. Keep compact-input materialization unchanged.

- [ ] **Step 5: Regenerate and verify**

Run:

```bash
artifacts/tabm_submission_python311/bin/python tools/build_tree_expert_e2_kaggle_cell.py
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_e2_*.py
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/tree_expert/e2_kaggle.py experiments/tree_expert/KAGGLE_E2_CELL.py
git diff --check
```

Expected: all E2 tests pass, the generated cell is below 1 MB, and rebuilding it twice produces the same SHA-256.
