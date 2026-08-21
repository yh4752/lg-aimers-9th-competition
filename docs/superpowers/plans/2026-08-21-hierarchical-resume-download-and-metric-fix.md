# Hierarchical Resume Download and Metric Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop per-epoch browser downloads, repair paired segment metrics, and safely reuse the exact completed legacy OOF resume.

**Architecture:** Keep checkpoint promotion and recursive ZIP verification unchanged, but separate verified local publication from browser delivery through a small policy gate. Source segment evidence from the row-aligned H1 prediction, and provide an exact-SHA, exact-state legacy resume upgrader before normal restore.

**Tech Stack:** Python 3.11, pandas, pytest, deterministic ZIP artifacts, Colab generated single cell.

---

### Task 1: Repair paired segment alignment

**Files:**
- Modify: `experiments/hierarchical_tabm/metrics.py`
- Test: `tests/test_hierarchical_tabm_metrics.py`

- [ ] Add a failing test whose anchor has only the official Stage C columns while the candidate has all `SEGMENT_COLUMNS` and `game_month`.
- [ ] Run `python -m pytest tests/test_hierarchical_tabm_metrics.py -q` and confirm `paired evidence missing segment`.
- [ ] After exact row/target alignment, copy candidate segment/month columns in anchor row order; reject missing candidate columns and conflicting same-name anchor values.
- [ ] Re-run the metric suite and confirm it passes.

### Task 2: Separate snapshot publication from browser delivery

**Files:**
- Modify: `experiments/hierarchical_tabm/colab.py`
- Test: `tests/test_hierarchical_tabm_colab.py`

- [ ] Add failing controlled-clock tests proving consecutive new epochs currently invoke the callback every time and terminal publication duplicates the final download.
- [ ] Run the focused tests and confirm the callback count is too high.
- [ ] Preserve verified local snapshots for every new epoch, deliver the first active checkpoint immediately, deliver later active snapshots only when `download_due`, deliver newly completed folds, and leave terminal downloads to the generated cell.
- [ ] Re-run the focused tests and confirm exact callback timing and recursive verification.

### Task 3: Upgrade the exact completed legacy resume

**Files:**
- Modify: `experiments/hierarchical_tabm/inputs.py`
- Modify: `experiments/hierarchical_tabm/artifacts.py`
- Modify: `experiments/hierarchical_tabm/runner.py`
- Test: `tests/test_hierarchical_tabm_inputs.py`
- Test: `tests/test_hierarchical_tabm_artifacts.py`
- Test: `tests/test_hierarchical_tabm_runner.py`

- [ ] Add failing fixtures for the exact old code SHA with two completed OOF jobs and for altered SHA/state/member evidence.
- [ ] Confirm the valid old resume is rejected by current exact binding and all invalid variants remain rejected.
- [ ] Implement a narrow migration that recursively verifies the old bundle, validates the exact pre-decision state and completed job evidence, records source provenance, and publishes a new-binding resume.
- [ ] Run focused input/artifact/runner tests and confirm only the sealed legacy case succeeds.

### Task 4: Regenerate and verify the Colab handoff

**Files:**
- Modify: `experiments/hierarchical_tabm/runtime_inventory.py`
- Regenerate: `experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py`
- Test: `tests/test_hierarchical_tabm_colab_cell.py`
- Update: `docs/HIERARCHICAL_TABM_COLAB_RUNBOOK.md`

- [ ] Update expected markers to distinguish local snapshots from user download requests and explain reuse of the `(24)` resume.
- [ ] Regenerate the checked-in cell twice and assert byte-identical output.
- [ ] Run all hierarchical tests, `py_compile`, `git diff --check`, and confirm unrelated dirty files are untouched.
- [ ] Commit only hierarchical implementation, tests, generated cell, and documentation; do not push.
