# Tree Privileged P-only Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified P-only Kaggle campaign that skips unusable TrackMan teacher work and starts CatBoost GPU workers with the CUDA-safe `spawn` method.

**Architecture:** Keep the existing Tree Privileged feature, decision, artifact, and submission gates. Seal the production grid to candidate `P`, make teacher evidence conditional on candidates that actually use it, and pass an explicit spawn multiprocessing context to the existing two-GPU scheduler.

**Tech Stack:** Python 3.11, pytest, pandas, CatBoost 1.2.10, deterministic ZIP artifacts.

---

### Task 1: Seal the campaign to P-only

**Files:**
- Modify: `tests/test_tree_privileged_contracts.py`
- Modify: `experiments/tree_privileged/contract.json`
- Modify: `experiments/tree_privileged/contracts.py`

- [ ] **Step 1: Write the failing contract test**

Change the production-grid assertion to:

```python
assert contract.candidates == ("P",)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv/bin/pytest -q tests/test_tree_privileged_contracts.py::test_contract_seals_budget_candidates_and_submission_boundary`

Expected: FAIL because the current contract still returns five candidates.

- [ ] **Step 3: Make the minimal contract change**

Set the JSON candidates to:

```json
"candidates": ["P"]
```

Set the sealed Python constant to:

```python
_CANDIDATES = ("P",)
```

Keep folds, seeds, gates and `submission_package: false` unchanged.

- [ ] **Step 4: Run the contract tests**

Run: `.venv/bin/pytest -q tests/test_tree_privileged_contracts.py tests/test_tree_privileged_inputs.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_privileged/contract.json experiments/tree_privileged/contracts.py tests/test_tree_privileged_contracts.py
git commit -m "fix: seal privileged recovery to profile candidate"
```

### Task 2: Skip teacher work for P and use CUDA-safe spawn

**Files:**
- Modify: `tests/test_tree_privileged_runner.py`
- Modify: `experiments/tree_privileged/runner.py`

- [ ] **Step 1: Write failing tests**

Add one test that asserts the production candidates do not require teacher evidence:

```python
def test_p_only_grid_does_not_require_teacher_evidence() -> None:
    assert _requires_teacher(("P",)) is False
    assert _requires_teacher(("P", "D15")) is True
```

Add one test that wraps `ProcessPoolExecutor`, runs a single fixture job and records:

```python
assert captured["start_method"] == "spawn"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest -q tests/test_tree_privileged_runner.py -k 'requires_teacher or spawn_context'`

Expected: collection or assertion failure because `_requires_teacher` and explicit spawn context do not exist.

- [ ] **Step 3: Implement the minimal scheduler fix**

In `runner.py`:

```python
import multiprocessing as mp


def _requires_teacher(candidates: Sequence[str]) -> bool:
    return any(candidate != "P" for candidate in candidates)
```

Load teacher evidence in `_worker` only when `_requires_teacher((job.candidate_id,))` is true. Build teacher caches in `run_campaign` only when `_requires_teacher(load_contract().candidates)` is true. Construct the executor with:

```python
ProcessPoolExecutor(
    max_workers=len(gpu_ids),
    mp_context=mp.get_context("spawn"),
)
```

- [ ] **Step 4: Run scheduler and campaign regression tests**

Run: `.venv/bin/pytest -q tests/test_tree_privileged_runner.py tests/test_tree_privileged_training.py tests/test_tree_privileged_decisions.py`

Expected: all tests pass, including two-GPU assignment and completed-job reuse.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_privileged/runner.py tests/test_tree_privileged_runner.py
git commit -m "fix: isolate privileged CatBoost GPU workers"
```

### Task 3: Regenerate and verify the handoff artifacts

**Files:**
- Regenerate: `experiments/tree_privileged/KAGGLE_CELL.py`
- Regenerate: `artifacts/tree_privileged_p_only_input.zip`

- [ ] **Step 1: Build the one-cell Kaggle launcher**

Run: `.venv/bin/python tools/build_tree_privileged_kaggle_cell.py`

Expected: `TREE_PRIV_KAGGLE_CELL_READY` with a SHA-256 and size below 1 MB.

- [ ] **Step 2: Rebind the unchanged accepted E2 payloads**

Read the four authorized payload members from `artifacts/tree_privileged_input.zip`, verify each against its old manifest, create a fresh manifest with the current `contract_sha256()`, and write them through `experiments.tree_privileged.inputs._write_zip` to `artifacts/tree_privileged_p_only_input.zip`.

Expected members:

```text
e2/model_delivery.bundle
e2/oof/2022.csv
e2/oof/2023.csv
e2/oof/2024.csv
manifest.json
```

- [ ] **Step 3: Verify the new input and generated runtime**

Run the input verifier against a temporary extraction directory, confirm the E2 handoff SHA-256 is `4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f`, and import `experiments.tree_privileged.runner` from the embedded runtime archive in an isolated Python process.

Expected: input verification and isolated runtime import both exit zero.

- [ ] **Step 4: Run the complete relevant test suite**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_tree_privileged_*.py \
  tests/test_temporal_portfolio_lupi.py \
  tests/test_tree_expert_e2_*.py
```

Expected: zero failed tests.

- [ ] **Step 5: Commit source changes only**

```bash
git add experiments/tree_privileged/KAGGLE_CELL.py
git commit -m "build: refresh privileged P-only Kaggle cell"
```

Do not add the 54 MB input ZIP to Git. Report its absolute path, size and SHA-256 to the user.
