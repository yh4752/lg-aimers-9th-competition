# Kaggle Extracted Resume Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow the generated TabM Kaggle cell to securely resume from either an intact resume ZIP or Kaggle's automatically extracted Dataset directory.

**Architecture:** Add a small, standard-library-only input normalizer embedded by the renderer. It discovers manifest-declared resume candidates, validates paths and hashes, rebuilds extracted input under `/kaggle/working`, and passes every form through the existing resume verifier before GPU work.

**Tech Stack:** Python 3.11+, `pathlib`, `json`, `hashlib`, `zipfile`, pytest.

---

### Task 1: Resume input normalizer

**Files:**
- Create: `experiments/tabm_campaign/resume_input.py`
- Test: `tests/test_tabm_campaign_resume_input.py`

- [ ] **Step 1: Write failing extracted-input tests**

Create a manifest-bound A-resume directory and assert `normalize_resume_input(input_root, working_root)` returns a verified rebuilt ZIP with source `extracted`. Add rejection cases for a modified member and two different valid resumes.

- [ ] **Step 2: Verify the focused tests fail**

Run:

```bash
python -m pytest tests/test_tabm_campaign_resume_input.py -q
```

Expected: collection failure because `experiments.tabm_campaign.resume_input` does not exist.

- [ ] **Step 3: Implement minimal secure normalization**

Define:

```python
@dataclass(frozen=True)
class NormalizedResume:
    path: Path | None
    source: str
    original_path: Path | None
    version: str | None
    manifest_sha256: str | None

def normalize_resume_input(input_root: Path, working_root: Path) -> NormalizedResume:
    ...
```

Validate `artifact_kind`, `review_only`, version, exact member set, safe regular-file paths, and all SHA-256 values. Rebuild extracted input deterministically and call `verify_resume_bundle` for both forms. Collapse identical manifest hashes; reject different logical resumes.

- [ ] **Step 4: Verify focused tests pass**

Run the focused test file and expect all cases to pass.

### Task 2: Generated Kaggle cell integration

**Files:**
- Modify: `tools/render_tabm_campaign_kaggle_cell.py`
- Modify (generated): `experiments/tabm_campaign/KAGGLE_CELL.py`
- Modify: `tests/test_tabm_campaign_kaggle_cell.py`

- [ ] **Step 1: Add a failing generated-runtime test**

Assert the embedded archive contains `resume_input.py`, and assert generated source calls `normalize_resume_input` and prints `source=extracted|zip|none` metadata.

- [ ] **Step 2: Verify the new test fails**

Run the focused generated-cell test and expect the missing embedded module assertion to fail.

- [ ] **Step 3: Integrate the normalizer**

Include `resume_input.py` in the embedded source set. Replace filename-only ZIP discovery with `normalize_resume_input(INPUT_ROOT, WORK_ROOT / "normalized_resume")`; log source, original path, normalized path, and verified version. Pass the normalized ZIP path to `run_one_version`.

- [ ] **Step 4: Regenerate and verify the cell**

Run the renderer twice, confirm byte determinism and size below 950,000 bytes, then run the focused normalizer and cell tests.

### Task 3: Full verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run complete static and unit verification**

Run:

```bash
python -m compileall -q experiments/tabm_campaign tools/render_tabm_campaign_kaggle_cell.py
python -m pytest -q
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 2: Report the exact new cell identity**

Report the generated cell path, byte size, file SHA-256, embedded runtime SHA-256 shown by `CODE_READY`, and the exact expected Stage B startup lines. Do not run full-data training or create a submission package.
