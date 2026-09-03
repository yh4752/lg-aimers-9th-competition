# Final Kaggle Failure Recovery Implementation Plan

> **For Codex:** Execute this plan inline in the current task. Do not run full-data training or create a submission package.

**Goal:** Recover the final `failure_regime_e3` experiment as reproducible source and tests, record the Kaggle out-of-memory failure with verifiable evidence, and synchronize the reviewed repository state to GitHub `main`.

**Architecture:** Preserve the existing E3 campaign implementation in its isolated worktree, verify it against the source embedded in the Kaggle result archive, and keep large runtime artifacts outside Git. Store only compact hashes, completion counts, failure classification, and human-readable experiment history in the repository.

**Tech Stack:** Python 3.11, pytest, CatBoost campaign code, JSON/Markdown evidence, Git.

---

### Task 1: Verify the recovered E3 implementation

**Files:**
- Verify: `experiments/failure_regime_e3/`
- Verify: `tests/test_failure_regime_e3_*.py`
- Verify: `/path/to/Downloads/results.zip`

1. Compare every E3 runtime source member embedded in `results.zip` with the recovered worktree source.
2. Stop if any source member differs or is missing.
3. Run the focused E3, competition-rule, and shared failure-label tests.
4. Compile the package and parse the Kaggle cell without starting training.

### Task 2: Commit the reproducible campaign source

**Files:**
- Add: `experiments/failure_regime_e3/**`
- Add: `tests/test_failure_regime_e3_*.py`
- Add: `docs/superpowers/plans/2026-09-01-failure-regime-e3.md`

1. Exclude caches, ZIP files, model files, predictions, and personal paths.
2. Review the staged diff.
3. Commit only the E3 implementation, tests, and original design plan.

### Task 3: Record the actual Kaggle outcome

**Files:**
- Add: `reports/evidence/failure_regime_e3_20260902.json`
- Modify: `reports/EXPERIMENT_LEDGER.md`
- Modify: `reports/experiment_registry.json`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/EXPERIMENT_JOURNEY.md`

1. Hash the final Kaggle result archive and prior E3 review/handoff bundles.
2. Record the exact failure: out-of-memory after 33,677.4 seconds.
3. Record 53 completed OOF jobs out of 63 planned, with 10 extra-seed jobs pending.
4. Mark the candidate `failed`, not accepted or rejected, and explicitly prohibit submission packaging from this evidence.
5. Explain what was learned and how the next competition workflow should avoid this failure.
6. Validate JSON and commit documentation/evidence separately.

### Task 4: Integrate E3 into main

1. Confirm the E3 worktree is clean.
2. Merge `codex/failure-regime-e3` into local `main` without discarding pre-existing user changes.
3. Re-run focused E3 tests from `main`.

### Task 5: Audit pre-existing main changes separately

**Files:**
- Review only the already-modified gated-residual and TabM recovery files.
- Leave unrelated untracked notebook and lock files untouched unless proven intentional.

1. Run focused tests for gated-residual row independence and TabM recovery/checkpoint handling.
2. Review the diff for scope and generated-cell consistency.
3. Commit only if the changes are coherent and tests pass; otherwise leave them local and report precisely.

### Task 6: Final repository verification and push

1. Confirm no large artifacts, datasets, models, caches, or personal absolute paths are staged.
2. Confirm `main` contains the E3 source, tests, compact evidence, and updated experiment history.
3. Fetch and verify that local `main` is not behind `origin/main`.
4. Push local `main` to `origin/main`.
5. Report pushed commits, remaining local-only files, and the exact documented failure status.
