# Tree Expert E2 Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a hash-gated, row-independent DACON `submit.zip` for the accepted three-seed E2 CatBoost candidate.

**Architecture:** Import the exact accepted E2 handoff into a frozen candidate directory, render a repository-independent evaluator runtime, convert reviewed E2 evidence into the repository's standard acceptance contract, and pass that contract to the sole writer in `submission/package.py`. Every stage is fail-closed and tested before the final local build.

**Tech Stack:** Python 3.11.15, pandas 2.0.3, NumPy 1.26.4, CatBoost 1.2.10, pytest, deterministic ZIP archives.

---

### Task 1: Exact E2 delivery import

**Files:**
- Create: `submission/tree_expert_e2_candidate.py`
- Test: `tests/test_tree_expert_e2_submission_candidate.py`

- [ ] **Step 1: Write failing tests for accepted and tampered handoffs**

Create fixtures with the exact nested manifest shape and assert that
`import_tree_expert_e2_candidate()` accepts the registered SHA only, extracts
the six frozen/model files, writes a candidate manifest, and rejects changed
bytes, unsafe members, non-accepted status, missing delivery, wrong candidate,
wrong predictor, wrong seeds, and mismatched member hashes.

- [ ] **Step 2: Run the candidate tests and verify the missing-module failure**

Run: `python -m pytest -q tests/test_tree_expert_e2_submission_candidate.py`

Expected: FAIL because `submission.tree_expert_e2_candidate` does not exist.

- [ ] **Step 3: Implement the minimal verified importer**

Define `TREE_E2_CANDIDATE_ID`, `TREE_E2_HANDOFF_SHA256`,
`ImportedTreeE2Candidate`, safe JSON/ZIP readers, manifest verification, and an
exclusive deterministic candidate export. Never accept additional members or
write outside the requested candidate directory.

- [ ] **Step 4: Run the candidate tests**

Run: `python -m pytest -q tests/test_tree_expert_e2_submission_candidate.py`

Expected: PASS.

- [ ] **Step 5: Commit the importer**

```bash
git add submission/tree_expert_e2_candidate.py tests/test_tree_expert_e2_submission_candidate.py
git commit -m "feat(submission): import accepted tree expert E2 delivery"
```

### Task 2: Reviewed standalone runtime and adapter registration

**Files:**
- Create: `submission/tree_expert_e2_script.py`
- Modify: `submission/tree_expert_e2_candidate.py`
- Modify: `submission/adapters.py`
- Modify: `submission/runtime.py`
- Test: `tests/test_tree_expert_e2_submission_script.py`

- [ ] **Step 1: Write failing feature, prediction, and evaluator tests**

Build a small frozen S1 fixture and fake residual models. Compare the standalone
runtime with `experiments.tree_expert.features.transform_tree_features` and
`E2InferenceRuntime`; assert reverse, shuffle, rebatch, and singleton equality,
sample-order restoration, exact output columns, target rejection, and missing
input/model errors. Assert the registered adapter and `render_script()` select
only the E2 candidate.

- [ ] **Step 2: Run runtime tests and verify they fail for missing behavior**

Run: `python -m pytest -q tests/test_tree_expert_e2_submission_script.py`

Expected: FAIL because the E2 standalone runtime and adapter are absent.

- [ ] **Step 3: Implement the standalone evaluator**

Port only the reviewed S1 restore/transform, row-local feature construction,
anchor calculation, CatBoost residual ensemble, input validation, and atomic
CSV output. Add `render_bound_script()` that binds the candidate manifest
hashes and register its factory in `submission/adapters.py` and renderer in
`submission/runtime.py`.

- [ ] **Step 4: Run runtime and existing submission tests**

Run: `python -m pytest -q tests/test_tree_expert_e2_submission_script.py tests/test_submission_runtime.py tests/test_submission_package.py`

Expected: PASS.

- [ ] **Step 5: Commit the runtime**

```bash
git add submission/tree_expert_e2_script.py submission/tree_expert_e2_candidate.py submission/adapters.py submission/runtime.py tests/test_tree_expert_e2_submission_script.py
git commit -m "feat(submission): add standalone E2 CatBoost runtime"
```

### Task 3: Existing-evidence acceptance adapter

**Files:**
- Create: `submission/tree_expert_e2_existing_evidence.py`
- Create: `reports/rules/2026-08-27-final-policy-review.json`
- Test: `tests/test_tree_expert_e2_submission_evidence.py`

- [ ] **Step 1: Write failing evidence-gate tests**

Assert that the accepted decision, full-fit manifest, inference audit, model
hashes, source manifest, 245,789-row capacity, zero independence deltas,
64.939-second runtime, memory limit, same-day rules review, exact sample parity,
and source gate are all required. Independently mutate each gate and assert the
acceptance directory is not published.

- [ ] **Step 2: Run evidence tests and verify the missing-module failure**

Run: `python -m pytest -q tests/test_tree_expert_e2_submission_evidence.py`

Expected: FAIL because the E2 acceptance adapter does not exist.

- [ ] **Step 3: Implement standard acceptance generation**

Create a standard `AuditIdentity`, run the current five-row phased audit, write
the standard seven-gate acceptance and runtime benchmark, and retain the
reviewed full-scale E2 audit as provenance. Use the same-date policy review with
`verdict=unchanged` and the current policy digest.

- [ ] **Step 4: Run evidence tests**

Run: `python -m pytest -q tests/test_tree_expert_e2_submission_evidence.py`

Expected: PASS.

- [ ] **Step 5: Commit the acceptance adapter**

```bash
git add submission/tree_expert_e2_existing_evidence.py reports/rules/2026-08-27-final-policy-review.json tests/test_tree_expert_e2_submission_evidence.py
git commit -m "feat(submission): bind E2 evidence to current rules"
```

### Task 4: Sole-packager build command

**Files:**
- Create: `tools/build_tree_expert_e2_submission.py`
- Modify: `submission/audit.py`
- Test: `tests/test_tree_expert_e2_submission_build_tool.py`

- [ ] **Step 1: Write failing end-to-end builder tests**

Assert that the command requires Python 3.11.15 and CatBoost 1.2.10, validates
the official five-row sample, imports the exact handoff, checks trusted/runtime
prediction parity, projects sizes, calls `build_submission_package()`, and
verifies exact final members and hashes. Assert no `submit.zip` is created for
wrong environment, existing output, stale policy review, or any artifact gate
failure.

- [ ] **Step 2: Run builder tests and verify failure**

Run: `python -m pytest -q tests/test_tree_expert_e2_submission_build_tool.py`

Expected: FAIL because the builder is absent.

- [ ] **Step 3: Implement the build command and E2 manifest audit**

Add E2 candidate-manifest validation to the existing audit path without adding
a second ZIP writer. The command creates the candidate and evidence in a new
output directory, then passes a `PackageRequest` to
`build_submission_package()` and prints `TREE_E2_SUBMISSION_READY` only after
post-build verification.

- [ ] **Step 4: Run builder and regression tests**

Run: `python -m pytest -q tests/test_tree_expert_e2_submission_build_tool.py tests/test_submission_package.py tests/test_submission_runtime.py tests/test_tabm_submission_build_tool.py`

Expected: PASS.

- [ ] **Step 5: Commit the command**

```bash
git add tools/build_tree_expert_e2_submission.py submission/audit.py tests/test_tree_expert_e2_submission_build_tool.py
git commit -m "feat(submission): build accepted E2 package through sole writer"
```

### Task 5: Final verification, build, and record

**Files:**
- Modify: `reports/EXPERIMENT_LEDGER.md`
- Create outside Git: `artifacts/tree_expert_e2_submission_*/submit.zip`
- Create outside Git: `artifacts/tree_expert_e2_submission_*/submission_receipt.json`

- [ ] **Step 1: Run the complete focused regression suite**

Run: `python -m pytest -q tests/test_tree_expert_e2_*.py tests/test_tree_expert_e2_submission_*.py tests/test_submission_*.py tests/test_tabm_submission_build_tool.py`

Expected: all tests PASS.

- [ ] **Step 2: Run static and syntax checks**

Run: `python -m compileall -q submission tools/build_tree_expert_e2_submission.py`

Expected: exit code 0.

- [ ] **Step 3: Build the exact local package**

Use the official Python 3.11.15 environment, exact accepted handoff, official
five-row local data directory, and a new timestamped artifact directory.

Expected terminal line: `TREE_E2_SUBMISSION_READY ...`.

- [ ] **Step 4: Inspect the produced ZIP and receipt**

Verify ZIP integrity, exact top-level layout, candidate/model/state hashes,
archive SHA against the receipt, and absence of data, predictions, training
state, unsafe paths, duplicate members, or external-access code.

- [ ] **Step 5: Record the package-ready state and commit**

Append the accepted E2 result and package-ready identity to the experiment
ledger without claiming a Public score.

```bash
git add reports/EXPERIMENT_LEDGER.md
git commit -m "docs: record E2 CatBoost package readiness"
```

## Self-review

- Every design requirement maps to a task above.
- No alternative ZIP writer is introduced.
- Heavy full-data inference is not repeated; only fixtures and the official
  five-row local sample run locally.
- The rejected TabM blend is excluded.
- The package cannot be produced before status, acceptance, audit, policy,
  runtime, and current member hashes all pass.
