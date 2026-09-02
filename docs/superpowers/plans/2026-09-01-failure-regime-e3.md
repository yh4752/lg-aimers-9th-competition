# Failure-Regime E3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one restart-safe Kaggle T4x2 campaign that independently reproduces the strongest externally supported structure—E2 anchor plus overlapping failure-subtype experts, global/R/F success experts, and a row-local adaptive gate—without creating a submission package before acceptance.

**Architecture:** Reuse the existing verified E2 input and direct-feature pipeline, but keep E3 in an independent `experiments/failure_regime_e3` package. Phase S chooses one fixed meta recipe on 2022·2023 OOF; Phase C runs 2024 and two extra seeds; Phase F trains the accepted recipe on all official training rows and runs exact row-independence audits. All artifacts bind code, contract, input, train, and Trackman hashes.

**Tech Stack:** Python 3.11, pandas 2.0.3, NumPy 1.26.4, CatBoost 1.2.10, pytest 8.4.1, Kaggle T4x2.

---

### Task 1: Freeze the E3 contract and overlapping labels

**Files:**
- Create: `experiments/failure_regime_e3/__init__.py`
- Create: `experiments/failure_regime_e3/contract.json`
- Create: `experiments/failure_regime_e3/contracts.py`
- Create: `experiments/failure_regime_e3/labels.py`
- Test: `tests/test_failure_regime_e3_contracts.py`
- Test: `tests/test_failure_regime_e3_labels.py`

- [ ] **Step 1: Write failing contract tests**

Assert the fixed folds `((2021, 2022), (2022, 2023), (2023, 2024))`, seeds `(42, 2026, 3407)`, seven roles, depth-10/2400 success capacity, depth-9/1800 subtype capacity, selection years `(2022, 2023)`, confirmation year `2024`, and acceptance thresholds `0.00025/0.00015/0.00010`.

- [ ] **Step 2: Verify contract tests fail**

Run: `python -m pytest tests/test_failure_regime_e3_contracts.py -q`
Expected: import failure because the E3 package does not exist.

- [ ] **Step 3: Implement strict contract loading**

Create frozen dataclasses `RoleSpec`, `E3Contract`, and exact-key JSON validation. Register roles `S_GLOBAL`, `S_FAST`, `S_R`, `S_F`, `MIDDLE`, `WILD`, `REVERSE`. Do not permit unregistered folds, seeds, sources, or gates.

- [ ] **Step 4: Write failing overlapping-label tests**

Construct cumulative pitcher rows where one failure increments both middle and reverse. Require a valid row with `middle=1`, `reverse=1`, `wild=0`; require ordinary ball deltas not to become a target; require `wild=1` only for unexplained failures; require invalid successor links to be masked.

- [ ] **Step 5: Verify label tests fail**

Run: `python -m pytest tests/test_failure_regime_e3_labels.py -q`
Expected: import failure for `recover_failure_targets`.

- [ ] **Step 6: Implement label recovery**

Expose:

```python
@dataclass(frozen=True)
class FailureTargets:
    frame: pd.DataFrame
    valid_mask: np.ndarray
    coverage: float
    overlap_rate: float
    positive_counts: Mapping[str, int]

def recover_failure_targets(rows: pd.DataFrame, *, valid_year: int, tolerance: float) -> FailureTargets:
    ...
```

Use only same-pitcher consecutive official training rows. Recover cumulative deltas for success, middle, ball, and reverse; retain middle/reverse overlap; define `wild=(1-success)*(1-middle)*(1-reverse)`; expose ball only in audit metadata.

- [ ] **Step 7: Run Task 1 tests**

Run: `python -m pytest tests/test_failure_regime_e3_contracts.py tests/test_failure_regime_e3_labels.py -q`
Expected: all pass.

### Task 2: Build row-local meta features and fixed recipe selection

**Files:**
- Create: `experiments/failure_regime_e3/meta.py`
- Create: `experiments/failure_regime_e3/selection.py`
- Test: `tests/test_failure_regime_e3_meta.py`
- Test: `tests/test_failure_regime_e3_selection.py`

- [ ] **Step 1: Write failing meta-feature tests**

Require exact features derived from current row and model streams: E2 logit, four success probabilities and disagreements, three subtype probabilities, game type, counts, inning, runners, leverage, as-of counts/rates, recent gaps, and reliability transforms. Shuffle/reverse/rebatch/singleton transformations must preserve predictions after row-id restoration.

- [ ] **Step 2: Verify meta tests fail**

Run: `python -m pytest tests/test_failure_regime_e3_meta.py -q`
Expected: import failure for `build_meta_frame`.

- [ ] **Step 3: Implement deterministic meta features**

Expose `build_meta_frame(rows, streams, *, include_target)` and reject missing, duplicate, non-finite, or misaligned row identities. No groupby, rolling, rank, shift, batch frequency, or test-derived aggregate is allowed in this module.

- [ ] **Step 4: Write failing selection tests**

Use synthetic 2022·2023·2024 OOF streams. Require recipe fitting on 2022·2023 only, immutable recipe identity, fixed E2 blend grid, no 2024-driven recipe changes, fold/R/F/bootstrap gates, and rejection when any mandatory acceptance gate fails.

- [ ] **Step 5: Verify selection tests fail**

Run: `python -m pytest tests/test_failure_regime_e3_selection.py -q`
Expected: missing selection API.

- [ ] **Step 6: Implement selection and evidence**

Train depth-3 CatBoost gates only through an injected estimator interface in unit tests. Search the contract-fixed E2 safety weights and gate strengths on 2022·2023. Freeze the best recipe, evaluate it without refitting on 2024, compute pitcher-cluster bootstrap and R/F regressions, then return `accepted` only when every gate passes.

- [ ] **Step 7: Run Task 2 tests**

Run: `python -m pytest tests/test_failure_regime_e3_meta.py tests/test_failure_regime_e3_selection.py -q`
Expected: all pass.

### Task 3: Train deep success and subtype experts

**Files:**
- Create: `experiments/failure_regime_e3/training.py`
- Create: `experiments/failure_regime_e3/state.py`
- Test: `tests/test_failure_regime_e3_training.py`
- Test: `tests/test_failure_regime_e3_state.py`

- [ ] **Step 1: Write failing training tests**

Test role masks, season decay, target alignment, GPU parameter identity, checkpoint reuse, model/prediction/metrics output membership, and subtype use of only valid recovered rows. Assert S_R and S_F fall back to S_GLOBAL outside their active game type at selection time.

- [ ] **Step 2: Verify training tests fail**

Run: `python -m pytest tests/test_failure_regime_e3_training.py -q`
Expected: missing `run_fold_job` and role helpers.

- [ ] **Step 3: Implement fold training**

Reuse `fit_direct_features` and `feature_profile`. Train success roles on `control_success`; train subtype roles on recovered middle/wild/reverse labels. Bind job, features, parameters, train/history hashes, fold, seed, and role to `job_identity.json`. Reuse only byte-identical completed jobs.

- [ ] **Step 4: Write and implement restart-state tests**

State transitions are `screening → confirmation → extra_seeds → decision → full_fit → audit → completed`. Completed jobs are immutable; failed jobs remain retryable; atomic JSON cannot contain NaN or duplicate job IDs.

- [ ] **Step 5: Run Task 3 tests**

Run: `python -m pytest tests/test_failure_regime_e3_training.py tests/test_failure_regime_e3_state.py -q`
Expected: all pass.

### Task 4: Orchestrate one restart-safe campaign and artifact chain

**Files:**
- Create: `experiments/failure_regime_e3/artifacts.py`
- Create: `experiments/failure_regime_e3/runtime.py`
- Create: `experiments/failure_regime_e3/runner.py`
- Test: `tests/test_failure_regime_e3_artifacts.py`
- Test: `tests/test_failure_regime_e3_runner.py`

- [ ] **Step 1: Write failing artifact tests**

Require deterministic manifests, safe ZIP paths, exact member sets, per-member hashes, binding verification, directory/ZIP input parity, and rejection of stale resumes.

- [ ] **Step 2: Verify artifact tests fail**

Run: `python -m pytest tests/test_failure_regime_e3_artifacts.py -q`
Expected: missing artifact API.

- [ ] **Step 3: Implement review and handoff bundles**

Create only `failure_regime_e3_review.zip` and `failure_regime_e3_handoff.zip`. Do not create a delivery or submission artifact. Handoff includes completed job evidence and full-fit payloads when available.

- [ ] **Step 4: Write failing runner tests**

Use a fake runtime to prove phase ordering, two-GPU scheduling, deadline pause before new work, exact resume reuse, early rejection isolation, acceptance-gated full fit, and audit failure blocking completion.

- [ ] **Step 5: Implement the runner**

Run 14 screening jobs for 2022·2023 seed 3407, freeze a recipe, then 2024 confirmation plus seeds 42/2026 for all folds. If accepted, train 21 full-fit expert models plus three small gate seeds. Pause safely when the deadline guard is reached and always emit one current handoff.

- [ ] **Step 6: Run Task 4 tests**

Run: `python -m pytest tests/test_failure_regime_e3_artifacts.py tests/test_failure_regime_e3_runner.py -q`
Expected: all pass.

### Task 5: Add Kaggle discovery, static policy gates, and one-cell renderer

**Files:**
- Create: `experiments/failure_regime_e3/kaggle.py`
- Create: `experiments/failure_regime_e3/runtime_inventory.py`
- Create: `experiments/failure_regime_e3/KAGGLE_CELL.py`
- Test: `tests/test_failure_regime_e3_kaggle.py`
- Test: `tests/test_failure_regime_e3_cell.py`

- [ ] **Step 1: Write failing Kaggle tests**

Require exactly one official dataset and one verified `direct_expert_input` source, zero or one E3 handoff, folder-or-ZIP discovery, deterministic selection when Kaggle adds metadata, T4x2 verification, and explicit duplicate-artifact diagnostics.

- [ ] **Step 2: Verify Kaggle tests fail**

Run: `python -m pytest tests/test_failure_regime_e3_kaggle.py -q`
Expected: missing discovery API.

- [ ] **Step 3: Implement Kaggle execution**

Run the competition source gate before training. Emit progress markers for phase, job, fold, seed, role, elapsed time, decision, pause, review, and handoff. Use a single wall deadline and never auto-submit or auto-upload.

- [ ] **Step 4: Write failing renderer tests**

Require deterministic embedded runtime, AST parseability, source inventory completeness, no network operations after dependency installation, and exact checked-in-cell equality.

- [ ] **Step 5: Render the checked-in cell**

Embed the required runtime as deterministic gzip+base64 and call `run_kaggle_campaign(Path('/kaggle/input'), Path('/kaggle/working/failure_regime_e3'))`.

- [ ] **Step 6: Run Task 5 tests**

Run: `python -m pytest tests/test_failure_regime_e3_kaggle.py tests/test_failure_regime_e3_cell.py -q`
Expected: all pass.

### Task 6: Verify the complete handoff without a heavy run

**Files:**
- Modify: `docs/ROADMAP.md`
- Modify: `reports/EXPERIMENT_LEDGER.md`
- Test: `tests/test_failure_regime_e3_repository.py`

- [ ] **Step 1: Add repository contract tests**

Require the E3 contract, policy version, cell, candidate state `code_ready`, user-run ownership, and absence of any E3 submission package entry point.

- [ ] **Step 2: Run all E3 tests and relevant regressions**

Run: `python -m pytest tests/test_failure_regime_e3_*.py tests/test_tree_expert_failure_labels.py tests/test_tree_expert_e2_inference.py tests/test_rules_code_gate.py tests/test_rules_entrypoints.py tests/test_rules_policy.py tests/test_rules_repository_enforcement.py -q`
Expected: all pass.

- [ ] **Step 3: Run static and syntax checks**

Compile all E3 Python sources, parse the generated cell, scan source through the current competition policy, and verify deterministic runtime hashes.

- [ ] **Step 4: Record code-ready status**

Document hypothesis, external evidence boundary, fixed validation, aggressive capacity, rules gate, expected T4x2 duration, artifact names, and exact success/error markers. Do not record a performance result before the user executes the full campaign.

- [ ] **Step 5: Review the final diff**

Confirm only the E3 package, E3 tests, plan, roadmap, and ledger changed. Confirm no data, model, ZIP, token, personal path, notebook, packaging, push, or submission code was added.
