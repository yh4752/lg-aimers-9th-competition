# Tree Heterogeneous Residual S3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable Kaggle T4x2 campaign that evaluates fixed XGBoost and LightGBM probability models as small residual corrections to the accepted E2 CatBoost OOF baseline.

**Architecture:** Reuse E2 handoff extraction, temporal folds, row-local tree features, deterministic artifacts, and the T3 handoff pattern. XGBoost and LightGBM learn the binary target directly; OOF-only correction weights blend each prediction toward the frozen E2 baseline. This phase produces review/resume/handoff artifacts only, never a submission package.

**Tech Stack:** Python 3.11/3.12, pandas, NumPy, XGBoost 3.0.2, LightGBM 4.6.0, pytest, deterministic ZIP artifacts, Kaggle T4x2

---

### Task 1: Seal the S3 contract and jobs

**Files:**
- Create: `experiments/tree_expert/hetero_contract.json`
- Create: `experiments/tree_expert/hetero_contracts.py`
- Test: `tests/test_tree_expert_hetero_contracts.py`

- [ ] Write tests asserting families `xgboost/lightgbm`, folds `2021→2022` through `2023→2024`, weights `0.05/0.10/0.15`, seed 3407, confirmation seeds 42/2026, gates, versions, and six unique structure jobs.
- [ ] Run `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hetero_contracts.py -q` and confirm missing-module failure.
- [ ] Implement strict JSON parsing with exact keys and immutable `HeteroContract`, `HeteroGates`, and `HeteroJob` dataclasses.
- [ ] Implement deterministic `structure_jobs()` and `confirmation_jobs(family)` registries.
- [ ] Run the contract tests and commit `feat: seal hetero residual contract`.

Required job identity:

```python
HeteroJob(
    job_id=f"hetero__{family}__tr{train_end}__va{valid_year}__s{seed}",
    family=family,
    train_end_year=train_end,
    valid_year=valid_year,
    seed=seed,
)
```

### Task 2: Reuse and verify the compact T3/E2 OOF input

**Files:**
- Reuse: `experiments/tree_expert/t3_inputs.py`
- Reuse: `tools/prepare_tree_expert_t3_input.py`
- Test: `tests/test_tree_expert_t3_inputs.py`

- [ ] Run the existing round-trip, changed-fold, row-order, target, duplicate-ID, unsafe-member, and wrong-E2-SHA tests.
- [ ] Confirm `hetero_contract.json` and `t3_contract.json` bind the same official train, history, and E2 handoff SHA-256 values.
- [ ] Import `verify_and_extract_t3_input()` from the sealed runtime instead of adding a second archive implementation.
- [ ] Record the reuse decision in this plan; no production file or commit is needed for this task.

The payload set is exactly:

```python
{
    "e2/fold_2021_2022.csv",
    "e2/fold_2022_2023.csv",
    "e2/fold_2023_2024.csv",
    "manifest.json",
}
```

Each CSV keeps `row_id,target,probability,game_type,game_month,pitcher_id_known,batter_id_known` unchanged.

### Task 3: Encode row-local features without validation fitting

**Files:**
- Create: `experiments/tree_expert/hetero_features.py`
- Test: `tests/test_tree_expert_hetero_features.py`

- [ ] Write failing neighbor-invariance, shuffle-invariance, OOV, schema-drift, infinity, and validation-target tests.
- [ ] Run tests and confirm missing encoder failure.
- [ ] Wrap existing `fit_tree_features()` and `transform_tree_features()` with train-only ordinal maps for categorical columns.
- [ ] Map unseen/missing categories to fixed `-1`; retain numeric NaN for native tree handling; reject infinity.
- [ ] Run tests and commit `feat: encode hetero tree features`.

Required output API:

```python
@dataclass(frozen=True)
class HeteroFeatureBatch:
    matrix: np.ndarray
    row_id: np.ndarray
    target: np.ndarray | None
```

### Task 4: Train direct-probability family jobs

**Files:**
- Create: `experiments/tree_expert/hetero_training.py`
- Test: `tests/test_tree_expert_hetero_training.py`

- [ ] Write failing XGBoost and LightGBM adapter tests using fake estimators; assert the observed training target is `control_success`, not a residual.
- [ ] Write failing tests for baseline row/target mismatch, invalid probabilities, atomic output, and family-local failure.
- [ ] Run tests and confirm missing training API failure.
- [ ] Implement XGBoost binary logistic GPU adapter and LightGBM binary CPU adapter with early stopping and pinned parameters.
- [ ] Atomically write job identity, metrics, model, predictions, and worker log.
- [ ] Run tests and commit `feat: train hetero probability models`.

Predictions must contain:

```python
["row_id", "target", "baseline_probability", "model_probability"]
```

### Task 5: Select correction weights and gate families

**Files:**
- Create: `experiments/tree_expert/hetero_decisions.py`
- Test: `tests/test_tree_expert_hetero_decisions.py`

- [ ] Write failing tests for fixed-grid selection on 2022/2023 only, one-time 2024 confirmation, all gate failures, seed consistency, and simpler-winner tie handling.
- [ ] Run tests and confirm missing decision API failure.
- [ ] Implement `p = clip(p0 + a * (ph - p0))` for `a ∈ {0.05,0.10,0.15}`.
- [ ] Implement family decisions and equal-residual blend only when both families pass confirmation.
- [ ] Run tests and commit `feat: gate hetero residual candidates`.

Fixed family gate:

```python
passed = (
    weighted_gain >= 0.00005
    and min(fold_gains.values()) >= -0.00003
    and fold_gains[(2023, 2024)] >= 0.0
    and maximum_segment_regression <= 0.00030
    and bootstrap_lower_95 > 0.0
    and residual_correlation < 0.995
)
```

The equal-residual blend must improve over the best single family by at least `0.00002`; otherwise the simpler single family wins.

### Task 6: Persist resumable state and review-only artifacts

**Files:**
- Create: `experiments/tree_expert/hetero_state.py`
- Create: `experiments/tree_expert/hetero_artifacts.py`
- Test: `tests/test_tree_expert_hetero_state.py`
- Test: `tests/test_tree_expert_hetero_artifacts.py`

- [ ] Write failing state transition, deterministic ZIP, binding mismatch, unsafe extraction, tamper, and review-only handoff tests.
- [ ] Run tests and confirm missing state/artifact APIs.
- [ ] Implement immutable state with phases `structure`, `confirmation`, and `completed`.
- [ ] Implement verified resume/review/handoff ZIPs bound to contract, code, input, official data, and E2 SHA-256.
- [ ] Run tests and commit `feat: persist hetero residual campaign`.

The handoff manifest always contains:

```python
{"delivery": False, "review_only": True, "submission_package": False}
```

### Task 7: Orchestrate structure and confirmation

**Files:**
- Create: `experiments/tree_expert/hetero_runner.py`
- Test: `tests/test_tree_expert_hetero_runner.py`

- [ ] Write failing tests for two-worker scheduling, family-local failure, deadline stop, verified resume reuse, rejected-family pruning, and incomplete-versus-rejected status.
- [ ] Run tests and confirm missing runner failure.
- [ ] Implement at most two concurrent jobs and the 600-second new-job guard.
- [ ] Publish a verified resume at stage changes and every 600 seconds without browser downloads.
- [ ] Schedule confirmation seeds only for structure-passed families.
- [ ] Run tests and commit `feat: run resumable hetero residual campaign`.

Deadline behavior:

```python
if time.monotonic() + contract.new_job_guard_seconds >= wall_deadline:
    return publish_incomplete_handoff(state)
```

### Task 8: Generate the Kaggle one-cell handoff

**Files:**
- Create: `experiments/tree_expert/hetero_kaggle.py`
- Create: `experiments/tree_expert/KAGGLE_HETERO_CELL.py`
- Create: `docs/TREE_HETERO_KAGGLE.md`
- Test: `tests/test_tree_expert_hetero_kaggle.py`
- Test: `tests/test_tree_expert_hetero_cell.py`

- [ ] Write failing tests for deterministic runtime packing, one-megabyte limit, official/input/resume discovery, expanded handoff handling, and one stable output filename.
- [ ] Run tests and confirm missing Kaggle APIs.
- [ ] Implement dependency checks for XGBoost 3.0.2 and LightGBM 4.6.0, require T4x2, and embed the sealed runtime.
- [ ] Emit stable progress/error markers and copy exactly one `/kaggle/working/tree_hetero_handoff.zip`.
- [ ] Run all `tests/test_tree_expert_hetero_*.py`, adjacent E2/feature regressions, `py_compile`, renderer equality, `git diff --check`, and the 1MB check.
- [ ] Commit `feat: hand off hetero residual Kaggle campaign`.

Required final markers:

```text
TREE_HETERO_CODE_READY
TREE_HETERO_DEPENDENCIES_READY
TREE_HETERO_GPU_READY
TREE_HETERO_INPUTS_VERIFIED
TREE_HETERO_JOB_START
TREE_HETERO_JOB_END
TREE_HETERO_DECISION
TREE_HETERO_HANDOFF_READY
TREE_HETERO_SUCCESS
TREE_HETERO_ERROR stage=<stage> type=<type> message=<message>
```

No full-data training is run by Codex. No accepted S3 result, exact artifact hashes, and row-independence audit means no full-fit delivery and no submission package.
