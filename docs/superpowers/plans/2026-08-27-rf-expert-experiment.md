# R/F Expert Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restart-safe Kaggle T4×2 experiment that tests R/F-specific CatBoost experts against the frozen E2 baseline and emits a deployable delivery only after all OOF and independence gates pass.

**Architecture:** Add an isolated `rf_*` campaign beside E2 and rejected T3. Reuse the existing row-local feature builder and E2 baseline artifacts, but give the R/F experiment its own immutable contract, input bundle, jobs, decision logic, artifacts, runner, inference audit, and one-cell Kaggle launcher. Selection uses only the first two temporal folds; the 2024 fold and three-seed evidence are confirmation gates.

**Tech Stack:** Python 3.11/3.12, pandas, NumPy, CatBoost 1.2.10, pytest, ZIP/JSON/SHA-256 artifact contracts, Kaggle Tesla T4×2.

---

## File map

Create these focused production files:

- `experiments/tree_expert/rf_contract.json`: immutable search grid, model settings, hashes, gates, and runtime budget.
- `experiments/tree_expert/rf_contracts.py`: strict contract parser and deterministic job enumeration.
- `experiments/tree_expert/rf_inputs.py`: prepare and verify the compact E2-based input archive.
- `experiments/tree_expert/rf_decisions.py`: pure R/F routing, blend search, holdout metrics, and acceptance logic.
- `experiments/tree_expert/rf_training.py`: one segment/fold/seed CatBoost job with atomic outputs.
- `experiments/tree_expert/rf_diagnostics.py`: fold, segment, calibration, correlation, and invariance tables.
- `experiments/tree_expert/rf_full_fit.py`: accepted-only full-fit token and final expert training.
- `experiments/tree_expert/rf_inference.py`: current-row R/F routing and fixed-weight inference.
- `experiments/tree_expert/rf_artifacts.py`: review, resume, and accepted-only delivery bundles.
- `experiments/tree_expert/rf_runner.py`: staged scheduler, two-GPU dispatch, deadline, and resume state machine.
- `experiments/tree_expert/rf_kaggle.py`: Kaggle input discovery and supervised snapshot/download flow.
- `experiments/tree_expert/KAGGLE_RF_CELL.py`: generated one-cell launcher below Kaggle's 1 MB source limit.
- `tools/prepare_tree_expert_rf_input.py`: local compact-input command.
- `tools/build_tree_expert_rf_kaggle_cell.py`: deterministic embedded-runtime cell builder.

Create matching tests named `tests/test_tree_expert_rf_<unit>.py`. Do not modify the existing E2 submission builder. The R/F campaign must not gain a DACON submission-packaging entry point.

### Task 1: Immutable contract and deterministic job grid

**Files:**
- Create: `experiments/tree_expert/rf_contract.json`
- Create: `experiments/tree_expert/rf_contracts.py`
- Test: `tests/test_tree_expert_rf_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from dataclasses import replace

import pytest

from experiments.tree_expert.rf_contracts import (
    RFContractError,
    confirmation_jobs,
    load_rf_contract,
    structure_jobs,
)


def test_rf_contract_has_fixed_search_and_budget():
    contract = load_rf_contract()
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.structure_seed == 3407
    assert contract.confirmation_seeds == (42, 2026)
    assert contract.alpha_values == (0.25, 0.50, 0.75, 1.00)
    assert contract.wall_seconds == 19_800
    assert contract.full_fit_guard_seconds == 1_800


def test_structure_grid_has_three_heads_across_three_folds():
    jobs = structure_jobs(load_rf_contract())
    assert len(jobs) == 9
    assert {job.head for job in jobs} == {"f_small", "f_wide", "r_expert"}
    assert len({job.job_id for job in jobs}) == 9


def test_confirmation_grid_only_uses_selected_heads():
    jobs = confirmation_jobs(load_rf_contract(), f_head="f_small", include_r=True)
    assert len(jobs) == 12
    assert {job.seed for job in jobs} == {42, 2026}
    assert {job.head for job in jobs} == {"f_small", "r_expert"}


def test_confirmation_rejects_unknown_head():
    with pytest.raises(RFContractError, match="F head differs"):
        confirmation_jobs(load_rf_contract(), f_head="other", include_r=True)
```

- [ ] **Step 2: Run the focused test and confirm the import failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_contracts.py -q
```

Expected: collection fails because `experiments.tree_expert.rf_contracts` does not exist.

- [ ] **Step 3: Add the exact contract**

Write `rf_contract.json` with these fixed values:

```json
{
  "schema_version": 1,
  "campaign_id": "tree_expert_rf_v1",
  "inputs": {
    "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
    "e2_handoff_sha256": "4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f"
  },
  "folds": [[2021, 2022], [2022, 2023], [2023, 2024]],
  "structure_seed": 3407,
  "confirmation_seeds": [42, 2026],
  "alpha_values": [0.25, 0.50, 0.75, 1.00],
  "experts": {
    "f_small": {"segment": "F", "iterations": 800, "depth": 6, "l2_leaf_reg": 10.0, "random_strength": 1.0, "bagging_temperature": 1.0},
    "f_wide": {"segment": "F", "iterations": 800, "depth": 8, "l2_leaf_reg": 8.0, "random_strength": 0.75, "bagging_temperature": 0.75},
    "r_expert": {"segment": "R", "iterations": 800, "depth": 8, "l2_leaf_reg": 5.0, "random_strength": 0.5, "bagging_temperature": 0.5}
  },
  "catboost_common": {
    "learning_rate": 0.04, "bootstrap_type": "Bayesian", "border_count": 128,
    "max_ctr_complexity": 2, "one_hot_max_size": 16, "od_type": "Iter",
    "od_wait": 60, "task_type": "GPU", "allow_writing_files": true
  },
  "gates": {
    "weighted_gain": 0.00010, "minimum_improved_folds": 2,
    "maximum_segment_regression": 0.00010, "recent_f_gain": 0.0,
    "minimum_segment_rows": 5000, "probability_tolerance": 0.000001
  },
  "runtime": {
    "wall_seconds": 19800, "new_job_guard_seconds": 600,
    "full_fit_guard_seconds": 1800, "snapshot_interval_seconds": 600,
    "progress_interval_seconds": 30, "inference_max_seconds": 480,
    "rss_max_bytes": 23622320128
  }
}
```

- [ ] **Step 4: Implement strict dataclasses and enumeration**

Expose these frozen types and functions from `rf_contracts.py`:

```python
@dataclass(frozen=True)
class RFJob:
    job_id: str
    head: str
    segment: str
    train_end_year: int
    valid_year: int
    seed: int


def structure_jobs(contract: RFContract) -> tuple[RFJob, ...]:
    return tuple(
        RFJob(
            job_id=f"rf__{head}__tr{train_end}__va{valid_year}__s{contract.structure_seed}",
            head=head,
            segment=str(contract.experts[head]["segment"]),
            train_end_year=train_end,
            valid_year=valid_year,
            seed=contract.structure_seed,
        )
        for train_end, valid_year in contract.folds
        for head in ("f_small", "f_wide", "r_expert")
    )
```

The parser must require exact top-level and nested key sets, finite numeric values,
the three fixed folds, seeds, alpha grid, `F/F/R` segment mapping, 19,800-second
wall budget, and 1e-6 tolerance. `contract_sha256()` must hash the JSON bytes.

- [ ] **Step 5: Run, verify, and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_contracts.py -q
git add experiments/tree_expert/rf_contract.json experiments/tree_expert/rf_contracts.py tests/test_tree_expert_rf_contracts.py
git commit -m "feat(tree-expert): define RF campaign contract"
```

Expected: all focused tests pass.

### Task 2: Compact input preparation and verification

**Files:**
- Create: `experiments/tree_expert/rf_inputs.py`
- Create: `tools/prepare_tree_expert_rf_input.py`
- Test: `tests/test_tree_expert_rf_inputs.py`

- [ ] **Step 1: Write failing tests for one immutable input**

```python
def test_prepare_and_verify_rf_input(tmp_path, e2_handoff_fixture):
    archive = prepare_rf_input(e2_handoff_fixture, tmp_path / "rf_input.zip")
    verified = verify_rf_input(archive, tmp_path / "verified")
    assert verified.e2_candidate_id == "c1_anchor_residual"
    assert set(verified.fold_predictions) == {(2021, 2022), (2022, 2023), (2023, 2024)}
    assert verified.full_fit_root.is_dir()


def test_rf_input_rejects_modified_member(tmp_path, e2_handoff_fixture):
    archive = prepare_rf_input(e2_handoff_fixture, tmp_path / "rf_input.zip")
    tampered = replace_zip_member(archive, "e2/decision.json", b"{}")
    with pytest.raises(RFInputError, match="member hash differs"):
        verify_rf_input(tampered, tmp_path / "verified")


def test_rf_input_rejects_wrong_e2_candidate(tmp_path, e2_handoff_fixture):
    archive = prepare_rf_input(e2_handoff_fixture, tmp_path / "rf_input.zip")
    altered = replace_manifest_value(archive, "e2_candidate_id", "c2_trackman_residual")
    with pytest.raises(RFInputError, match="E2 candidate differs"):
        verify_rf_input(altered, tmp_path / "verified")
```

- [ ] **Step 2: Confirm the tests fail before implementation**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_inputs.py -q
```

Expected: import failure for `rf_inputs`.

- [ ] **Step 3: Implement a minimal typed input boundary**

```python
@dataclass(frozen=True)
class VerifiedRFInput:
    root: Path
    manifest_sha256: str
    e2_candidate_id: str
    fold_predictions: Mapping[tuple[int, int], Path]
    full_fit_root: Path


def prepare_rf_input(e2_handoff: Path, output: Path) -> Path:
    verified = verify_e2_handoff_for_rf(Path(e2_handoff))
    members = collect_required_e2_members(verified)
    write_deterministic_zip(
        Path(output),
        members,
        kind="tree_expert_rf_input_v1",
        bindings={
            "rf_contract_sha256": contract_sha256(),
            "e2_handoff_sha256": file_sha256(Path(e2_handoff)),
            "e2_candidate_id": "c1_anchor_residual",
        },
    )
    return Path(output)
```

Copy only the three E2 OOF prediction sets, their targets/row identities, the accepted
E2 decision, frozen state, three full-fit models, and E2 inference metadata. Reject
symlinks, absolute paths, `..`, duplicate ZIP names, undeclared members, encrypted
members, changed hashes, non-finite probabilities, duplicate row IDs, and an E2
candidate other than `c1_anchor_residual`.

- [ ] **Step 4: Implement the local command with repository import bootstrapping**

At the top of `tools/prepare_tree_expert_rf_input.py`, use:

```python
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tree_expert.rf_inputs import file_sha256, prepare_rf_input
```

The command must require `--e2-handoff` and `--output`, print exactly one terminal
success line beginning `TREE_RF_INPUT_SUCCESS`, and print `TREE_RF_INPUT_ERROR`
with stage, exception type, and message on failure.

- [ ] **Step 5: Run, verify, and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_inputs.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile tools/prepare_tree_expert_rf_input.py
git add experiments/tree_expert/rf_inputs.py tools/prepare_tree_expert_rf_input.py tests/test_tree_expert_rf_inputs.py
git commit -m "feat(tree-expert): prepare verified RF inputs"
```

### Task 3: Pure routing, blend search, and acceptance decisions

**Files:**
- Create: `experiments/tree_expert/rf_decisions.py`
- Test: `tests/test_tree_expert_rf_decisions.py`

- [ ] **Step 1: Write failing routing and leakage-boundary tests**

```python
def test_route_probability_uses_only_current_row_game_type():
    base = np.array([0.2, 0.4, 0.6])
    r = np.array([0.3, 0.5, 0.7])
    f = np.array([0.1, 0.2, 0.8])
    game_type = np.array(["R", "F", "R"])
    actual = route_probability(base, r, f, game_type, alpha_r=0.5, alpha_f=0.25)
    np.testing.assert_allclose(actual, [0.25, 0.35, 0.65])


def test_route_probability_is_permutation_invariant():
    order = np.array([2, 0, 1])
    original = route_probability(BASE, R_PRED, F_PRED, GAME_TYPE, 0.5, 0.75)
    shuffled = route_probability(BASE[order], R_PRED[order], F_PRED[order], GAME_TYPE[order], 0.5, 0.75)
    np.testing.assert_allclose(shuffled, original[order], atol=1e-12)


def test_structure_selection_never_reads_confirmation_fold():
    first = structure_evidence(confirmation_f_probability=0.0)
    second = structure_evidence(confirmation_f_probability=1.0)
    assert select_rf_structure(first, CONTRACT).selection_key == select_rf_structure(second, CONTRACT).selection_key
```

- [ ] **Step 2: Write failing gate and tie-break tests**

```python
def test_tie_break_prefers_small_f_then_lower_alpha_sum_then_f_only():
    decision = select_rf_structure(exact_tie_evidence(), CONTRACT)
    assert decision.f_head == "f_small"
    assert decision.alpha_r == 0.0
    assert decision.alpha_f == 0.25


def test_acceptance_requires_recent_f_improvement():
    evidence = acceptance_evidence(weighted_gain=0.0002, improved_folds=3, recent_f_gain=0.0)
    assert accept_rf(evidence, CONTRACT).status == "rejected"


def test_acceptance_requires_two_improved_folds_and_segment_safety():
    assert accept_rf(acceptance_evidence(improved_folds=1), CONTRACT).status == "rejected"
    assert accept_rf(acceptance_evidence(max_segment_regression=0.00011), CONTRACT).status == "rejected"
```

- [ ] **Step 3: Implement the row-local blend primitive**

```python
def route_probability(
    baseline: np.ndarray,
    r_probability: np.ndarray,
    f_probability: np.ndarray,
    game_type: np.ndarray,
    alpha_r: float,
    alpha_f: float,
) -> np.ndarray:
    base = probability_vector(baseline)
    r_pred = probability_vector(r_probability, len(base))
    f_pred = probability_vector(f_probability, len(base))
    segment = np.asarray(game_type, dtype=str)
    if segment.shape != base.shape or not np.isin(segment, ["R", "F"]).all():
        raise RFDecisionError("game_type differs")
    alpha = np.where(segment == "R", checked_alpha(alpha_r), checked_alpha(alpha_f))
    expert = np.where(segment == "R", r_pred, f_pred)
    return np.clip((1.0 - alpha) * base + alpha * expert, 1e-5, 1 - 1e-5)
```

For F-only candidates, pass `alpha_r=0.0` and use the baseline vector as the R expert
stand-in. `checked_alpha` accepts zero only for F-only routing; searched non-zero
values must come from the contract grid.

- [ ] **Step 4: Implement deterministic two-fold selection and three-fold acceptance**

Define frozen `RFStructureEvidence`, `RFStructureDecision`, `RFAcceptanceEvidence`,
and `RFAcceptanceDecision`. Enumerate both F heads, four F-only weights, and all 16
R/F weight pairs. Score only `contract.folds[:2]`. Apply tie-break keys in this exact
order:

```python
key = (
    selection_brier,
    0 if f_head == "f_small" else 1,
    alpha_r + alpha_f,
    0 if alpha_r == 0.0 else 1,
    alpha_r,
    alpha_f,
)
```

After selection, compute the untouched recent-fold metrics without allowing them to
change `selection_key`. Acceptance must enforce weighted gain ≥ 0.00010, at least two
positive fold gains, R and F regression ≤ 0.00010, recent F gain > 0, finite bounded
probabilities, and all required seeds.

- [ ] **Step 5: Run, verify, and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_decisions.py -q
git add experiments/tree_expert/rf_decisions.py tests/test_tree_expert_rf_decisions.py
git commit -m "feat(tree-expert): select and gate RF experts"
```

### Task 4: Atomic segment-specific training jobs

**Files:**
- Create: `experiments/tree_expert/rf_training.py`
- Test: `tests/test_tree_expert_rf_training.py`

- [ ] **Step 1: Write failing job tests with a recording model**

```python
def test_f_job_trains_only_f_rows_and_uses_anchor_residual(tmp_path):
    model = RecordingRegressor(prediction=0.05)
    result = run_rf_job(
        job=job(head="f_small", segment="F"), train=TRAIN, valid=VALID,
        baseline=BASELINE, output_dir=tmp_path, model_factory=lambda _: model,
        feature_builder=fake_feature_builder, feature_transformer=fake_transformer,
    )
    assert model.fit_row_ids == tuple(TRAIN.loc[TRAIN.game_type.eq("F"), "row_id"])
    np.testing.assert_allclose(model.fit_target, F_TARGET - F_ANCHOR)
    assert result.status == "completed"


def test_job_writes_model_prediction_and_metrics_atomically(tmp_path):
    result = completed_job(tmp_path)
    assert result.model_path.name == "checkpoint.cbm"
    assert result.predictions_path.name == "predictions.csv"
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_job_rejects_missing_segment_in_training_fold(tmp_path):
    with pytest.raises(RFTrainingError, match="segment training rows are empty"):
        run_rf_job(job=f_job(), train=R_ONLY_TRAIN, valid=VALID, baseline=BASELINE, output_dir=tmp_path)
```

- [ ] **Step 2: Confirm the focused test fails**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_training.py -q
```

- [ ] **Step 3: Implement fold-first, segment-second training**

Use this ordering so validation-year rows can never enter feature fitting:

```python
prefix = train.loc[pd.to_numeric(train["season"], errors="raise").le(job.train_end_year)].copy()
validation = valid.loc[pd.to_numeric(valid["season"], errors="raise").eq(job.valid_year)].copy()
segment_train = prefix.loc[prefix["game_type"].astype(str).eq(job.segment)].copy()
state, train_batch = feature_builder(segment_train, None, valid_year=job.valid_year, use_trackman=False)
valid_batch = feature_transformer(validation.drop(columns="control_success"), state)
residual = np.asarray(train_batch.target, dtype="float64") - np.asarray(train_batch.anchor, dtype="float64")
```

The expert predicts every validation row so routing can remain a pure current-row
operation. Compute expert probability as `clip(valid_anchor + model.predict(x), 1e-5,
1-1e-5)`. Save `job.json`, `metrics.json`, `predictions.csv`, `checkpoint.cbm`, and
`worker.log` via temporary sibling files followed by `os.replace`.

- [ ] **Step 4: Bind parameters to the chosen head**

Build CatBoost parameters from `catboost_common` plus the exact head settings and:

```python
parameters.update(
    random_seed=job.seed,
    devices=str(gpu_id),
    train_dir=str(output_dir / "catboost_info"),
    loss_function="RMSE",
    eval_metric="RMSE",
)
```

Log segment row counts, validation row counts, best iteration, segment Brier, overall
Brier, baseline Brier, gain, and elapsed seconds. Reject non-finite predictions,
misaligned row IDs/targets, unsupported game types, and fewer than 5,000 training rows
for the selected segment.

- [ ] **Step 5: Run, verify, and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_training.py -q
git add experiments/tree_expert/rf_training.py tests/test_tree_expert_rf_training.py
git commit -m "feat(tree-expert): train RF expert jobs"
```

### Task 5: Diagnostics, final fit, and row-independent inference

**Files:**
- Create: `experiments/tree_expert/rf_diagnostics.py`
- Create: `experiments/tree_expert/rf_full_fit.py`
- Create: `experiments/tree_expert/rf_inference.py`
- Test: `tests/test_tree_expert_rf_diagnostics.py`
- Test: `tests/test_tree_expert_rf_full_fit.py`
- Test: `tests/test_tree_expert_rf_inference.py`

- [ ] **Step 1: Write failing diagnostics tests**

```python
def test_diagnostics_reports_required_segments_and_calibration():
    report = build_rf_diagnostics(FRAME, BASELINE, CANDIDATE, minimum_rows=2)
    assert {row["segment"] for row in report["game_type"]} == {"R", "F"}
    assert len(report["calibration"]) == 10
    assert "prediction_correlation" in report


def test_diagnostics_ignores_small_reporting_segments_without_changing_predictions():
    report = build_rf_diagnostics(FRAME, BASELINE, CANDIDATE, minimum_rows=10_000)
    assert report["game_type"] == []
    np.testing.assert_array_equal(CANDIDATE, CANDIDATE_COPY)
```

- [ ] **Step 2: Write failing full-fit and inference tests**

```python
def test_full_fit_is_blocked_for_rejected_decision(tmp_path):
    with pytest.raises(RFFullFitError, match="accepted decision is required"):
        create_rf_full_fit_token(REJECTED, EVIDENCE, tmp_path)


def test_inference_is_identical_for_singleton_shuffle_and_batches():
    full = predictor.predict(FRAME)
    singleton = np.array([predictor.predict(FRAME.iloc[[i]])[0] for i in range(len(FRAME))])
    reverse = predictor.predict(FRAME.iloc[::-1])[::-1]
    batched = np.concatenate([predictor.predict(part) for part in np.array_split(FRAME, 3)])
    np.testing.assert_allclose(full, singleton, atol=1e-6)
    np.testing.assert_allclose(full, reverse, atol=1e-6)
    np.testing.assert_allclose(full, batched, atol=1e-6)
```

- [ ] **Step 3: Implement diagnostics without feeding metrics back into inference**

Return JSON-safe objects for overall and fold Brier, R/F Brier and counts, calibration
deciles defined from fixed probability edges `np.linspace(0, 1, 11)`, prediction and
residual correlations, and maximum eligible segment regression. Diagnostics may inspect
validation targets but must not mutate the selected weights or any model state.

- [ ] **Step 4: Implement accepted-only full-fit tokens**

```python
@dataclass(frozen=True)
class RFFullFitToken:
    f_head: str
    include_r: bool
    alpha_r: float
    alpha_f: float
    seeds: tuple[int, ...]
    iterations: Mapping[str, Mapping[int, int]]
```

Derive each full-fit iteration count as the rounded median of the three fold best
iterations plus one, clamped to `[1, contract.experts[head]["iterations"]]`. Train only
the selected F head and, when selected, R head for all three seeds. Fit each head's
feature state using only its segment's official training rows. Save each state and model
under `full_fit/<head>/frozen_state/` once per head and save models under
`full_fit/<head>/models/seed_<seed>.cbm`.

- [ ] **Step 5: Implement inference with fixed routing only**

Load the E2 baseline predictor from the verified input, average the three E2 seed
probabilities, average each selected expert's three seed probabilities, and call
`route_probability` with the frozen `alpha_r` and `alpha_f`. Unknown or missing
`game_type` must raise `RFInferenceError`; it must never fall back based on other test
rows. Expose `audit_row_independence(frame, predictor, tolerance)` that performs full,
singleton, shuffled, reversed, companion-row, and batch-size checks.

- [ ] **Step 6: Run, verify, and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_rf_diagnostics.py \
  tests/test_tree_expert_rf_full_fit.py \
  tests/test_tree_expert_rf_inference.py -q
git add experiments/tree_expert/rf_diagnostics.py experiments/tree_expert/rf_full_fit.py \
  experiments/tree_expert/rf_inference.py tests/test_tree_expert_rf_diagnostics.py \
  tests/test_tree_expert_rf_full_fit.py tests/test_tree_expert_rf_inference.py
git commit -m "feat(tree-expert): finalize and audit RF experts"
```

### Task 6: Review, resume, and accepted-only delivery artifacts

**Files:**
- Create: `experiments/tree_expert/rf_artifacts.py`
- Test: `tests/test_tree_expert_rf_artifacts.py`

- [ ] **Step 1: Write failing artifact-contract tests**

```python
def test_resume_round_trip_preserves_completed_jobs(tmp_path):
    bundle = create_rf_resume(CAMPAIGN_ROOT, tmp_path / "resume.zip", BINDINGS)
    restored = restore_rf_resume(bundle, tmp_path / "restored", BINDINGS)
    assert restored.completed_job_ids == COMPLETED_IDS
    assert not tuple(restored.root.rglob("*.tmp"))


def test_resume_rejects_identity_mismatch(tmp_path):
    bundle = create_rf_resume(CAMPAIGN_ROOT, tmp_path / "resume.zip", BINDINGS)
    with pytest.raises(RFArtifactError, match="artifact bindings differ"):
        verify_rf_resume(bundle, replace(BINDINGS, code_sha256="f" * 64))


def test_delivery_is_blocked_until_accepted(tmp_path):
    with pytest.raises(RFArtifactError, match="accepted decision is required"):
        create_rf_delivery(CAMPAIGN_ROOT, tmp_path / "delivery.zip", REJECTED, BINDINGS)
```

- [ ] **Step 2: Confirm the focused test fails**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_artifacts.py -q
```

- [ ] **Step 3: Implement deterministic manifests and safe ZIP handling**

Define `RFBindings` with contract, code, input manifest, official train/history, and E2
handoff hashes. Each bundle manifest must contain exact kind, schema version, bindings,
member size, and SHA-256. ZIP validation must reject duplicate members, unsafe paths,
symlinks, encryption, undeclared content, missing content, and hash/size differences.

Resume collection must use an allowlist:

```python
def is_stable_member(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and not path.name.startswith(".")
        and not path.name.endswith(".tmp")
        and "catboost_info" not in path.parts
    )
```

Create a ZIP in a temporary sibling and `os.replace` only after verifying the temporary
ZIP. Review always includes decisions, metrics, predictions, diagnostics, logs, and
manifests. Delivery additionally requires `decision.status == "accepted"`, full-fit
models/states, fixed routing weights, and a passing independence audit.

- [ ] **Step 4: Run, verify, and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_artifacts.py -q
git add experiments/tree_expert/rf_artifacts.py tests/test_tree_expert_rf_artifacts.py
git commit -m "feat(tree-expert): package RF campaign artifacts"
```

### Task 7: Staged two-GPU runner and safe deadlines

**Files:**
- Create: `experiments/tree_expert/rf_runner.py`
- Test: `tests/test_tree_expert_rf_runner.py`

- [ ] **Step 1: Write failing state-machine tests**

```python
def test_runner_reuses_completed_jobs_and_runs_only_missing_jobs(tmp_path):
    restored_campaign(tmp_path, completed={"rf__f_small__tr2021__va2022__s3407"})
    result = run_rf_campaign(VERIFIED, tmp_path, resume_bundle=RESUME, runtime=FAKE_RUNTIME)
    assert "rf__f_small__tr2021__va2022__s3407" in result.reused_jobs
    assert "rf__f_small__tr2021__va2022__s3407" not in FAKE_RUNTIME.started_jobs


def test_runner_stops_after_structure_rejection(tmp_path):
    result = run_rf_campaign(VERIFIED, tmp_path, runtime=rejected_structure_runtime())
    assert result.status == "rejected"
    assert not any("s42" in item or "s2026" in item for item in result.started_jobs)
    assert result.delivery_path is None


def test_runner_pauses_before_new_job_guard_and_writes_resume(tmp_path):
    result = run_rf_campaign(VERIFIED, tmp_path, absolute_deadline=NEAR_DEADLINE, runtime=FAKE_RUNTIME)
    assert result.status == "paused"
    assert result.resume_path.is_file()
```

- [ ] **Step 2: Write the snapshot race regression**

```python
def test_snapshot_ignores_prediction_temp_file_during_atomic_replace(tmp_path):
    active = tmp_path / "jobs" / "rf__f_small" / ".predictions.csv.tmp"
    active.parent.mkdir(parents=True)
    active.write_text("partial", encoding="utf-8")
    snapshot = create_stable_snapshot(tmp_path, tmp_path / "snapshot.zip", BINDINGS)
    assert ".predictions.csv.tmp" not in zip_names(snapshot)
    verify_rf_resume(snapshot, BINDINGS)
```

- [ ] **Step 3: Implement explicit campaign stages**

Persist `campaign_state.json` with one of:

```python
STRUCTURE = "structure"
CONFIRMATION = "confirmation"
FULL_FIT = "full_fit"
COMPLETED = "completed"
REJECTED = "rejected"
PAUSED = "paused"
```

Run the six jobs for the first two selection folds first. For each head, evaluate every
allowed blend weight on its own segment. Drop a head only when no allowed weight improves
either selection fold; this avoids discarding an overconfident expert that becomes useful
after shrinkage. If no F head survives, reject because the required recent-F gate cannot
be met. Run recent-fold jobs only for surviving heads, then build and persist the fixed
structure decision. If rejected, create review/resume and stop. Otherwise enumerate only
the selected confirmation heads. Build three-seed acceptance evidence. If rejected,
create review/resume and stop. Only an accepted decision can enter full-fit and delivery
creation. `structure_jobs()` returns nine potential jobs, but the runner deliberately
does not launch recent-fold jobs for heads eliminated by the two-fold screen.

- [ ] **Step 4: Implement two-worker dispatch and progress logs**

Use two spawned worker processes with explicit GPU IDs `0` and `1`; do not rely on
`DataParallel`. The parent owns the queue and emits these stable prefixes:

```text
TREE_RF_STAGE_SELECTED stage=structure
TREE_RF_JOB_START job=<id> gpu=<0|1>
TREE_RF_JOB_REUSED job=<id>
TREE_RF_TRAINING_PROGRESS job=<id> iteration=<n>/800 best_brier=<value> eta_seconds=<n>
TREE_RF_JOB_END job=<id> status=<completed|failed>
TREE_RF_DECISION status=<passed|accepted|rejected> reason=<reason>
```

Before dispatching a new job, require at least `new_job_guard_seconds`. Before full-fit,
require at least `full_fit_guard_seconds`. On SIGTERM, KeyboardInterrupt, worker failure,
or deadline, stop launching jobs, terminate only active child processes, validate stable
outputs, and create one resume.

- [ ] **Step 5: Run, verify, and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_runner.py -q
git add experiments/tree_expert/rf_runner.py tests/test_tree_expert_rf_runner.py
git commit -m "feat(tree-expert): orchestrate RF campaign"
```

### Task 8: Kaggle discovery, one-download supervision, and one-cell launcher

**Files:**
- Create: `experiments/tree_expert/rf_kaggle.py`
- Create: `experiments/tree_expert/KAGGLE_RF_CELL.py`
- Create: `tools/build_tree_expert_rf_kaggle_cell.py`
- Test: `tests/test_tree_expert_rf_kaggle.py`
- Test: `tests/test_tree_expert_rf_kaggle_cell.py`

- [ ] **Step 1: Write failing discovery tests for ZIP and expanded datasets**

```python
def test_discovery_accepts_expanded_compact_input(tmp_path):
    expanded_dataset(tmp_path / "tree_expert_rf_input")
    official_dataset(tmp_path / "lg-aimers-9th-data")
    found = discover_rf_inputs(tmp_path)
    assert found.rf_input.name == "tree_expert_rf_input"


def test_discovery_rejects_two_resumes_even_if_nested(tmp_path):
    resume_fixture(tmp_path / "a" / "tree_expert_rf_resume")
    resume_fixture(tmp_path / "b" / "tree_expert_rf_resume.zip")
    with pytest.raises(RFKaggleError, match="resume count must be zero or one"):
        discover_rf_inputs(tmp_path)
```

- [ ] **Step 2: Write failing single-download tests**

```python
def test_supervisor_downloads_only_final_stable_resume():
    downloads = []
    run_supervised_rf_campaign(run_campaign=paused_campaign, download=downloads.append)
    assert len([item for item in downloads if item.kind == "resume"]) == 1


def test_accepted_run_downloads_review_resume_and_delivery_once_each():
    downloads = []
    run_supervised_rf_campaign(run_campaign=accepted_campaign, download=downloads.append)
    assert [item.kind for item in downloads] == ["resume", "review", "delivery"]
```

- [ ] **Step 3: Implement discovery by artifact identity, not filename alone**

Scan `/kaggle/input` for both `.zip` files and expanded directories. Read only small
manifest candidates to classify `tree_expert_rf_input_v1` and `tree_expert_rf_resume_v1`.
Official data discovery must verify the fixed train/history hashes. Require exactly one
official dataset, exactly one RF input, and zero or one resume. Deduplicate a ZIP and its
expanded copy only when their manifest identity and member hashes are identical;
otherwise report both paths and stop.

- [ ] **Step 4: Implement the generated cell contract**

The checked-in cell must:

1. print `TREE_RF_CODE_READY sha256=<...> size_bytes=<...>`;
2. install only missing pinned requirements;
3. print both T4 devices or stop with `TREE_RF_ERROR stage=gpu`;
4. discover and verify inputs before creating campaign outputs;
5. run the campaign with a 5h30 deadline;
6. stream progress at least every 30 seconds while work advances;
7. retain uploaded paths in an in-memory cache for same-runtime reruns;
8. request each final artifact download once;
9. print `TREE_RF_CAMPAIGN_SUCCESS status=<accepted|rejected|paused>` or
   `TREE_RF_ERROR stage=<stage> type=<type> message=<message>`.

Embed a deterministic tar.gz runtime as base64 and verify its SHA-256 before extraction.
Reject an embedded cell at or above 900,000 UTF-8 bytes, leaving margin below Kaggle's
1 MB kernel-source limit.

- [ ] **Step 5: Build twice and assert byte stability**

```bash
artifacts/tabm_submission_python311/bin/python tools/build_tree_expert_rf_kaggle_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_RF_CELL.py
artifacts/tabm_submission_python311/bin/python tools/build_tree_expert_rf_kaggle_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_RF_CELL.py
```

Expected: both SHA-256 values are identical.

- [ ] **Step 6: Run, verify, and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_rf_kaggle.py tests/test_tree_expert_rf_kaggle_cell.py -q
git add experiments/tree_expert/rf_kaggle.py experiments/tree_expert/KAGGLE_RF_CELL.py \
  tools/build_tree_expert_rf_kaggle_cell.py tests/test_tree_expert_rf_kaggle.py \
  tests/test_tree_expert_rf_kaggle_cell.py
git commit -m "feat(tree-expert): add RF Kaggle handoff"
```

### Task 9: Cross-module regression and handoff preparation

**Files:**
- Modify: `README.md`
- Test: all `tests/test_tree_expert_rf_*.py`
- Test: existing E2/T3 regression tests

- [ ] **Step 1: Add a short human-readable README entry**

Document only:

```markdown
### R/F 경기 유형 전문가

기존 977점 CatBoost 모델을 고정하고 정규 시즌(R)과 F 경기의 전용 모델이
시간 순서 검증에서 추가 개선을 만드는지 확인하는 독립 실험이다. Kaggle
실행 결과가 `accepted`인 경우에만 제출 후보 제작 단계로 넘어간다.
```

Link the design and implementation-plan documents. Do not describe an unverified score
or imply the candidate is accepted.

- [ ] **Step 2: Run all new focused tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_rf_*.py -q
```

Expected: every RF test passes.

- [ ] **Step 3: Run existing E2 and T3 regression tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_e2_*.py tests/test_tree_expert_t3_*.py -q
```

Expected: every selected regression test passes. Do not change user-owned TabM files to
resolve failures unrelated to this campaign.

- [ ] **Step 4: Run static checks and inspect campaign boundaries**

```bash
artifacts/tabm_submission_python311/bin/python -m compileall -q experiments/tree_expert tools/prepare_tree_expert_rf_input.py tools/build_tree_expert_rf_kaggle_cell.py
git diff --check
rg -n "submission|test.*groupby|test.*value_counts|test.*rank|test.*rolling" experiments/tree_expert/rf_*.py experiments/tree_expert/KAGGLE_RF_CELL.py
```

Expected: compile and diff checks succeed. Any search hit must be either the explicit
submission-package prohibition or a test/audit name, never an evaluation-data aggregate.

- [ ] **Step 5: Commit documentation and final generated artifacts**

```bash
git add README.md experiments/tree_expert/KAGGLE_RF_CELL.py
git commit -m "docs: explain RF expert experiment"
```

- [ ] **Step 6: Prepare the user-run input without starting heavy work**

After confirming the exact local E2 handoff path, run only the local archive preparation:

```bash
cd /Users/yonghyun/Documents/lg-aimers-9th-competition
artifacts/tabm_submission_python311/bin/python tools/prepare_tree_expert_rf_input.py \
  --e2-handoff "/Users/yonghyun/Downloads/tree_expert_e2_handoff (1).zip" \
  --output artifacts/tree_expert_rf_input.zip
```

Expected terminal prefix: `TREE_RF_INPUT_SUCCESS`. Record the output SHA-256 and the
checked-in cell SHA-256. This operation is local packaging only; it does not train a
model or create a submission.

### Task 10: Final handoff review

**Files:**
- Review: `experiments/tree_expert/KAGGLE_RF_CELL.py`
- Review: `artifacts/tree_expert_rf_input.zip`
- Review: current git status and commit history

- [ ] **Step 1: Verify only intended files were committed**

```bash
git status --short
git log --oneline --decorate -12
```

Expected: the user's pre-existing TabM modifications may remain unstaged, but no RF file
is untracked or unexpectedly modified.

- [ ] **Step 2: Produce one complete user-run instruction**

The handoff message must state:

- purpose: test R/F experts against the 977-point E2 baseline;
- Kaggle inputs: official `lg-aimers-9th-data` plus `tree_expert_rf_input` only;
- accelerator: two Tesla T4 GPUs;
- Internet: off after any required dependency availability is confirmed;
- expected runtime: 1.5–3 hours if rejected early, up to 5.5 hours if accepted/full-fit;
- rerun safety: add only the newest `tree_expert_rf_resume` dataset and rerun the same cell;
- success markers: `TREE_RF_CAMPAIGN_SUCCESS`, `TREE_RF_DECISION`;
- error marker: the complete line beginning `TREE_RF_ERROR`;
- files to return: review and resume always, delivery only when produced;
- omission impact: without this run there is no evidence to include R/F experts in a submission.

- [ ] **Step 3: Stop before heavy execution or submission packaging**

Do not run Kaggle, full-data OOF, full-fit, or submission creation locally. Wait for the
user to run the cell and return its artifacts. After artifacts arrive, verify acceptance
and hashes before designing or invoking any DACON submission-packaging path.
