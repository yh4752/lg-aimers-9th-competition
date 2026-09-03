# Tree Hierarchical Residual Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rule-safe, resumable H1/H2/H3 Kaggle campaign that tests a hierarchical second CatBoost residual and rolling hierarchical calibration on top of the accepted Tree Expert E2 candidate, while preventing any model delivery before fixed temporal OOF gates pass.

**Architecture:** Add one isolated `tree_hierarchical` campaign beside the existing E2/T3 code. It verifies and reuses the accepted E2 handoff, creates only the missing frozen `2020→2021` E2 source fold, builds prior-season hierarchy tables, trains rolling C1 residual models, evaluates optional C2 calibration, and emits a deterministic stage-aware handoff. Training, validation, state, artifact, and Kaggle orchestration boundaries stay separate so interrupted work can be safely reused.

**Tech Stack:** Python 3.11/3.12, pandas, NumPy, CatBoost 1.2.10, PyTorch/TabM only for the frozen source baseline, pytest, deterministic ZIP artifacts, Kaggle T4x2.

---

## Guardrails

- Full-data training and Kaggle execution are user-run operations. Development runs only static, synthetic, and fixture-based tests.
- Do not modify or regenerate existing TabM notebooks or the user's unrelated dirty files.
- Do not create a submission ZIP, submission script, or submission-package entry point in this plan.
- H3 may create a verified model delivery only after C1 or C2 has passed its registered gates with matching hashes.
- T3 is closed negative evidence and is not an input, fallback, or hidden blend component.
- The failure-label audit remains an independent CPU experiment and is not read by this campaign.

## Fixed experiment registry

The contract written in Task 1 must contain these exact values. Tests compare the entire object, not selected fields.

```json
{
  "schema_version": 1,
  "campaign_id": "tree_hierarchical_residual_v1",
  "review_only": false,
  "submission_package": false,
  "inputs": {
    "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
    "e2_handoff_sha256": "4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f"
  },
  "folds": [[2020, 2021], [2021, 2022], [2022, 2023], [2023, 2024]],
  "structure_folds": [[2021, 2022], [2022, 2023]],
  "confirmation_fold": [2023, 2024],
  "seeds": [3407, 42, 2026],
  "profiles": {
    "hc_strong": {"identity_k": 400.0, "context_k": 200.0, "interaction_k": 1600.0},
    "hc_balanced": {"identity_k": 200.0, "context_k": 100.0, "interaction_k": 800.0},
    "hc_light": {"identity_k": 80.0, "context_k": 40.0, "interaction_k": 320.0}
  },
  "minimum_group_rows": {"identity": 20, "context": 50, "interaction": 100},
  "profile_tie_order": ["hc_strong", "hc_balanced", "hc_light"],
  "calibration_alphas": [0.25, 0.5, 0.75, 1.0],
  "seed_ensemble": "arithmetic_probability_mean",
  "full_fit_iterations": {"rule": "median_best_iteration_plus_one", "minimum": 50, "maximum": 800},
  "residual_catboost": {
    "iterations": 800,
    "depth": 8,
    "learning_rate": 0.04,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.5,
    "border_count": 128,
    "max_ctr_complexity": 2,
    "one_hot_max_size": 16,
    "od_type": "Iter",
    "od_wait": 60,
    "task_type": "GPU",
    "allow_writing_files": true,
    "loss_function": "RMSE"
  },
  "calibration": {"effect_clip": 0.25, "probability_clip": 0.00001, "ece_bins": 10},
  "gates": {
    "c1_weighted_gain": 0.00010,
    "c1_confirmation_gain": 0.0,
    "c1_min_fold_gain": -0.00005,
    "c1_max_segment_regression": 0.00035,
    "c2_weighted_gain": 0.00013,
    "c2_incremental_gain": 0.00003,
    "c2_min_fold_gain": -0.00003,
    "c2_max_segment_regression": 0.00020,
    "bootstrap_repeats": 1000,
    "bootstrap_seed": 3407,
    "minimum_segment_rows": 5000,
    "simpler_tie_margin": 0.00002,
    "required_non_worse_seeds": 2,
    "seed_fold_tolerance": 0.0
  },
  "runtime": {
    "h1_wall_seconds": 21600,
    "h2_wall_seconds": 28800,
    "h3_wall_seconds": 18000,
    "new_job_guard_seconds": 600,
    "final_fit_guard_seconds": 1800,
    "snapshot_interval_seconds": 900,
    "inference_rows": 245789,
    "inference_max_seconds": 480,
    "model_state_max_bytes": 2147483648,
    "probability_tolerance": 1e-12
  }
}
```

Calibration source policy is also fixed:

- C1 prediction for 2022 trains on 2021 OOF only.
- C1 prediction for 2023 trains on 2021–2022 OOF only.
- C1 prediction for 2024 trains on 2021–2023 OOF only.
- Final C1 trains on 2021–2024 verified OOF pairs only.
- C2 prediction for 2022 uses the frozen E2 `2020→2021` error table because no earlier C1 prediction exists.
- C2 prediction for 2023 uses C1 OOF errors from 2022.
- C2 prediction for 2024 uses C1 OOF errors from 2022–2023.
- Final C2 state uses the frozen E2 2021 errors and C1 OOF errors from 2022–2024.

Profile selection uses the lowest row-count-weighted Brier over the two structure folds at
seed 3407. An exact tie within `1e-12` follows `hc_strong`, `hc_balanced`, then `hc_light`.
The 2024 result confirms or rejects evidence but never reopens profile selection. H2 C1
probabilities are the arithmetic probability mean of seeds 3407, 42, and 2026. H3 uses
the same mean, with each seed's full-fit iteration count equal to the median of that seed's
three temporal best iterations plus one, clipped to `[50, 800]`.

## File map

| Path | Responsibility |
|---|---|
| `experiments/tree_expert/hc_contract.json` | Immutable registry above. |
| `experiments/tree_expert/hc_contracts.py` | Strict parsing, hashes, stage/job schedule. |
| `experiments/tree_expert/hc_inputs.py` | Official-data and accepted-E2 handoff verification/extraction/deduplication. |
| `experiments/tree_expert/hc_base.py` | Frozen TabM plus E2 residual extension for `2020→2021`. |
| `experiments/tree_expert/hc_features.py` | Prior-season hierarchy tables and row-local lookups. |
| `experiments/tree_expert/hc_training.py` | Rolling C1 datasets, CatBoost residual jobs, prediction validation. |
| `experiments/tree_expert/hc_calibration.py` | Rolling C2 effect state, fallback, alpha application. |
| `experiments/tree_expert/hc_metrics.py` | Paired Brier, ECE, calibration gap, segments, clustered bootstrap. |
| `experiments/tree_expert/hc_decisions.py` | Profile selection and independent C1/C2 gates. |
| `experiments/tree_expert/hc_state.py` | Immutable completed-job state and stage transition rules. |
| `experiments/tree_expert/hc_artifacts.py` | Deterministic review/resume/handoff/model-delivery artifacts. |
| `experiments/tree_expert/hc_runner.py` | H1/H2/H3 scheduler and two-GPU orchestration. |
| `experiments/tree_expert/hc_kaggle.py` | Kaggle discovery, stage selection, logging, deadline/error handoff. |
| `experiments/tree_expert/KAGGLE_HC_CELL.py` | Generated single-cell user entry point. |
| `tools/build_tree_hc_kaggle_cell.py` | Deterministically embeds the runtime below 1 MB. |
| `tests/test_tree_expert_hc_contracts.py` | Contract and job registry tests. |
| `tests/test_tree_expert_hc_inputs.py` | Input, archive, identity, and deduplication tests. |
| `tests/test_tree_expert_hc_base.py` | Frozen 2021 OOF source tests. |
| `tests/test_tree_expert_hc_features.py` | Leakage, shrinkage, fallback, row-independence tests. |
| `tests/test_tree_expert_hc_training.py` | Rolling C1 data and restart tests. |
| `tests/test_tree_expert_hc_calibration.py` | C2 source windows and fallback tests. |
| `tests/test_tree_expert_hc_metrics.py` | Metric and bootstrap tests. |
| `tests/test_tree_expert_hc_decisions.py` | Gate and tie-break tests. |
| `tests/test_tree_expert_hc_state.py` | Stage/state/reuse tests. |
| `tests/test_tree_expert_hc_artifacts.py` | Deterministic and safe artifact tests. |
| `tests/test_tree_expert_hc_runner.py` | Synthetic H1/H2/H3 orchestration tests. |
| `tests/test_tree_expert_hc_kaggle.py` | Discovery, logging, and error-path tests. |
| `tests/test_tree_expert_hc_kaggle_cell.py` | Generated-cell determinism and size tests. |
| `docs/tree_hierarchical_campaign.md` | Human runbook and evidence interpretation. |

## Task 1: Lock the contract and stage schedule

**Files:**
- Create: `experiments/tree_expert/hc_contract.json`
- Create: `experiments/tree_expert/hc_contracts.py`
- Test: `tests/test_tree_expert_hc_contracts.py`

- [ ] **Step 1: Write the failing strict-contract tests**

```python
def test_contract_matches_registered_experiment():
    contract = load_hc_contract()
    assert contract.campaign_id == "tree_hierarchical_residual_v1"
    assert contract.profiles["hc_balanced"].interaction_k == 800.0
    assert contract.calibration_alphas == (0.25, 0.5, 0.75, 1.0)
    assert contract.runtime.h1_wall_seconds == 21600
    assert contract.submission_package is False


def test_jobs_are_finite_and_stage_registered():
    jobs = build_hc_jobs(load_hc_contract())
    assert {job.stage for job in jobs} == {"H1", "H2", "H3"}
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert all(job.seed in {42, 2026, 3407} for job in jobs)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_contracts.py -q`

Expected: `ModuleNotFoundError: No module named 'experiments.tree_expert.hc_contracts'`.

- [ ] **Step 3: Add the exact JSON registry and frozen dataclasses**

Implement strict exact-key validation. Reject booleans where integers are expected, non-finite floats, reordered/extra folds, unknown profiles, and any `submission_package != false`.

```python
@dataclass(frozen=True)
class HCJob:
    job_id: str
    stage: Literal["H1", "H2", "H3"]
    kind: Literal["source_e2", "c1_residual", "c2_calibration", "full_fit"]
    train_end_year: int | None
    valid_year: int | None
    seed: int
    profile: str | None


def contract_sha256() -> str:
    return file_sha256(Path(__file__).with_name("hc_contract.json"))
```

The job builder must register:

- H1: one frozen TabM baseline for `2020→2021`, three E2 residual seeds for that source fold, six C1 structure jobs (three profiles × two structure folds, seed 3407), and three conditional 2024 confirmation jobs. Exactly the selected profile's confirmation job runs; the other two receive an immutable `skipped_not_selected` state.
- H2: selected-profile C1 jobs for seeds 42 and 2026 over 2022, 2023, and 2024; C2 alpha evaluation is CPU work recorded as registered decision jobs.
- H3: conditional full C1 fit jobs for seeds 3407, 42, and 2026 plus one conditional C2 state fit. C1 fits run only when C1 or C2 is accepted; the C2 state fit runs only when C2 wins. Every inapplicable registered job is explicitly skipped.

- [ ] **Step 4: Run contract tests and confirm GREEN**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_contracts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/hc_contract.json experiments/tree_expert/hc_contracts.py tests/test_tree_expert_hc_contracts.py
git commit -m "feat: register hierarchical residual campaign"
```

## Task 2: Verify and materialize the accepted E2 evidence

**Files:**
- Create: `experiments/tree_expert/hc_inputs.py`
- Test: `tests/test_tree_expert_hc_inputs.py`

- [ ] **Step 1: Write failing input and archive tests**

Cover exact handoff SHA, accepted status, required nested E2 review members, row hashes, official train/history hashes, ZIP-slip, duplicate members, symlinks, compression ratio limits, expanded-directory inputs, and same-identity deduplication.

```python
def test_same_resume_identity_is_deduplicated(tmp_path, accepted_handoff):
    expanded = expand(accepted_handoff, tmp_path / "expanded")
    found = discover_hc_inputs([accepted_handoff, expanded])
    assert found.previous_handoff is None
    assert found.e2_handoff == accepted_handoff


def test_distinct_previous_handoffs_fail(tmp_path, two_distinct_handoffs):
    with pytest.raises(HCInputError, match="distinct handoff identities"):
        discover_hc_inputs(two_distinct_handoffs)
```

- [ ] **Step 2: Run and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_inputs.py -q`

Expected: missing `hc_inputs` module.

- [ ] **Step 3: Implement verified extraction**

Call public `verify_e2_handoff()` before opening nested members. Extract only:

```text
decisions/acceptance.json
ensembles/2021_2022.csv
ensembles/2022_2023.csv
ensembles/2023_2024.csv
tree_expert_e2_model_delivery.zip
```

Every prediction CSV must have unique ordered `row_id`, binary target, finite clipped probability, and the expected validation season. Do not import private `t3_inputs._read_e2_handoff`.

Expose:

```python
@dataclass(frozen=True)
class VerifiedHCEvidence:
    root: Path
    manifest_sha256: str
    e2_handoff_sha256: str
    acceptance: Path
    e2_delivery: Path
    fold_predictions: Mapping[tuple[int, int], Path]
```

- [ ] **Step 4: Add a deterministic local input preparer**

Add `prepare_hc_input(e2_handoff, output)` in the same module. It creates a small verified input ZIP containing E2 evidence only; official data remains a separate Kaggle dataset. The outer manifest binds every member and the contract hash.

- [ ] **Step 5: Run focused and existing E2 artifact tests**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_inputs.py tests/test_tree_expert_e2_artifacts.py tests/test_tree_expert_t3_inputs.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/tree_expert/hc_inputs.py tests/test_tree_expert_hc_inputs.py
git commit -m "feat: verify hierarchical campaign inputs"
```

## Task 3: Create the missing frozen E2 `2020→2021` source fold

**Files:**
- Create: `experiments/tree_expert/hc_base.py`
- Test: `tests/test_tree_expert_hc_base.py`

- [ ] **Step 1: Write failing source-fold tests**

Test that the TabM request exactly matches the E2 p2 configuration except for fold and candidate ID, that all three E2 CatBoost residual seeds share the same frozen baseline, and that an existing complete source fold is reused only when its identity and files match.

```python
def test_source_baseline_is_2020_to_2021_and_frozen(contract, cache):
    request = make_source_request(cache, contract, checkpoint_binding=BINDING)
    assert request.seed == 3407
    assert request.model_config["architecture"] == "tabm"
    assert request.model_config["num_embedding"] == "piecewise_linear"
    assert request.candidate_id.endswith("__tr2020__va2021__s3407")
```

- [ ] **Step 2: Run and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_base.py -q`

Expected: missing `hc_base` module.

- [ ] **Step 3: Implement the frozen baseline adapter**

Reuse the public cache and training abstractions used by `e2_baseline.py`, but define an HC-owned identity for fold `(2020, 2021)`. Copy no checkpoint from another fold. Align the resulting validation rows against official 2021 rows and emit the standard E2 prediction columns.

- [ ] **Step 4: Wrap existing E2 residual training for three source seeds**

Create HC-owned `E2Job` values with candidate `c1_anchor_residual`, fold `(2020, 2021)`, and seeds `(3407, 42, 2026)`. Call `run_e2_fold_job()` with the verified frozen baseline, then average the three row-aligned probabilities into `source_e2_2021.csv`.

Reject any seed output whose target, row order, fold identity, or baseline binding differs.

- [ ] **Step 5: Test restart identity and ensemble determinism**

The same completed jobs must be reused byte-for-byte. A changed contract/code/data/input hash must fail instead of silently retraining into the old directory.

- [ ] **Step 6: Run focused regression tests**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_base.py tests/test_tree_expert_e2_baseline.py tests/test_tree_expert_e2_training.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tree_expert/hc_base.py tests/test_tree_expert_hc_base.py
git commit -m "feat: add frozen earliest E2 source fold"
```

## Task 4: Build leakage-safe hierarchical features

**Files:**
- Create: `experiments/tree_expert/hc_features.py`
- Test: `tests/test_tree_expert_hc_features.py`

- [ ] **Step 1: Write failing shrinkage and chronology tests**

Use a synthetic four-season frame where future seasons have deliberately inverted targets. Assert that changing 2023–2024 targets cannot change a 2022 feature row.

```python
def test_future_targets_cannot_change_earlier_hierarchy(synthetic_rows):
    original = build_rolling_hierarchy(synthetic_rows, profile=BALANCED)
    changed = synthetic_rows.copy()
    changed.loc[changed.season >= 2023, "control_success"] ^= 1
    mutated = build_rolling_hierarchy(changed, profile=BALANCED)
    pd.testing.assert_frame_equal(original[2022], mutated[2022])


def test_shrinkage_uses_parent_rate():
    assert shrink(raw=0.8, parent=0.5, count=20, strength=80) == pytest.approx(0.56)
```

- [ ] **Step 2: Run and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_features.py -q`

Expected: missing `hc_features` module.

- [ ] **Step 3: Implement hierarchy table fitting**

Fit and persist levels in this order:

```python
LEVELS = (
    ("global", ()),
    ("game_type", ("game_type",)),
    ("pitcher", ("pitcher_id",)),
    ("batter", ("batter_id",)),
    ("hand_matchup", ("pitcher_hand", "batter_hand")),
    ("count", ("balls_before", "strikes_before")),
    ("outs", ("outs_before",)),
    ("base_state", ("base_state",)),
    ("pitcher_game", ("pitcher_id", "game_type")),
    ("batter_pitcher_hand", ("batter_id", "pitcher_hand")),
)
```

Each stored table includes `count`, `raw_rate`, `parent_rate`, `shrunk_rate`, `parent_level`, source seasons, and key columns. Use exact minimum counts from the contract. Sparse or unseen keys fall back to parent, then global, then zero-centered effect.

- [ ] **Step 4: Implement row-local transformation**

`transform_hierarchy(rows, frozen_state)` may read only the supplied row and frozen mappings. It produces counts/rates/effects plus known flags. It must not call `groupby`, `rolling`, `expanding`, `rank`, `value_counts`, or batch mean on evaluation rows.

- [ ] **Step 5: Add independence tests**

For one frozen state, compare singleton, original batch, shuffled batch, rebatches, and a duplicated row. Maximum numerical difference must be `<= 1e-12`.

- [ ] **Step 6: Run feature and policy tests**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_features.py tests/test_rules_policy.py tests/test_row_independence_evidence.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tree_expert/hc_features.py tests/test_tree_expert_hc_features.py
git commit -m "feat: add prior-season hierarchy features"
```

## Task 5: Train rolling C1 second-residual models

**Files:**
- Create: `experiments/tree_expert/hc_training.py`
- Test: `tests/test_tree_expert_hc_training.py`

- [ ] **Step 1: Write failing rolling-window tests**

```python
@pytest.mark.parametrize((valid_year, allowed), [
    (2022, {2021}),
    (2023, {2021, 2022}),
    (2024, {2021, 2022, 2023}),
])
def test_c1_training_uses_only_earlier_oof(valid_year, allowed, oof_rows):
    frame = build_c1_training_rows(oof_rows, valid_year=valid_year)
    assert set(frame["oof_year"]) == allowed
    assert frame["oof_year"].max() < valid_year
```

Also test exact row alignment, `residual_target == target - p0`, probability clipping, CatBoost parameters, feature-list persistence, non-finite rejection, and restart identity.

- [ ] **Step 2: Run and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_training.py -q`

Expected: missing `hc_training` module.

- [ ] **Step 3: Implement rolling C1 materialization**

Build C1 training rows from verified OOF pairs only. Join the existing E2 feature frame, `p0`, and hierarchy columns by unique `row_id` with one-to-one validation. Validation features are built from rows available through `train_end_year`; residual training evidence is limited by the OOF schedule above.

For an OOF row from season `y`, reconstruct its E2 current-row features and hierarchy
lookups with a state fit only through season `y-1`. Concatenating OOF years must preserve
those per-year frozen transforms; it must not refit one feature state across 2021–2024.
The final evaluation transform is the separate full-training state fit through 2024.

- [ ] **Step 4: Implement CatBoost residual jobs**

Use `CatBoostRegressor` and the exact registered parameters. Persist:

```text
model.cbm
predictions.csv
feature_schema.json
hierarchy_state.json
job.json
```

`predictions.csv` contains `row_id,target,p0,p1,season,game_type,pitcher_id,batter_id` and segment columns required by Task 7. `p1 = clip(p0 + residual, 1e-5, 1-1e-5)`.

- [ ] **Step 5: Add deterministic synthetic trainer injection**

The production default is CatBoost, but tests inject a tiny deterministic regressor. This validates scheduling and state without installing packages or running GPU training.

- [ ] **Step 6: Run training and E2 feature regression tests**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_training.py tests/test_tree_expert_features.py tests/test_tree_expert_e2_training.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tree_expert/hc_training.py tests/test_tree_expert_hc_training.py
git commit -m "feat: train rolling hierarchical residuals"
```

## Task 6: Add rolling C2 hierarchical calibration

**Files:**
- Create: `experiments/tree_expert/hc_calibration.py`
- Test: `tests/test_tree_expert_hc_calibration.py`

- [ ] **Step 1: Write failing source-window and fallback tests**

```python
def test_2022_calibration_uses_only_frozen_e2_2021(source_rows, c1_rows):
    state = fit_rolling_calibrator(2022, source_rows, c1_rows, BALANCED)
    assert state.source == {2021: "e2"}


def test_2024_calibration_uses_only_prior_c1_errors(source_rows, c1_rows):
    state = fit_rolling_calibrator(2024, source_rows, c1_rows, BALANCED)
    assert state.source == {2022: "c1", 2023: "c1"}
```

Also test unseen keys, sparse keys, parent fallback, effect clipping, logit stability, and identical row predictions across batch arrangements.

- [ ] **Step 2: Run and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_calibration.py -q`

Expected: missing `hc_calibration` module.

- [ ] **Step 3: Implement effect estimation**

Convert group calibration error into an additive logit effect. For source probability `p`, define a group's raw cumulative correction as

```text
raw_delta = logit(clip(mean(target), 1e-5, 1-1e-5))
            - logit(clip(mean(p), 1e-5, 1-1e-5))
shrunk_cumulative = parent_cumulative
                      + n / (n + k) * (raw_delta - parent_cumulative)
incremental_effect = shrunk_cumulative - parent_cumulative
```

The global parent is zero. Identity/context effects use global as parent. Interaction effects use the sum of their registered lower-level cumulative parents. A row adds stored incremental effects in the fixed hierarchy order, multiplies the sum by alpha, then clips the final correction to `[-0.25, 0.25]`. Persist counts, raw deltas, parent values, incremental effects, and source years.

- [ ] **Step 4: Implement fixed alpha selection inputs**

Produce C2 predictions for only `(0.25, 0.5, 0.75, 1.0)`. Alpha selection uses structure folds; ties within `1e-12` choose the smaller alpha. The 2024 confirmation fold is never used to invent another alpha.

- [ ] **Step 5: Run calibration and feature tests**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_calibration.py tests/test_tree_expert_hc_features.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/tree_expert/hc_calibration.py tests/test_tree_expert_hc_calibration.py
git commit -m "feat: add rolling hierarchical calibration"
```

## Task 7: Implement paired evidence, segment diagnostics, and fixed decisions

**Files:**
- Create: `experiments/tree_expert/hc_metrics.py`
- Create: `experiments/tree_expert/hc_decisions.py`
- Test: `tests/test_tree_expert_hc_metrics.py`
- Test: `tests/test_tree_expert_hc_decisions.py`

- [ ] **Step 1: Write failing metric tests with hand-calculated fixtures**

Cover Brier, weighted fold gain, 10-bin ECE, calibration gap, eligible segments, prediction-decile edges from training evidence, and pitcher-cluster bootstrap reproducibility.

Weighted Brier is the Brier score over all row-aligned folds, which is equivalent to a
fold Brier weighted by validation row count. Calibration gap is
`abs(mean(target) - mean(probability))`. ECE uses the fixed probability intervals
`[0.0,0.1), ... , [0.9,1.0]`. For each validation year, C0 decile boundaries are fit from
the permitted earlier OOF evidence and then applied to that validation fold; validation
probabilities never define their own boundaries.

```python
def test_pitcher_cluster_bootstrap_resamples_clusters_not_rows(frame):
    result = paired_cluster_bootstrap(frame, "p0", "p1", repeats=1000, seed=3407)
    assert result.cluster_column == "pitcher_id"
    assert result.repeats == 1000
    assert result.lower_95 == pytest.approx(EXPECTED_LOWER)
```

Use explicit numeric arrays in the fixture and store the hand-computed expected value; do not compute `EXPECTED_LOWER` by calling production code.

- [ ] **Step 2: Write failing decision tests**

Create one fixture for every individual gate failure, C2 passing while C1 fails, both candidates passing, and the `0.00002` simpler-C1 tie.

- [ ] **Step 3: Run and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_metrics.py tests/test_tree_expert_hc_decisions.py -q`

Expected: missing HC metric/decision modules.

- [ ] **Step 4: Implement evidence calculations**

Hard-gated segments require at least 5,000 validation rows. Report smaller groups with `eligible=false`. Segment families are validation year, game type, known identity, hand matchup, count, base state, and C0 decile. Store per-fold/per-seed values as well as aggregate values.

- [ ] **Step 5: Implement pure decision functions**

```python
def decide_c1(evidence: C1Evidence, contract: HCContract) -> CandidateDecision: ...
def decide_c2(evidence: C2Evidence, contract: HCContract) -> CandidateDecision: ...
def choose_winner(c1: CandidateDecision, c2: CandidateDecision) -> WinnerDecision: ...
```

Every rejected decision includes machine-readable failed gate names. Decisions never read leaderboard scores or environment variables.

- [ ] **Step 6: Run metric, decision, and existing metric regression tests**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_metrics.py tests/test_tree_expert_hc_decisions.py tests/test_tree_expert_metrics.py tests/test_tree_expert_e2_decisions.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tree_expert/hc_metrics.py experiments/tree_expert/hc_decisions.py tests/test_tree_expert_hc_metrics.py tests/test_tree_expert_hc_decisions.py
git commit -m "feat: gate hierarchical residual candidates"
```

## Task 8: Add immutable state and deterministic artifacts

**Files:**
- Create: `experiments/tree_expert/hc_state.py`
- Create: `experiments/tree_expert/hc_artifacts.py`
- Test: `tests/test_tree_expert_hc_state.py`
- Test: `tests/test_tree_expert_hc_artifacts.py`

- [ ] **Step 1: Write failing state transition tests**

Test `H1→H2→H3`, immutable complete jobs, failed-job retry, mismatched identity refusal, H3 blocking without acceptance, and independent C1/C2 rejection semantics.

- [ ] **Step 2: Write failing artifact tests**

Require byte-identical ZIP output for identical inputs, fixed timestamps, sorted members, member hashes/sizes, no symlinks/duplicate/traversal/oversized members, and one stable handoff filename.

```python
def test_model_delivery_requires_accepted_hash_bound_evidence(tmp_path, rejected_state):
    with pytest.raises(HCArtifactError, match="accepted evidence"):
        create_model_delivery(rejected_state, tmp_path / "delivery.zip")
```

- [ ] **Step 3: Run and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_state.py tests/test_tree_expert_hc_artifacts.py -q`

Expected: missing HC state/artifact modules.

- [ ] **Step 4: Implement atomic state and resume**

State records schema/campaign/bindings/stage, completed/skipped/failed/active jobs, profile/alpha/acceptance decisions, and artifact paths. Write via temporary file, `fsync`, and atomic replace. Completed job files are verified before reuse.

- [ ] **Step 5: Implement four artifact kinds**

```text
tree_hierarchical_review_v1
tree_hierarchical_resume_v1
tree_hierarchical_handoff_v1
tree_hierarchical_model_delivery_v1
```

The handoff contains one review, one resume, optional acceptance, and optional model delivery. A rejected H2 handoff is valid evidence but contains no delivery. No artifact may contain official evaluation rows or a submission package.

- [ ] **Step 6: Run artifact and repository policy tests**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_state.py tests/test_tree_expert_hc_artifacts.py tests/test_rules_repository_enforcement.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tree_expert/hc_state.py experiments/tree_expert/hc_artifacts.py tests/test_tree_expert_hc_state.py tests/test_tree_expert_hc_artifacts.py
git commit -m "feat: persist hierarchical campaign evidence"
```

## Task 9: Orchestrate H1, H2, and H3 with safe recovery

**Files:**
- Create: `experiments/tree_expert/hc_runner.py`
- Test: `tests/test_tree_expert_hc_runner.py`

- [ ] **Step 1: Write failing synthetic stage tests**

Test these complete paths:

1. fresh H1 creates source evidence, selects one profile, seals 2024 confirmation, and emits H1 review/resume;
2. resumed H2 reuses every H1 job, trains only seeds 42/2026, decides C1/C2 independently, and emits acceptance evidence;
3. H3 refuses a rejected handoff;
4. accepted H3 full-fits only the winner and emits model delivery;
5. deadline before a new job emits a valid resume rather than beginning work;
6. interrupted active job is retried without deleting complete siblings.

- [ ] **Step 2: Run and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_runner.py -q`

Expected: missing `hc_runner` module.

- [ ] **Step 3: Implement stage selection and finite scheduling**

`run_campaign()` receives injected clocks/trainers for tests and production defaults for Kaggle. H1 profile selection uses only structure folds and the registered tie order, then evaluates the selected profile on 2024 without reopening selection. H2 forms the registered arithmetic probability mean across three seeds and applies all gates. H3 trains on official train only after verifying the exact accepted review and resume hashes, using the registered median-plus-one iteration rule.

- [ ] **Step 4: Implement two-GPU dispatch**

Only independent GPU jobs run in parallel. CPU hierarchy/calibration/decision work remains in the parent process. Set `CUDA_VISIBLE_DEVICES` per worker and stream stable progress lines. A failed worker is recorded and does not mark its job complete.

- [ ] **Step 5: Implement H3 inference audits**

Before delivery, measure 245,789-row inference, enforce `<=480s`, enforce serialized state `<=2GB`, and run singleton/shuffle/rebatch/duplicate maximum difference `<=1e-12`. Tests use a small injected row count but assert the production contract keeps 245,789.

- [ ] **Step 6: Run runner and focused campaign tests**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_runner.py tests/test_tree_expert_hc_training.py tests/test_tree_expert_hc_calibration.py tests/test_tree_expert_hc_decisions.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tree_expert/hc_runner.py tests/test_tree_expert_hc_runner.py
git commit -m "feat: orchestrate hierarchical campaign stages"
```

## Task 10: Build one deterministic Kaggle cell

**Files:**
- Create: `experiments/tree_expert/hc_kaggle.py`
- Create: `tools/build_tree_hc_kaggle_cell.py`
- Generate: `experiments/tree_expert/KAGGLE_HC_CELL.py`
- Test: `tests/test_tree_expert_hc_kaggle.py`
- Test: `tests/test_tree_expert_hc_kaggle_cell.py`

- [ ] **Step 1: Write failing discovery and error-path tests**

Cover fresh H1 inputs, one previous handoff, expanded/ZIP duplicate identity, two distinct handoffs, missing official data, CPU accelerator, one or two T4 GPUs, deadline, and exception-time resume creation.

- [ ] **Step 2: Write failing cell-generation tests**

Assert deterministic bytes, no GitHub/network source fetch, embedded runtime inventory completeness, source size below 1,000,000 bytes, no Drive mount, no repeated browser download loop, and these markers:

```text
TREE_HC_CODE_READY
TREE_HC_STAGE_SELECTED
TREE_HC_JOB_START
TREE_HC_TRAINING_PROGRESS
TREE_HC_JOB_END
TREE_HC_DECISION
TREE_HC_ARTIFACT_READY
TREE_HC_SUCCESS
TREE_HC_ERROR
```

- [ ] **Step 3: Run and confirm RED**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_kaggle.py tests/test_tree_expert_hc_kaggle_cell.py -q`

Expected: missing HC Kaggle modules/cell.

- [ ] **Step 4: Implement Kaggle discovery and supervisor**

H1 discovers exactly official data plus E2 input/handoff. H2/H3 add at most one distinct prior HC handoff. The supervisor selects the next incomplete stage, enforces its wall deadline, snapshots at most every 900 seconds, and publishes only the final handoff through Kaggle output. It never initiates repeated browser downloads.

- [ ] **Step 5: Implement deterministic runtime embedding**

The build tool inventories only required modules and contract files, creates a compressed base64 runtime, validates its SHA-256 at startup, installs only missing pinned packages, and writes `KAGGLE_HC_CELL.py` atomically. Runtime extraction rejects unsafe members.

- [ ] **Step 6: Generate the cell twice and compare hashes**

Run:

```bash
/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python tools/build_tree_hc_kaggle_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_HC_CELL.py
/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python tools/build_tree_hc_kaggle_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_HC_CELL.py
wc -c experiments/tree_expert/KAGGLE_HC_CELL.py
```

Expected: both hashes match; size is below `1000000` bytes.

- [ ] **Step 7: Run Kaggle and existing cell regression tests**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_kaggle.py tests/test_tree_expert_hc_kaggle_cell.py tests/test_tree_expert_e2_kaggle_cell.py tests/test_tree_expert_t3_kaggle.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add experiments/tree_expert/hc_kaggle.py experiments/tree_expert/KAGGLE_HC_CELL.py tools/build_tree_hc_kaggle_cell.py tests/test_tree_expert_hc_kaggle.py tests/test_tree_expert_hc_kaggle_cell.py
git commit -m "feat: add hierarchical campaign Kaggle cell"
```

## Task 11: Document, audit, and hand off H1

**Files:**
- Create: `docs/tree_hierarchical_campaign.md`
- Modify only if needed for module export: `experiments/tree_expert/__init__.py`

- [ ] **Step 1: Write the operator runbook**

Document in Korean:

- what C0/C1/C2 mean;
- why 2021 source evidence is newly trained;
- exact H1/H2/H3 inputs and outputs;
- expected stage times and rerun behavior;
- success/error markers to return to Codex;
- how to interpret a rejection;
- why no submission file exists yet;
- DACON evaluation-row independence boundary.

- [ ] **Step 2: Run formatting/static scans**

Run:

```bash
rg -n "TO[D]O|TB[D]|pass$|NotImplemented|submission\.zip|files\.download" experiments/tree_expert/hc_* experiments/tree_expert/KAGGLE_HC_CELL.py tools/build_tree_hc_kaggle_cell.py docs/tree_hierarchical_campaign.md
rg -n "groupby|rolling|expanding|rank|value_counts|\.mean\(" experiments/tree_expert/hc_features.py experiments/tree_expert/hc_calibration.py
```

Expected:

- no unfinished implementation markers;
- no submission ZIP creation;
- `files.download` absent from the generated cell/runtime;
- aggregation calls, if present, occur only in clearly named training-fit functions and never in row transform/inference functions.

- [ ] **Step 3: Run the focused HC suite**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_contracts.py tests/test_tree_expert_hc_inputs.py tests/test_tree_expert_hc_base.py tests/test_tree_expert_hc_features.py tests/test_tree_expert_hc_training.py tests/test_tree_expert_hc_calibration.py tests/test_tree_expert_hc_metrics.py tests/test_tree_expert_hc_decisions.py tests/test_tree_expert_hc_state.py tests/test_tree_expert_hc_artifacts.py tests/test_tree_expert_hc_runner.py tests/test_tree_expert_hc_kaggle.py tests/test_tree_expert_hc_kaggle_cell.py -q`

Expected: all tests pass.

- [ ] **Step 4: Run existing tree-expert and rules regressions**

Run: `/path/to/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_artifacts.py tests/test_tree_expert_e2_baseline.py tests/test_tree_expert_e2_training.py tests/test_tree_expert_e2_decisions.py tests/test_tree_expert_t3_inputs.py tests/test_tree_expert_t3_decisions.py tests/test_rules_policy.py tests/test_rules_repository_enforcement.py tests/test_row_independence_evidence.py -q`

Expected: all tests pass.

- [ ] **Step 5: Inspect the final diff and dirty-worktree boundary**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only HC campaign files and the runbook are new/changed for this implementation; pre-existing unrelated TabM changes remain untouched and uncommitted.

- [ ] **Step 6: Commit documentation**

```bash
git add docs/tree_hierarchical_campaign.md experiments/tree_expert/__init__.py
git commit -m "docs: explain hierarchical campaign operation"
```

If `experiments/tree_expert/__init__.py` required no change, omit it from `git add`.

- [ ] **Step 7: Provide exactly one H1 user-run operation**

The handoff message must include:

- purpose: generate frozen 2021 source evidence and screen/confirm hierarchy profiles;
- inputs: official Kaggle dataset and accepted E2 handoff/input with SHA-256 `4dd0c901...384050f`;
- accelerator: Kaggle T4x2;
- expected duration: up to 6 hours;
- output: `tree_hierarchical_handoff.zip` from Kaggle Output;
- rerun safety: Save Version reuses hash-matching completed jobs and starts the next incomplete work;
- success text: `TREE_HC_SUCCESS stage=H1 handoff=<path>`;
- error text: the complete `TREE_HC_ERROR ...` line plus the final 80 log lines;
- return to Codex: the one handoff ZIP and final log block.

Do not provide an H2 or H3 run operation until the preceding handoff is verified.

## Final completion check

Implementation is ready for the user's H1 run only when all conditions below are true:

- the generated cell is deterministic and under Kaggle's 1 MB source limit;
- every HC synthetic/static test passes;
- selected existing tree-expert and rule-policy regression tests pass;
- source/archive/input hashes are bound in every resume and review;
- no inference path aggregates evaluation rows;
- no model delivery can be created before accepted H2 evidence;
- no submission-package code has been created;
- the user receives one complete, copyable H1 operation rather than fragmented instructions.
