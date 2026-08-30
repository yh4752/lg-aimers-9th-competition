# Direct Target Expert Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage, resumable Kaggle T4 x2 campaign that trains eight high-capacity direct-target CatBoost experts, selects and confirms them with temporal OOF evidence, and emits a delivery only for a fully accepted candidate.

**Architecture:** Add an isolated `experiments/direct_expert` package that reuses the proven tree feature and TrackMan primitives but owns its contract, expert registry, OOF selection, stacking, state, artifacts, inference audit, and generated Kaggle cells. Stage A performs two-fold structure screening and writes one handoff; Stage B performs the locked 2024 confirmation, extra seeds, full fit, and fail-closed delivery generation.

**Tech Stack:** Python 3.11/3.12, pandas 2.0.3, NumPy 1.26.4, CatBoost 1.2.10 on Kaggle T4 x2, pytest 8.4.1, deterministic ZIP/JSON artifacts.

---

## File map

Create the following focused package. Do not modify the user's currently dirty TabM files.

- `experiments/direct_expert/contract.json`: sealed folds, experts, parameters, gates, runtime and artifact identities.
- `experiments/direct_expert/contracts.py`: strict contract parsing and deterministic expert/job registries.
- `experiments/direct_expert/inputs.py`: safe ZIP/expanded-directory reconstruction and source binding.
- `experiments/direct_expert/features.py`: direct-target feature state layered over proven tree/S1/TrackMan transforms.
- `experiments/direct_expert/training.py`: CatBoost parameterization, season weights, specialist routing and fold worker.
- `experiments/direct_expert/selection.py`: Brier, segment, bootstrap, seed and stable/aggressive decisions.
- `experiments/direct_expert/stacking.py`: constrained probability/logit blending on structure folds only.
- `experiments/direct_expert/state.py`: atomic campaign phase and completed-job state.
- `experiments/direct_expert/artifacts.py`: deterministic Stage A handoff, review, resume, handoff and delivery bundles.
- `experiments/direct_expert/stage_a.py`: structure-fold data materialization and eight-expert screening.
- `experiments/direct_expert/stage_b.py`: locked confirmation, extra seeds, full fit and final decision orchestration.
- `experiments/direct_expert/full_fit.py`: accepted-token issuance, frozen state export and final seed training.
- `experiments/direct_expert/inference.py`: row-local expert routing, fixed stacking and invariance audit.
- `experiments/direct_expert/runtime_inventory.py`: exact embedded-runtime member registry and code identity.
- `experiments/direct_expert/kaggle.py`: input discovery, two-T4 verification and deterministic cell rendering.
- `experiments/direct_expert/KAGGLE_STAGE_A_CELL.py`: committed generated Stage A one-cell source.
- `experiments/direct_expert/KAGGLE_STAGE_B_CELL.py`: committed generated Stage B one-cell source.
- `tools/prepare_direct_expert_input.py`: bind S4-derived E2 OOF evidence and the accepted E2 submission model into one compact input.
- `docs/DIRECT_EXPERT_KAGGLE.md`: exact user-run instructions, logs, runtimes and returned files.
- `tests/direct_expert_fixtures.py`: deterministic tiny rows and bound S4/E2 evidence ZIPs shared by the new tests.

Create one matching test file per component under `tests/test_direct_expert_*.py`.

Use this interpreter for local tests because it has the project's pinned NumPy/pandas/PyTorch stack:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest
```

CatBoost is not installed locally. Unit tests must inject a fake estimator or model factory and must not install packages or run full official training.

### Task 1: Seal the campaign contract and expert registry

**Files:**
- Create: `experiments/direct_expert/__init__.py`
- Create: `experiments/direct_expert/contract.json`
- Create: `experiments/direct_expert/contracts.py`
- Test: `tests/test_direct_expert_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from dataclasses import replace
import pytest

from experiments.direct_expert.contracts import (
    DirectExpertContractError,
    expert_specs,
    load_contract,
    screening_jobs,
)


def test_contract_seals_deep_direct_campaign():
    contract = load_contract()
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.screening_seed == 3407
    assert contract.confirmation_seeds == (42, 2026)
    assert contract.screening_parameters["iterations"] == 1800
    assert contract.screening_parameters["depth"] == 9
    assert contract.final_parameters["iterations"] == 2400
    assert contract.final_parameters["depth"] == 10
    assert contract.maximum_deployed_models == 9


def test_exact_eight_experts_and_sixteen_stage_a_jobs():
    specs = expert_specs(load_contract())
    assert tuple(item.expert_id for item in specs) == tuple(f"D{i}" for i in range(8))
    jobs = screening_jobs(load_contract())
    assert len(jobs) == 16
    assert len({job.job_id for job in jobs}) == 16
    assert {job.fold for job in jobs} == {(2021, 2022), (2022, 2023)}


def test_changed_contract_is_rejected():
    contract = load_contract()
    with pytest.raises(DirectExpertContractError, match="final depth differs"):
        replace(contract, final_parameters={**contract.final_parameters, "depth": 8})
```

- [ ] **Step 2: Run the contract tests and verify the import failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_contracts.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'experiments.direct_expert'`.

- [ ] **Step 3: Add the sealed JSON contract**

Use these exact top-level values:

```json
{
  "schema_version": 1,
  "campaign_id": "direct_target_expert_v1",
  "folds": [[2021, 2022], [2022, 2023], [2023, 2024]],
  "screening_seed": 3407,
  "confirmation_seeds": [42, 2026],
  "experts": ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"],
  "screening_parameters": {"iterations": 1800, "depth": 9, "learning_rate": 0.03, "max_ctr_complexity": 3, "border_count": 128, "od_wait": 150},
  "final_parameters": {"iterations": 2400, "depth": 10, "learning_rate": 0.03, "max_ctr_complexity": 3, "border_count": 128, "od_wait": 150},
  "maximum_selected_experts": 4,
  "maximum_deployed_models": 9,
  "probability_tolerance": 0.000001,
  "runtime": {"stage_a_wall_seconds": 39600, "stage_b_wall_seconds": 32400, "new_job_guard_seconds": 1800, "artifact_reserve_seconds": 1200, "snapshot_interval_seconds": 600, "minimum_free_bytes": 8589934592},
  "versions": {"catboost": "1.2.10", "pandas": "2.0.3", "numpy": "1.26.4"}
}
```

Add nested stable and aggressive gate objects exactly as approved in the design: stable weighted `0.00005`, latest `0`, fold floor `-0.00003`, segment cap `0.00030`, bootstrap floor `0`, two non-worse seeds; aggressive latest `0.00015`, recent-heavy `0.00008`, fold floor `-0.00025`, segment cap `0.001`, latest bootstrap floor `-0.00005`, two improving latest seeds.

- [ ] **Step 4: Implement strict contract dataclasses and registries**

Expose these immutable interfaces:

```python
@dataclass(frozen=True)
class ExpertSpec:
    expert_id: str
    objective: str
    decay: float | None
    recent_seasons: int | None
    game_type: str | None
    interaction_profile: str


@dataclass(frozen=True)
class ExpertJob:
    job_id: str
    expert_id: str
    fold: tuple[int, int]
    seed: int
    phase: str


def expert_specs(contract: DirectExpertContract) -> tuple[ExpertSpec, ...]:
    return (
        ExpertSpec("D0", "Logloss", None, None, None, "standard"),
        ExpertSpec("D1", "Logloss", 0.75, None, None, "standard"),
        ExpertSpec("D2", "Logloss", 0.55, None, None, "standard"),
        ExpertSpec("D3", "Logloss", None, 2, None, "standard"),
        ExpertSpec("D4", "RMSE", 0.55, None, None, "standard"),
        ExpertSpec("D5", "Logloss", 0.55, None, "R", "standard"),
        ExpertSpec("D6", "Logloss", 0.55, None, "F", "standard"),
        ExpertSpec("D7", "Logloss", None, None, None, "high_ctr"),
    )
```

Reject unknown JSON keys, booleans where integers are expected, non-finite numbers, changed registered constants and duplicate jobs.

- [ ] **Step 5: Run the tests and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_contracts.py -q
```

Expected: all tests pass.

Commit:

```bash
git add experiments/direct_expert/__init__.py experiments/direct_expert/contract.json experiments/direct_expert/contracts.py tests/test_direct_expert_contracts.py
git commit -m "feat: seal direct expert campaign contract"
```

### Task 2: Build and verify the bound campaign input

**Files:**
- Create: `experiments/direct_expert/inputs.py`
- Create: `tools/prepare_direct_expert_input.py`
- Create: `tests/direct_expert_fixtures.py`
- Test: `tests/test_direct_expert_inputs.py`
- Test: `tests/test_prepare_direct_expert_input.py`

- [ ] **Step 1: Write failing tests for the available evidence sources**

```python
def test_input_contains_s4_e2_oof_and_accepted_e2_runtime(tmp_path, s4_handoff, e2_submit):
    output = prepare_direct_expert_input(s4_handoff, e2_submit, tmp_path / "input.zip")
    verified = verify_and_extract_input(output, tmp_path / "verified")
    assert verified.e2_oof_years == (2021, 2022, 2023, 2024)
    assert verified.e2_submission_sha256 == "8bc33042b9258049f905df0c733b1153d1cedfac39cb0732b2a7e154629d779a"
    assert verified.logical_e2_handoff_sha256 == "4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f"


def test_expanded_kaggle_directory_matches_zip(tmp_path, prepared_input):
    expanded = tmp_path / "expanded"
    with ZipFile(prepared_input) as archive:
        archive.extractall(expanded)
    left = verify_and_extract_input(prepared_input, tmp_path / "left")
    right = verify_and_extract_input(expanded, tmp_path / "right")
    assert left.manifest_sha256 == right.manifest_sha256


def test_tampered_e2_oof_is_rejected(tmp_path, prepared_input):
    changed = rewrite_member(prepared_input, "e2_oof/2024.csv", b"row_id,target,p_anchor\nX,0,1\n")
    with pytest.raises(DirectExpertInputError, match="member digest differs"):
        verify_and_extract_input(changed, tmp_path / "bad")
```

- [ ] **Step 2: Run the input tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_inputs.py tests/test_prepare_direct_expert_input.py -q
```

Expected: import or missing-function failures.

- [ ] **Step 3: Implement the exact input member contract**

The prepared archive must contain only:

```python
INPUT_MEMBERS = {
    "e2_oof/2021.csv",
    "e2_oof/2022.csv",
    "e2_oof/2023.csv",
    "e2_oof/2024.csv",
    "e2_submission/catboost_3seed_v1.zip",
    "evidence/s4_manifest.json",
    "evidence/e2_submission_receipt.json",
    "manifest.json",
}
```

Read `anchors/e2/{year}.csv` from the verified S4 handoff's nested resume without loading all members into memory. Verify the outer S4 SHA, outer manifest, nested resume manifest, row identity, binary target and probability range. Verify the accepted E2 archive SHA and receipt before copying it.

- [ ] **Step 4: Implement shared deterministic test fixtures**

`tests/direct_expert_fixtures.py` must expose only `make_train_rows`, `make_history_rows`,
`make_s4_handoff`, `make_e2_submission` and `rewrite_member`. Import the exact six-row
official-schema fixture and TrackMan fixture from `tests/test_tree_expert_features.py` into the
first two builders, then make detached copies so new tests cannot mutate the source fixtures. The
ZIP builders must use the production canonical JSON and ZIP member writers rather than
hand-written manifests.

The training fixture contains contiguous seasons 2020–2024, both R/F, two pitchers, two batters,
binary targets and every official column required by the reused tree/S1/TrackMan transforms. ZIP
builders use fixed timestamps and the production manifest writer. Each later test module defines
its own task-specific pytest fixtures for fold data, structure streams, bindings, gate evidence,
fake estimators and fake runtimes only after the corresponding production interface has been
introduced. Fake estimators return fixed finite probabilities and record fit parameters; fake
runtimes record phase, job, seed and GPU without importing CatBoost.

- [ ] **Step 5: Make the CLI import-safe from any working directory**

At the top of `tools/prepare_direct_expert_input.py`, resolve and insert the repository root before importing `experiments`:

```python
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

Support these exact arguments:

```text
--s4-handoff
--e2-submission
--e2-receipt
--output
```

Print `DIRECT_EXPERT_INPUT_READY path=<absolute> sha256=<digest> size_bytes=<integer>` only after reopening and verifying the output.

- [ ] **Step 6: Run tests and a fixture CLI probe, then commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_inputs.py tests/test_prepare_direct_expert_input.py -q
```

Expected: all tests pass without reading official full CSVs.

Commit:

```bash
git add experiments/direct_expert/inputs.py tools/prepare_direct_expert_input.py tests/direct_expert_fixtures.py tests/test_direct_expert_inputs.py tests/test_prepare_direct_expert_input.py
git commit -m "feat: bind direct expert evidence input"
```

### Task 3: Implement cutoff-safe direct feature states

**Files:**
- Create: `experiments/direct_expert/features.py`
- Test: `tests/test_direct_expert_features.py`

- [ ] **Step 1: Write failing cutoff, schema and order-invariance tests**

```python
def test_direct_features_are_target_blind_and_order_independent(train_rows, history_rows):
    prefix = train_rows.loc[train_rows["season"].lt(2024)]
    state, fitted = fit_direct_features(prefix, history_rows, valid_year=2024)
    valid = train_rows.loc[train_rows["season"].eq(2024)].drop(columns="control_success")
    forward = transform_direct_features(valid, state)
    reverse = transform_direct_features(valid.iloc[::-1], state)
    aligned = reverse.frame.set_index(reverse.row_id).loc[forward.row_id]
    assert tuple(forward.frame.columns) == state.feature_columns
    np.testing.assert_allclose(forward.frame.select_dtypes("number"), aligned.select_dtypes("number"))
    assert "control_success" not in forward.frame


def test_snapshot_rejects_future_fit_rows(train_rows, history_rows):
    with pytest.raises(DirectFeatureError, match="training rows reach validation season"):
        fit_direct_features(train_rows, history_rows, valid_year=2024)


def test_high_ctr_profile_adds_registered_interactions(train_rows, history_rows):
    prefix = train_rows.loc[train_rows["season"].lt(2024)]
    state, batch = fit_direct_features(prefix, history_rows, valid_year=2024)
    required = {"pitcher_batter", "pitcher_batter_count", "batter_count", "team_matchup_game_type"}
    assert required.issubset(batch.frame.columns)
    assert required.issubset(state.high_ctr_columns)
```

- [ ] **Step 2: Run the feature tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_features.py -q
```

Expected: missing-module failure.

- [ ] **Step 3: Wrap the proven tree feature state and append only registered columns**

Define:

```python
@dataclass(frozen=True)
class DirectFeatureState:
    valid_year: int
    tree_state: TreeFeatureState
    batter_trackman_state: BatterTrackmanState | None
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    high_ctr_columns: tuple[str, ...]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class DirectFeatureBatch:
    frame: pd.DataFrame
    row_id: np.ndarray
    target: np.ndarray | None
    season: np.ndarray
    game_type: np.ndarray
```

Call `fit_tree_features(train, history, valid_year=valid_year, use_trackman=True)` for the proven S1 and pitcher TrackMan features. Fit `BatterTrackmanState` on the same cutoff. Attach batter exposure only when its status is `exploratory` or `eligible`; otherwise add fixed missing indicators and do not infer values.

- [ ] **Step 4: Add deterministic current-season and interaction formulas**

Use only current-row values and stored snapshot columns. Counts use `max(current_n - snapshot_n, 0)`. Rates use smoothed counts:

```python
def smoothed_rate(count: pd.Series, total: pd.Series, prior: float, strength: float) -> pd.Series:
    return (count + strength * prior) / (total + strength)
```

Register category strings with `astype("string").fillna("__MISSING__")` and pairwise `left.str.cat(right, sep="|")`. Convert all non-categorical columns to finite `float32`. Reject column collisions, target presence at inference, changed column order and non-unique row IDs.

- [ ] **Step 5: Run feature and regression tests, then commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_features.py tests/test_tree_expert_features.py -q
```

Expected: all new and existing tree feature tests pass.

Commit:

```bash
git add experiments/direct_expert/features.py tests/test_direct_expert_features.py
git commit -m "feat: build cutoff safe direct expert features"
```

### Task 4: Add season weighting, specialist routing and fold training

**Files:**
- Create: `experiments/direct_expert/training.py`
- Test: `tests/test_direct_expert_training.py`

- [ ] **Step 1: Write failing tests for every expert behavior**

```python
@pytest.mark.parametrize(
    ("expert_id", "expected"),
    [
        ("D0", [1.0, 1.0, 1.0]),
        ("D1", [0.75**2, 0.75, 1.0]),
        ("D2", [0.55**2, 0.55, 1.0]),
        ("D3", [0.0, 1.0, 1.0]),
    ],
)
def test_registered_season_weights(expert_id, expected):
    seasons = np.asarray([2021, 2022, 2023])
    np.testing.assert_allclose(season_weights(expert_spec(expert_id), seasons, 2024), expected)


def test_r_and_f_specialists_train_only_their_rows(batch):
    assert training_mask(expert_spec("D5"), batch).tolist() == [True, False, True, False]
    assert training_mask(expert_spec("D6"), batch).tolist() == [False, True, False, True]


def test_classifier_and_regressor_emit_probabilities(fake_factory, fold_data, tmp_path):
    left = run_fold_job(job("D0"), fold_data, tmp_path / "d0", model_factory=fake_factory)
    right = run_fold_job(job("D4"), fold_data, tmp_path / "d4", model_factory=fake_factory)
    assert left.status == right.status == "completed"
    assert left.predictions["probability"].between(0, 1).all()
    assert right.predictions["probability"].between(0, 1).all()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_training.py -q
```

Expected: missing-function failures.

- [ ] **Step 3: Implement exact CatBoost parameter construction**

For Logloss experts use `CatBoostClassifier(loss_function="Logloss", eval_metric="BrierScore")`. For D4 use `CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE")` and clip its raw validation output to `[1e-6, 1 - 1e-6]` before recording probabilities or Brier. Add seed, GPU device, categorical indices, `allow_writing_files=True`, `save_snapshot=True`, `snapshot_interval=600`, and the sealed structural parameters. Never catch OOM to change depth or iteration count.

- [ ] **Step 4: Implement atomic fold outputs and reuse identity**

Each completed job writes:

```text
predictions.csv
metrics.json
job_identity.json
model.cbm
worker.log
```

`predictions.csv` columns are exactly `row_id,target,probability,game_type,pitcher_id,oof_year`. The identity binds contract, code, train, history, input manifest, expert, fold and seed hashes. A reusable job must have all files and an exact identity match.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_training.py -q
```

Expected: all tests pass with fake estimators and no local CatBoost import at module import time.

Commit:

```bash
git add experiments/direct_expert/training.py tests/test_direct_expert_training.py
git commit -m "feat: train direct target experts"
```

### Task 5: Implement evidence, selection and the two acceptance gates

**Files:**
- Create: `experiments/direct_expert/selection.py`
- Test: `tests/test_direct_expert_selection.py`

- [ ] **Step 1: Write boundary tests for stable and aggressive decisions**

```python
def test_stable_gate_requires_every_boundary():
    evidence = accepted_stable_evidence()
    assert decide(evidence).status == "accepted_stable"
    assert "weighted_gain" in decide(replace(evidence, weighted_gain=0.000049999)).failed_gates
    assert "minimum_fold_gain" in decide(replace(evidence, minimum_fold_gain=-0.000030001)).failed_gates
    assert "bootstrap_lower" in decide(replace(evidence, bootstrap_lower=-1e-12)).failed_gates


def test_aggressive_gate_does_not_accept_s4_c00():
    evidence = accepted_aggressive_evidence()
    s4_like = replace(evidence, recent_heavy_gain=0.0000473396)
    result = decide(s4_like)
    assert result.status == "rejected"
    assert result.failed_gates == ("recent_heavy_gain",)


def test_structure_selection_never_reads_2024():
    selected = select_structure_experts(structure_frames(), confirmation_frames=ExplodingMapping())
    assert len(selected) == 4
    assert selected.locked_on_years == (2022, 2023)


def test_specialist_selection_is_dependency_closed():
    selected = select_structure_experts(structure_frames(prefer=("D5", "D6")))
    assert len(selected.expert_ids) == 4
    assert {"D0", "D5", "D6"}.issubset(selected.expert_ids)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_selection.py -q
```

Expected: missing-module failure.

- [ ] **Step 3: Implement aligned Brier and segment evidence**

Join candidates to E2 by exact `row_id` and reject missing, duplicate or reordered identities. Compute row gain as `(target - p_e2)**2 - (target - p_candidate)**2`. Use fold weights `(0.60, 0.75, 0.90)` for stable and `(0.15, 0.25, 0.60)` for recent-heavy evidence. Compute R/F segment regression only for segments with at least 5,000 rows.

- [ ] **Step 4: Implement deterministic pitcher-cluster bootstrap and seed counts**

Use seed 3407 and 1,000 resamples of unique pitcher IDs. Record both combined-fold and 2024-only 95% bounds. Count a seed as stable non-worse only when its weighted gain is nonnegative; count aggressive improvement only when its 2024 gain is positive.

- [ ] **Step 5: Implement four-role structure selection and fail-closed decisions**

Select `best_weighted`, `best_latest_structure`, `most_diverse`, and `wildcard`. Resolve ties by expert ID. D4 or an unrepresented R/F/high-CTR expert supplies the wildcard only if it has no catastrophic structure-fold regression below `-0.001`; otherwise choose the next diverse eligible expert. Return exactly four unique experts. If D5 or D6 is selected, D0 must occupy one of those four slots; replace the lowest-priority non-mandatory role winner deterministically until the set is dependency-closed. Build D5/D6 OOF streams by using the specialist prediction only on its registered current-row `game_type` and the aligned D0 prediction elsewhere.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_selection.py -q
```

Expected: all tests pass.

Commit:

```bash
git add experiments/direct_expert/selection.py tests/test_direct_expert_selection.py
git commit -m "feat: gate direct expert evidence"
```

### Task 6: Add constrained probability and logit stacking

**Files:**
- Create: `experiments/direct_expert/stacking.py`
- Test: `tests/test_direct_expert_stacking.py`

- [ ] **Step 1: Write failing causality and weight-constraint tests**

```python
def test_probability_stack_is_nonnegative_sparse_and_normalized():
    recipe = fit_stack(structure_streams(), method="probability")
    assert len(recipe.weights) <= 3
    assert all(weight >= 0.05 for weight in recipe.weights.values())
    assert sum(recipe.weights.values()) == pytest.approx(1.0)


def test_logit_stack_clips_before_transform():
    recipe = StackRecipe("logit", {"D0": 0.5, "D2": 0.5}, (2022, 2023))
    result = apply_stack(recipe, {"D0": np.array([0.0]), "D2": np.array([1.0])})
    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(0.5)


def test_2024_stream_is_never_opened_while_fitting():
    recipe = fit_stack(structure_streams(), method="probability", confirmation=ExplodingMapping())
    assert recipe.selection_years == (2022, 2023)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_stacking.py -q
```

Expected: missing-module failure.

- [ ] **Step 3: Implement deterministic sparse simplex search with the deployment budget**

Enumerate active expert sets of sizes one through three from the four locked candidates. Start weights on a `0.05` grid, retain the best Brier recipe, then run deterministic coordinate refinements at `0.02`, `0.01` and `0.005`. Reject negative weights, active weights below `0.05`, sums outside `1e-12` of one, changed row alignment and unregistered methods. `required_experts("D5")` and `required_experts("D6")` return the specialist plus D0; every direct recipe must have at most three unique required experts.

- [ ] **Step 4: Add the fixed E2 safety blend search**

Evaluate direct-champion weights `(0.60, 0.70, 0.80, 0.90)` against E2 on structure folds only in probability and logit space. An E2 recipe may require at most two unique direct experts because the bound E2 runtime contains three seed models. Store the single locked recipe and its structure-fold digest before 2024 evaluation.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_stacking.py -q
```

Expected: all tests pass.

Commit:

```bash
git add experiments/direct_expert/stacking.py tests/test_direct_expert_stacking.py
git commit -m "feat: stack direct expert predictions"
```

### Task 7: Add atomic state and deterministic artifacts

**Files:**
- Create: `experiments/direct_expert/state.py`
- Create: `experiments/direct_expert/artifacts.py`
- Test: `tests/test_direct_expert_state.py`
- Test: `tests/test_direct_expert_artifacts.py`

- [ ] **Step 1: Write failing state and artifact tests**

```python
def test_phase_order_and_retry_state_round_trip(tmp_path):
    state = initial_state("stage_a")
    state = complete_job(state, "screen__D0__2021_2022__s3407")
    state = fail_job(state, "screen__D1__2021_2022__s3407", "oom")
    state = retry_job(state, "screen__D1__2021_2022__s3407")
    save_state(tmp_path / "state.json", state)
    assert load_state(tmp_path / "state.json") == state


def test_rejected_campaign_has_no_delivery(tmp_path, bindings):
    paths = write_stage_b_bundles(rejected_campaign(), tmp_path, bindings)
    assert paths.review.is_file()
    assert paths.handoff.is_file()
    assert paths.delivery is None


def test_changed_binding_rejects_resume(tmp_path, resume, bindings):
    with pytest.raises(DirectExpertArtifactError, match="artifact bindings differ"):
        verify_resume(resume, replace(bindings, contract_sha256="f" * 64))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_state.py tests/test_direct_expert_artifacts.py -q
```

Expected: missing-module failures.

- [ ] **Step 3: Implement exact phases and atomic state writes**

Stage A phases are `screening`, `selection`, `completed`. Stage B phases are `confirmation`, `extra_seeds`, `stacking`, `decision`, `full_fit`, `audit`, `completed`. Completed and failed job sets must be disjoint. Save canonical JSON to a sibling temporary file, `fsync`, then `os.replace` it.

- [ ] **Step 4: Implement deterministic ZIP writers and verifiers**

Use fixed timestamps, sorted POSIX member names, stored manifest member size and SHA-256, no symlinks, no encryption, no absolute or `..` paths, and bounded decompressed size. Stream file members; do not call `Path.read_bytes()` on model or prediction files.

Stage A handoff includes state, selection evidence, locked expert IDs, required OOF predictions, logs and bindings. Stage B review includes all decision evidence but no models. Stage B handoff includes resumable completed jobs. Delivery includes only accepted model/state/recipe/runtime/audit files and never a submission archive.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_state.py tests/test_direct_expert_artifacts.py -q
```

Expected: all tests pass.

Commit:

```bash
git add experiments/direct_expert/state.py experiments/direct_expert/artifacts.py tests/test_direct_expert_state.py tests/test_direct_expert_artifacts.py
git commit -m "feat: persist direct expert campaign state"
```

### Task 8: Implement Stage A production and resumable two-GPU runner

**Files:**
- Create: `experiments/direct_expert/stage_a.py`
- Test: `tests/test_direct_expert_stage_a.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_stage_a_runs_sixteen_unique_jobs_on_two_workers(tmp_path, fake_runtime):
    result = run_stage_a(fake_runtime, tmp_path, absolute_deadline=10_000)
    assert result.status == "completed"
    assert len(fake_runtime.started) == 16
    assert len(set(fake_runtime.started)) == 16
    assert set(fake_runtime.gpus) == {0, 1}


def test_one_failed_candidate_does_not_stop_siblings(tmp_path, fake_runtime):
    fake_runtime.fail("screen__D7__2021_2022__s3407")
    result = run_stage_a(fake_runtime, tmp_path, absolute_deadline=10_000)
    assert result.status == "completed_with_candidate_failure"
    assert len(result.completed_jobs) == 15


def test_deadline_stops_new_jobs_and_publishes_handoff(tmp_path, fake_runtime):
    result = run_stage_a(fake_runtime, tmp_path, absolute_deadline=1)
    assert result.status == "incomplete"
    assert result.handoff.is_file()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_stage_a.py -q
```

Expected: missing-module failure.

- [ ] **Step 3: Materialize each fold once and schedule jobs deterministically**

Cache one `DirectFeatureBatch` per structure fold. Jobs are sorted by fold, then expert ID. Start at most two concurrent worker processes and assign GPU IDs `0` and `1` explicitly. A worker returns only its result path and terminal status; the parent verifies identity before marking completion.

- [ ] **Step 4: Lock selection before any confirmation data is available**

After eligible screening jobs finish, load only 2022/2023 predictions, compare to the bound E2 OOF streams, write `selection/locked_selection.json`, and store its SHA in state. Do not accept a confirmation path or frame in the Stage A API.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_stage_a.py -q
```

Expected: all tests pass.

Commit:

```bash
git add experiments/direct_expert/stage_a.py tests/test_direct_expert_stage_a.py
git commit -m "feat: run direct expert stage A"
```

### Task 9: Generate and verify the Stage A Kaggle cell

**Files:**
- Create: `experiments/direct_expert/runtime_inventory.py`
- Create: `experiments/direct_expert/kaggle.py`
- Create: `experiments/direct_expert/KAGGLE_STAGE_A_CELL.py`
- Test: `tests/test_direct_expert_kaggle.py`
- Test: `tests/test_direct_expert_kaggle_cells.py`

- [ ] **Step 1: Write failing discovery, GPU and renderer tests**

```python
def test_discovery_accepts_zip_or_expanded_input_and_rejects_two_logical_inputs(tmp_path):
    found = discover_inputs(single_expanded_fixture(tmp_path), stage="A")
    assert found.official_data.is_dir()
    assert found.campaign_input.exists()
    with pytest.raises(DirectExpertKaggleError, match="campaign input count must be one"):
        discover_inputs(duplicate_input_fixture(tmp_path), stage="A")


def test_gpu_contract_requires_exactly_two_t4_devices():
    assert verify_t4x2(FakeTorch(["Tesla T4", "Tesla T4"])) == ("Tesla T4", "Tesla T4")
    with pytest.raises(DirectExpertKaggleError, match="two Tesla T4"):
        verify_t4x2(FakeTorch(["Tesla T4"]))


def test_stage_a_cell_is_deterministic_small_and_has_no_submission(tmp_path):
    left = build_kaggle_cell("A", tmp_path / "left.py")
    right = build_kaggle_cell("A", tmp_path / "right.py")
    assert left.read_bytes() == right.read_bytes()
    assert left.stat().st_size < 1_000_000
    assert b"submission.zip" not in left.read_bytes()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_kaggle.py tests/test_direct_expert_kaggle_cells.py -q
```

Expected: missing-module failures.

- [ ] **Step 3: Implement an exact runtime inventory before rendering**

`runtime_inventory.py` owns a sorted tuple of every embedded source, JSON contract and required shared tree/temporal module. `code_identity_sha256(root)` must hash the relative name and bytes of every member. The renderer and extracted runtime import the same inventory module so the earlier missing-`runtime_inventory.py` failure cannot recur.

- [ ] **Step 4: Render the one-cell Stage A source**

The generated cell must:

1. print `DIRECT_EXPERT_CODE_READY` after archive hash verification;
2. install only `catboost==1.2.10` when the exact version is absent;
3. discover and verify official data and the single campaign input before training;
4. verify two T4 GPUs;
5. run a 20-iteration, 10,000-row smoke fit through both devices;
6. call Stage A with an 11-hour deadline and artifact reserve;
7. print `DIRECT_EXPERT_HANDOFF_READY path=<absolute_handoff_path>` for one final handoff;
8. never call browser download APIs.

- [ ] **Step 5: Generate the committed cell and run parity tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m experiments.direct_expert.kaggle --stage A --output experiments/direct_expert/KAGGLE_STAGE_A_CELL.py
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_kaggle.py tests/test_direct_expert_kaggle_cells.py -q
```

Expected: committed-cell parity and all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/direct_expert/runtime_inventory.py experiments/direct_expert/kaggle.py experiments/direct_expert/KAGGLE_STAGE_A_CELL.py tests/test_direct_expert_kaggle.py tests/test_direct_expert_kaggle_cells.py
git commit -m "feat: generate direct expert stage A cell"
```

### Task 10: Implement accepted-token full fit and frozen inference

**Files:**
- Create: `experiments/direct_expert/full_fit.py`
- Create: `experiments/direct_expert/inference.py`
- Test: `tests/test_direct_expert_full_fit.py`
- Test: `tests/test_direct_expert_inference.py`

- [ ] **Step 1: Write failing full-fit and prediction tests**

```python
def test_only_accepted_decision_can_issue_full_fit_token(bindings):
    with pytest.raises(DirectExpertFullFitError, match="candidate is not accepted"):
        accepted_token(rejected_decision(), locked_recipe(), bindings)


def test_full_fit_iterations_use_median_and_clip():
    assert full_fit_iterations([1200, 1500, 2100], maximum=2400) == 1500
    assert full_fit_iterations([2300, 2400, 2400], maximum=2400) == 2400


def test_predictor_routes_only_by_current_row_game_type(frozen_runtime, rows):
    predicted = frozen_runtime.predict(rows)
    changed = rows.copy()
    changed.loc[1:, "game_type"] = changed.loc[1:, "game_type"].map({"R": "F", "F": "R"})
    repeated = frozen_runtime.predict(changed)
    assert predicted[0] == pytest.approx(repeated[0])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_full_fit.py tests/test_direct_expert_inference.py -q
```

Expected: missing-module failures.

- [ ] **Step 3: Implement fail-closed tokens and frozen feature export**

The token binds accepted gate kind, candidate ID, recipe, selected experts, seeds, per-expert iterations, contract/code/data/input hashes and model count. Export feature state frames with their dtypes and SHA-256. Reject any final model not named by the token.

- [ ] **Step 4: Implement row-local fixed-recipe inference**

Load CatBoost lazily, transform rows once, select standard or high-CTR columns per expert, route D5/D6 using the current row's `game_type`, average the registered seeds, then apply the frozen probability or logit recipe. Clip only to `[1e-6, 1 - 1e-6]`. Reject target presence, unknown game type, changed feature schema and non-finite output.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_full_fit.py tests/test_direct_expert_inference.py -q
```

Expected: all tests pass using fake models.

Commit:

```bash
git add experiments/direct_expert/full_fit.py experiments/direct_expert/inference.py tests/test_direct_expert_full_fit.py tests/test_direct_expert_inference.py
git commit -m "feat: freeze accepted direct expert inference"
```

### Task 11: Add inference invariance and source-compliance audits

**Files:**
- Modify: `experiments/direct_expert/inference.py`
- Create: `experiments/direct_expert/compliance.py`
- Test: `tests/test_direct_expert_compliance.py`

- [ ] **Step 1: Write failing invariance tests**

```python
def test_all_row_independence_modes_match(frozen_runtime, audit_rows):
    report = audit_inference(frozen_runtime, audit_rows, batch_sizes=(1, 7, 64))
    assert report.singleton_max_abs <= 1e-6
    assert report.reverse_max_abs <= 1e-6
    assert report.shuffle_max_abs <= 1e-6
    assert report.rebatch_max_abs <= 1e-6
    assert report.companion_max_abs <= 1e-6
    assert report.same_feature_audit_row_max_abs <= 1e-6


def test_submission_source_rejects_cross_row_operations():
    with pytest.raises(DirectExpertComplianceError, match="forbidden evaluation operation"):
        audit_source("def predict(test):\n    return test.groupby('pitcher_id').size()\n")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_compliance.py -q
```

Expected: missing-module failure.

- [ ] **Step 3: Implement dynamic invariance audits**

Align by audit row identity, not input position. For the same-feature test, copy a row, assign a distinct audit-only row ID, and compare probabilities after removing identity from model features. Record maximum absolute differences and tested row counts.

- [ ] **Step 4: Implement AST-based inference-source checks**

Inspect only the exported inference path. Reject evaluation-frame uses of `groupby`, `rolling`, `expanding`, `rank`, `value_counts`, `shift`, `diff`, global `mean` and `transform`; reject network modules and external-process calls. Permit training orchestration workers because they are absent from delivery inference source.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_inference.py tests/test_direct_expert_compliance.py -q
```

Expected: all tests pass.

Commit:

```bash
git add experiments/direct_expert/inference.py experiments/direct_expert/compliance.py tests/test_direct_expert_compliance.py
git commit -m "feat: audit direct expert row independence"
```

### Task 12: Implement Stage B confirmation, stacking, decision and delivery

**Files:**
- Create: `experiments/direct_expert/stage_b.py`
- Test: `tests/test_direct_expert_stage_b.py`

- [ ] **Step 1: Write failing phase and package-blocking tests**

```python
def test_stage_b_runs_locked_confirmation_before_extra_seeds(tmp_path, fake_runtime):
    result = run_stage_b(fake_runtime, stage_a_handoff(), tmp_path, absolute_deadline=20_000)
    phases = [event.phase for event in fake_runtime.events]
    assert phases.index("confirmation") < phases.index("extra_seeds")
    assert phases.index("extra_seeds") < phases.index("stacking")
    assert phases.index("stacking") < phases.index("decision")


def test_rejected_result_creates_review_and_handoff_only(tmp_path, rejecting_runtime):
    result = run_stage_b(rejecting_runtime, stage_a_handoff(), tmp_path, absolute_deadline=20_000)
    assert result.review.is_file()
    assert result.handoff.is_file()
    assert result.delivery is None


def test_accepted_result_full_fits_at_most_nine_models(tmp_path, accepting_runtime):
    result = run_stage_b(accepting_runtime, stage_a_handoff(), tmp_path, absolute_deadline=20_000)
    assert result.delivery.is_file()
    assert len(accepting_runtime.full_fit_jobs) <= 9
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_stage_b.py -q
```

Expected: missing-module failure.

- [ ] **Step 3: Implement the exact maximum-44 OOF schedule**

Reuse the 16 Stage A jobs. Run seed 3407 on 2024 for four locked experts, then seeds 42 and 2026 on all three folds for those experts. Verify that the total OOF job registry cannot exceed 44 and that Stage B cannot add another expert after reading 2024.

- [ ] **Step 4: Evaluate single experts, direct stacks and E2 safety blend**

Create evidence for every confirmed expert and both stacking spaces. Apply stable and aggressive gates without fallback. If no candidate is accepted, stop before full fit. If candidates pass, prefer the best accepted direct champion and retain one accepted E2 safety blend only when its sequential gain and all applicable gates pass.

- [ ] **Step 5: Full fit, audit and write bundles**

Train only token-authorized expert/seed pairs. A direct champion may use at most three expert structures × three seeds. An E2 safety blend may use at most two direct structures × three seeds plus the three bound E2 seed models. Load the frozen inference runtime, run all dynamic and source audits, then call the artifact writer. Any model, audit, version or binding mismatch changes the candidate to rejected and prevents delivery creation.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_stage_b.py -q
```

Expected: all tests pass.

Commit:

```bash
git add experiments/direct_expert/stage_b.py tests/test_direct_expert_stage_b.py
git commit -m "feat: confirm and deliver direct experts"
```

### Task 13: Generate the Stage B cell and write the user runbook

**Files:**
- Modify: `experiments/direct_expert/kaggle.py`
- Create: `experiments/direct_expert/KAGGLE_STAGE_B_CELL.py`
- Create: `docs/DIRECT_EXPERT_KAGGLE.md`
- Modify: `tests/test_direct_expert_kaggle.py`
- Modify: `tests/test_direct_expert_kaggle_cells.py`
- Create: `tests/test_direct_expert_runbook.py`

- [ ] **Step 1: Write failing Stage B renderer and runbook tests**

```python
def test_stage_b_discovers_exactly_one_stage_a_handoff(tmp_path):
    found = discover_inputs(stage_b_fixture(tmp_path), stage="B")
    assert found.stage_a_handoff.exists()
    assert found.previous_stage_b_handoff is None


def test_stage_b_cell_prints_all_terminal_artifacts(tmp_path):
    cell = build_kaggle_cell("B", tmp_path / "cell.py").read_text()
    assert "DIRECT_EXPERT_REVIEW_READY" in cell
    assert "DIRECT_EXPERT_HANDOFF_READY" in cell
    assert "DIRECT_EXPERT_DELIVERY_READY" in cell
    assert "submission.zip" not in cell


def test_runbook_names_inputs_time_and_return_files():
    text = Path("docs/DIRECT_EXPERT_KAGGLE.md").read_text()
    for required in ("10~11시간", "8~9시간", "T4 x2", "direct_expert_stage_A_handoff.zip", "direct_expert_review.zip"):
        assert required in text
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_kaggle.py tests/test_direct_expert_kaggle_cells.py tests/test_direct_expert_runbook.py -q
```

Expected: Stage B and runbook assertions fail.

- [ ] **Step 3: Extend discovery and render Stage B**

Stage B accepts official data, one Stage A handoff, and zero or one matching Stage B handoff. Deduplicate an expanded handoff and its nested resume as one logical source; reject distinct duplicates. Run with a 9-hour deadline and print review/handoff paths unconditionally, delivery only when present. Do not auto-download.

- [ ] **Step 4: Write the complete Korean runbook**

Document for each Version:

- purpose;
- exact Kaggle inputs;
- accelerator and Internet setting;
- expected runtime;
- one-cell execution method;
- safe rerun/resume behavior;
- success markers;
- error text to return;
- exact ZIP files to download and send back.

Explicitly state that the input preparation tool is local and light, full training is user-run, and no submission package exists yet.

- [ ] **Step 5: Generate both committed cells and run tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m experiments.direct_expert.kaggle --stage A --output experiments/direct_expert/KAGGLE_STAGE_A_CELL.py
artifacts/tabm_submission_python311/bin/python -m experiments.direct_expert.kaggle --stage B --output experiments/direct_expert/KAGGLE_STAGE_B_CELL.py
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_kaggle.py tests/test_direct_expert_kaggle_cells.py tests/test_direct_expert_runbook.py -q
```

Expected: all tests and generated-cell parity checks pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/direct_expert/kaggle.py experiments/direct_expert/KAGGLE_STAGE_A_CELL.py experiments/direct_expert/KAGGLE_STAGE_B_CELL.py docs/DIRECT_EXPERT_KAGGLE.md tests/test_direct_expert_kaggle.py tests/test_direct_expert_kaggle_cells.py tests/test_direct_expert_runbook.py
git commit -m "docs: hand off direct expert Kaggle campaign"
```

### Task 14: Run integration, regression and packaging-blocker verification

**Files:**
- Create: `tests/test_direct_expert_integration.py`
- Modify only if needed: files created in Tasks 1–13

- [ ] **Step 1: Add a fixture-only end-to-end integration test**

```python
def test_two_stage_fixture_campaign_is_deterministic_and_fail_closed(tmp_path):
    first = run_fixture_campaign(tmp_path / "first")
    second = run_fixture_campaign(tmp_path / "second")
    assert first.stage_a_sha256 == second.stage_a_sha256
    assert first.review_sha256 == second.review_sha256
    assert first.decision.status == "rejected"
    assert first.delivery is None
    assert not tuple(tmp_path.rglob("submission.zip"))
```

- [ ] **Step 2: Run all direct-expert tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_direct_expert_*.py -q
```

Expected: all direct-expert tests pass.

- [ ] **Step 3: Run tree-feature, TrackMan and S4 regression tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_features.py \
  tests/test_tree_expert_s4_inputs.py \
  tests/test_tree_expert_s4_inference.py \
  tests/test_temporal_portfolio_trackman_pitcher.py \
  tests/test_temporal_portfolio_trackman_batter.py -q
```

Expected: all selected regression tests pass.

- [ ] **Step 4: Run static and artifact checks**

Run:

```bash
git diff --check
artifacts/tabm_submission_python311/bin/python -m compileall -q experiments/direct_expert tools/prepare_direct_expert_input.py
rg -n "submission\.zip|files\.download|drive\.mount" experiments/direct_expert tools/prepare_direct_expert_input.py
```

Expected: `git diff --check` and compileall succeed. The search may show assertions or documentation, but no code path creates `submission.zip`, calls a browser download API, or mounts Drive.

- [ ] **Step 5: Run the repository test suite and classify only pre-existing failures**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q
```

Expected: all direct-expert tests pass. If unrelated failures remain, reproduce them at the pre-implementation commit before claiming they are pre-existing; do not edit the user's dirty TabM files to hide them.

- [ ] **Step 6: Verify the exact staged diff and commit**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Stage only the direct-expert integration test and any direct-expert corrections from this task.

Commit:

```bash
git add tests/test_direct_expert_integration.py experiments/direct_expert tools/prepare_direct_expert_input.py docs/DIRECT_EXPERT_KAGGLE.md
git commit -m "test: verify direct expert campaign end to end"
```

Do not push, prepare the heavy input archive, run official full-data training, or create a submission ZIP without a separate explicit user request and the required accepted delivery evidence.
