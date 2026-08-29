# S4 Aggressive Anchor·Residual·Hierarchical Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one resumable Kaggle T4×2 campaign that compares seasonal anchors, residual experts, and hierarchical calibration as at least twelve complete pipelines and returns one hash-bound review handoff.

**Architecture:** Add an S4-specific set of focused modules under `experiments/tree_expert` while reusing the verified official-data, E2, hierarchy, metric, and artifact primitives. A strict JSON contract generates anchor and residual jobs; a two-worker runner advances anchor → residual → calibration → confirmation phases and persists atomic state after every job. The Kaggle renderer embeds the exact runtime into one source cell and never creates a DACON submission package.

**Tech Stack:** Python 3.11/3.12, pandas, NumPy, CatBoost 1.2.10, XGBoost 3.0.2, LightGBM 4.6.0, scikit-learn, pytest, deterministic ZIP/JSON artifacts, Kaggle T4×2

---

## File map

New runtime files:

- `experiments/tree_expert/s4_contract.json`: sealed grids, folds, capacities, gates, runtime budget.
- `experiments/tree_expert/s4_contracts.py`: strict contract parser and deterministic job registry.
- `experiments/tree_expert/s4_inputs.py`: exact E2-bound input preparation and verification.
- `experiments/tree_expert/s4_temporal.py`: recent/multi-season weighting and anchor formulas.
- `experiments/tree_expert/s4_features.py`: cutoff-fitted current-row context state.
- `experiments/tree_expert/s4_training.py`: anchor classifiers and residual regressors.
- `experiments/tree_expert/s4_calibration.py`: train-only hierarchical logit calibration.
- `experiments/tree_expert/s4_decisions.py`: coverage roles, full-chain archetypes, confirmation and acceptance.
- `experiments/tree_expert/s4_state.py`: immutable campaign state and atomic serialization.
- `experiments/tree_expert/s4_artifacts.py`: resume, review, optional model delivery, and one outer handoff.
- `experiments/tree_expert/s4_full_fit.py`: accepted-only full fitting token and frozen model state.
- `experiments/tree_expert/s4_inference.py`: current-row-only frozen predictor and independence audit.
- `experiments/tree_expert/s4_runner.py`: phase orchestration and two-worker scheduling.
- `experiments/tree_expert/s4_kaggle.py`: input discovery, embedded runtime, and cell renderer.
- `experiments/tree_expert/KAGGLE_S4_CELL.py`: committed one-cell Kaggle entry point.
- `tools/prepare_tree_s4_input.py`: local deterministic S4 input builder.
- `docs/TREE_S4_KAGGLE.md`: user runbook and exact return contract.

New tests mirror each runtime boundary under `tests/test_tree_expert_s4_*.py`. Existing TabM files and the untracked recovery notebook are outside this plan.

### Task 1: Seal the S4 contract and job registry

**Files:**
- Create: `experiments/tree_expert/s4_contract.json`
- Create: `experiments/tree_expert/s4_contracts.py`
- Create: `tests/test_tree_expert_s4_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
from dataclasses import replace

import pytest

from experiments.tree_expert.s4_contracts import (
    S4ContractError,
    anchor_specs,
    load_s4_contract,
    residual_specs,
)


def test_contract_seals_aggressive_search_and_runtime() -> None:
    contract = load_s4_contract()
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.recent_weights == (0.60, 0.75, 0.90)
    assert contract.decays == (0.30, 0.55, 0.75)
    assert contract.residual_alphas == (0.25, 0.50, 0.75, 1.00)
    assert contract.calibration_betas == (0.10, 0.25, 0.50, 0.75)
    assert contract.minimum_full_chains == 12
    assert contract.runtime.wall_seconds == 43200
    assert contract.runtime.artifact_reserve_seconds >= 2700


def test_anchor_and_residual_registries_are_unique_and_cover_external_template() -> None:
    contract = load_s4_contract()
    anchors = anchor_specs(contract)
    residuals = residual_specs(contract)
    assert len({item.candidate_id for item in anchors}) == len(anchors)
    assert any(item.recent_weight == 0.75 and item.decay == 0.55 for item in anchors)
    assert {item.family for item in residuals} == {
        "catboost", "catboost_rf", "xgboost", "lightgbm", "dual_temporal"
    }


def test_changed_contract_is_rejected(tmp_path) -> None:
    contract = load_s4_contract()
    bad = replace(contract, minimum_full_chains=11)
    with pytest.raises(S4ContractError, match="minimum_full_chains"):
        anchor_specs(bad)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_contracts.py -q`

Expected: collection fails because `experiments.tree_expert.s4_contracts` does not exist.

- [ ] **Step 3: Add the sealed JSON and strict parser**

The JSON must contain exact keys for inputs, folds, grids, five residual families, full model capacities, gates, and runtime. Implement these public immutable types and registries:

```python
@dataclass(frozen=True)
class AnchorSpec:
    candidate_id: str
    recent_weight: float | None
    decay: float | None
    route_by_game_type: bool
    mandatory_role: str | None


@dataclass(frozen=True)
class ResidualSpec:
    family: str
    route_by_game_type: bool


def anchor_specs(contract: S4Contract) -> tuple[AnchorSpec, ...]:
    base = [AnchorSpec("s4__anchor__e2", None, None, False, "e2_control")]
    for weight in contract.recent_weights:
        for decay in contract.decays:
            role = "external_template" if (weight, decay) == (0.75, 0.55) else None
            token = f"w{int(weight * 100):02d}__d{int(decay * 100):02d}"
            base.append(AnchorSpec(f"s4__anchor__{token}", weight, decay, False, role))
    if len(base) != 10:
        raise S4ContractError("anchor registry must contain ten base anchors")
    return tuple(base)
```

Reject booleans as numbers, non-finite values, extra keys, wrong tuple order, duplicate IDs, a missing external template, and any contract whose minimum full chains is below 12.

- [ ] **Step 4: Run contract tests and the repository contract**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_contracts.py tests/test_experiment_contract_gate.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add experiments/tree_expert/s4_contract.json experiments/tree_expert/s4_contracts.py tests/test_tree_expert_s4_contracts.py
git commit -m "feat: seal S4 campaign contract"
```

### Task 2: Build and verify the exact S4 input

**Files:**
- Create: `experiments/tree_expert/s4_inputs.py`
- Create: `tools/prepare_tree_s4_input.py`
- Create: `tests/test_tree_expert_s4_inputs.py`

- [ ] **Step 1: Write failing round-trip and deduplication tests**

```python
def test_prepare_and_verify_s4_input_round_trip(valid_e2_handoff, tmp_path) -> None:
    archive = prepare_s4_input(valid_e2_handoff, tmp_path / "s4_input.zip")
    verified = verify_and_extract_s4_input(archive, tmp_path / "verified")
    assert verified.artifact_kind == "tree_s4_input_v1"
    assert verified.e2_handoff_sha256 == EXPECTED_E2_HANDOFF_SHA256
    assert verified.e2_root.is_dir()


def test_zip_and_expanded_copy_of_same_identity_are_deduplicated(valid_s4_input, tmp_path) -> None:
    expanded = tmp_path / "expanded"
    expanded.mkdir()
    safe_extract(valid_s4_input, expanded)
    found = discover_s4_input_candidates((valid_s4_input, expanded))
    assert len(found) == 1


def test_distinct_s4_inputs_are_rejected(valid_s4_input, changed_s4_input) -> None:
    with pytest.raises(S4InputError, match="distinct S4 input identities"):
        discover_s4_input_candidates((valid_s4_input, changed_s4_input))
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_inputs.py -q`

Expected: import failure for `s4_inputs`.

- [ ] **Step 3: Implement deterministic preparation and verification**

Expose:

```python
@dataclass(frozen=True)
class VerifiedS4Input:
    artifact_kind: str
    manifest_sha256: str
    e2_handoff_sha256: str
    e2_root: Path


def prepare_s4_input(e2_handoff: Path, output: Path) -> Path:
    verified = verify_e2_handoff(Path(e2_handoff))
    if verified.delivery is not True or file_sha256(e2_handoff) != EXPECTED_E2_HANDOFF_SHA256:
        raise S4InputError("E2 handoff identity differs")
    return write_deterministic_input(output, {"e2/handoff.zip": Path(e2_handoff).read_bytes()})
```

Use safe ZIP member validation, size limits, canonical JSON, exclusive output creation, and exact member hashes. The CLI must prepend the repository root to `sys.path` so it runs directly from the terminal.

- [ ] **Step 4: Run the S4 input and existing HC input tests**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_inputs.py tests/test_tree_expert_hc_inputs.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit input preparation**

```bash
git add experiments/tree_expert/s4_inputs.py tools/prepare_tree_s4_input.py tests/test_tree_expert_s4_inputs.py
git commit -m "feat: prepare exact S4 input artifact"
```

### Task 3: Implement leakage-safe temporal anchors

**Files:**
- Create: `experiments/tree_expert/s4_temporal.py`
- Create: `tests/test_tree_expert_s4_temporal.py`

- [ ] **Step 1: Write tests for weights, blending, and holdout blindness**

```python
def test_decay_weights_use_only_training_seasons() -> None:
    years = np.array([2020, 2021, 2022, 2023])
    weights = season_decay_weights(years, cutoff_year=2023, decay=0.55)
    np.testing.assert_allclose(weights, [0.55**3, 0.55**2, 0.55, 1.0])


def test_anchor_probability_matches_fixed_mixture() -> None:
    recent = np.array([0.2, 0.8])
    multi = np.array([0.6, 0.4])
    actual = blend_anchor(recent, multi, recent_weight=0.75)
    np.testing.assert_allclose(actual, [0.3, 0.7])


def test_structure_selector_never_receives_2024_predictions(monkeypatch) -> None:
    seen: list[tuple[int, int]] = []
    select_anchor_coverage(structure_frames(), on_read=lambda fold: seen.append(fold))
    assert seen == [(2021, 2022), (2022, 2023)]
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_temporal.py -q`

Expected: missing module failure.

- [ ] **Step 3: Implement strict temporal functions**

```python
def season_decay_weights(years: object, *, cutoff_year: int, decay: float) -> np.ndarray:
    values = np.asarray(years, dtype=np.int64)
    if values.size == 0 or np.any(values > cutoff_year):
        raise S4TemporalError("training seasons exceed cutoff")
    if not 0.0 < decay <= 1.0:
        raise S4TemporalError("decay must be in (0, 1]")
    return np.power(decay, cutoff_year - values, dtype=np.float64)


def blend_anchor(recent: object, multi: object, *, recent_weight: float) -> np.ndarray:
    left = probability_vector(recent, "recent")
    right = probability_vector(multi, "multi")
    if left.shape != right.shape or not 0.0 <= recent_weight <= 1.0:
        raise S4TemporalError("anchor mixture differs")
    return np.clip(recent_weight * left + (1.0 - recent_weight) * right, 1e-5, 1.0 - 1e-5)
```

Add row-ID alignment checks and a selection API that accepts only the two registered structure folds.

- [ ] **Step 4: Run temporal tests**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_temporal.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit temporal anchors**

```bash
git add experiments/tree_expert/s4_temporal.py tests/test_tree_expert_s4_temporal.py
git commit -m "feat: add causal seasonal anchor logic"
```

### Task 4: Add cutoff-fitted context features

**Files:**
- Create: `experiments/tree_expert/s4_features.py`
- Create: `tests/test_tree_expert_s4_features.py`

- [ ] **Step 1: Write leave-one-out, backoff, and permutation tests**

```python
def test_fit_context_never_uses_validation_target(tiny_train, tiny_valid) -> None:
    state = fit_s4_context(tiny_train, cutoff_year=2022)
    first = transform_s4_context(tiny_valid.drop(columns=["control_success"]), state)
    changed = tiny_valid.assign(control_success=1 - tiny_valid["control_success"])
    second = transform_s4_context(changed.drop(columns=["control_success"]), state)
    pd.testing.assert_frame_equal(first, second)


def test_unseen_matchup_backs_off_to_parent(tiny_train, unseen_valid) -> None:
    state = fit_s4_context(tiny_train, cutoff_year=2022)
    transformed = transform_s4_context(unseen_valid, state)
    assert transformed.loc[0, "matchup_rate"] == transformed.loc[0, "hand_rate"]


def test_context_transform_is_order_independent(tiny_train, tiny_valid) -> None:
    state = fit_s4_context(tiny_train, cutoff_year=2022)
    normal = transform_s4_context(tiny_valid, state).sort_values("row_id")
    shuffled = transform_s4_context(tiny_valid.sample(frac=1, random_state=7), state).sort_values("row_id")
    pd.testing.assert_frame_equal(normal.reset_index(drop=True), shuffled.reset_index(drop=True))
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_features.py -q`

Expected: missing module failure.

- [ ] **Step 3: Implement immutable context state**

Reuse `hc_features.shrink` and the existing canonical hierarchy payload pattern. Fit global → game type → hand → pitcher/batter → matchup entries only on `year <= cutoff_year`; store counts and smoothed rates; transform without target. Reject duplicate row IDs, missing pre-pitch columns, future years, and non-binary targets.

Public interfaces:

```python
@dataclass(frozen=True)
class S4ContextState:
    cutoff_year: int
    levels: Mapping[str, Mapping[tuple[str, ...], ContextEntry]]
```

Implement `fit_s4_context(rows, cutoff_year)`, `transform_s4_context(rows, state)`,
`context_state_payload(state)`, and `context_state_from_payload(payload)` around this type. The
functions must return fresh frames and immutable mapping proxies; they must not mutate input rows.

- [ ] **Step 4: Run S4 and existing hierarchy feature tests**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_features.py tests/test_tree_expert_hc_features.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit context features**

```bash
git add experiments/tree_expert/s4_features.py tests/test_tree_expert_s4_features.py
git commit -m "feat: add S4 cutoff context features"
```

### Task 5: Train anchor and residual jobs

**Files:**
- Create: `experiments/tree_expert/s4_training.py`
- Create: `tests/test_tree_expert_s4_training.py`

- [ ] **Step 1: Write model-factory and target tests**

```python
def test_residual_target_is_recomputed_for_each_anchor() -> None:
    target = np.array([0.0, 1.0])
    first = residual_target(target, np.array([0.2, 0.7]))
    second = residual_target(target, np.array([0.4, 0.6]))
    np.testing.assert_allclose(first, [-0.2, 0.3])
    np.testing.assert_allclose(second, [-0.4, 0.4])


def test_model_factory_registers_full_capacity_families(contract) -> None:
    assert model_parameters(contract, "catboost", gpu_id=0)["task_type"] == "GPU"
    assert model_parameters(contract, "xgboost", gpu_id=1)["device"] == "cuda:1"
    assert model_parameters(contract, "lightgbm", gpu_id=0)["num_threads"] >= 4


def test_rf_residual_routes_each_row_without_population_statistics() -> None:
    routed = route_rf(np.array(["R", "F", "R"]), np.array([0.1, 0.2, 0.3]), np.array([0.7, 0.8, 0.9]))
    np.testing.assert_allclose(routed, [0.1, 0.8, 0.3])
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_training.py -q`

Expected: missing module failure.

- [ ] **Step 3: Implement real full-capacity training adapters**

```python
def residual_target(target: object, anchor: object) -> np.ndarray:
    y = binary_target(target)
    p = probability_vector(anchor, "anchor")
    if y.shape != p.shape:
        raise S4TrainingError("residual rows differ")
    return y - p


def corrected_probability(anchor: object, correction: object, *, alpha: float) -> np.ndarray:
    p = probability_vector(anchor, "anchor")
    r = finite_vector(correction, "correction")
    if p.shape != r.shape or alpha <= 0.0:
        raise S4TrainingError("correction differs")
    return np.clip(p + alpha * r, 1e-5, 1.0 - 1e-5)
```

Use CatBoost classifier jobs for `p_recent`/`p_multi`; CatBoost/XGBoost/LightGBM regressors for residual targets; fixed categorical encoding fitted only on training rows; early stopping only against the registered fold. Save `job.json`, `metrics.json`, `predictions.csv`, model checkpoint, context state, and worker log atomically.

- [ ] **Step 4: Run training tests and existing hetero training regression tests**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_training.py tests/test_tree_expert_hetero_training.py tests/test_tree_expert_hc_training.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit training adapters**

```bash
git add experiments/tree_expert/s4_training.py tests/test_tree_expert_s4_training.py
git commit -m "feat: train S4 anchor and residual experts"
```

### Task 6: Apply train-only hierarchical calibration

**Files:**
- Create: `experiments/tree_expert/s4_calibration.py`
- Create: `tests/test_tree_expert_s4_calibration.py`

- [ ] **Step 1: Write calibration formula and target-blind tests**

```python
def test_apply_calibration_matches_additive_logit() -> None:
    probability = np.array([0.25])
    effect = np.array([0.2])
    actual = apply_s4_calibration(probability, effect, beta=0.5)
    expected = expit(logit(probability) + 0.1)
    np.testing.assert_allclose(actual, expected)


def test_calibrator_uses_only_earlier_oof_rows(rolling_rows) -> None:
    state = fit_s4_calibrator(rolling_rows, prediction_year=2024, profile="pitcher")
    assert state.maximum_source_year == 2023


def test_calibration_is_permutation_and_population_invariant(frozen_calibrator, rows) -> None:
    one = predict_calibrated(rows, frozen_calibrator).sort_values("row_id")
    two = predict_calibrated(rows.sample(frac=1, random_state=5), frozen_calibrator).sort_values("row_id")
    pd.testing.assert_frame_equal(one.reset_index(drop=True), two.reset_index(drop=True))
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_calibration.py -q`

Expected: missing module failure.

- [ ] **Step 3: Implement profiles using HC primitives**

Wrap the validated `hc_calibration` state format but bind every state to anchor ID, residual ID, seed, source years, and profile. Profiles are `global_game`, `pitcher`, `batter`, `matchup`, and `rf_matchup`; effect tables use training OOF only and back off to parents.

```python
def apply_s4_calibration(probability: object, effect: object, *, beta: float) -> np.ndarray:
    p = probability_vector(probability, "probability")
    delta = finite_vector(effect, "effect")
    if p.shape != delta.shape or beta not in (0.10, 0.25, 0.50, 0.75):
        raise S4CalibrationError("calibration arguments differ")
    z = np.log(p / (1.0 - p)) + beta * delta
    return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-5, 1.0 - 1e-5)
```

- [ ] **Step 4: Run S4 and HC calibration tests**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_calibration.py tests/test_tree_expert_hc_calibration.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit calibration**

```bash
git add experiments/tree_expert/s4_calibration.py tests/test_tree_expert_s4_calibration.py
git commit -m "feat: calibrate S4 full-chain candidates"
```

### Task 7: Select coverage and make final decisions

**Files:**
- Create: `experiments/tree_expert/s4_decisions.py`
- Create: `tests/test_tree_expert_s4_decisions.py`

- [ ] **Step 1: Write tests that prevent conservative early pruning**

```python
def test_anchor_coverage_keeps_all_six_roles(anchor_evidence) -> None:
    selected = select_anchor_coverage(anchor_evidence)
    assert {item.role for item in selected} == {
        "e2_control", "external_template", "best_weighted",
        "best_worst_fold", "most_diverse", "best_rf",
    }


def test_full_chain_grid_contains_at_least_twelve_archetypes(contract, coverage) -> None:
    candidates = full_chain_archetypes(contract, coverage)
    assert len(candidates) >= 12
    assert any(item.anchor_role == "external_template" for item in candidates)
    assert any(item.residual_family == "catboost_rf" for item in candidates)
    assert {item.calibration_profile for item in candidates} >= {"pitcher", "batter", "matchup"}


def test_acceptance_requires_every_registered_gate(passing_evidence) -> None:
    assert decide_submission_eligibility(passing_evidence).status == "accepted"
    for field in ("weighted_gain", "recent_gain", "minimum_fold_gain", "segment_regression", "bootstrap_lower"):
        changed = failing_copy(passing_evidence, field)
        assert decide_submission_eligibility(changed).status == "rejected"
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_decisions.py -q`

Expected: missing module failure.

- [ ] **Step 3: Implement role selection, Pareto confirmation, and gates**

Use only structure folds for role selection and calibration strength. Keep up to four Pareto candidates plus the external template. Compute paired Brier, HC segment diagnostics, pitcher-cluster bootstrap, calibration gap/ECE, seed counts, and residual correlation. Return explicit `research_only`, `accepted`, or `rejected` decisions with failed gate names.

The 2021→2022 and 2022→2023 folds select structure. The frozen candidates then run once on
2023→2024; no value read from 2023→2024 may alter a weight, decay, model family, profile, or
calibration strength. Confirmation uses the exact seeds `42`, `2026`, `3407`.

```python
@dataclass(frozen=True)
class S4Decision:
    candidate_id: str
    status: str
    failed_gates: tuple[str, ...]
    weighted_gain: float
    recent_gain: float
    minimum_fold_gain: float
    maximum_segment_regression: float
    bootstrap_lower: float
    non_worse_seed_count: int
```

Residual correlation is recorded but never added to `failed_gates` for a standalone full chain.

- [ ] **Step 4: Run decision and metric tests**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_decisions.py tests/test_tree_expert_hc_metrics.py tests/test_tree_expert_hc_decisions.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit decisions**

```bash
git add experiments/tree_expert/s4_decisions.py tests/test_tree_expert_s4_decisions.py
git commit -m "feat: gate S4 full-chain candidates"
```

### Task 8: Persist resumable state and deterministic artifacts

**Files:**
- Create: `experiments/tree_expert/s4_state.py`
- Create: `experiments/tree_expert/s4_artifacts.py`
- Create: `tests/test_tree_expert_s4_state.py`
- Create: `tests/test_tree_expert_s4_artifacts.py`

- [ ] **Step 1: Write state and artifact failure tests**

```python
def test_state_round_trip_preserves_phase_jobs_and_decisions(tmp_path) -> None:
    state = mark_completed(initial_s4_state(), "anchor__a0")
    state = record_s4_decision(state, "full__c1", "research_only")
    path = save_s4_state(state, tmp_path / "state.json")
    assert load_s4_state(path) == state


def test_rejected_handoff_has_no_model_or_submission(tmp_path, bindings) -> None:
    handoff = create_s4_handoff(completed_no_candidate_root(tmp_path), tmp_path / "handoff.zip", bindings)
    names = zip_names(handoff)
    assert "review.zip" in names and "resume.zip" in names
    assert "model_delivery.zip" not in names
    assert all("submission" not in name for name in names)


def test_binding_change_rejects_resume(valid_resume, bindings) -> None:
    changed = replace(bindings, contract_sha256="0" * 64)
    with pytest.raises(S4ArtifactError, match="bindings differ"):
        verify_s4_resume(valid_resume, changed)
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_state.py tests/test_tree_expert_s4_artifacts.py -q`

Expected: missing module failures.

- [ ] **Step 3: Implement atomic state and deterministic ZIPs**

Follow the verified `hetero_artifacts`/`hc_artifacts` member validation patterns. Bind contract, code, official train/history, E2 handoff, and input manifest hashes. Review must contain decisions and aggregate diagnostics; resume contains completed terminal jobs; delivery requires an accepted token. Outer handoff has one of `review_ready`, `accepted_review_ready`, `delivery_ready`, or `completed_no_candidate`.

- [ ] **Step 4: Run artifact regression tests**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_state.py tests/test_tree_expert_s4_artifacts.py tests/test_tree_expert_hetero_artifacts.py tests/test_tree_expert_hc_artifacts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit state and artifacts**

```bash
git add experiments/tree_expert/s4_state.py experiments/tree_expert/s4_artifacts.py tests/test_tree_expert_s4_state.py tests/test_tree_expert_s4_artifacts.py
git commit -m "feat: persist S4 campaign evidence"
```

### Task 9: Add accepted-only full fit and current-row inference

**Files:**
- Create: `experiments/tree_expert/s4_full_fit.py`
- Create: `experiments/tree_expert/s4_inference.py`
- Create: `tests/test_tree_expert_s4_full_fit.py`
- Create: `tests/test_tree_expert_s4_inference.py`

- [ ] **Step 1: Write token and independence tests**

```python
def test_rejected_decision_cannot_issue_full_fit_token(rejected_decision) -> None:
    with pytest.raises(S4FullFitError, match="accepted decision required"):
        issue_full_fit_token(rejected_decision, bindings())


def test_frozen_predictor_is_singleton_shuffle_reverse_and_batch_invariant(frozen_predictor, rows) -> None:
    audit = audit_s4_independence(frozen_predictor, rows, batch_sizes=(1, 17, 256))
    assert audit.maximum_absolute_difference <= 1e-6


def test_predictor_rejects_target_and_unknown_game_type(frozen_predictor, rows) -> None:
    with pytest.raises(S4InferenceError):
        frozen_predictor.predict(rows.assign(control_success=1))
    with pytest.raises(S4InferenceError):
        frozen_predictor.predict(rows.assign(game_type="UNKNOWN"))
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_full_fit.py tests/test_tree_expert_s4_inference.py -q`

Expected: missing module failures.

- [ ] **Step 3: Implement accepted-only frozen state**

Median-plus-one best iterations are computed from the three registered folds and clipped by the contract. Fit recent, multi, residual, and calibration states on official train only. The predictor takes a frame of independent current rows and applies frozen transforms, current-row R/F routing, and fixed formulas without groupby, rank, rolling, expanding, shift, or evaluation-population aggregation.

- [ ] **Step 4: Run inference and E2 production regression tests**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_full_fit.py tests/test_tree_expert_s4_inference.py tests/test_tree_expert_e2_production.py tests/test_tree_expert_rf_inference.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit full fit and inference**

```bash
git add experiments/tree_expert/s4_full_fit.py experiments/tree_expert/s4_inference.py tests/test_tree_expert_s4_full_fit.py tests/test_tree_expert_s4_inference.py
git commit -m "feat: freeze accepted S4 predictor"
```

### Task 10: Orchestrate one dual-GPU campaign

**Files:**
- Create: `experiments/tree_expert/s4_runner.py`
- Create: `tests/test_tree_expert_s4_runner.py`

- [ ] **Step 1: Write scheduler, priority, and emergency tests**

```python
def test_runner_uses_two_workers_without_duplicate_jobs(tmp_path) -> None:
    result = run_s4_campaign(fake_verified(), tmp_path, fake_two_gpu_runtime())
    assert result.maximum_concurrent_gpu_jobs == 2
    assert len(result.started_jobs) == len(set(result.started_jobs))


def test_priority_finishes_twelve_full_chains_before_optional_seeds(tmp_path) -> None:
    result = run_s4_campaign(fake_verified(), tmp_path, deadline_runtime(stop_after="full_chains"))
    assert result.completed_full_chain_count >= 12
    assert result.optional_confirmation_started is False


def test_exception_publishes_one_emergency_handoff(tmp_path) -> None:
    with pytest.raises(ExpectedWorkerFailure):
        run_s4_campaign(fake_verified(), tmp_path, failing_runtime())
    assert [path.name for path in tmp_path.glob("*handoff.zip")] == ["anchor_residual_hierarchical_handoff.zip"]
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_runner.py -q`

Expected: missing module failure.

- [ ] **Step 3: Implement phase orchestration**

Use a process pool with explicit `CUDA_VISIBLE_DEVICES` per worker, no shell execution, and one terminal result message per job. Phase transitions require validated artifacts. The priority queue is E2/external anchors → six coverage anchors → residual archetypes → twelve full chains → confirmation seeds → optional full fit. Stop starting new jobs at `wall_deadline - artifact_reserve_seconds`; always publish state and one handoff in `finally`.

- [ ] **Step 4: Run runner regression tests**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_runner.py tests/test_tree_expert_hetero_runner.py tests/test_tree_expert_rf_runner.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit orchestration**

```bash
git add experiments/tree_expert/s4_runner.py tests/test_tree_expert_s4_runner.py
git commit -m "feat: orchestrate single-run S4 campaign"
```

### Task 11: Render the one-cell Kaggle runtime

**Files:**
- Create: `experiments/tree_expert/s4_kaggle.py`
- Create: `experiments/tree_expert/KAGGLE_S4_CELL.py`
- Create: `tests/test_tree_expert_s4_kaggle.py`
- Create: `tests/test_tree_expert_s4_kaggle_cell.py`

- [ ] **Step 1: Write discovery and cell tests**

```python
def test_discovery_finds_one_official_and_one_logical_s4_input(kaggle_root) -> None:
    found = discover_s4_inputs(kaggle_root)
    assert found.official_data.name == "kaggle-lg-aimers-9th-data-upload"
    assert found.s4_input_identity == EXPECTED_S4_INPUT_IDENTITY


def test_rendered_cell_is_deterministic_small_and_has_no_submission(tmp_path) -> None:
    first = build_s4_kaggle_cell(tmp_path / "first.py")
    second = build_s4_kaggle_cell(tmp_path / "second.py")
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 1_000_000
    text = first.read_text()
    assert "S4_HANDOFF_READY" in text and "S4_SUCCESS" in text
    assert "files.download" not in text
    assert "submission/package.py" not in text


def test_committed_cell_matches_renderer(tmp_path) -> None:
    rendered = build_s4_kaggle_cell(tmp_path / "rendered.py")
    committed = Path("experiments/tree_expert/KAGGLE_S4_CELL.py")
    assert rendered.read_bytes() == committed.read_bytes()
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_kaggle.py tests/test_tree_expert_s4_kaggle_cell.py -q`

Expected: missing module and cell failures.

- [ ] **Step 3: Implement exact runtime inventory and renderer**

Build a deterministic tar archive containing only the import closure for S4 and reused tree-expert modules. The cell must install sealed dependency versions, verify two T4 devices, discover ZIP or expanded inputs by artifact identity, stream progress logs, run one campaign, and leave one handoff in `/kaggle/working`. It must not auto-download files or create a submission package.

- [ ] **Step 4: Generate the committed cell and run isolation tests**

Run: `artifacts/tabm_submission_python311/bin/python -c "from pathlib import Path; from experiments.tree_expert.s4_kaggle import build_s4_kaggle_cell; build_s4_kaggle_cell(Path('experiments/tree_expert/KAGGLE_S4_CELL.py'))"`

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_kaggle.py tests/test_tree_expert_s4_kaggle_cell.py -q`

Expected: all tests pass and the embedded runtime imports from an isolated directory.

- [ ] **Step 5: Commit Kaggle runtime**

```bash
git add experiments/tree_expert/s4_kaggle.py experiments/tree_expert/KAGGLE_S4_CELL.py tests/test_tree_expert_s4_kaggle.py tests/test_tree_expert_s4_kaggle_cell.py
git commit -m "feat: render one-cell S4 Kaggle campaign"
```

### Task 12: Add the user handoff commands and runbook

**Files:**
- Modify: `tools/prepare_tree_s4_input.py`
- Create: `docs/TREE_S4_KAGGLE.md`
- Create: `tests/test_tree_expert_s4_runbook.py`

- [ ] **Step 1: Write CLI and runbook contract tests**

```python
def test_prepare_cli_runs_outside_repository(tmp_path, valid_e2_handoff) -> None:
    result = subprocess.run(
        [PYTHON, str(ROOT / "tools/prepare_tree_s4_input.py"), "--e2-handoff", str(valid_e2_handoff), "--output", str(tmp_path / "s4.zip")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "TREE_S4_INPUT_READY" in result.stdout


def test_runbook_names_exact_inputs_runtime_and_return_file() -> None:
    text = Path("docs/TREE_S4_KAGGLE.md").read_text()
    for phrase in ("T4 x2", "Save Version", "10~12시간", "KAGGLE_S4_CELL.py", "anchor_residual_hierarchical_handoff.zip", "S4_SUCCESS"):
        assert phrase in text
```

- [ ] **Step 2: Verify RED**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_runbook.py -q`

Expected: missing runbook or marker failure.

- [ ] **Step 3: Finish the CLI and write the Korean runbook**

Document purpose, two Kaggle inputs, one-cell execution, expected start/progress/success logs, 10–12 hour estimate, no manual intermediate downloads, rerun behavior, and exactly one ZIP to return. Do not include personal absolute paths or claim an expected Public score.

- [ ] **Step 4: Run runbook tests and safety scan**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_runbook.py -q`

Run: `rg -n '/Users/|GHS[A-Z0-9]|[T]ODO|[T]BD' docs/TREE_S4_KAGGLE.md tools/prepare_tree_s4_input.py`

Expected: tests pass and the scan prints nothing.

- [ ] **Step 5: Commit the handoff UX**

```bash
git add tools/prepare_tree_s4_input.py docs/TREE_S4_KAGGLE.md tests/test_tree_expert_s4_runbook.py
git commit -m "docs: add S4 Kaggle handoff guide"
```

### Task 13: Run focused and repository verification

**Files:**
- Verify only; do not modify unrelated dirty files.

- [ ] **Step 1: Run the complete S4 suite**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_*.py -q`

Expected: all S4 tests pass.

- [ ] **Step 2: Run reused tree-expert regression suites**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_hc_*.py tests/test_tree_expert_hetero_*.py tests/test_tree_expert_rf_*.py tests/test_tree_expert_e2_*.py -q`

Expected: all selected tests pass.

- [ ] **Step 3: Run policy and repository contracts**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_competition_policy.py tests/test_experiment_contract_gate.py tests/test_repository_contract.py -q`

Expected: all tests pass.

- [ ] **Step 4: Run source, syntax, and patch checks**

Run: `artifacts/tabm_submission_python311/bin/python -m compileall -q experiments/tree_expert tools/prepare_tree_s4_input.py`

Run: `test "$(wc -c < experiments/tree_expert/KAGGLE_S4_CELL.py)" -lt 1000000`

Run: `rg -n 'groupby|rolling|expanding|shift|rank' experiments/tree_expert/s4_inference.py`

Expected: compile and size checks exit zero; forbidden population-operation scan prints nothing.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 5: Review the final tracked scope**

Run: `git status --short && git log --oneline --max-count=15`

Expected: only the pre-existing unrelated TabM edits and notebook remain dirty; S4 work is committed in task-sized commits.

### Task 14: Prepare—but do not run—the user GPU handoff

**Files:**
- Read: `docs/TREE_S4_KAGGLE.md`
- Read: `experiments/tree_expert/KAGGLE_S4_CELL.py`

- [ ] **Step 1: Report the exact local preparation command**

Provide the user one copyable command using their actual E2 handoff path only after resolving it read-only. The command must produce `artifacts/tree_s4_input.zip` and print `TREE_S4_INPUT_READY` with its SHA-256.

- [ ] **Step 2: Report the Kaggle input and execution checklist**

The checklist must state:

```text
Purpose: compare at least twelve anchor + residual + hierarchical calibration pipelines
Inputs: official lg-aimers-9th-data and tree_s4_input
Accelerator: GPU T4 x2
Execution: paste KAGGLE_S4_CELL.py into one cell and Save Version once
Expected runtime: 10~12 hours
Rerun safety: terminal jobs are internally checkpointed; no manual intermediate download
Success text: S4_HANDOFF_READY and S4_SUCCESS
Return file: anchor_residual_hierarchical_handoff.zip
```

- [ ] **Step 3: Stop before full-data execution**

Do not run official preprocessing, temporal OOF, GPU training, full fit, or submission packaging locally. Wait for the user to run Kaggle and return the handoff.
