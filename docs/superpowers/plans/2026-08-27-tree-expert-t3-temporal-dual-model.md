# Tree Expert T3 Temporal Dual Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restartable Kaggle T4×2 campaign that compares a previous-season CatBoost residual expert with time-decayed multi-season experts, accepts only a temporally stable blend, and returns auditable review, resume, and model-delivery artifacts.

**Architecture:** Extend `experiments/tree_expert` with isolated `t3_*` modules while reusing the accepted E2 feature and anchor implementation. Structure selection uses seed 3407 on three temporal folds, chooses decay and recent-model weight on the first two folds, reserves 2023→2024 as confirmation, and runs two additional seeds only for the selected structure. An accepted candidate is full-fitted and exported as a model delivery, but no DACON `submit.zip` writer is added until the returned evidence is independently reviewed.

**Tech Stack:** Python 3.11, pandas 2.0.3, NumPy 1.26.4, CatBoost 1.2.10 GPU, pytest, deterministic ZIP/JSON artifacts, Kaggle T4×2.

---

## File map

- Create `experiments/tree_expert/t3_contract.json`: immutable experiment grid, hashes, gates, and runtime budget.
- Create `experiments/tree_expert/t3_contracts.py`: strict contract loader and deterministic job planner.
- Create `experiments/tree_expert/t3_temporal.py`: recent-row selection and time-decay weights.
- Create `experiments/tree_expert/t3_training.py`: one CatBoost residual job with progress and checkpoint metadata.
- Create `experiments/tree_expert/t3_decisions.py`: structure selection, seed confirmation, and acceptance gates.
- Create `experiments/tree_expert/t3_diagnostics.py`: OOF-only segment, calibration, and residual-correlation evidence.
- Create `experiments/tree_expert/t3_artifacts.py`: deterministic resume, review, model-delivery, and handoff bundles.
- Create `experiments/tree_expert/t3_inputs.py`: compact input creation and official-data verification.
- Create `experiments/tree_expert/t3_full_fit.py`: accepted-only two-head full fit and frozen-state export.
- Create `experiments/tree_expert/t3_inference.py`: row-independent recent/multi blend inference.
- Create `experiments/tree_expert/t3_runner.py`: two-GPU scheduling, resume restore, decisions, and full fit.
- Create `experiments/tree_expert/t3_kaggle.py`: runtime archive and one-cell renderer.
- Generate `experiments/tree_expert/KAGGLE_T3_CELL.py`: user-facing Kaggle cell.
- Create `tools/prepare_tree_expert_t3_input.py`: local compact input builder.
- Create `tools/build_tree_expert_t3_kaggle_cell.py`: deterministic cell generator.
- Create focused tests named `tests/test_tree_expert_t3_*.py` for every module.

The existing E2 source, accepted artifacts, submission builder, and 977-point ZIP remain unchanged.

### Task 1: Freeze the T3 contract and job identities

**Files:**
- Create: `experiments/tree_expert/t3_contract.json`
- Create: `experiments/tree_expert/t3_contracts.py`
- Test: `tests/test_tree_expert_t3_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from dataclasses import FrozenInstanceError

import pytest

from experiments.tree_expert.t3_contracts import load_t3_contract, structure_jobs


def test_t3_contract_fixes_the_small_search_space():
    contract = load_t3_contract()
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.decays == (0.35, 0.55, 0.75)
    assert contract.recent_weights == (0.70, 0.80, 0.90)
    assert contract.structure_seed == 3407
    assert contract.confirmation_seeds == (42, 2026)
    assert contract.wall_seconds == 28_800
    with pytest.raises(FrozenInstanceError):
        contract.structure_seed = 42


def test_structure_jobs_have_unique_stable_ids():
    jobs = structure_jobs(load_t3_contract())
    assert len(jobs) == 12
    assert len({job.job_id for job in jobs}) == 12
    assert jobs[0].job_id == "t3__recent__tr2021__va2022__s3407"
    assert jobs[-1].job_id == "t3__multi_d075__tr2023__va2024__s3407"
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_contracts.py
```

Expected: collection fails with `ModuleNotFoundError: experiments.tree_expert.t3_contracts`.

- [ ] **Step 3: Add the immutable JSON contract**

```json
{
  "schema_version": 1,
  "campaign_id": "tree_expert_t3_temporal_dual_v1",
  "folds": [[2021, 2022], [2022, 2023], [2023, 2024]],
  "decays": [0.35, 0.55, 0.75],
  "recent_weights": [0.70, 0.80, 0.90],
  "structure_seed": 3407,
  "confirmation_seeds": [42, 2026],
  "gates": {
    "weighted_gain": 0.00015,
    "recent_fold_gain": 0.0,
    "maximum_fold_regression": 0.00005,
    "maximum_segment_regression": 0.0005,
    "minimum_segment_rows": 5000,
    "minimum_non_worse_seed_count": 2
  },
  "runtime": {
    "wall_seconds": 28800,
    "new_job_guard_seconds": 600,
    "snapshot_interval_seconds": 600,
    "inference_max_seconds": 480,
    "rss_max_bytes": 23622320128,
    "probability_tolerance": 0.000001
  }
}
```

- [ ] **Step 4: Implement strict parsing and deterministic jobs**

```python
@dataclass(frozen=True)
class T3Job:
    job_id: str
    head: str
    decay: float | None
    train_end_year: int
    valid_year: int
    seed: int


def structure_jobs(contract: T3Contract) -> tuple[T3Job, ...]:
    jobs: list[T3Job] = []
    for train_end, valid_year in contract.folds:
        jobs.append(T3Job(
            f"t3__recent__tr{train_end}__va{valid_year}__s{contract.structure_seed}",
            "recent", None, train_end, valid_year, contract.structure_seed,
        ))
        for decay in contract.decays:
            label = f"d{int(round(decay * 100)):03d}"
            jobs.append(T3Job(
                f"t3__multi_{label}__tr{train_end}__va{valid_year}__s{contract.structure_seed}",
                "multi", decay, train_end, valid_year, contract.structure_seed,
            ))
    return tuple(jobs)
```

The loader must reject extra keys, booleans in numeric fields, non-finite values, changed fold order, and any grid other than the JSON above.

- [ ] **Step 5: Run and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_contracts.py
```

Expected: `2 passed`.

Commit:

```bash
git add experiments/tree_expert/t3_contract.json experiments/tree_expert/t3_contracts.py tests/test_tree_expert_t3_contracts.py
git commit -m "feat(tree-expert): define T3 temporal contract"
```

### Task 2: Implement leakage-safe temporal row weights

**Files:**
- Create: `experiments/tree_expert/t3_temporal.py`
- Test: `tests/test_tree_expert_t3_temporal.py`

- [ ] **Step 1: Write exact row-selection tests**

```python
import numpy as np
import pandas as pd

from experiments.tree_expert.t3_temporal import temporal_training_weights


def test_recent_head_uses_only_the_immediate_previous_season():
    seasons = pd.Series([2019, 2021, 2022, 2023, 2023])
    weights = temporal_training_weights(seasons, valid_year=2024, head="recent")
    np.testing.assert_array_equal(weights, [0.0, 0.0, 0.0, 1.0, 1.0])


def test_multi_head_applies_decay_from_the_immediate_previous_season():
    seasons = pd.Series([2021, 2022, 2023])
    weights = temporal_training_weights(
        seasons, valid_year=2024, head="multi", decay=0.55,
    )
    np.testing.assert_allclose(weights, [0.55**2, 0.55, 1.0])
```

- [ ] **Step 2: Verify failure before implementation**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_temporal.py
```

Expected: missing-module failure.

- [ ] **Step 3: Implement the exact weighting rule**

```python
def temporal_training_weights(
    seasons: pd.Series,
    *,
    valid_year: int,
    head: str,
    decay: float | None = None,
) -> np.ndarray:
    numeric = pd.to_numeric(seasons, errors="raise").to_numpy(dtype="float64")
    if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
        raise T3TemporalError("training seasons must be finite integers")
    values = numeric.astype("int64")
    if values.size == 0 or np.any(values >= valid_year):
        raise T3TemporalError("training seasons must precede validation")
    age = (valid_year - 1) - values
    if age.min() != 0:
        raise T3TemporalError("immediate previous season is missing")
    if head == "recent" and decay is None:
        result = (age == 0).astype("float64")
    elif head == "multi" and decay in {0.35, 0.55, 0.75}:
        result = np.power(float(decay), age, dtype="float64")
    else:
        raise T3TemporalError("temporal head or decay differs")
    if not np.any(result > 0) or not np.isfinite(result).all():
        raise T3TemporalError("temporal weights are invalid")
    result.setflags(write=False)
    return result
```

- [ ] **Step 4: Add rejection tests and run**

```python
@pytest.mark.parametrize(
    ("seasons", "head", "decay", "message"),
    [
        ([2023, 2024], "recent", None, "must precede validation"),
        ([2022, 2023], "multi", 0.50, "head or decay differs"),
        ([2021, 2022], "recent", None, "previous season is missing"),
        ([2022.5, 2023], "multi", 0.55, "finite integers"),
    ],
)
def test_temporal_weights_reject_invalid_inputs(seasons, head, decay, message):
    with pytest.raises(T3TemporalError, match=message):
        temporal_training_weights(
            pd.Series(seasons), valid_year=2024, head=head, decay=decay,
        )
```

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_temporal.py
```

Expected: all temporal tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/t3_temporal.py tests/test_tree_expert_t3_temporal.py
git commit -m "feat(tree-expert): add leakage-safe temporal weights"
```

### Task 3: Train one recent or decayed residual job

**Files:**
- Create: `experiments/tree_expert/t3_training.py`
- Test: `tests/test_tree_expert_t3_training.py`

- [ ] **Step 1: Write a fake-model test for weights and predictions**

```python
def test_t3_job_passes_temporal_weights_and_returns_row_aligned_probabilities(tmp_path):
    model = RecordingRegressor(prediction=0.02, best_iteration=7)
    result = run_t3_job(
        job=multi_job(decay=0.55),
        train=synthetic_train(),
        valid=synthetic_valid(),
        baseline=synthetic_baseline(),
        output_dir=tmp_path,
        model_factory=lambda _: model,
    )
    assert result.status == "completed"
    assert model.sample_weight.tolist() == [0.55**2, 0.55, 1.0]
    assert result.best_iteration == 7
    assert list(result.predictions.columns) == ["row_id", "control_success"]
```

- [ ] **Step 2: Run and confirm the missing implementation**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_training.py
```

Expected: missing-module failure.

- [ ] **Step 3: Implement training by reusing the accepted feature path**

```python
def run_t3_job(
    *,
    job: T3Job,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    baseline: pd.DataFrame,
    output_dir: Path,
    gpu_id: int = 0,
    contract: T3Contract | None = None,
    model_factory: ModelFactory = _default_model_factory,
) -> T3JobResult:
    prefix = train.loc[pd.to_numeric(train["season"]).le(job.train_end_year)].copy()
    validation = valid.loc[pd.to_numeric(valid["season"]).eq(job.valid_year)].copy()
    state, train_batch = fit_tree_features(
        prefix, None, valid_year=job.valid_year, use_trackman=False,
    )
    valid_batch = transform_tree_features(
        validation.drop(columns="control_success"), state,
    )
    sample_weight = temporal_training_weights(
        prefix["season"], valid_year=job.valid_year,
        head=job.head, decay=job.decay,
    )
    selected = sample_weight > 0
    residual = train_batch.target[selected] - train_batch.anchor[selected]
    model = model_factory(catboost_parameters(job, output_dir))
    model.fit(
        train_batch.frame.loc[selected], residual,
        sample_weight=sample_weight[selected],
        cat_features=list(state.categorical_columns),
        eval_set=(valid_batch.frame, validation["control_success"] - valid_batch.anchor),
        use_best_model=True,
        verbose=50,
    )
    probability = np.clip(
        valid_batch.anchor + np.asarray(model.predict(valid_batch.frame)),
        1e-5, 1 - 1e-5,
    )
    prediction = pd.DataFrame({
        "row_id": valid_batch.row_id,
        "control_success": probability,
    })
    target = pd.to_numeric(validation["control_success"]).to_numpy(dtype="float64")
    baseline_probability = aligned_baseline(baseline, valid_batch.row_id)
    brier = float(np.mean(np.square(probability - target)))
    baseline_brier = float(np.mean(np.square(baseline_probability - target)))
    result = T3JobResult(
        status="completed",
        job_id=job.job_id,
        predictions=prediction,
        brier=brier,
        baseline_brier=baseline_brier,
        gain=baseline_brier - brier,
        best_iteration=max(0, int(model.get_best_iteration())),
        failure=None,
    )
    write_completed_job(output_dir, job, result, model, float(sample_weight.sum()))
    return result
```

Persist `job.json`, `metrics.json`, `predictions.csv`, `checkpoint.cbm`, and `worker.log` atomically. Metrics must include Brier, baseline Brier, gain, best iteration, row count, weight sum, and elapsed seconds.

- [ ] **Step 4: Test failure isolation and deterministic metadata**

```python
def test_failed_model_writes_no_checkpoint(tmp_path):
    result = run_t3_job(
        job=recent_job(), train=synthetic_train(), valid=synthetic_valid(),
        baseline=synthetic_baseline(), output_dir=tmp_path,
        model_factory=lambda _: RaisingRegressor("fit failed"),
    )
    assert result.status == "failed"
    assert not (tmp_path / "checkpoint.cbm").exists()
    assert "fit failed" in (tmp_path / "worker.log").read_text()


def test_duplicate_validation_row_id_is_rejected(tmp_path):
    valid = synthetic_valid()
    valid.loc[valid.index[1], "row_id"] = valid.loc[valid.index[0], "row_id"]
    with pytest.raises(TreeFeatureError, match="row_id"):
        run_t3_job(
            job=recent_job(), train=synthetic_train(), valid=valid,
            baseline=synthetic_baseline(), output_dir=tmp_path,
            model_factory=lambda _: RecordingRegressor(0.0, 1),
        )


def test_baseline_row_identity_must_match(tmp_path):
    baseline = synthetic_baseline().iloc[::-1].reset_index(drop=True)
    with pytest.raises(T3TrainingError, match="baseline row identity"):
        run_t3_job(
            job=recent_job(), train=synthetic_train(), valid=synthetic_valid(),
            baseline=baseline, output_dir=tmp_path,
            model_factory=lambda _: RecordingRegressor(0.0, 1),
        )
```

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_training.py
```

Expected: all training tests pass without importing CatBoost.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/t3_training.py tests/test_tree_expert_t3_training.py
git commit -m "feat(tree-expert): train T3 temporal residual heads"
```

### Task 4: Select decay and blend without using the confirmation fold

**Files:**
- Create: `experiments/tree_expert/t3_decisions.py`
- Create: `experiments/tree_expert/t3_diagnostics.py`
- Test: `tests/test_tree_expert_t3_decisions.py`
- Test: `tests/test_tree_expert_t3_diagnostics.py`

- [ ] **Step 1: Write a selection-isolation test**

```python
def test_structure_selection_ignores_f3_when_choosing_parameters():
    evidence_a = synthetic_structure_evidence(f3_override=0.01)
    evidence_b = synthetic_structure_evidence(f3_override=0.99)
    first = select_structure(evidence_a, load_t3_contract())
    second = select_structure(evidence_b, load_t3_contract())
    assert (first.decay, first.recent_weight) == (second.decay, second.recent_weight)
    assert first.selection_folds == ((2021, 2022), (2022, 2023))
    assert first.confirmation_fold == (2023, 2024)
```

- [ ] **Step 2: Run and verify the missing-module failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_decisions.py
```

Expected: missing-module failure.

- [ ] **Step 3: Implement deterministic probability blending and selection**

```python
def blended_probability(
    recent: np.ndarray,
    multi: np.ndarray,
    recent_weight: float,
) -> np.ndarray:
    if recent_weight not in {0.70, 0.80, 0.90}:
        raise T3DecisionError("recent weight differs")
    return np.clip(
        recent_weight * recent + (1.0 - recent_weight) * multi,
        1e-5, 1 - 1e-5,
    )


def select_structure(evidence: StructureEvidence, contract: T3Contract) -> StructureDecision:
    candidates = []
    for decay in contract.decays:
        for recent_weight in contract.recent_weights:
            score = weighted_brier_for_folds(
                evidence, decay, recent_weight,
                folds=contract.folds[:2],
            )
            candidates.append((score, decay, recent_weight))
    _, decay, recent_weight = min(candidates)
    return evaluate_selected_structure(evidence, decay, recent_weight, contract)
```

Tie-breaking must be lower Brier, then higher recent weight, then decay closest to 0.55. The returned decision records all nine selection scores and the untouched F3 result.

- [ ] **Step 4: Compute OOF-only segment and calibration evidence**

```python
SEGMENTS = (
    "game_type", "control_success", "pitcher_n_bucket",
    "hand_matchup", "count_state",
)


def segment_values(rows: pd.DataFrame, column: str) -> pd.Series:
    if column in rows:
        return rows[column].astype("string").fillna("__MISSING__")
    if column == "pitcher_n_bucket":
        return pd.cut(
            pd.to_numeric(rows["asof_pitcher_n"], errors="coerce"),
            [-np.inf, 0, 25, 100, 500, np.inf],
            labels=["zero", "1_25", "26_100", "101_500", "over_500"],
        ).astype("string").fillna("__MISSING__")
    if column == "hand_matchup":
        return rows["pitcher_hand"].astype("string").str.cat(
            rows["batter_hand"].astype("string"), sep="|",
        ).fillna("__MISSING__")
    if column == "count_state":
        return rows["balls_before"].astype("string").str.cat(
            rows["strikes_before"].astype("string"), sep="|",
        ).fillna("__MISSING__")
    raise T3DiagnosticError(f"unsupported segment: {column}")


def segment_diagnostics(
    rows: pd.DataFrame,
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    minimum_rows: int,
) -> pd.DataFrame:
    if "control_success" not in rows or len(rows) != len(target):
        raise T3DiagnosticError("labeled OOF rows are required")
    records: list[dict[str, object]] = []
    for column in SEGMENTS:
        values = segment_values(rows.reset_index(drop=True), column)
        for value in sorted(values.unique().tolist()):
            positions = np.flatnonzero(values.eq(value).to_numpy())
            if len(positions) < minimum_rows:
                continue
            base_brier = float(np.mean(np.square(baseline[positions] - target[positions])))
            candidate_brier = float(np.mean(np.square(candidate[positions] - target[positions])))
            records.append({
                "segment": column, "value": str(value), "rows": len(positions),
                "baseline_brier": base_brier, "candidate_brier": candidate_brier,
                "gain": base_brier - candidate_brier,
            })
    return pd.DataFrame.from_records(records).sort_values(
        ["segment", "value"], kind="stable", ignore_index=True,
    )


def calibration_diagnostics(target: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, 11)
    bucket = np.clip(np.digitize(prediction, bins[1:-1]), 0, 9)
    return pd.DataFrame({"target": target, "prediction": prediction, "bin": bucket}).groupby(
        "bin", sort=True, observed=False,
    ).agg(rows=("target", "size"), mean_target=("target", "mean"), mean_prediction=("prediction", "mean")).reset_index()


def residual_correlation(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    names = tuple(sorted(predictions))
    residuals = np.column_stack([target - predictions[name] for name in names])
    matrix = np.corrcoef(residuals, rowvar=False)
    return pd.DataFrame(matrix, index=names, columns=names)
```

```python
def test_diagnostics_use_only_labeled_oof_rows():
    report = segment_diagnostics(
        labeled_oof_rows(), oof_target(), oof_candidate(), oof_baseline(), minimum_rows=2,
    )
    assert set(report.columns) == {
        "segment", "value", "rows", "baseline_brier", "candidate_brier", "gain",
    }
    assert (report["rows"] >= 2).all()
    correlation = residual_correlation(
        oof_target(), {"baseline": oof_baseline(), "candidate": oof_candidate()},
    )
    assert correlation.index.tolist() == ["baseline", "candidate"]
    assert np.allclose(np.diag(correlation), 1.0)
    with pytest.raises(T3DiagnosticError, match="labeled OOF"):
        segment_diagnostics(
            unlabeled_evaluation_rows(), oof_target(), oof_candidate(), oof_baseline(),
            minimum_rows=2,
        )
```

- [ ] **Step 5: Implement seed confirmation and acceptance gates**

```python
def accept_t3(evidence: AcceptanceEvidence, contract: T3Contract) -> AcceptanceDecision:
    gains = fold_gains(evidence)
    accepted = (
        evidence.weighted_gain >= contract.gates.weighted_gain
        and gains[(2023, 2024)] > contract.gates.recent_fold_gain
        and min(gains.values()) >= -contract.gates.maximum_fold_regression
        and evidence.maximum_segment_regression <= contract.gates.maximum_segment_regression
        and evidence.non_worse_seed_count >= contract.gates.minimum_non_worse_seed_count
    )
    return AcceptanceDecision(
        status="accepted" if accepted else "rejected",
        reason="all_t3_gates_passed" if accepted else first_failed_gate(evidence, contract),
        weighted_gain=evidence.weighted_gain,
        fold_gains=MappingProxyType(dict(gains)),
    )
```

- [ ] **Step 6: Run and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_decisions.py tests/test_tree_expert_t3_diagnostics.py
```

Expected: selection, tie-break, rejection, and acceptance cases pass.

Commit:

```bash
git add experiments/tree_expert/t3_decisions.py experiments/tree_expert/t3_diagnostics.py tests/test_tree_expert_t3_decisions.py tests/test_tree_expert_t3_diagnostics.py
git commit -m "feat(tree-expert): gate T3 temporal candidates"
```

### Task 5: Bind compact inputs to the accepted E2 evidence

**Files:**
- Create: `experiments/tree_expert/t3_inputs.py`
- Create: `tools/prepare_tree_expert_t3_input.py`
- Test: `tests/test_tree_expert_t3_inputs.py`

- [ ] **Step 1: Write archive identity and tamper tests**

```python
def test_prepare_and_verify_t3_input_round_trip(tmp_path):
    archive = prepare_t3_input(
        e2_handoff=fixture_e2_handoff(), output=tmp_path / "t3_input.zip",
    )
    verified = verify_and_extract_t3_input(archive, tmp_path / "verified")
    assert verified.e2_candidate_id == "c1_anchor_residual"
    assert verified.e2_handoff_sha256 == TREE_E2_HANDOFF_SHA256
    assert set(verified.fold_predictions) == {(2021, 2022), (2022, 2023), (2023, 2024)}


def test_t3_input_rejects_changed_prediction_bytes(tmp_path):
    archive = tampered_t3_input(tmp_path)
    with pytest.raises(T3InputError, match="member SHA-256 differs"):
        verify_and_extract_t3_input(archive, tmp_path / "verified")
```

- [ ] **Step 2: Run and confirm the failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_inputs.py
```

Expected: missing-module failure.

- [ ] **Step 3: Implement a minimal deterministic input archive**

The archive member set is exact:

```python
T3_INPUT_MEMBERS = {
    "e2/acceptance.json",
    "e2/fold_2021_2022.csv",
    "e2/fold_2022_2023.csv",
    "e2/fold_2023_2024.csv",
    "e2/handoff_manifest.json",
    "manifest.json",
}
```

`manifest.json` records the T3 contract hash, exact E2 handoff hash, each member hash and size, official train/history hashes, and row-id hashes. Extraction rejects unsafe paths, encryption, duplicate names, extra names, oversize expansion, invalid JSON, changed row order, and mismatched hashes.

- [ ] **Step 4: Add the local preparation command**

```python
def main() -> int:
    args = parser().parse_args()
    result = prepare_t3_input(
        e2_handoff=Path(args.e2_handoff), output=Path(args.output),
    )
    print(f"TREE_T3_INPUT_READY path={result} sha256={file_sha256(result)}")
    return 0
```

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_inputs.py
```

Expected: all input tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/t3_inputs.py tools/prepare_tree_expert_t3_input.py tests/test_tree_expert_t3_inputs.py
git commit -m "feat(tree-expert): bind T3 inputs to E2 evidence"
```

### Task 6: Create deterministic review and single-resume artifacts

**Files:**
- Create: `experiments/tree_expert/t3_artifacts.py`
- Test: `tests/test_tree_expert_t3_artifacts.py`

- [ ] **Step 1: Write artifact round-trip and single-resume tests**

```python
def test_resume_round_trip_preserves_completed_jobs(tmp_path):
    bundle = create_resume_bundle(campaign_fixture(tmp_path), tmp_path / "resume.zip")
    restored = restore_resume_bundle(bundle, tmp_path / "restored", expected_bindings())
    assert restored.completed_jobs == ("job_a", "job_b")


def test_snapshot_directory_contains_only_latest_verified_resume(tmp_path):
    first = publish_stable_resume(campaign_fixture(tmp_path), tmp_path / "snapshots")
    second = publish_stable_resume(campaign_fixture(tmp_path), tmp_path / "snapshots")
    assert first != second
    assert [path.name for path in (tmp_path / "snapshots").glob("*.zip")] == [second.name]
```

- [ ] **Step 2: Run and verify the missing-module failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_artifacts.py
```

Expected: missing-module failure.

- [ ] **Step 3: Implement bundle bindings and atomic publication**

```python
@dataclass(frozen=True)
class T3Bindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str
    e2_handoff_sha256: str


def publish_stable_resume(campaign_root: Path, snapshot_root: Path) -> Path:
    candidate = snapshot_root / ".next_resume.zip"
    create_resume_bundle(campaign_root, candidate)
    verify_resume_bundle(candidate, load_bindings(campaign_root))
    final = snapshot_root / "tree_expert_t3_resume.zip"
    os.replace(candidate, final)
    for stale in snapshot_root.glob("*.zip"):
        if stale != final:
            stale.unlink()
    return final
```

Review artifacts include predictions and diagnostics but no training checkpoint required for resume. Resume artifacts include only completed-job checkpoints, state, decisions, logs, and exact bindings. Model delivery is emitted only after an accepted full fit.

- [ ] **Step 4: Add corruption, wrong-binding, and path-safety tests**

```python
def test_resume_rejects_changed_member(tmp_path):
    bundle = create_resume_bundle(campaign_fixture(tmp_path), tmp_path / "resume.zip")
    changed = rewrite_zip_member(bundle, "state/stage_state.json", b"{}")
    with pytest.raises(T3ArtifactError, match="member SHA-256 differs"):
        verify_resume_bundle(changed, expected_bindings())


def test_resume_rejects_wrong_bindings(tmp_path):
    bundle = create_resume_bundle(campaign_fixture(tmp_path), tmp_path / "resume.zip")
    wrong = replace(expected_bindings(), e2_handoff_sha256="0" * 64)
    with pytest.raises(T3ArtifactError, match="bindings differ"):
        verify_resume_bundle(bundle, wrong)


def test_resume_rejects_parent_path(tmp_path):
    unsafe = zip_with_members(tmp_path / "unsafe.zip", {"../state.json": b"{}"})
    with pytest.raises(T3ArtifactError, match="unsafe archive member"):
        verify_resume_bundle(unsafe, expected_bindings())
```

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_artifacts.py
```

Expected: all artifact tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/t3_artifacts.py tests/test_tree_expert_t3_artifacts.py
git commit -m "feat(tree-expert): add T3 review and resume bundles"
```

### Task 7: Add accepted-only full fit and row-independent inference

**Files:**
- Create: `experiments/tree_expert/t3_full_fit.py`
- Create: `experiments/tree_expert/t3_inference.py`
- Test: `tests/test_tree_expert_t3_full_fit.py`
- Test: `tests/test_tree_expert_t3_inference.py`

- [ ] **Step 1: Write the acceptance hard-gate test**

```python
def test_full_fit_rejects_nonaccepted_decision(tmp_path):
    with pytest.raises(T3FullFitError, match="candidate is not accepted"):
        full_fit_t3(
            decision=rejected_decision(), data=official_fixture(), output_dir=tmp_path,
        )
```

- [ ] **Step 2: Write the two-head inference formula test**

```python
def test_inference_blends_recent_and_multi_probabilities():
    predictor = T3Predictor(
        state=feature_state_fixture(),
        recent_models=(ConstantResidual(0.02),),
        multi_models=(ConstantResidual(-0.01),),
        recent_weight=0.80,
    )
    prediction = predictor.predict(singleton_rows())
    expected = 0.80 * (anchor() + 0.02) + 0.20 * (anchor() - 0.01)
    np.testing.assert_allclose(prediction, np.clip(expected, 1e-5, 1 - 1e-5))
```

- [ ] **Step 3: Run and verify both modules are absent**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_full_fit.py tests/test_tree_expert_t3_inference.py
```

Expected: missing-module failure.

- [ ] **Step 4: Implement full fit with fold-derived iteration counts**

```python
def full_fit_iterations(best_iterations: Sequence[int]) -> int:
    values = tuple(best_iterations)
    if len(values) != 3 or any(type(value) is not int or value < 0 for value in values):
        raise T3FullFitError("three fold iterations are required")
    return min(400, max(50, int(statistics.median(values)) + 1))


def full_fit_t3(
    *,
    decision: AcceptanceDecision,
    e2_token: AcceptedForFullFit,
    data: VerifiedOfficialData,
    output_dir: Path,
    gpu_ids: tuple[int, int] = (0, 1),
    model_factory: ModelFactory = _default_model_factory,
) -> T3FullFitResult:
    if decision.status != "accepted":
        raise T3FullFitError("candidate is not accepted")
    state, batch = prepare_full_fit_features(e2_token, data)
    seasons = pd.to_numeric(pd.read_csv(data.train, usecols=["season"])["season"])
    recent_weights = temporal_training_weights(
        seasons, valid_year=2025, head="recent",
    )
    multi_weights = temporal_training_weights(
        seasons, valid_year=2025, head="multi", decay=decision.decay,
    )
    return fit_heads_for_seeds(
        state, batch, recent_weights, multi_weights,
        seeds=(42, 2026, 3407), decision=decision, output_dir=output_dir,
        gpu_ids=gpu_ids, model_factory=model_factory,
    )
```

Each seed produces one recent and one multi model. The frozen state records the chosen decay, recent weight, E2 feature-state hash, acceptance hash, and all six model hashes.

- [ ] **Step 5: Implement batch-independent inference**

```python
def predict(self, rows: pd.DataFrame, *, batch_size: int = 4096) -> np.ndarray:
    batch = transform_tree_features(rows, self.state)
    anchor = np.asarray(batch.anchor, dtype="float64")
    recent = np.mean([
        np.clip(anchor + model.predict(batch.frame), 1e-5, 1 - 1e-5)
        for model in self.recent_models
    ], axis=0)
    multi = np.mean([
        np.clip(anchor + model.predict(batch.frame), 1e-5, 1 - 1e-5)
        for model in self.multi_models
    ], axis=0)
    return np.clip(
        self.recent_weight * recent + (1.0 - self.recent_weight) * multi,
        1e-5, 1 - 1e-5,
    )
```

The method must not inspect other rows except for model batch evaluation. Add this exact invariance test:

```python
def test_prediction_is_independent_of_companion_rows_and_order():
    predictor = predictor_fixture()
    rows = inference_rows_fixture().reset_index(drop=True)
    full = predictor.predict(rows, batch_size=4096)
    reversed_prediction = predictor.predict(rows.iloc[::-1].reset_index(drop=True), batch_size=257)[::-1]
    np.testing.assert_allclose(full, reversed_prediction, atol=1e-6, rtol=0.0)
    for position in (0, len(rows) // 2, len(rows) - 1):
        singleton = predictor.predict(rows.iloc[[position]].copy(), batch_size=1)
        np.testing.assert_allclose(singleton, full[[position]], atol=1e-6, rtol=0.0)
```

- [ ] **Step 6: Run and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_full_fit.py tests/test_tree_expert_t3_inference.py
```

Expected: all full-fit and inference tests pass using fake models.

Commit:

```bash
git add experiments/tree_expert/t3_full_fit.py experiments/tree_expert/t3_inference.py tests/test_tree_expert_t3_full_fit.py tests/test_tree_expert_t3_inference.py
git commit -m "feat(tree-expert): full-fit accepted T3 candidates"
```

### Task 8: Orchestrate two GPUs with safe resume

**Files:**
- Create: `experiments/tree_expert/t3_runner.py`
- Test: `tests/test_tree_expert_t3_runner.py`

- [ ] **Step 1: Write scheduling and reuse tests**

```python
def test_runner_uses_two_workers_and_reuses_completed_jobs(tmp_path):
    runtime = FakeRuntime(gpu_count=2, completed={"t3__recent__tr2021__va2022__s3407"})
    result = run_t3_campaign(verified_fixture(), tmp_path, runtime=runtime)
    assert runtime.maximum_concurrency == 2
    assert "t3__recent__tr2021__va2022__s3407" not in runtime.started_jobs
    assert result.review_bundle.is_file()
    assert result.resume_bundle.is_file()
```

- [ ] **Step 2: Run and confirm the missing-module failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_runner.py
```

Expected: missing-module failure.

- [ ] **Step 3: Implement the campaign phases**

```python
class T3Phase(str, Enum):
    STRUCTURE = "structure"
    CONFIRMATION = "confirmation"
    FULL_FIT = "full_fit"
    COMPLETE = "complete"


def run_t3_campaign(
    verified: VerifiedT3Input,
    output_dir: Path,
    *,
    resume_bundle: Path | None = None,
    runtime: T3Runtime | None = None,
    wall_deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> T3CampaignResult:
    state = restore_or_initialize(verified, output_dir, resume_bundle)
    run_pending_jobs(
        state, structure_jobs(state.contract), gpu_ids=(0, 1), runtime=runtime,
        wall_deadline=wall_deadline, clock=clock,
    )
    state.structure_decision = select_structure(load_structure_evidence(state), state.contract)
    if state.structure_decision.status == "passed":
        run_pending_jobs(
            state, confirmation_jobs(state.structure_decision), gpu_ids=(0, 1),
            runtime=runtime, wall_deadline=wall_deadline, clock=clock,
        )
    state.acceptance = accept_t3(load_acceptance_evidence(state), state.contract)
    if state.acceptance.status == "accepted":
        state.delivery = full_fit_t3_from_state(state, gpu_ids=(0, 1), runtime=runtime)
    return publish_campaign_outputs(state)
```

Before starting a job, require at least 600 seconds before the wall deadline. After every terminal job and every 600 seconds, atomically replace the single stable resume. A failed job blocks only its own candidate identity.

- [ ] **Step 4: Add deadline and interrupted-worker tests**

```python
def test_runner_stops_before_new_job_guard(tmp_path):
    clock = FakeClock(now=10_000.0)
    runtime = FakeRuntime(gpu_count=2)
    result = run_t3_campaign(
        verified_fixture(), tmp_path, runtime=runtime,
        wall_deadline=10_599.0, clock=clock,
    )
    assert runtime.started_jobs == []
    assert result.resume_bundle.is_file()


def test_interrupted_campaign_resumes_without_retraining_completed_job(tmp_path):
    first_runtime = InterruptAfterCompletedJob(gpu_count=2, completed_count=1)
    first = run_t3_campaign(verified_fixture(), tmp_path / "first", runtime=first_runtime)
    second_runtime = FakeRuntime(gpu_count=2)
    second = run_t3_campaign(
        verified_fixture(), tmp_path / "second",
        resume_bundle=first.resume_bundle, runtime=second_runtime,
    )
    assert first_runtime.completed_job_id not in second_runtime.started_jobs
    assert second.review_bundle.is_file()


def test_partial_checkpoint_is_not_reused(tmp_path):
    resume = resume_with_unbound_checkpoint(tmp_path)
    with pytest.raises(T3ArtifactError, match="checkpoint binding differs"):
        run_t3_campaign(
            verified_fixture(), tmp_path / "run",
            resume_bundle=resume, runtime=FakeRuntime(gpu_count=2),
        )
```

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_runner.py
```

Expected: all runner tests pass with fake workers and clocks.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/t3_runner.py tests/test_tree_expert_t3_runner.py
git commit -m "feat(tree-expert): orchestrate restartable T3 campaign"
```

### Task 9: Generate the one-cell Kaggle handoff

**Files:**
- Create: `experiments/tree_expert/t3_kaggle.py`
- Create: `tools/build_tree_expert_t3_kaggle_cell.py`
- Generate: `experiments/tree_expert/KAGGLE_T3_CELL.py`
- Test: `tests/test_tree_expert_t3_kaggle.py`
- Test: `tests/test_tree_expert_t3_kaggle_cell.py`

- [ ] **Step 1: Write runtime-inventory and upload-discovery tests**

```python
def test_t3_runtime_archive_contains_every_imported_member():
    members = runtime_member_names()
    assert "experiments/tree_expert/t3_runner.py" in members
    assert "experiments/tree_expert/t3_inference.py" in members
    assert "experiments/tree_expert/features.py" in members
    assert "experiments/temporal_portfolio/seasonal_features.py" in members


def test_discovery_accepts_unpacked_input_and_at_most_one_resume(tmp_path):
    make_official_dataset(tmp_path)
    make_unpacked_t3_input(tmp_path)
    found = discover_t3_inputs(tmp_path)
    assert found.resume is None
    make_unpacked_t3_resume(tmp_path)
    assert discover_t3_inputs(tmp_path).resume is not None
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_kaggle.py tests/test_tree_expert_t3_kaggle_cell.py
```

Expected: missing-module failure.

- [ ] **Step 3: Implement deterministic runtime embedding and discovery**

The generated cell must:

1. decode the embedded runtime and verify its SHA-256;
2. find exactly one official dataset and one zipped or unpacked T3 input;
3. accept zero or one zipped or unpacked T3 resume;
4. verify CatBoost 1.2.10 and exactly two CUDA devices;
5. print `TREE_T3_JOB_START`, structured progress, checkpoint, decision, and success events;
6. keep only one stable emergency resume and request its download only on failure or session deadline;
7. download exactly one final handoff on success.

The final success line is exact:

```text
TREE_T3_SUCCESS handoff=/kaggle/working/tree_expert_t3/tree_expert_t3_handoff.zip
```

The error prefix is exact:

```text
TREE_T3_ERROR stage=<stage> type=<exception> message=<message>
```

- [ ] **Step 4: Generate twice and prove byte stability**

Run:

```bash
artifacts/tabm_submission_python311/bin/python tools/build_tree_expert_t3_kaggle_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_T3_CELL.py
artifacts/tabm_submission_python311/bin/python tools/build_tree_expert_t3_kaggle_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_T3_CELL.py
```

Expected: both hashes are identical.

- [ ] **Step 5: Run and commit**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_kaggle.py tests/test_tree_expert_t3_kaggle_cell.py
```

Expected: runtime, discovery, cell syntax, and deterministic-generation tests pass.

Commit:

```bash
git add experiments/tree_expert/t3_kaggle.py experiments/tree_expert/KAGGLE_T3_CELL.py tools/build_tree_expert_t3_kaggle_cell.py tests/test_tree_expert_t3_kaggle.py tests/test_tree_expert_t3_kaggle_cell.py
git commit -m "feat(tree-expert): add T3 Kaggle handoff cell"
```

### Task 10: Verify the complete local implementation and prepare the user run

**Files:**
- Modify: `reports/EXPERIMENT_LEDGER.md`
- Verify: all `tests/test_tree_expert_t3_*.py`

- [ ] **Step 1: Run the focused T3 suite**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_expert_t3_*.py
```

Expected: all T3 tests pass.

- [ ] **Step 2: Run regression tests for reused E2 components**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q \
  tests/test_tree_expert_features.py \
  tests/test_tree_expert_e2_*.py \
  tests/test_tree_expert_e2_submission_*.py
```

Expected: all selected regression tests pass.

- [ ] **Step 3: Compile every new runtime member**

```bash
artifacts/tabm_submission_python311/bin/python -m compileall -q \
  experiments/tree_expert tools/prepare_tree_expert_t3_input.py \
  tools/build_tree_expert_t3_kaggle_cell.py
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Build and verify the compact user input without training**

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tree_expert_t3_input.py \
  --e2-handoff "/Users/yonghyun/Downloads/tree_expert_e2_handoff (1).zip" \
  --output artifacts/tree_expert_t3_input.zip
```

Expected output begins with:

```text
TREE_T3_INPUT_READY path=artifacts/tree_expert_t3_input.zip sha256=
```

This is a small local archive operation, not a full-data preprocessing or GPU training run.

- [ ] **Step 5: Record pending status and commit**

Add this ledger row, replacing `<input-sha256>` with the hash printed in Step 4:

```markdown
| 18 | `tree_expert_t3_temporal_dual` | recent-season + decayed multi-season CatBoost | 2021→2022, 2022→2023 selection; 2023→2024 confirmation | pending user Kaggle T4×2 | prepared | — | contract `tree_expert_t3_temporal_dual_v1`, input `<input-sha256>` |
```

Commit:

```bash
git add reports/EXPERIMENT_LEDGER.md artifacts/tree_expert_t3_input.zip
git commit -m "docs: prepare T3 temporal campaign handoff"
```

If `artifacts/` is ignored, commit only the ledger; report the archive path and SHA-256 to the user without forcing it into Git.

- [ ] **Step 6: Hand off one complete Kaggle operation**

Tell the user to add only these Kaggle inputs:

- official `lg-aimers-9th-data`
- `tree_expert_t3_input`
- optional `tree_expert_t3_resume` only when resuming

They run `experiments/tree_expert/KAGGLE_T3_CELL.py` as one cell with T4×2. State:

- Purpose: select and confirm the temporal dual model, then full-fit only if accepted.
- Expected wall time: up to 8 hours for structure; up to 14 hours if confirmation and full fit run.
- Rerun safety: verified completed jobs are reused from the one resume bundle.
- Success return: `tree_expert_t3_handoff.zip` plus the final `TREE_T3_SUCCESS` line.
- Error return: the latest `tree_expert_t3_resume.zip` plus the first `TREE_T3_ERROR` traceback.

Do not build or provide a DACON submission ZIP at this task. The returned handoff must first prove candidate acceptance, all gates, row-independence, and exact current hashes. A rejected T3 candidate leaves the existing 977-point submission valid and does not block the later R/F experiment.
