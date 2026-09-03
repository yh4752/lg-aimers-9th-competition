# Tree Expert E1 Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline, task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restartable Kaggle T4 x2 E1 campaign that screens four independently implemented CatBoost tree-expert structures on the 2023→2024 fold and produces review/resume evidence without any submission package.

**Architecture:** Add a new `experiments/tree_expert` package instead of modifying the existing TabM, temporal-portfolio, or CatBoost-blend campaigns. Reuse only their public, tested cutoff-safe S1 and TrackMan builders, bind the exact deployed TabM F3 OOF as the comparison baseline, run each CatBoost candidate as an isolated GPU worker, and publish one verified E1 handoff ZIP. E2 confirmation, final fitting, and submission packaging remain separate implementation plans because their exact job set depends on the E1 decision.

**Tech Stack:** Python 3.11, pandas/numpy, CatBoost 1.2.10, pytest, ZIP/JSON/SHA-256 artifact contracts, Kaggle T4 x2

---

## Scope and file map

Create these focused production files:

- `experiments/tree_expert/__init__.py` — package marker and public exception exports.
- `experiments/tree_expert/e1_contract.json` — immutable E1 candidates, hashes, gates, and budget.
- `experiments/tree_expert/contracts.py` — strict contract parser and deterministic job planner.
- `experiments/tree_expert/inputs.py` — official-data hash verification and exact Stage C F3 baseline extraction.
- `experiments/tree_expert/features.py` — row-local categorical crosses, S1, anchor, and optional TrackMan composition.
- `experiments/tree_expert/failure_labels.py` — train-only failure-type recovery and audit gate.
- `experiments/tree_expert/training.py` — one isolated CatBoost fold job.
- `experiments/tree_expert/metrics.py` — exact OOF alignment, Brier/bootstrap/segments, and E1 decision.
- `experiments/tree_expert/artifacts.py` — deterministic review/resume/handoff writers and verifiers.
- `experiments/tree_expert/runner.py` — restartable two-GPU E1 scheduler.
- `experiments/tree_expert/kaggle.py` — runtime inventory, embedded launcher, Kaggle input discovery.
- `experiments/tree_expert/KAGGLE_E1_CELL.py` — generated single-cell launcher under 1MB.
- `experiments/tree_expert/requirements-kaggle.txt` — only `catboost==1.2.10`.
- `tools/prepare_tree_expert_e1_input.py` — seal the trusted Stage C delivery into a small E1 input.
- `tools/build_tree_expert_e1_kaggle_cell.py` — regenerate the deterministic cell.
- `docs/TREE_EXPERT_E1_KAGGLE.md` — user-facing one-cell runbook.

Create these tests:

- `tests/test_tree_expert_contracts.py`
- `tests/test_tree_expert_inputs.py`
- `tests/test_tree_expert_features.py`
- `tests/test_tree_expert_failure_labels.py`
- `tests/test_tree_expert_training.py`
- `tests/test_tree_expert_metrics.py`
- `tests/test_tree_expert_artifacts.py`
- `tests/test_tree_expert_runner.py`
- `tests/test_tree_expert_kaggle.py`
- `tests/test_tree_expert_kaggle_cell.py`

Do not modify the user's dirty `experiments/tabm_campaign`, related tests/tools, or
`notebooks/INDEPENDENT_DL_RECOVERY_AND_TABR.ipynb`.

### Task 1: Seal the E1 contract and job schedule

**Files:**
- Create: `experiments/tree_expert/__init__.py`
- Create: `experiments/tree_expert/e1_contract.json`
- Create: `experiments/tree_expert/contracts.py`
- Test: `tests/test_tree_expert_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
from dataclasses import FrozenInstanceError
import json

import pytest

from experiments.tree_expert.contracts import (
    TreeExpertContractError,
    build_e1_jobs,
    load_e1_contract,
)


def test_e1_contract_is_review_only_and_has_four_fixed_candidates():
    contract = load_e1_contract()
    assert contract.review_only is True
    assert contract.submission_package is False
    assert contract.fold == (2023, 2024)
    assert contract.seed == 3407
    assert tuple(job.candidate_id for job in build_e1_jobs(contract)) == (
        "c0_native_ctr",
        "c1_anchor_residual",
        "c2_trackman_residual",
        "c3_failure_aware",
    )
    with pytest.raises(FrozenInstanceError):
        contract.seed = 1


def test_e1_contract_rejects_one_changed_gate(tmp_path):
    source = json.loads(
        __import__("pathlib").Path(
            "experiments/tree_expert/e1_contract.json"
        ).read_text()
    )
    source["gates"]["stop_if_all_regress_more_than"] = 0.5
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(source))
    with pytest.raises(TreeExpertContractError, match="gates differ"):
        load_e1_contract(path)
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
pytest -q tests/test_tree_expert_contracts.py
```

Expected: collection fails with `ModuleNotFoundError: experiments.tree_expert`.

- [ ] **Step 3: Create the exact JSON contract**

Use these policy values in `e1_contract.json`:

```json
{
  "schema_version": 1,
  "campaign_id": "tree_expert_e1_v1",
  "review_only": true,
  "submission_package": false,
  "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
  "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
  "stage_c_delivery_sha256": "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a",
  "stage_c_review_sha256": "461c4422cebdc7383577ad6c53b044bfe3d9d15b667383e454591135e4242d2c",
  "baseline_predictor": "single_s3407",
  "fold": {"train_end_year": 2023, "valid_year": 2024},
  "seed": 3407,
  "candidates": [
    {"candidate_id": "c0_native_ctr", "objective": "binary", "use_trackman": false, "use_failure_labels": false},
    {"candidate_id": "c1_anchor_residual", "objective": "residual", "use_trackman": false, "use_failure_labels": false},
    {"candidate_id": "c2_trackman_residual", "objective": "residual", "use_trackman": true, "use_failure_labels": false},
    {"candidate_id": "c3_failure_aware", "objective": "multiclass", "use_trackman": false, "use_failure_labels": true}
  ],
  "catboost": {
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
    "allow_writing_files": true
  },
  "failure_label_gate": {
    "minimum_coverage": 0.98,
    "minimum_binary_delta_fraction": 0.999,
    "minimum_success_agreement": 0.999,
    "maximum_middle_reverse_overlap": 0.001,
    "minimum_class_rows": 10000,
    "delta_tolerance": 0.02
  },
  "trackman_gate": {
    "minimum_accepted_coverage": 0.30
  },
  "metrics": {
    "bootstrap_repeats": 1000,
    "bootstrap_seed": 3407,
    "minimum_segment_rows": 5000
  },
  "gates": {
    "stop_if_all_regress_more_than": 0.00005,
    "maximum_promoted": 2
  },
  "budget": {
    "wall_seconds": 14400,
    "new_job_guard_seconds": 600,
    "snapshot_interval_seconds": 300
  }
}
```

- [ ] **Step 4: Implement strict frozen parsing and deterministic jobs**

The public shape in `contracts.py` must be:

```python
@dataclass(frozen=True)
class E1Candidate:
    candidate_id: str
    objective: str
    use_trackman: bool
    use_failure_labels: bool


@dataclass(frozen=True)
class E1Job:
    job_id: str
    candidate_id: str
    train_end_year: int
    valid_year: int
    seed: int
    objective: str
    use_trackman: bool
    use_failure_labels: bool


@dataclass(frozen=True)
class E1Contract:
    campaign_id: str
    review_only: bool
    submission_package: bool
    official_train_sha256: str
    official_history_sha256: str
    stage_c_delivery_sha256: str
    stage_c_review_sha256: str
    baseline_predictor: str
    fold: tuple[int, int]
    seed: int
    candidates: tuple[E1Candidate, ...]
    catboost_parameters: Mapping[str, object]
    failure_label_gate: Mapping[str, object]
    trackman_gate: Mapping[str, object]
    bootstrap_repeats: int
    bootstrap_seed: int
    minimum_segment_rows: int
    stop_if_all_regress_more_than: float
    maximum_promoted: int
    wall_seconds: int
    new_job_guard_seconds: int
    snapshot_interval_seconds: int


def build_e1_jobs(contract: E1Contract) -> tuple[E1Job, ...]:
    train_end, valid = contract.fold
    return tuple(
        E1Job(
            job_id=f"e1__{item.candidate_id}__tr{train_end}__va{valid}__s{contract.seed}",
            candidate_id=item.candidate_id,
            train_end_year=train_end,
            valid_year=valid,
            seed=contract.seed,
            objective=item.objective,
            use_trackman=item.use_trackman,
            use_failure_labels=item.use_failure_labels,
        )
        for item in contract.candidates
    )
```

Validate exact top-level and nested key sets, exact policy values, lowercase 64-character hashes, exact booleans rather than truthy values, and wrap nested dictionaries in `MappingProxyType`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest -q tests/test_tree_expert_contracts.py
git diff --check
```

Expected: all contract tests pass and `git diff --check` prints nothing.

Commit:

```bash
git add experiments/tree_expert/__init__.py experiments/tree_expert/e1_contract.json experiments/tree_expert/contracts.py tests/test_tree_expert_contracts.py
git commit -m "feat: seal tree expert E1 contract"
```

### Task 2: Verify official inputs and bind the exact F3 TabM baseline

**Files:**
- Create: `experiments/tree_expert/inputs.py`
- Create: `tools/prepare_tree_expert_e1_input.py`
- Test: `tests/test_tree_expert_inputs.py`

- [ ] **Step 1: Write failing input tests**

```python
def test_official_data_requires_exact_top_level_train_and_history(fixture_data, contract):
    verified = verify_official_data(fixture_data, contract, testing=True)
    assert verified.train.name == "train.csv"
    assert verified.history.name == "trackman_history.csv"
    assert verified.train_sha256 == file_sha256(verified.train)


def test_e1_input_extracts_only_bound_2023_to_2024_single_seed(tmp_path, stage_c_delivery, contract):
    archive = prepare_e1_input(stage_c_delivery, tmp_path / "input.zip", contract)
    verified = verify_and_extract_e1_input(archive, tmp_path / "verified", contract)
    assert verified.baseline_fold == "2023->2024"
    frame = __import__("pandas").read_csv(verified.baseline_predictions)
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert frame["row_id"].is_unique


def test_e1_input_rejects_same_named_unbound_delivery(tmp_path, stage_c_delivery, contract):
    changed = mutate_nested_review_prediction(stage_c_delivery, tmp_path)
    with pytest.raises(TreeExpertInputError, match="Stage C delivery SHA-256 differs"):
        prepare_e1_input(changed, tmp_path / "input.zip", contract)
```

- [ ] **Step 2: Verify failure**

Run `pytest -q tests/test_tree_expert_inputs.py`.

Expected: import failure for `experiments.tree_expert.inputs`.

- [ ] **Step 3: Implement trusted input dataclasses and exact extraction**

Use this public interface:

```python
PREDICTION_COLUMNS = (
    "row_id", "target", "probability", "game_type", "game_month",
    "pitcher_id_known", "batter_id_known",
)


@dataclass(frozen=True)
class VerifiedOfficialData:
    root: Path
    train: Path
    history: Path
    train_sha256: str
    history_sha256: str


@dataclass(frozen=True)
class VerifiedE1Input:
    archive_sha256: str
    manifest_sha256: str
    stage_c_delivery_sha256: str
    stage_c_review_sha256: str
    baseline_fold: str
    baseline_predictions: Path
```

`prepare_e1_input` must accept exactly the Stage C delivery SHA from the contract, verify its nested review with the existing `experiments.tabm_campaign.artifacts.verify_review_bundle`, extract only the `single_s3407` F3 prediction member into a deterministic ZIP, and write a manifest containing its original nested member path and all hashes. Never include `test.csv`, `sample_submission.csv`, a model, or a submission file.

`verify_official_data` must walk without following symlinks, require exactly one top-level `train.csv` and `trackman_history.csv`, and verify the contract hashes in real runs. The explicit `testing=True` switch exists only for synthetic tests and must reject use unless the supplied root resolves below the system temporary directory; do not inspect environment variables.

- [ ] **Step 4: Implement the preparation CLI**

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-c-delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = prepare_e1_input(
        args.stage_c_delivery, args.output, load_e1_contract()
    )
    print(
        f"TREE_E1_INPUT_READY path={output.resolve()} "
        f"sha256={file_sha256(output)} size_bytes={output.stat().st_size}",
        flush=True,
    )
    return 0
```

- [ ] **Step 5: Test and commit**

Run:

```bash
pytest -q tests/test_tree_expert_inputs.py
git diff --check
```

Expected: PASS.

Commit only the four Task 2 files with message `feat: bind tree expert E1 inputs`.

### Task 3: Build row-local CTR, S1, and anchor features

**Files:**
- Create: `experiments/tree_expert/features.py`
- Test: `tests/test_tree_expert_features.py`

- [ ] **Step 1: Write row-independence and cutoff tests**

```python
def test_tree_features_are_invariant_to_validation_neighbors(train_prefix, valid_rows):
    state, train = fit_tree_features(
        train_prefix, history=None, valid_year=2024, use_trackman=False
    )
    whole = transform_tree_features(valid_rows, state)
    for position in (0, len(valid_rows) // 2, len(valid_rows) - 1):
        one = transform_tree_features(valid_rows.iloc[[position]], state)
        pd.testing.assert_frame_equal(
            one.frame.reset_index(drop=True),
            whole.frame.iloc[[position]].reset_index(drop=True),
        )


def test_tree_features_have_fixed_high_cardinality_crosses(train_prefix):
    state, batch = fit_tree_features(
        train_prefix, history=None, valid_year=2024, use_trackman=False
    )
    assert {
        "pitcher_batter_hand", "pitcher_count", "pitcher_base",
        "pitcher_game_type", "batter_pitcher_hand", "team_count",
    } <= set(state.categorical_columns)
    assert np.isfinite(batch.anchor).all()
    assert np.all((batch.anchor > 0) & (batch.anchor < 1))


def test_training_s1_never_reads_validation_target(train_prefix, valid_rows):
    state, _ = fit_tree_features(
        train_prefix, history=None, valid_year=2024, use_trackman=False
    )
    with pytest.raises(TreeFeatureError, match="evaluation rows contain target"):
        transform_tree_features(valid_rows.assign(control_success=1), state)
```

- [ ] **Step 2: Verify the tests fail**

Run `pytest -q tests/test_tree_expert_features.py`.

Expected: missing module or missing functions.

- [ ] **Step 3: Implement immutable state and feature batches**

Use the existing tested APIs `build_training_s1`, `fit_s1_state`, and `transform_s1` from `experiments.temporal_portfolio.seasonal_features`; do not copy their math.

```python
@dataclass(frozen=True)
class TreeFeatureState:
    valid_year: int
    prior_rate: float
    categorical_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    s1_state: S1State
    trackman_state: PitcherTrackmanState | None
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class TreeFeatureBatch:
    frame: pd.DataFrame
    anchor: np.ndarray
    row_id: np.ndarray
    target: np.ndarray | None
```

Create one `_row_local_features` function that copies the supplied rows, drops only `row_id` and `control_success`, and adds exactly the crosses in the design. Use fixed bins:

```python
inning_bin = pd.cut(inning, [-np.inf, 3, 6, 9, np.inf], labels=["early", "middle", "late", "extra"])
score_bin = pd.cut(score, [-np.inf, -3, -1, 1, 3, np.inf], labels=["far_behind", "behind", "close", "ahead", "far_ahead"])
leverage_bin = pd.cut(li, [-np.inf, 0.75, 1.5, 3, np.inf], labels=["low", "normal", "high", "extreme"])
```

Define the anchor only from the fold training prior and row-local/S1 values:

```python
n = numeric["asof_pitcher_n"].clip(lower=0)
career = numeric["asof_pitcher_success_rate"].fillna(prior)
recent = numeric[[
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]].mean(axis=1).fillna(prior)
season_n = s1["season_pitcher_n"].clip(lower=0)
season_rate = s1["season_pitcher_success_smooth_100"]
career_weight = n / (n + 100.0)
season_weight = 0.15 + 0.30 * season_n / (season_n + 80.0)
career_anchor = prior + career_weight * (career - prior)
anchor = np.clip(
    career_anchor + season_weight * (season_rate - career_anchor)
    + 0.10 * (recent - career_anchor),
    1e-5, 1 - 1e-5,
)
```

Ensure validation transforms reject `control_success`, preserve row order/index, use only frozen state, and produce the exact same columns/dtypes as training.

- [ ] **Step 4: Run focused and existing feature tests**

Run:

```bash
pytest -q tests/test_tree_expert_features.py tests/test_temporal_portfolio_features.py tests/test_temporal_portfolio_trackman.py
```

Expected: PASS without changing existing modules.

- [ ] **Step 5: Commit**

Commit the new feature module and its tests with message `feat: build tree expert row features`.

### Task 4: Audit and recover failure-type labels from train only

**Files:**
- Create: `experiments/tree_expert/failure_labels.py`
- Test: `tests/test_tree_expert_failure_labels.py`

- [ ] **Step 1: Write exact arithmetic tests**

```python
def test_failure_recovery_uses_next_train_cumulative_state_only():
    rows = failure_fixture()  # four pitches for one pitcher plus terminal row
    audit = audit_failure_labels(rows, gate=gate())
    assert audit.status == "passed"
    assert audit.labels.tolist() == ["success", "middle", "reverse", "other_failure"]
    assert audit.source_positions.tolist() == [0, 1, 2, 3]


def test_failure_recovery_rejects_validation_rows():
    with pytest.raises(FailureLabelError, match="rows reach validation season"):
        audit_failure_labels(
            failure_fixture().assign(season=2024), gate=gate(), valid_year=2024
        )


def test_failed_label_gate_skips_only_c3():
    result = audit_failure_labels(low_coverage_fixture(), gate=gate())
    assert result.status == "skipped_unreliable_labels"
    assert result.labels.size == 0
```

- [ ] **Step 2: Verify failure**

Run `pytest -q tests/test_tree_expert_failure_labels.py`.

- [ ] **Step 3: Implement deterministic recovery**

Use original input position as a stable tiebreaker, group only official training rows by `pitcher_id`, and sort by `asof_pitcher_n` then position. Link a row only when the next same-pitcher cumulative count equals `n + 1` within tolerance.

```python
success_delta = next_n * next_success_rate - n * success_rate
middle_delta = next_n * next_middle_rate - n * middle_rate
reverse_delta = next_n * next_reverse_rate - n * reverse_rate
```

Snap a delta to zero or one only when its distance is at most `delta_tolerance`. Require the snapped success delta to equal `control_success`; label failures as `middle`, `reverse`, or `other_failure`. Rows with invalid, overlapping, skipped-count, or non-binary deltas are excluded and counted in the audit.

Return an immutable result:

```python
@dataclass(frozen=True)
class FailureLabelAudit:
    status: str
    reason: str
    labels: np.ndarray
    source_positions: np.ndarray
    coverage: float
    binary_delta_fraction: float
    success_agreement: float
    middle_reverse_overlap: float
    class_counts: Mapping[str, int]
```

Do not derive labels from TrackMan, validation targets, test rows, or the competitor ZIP.

- [ ] **Step 4: Test and commit**

Run `pytest -q tests/test_tree_expert_failure_labels.py` and commit with message `feat: audit train-only failure labels`.

### Task 5: Add cutoff-bound TrackMan composition and its candidate-local gate

**Files:**
- Modify: `experiments/tree_expert/features.py`
- Modify: `tests/test_tree_expert_features.py`

- [ ] **Step 1: Add failing TrackMan tests**

```python
def test_trackman_candidate_uses_only_cutoff_history(train_prefix, history):
    state, first = fit_tree_features(
        train_prefix, history=history, valid_year=2024, use_trackman=True
    )
    changed = history.copy()
    changed.loc[changed["season"].eq(2024), "rel_speed"] = 999.0
    replay_state, replay = fit_tree_features(
        train_prefix, history=changed, valid_year=2024, use_trackman=True
    )
    assert state.source_hashes == replay_state.source_hashes
    pd.testing.assert_frame_equal(first.frame, replay.frame)


def test_low_trackman_coverage_skips_only_c2(train_prefix, sparse_history):
    with pytest.raises(TreeFeatureSkip, match="trackman_coverage"):
        fit_tree_features(
            train_prefix, history=sparse_history, valid_year=2024,
            use_trackman=True, minimum_trackman_coverage=0.30,
        )
```

- [ ] **Step 2: Verify failure**

Run the two tests by node ID and expect failure.

- [ ] **Step 3: Reuse the immutable TrackMan state**

Call `fit_pitcher_trackman(train, history, cutoff_year=valid_year - 1)`. Attach its P0, P1, P2, and P3 bundle columns by mapping the current row's `pitcher_id`; do not perform a many-row evaluation merge or derive an evaluation-set coverage statistic. Compute the gate from the frozen training lookup:

```python
lookup = state.lookup
accepted = pd.to_numeric(lookup["tm_match_accepted"], errors="coerce").fillna(0)
coverage = float(accepted.eq(1).mean())
if coverage < minimum_trackman_coverage:
    raise TreeFeatureSkip(f"trackman_coverage={coverage:.6f}")
```

Add `tm_pitcher_mapping_missing` and keep all missing numeric TrackMan values as NaN for CatBoost native missing handling.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
pytest -q tests/test_tree_expert_features.py tests/test_temporal_portfolio_trackman.py
```

Commit with message `feat: attach cutoff-safe TrackMan tree features`.

### Task 6: Train one isolated CatBoost candidate

**Files:**
- Create: `experiments/tree_expert/training.py`
- Create: `experiments/tree_expert/requirements-kaggle.txt`
- Test: `tests/test_tree_expert_training.py`

- [ ] **Step 1: Write fake-model tests for all objectives**

```python
@pytest.mark.parametrize(
    ("candidate", "expected_mode"),
    [
        ("c0_native_ctr", "binary"),
        ("c1_anchor_residual", "residual"),
        ("c2_trackman_residual", "residual"),
        ("c3_failure_aware", "multiclass"),
    ],
)
def test_fold_worker_writes_valid_probability_for_each_objective(
    tmp_path, candidate, expected_mode, e1_fixture
):
    result = run_e1_job(
        job=e1_fixture.job(candidate),
        data=e1_fixture.data,
        baseline=e1_fixture.baseline,
        output_dir=tmp_path / candidate,
        absolute_deadline=time.time() + 60,
        gpu_id=0,
        model_factory=FakeCatBoostFactory(),
    )
    assert result.status == "completed"
    frame = pd.read_csv(result.predictions_path)
    assert frame["probability"].between(0, 1).all()
    assert json.loads((tmp_path / candidate / "metrics.json").read_text())["objective"] == expected_mode


def test_skippable_candidate_does_not_fail_worker(tmp_path, e1_fixture):
    def skipped_builder(*args, **kwargs):
        raise TreeFeatureSkip("trackman_coverage=0.100000")

    result = run_e1_job(
        job=e1_fixture.job("c2_trackman_residual"),
        data=e1_fixture.data,
        baseline=e1_fixture.baseline,
        output_dir=tmp_path / "c2_trackman_residual",
        absolute_deadline=time.time() + 60,
        gpu_id=0,
        model_factory=FakeCatBoostFactory(),
        feature_builder=skipped_builder,
    )
    assert result.status == "skipped"
    assert result.model_path is None
```

- [ ] **Step 2: Verify failure**

Run `pytest -q tests/test_tree_expert_training.py`.

- [ ] **Step 3: Implement objective-specific CatBoost calls**

Create `FoldResult(job_id, candidate_id, status, brier, model_path, predictions_path, snapshot_path, failure)`. Keep the worker dependency-injection boundary explicit so synthetic tests do not import or train real CatBoost:

```python
def run_e1_job(
    *,
    job: E1Job,
    data: VerifiedOfficialData,
    baseline: pd.DataFrame,
    output_dir: Path,
    absolute_deadline: float,
    gpu_id: int,
    contract: E1Contract | None = None,
    model_factory: CatBoostFactory | None = None,
    feature_builder: Callable[..., tuple[TreeFeatureState, TreeFeatureBatch]] = fit_tree_features,
) -> FoldResult:
```

For C0:

```python
model = CatBoostClassifier(
    **common, loss_function="Logloss", eval_metric="Logloss",
    random_seed=job.seed, devices=str(gpu_id),
)
model.fit(
    x_train,
    y_train,
    cat_features=cat_columns,
    eval_set=(x_valid, y_valid),
    use_best_model=True,
    early_stopping_rounds=60,
    save_snapshot=True,
    snapshot_file=str(snapshot_path),
    snapshot_interval=contract.snapshot_interval_seconds,
    verbose=50,
)
probability = model.predict_proba(x_valid)[:, 1]
```

For C1/C2:

```python
residual = y_train.astype("float64") - train_batch.anchor
model = CatBoostRegressor(
    **common, loss_function="RMSE", eval_metric="RMSE",
    random_seed=job.seed, devices=str(gpu_id),
)
model.fit(
    x_train,
    residual,
    cat_features=cat_columns,
    eval_set=(x_valid, y_valid.astype("float64") - valid_batch.anchor),
    use_best_model=True,
    early_stopping_rounds=60,
    save_snapshot=True,
    snapshot_file=str(snapshot_path),
    snapshot_interval=contract.snapshot_interval_seconds,
    verbose=50,
)
probability = np.clip(valid_batch.anchor + model.predict(x_valid), 1e-5, 1 - 1e-5)
```

For C3, subset the training feature rows by `source_positions`, map classes in fixed order `success=0`, `middle=1`, `reverse=2`, `other_failure=3`, fit `CatBoostClassifier(loss_function="MultiClass")`, and use column 0 as `P(success)`.

Every completed job must atomically write:

- `job.json`
- `worker.log`
- `worker_result.json`
- `metrics.json`
- `predictions.csv`
- `model.cbm`
- `experiment.cbsnapshot` when CatBoost produced one

Prediction columns must be exactly `PREDICTION_COLUMNS`. Calculate `pitcher_id_known` and `batter_id_known` from training-only ID sets, not validation frequency.

- [ ] **Step 4: Add the one-line dependency file**

```text
catboost==1.2.10
```

Do not include pandas, numpy, sklearn, torch, or Kaggle packages.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest -q tests/test_tree_expert_training.py tests/test_tree_expert_features.py tests/test_tree_expert_failure_labels.py
```

Commit with message `feat: train tree expert E1 candidates`.

### Task 7: Evaluate exact OOF evidence and select at most two structures

**Files:**
- Create: `experiments/tree_expert/metrics.py`
- Test: `tests/test_tree_expert_metrics.py`

- [ ] **Step 1: Write metric and decision tests**

```python
def test_e1_metrics_reject_row_reordering(baseline, candidate):
    with pytest.raises(TreeMetricError, match="row_id alignment differs"):
        evaluate_e1_candidate(baseline, candidate.iloc[::-1], contract())


def test_e1_decision_promotes_best_two_noncatastrophic_candidates(metric_fixture):
    decision = decide_e1([
        metric_fixture("c0_native_ctr", gain=0.00020),
        metric_fixture("c1_anchor_residual", gain=0.00010),
        metric_fixture("c2_trackman_residual", gain=-0.00004),
        metric_fixture("c3_failure_aware", status="skipped"),
    ], contract())
    assert decision.status == "completed"
    assert decision.promoted == ("c0_native_ctr", "c1_anchor_residual")


def test_e1_decision_stops_when_every_candidate_regresses_too_much(metric_fixture):
    decision = decide_e1([
        metric_fixture("c0_native_ctr", gain=-0.00006),
        metric_fixture("c1_anchor_residual", gain=-0.00010),
    ], contract())
    assert decision.status == "rejected"
    assert decision.promoted == ()
```

- [ ] **Step 2: Verify failure**

Run `pytest -q tests/test_tree_expert_metrics.py`.

- [ ] **Step 3: Implement metrics**

Require exact equality of `row_id`, `target`, and diagnostic columns. Compute:

- baseline and candidate Brier
- `gain = baseline_brier - candidate_brier`
- prediction and residual correlation
- 1000 pitcher-block bootstrap gains with seed 3407
- segment Brier for `game_type`, `game_month`, `pitcher_id_known`, `batter_id_known`
- maximum candidate-vs-baseline regression among segments with at least 5000 rows

Selection order is:

```python
eligible = [m for m in metrics if m.status == "completed"]
if eligible and all(m.gain < -contract.stop_if_all_regress_more_than for m in eligible):
    return E1Decision("rejected", (), "all_candidates_regressed_more_than_gate")
ranked = sorted(
    eligible,
    key=lambda m: (-m.gain, -m.bootstrap_lower, m.maximum_segment_regression, m.candidate_id),
)
return E1Decision("completed", tuple(m.candidate_id for m in ranked[:2]), "top_structures_selected")
```

Skipped C2/C3 do not count as failed and cannot be promoted.

- [ ] **Step 4: Test and commit**

Run `pytest -q tests/test_tree_expert_metrics.py` and commit with message `feat: decide tree expert E1 screening`.

### Task 8: Publish deterministic review, resume, and one handoff

**Files:**
- Create: `experiments/tree_expert/artifacts.py`
- Test: `tests/test_tree_expert_artifacts.py`

- [ ] **Step 1: Write archive safety tests**

```python
def test_e1_resume_round_trip_preserves_completed_and_skipped_jobs(tmp_path, campaign_fixture):
    bundles = write_e1_bundles(
        output_dir=tmp_path / "bundles",
        contract_path=campaign_fixture.contract_path,
        bindings=campaign_fixture.bindings,
        state_path=campaign_fixture.state_path,
        log_path=campaign_fixture.log_path,
        job_directories=campaign_fixture.job_directories,
        decision_path=campaign_fixture.decision_path,
        audit_paths=campaign_fixture.audit_paths,
    )
    verified = verify_e1_resume(bundles.resume, campaign_fixture.bindings)
    assert verified.completed == (
        "e1__c0_native_ctr__tr2023__va2024__s3407",
    )
    assert verified.skipped == (
        "e1__c3_failure_aware__tr2023__va2024__s3407",
    )
    assert verified.submission_package is False


def test_rejected_e1_never_contains_delivery_or_submission(tmp_path, campaign_fixture):
    campaign_fixture.set_decision_status("rejected")
    bundles = write_e1_bundles(
        output_dir=tmp_path / "bundles",
        contract_path=campaign_fixture.contract_path,
        bindings=campaign_fixture.bindings,
        state_path=campaign_fixture.state_path,
        log_path=campaign_fixture.log_path,
        job_directories=campaign_fixture.job_directories,
        decision_path=campaign_fixture.decision_path,
        audit_paths=campaign_fixture.audit_paths,
    )
    assert bundles.review.is_file()
    assert bundles.resume.is_file()
    with ZipFile(bundles.review) as archive:
        assert not any("submission" in name or "delivery" in name for name in archive.namelist())


@pytest.mark.parametrize("unsafe", ["../x", "/x", "a/../../x"])
def test_e1_verifier_rejects_unsafe_members(tmp_path, unsafe):
    archive = malicious_zip(tmp_path, unsafe)
    with pytest.raises(TreeArtifactError, match="unsafe"):
        verify_e1_resume(archive, expected_bindings())
```

- [ ] **Step 2: Verify failure**

Run `pytest -q tests/test_tree_expert_artifacts.py`.

- [ ] **Step 3: Implement fail-closed bundle contracts**

Bindings must contain exactly:

```python
{
    "contract_sha256", "code_sha256", "input_manifest_sha256",
    "train_sha256", "history_sha256", "stage_c_delivery_sha256",
    "stage_c_review_sha256", "baseline_predictions_sha256",
}
```

Write archives atomically with canonical JSON, fixed ZIP timestamp `(2026, 1, 1, 0, 0, 0)`, no duplicate member, no symlink, per-member size/ratio limits, exact allowlists, and post-write verification before `os.replace`.

Resume includes completed job model/snapshot artifacts, skipped evidence, state, contract, and audit files. Review includes decision, metrics, predictions, label/TrackMan audits, log, state, and contract but no model. Handoff contains exactly:

```text
handoff_manifest.json
tree_expert_e1_review.zip
tree_expert_e1_resume.zip
tree_expert_e1.log
```

Every manifest must contain `review_only: true` and `submission_package: false`.

- [ ] **Step 4: Test and commit**

Run `pytest -q tests/test_tree_expert_artifacts.py` and commit with message `feat: seal tree expert E1 artifacts`.

### Task 9: Run E1 restartably across two GPUs

**Files:**
- Create: `experiments/tree_expert/runner.py`
- Test: `tests/test_tree_expert_runner.py`

- [ ] **Step 1: Write scheduler and resume tests**

```python
def test_runner_assigns_at_most_one_job_per_gpu(tmp_path, e1_context):
    runtime = RecordingRuntime()
    result = run_e1_campaign(
        verified_data=e1_context.verified_data,
        verified_input=e1_context.verified_input,
        output_dir=tmp_path / "first",
        resume_bundle=None,
        absolute_deadline=time.time() + 3600,
        gpu_ids=(0, 1),
        runtime=runtime,
    )
    assert runtime.maximum_concurrent == 2
    assert set(runtime.gpus_used) == {0, 1}
    assert len(result.completed) + len(result.skipped) == 4


def test_runner_reuses_completed_jobs_after_resume(tmp_path, e1_context):
    first = run_e1_campaign(
        verified_data=e1_context.verified_data,
        verified_input=e1_context.verified_input,
        output_dir=tmp_path / "first",
        resume_bundle=None,
        absolute_deadline=time.time() + 3600,
        gpu_ids=(0, 1),
        runtime=StopAfterTwoRuntime(),
    )
    second = run_e1_campaign(
        verified_data=e1_context.verified_data,
        verified_input=e1_context.verified_input,
        output_dir=tmp_path / "second",
        resume_bundle=first.bundles.resume,
        absolute_deadline=time.time() + 3600,
        gpu_ids=(0, 1),
        runtime=RecordingRuntime(),
    )
    assert second.reused == first.completed
    assert not set(second.executed).intersection(second.reused)


def test_runner_stops_starting_jobs_inside_guard(tmp_path, e1_context, fake_clock):
    result = run_e1_campaign(
        verified_data=e1_context.verified_data,
        verified_input=e1_context.verified_input,
        output_dir=tmp_path / "guard",
        resume_bundle=None,
        absolute_deadline=fake_clock.now + 599,
        gpu_ids=(0, 1),
        runtime=RecordingRuntime(),
        clock=fake_clock,
    )
    assert result.status == "budget_inconclusive"
    assert result.bundles.resume.is_file()
```

- [ ] **Step 2: Verify failure**

Run `pytest -q tests/test_tree_expert_runner.py`.

- [ ] **Step 3: Implement job-level parallelism**

The public orchestration boundary must accept a clock for deterministic guard tests while production defaults to `time.time`:

```python
def run_e1_campaign(
    *,
    verified_data: VerifiedOfficialData,
    verified_input: VerifiedE1Input,
    output_dir: Path,
    resume_bundle: Path | None,
    absolute_deadline: float,
    gpu_ids: tuple[int, int],
    runtime: CampaignRuntime | None = None,
    clock: Callable[[], float] = time.time,
) -> E1CampaignResult:
```

The parent process verifies inputs, creates immutable bindings, restores a verified resume, and maintains this state:

```json
{
  "schema_version": 1,
  "campaign_id": "tree_expert_e1_v1",
  "status": "running",
  "completed": [],
  "skipped": [],
  "failed": [],
  "active": {},
  "bindings": {}
}
```

Use one worker process per job and pass its explicit GPU ID. Never allow two active jobs to share one ID. After each terminal worker result, verify the job directory before updating state and republishing resume. Candidate-local `skipped` is terminal but not a campaign failure. A real failed worker stops new jobs, waits for active jobs, publishes the latest verified resume, and returns `failed`.

After all four candidates are completed or skipped, call `decide_e1`, publish review/resume, and return `completed` or `rejected`. No code path may call final fit, delivery, inference on test, or submission packaging.

Required logs:

The concrete prefix sequence is `TREE_E1_INPUTS_VERIFIED`,
`TREE_E1_GPU_READY device_count=2`, `TREE_E1_JOB_START`,
`TREE_E1_JOB_END`, `TREE_E1_DECISION`, and `TREE_E1_HANDOFF_READY`. Each line adds
the exact candidate, GPU, status, path, or SHA-256 as named key-value fields.

- [ ] **Step 4: Test and commit**

Run `pytest -q tests/test_tree_expert_runner.py tests/test_tree_expert_artifacts.py tests/test_tree_expert_training.py` and commit with message `feat: orchestrate tree expert E1 screening`.

### Task 10: Generate a self-contained Kaggle cell below 1MB

**Files:**
- Create: `experiments/tree_expert/kaggle.py`
- Create: `tools/build_tree_expert_e1_kaggle_cell.py`
- Generate: `experiments/tree_expert/KAGGLE_E1_CELL.py`
- Test: `tests/test_tree_expert_kaggle.py`
- Test: `tests/test_tree_expert_kaggle_cell.py`

- [ ] **Step 1: Write runtime and launcher tests**

```python
def test_runtime_inventory_contains_only_declared_tree_expert_and_reused_modules():
    names = runtime_member_names()
    assert "experiments/tree_expert/runner.py" in names
    assert "experiments/temporal_portfolio/seasonal_features.py" in names
    assert "experiments/temporal_portfolio/trackman_pitcher.py" in names
    assert all("submission" not in name for name in names)


def test_generated_kaggle_cell_is_small_and_has_one_final_handoff(tmp_path):
    output = build_e1_kaggle_cell(tmp_path / "cell.py")
    text = output.read_text()
    assert output.stat().st_size < 1_000_000
    assert "TREE_E1_HANDOFF_READY" in text
    assert "tree_expert_e1_handoff.zip" in text
    assert "files.download" not in text


def test_kaggle_input_discovery_accepts_expanded_dataset_and_zip(tmp_path):
    assert classify_e1_input(expanded_e1_input(tmp_path)) == "tree_expert_e1_input_v1"
    assert classify_e1_input(zipped_e1_input(tmp_path)) == "tree_expert_e1_input_v1"
```

- [ ] **Step 2: Verify failure**

Run `pytest -q tests/test_tree_expert_kaggle.py tests/test_tree_expert_kaggle_cell.py`.

- [ ] **Step 3: Implement deterministic runtime inventory**

List exact source members rather than recursively archiving the repository. Include the new E1 modules and only the imported public temporal/independent-DL preprocessing modules. Hash member name and bytes in sorted order. The builder must fail if a declared source is missing or the rendered source reaches 1MB.

- [ ] **Step 4: Implement the one-cell launcher**

The generated cell must:

1. decode and safely extract the embedded runtime;
2. import CatBoost and install only `catboost==1.2.10` if absent/wrong;
3. find exactly one Kaggle data root containing top-level `train.csv` and `trackman_history.csv`;
4. find exactly one expanded or zipped `tree_expert_e1_input_v1`;
5. find zero or one `tree_expert_e1_resume_v1` or prior E1 handoff;
6. verify both inputs before reporting GPU readiness;
7. require exactly two CUDA devices for the planned run;
8. call `run_e1_campaign` with `time.time() + 14400`;
9. build one `/kaggle/working/tree_expert_e1_handoff.zip`;
10. print its path and SHA-256 without invoking repeated browser downloads.

Errors must end with:

```python
print(
    f"TREE_EXPERT_ERROR stage={stage} type={type(error).__name__} "
    f"message={str(error).replace(' ', '_')}",
    flush=True,
)
```

On failure after verified input, publish one emergency handoff if and only if at least one verified terminal job exists.

- [ ] **Step 5: Build, test, and commit**

Run:

```bash
python tools/build_tree_expert_e1_kaggle_cell.py
pytest -q tests/test_tree_expert_kaggle.py tests/test_tree_expert_kaggle_cell.py
python -m py_compile experiments/tree_expert/KAGGLE_E1_CELL.py
```

Expected: generated cell below 1MB and all tests pass.

Commit with message `feat: launch tree expert E1 on Kaggle`.

### Task 11: Write the exact user handoff runbook

**Files:**
- Create: `docs/TREE_EXPERT_E1_KAGGLE.md`
- Test: `tests/test_tree_expert_kaggle_cell.py`

- [ ] **Step 1: Add a documentation-contract test**

```python
def test_e1_runbook_names_inputs_outputs_runtime_and_return_logs():
    text = Path("docs/TREE_EXPERT_E1_KAGGLE.md").read_text()
    for required in (
        "lg-aimers-9th-data",
        "tree_expert_e1_input",
        "tree_expert_e1_handoff.zip",
        "3~4시간",
        "재실행",
        "TREE_E1_HANDOFF_READY",
        "TREE_EXPERT_ERROR",
    ):
        assert required in text
```

- [ ] **Step 2: Verify failure**

Run the test and expect failure because the runbook is absent.

- [ ] **Step 3: Write the runbook**

It must give one exact local preparation command:

```bash
cd /path/to/lg-aimers-9th-competition
artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tree_expert_e1_input.py \
  --stage-c-delivery /path/to/Downloads/tabm_colab_stage_C_delivery.zip \
  --output artifacts/tree_expert_e1_input.zip
```

Then explain how to add `lg-aimers-9th-data` and the expanded E1 input as Kaggle datasets, select T4 x2, paste the single generated cell, and use Save Version. State:

- purpose: latest-fold structure screening only;
- required inputs: official dataset + E1 input + optional resume;
- expected output: one handoff ZIP;
- approximate runtime: 3–4 hours;
- rerun safety: completed/skipped jobs are reused only when hashes match;
- success text to return: the full `TREE_E1_HANDOFF_READY path=<absolute-path> sha256=<sha256>` line;
- error text to return: the full final `TREE_EXPERT_ERROR stage=<stage> type=<type> message=<message>` line plus the emergency handoff path if present;
- the handoff is not a DACON submission.

- [ ] **Step 4: Test and commit**

Run `pytest -q tests/test_tree_expert_kaggle_cell.py` and commit with message `docs: add tree expert E1 Kaggle runbook`.

### Task 12: Run the complete local verification gate

**Files:**
- Modify only if verification reveals an E1 defect: files created in Tasks 1–11

- [ ] **Step 1: Run focused E1 tests**

```bash
pytest -q \
  tests/test_tree_expert_contracts.py \
  tests/test_tree_expert_inputs.py \
  tests/test_tree_expert_features.py \
  tests/test_tree_expert_failure_labels.py \
  tests/test_tree_expert_training.py \
  tests/test_tree_expert_metrics.py \
  tests/test_tree_expert_artifacts.py \
  tests/test_tree_expert_runner.py \
  tests/test_tree_expert_kaggle.py \
  tests/test_tree_expert_kaggle_cell.py
```

Expected: all E1 tests pass.

- [ ] **Step 2: Run reused-component regressions**

```bash
pytest -q \
  tests/test_temporal_portfolio_features.py \
  tests/test_temporal_portfolio_trackman.py \
  tests/test_tabm_campaign_artifacts.py \
  tests/test_catboost_tabm_blend_inputs.py
```

Expected: all pass without modifying existing components.

- [ ] **Step 3: Run static checks**

```bash
python -m compileall -q experiments/tree_expert tools/prepare_tree_expert_e1_input.py tools/build_tree_expert_e1_kaggle_cell.py
git diff --check
rg -n "test\.csv|sample_submission|submission\.csv|package_submission|files\.download" experiments/tree_expert tools/prepare_tree_expert_e1_input.py
```

Expected:

- compile succeeds;
- `git diff --check` is empty;
- the search returns no E1 training/runtime submission path; test assertions may contain prohibited strings only when explicitly verifying absence.

- [ ] **Step 4: Rebuild and compare the cell**

```bash
cp experiments/tree_expert/KAGGLE_E1_CELL.py /tmp/tree_expert_e1_cell.before.py
python tools/build_tree_expert_e1_kaggle_cell.py
cmp /tmp/tree_expert_e1_cell.before.py experiments/tree_expert/KAGGLE_E1_CELL.py
wc -c experiments/tree_expert/KAGGLE_E1_CELL.py
```

Expected: `cmp` succeeds and size is below `1000000`.

- [ ] **Step 5: Inspect the final diff and commit any verification-only fix**

```bash
git status --short
git diff --stat
git diff --check
```

Do not stage or commit the pre-existing dirty TabM/notebook files. If Tasks 1–11 already left a clean E1 diff, no extra commit is needed. If a verification defect required a fix, commit only the E1 files with message `fix: close tree expert E1 verification gaps`.

## Deferred dependent plans

Do not implement these before the stated evidence exists:

1. **E2 confirmation and full-fit plan** — write only after `tree_expert_e1_handoff.zip` verifies and contains one or two promoted candidates. Its contract must bind the E1 review SHA and exact promoted IDs.
2. **Local final-audit and submission plan** — write only after E2 status is `accepted`, the delivery manifest and current artifact hashes pass, and the 480-second inference gate has evidence. This is required by the repository's fail-closed submission contract.

No E1 task creates a submission ZIP, full-data model, test prediction, or automatic DACON upload.
