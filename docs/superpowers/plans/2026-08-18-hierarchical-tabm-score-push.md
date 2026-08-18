# Hierarchical TabM Score Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rule-safe, resumable Colab campaign that tests one hierarchical-context TabM and its two preregistered calibration variants, then emits candidate evidence only for variants that pass their own temporal and segment gates.

**Architecture:** Add an isolated `experiments/hierarchical_tabm` package. It fits a train-only hierarchical context encoder, injects eight numeric columns before the existing `dl_standard + hand_matchup` preprocessing, reuses the current P2 TabM trainer for two temporal OOF folds, and evaluates H1/H2/H3 with separately sealed gates. A deterministic one-cell Colab supervisor verifies two required uploads plus an optional resume, checkpoints every epoch, and can produce review/resume/candidate-delivery archives; no module in this plan reads evaluation data or creates a submission package.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, PyTorch, TabM 0.0.3, rtdl-num-embeddings 0.0.12, standard-library JSON/ZIP/hash/subprocess utilities, pytest with synthetic fixtures and injected lightweight runtimes.

---

## Execution boundary

Codex may write and review code, run import/static checks, and run only synthetic
or fixture-based tests. The user owns all official-data reads, package installation,
GPU OOF, full training, Colab runtime, evaluation inference, and leaderboard
submissions.

This plan stops after a verified
`hierarchical_tabm_candidate_delivery.zip`. That file is research evidence, not a
DACON submission. No task may modify `submission/`, call a submission builder,
read `test.csv`, or create `submit.zip`. A later packaging plan is allowed only
after at least one candidate has passed its sealed gate and its returned hashes
have been verified.

The following currently modified user files are outside this plan and must stay
untouched:

```text
experiments/tabm_campaign/COLAB_ROW_FEATURE_PROXY_CELL.py
experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py
experiments/tabm_campaign/KAGGLE_CELL.py
experiments/tabm_campaign/colab_recovery.py
experiments/tabm_campaign/colab_stage_c_contract.json
experiments/tabm_campaign/row_feature_proxy.py
tests/test_tabm_campaign_colab_cell.py
tests/test_tabm_campaign_colab_recovery.py
tests/test_tabm_row_feature_colab.py
tests/test_tabm_row_feature_proxy.py
tools/prepare_tabm_colab_stage_c_handoff.py
tools/prepare_tabm_row_feature_colab_input.py
notebooks/INDEPENDENT_DL_RECOVERY_AND_TABR.ipynb
```

## Locked experiment policy

The implementation must encode these values in `contract.json` and reject any
mutation rather than silently using defaults.

```text
campaign_id: hierarchical_tabm_score_push_v1
official train SHA-256: d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff
official history SHA-256: f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9
Stage C delivery SHA-256: f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a
development fold: 2021 -> 2022, CPU-only K selection
OOF folds: 2022 -> 2023, 2023 -> 2024
K candidates: 32, 128, 512
K tie tolerance: 0.000001; choose larger K inside the tie
pitcher reliability K: 100
batter reliability K: 250
model: TabM P2, k=32, width=512, blocks=4, dropout=0.1
numeric embedding: piecewise_linear
loss: bce
scheduler: plateau
learning rate: 0.0006
weight decay: 0.0001
effective batch: 4096
micro batch: 512
seed: 3407
OOF maximum epochs: 40
OOF minimum epochs: 3
OOF patience: 10
final epoch clamp: 2..8
calibration regularization: 0.0001, 0.001, 0.01, 0.1
probability clip: 0.000001
segment hard-gate minimum rows: 5000
Colab session: 10800 seconds
new-job guard: 900 seconds
snapshot interval: 300 seconds
download interval: 1200 seconds
Python inference runtime: 3.11.15
inference limit: 480 seconds
GPU allocation limit: 21474836480 bytes
RSS limit: 22000000000 bytes
candidate artifact limit: 2000000000 bytes
H1 strong: weighted gain >=0.00010, latest gain >=0.00005,
           old-fold regression <=0.00015, eligible segment regression <=0.00075
H1 frontier: weighted and latest gains >0, old-fold regression <=0.00025,
             and not H1 strong
H2: latest gain vs H1 >=0.00003, latest gain vs anchor >=0.00008,
    eligible segment regression vs H1 <=0.00020
H3: all H2 gates and latest gain vs H2 >=0.00002
```

H1, H2, and H3 retain separate decisions. A failure or incomplete result for one
candidate must not rewrite another candidate's status. `incomplete` is not a
performance rejection.

## File map

```text
experiments/hierarchical_tabm/
├── __init__.py                       # package marker, no eager imports
├── contract.json                     # immutable hypotheses, inputs, model and gates
├── contracts.py                     # strict parser and job definitions
├── context_features.py              # hierarchy fit/LOO/frozen transform/state JSON
├── feature_adapter.py               # hierarchy + existing DL preprocessing bridge
├── calibration.py                   # H2/H3 fitting, transform and state JSON
├── metrics.py                       # alignment, Brier, segments and H1/H2/H3 gates
├── inputs.py                        # content-based upload classification and extraction
├── training.py                      # one OOF/full TabM worker and checkpoint CLI
├── artifacts.py                     # review/resume/candidate-delivery write and verify
├── runner.py                        # K -> OOF -> calibration -> decision -> full-fit state machine
├── colab.py                         # subprocess supervision, recovery and download cadence
├── inference.py                     # frozen H1/H2/H3 predictor and row-independence audit
├── runtime_inventory.py             # exact embedded source/dependency identity
├── requirements-colab.txt           # exact Colab dependency pins
└── COLAB_HIERARCHICAL_TABM_CELL.py  # generated one-cell user handoff

tools/build_hierarchical_tabm_colab_cell.py
docs/HIERARCHICAL_TABM_COLAB.md
tests/test_hierarchical_tabm_contracts.py
tests/test_hierarchical_tabm_context_features.py
tests/test_hierarchical_tabm_feature_adapter.py
tests/test_hierarchical_tabm_calibration.py
tests/test_hierarchical_tabm_metrics.py
tests/test_hierarchical_tabm_inputs.py
tests/test_hierarchical_tabm_training.py
tests/test_hierarchical_tabm_artifacts.py
tests/test_hierarchical_tabm_runner.py
tests/test_hierarchical_tabm_colab.py
tests/test_hierarchical_tabm_inference.py
tests/test_hierarchical_tabm_colab_cell.py
```

The new package may import stable computation primitives from
`experiments.independent_dl` and content verifiers from
`experiments.catboost_tabm_blend`. It must not modify those packages. Runtime
identity must include every imported local source and both dependency files.

### Task 1: Seal the experiment contract and job order

**Files:**
- Create: `experiments/hierarchical_tabm/__init__.py`
- Create: `experiments/hierarchical_tabm/contract.json`
- Create: `experiments/hierarchical_tabm/contracts.py`
- Create: `tests/test_hierarchical_tabm_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

Use exact assertions, not subset assertions:

```python
def test_contract_seals_three_hypotheses_and_current_tabm() -> None:
    contract = load_contract()
    assert contract.campaign_id == "hierarchical_tabm_score_push_v1"
    assert contract.k_candidates == (32.0, 128.0, 512.0)
    assert contract.development_fold == Fold(2021, 2022)
    assert contract.oof_folds == (Fold(2022, 2023), Fold(2023, 2024))
    assert contract.candidate_ids == ("H1", "H2", "H3")
    assert contract.model == ModelPolicy(
        k=32, width=512, blocks=4, dropout=0.1,
        num_embedding="piecewise_linear", loss="bce",
        scheduler="plateau", learning_rate=0.0006,
        weight_decay=0.0001, effective_batch_size=4096,
        micro_batch_size=512, seed=3407,
    )


def test_contract_builds_only_two_oof_jobs_then_one_full_job() -> None:
    jobs = build_jobs(load_contract())
    assert [(job.kind, job.train_end_year, job.valid_year) for job in jobs] == [
        ("oof", 2022, 2023),
        ("oof", 2023, 2024),
        ("full_fit", 2024, None),
    ]
```

Parametrize mutations of every exact key, source SHA, hierarchy column, K,
model value, gate boundary, calibration grid, budget, and list order. Require
`HierarchicalContractError`. Reject duplicate JSON keys, booleans as integers,
non-finite numbers, unknown keys, uppercase SHA text, and reordered folds.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_contracts.py -q
```

Expected: collection fails with
`ModuleNotFoundError: experiments.hierarchical_tabm`.

- [ ] **Step 3: Implement the strict contract types and parser**

The public interface is fixed as follows:

```python
@dataclass(frozen=True)
class Fold:
    train_end_year: int
    valid_year: int


@dataclass(frozen=True)
class ModelPolicy:
    k: int
    width: int
    blocks: int
    dropout: float
    num_embedding: str
    loss: str
    scheduler: str
    learning_rate: float
    weight_decay: float
    effective_batch_size: int
    micro_batch_size: int
    seed: int


@dataclass(frozen=True)
class HierarchicalJob:
    job_id: str
    kind: str
    train_end_year: int
    valid_year: int | None


@dataclass(frozen=True)
class HierarchicalContract:
    schema_version: int
    campaign_id: str
    review_only: bool
    submission_package: bool
    official_train_sha256: str
    official_history_sha256: str
    source_stage_c_delivery_sha256: str
    development_fold: Fold
    oof_folds: tuple[Fold, ...]
    candidate_ids: tuple[str, ...]
    k_candidates: tuple[float, ...]
    k_tie_tolerance: float
    pitcher_reliability_k: float
    batter_reliability_k: float
    hierarchy_columns: tuple[str, ...]
    model: ModelPolicy
    max_epochs: int
    min_epochs: int
    patience: int
    final_min_epochs: int
    final_max_epochs: int
    calibration_grid: tuple[float, ...]
    probability_clip: float
    segment_min_rows: int
    gates: Mapping[str, Mapping[str, float]]
    inference_limits: Mapping[str, int | str]
    session_seconds: int
    new_job_guard_seconds: int
    snapshot_interval_seconds: int
    download_interval_seconds: int


def load_contract(path: Path = DEFAULT_CONTRACT) -> HierarchicalContract: ...
def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str: ...
def build_jobs(contract: HierarchicalContract) -> tuple[HierarchicalJob, ...]: ...
```

Use exact-key validation at every nesting level and `MappingProxyType` for gate
mappings. `contract_sha256` must validate before hashing exact bytes.

- [ ] **Step 4: Run GREEN checks**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_contracts.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/contracts.py
git diff --check
```

- [ ] **Step 5: Commit Task 1**

```bash
git add experiments/hierarchical_tabm/__init__.py \
  experiments/hierarchical_tabm/contract.json \
  experiments/hierarchical_tabm/contracts.py \
  tests/test_hierarchical_tabm_contracts.py
git commit -m "feat: seal hierarchical TabM contract"
```

### Task 2: Fit and transform the hierarchical context state

**Files:**
- Create: `experiments/hierarchical_tabm/context_features.py`
- Create: `tests/test_hierarchical_tabm_context_features.py`

- [ ] **Step 1: Write hand-calculated hierarchy tests**

Build an eight-row frame containing repeated and unseen paths. Assert the exact
global, count, hand, base-out, and game-type values for K=2 so that the smoothing
equation is visible in the test:

```python
def test_leave_one_out_uses_parent_and_excludes_own_target() -> None:
    state = fit_context_state(rows, smoothing_k=2.0)
    transformed = transform_training_loo(rows, state)
    expected_first = (1.0 + 2.0 * ((3.0 + 2.0 * (4.0 / 8.0)) / 9.0)) / 5.0
    assert transformed.loc[0, "hier_context_rate"] == pytest.approx(expected_first)
    flipped = rows.copy()
    flipped.loc[0, "control_success"] = 1 - flipped.loc[0, "control_success"]
    changed = transform_training_loo(flipped, fit_context_state(flipped, smoothing_k=2.0))
    assert changed.loc[0, "hier_context_rate"] == pytest.approx(
        transformed.loc[0, "hier_context_rate"]
    )


def test_frozen_transform_backs_off_one_level_at_a_time() -> None:
    state = fit_context_state(rows, smoothing_k=2.0)
    output = transform_frozen(unseen_rows, state)
    assert output["hier_context_rate"].tolist() == pytest.approx([
        expected_base_out_parent,
        expected_hand_parent,
        expected_count_parent,
        state.global_rate,
    ])
```

Also cover duplicate/non-null `row_id`, binary targets, balls 0..3, strikes
0..2, outs 0..2, normalized `base_state`, missing hand/game type markers,
invalid numeric input, empty frames, and the single-row LOO fallback to the
contract-safe 0.5 prior.

- [ ] **Step 2: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_context_features.py -q
```

Expected: import fails because `context_features.py` is absent.

- [ ] **Step 3: Implement immutable state and exact hierarchy traversal**

Use these structures and functions:

```python
HIERARCHY_LEVELS = (
    ("balls_before", "strikes_before"),
    ("balls_before", "strikes_before", "pitcher_hand", "batter_hand"),
    ("balls_before", "strikes_before", "pitcher_hand", "batter_hand",
     "base_state", "outs_before"),
    ("balls_before", "strikes_before", "pitcher_hand", "batter_hand",
     "base_state", "outs_before", "game_type"),
)


@dataclass(frozen=True)
class LevelState:
    columns: tuple[str, ...]
    counts: Mapping[tuple[str, ...], int]
    successes: Mapping[tuple[str, ...], float]
    rates: Mapping[tuple[str, ...], float]


@dataclass(frozen=True)
class ContextState:
    schema_version: int
    smoothing_k: float
    row_count: int
    target_sum: float
    global_rate: float
    levels: tuple[LevelState, ...]


def fit_context_state(frame: pd.DataFrame, *, smoothing_k: float) -> ContextState: ...
def transform_training_loo(frame: pd.DataFrame, state: ContextState) -> pd.DataFrame: ...
def transform_frozen(frame: pd.DataFrame, state: ContextState) -> pd.DataFrame: ...
def context_state_payload(state: ContextState) -> dict[str, object]: ...
def context_state_from_payload(payload: Mapping[str, object]) -> ContextState: ...
def context_state_sha256(state: ContextState) -> str: ...
```

`fit_context_state` may aggregate only the provided labeled training frame.
`transform_training_loo` performs leave-one-out and recomputes every node
numerator and denominator after
subtracting that row's target and count. It must not call `transform_frozen` for
training rows. Encode tuple keys as canonical JSON arrays sorted lexicographically;
never use `repr(tuple)` as persisted identity.

- [ ] **Step 4: Add target-blindness and state round-trip tests**

```python
def test_frozen_transform_never_reads_target() -> None:
    state = fit_context_state(rows, smoothing_k=32.0)
    without = transform_frozen(valid.drop(columns="control_success"), state)
    with_changed = valid.assign(control_success=1 - valid.control_success)
    observed = transform_frozen(with_changed, state)
    pd.testing.assert_frame_equal(without, observed)


def test_context_state_round_trip_is_canonical() -> None:
    state = fit_context_state(rows, smoothing_k=32.0)
    restored = context_state_from_payload(context_state_payload(state))
    assert context_state_payload(restored) == context_state_payload(state)
    assert context_state_sha256(restored) == context_state_sha256(state)
```

Reject missing/extra/reordered levels, duplicate encoded keys, inconsistent
counts/sums/rates, non-finite values, and a state whose child rate does not equal
the stored parent-smoothed value.

- [ ] **Step 5: Run GREEN and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_context_features.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/context_features.py
git diff --check
git add experiments/hierarchical_tabm/context_features.py \
  tests/test_hierarchical_tabm_context_features.py
git commit -m "feat: add train-only hierarchical context state"
```

### Task 3: Build the eight TabM inputs without changing existing preprocessing

**Files:**
- Create: `experiments/hierarchical_tabm/feature_adapter.py`
- Create: `tests/test_hierarchical_tabm_feature_adapter.py`

- [ ] **Step 1: Write failing exact-column and arithmetic tests**

```python
EXPECTED = (
    "hier_context_rate", "hier_context_logit",
    "hier_pitcher_reliability", "hier_batter_reliability",
    "hier_pitcher_context_gap", "hier_batter_context_gap",
    "hier_pitcher_weighted_gap", "hier_batter_weighted_gap",
)


def test_adds_exact_hierarchy_columns_in_order() -> None:
    output = attach_hierarchical_numeric(rows, context_rate, pitcher_k=100.0, batter_k=250.0)
    assert tuple(output.columns[-8:]) == EXPECTED
    assert output.loc[0, "hier_pitcher_reliability"] == pytest.approx(100 / 200)
    assert output.loc[0, "hier_batter_reliability"] == pytest.approx(80 / 330)
    assert output.loc[0, "hier_pitcher_weighted_gap"] == pytest.approx(
        (rows.loc[0, "asof_pitcher_success_rate"] - context_rate[0]) * 0.5
    )
```

Require nonnegative counts, source success rates in `[0,1]` when present,
finite output, logit clipping at `1e-6`, row/index preservation, and output
collision rejection.

- [ ] **Step 2: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_feature_adapter.py -q
```

- [ ] **Step 3: Implement the preprocessing bridge**

```python
@dataclass(frozen=True)
class HierarchicalFeatureState:
    context: ContextState
    preprocessing_payload: Mapping[str, object]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_maps: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class PreparedFold:
    train: FeatureBatch
    valid: FeatureBatch
    state: HierarchicalFeatureState
    metadata: ModelMetadata


def attach_hierarchical_numeric(
    frame: pd.DataFrame,
    context_rate: Sequence[float],
    *,
    pitcher_k: float,
    batter_k: float,
) -> pd.DataFrame: ...


def prepare_fold(
    fit_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    *,
    context_state: ContextState,
    pitcher_k: float,
    batter_k: float,
) -> PreparedFold: ...


def transform_with_state(frame: pd.DataFrame, state: HierarchicalFeatureState) -> FeatureBatch: ...
def feature_state_payload(state: HierarchicalFeatureState) -> dict[str, object]: ...
def feature_state_from_payload(payload: Mapping[str, object]) -> HierarchicalFeatureState: ...
def feature_state_sha256(state: HierarchicalFeatureState) -> str: ...
```

`prepare_fold` must attach LOO rates to `fit_rows`, frozen rates to
`valid_rows`, then call the existing public `fit_preprocessor` and
`transform_preprocessor` with `PreprocessingSpec("dl_standard",
("hand_matchup",))`. Build category maps from training output only and map
unseen categories to 0. Reimplement only the small `FeatureBatch` assembly
needed by this package; do not modify or monkeypatch
`experiments.independent_dl.features`. The feature-state serializer must include
the context payload, exact preprocessing state, ordered numeric/categorical
columns and training-only category maps. It must reject altered means, scales,
medians, category indices, output order or context digest.

- [ ] **Step 4: Add row-independence and preprocessing parity tests**

```python
def test_frozen_feature_batch_is_order_and_batch_independent() -> None:
    prepared = prepare_fold(train, valid, context_state=state, pitcher_k=100, batter_k=250)
    baseline = by_row_id(transform_with_state(valid, prepared.state))
    assert by_row_id(transform_with_state(valid.iloc[::-1], prepared.state)) == baseline
    assert merge_batches(valid, prepared.state, batch_size=1) == baseline
    assert merge_batches(valid, prepared.state, batch_size=257) == baseline


def test_existing_hand_matchup_is_still_present_once() -> None:
    prepared = prepare_fold(train, valid, context_state=state, pitcher_k=100, batter_k=250)
    assert prepared.state.categorical_columns.count("hand_matchup") == 1
```

Also assert that changing another validation row, validation row count, or
validation target cannot change a fixed row's features, and that the fitted
state digest remains unchanged.

- [ ] **Step 5: Run GREEN with preprocessing regressions and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_feature_adapter.py \
  tests/test_preprocessing_profiles.py \
  tests/test_independent_dl_training.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/feature_adapter.py
git diff --check
git add experiments/hierarchical_tabm/feature_adapter.py \
  tests/test_hierarchical_tabm_feature_adapter.py
git commit -m "feat: bridge hierarchical features into TabM"
```

### Task 4: Select K on the CPU development fold

**Files:**
- Modify: `experiments/hierarchical_tabm/context_features.py`
- Create: `tests/test_hierarchical_tabm_k_selection.py`

- [ ] **Step 1: Write failing development-only selection tests**

```python
def test_selects_lowest_development_brier() -> None:
    result = select_smoothing_k(train_2021, valid_2022, (32.0, 128.0, 512.0), tie_tolerance=1e-6)
    assert result.selected_k == 128.0
    assert result.fold == "2021->2022"


def test_tie_within_tolerance_selects_larger_k() -> None:
    selected = select_k_from_scores(
        {32.0: 0.2500000, 128.0: 0.2500008, 512.0: 0.2500020},
        tie_tolerance=1e-6,
    )
    assert selected == 128.0
```

Verify row-weighted mean Brier, exact candidate order, finite binary target,
train seasons `<=2021`, validation season `==2022`, and target-blind frozen
validation transform. Reject any frame containing 2023/2024 in this API.

- [ ] **Step 2: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_k_selection.py -q
```

Expected: import or attribute failure for `select_smoothing_k`.

- [ ] **Step 3: Implement the pure selector**

```python
@dataclass(frozen=True)
class KSelection:
    fold: str
    selected_k: float
    scores: Mapping[float, float]
    row_count: int


def select_k_from_scores(scores: Mapping[float, float], *, tie_tolerance: float) -> float: ...


def select_smoothing_k(
    fit_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    candidates: tuple[float, ...],
    *,
    tie_tolerance: float,
) -> KSelection: ...
```

For each K, fit on `fit_rows`, transform `valid_rows` without target access, and
compute float64 Brier against a separately extracted target vector. Sort
eligible tied candidates by K descending. Persist every candidate score.

- [ ] **Step 4: Run GREEN and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_context_features.py \
  tests/test_hierarchical_tabm_k_selection.py -q
git diff --check
git add experiments/hierarchical_tabm/context_features.py \
  tests/test_hierarchical_tabm_k_selection.py
git commit -m "feat: select hierarchy smoothing on development fold"
```

### Task 5: Fit and serialize H2/H3 calibration

**Files:**
- Create: `experiments/hierarchical_tabm/calibration.py`
- Create: `tests/test_hierarchical_tabm_calibration.py`

- [ ] **Step 1: Write failing H2 tests**

```python
def test_affine_logit_recovers_known_parameters() -> None:
    state = fit_h2(probability, target, regularization=1e-4, clip=1e-6)
    calibrated = apply_calibration(probability, segments, state)
    assert state.kind == "H2"
    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 1e-6) & (calibrated <= 1 - 1e-6)).all()
    assert brier(target, calibrated) < brier(target, probability)


def test_h2_penalty_shrinks_to_identity() -> None:
    weak = fit_h2(probability, target, regularization=1e-4, clip=1e-6)
    strong = fit_h2(probability, target, regularization=0.1, clip=1e-6)
    assert abs(strong.bias) <= abs(weak.bias)
    assert abs(strong.slope - 1.0) <= abs(weak.slope - 1.0)
```

- [ ] **Step 2: Write failing H3 and time-split tests**

```python
def test_h3_uses_only_preregistered_main_effects() -> None:
    state = fit_h3(probability, target, segment_frame, regularization=0.001, clip=1e-6)
    assert tuple(state.effects) == ("game_type", "count_state", "hand_matchup", "base_out_state")
    assert "pitcher_id" not in canonical_state_json(state).decode()


def test_regularization_selection_never_fits_on_late_months() -> None:
    chosen = select_calibration(
        oof_2023, kind="H3", grid=(1e-4, 1e-3, 1e-2, 1e-1),
        fit_month_max=7, validation_month_min=8, optimizer=fake_optimizer,
    )
    assert fake_optimizer.fit_row_ids == set(oof_2023.loc[oof_2023.game_month <= 7, "row_id"])
    assert chosen.selection_row_ids == tuple(oof_2023.loc[oof_2023.game_month >= 8, "row_id"])
```

Cover unseen category offset=0, sorted canonical levels, duplicate/missing
`row_id`, target mismatch, months outside 1..12, non-finite optimizer output,
regularization tie choosing the larger value, and refitting the chosen value on
all 2023 OOF rows.

- [ ] **Step 3: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_calibration.py -q
```

- [ ] **Step 4: Implement exact objectives and state**

```python
@dataclass(frozen=True)
class CalibrationState:
    schema_version: int
    kind: str
    regularization: float
    clip: float
    bias: float
    slope: float
    effects: Mapping[str, Mapping[str, float]]
    fit_row_ids_sha256: str


@dataclass(frozen=True)
class CalibrationSelection:
    kind: str
    selected_regularization: float
    validation_brier: Mapping[float, float]
    state: CalibrationState
    selection_row_ids: tuple[str, ...]


def fit_h2(probability: np.ndarray, target: np.ndarray, *, regularization: float, clip: float) -> CalibrationState: ...
def fit_h3(probability: np.ndarray, target: np.ndarray, segments: pd.DataFrame, *, regularization: float, clip: float) -> CalibrationState: ...
def select_calibration(oof_2023: pd.DataFrame, *, kind: str, grid: tuple[float, ...], fit_month_max: int, validation_month_min: int, optimizer: Callable = scipy.optimize.minimize) -> CalibrationSelection: ...
def apply_calibration(probability: np.ndarray, segments: pd.DataFrame, state: CalibrationState) -> np.ndarray: ...
def calibration_state_payload(state: CalibrationState) -> dict[str, object]: ...
def calibration_state_from_payload(payload: Mapping[str, object]) -> CalibrationState: ...
```

Use `sigmoid(bias + slope * clipped_logit + sum(main_effect))`. The fitting
objective is mean float64 Brier plus
`lambda * (bias**2 + (slope-1)**2 + mean(all_offset**2))`; omit the offset term
for H2. Select lambda by pure late-month Brier, with a `1e-12` tie choosing the
larger lambda. L2 makes the all-level representation identifiable; do not add a
reference-category heuristic. The final persisted state is refitted with the
chosen lambda on all supplied 2023 OOF rows.

- [ ] **Step 5: Run GREEN and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_calibration.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/calibration.py
git diff --check
git add experiments/hierarchical_tabm/calibration.py \
  tests/test_hierarchical_tabm_calibration.py
git commit -m "feat: add temporal H2 and H3 calibration"
```

### Task 6: Compute paired fold and segment decisions

**Files:**
- Create: `experiments/hierarchical_tabm/metrics.py`
- Create: `tests/test_hierarchical_tabm_metrics.py`

- [ ] **Step 1: Write failing alignment and segment tests**

```python
def test_aligns_only_exact_row_id_and_target_sets() -> None:
    aligned = align_anchor_and_h1(anchor, h1)
    assert aligned["row_id"].tolist() == anchor["row_id"].tolist()
    with pytest.raises(HierarchicalMetricError, match="target differs"):
        align_anchor_and_h1(anchor, h1.assign(target=1 - h1.target))


def test_builds_exact_preregistered_segments() -> None:
    labels = build_segment_columns(valid, fit_pitcher_ids={"p1"}, fit_batter_ids={"b1"})
    assert tuple(labels) == (
        "game_type", "count_state", "hand_matchup", "base_out_state",
        "pitcher_known", "batter_known",
    )
    assert labels.loc[0, "pitcher_known"] in {"known", "oov"}
```

Changing validation order, batch size, or another row must not change segment
labels for a fixed row. Known/OOV sets come only from the fold fit IDs.

- [ ] **Step 2: Write all gate boundary tests**

```python
def test_h1_strong_exact_boundary_passes() -> None:
    decision = decide_h1(metrics_at(
        weighted_gain=0.00010, latest_gain=0.00005,
        old_regression=0.00015, worst_segment_regression=0.00075,
    ), contract)
    assert decision.status == "strong"


def test_h1_frontier_is_not_final_acceptance() -> None:
    decision = decide_h1(metrics_at(
        weighted_gain=0.00001, latest_gain=0.00001,
        old_regression=0.00020, worst_segment_regression=0.00010,
    ), contract)
    assert decision.status == "frontier"
    assert decision.final_acceptance is False
    assert decision.delivery_role == "public_diagnostic_only"


def test_h2_can_pass_when_raw_h1_is_not_strong() -> None:
    decision = decide_calibrated("H2", h1_failed, calibrated_metrics, contract)
    assert decision.status == "accepted"
```

Test every exact H1 strong/frontier boundary, H2 latest-vs-H1 and
latest-vs-anchor boundary, H3 improvement-vs-H2 boundary, segment minimum row
count 5000, H3/H2 simplicity tie, independent candidate failure, and
`incomplete` propagation.

- [ ] **Step 3: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_metrics.py -q
```

- [ ] **Step 4: Implement pure metric and decision APIs**

```python
@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    status: str
    final_acceptance: bool
    delivery_role: str | None
    fold_brier: Mapping[str, float]
    fold_gain_vs_anchor: Mapping[str, float]
    weighted_gain_vs_anchor: float
    worst_eligible_segment_regression: float
    reason: str


def brier(target: Sequence[float], probability: Sequence[float]) -> float: ...
def build_segment_columns(frame: pd.DataFrame, *, fit_pitcher_ids: set[str], fit_batter_ids: set[str]) -> pd.DataFrame: ...
def paired_fold_metrics(anchor_by_fold: Mapping[str, pd.DataFrame], candidate_by_fold: Mapping[str, pd.DataFrame], *, segment_min_rows: int) -> Mapping[str, object]: ...
def decide_h1(metrics: Mapping[str, object], contract: HierarchicalContract) -> CandidateDecision: ...
def decide_calibrated(candidate_id: str, h1_metrics: Mapping[str, object], calibrated_metrics: Mapping[str, object], contract: HierarchicalContract, *, h2_metrics: Mapping[str, object] | None = None) -> CandidateDecision: ...
def candidate_decision_payload(decision: CandidateDecision) -> dict[str, object]: ...
```

All calculations use paired rows and float64. Record small segments but exclude
them from hard maxima. Frontier H1 uses
`delivery_role="public_diagnostic_only"`, may trigger the shared full fit, and
may appear in candidate delivery, but `final_acceptance` remains false. It still
requires separate human acceptance before a later submission-packaging task,
and at most one frontier Public diagnostic is allowed in the round. H2/H3
decisions use the locked 2024 fold as the independent gate; do not present
calibration-fitted 2023 performance as independent evidence.

- [ ] **Step 5: Run GREEN and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_metrics.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/metrics.py
git diff --check
git add experiments/hierarchical_tabm/metrics.py \
  tests/test_hierarchical_tabm_metrics.py
git commit -m "feat: decide hierarchical TabM candidates"
```

### Task 7: Verify the two required uploads and optional resume

**Files:**
- Create: `experiments/hierarchical_tabm/inputs.py`
- Create: `tests/test_hierarchical_tabm_inputs.py`

- [ ] **Step 1: Write failing content-based classification tests**

```python
def test_classifies_renamed_required_uploads_by_exact_members(tmp_path: Path) -> None:
    training, stage_c = make_required_uploads(
        tmp_path, training_name="anything.zip", stage_c_name="other.zip"
    )
    assert classify_upload(training) == "training_input"
    assert classify_upload(stage_c) == "stage_c_delivery"


def test_rejects_test_or_submission_members(tmp_path: Path) -> None:
    bad = write_zip(tmp_path / "bad.zip", {"test.csv": b"x"})
    with pytest.raises(HierarchicalInputError, match="unknown upload"):
        classify_upload(bad)
```

Cover two files fresh, three with resume, one-by-one upload selection, duplicate
kinds, wrong source Stage C outer SHA, wrong official train/history SHA,
traversal, duplicate members, symlink entries, excessive size/compression,
changed-during-copy files, and consistently rehashed inner forgery.

- [ ] **Step 2: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_inputs.py -q
```

- [ ] **Step 3: Implement the upload trust boundary**

```python
@dataclass(frozen=True)
class VerifiedHierarchicalInputs:
    training: VerifiedTrainingInput
    stage_c: VerifiedStageC
    source_stage_c_path: Path
    source_stage_c_sha256: str
    anchor_predictions: Mapping[str, Path]


def classify_upload(path: Path) -> str: ...


def classify_and_verify_uploads(
    paths: Sequence[Path],
    *,
    run_root: Path,
    contract: HierarchicalContract,
    expected_contract_sha256: str,
    expected_code_sha256: str,
) -> tuple[VerifiedHierarchicalInputs, Path | None]: ...
```

Reuse `verify_and_extract_training_input` and `verify_and_extract_stage_c` from
`experiments.catboost_tabm_blend.inputs`. Extract Stage C anchor predictions for
exactly `2022->2023` and `2023->2024`, validate their row/target/probability
schema, and copy them below the exclusive run root. Do not extract any model or
submission member not needed by this campaign. Resume verification is delegated
to Task 9 and must bind contract, code, input manifest, train, history, Stage C
outer delivery, Stage C state, and anchor prediction hashes.

The exact binding keys are:

```python
EXPECTED_BINDING_KEYS = {
    "contract_sha256", "code_sha256", "environment_sha256",
    "input_manifest_sha256", "train_sha256", "history_sha256",
    "stage_c_delivery_sha256", "stage_c_review_sha256",
    "stage_c_state_sha256", "anchor_2022_2023_sha256",
    "anchor_2023_2024_sha256",
}
```

Every value is a lowercase SHA-256. The environment digest covers Python,
PyTorch, NumPy, pandas, SciPy, TabM, rtdl-num-embeddings and CUDA runtime
versions recorded before any worker starts.

- [ ] **Step 4: Run GREEN and old-verifier regressions**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_inputs.py \
  tests/test_catboost_tabm_blend_inputs.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/inputs.py
git diff --check
```

- [ ] **Step 5: Commit Task 7**

```bash
git add experiments/hierarchical_tabm/inputs.py \
  tests/test_hierarchical_tabm_inputs.py
git commit -m "feat: verify hierarchical TabM inputs"
```

### Task 8: Train one OOF or full TabM job with resumable checkpoints

**Files:**
- Create: `experiments/hierarchical_tabm/training.py`
- Create: `tests/test_hierarchical_tabm_training.py`

- [ ] **Step 1: Write failing injected-runtime OOF tests**

```python
def test_oof_worker_uses_only_fold_training_for_context(tmp_path: Path) -> None:
    result = run_training_job(job_2022_2023, data_dir, tmp_path, deadline, fit=fake_fit)
    assert fake_fit.context_fit_seasons.max() == 2022
    assert fake_fit.valid_seasons.tolist() == [2023]
    assert result.status == "completed"
    assert result.predictions_path.name == "predictions.csv"


def test_worker_emits_exact_anchor_aligned_prediction_schema(tmp_path: Path) -> None:
    result = run_training_job(job_2022_2023, data_dir, tmp_path, deadline, fit=fake_fit)
    frame = pd.read_csv(result.predictions_path)
    assert tuple(frame) == (
        "row_id", "target", "probability", "season", "game_month", "game_type",
        "count_state", "hand_matchup", "base_out_state",
        "pitcher_known", "batter_known",
    )
```

Verify exact row alignment, finite probabilities, best epoch and best Brier,
fold identity, K/context/preprocessing hash binding, and all progress markers.

- [ ] **Step 2: Write deadline and checkpoint RED cases**

```python
def test_deadline_returns_incomplete_with_latest_verified_checkpoint(tmp_path: Path) -> None:
    result = run_training_job(job, data_dir, tmp_path, expired_deadline, fit=checkpointing_fit)
    assert result.status == "incomplete"
    assert result.checkpoint_path.is_file()
    assert result.completed_epochs == checkpointing_fit.completed_epochs


def test_resume_rejects_identity_or_optimizer_mismatch(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path, code_sha256="0" * 64)
    with pytest.raises(HierarchicalTrainingError, match="identity differs"):
        run_training_job(job, data_dir, tmp_path / "out", deadline, resume=checkpoint)
```

Cover model key/shape, AdamW moments, plateau scheduler, scaler, Python/NumPy/
Torch/CUDA RNG, epoch curves, best epoch, best Brier, adapter state and maximum
tensor dimensions. Validate untrusted checkpoint payload in an isolated process
with a hard timeout before allocating a GPU model.

- [ ] **Step 3: Implement the job types and injected training seam**

```python
@dataclass(frozen=True)
class TrainingJobResult:
    job_id: str
    kind: str
    status: str
    train_rows: int
    valid_rows: int | None
    best_epoch: int | None
    best_brier: float | None
    completed_epochs: int
    predictions_path: Path | None
    checkpoint_path: Path | None
    final_model_path: Path | None
    feature_state_path: Path | None
    failure: str | None


def run_training_job(
    job: HierarchicalJob,
    *,
    data_dir: Path,
    output_dir: Path,
    selected_k: float,
    absolute_deadline: float,
    identity: Mapping[str, str],
    final_epochs: int | None = None,
    resume_checkpoint: Path | None = None,
    on_epoch_checkpoint: Callable[[Path, int], None] | None = None,
    fit: Callable = fit_candidate,
) -> TrainingJobResult: ...
```

For OOF jobs, split by season, fit `ContextState` on fit rows, prepare LOO/frozen
batches, and call the existing `TabMAdapter` plus `fit_candidate`. For full fit,
fit hierarchy and preprocessing on all official 2019..2024 rows, train exactly
`final_epochs` with a constant schedule, and save the model plus frozen feature
state. The full job has no validation and may start only after Task 10 authorizes
it. Write `worker_result.json` atomically only after every referenced file is
hashed.

- [ ] **Step 4: Add CLI and real-shape synthetic adapter test**

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = job_from_json(args.job)
    result = run_training_job(
        job, data_dir=args.data_dir, output_dir=args.output_dir,
        selected_k=args.selected_k, absolute_deadline=args.absolute_deadline,
        identity=json.loads(args.identity), final_epochs=args.final_epochs,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(canonical_json(training_result_payload(result)).decode(), flush=True)
    return 0 if result.status in {"completed", "incomplete"} else 1
```

Use a tiny synthetic `FeatureBatch` to build the actual TabM P2 adapter on CPU,
verify feature-width/model-state expectations, and avoid fitting official data or
requiring CUDA.

- [ ] **Step 5: Run GREEN and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_training.py \
  tests/test_independent_dl_training.py \
  tests/test_tabm_campaign_model.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/training.py
git diff --check
git add experiments/hierarchical_tabm/training.py \
  tests/test_hierarchical_tabm_training.py
git commit -m "feat: train resumable hierarchical TabM jobs"
```

### Task 9: Write and verify review, resume, and candidate delivery artifacts

**Files:**
- Create: `experiments/hierarchical_tabm/artifacts.py`
- Create: `tests/test_hierarchical_tabm_artifacts.py`

- [ ] **Step 1: Write failing exact-inventory artifact tests**

```python
def test_review_and_resume_have_exact_members(tmp_path: Path) -> None:
    bundles = write_campaign_bundles(evidence_fixture, tmp_path)
    assert zip_names(bundles.review) == EXPECTED_REVIEW_MEMBERS
    assert zip_names(bundles.resume) == EXPECTED_RESUME_MEMBERS
    verify_review_bundle(bundles.review, expected_bindings=evidence_fixture.bindings)
    verify_resume_bundle(bundles.resume, expected_bindings=evidence_fixture.bindings)


def test_candidate_delivery_contains_no_test_or_submission_member(tmp_path: Path) -> None:
    delivery = write_candidate_delivery(delivery_fixture, tmp_path)
    names = zip_names(delivery)
    assert not any("test" in name.lower() or "submit" in name.lower() for name in names)
    assert {
        "model/final_checkpoint.pt",
        "state/feature_state.json",
        "state/calibration_H2.json",
    } <= names
```

Review exact members include contract, campaign log, K selection, both OOF
predictions/metrics, calibration selection, three decisions, and stage state.
Resume includes those completed items plus at most one active checkpoint. Delivery
includes one full model/feature state and the calibration states for accepted
H2/H3 candidates. A frontier-only H1 may create a delivery whose manifest role
is exactly `public_diagnostic_only`; it is not final acceptance and it is not a
submission package.

- [ ] **Step 2: Add hostile ZIP and consistent-forgery RED tests**

Test duplicate/traversal/symlink/oversize/high-ratio members, missing/extra
members, manifest type errors, payload/manifest SHA mismatch, stage state whose
completed job lacks evidence, active checkpoint whose epoch differs, decision
whose Brier differs from predictions, calibration whose fit-row SHA differs,
and a consistently rehashed archive whose semantic binding is wrong.

- [ ] **Step 3: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_artifacts.py -q
```

- [ ] **Step 4: Implement canonical, streaming artifact APIs**

```python
@dataclass(frozen=True)
class CampaignBundles:
    review: Path
    resume: Path


def write_campaign_bundles(evidence: CampaignEvidence, output_dir: Path, *, check_deadline: Callable[[], None] | None = None) -> CampaignBundles: ...
def verify_review_bundle(path: Path, *, expected_bindings: Mapping[str, str]) -> Mapping[str, object]: ...
def verify_resume_bundle(path: Path, *, expected_bindings: Mapping[str, str]) -> Mapping[str, object]: ...
def restore_resume(path: Path, destination: Path, *, expected_bindings: Mapping[str, str], check_deadline: Callable[[], None] | None = None) -> RestoredCampaign: ...
def write_candidate_delivery(evidence: DeliveryEvidence, output_dir: Path, *, check_deadline: Callable[[], None] | None = None) -> Path: ...
def verify_candidate_delivery(path: Path, *, expected_bindings: Mapping[str, str]) -> Mapping[str, object]: ...
```

Define the referenced evidence types in the same module:

```python
@dataclass(frozen=True)
class CampaignEvidence:
    bindings: Mapping[str, str]
    contract_path: Path
    log_path: Path
    state_path: Path
    k_selection_path: Path | None
    completed_job_directories: Mapping[str, Path]
    calibration_paths: Mapping[str, Path]
    decision_paths: Mapping[str, Path]
    active_job_directory: Path | None


@dataclass(frozen=True)
class DeliveryEvidence:
    bindings: Mapping[str, str]
    delivery_roles: Mapping[str, str]
    final_checkpoint_path: Path
    feature_state_path: Path
    calibration_paths: Mapping[str, Path]
    independence_report_paths: Mapping[str, Path]


@dataclass(frozen=True)
class RestoredCampaign:
    root: Path
    state: Mapping[str, object]
    completed_job_directories: Mapping[str, Path]
    active_job_directory: Path | None
```

Hash files by streaming from trusted descriptors and carry expected SHA values
from stage state to the ZIP writer to close the validate/write race. Publish by
temporary file plus `os.replace`. Reopen and recursively verify the finished ZIP
before returning it. Candidate delivery manifest must contain
`review_only=true`, `submission_package=false`, and ordered
`delivery_candidate_ids` with each candidate's exact delivery role.

- [ ] **Step 5: Run GREEN with legacy artifact regressions and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_artifacts.py \
  tests/test_catboost_deployment_artifacts.py \
  tests/test_tabm_campaign_artifacts.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/artifacts.py
git diff --check
git add experiments/hierarchical_tabm/artifacts.py \
  tests/test_hierarchical_tabm_artifacts.py
git commit -m "feat: seal hierarchical TabM evidence"
```

### Task 10: Orchestrate K, OOF, calibration, decisions, and conditional full fit

**Files:**
- Create: `experiments/hierarchical_tabm/runner.py`
- Create: `tests/test_hierarchical_tabm_runner.py`

- [ ] **Step 1: Write failing stage-order and resume tests**

```python
def test_campaign_runs_preregistered_order(tmp_path: Path) -> None:
    run = run_campaign(verified, tmp_path, runtime=fake_runtime)
    assert fake_runtime.calls == [
        "select_k_2021_2022", "oof_2022_2023", "oof_2023_2024",
        "calibrate_H2", "calibrate_H3", "decide_H1", "decide_H2", "decide_H3",
        "full_fit_2019_2024",
    ]
    assert run.state.selected_k in {32.0, 128.0, 512.0}


def test_resume_reuses_only_verified_completed_work(tmp_path: Path) -> None:
    run = run_campaign(verified, tmp_path, resume_bundle=completed_first_fold, runtime=fake_runtime)
    assert "oof_2022_2023" not in fake_runtime.calls
    assert "oof_2023_2024" in fake_runtime.calls
```

Cover no-time-for-new-job, deadline during OOF, completed K reuse, calibration
reuse, per-candidate failure isolation, H1 frontier only, H2-only acceptance,
H3 acceptance, no accepted candidates, and full-fit resume.

- [ ] **Step 2: Write final-epoch and final-calibration RED tests**

```python
def test_final_epoch_is_row_weighted_median_clamped_to_contract() -> None:
    assert choose_final_epochs([(100, 1), (300, 9)], minimum=2, maximum=8) == 8
    assert choose_final_epochs([(300, 3), (100, 7)], minimum=2, maximum=8) == 3


def test_final_calibration_uses_2023_and_2024_oof_only_after_selection() -> None:
    state = fit_final_calibration(selection=selection_h2, oof_by_fold=oof)
    assert state.fit_row_ids_sha256 == sha256_ids((*ids_2023, *ids_2024))
    assert not set(ids_2022) & set((*ids_2023, *ids_2024))
```

- [ ] **Step 3: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_runner.py -q
```

- [ ] **Step 4: Implement the restartable state machine**

```python
@dataclass(frozen=True)
class CampaignState:
    schema_version: int
    status: str
    bindings: Mapping[str, str]
    selected_k: float | None
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None
    decisions: Mapping[str, CandidateDecision]
    delivery_candidate_ids: tuple[str, ...]
    final_epochs: int | None
    final_fit_completed: bool


@dataclass(frozen=True)
class CampaignRun:
    state: CampaignState
    bundles: CampaignBundles
    candidate_delivery: Path | None


class CampaignRuntime(Protocol):
    def select_k(self, fit_rows: pd.DataFrame, valid_rows: pd.DataFrame) -> KSelection: ...
    def run_oof(self, job: HierarchicalJob, **kwargs: object) -> TrainingJobResult: ...
    def run_full_fit(self, job: HierarchicalJob, **kwargs: object) -> TrainingJobResult: ...


def choose_final_epochs(weighted_epochs: Sequence[tuple[int, int]], *, minimum: int, maximum: int) -> int: ...
def fit_final_calibration(selection: CalibrationSelection, oof_by_fold: Mapping[str, pd.DataFrame]) -> CalibrationState: ...
def run_campaign(verified: VerifiedHierarchicalInputs, output_dir: Path, *, resume_bundle: Path | None, absolute_deadline: float, runtime: CampaignRuntime, on_verified_resume: Callable[[Path], None]) -> CampaignRun: ...
```

After both OOF folds, align H1 with Stage C anchors, fit/select H2 and H3 on
2023 OOF, evaluate locked states once on 2024 OOF, and write all decisions before
considering full fit. If H1-strong, H1-frontier diagnostic eligibility,
H2-accepted, or H3-accepted is present, fit one shared full H1 model. Refit only
the already selected H2/H3 calibration structures on combined 2023+2024 OOF for
delivery. A frontier-only delivery is marked diagnostic-only and never treated
as final acceptance. This task stops the state at
`delivery_pending_validation` after full fit and final calibration; Task 11 is
the only step that may turn that state into a candidate delivery.

- [ ] **Step 5: Run GREEN and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_runner.py \
  tests/test_hierarchical_tabm_metrics.py \
  tests/test_hierarchical_tabm_artifacts.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/runner.py
git diff --check
git add experiments/hierarchical_tabm/runner.py \
  tests/test_hierarchical_tabm_runner.py
git commit -m "feat: orchestrate hierarchical TabM campaign"
```

### Task 11: Validate frozen candidate inference and row independence

**Files:**
- Create: `experiments/hierarchical_tabm/inference.py`
- Modify: `experiments/hierarchical_tabm/runner.py`
- Create: `tests/test_hierarchical_tabm_inference.py`
- Modify: `tests/test_hierarchical_tabm_runner.py`

- [ ] **Step 1: Write failing frozen-predictor tests**

```python
def test_frozen_predictor_applies_only_train_state(tmp_path: Path) -> None:
    predictor = load_candidate_predictor(delivery, candidate_id="H2", device="cpu")
    before = predictor.state_digest()
    probability = predictor.predict_batch(eval_rows.drop(columns="control_success"), batch_size=2)
    assert probability.shape == (len(eval_rows),)
    assert predictor.state_digest() == before


def test_prediction_is_row_order_and_batch_invariant(tmp_path: Path) -> None:
    predictor = load_candidate_predictor(delivery, candidate_id="H3", device="cpu")
    report = audit_frozen_predictor(predictor, eval_rows, batch_sizes=(1, 3, 257))
    assert report.features_exact is True
    assert report.maximum_probability_delta <= 1e-6
    assert report.state_before == report.state_after


def test_scale_gate_uses_sealed_limits() -> None:
    report = audit_inference_limits(
        fake_predictor, scale_frame, contract,
        resource_probe=lambda: (120.0, 2_000_000_000, 3_000_000_000),
    )
    assert report.passed is True
    assert report.row_count == len(scale_frame)
```

Test a row alone, reverse, deterministic shuffle, several batches, unseen every
hierarchy level, unseen calibration categories, absent target, extra unrelated
row, and changed total row count. Explicitly fail if prediction code calls
`groupby`, rolling, rank, cumulative operations, filesystem reads, network, or
state mutation during `predict_batch`.

- [ ] **Step 2: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_inference.py -q
```

- [ ] **Step 3: Implement the frozen inference interface**

```python
@dataclass(frozen=True)
class IndependenceReport:
    row_count: int
    features_exact: bool
    maximum_probability_delta: float
    state_before: str
    state_after: str
    batch_sizes: tuple[int, ...]


@dataclass(frozen=True)
class InferenceResourceReport:
    row_count: int
    python_version: str
    elapsed_seconds: float
    peak_gpu_bytes: int
    peak_rss_bytes: int
    artifact_bytes: int
    passed: bool


class FrozenHierarchicalPredictor:
    def state_digest(self) -> str: ...
    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: ...
    def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 2048) -> np.ndarray: ...


def load_candidate_predictor(delivery: Path, *, candidate_id: str, device: str) -> FrozenHierarchicalPredictor: ...
def audit_frozen_predictor(predictor: FrozenHierarchicalPredictor, frame: pd.DataFrame, *, batch_sizes: tuple[int, ...] = (1, 257, 2048), tolerance: float = 1e-6) -> IndependenceReport: ...
def audit_inference_limits(predictor: FrozenHierarchicalPredictor, frame: pd.DataFrame, contract: HierarchicalContract, *, resource_probe: Callable | None = None) -> InferenceResourceReport: ...
```

The loader recursively verifies candidate delivery first. H1 returns raw TabM
probability, H2/H3 apply only the persisted calibration state. `predict_batch`
may inspect only the supplied row, frozen context/preprocessing/category/model/
calibration state, and the requested batch size.
`audit_inference_limits` runs in a bounded child on 245,789 train-derived audit
rows during the user-owned Colab run, records Python 3.11.15, elapsed time, peak
CUDA/RSS and delivery bytes, and fails before delivery when a sealed limit is
exceeded. Synthetic tests inject the probe and never allocate a GPU.

Wire this into `run_campaign` after the full model and final calibration states
exist. Load each delivery-role candidate, run independence and scale checks, and
call `write_candidate_delivery` only for candidates that pass. A scale or
independence failure blocks only that candidate. If none pass, keep review/resume
evidence and return `candidate_delivery=None`.

- [ ] **Step 4: Run GREEN with rule regressions and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_inference.py \
  tests/test_row_independence_evidence.py \
  tests/test_rules_code_gate.py \
  tests/test_rules_repository_enforcement.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/inference.py
git diff --check
git add experiments/hierarchical_tabm/inference.py \
  experiments/hierarchical_tabm/runner.py \
  tests/test_hierarchical_tabm_inference.py \
  tests/test_hierarchical_tabm_runner.py
git commit -m "feat: verify frozen hierarchical inference"
```

### Task 12: Supervise Colab subprocesses and verified recovery

**Files:**
- Create: `experiments/hierarchical_tabm/colab.py`
- Create: `tests/test_hierarchical_tabm_colab.py`

- [ ] **Step 1: Write failing supervisor and cadence tests**

```python
def test_supervisor_publishes_first_checkpoint_then_cadence(tmp_path: Path) -> None:
    downloads = []
    result = run_supervised_campaign(
        verified, output_dir=tmp_path / "out", snapshot_dir=tmp_path / "snap",
        resume_bundle=None, wall_deadline=clock.time() + 100,
        on_verified_resume=downloads.append, runtime=fake_subprocess_runtime,
        clock=clock,
    )
    assert downloads[0].name.startswith("hierarchical_tabm_resume")
    assert all(verify_resume_bundle(path, expected_bindings=bindings) for path in downloads)


def test_stalled_first_epoch_still_republishes_verified_resume(tmp_path: Path) -> None:
    runtime = never_checkpoint_runtime(duration=1300)
    downloads = []
    run_supervised_campaign(..., runtime=runtime, on_verified_resume=downloads.append)
    assert len(downloads) >= 1
```

Cover first checkpoint immediately, 20-minute unchanged republish, fold-complete
publish, callback failure preserving previous resume, no checkpoint before
deadline, subprocess exit without result, terminate then kill timeout, log relay,
and exclusive run roots across reruns.

- [ ] **Step 2: Write active-checkpoint promotion RED tests**

```python
def test_active_checkpoint_becomes_verified_incomplete_resume(tmp_path: Path) -> None:
    active = make_atomic_epoch_checkpoint(tmp_path, completed_epochs=4)
    resume = promote_active_checkpoint(active, evidence, tmp_path / "snap")
    restored = restore_resume(resume, tmp_path / "restored", expected_bindings=evidence.bindings)
    assert restored.active.completed_epochs == 4
    assert file_sha256(restored.active.checkpoint) == file_sha256(active.checkpoint)
```

Reject half-written metadata, changed-during-copy checkpoint, wrong epoch/hash/
identity, invalid optimizer/RNG state, and validation that exceeds the absolute
deadline. Run untrusted checkpoint validation in a killable child.

- [ ] **Step 3: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_colab.py -q
```

- [ ] **Step 4: Implement the supervisor**

```python
@dataclass
class SnapshotCadence:
    snapshot_interval_seconds: float
    download_interval_seconds: float
    started_at: float
    last_snapshot_at: float | None = None
    last_download_at: float | None = None


@dataclass(frozen=True)
class ActiveCheckpoint:
    job_id: str
    job_directory: Path
    checkpoint: Path
    checkpoint_meta: Path
    completed_epochs: int


class Clock(Protocol):
    def time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class SubprocessRuntime(Protocol):
    def run_job(self, job: HierarchicalJob, **kwargs: object) -> TrainingJobResult: ...
    def active_checkpoint(self) -> ActiveCheckpoint | None: ...
    def terminate(self) -> None: ...


def promote_active_checkpoint(active: ActiveCheckpoint, evidence: CampaignEvidence, snapshot_dir: Path, *, check_deadline: Callable[[], None]) -> Path: ...


def run_supervised_campaign(
    verified: VerifiedHierarchicalInputs,
    *,
    output_dir: Path,
    snapshot_dir: Path,
    resume_bundle: Path | None,
    wall_deadline: float,
    on_verified_resume: Callable[[Path], None],
    log_path: Path,
    runtime: SubprocessRuntime | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> CampaignRun: ...
```

Launch one GPU worker at a time with explicit `PYTHONPATH` and no shell. Relay
stdout to console and `hierarchical_tabm.log`. Poll at most every five seconds.
Do not start a new job inside the 900-second guard. Every download callback must
receive a recursively verified resume; callback ownership never deletes the
user-uploaded resume.

- [ ] **Step 5: Run GREEN and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_colab.py \
  tests/test_hierarchical_tabm_runner.py \
  tests/test_hierarchical_tabm_artifacts.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/colab.py
git diff --check
git add experiments/hierarchical_tabm/colab.py \
  tests/test_hierarchical_tabm_colab.py
git commit -m "feat: supervise hierarchical TabM Colab run"
```

### Task 13: Build a deterministic one-cell Colab handoff

**Files:**
- Create: `experiments/hierarchical_tabm/runtime_inventory.py`
- Create: `experiments/hierarchical_tabm/requirements-colab.txt`
- Create: `experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py`
- Create: `tools/build_hierarchical_tabm_colab_cell.py`
- Create: `tests/test_hierarchical_tabm_colab_cell.py`

- [ ] **Step 1: Write failing runtime-closure tests**

```python
def test_runtime_archive_has_exact_import_closure(repo_root: Path) -> None:
    names = archive_names(runtime_archive(repo_root))
    assert names == set(RUNTIME_MEMBERS)
    assert "experiments/hierarchical_tabm/requirements-colab.txt" in names
    assert not any("submission" in name or "test.csv" in name for name in names)


def test_dependency_change_changes_code_identity(repo_root: Path, tmp_path: Path) -> None:
    copied = copy_runtime_sources(repo_root, tmp_path)
    before = code_identity_sha256(copied)
    (copied / "experiments/hierarchical_tabm/requirements-colab.txt").write_text(
        "tabm==0.0.4\n"
    )
    assert code_identity_sha256(copied) != before
```

Run an isolated import and synthetic contract/job construction using only the
embedded archive. Require no accidental eager import through package
`__init__.py`.

- [ ] **Step 2: Write failing cell-renderer tests**

```python
def test_checked_in_cell_matches_renderer(repo_root: Path) -> None:
    rendered = render_cell(repo_root)
    checked_in = (repo_root / "experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py").read_bytes()
    assert rendered == checked_in
    assert len(rendered) < 1_000_000


def test_session_deadline_precedes_upload_and_setup() -> None:
    text = render_cell(REPO_ROOT).decode()
    assert text.index("SESSION_DEADLINE = time.time() + 10800") < text.index("files.upload()")
    assert "drive.mount" not in text
    assert "github" not in text.lower()
```

- [ ] **Step 3: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_colab_cell.py -q
```

- [ ] **Step 4: Implement deterministic inventory and builder**

```python
RUNTIME_MEMBERS = (
    "experiments/hierarchical_tabm/__init__.py",
    "experiments/hierarchical_tabm/contract.json",
    "experiments/hierarchical_tabm/contracts.py",
    "experiments/hierarchical_tabm/context_features.py",
    "experiments/hierarchical_tabm/feature_adapter.py",
    "experiments/hierarchical_tabm/calibration.py",
    "experiments/hierarchical_tabm/metrics.py",
    "experiments/hierarchical_tabm/inputs.py",
    "experiments/hierarchical_tabm/training.py",
    "experiments/hierarchical_tabm/artifacts.py",
    "experiments/hierarchical_tabm/runner.py",
    "experiments/hierarchical_tabm/colab.py",
    "experiments/hierarchical_tabm/inference.py",
    "experiments/hierarchical_tabm/runtime_inventory.py",
    "experiments/hierarchical_tabm/requirements-colab.txt",
    "experiments/independent_dl/row_features.py",
    "experiments/independent_dl/preprocessing.py",
    "experiments/independent_dl/feature_sources/trackman.py",
    "experiments/independent_dl/features.py",
    "experiments/independent_dl/progress.py",
    "experiments/independent_dl/training.py",
    "experiments/independent_dl/models/common.py",
    "experiments/independent_dl/models/tabm.py",
    "experiments/catboost_tabm_blend/contract.json",
    "experiments/catboost_tabm_blend/contracts.py",
    "experiments/catboost_tabm_blend/metrics.py",
    "experiments/catboost_tabm_blend/inputs.py",
    "experiments/tabm_campaign/artifacts.py",
)


def runtime_archive(root: Path) -> bytes: ...
def code_identity_sha256(root: Path) -> str: ...
def environment_identity() -> Mapping[str, str]: ...
def environment_identity_sha256() -> str: ...
def render_cell(root: Path) -> bytes: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

Create deterministic tar members with mode 0644, uid/gid 0, mtime 0, then gzip
with mtime 0 and base64-embed it. Pin exactly:

```text
tabm==0.0.3
rtdl-num-embeddings==0.0.12
```

`main()` writes the checked-in cell atomically by default. With `--check`, it
compares rendered bytes to the checked-in file, prints both SHA-256 values on a
mismatch, and exits nonzero without writing. Do not archive the eager
`experiments.independent_dl.models.__init__`; the isolated runtime test must
prove that `models.tabm` loads as a namespace subpackage using only the exact
allowlist above.

The cell must:

1. set the 10,800-second deadline before upload or install;
2. upload two or three ZIPs together in one picker;
3. classify by content, not filename;
4. verify code, contract, inputs, Stage C and optional resume;
5. require one Tesla T4-compatible CUDA device;
6. run the supervised campaign;
7. request verified resume downloads at cadence and fold completion;
8. request final review/resume/candidate delivery downloads when present;
9. print `HIER_ERROR stage=<stage> type=<type> message=<message>` and re-raise.

Expected success markers are:

```text
HIER_INPUTS_VERIFIED
HIER_CONTEXT_SELECTED k=<K>
HIER_JOB_START fold=<fold>
HIER_TRAINING_PROGRESS
HIER_JOB_END fold=<fold>
HIER_DECISION candidate=<H1|H2|H3> status=<status>
HIER_FULL_TRAIN_START
HIER_DELIVERY_READY
```

- [ ] **Step 5: Generate twice and run GREEN**

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/build_hierarchical_tabm_colab_cell.py
shasum -a 256 experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py
artifacts/tabm_submission_python311/bin/python \
  tools/build_hierarchical_tabm_colab_cell.py
shasum -a 256 experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_colab_cell.py \
  tests/test_hierarchical_tabm_colab.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/hierarchical_tabm/runtime_inventory.py \
  experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py \
  tools/build_hierarchical_tabm_colab_cell.py
git diff --check
```

Expected: both `shasum` values are identical and the cell is below 1 MB.

- [ ] **Step 6: Commit Task 13**

```bash
git add experiments/hierarchical_tabm/runtime_inventory.py \
  experiments/hierarchical_tabm/requirements-colab.txt \
  experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py \
  tools/build_hierarchical_tabm_colab_cell.py \
  tests/test_hierarchical_tabm_colab_cell.py
git commit -m "feat: add hierarchical TabM Colab handoff"
```

### Task 14: Document the user run and perform final verification

**Files:**
- Create: `docs/HIERARCHICAL_TABM_COLAB.md`
- Modify: `README.md`

- [ ] **Step 1: Write the runbook in plain Korean**

The runbook must give one unambiguous sequence:

```text
1. Colab 새 노트에서 T4 GPU를 선택한다.
2. COLAB_HIERARCHICAL_TABM_CELL.py 전체를 한 셀에 붙여 넣는다.
3. 업로드 창 하나에서 catboost_tabm_blend_input.zip과
   tabm_colab_stage_C_delivery.zip을 동시에 선택한다.
4. 재개할 때만 최신 hierarchical_tabm_resume.zip을 함께 선택한다.
5. 완료 후 hierarchical_tabm_review.zip,
   hierarchical_tabm_resume.zip, 그리고 생성된 경우에만
   hierarchical_tabm_candidate_delivery.zip을 Codex에 전달한다.
```

Explain purpose, required inputs, expected 1–3 hour T4 runtime, 20-minute
download cadence, rerun safety, success markers, error marker, and the fact that
candidate delivery is not submit-ready. Explain H1/H2/H3 and every gate in
language a teammate can understand. Do not tell the user to mount Drive, clone
GitHub, rename uploads, or upload evaluation data.

- [ ] **Step 2: Add one README link**

Add only this link near the experiment-runbook section:

```markdown
- 계층적 문맥 TabM H1/H2/H3 Colab 실행: `docs/HIERARCHICAL_TABM_COLAB.md`
```

- [ ] **Step 3: Run focused, rule, and full regressions**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_contracts.py \
  tests/test_hierarchical_tabm_context_features.py \
  tests/test_hierarchical_tabm_k_selection.py \
  tests/test_hierarchical_tabm_feature_adapter.py \
  tests/test_hierarchical_tabm_calibration.py \
  tests/test_hierarchical_tabm_metrics.py \
  tests/test_hierarchical_tabm_inputs.py \
  tests/test_hierarchical_tabm_training.py \
  tests/test_hierarchical_tabm_artifacts.py \
  tests/test_hierarchical_tabm_runner.py \
  tests/test_hierarchical_tabm_colab.py \
  tests/test_hierarchical_tabm_inference.py \
  tests/test_hierarchical_tabm_colab_cell.py -q

artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_rules_code_gate.py \
  tests/test_rules_entrypoints.py \
  tests/test_rules_repository_enforcement.py \
  tests/test_repository_contract.py \
  tests/test_submission_package.py -q

artifacts/tabm_submission_python311/bin/python -m pytest -q
artifacts/tabm_submission_python311/bin/python -m compileall -q \
  experiments/hierarchical_tabm tools
git diff --check
```

Expected: every command exits 0. The full suite count may grow; record its exact
reported pass count in the final handoff rather than predicting it here.

- [ ] **Step 4: Audit forbidden scope and generated artifacts**

```bash
rg -n "test\.csv|submit\.zip|submission/package|drive\.mount|github\.com|requests\.|urllib" \
  experiments/hierarchical_tabm tools/build_hierarchical_tabm_colab_cell.py \
  docs/HIERARCHICAL_TABM_COLAB.md
git ls-files '*.zip' '*.pt' '*.pth' '*.npy' '*.npz'
git status --short
```

Expected: no runtime access to evaluation/submission/network/Drive/GitHub, no
tracked training artifact, and only the pre-existing user-modified files plus
the intentional Task 14 docs before commit.

- [ ] **Step 5: Commit the runbook**

```bash
git add README.md docs/HIERARCHICAL_TABM_COLAB.md
git commit -m "docs: explain hierarchical TabM Colab run"
```

- [ ] **Step 6: Verify the final branch without running official data**

```bash
git log --oneline --decorate -16
git status --short
artifacts/tabm_submission_python311/bin/python \
  tools/build_hierarchical_tabm_colab_cell.py --check
shasum -a 256 experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py
```

Report the exact generated-cell SHA-256, code identity SHA-256, contract
SHA-256, focused/full pytest counts, and preserved pre-existing dirty paths.
Do not push.

## User-run handoff after implementation

After the implementation and verification above, the user receives one file to
copy:

```text
experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py
```

Required uploads in the single Colab picker:

```text
catboost_tabm_blend_input.zip
tabm_colab_stage_C_delivery.zip
hierarchical_tabm_resume.zip  # optional; latest one only
```

Approximate user runtime is 1–3 hours on one T4. The run is restart-safe at the
last recursively verified epoch/fold resume. The user returns:

```text
hierarchical_tabm_review.zip
hierarchical_tabm_resume.zip
hierarchical_tabm_candidate_delivery.zip  # only if created
```

If an error occurs, the user returns the complete line beginning with
`HIER_ERROR` plus the latest verified resume ZIP. Codex then verifies returned
hashes and decisions before any separate submission-packaging work is allowed.
