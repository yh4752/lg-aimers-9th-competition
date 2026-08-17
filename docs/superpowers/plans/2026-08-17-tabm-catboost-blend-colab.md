# TabM CatBoost OOF Blend Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a review-only, resumable Colab T4 campaign that reuses the sealed Stage C `single_s3407` TabM OOF predictions, trains one fixed CatBoost configuration on the same two temporal folds, and evaluates only the preregistered `0.70`, `0.80`, and `0.90` TabM blend weights.

**Architecture:** Add an isolated `experiments/catboost_tabm_blend` package rather than expanding the Stage P TabM runtime. Reuse the existing fold-fitted `tree_native + hand_matchup` transformer, but keep the blend contract, Stage C verification, CatBoost worker, pure metrics, artifact format, resume state, and Colab supervisor in focused modules. The one-cell entry point embeds only an explicit runtime inventory and never contains full-data inference or submission packaging.

**Tech Stack:** Python 3.11, pandas, NumPy, CatBoost 1.2.10 on a Colab Tesla T4, pytest with fake CatBoost fixtures, and standard-library JSON/ZIP/hash/subprocess utilities.

---

## Fixed scope and file map

Create one package with these responsibilities:

```text
experiments/catboost_tabm_blend/
├── __init__.py
├── contract.json
├── contracts.py
├── metrics.py
├── inputs.py
├── training.py
├── artifacts.py
├── runner.py
├── colab.py
├── runtime_inventory.py
├── requirements-colab.txt
└── COLAB_CATBOOST_TABM_BLEND_CELL.py
```

Create matching tests:

```text
tests/test_catboost_tabm_blend_contracts.py
tests/test_catboost_tabm_blend_metrics.py
tests/test_catboost_tabm_blend_inputs.py
tests/test_catboost_tabm_blend_training.py
tests/test_catboost_tabm_blend_artifacts.py
tests/test_catboost_tabm_blend_runner.py
tests/test_catboost_tabm_blend_colab.py
tests/test_catboost_tabm_blend_colab_cell.py
```

Create `tools/prepare_catboost_tabm_blend_input.py`, `tools/build_catboost_tabm_blend_colab_cell.py`, and `docs/CATBOOST_TABM_BLEND_COLAB.md`. Do not modify an existing Stage P/Stage C/Version D cell, notebook, or submission package. Codex runs fixture and static tests only; the user runs the official-data T4 job.

## Task 1: Seal the campaign contract and two-job schedule

**Files:**

- Create: `experiments/catboost_tabm_blend/__init__.py`
- Create: `experiments/catboost_tabm_blend/contract.json`
- Create: `experiments/catboost_tabm_blend/contracts.py`
- Create: `tests/test_catboost_tabm_blend_contracts.py`

- [ ] **Step 1: Write contract-loader tests before the package exists**

Test this exact public surface:

```python
from experiments.catboost_tabm_blend.contracts import (
    BlendContractError,
    build_jobs,
    contract_sha256,
    load_contract,
)


def test_contract_seals_two_folds_one_seed_and_three_weights() -> None:
    contract = load_contract()
    assert [(job.train_end_year, job.valid_year) for job in build_jobs(contract)] == [
        (2022, 2023),
        (2023, 2024),
    ]
    assert {job.seed for job in build_jobs(contract)} == {42}
    assert contract.tabm_weights == (0.9, 0.8, 0.7)
    assert contract.minimum_weighted_gain == 0.00003
    assert contract.maximum_fold_regression == 0.00003
    assert contract.review_only is True
    assert contract.submission_package is False
```

Add mutation tests for wrong/extra/missing schema keys, reversed or missing fold, seed drift, weight drift, CatBoost parameter drift, malformed SHA-256, `review_only=false`, `submission_package=true`, snapshot interval other than `300`, download interval other than `1200`, and wall budget other than `10800`.

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_tabm_blend_contracts.py -q
```

Expected: collection fails with `ModuleNotFoundError: experiments.catboost_tabm_blend`.

- [ ] **Step 2: Add the exact JSON contract**

Write `contract.json` with these values and no additional model choices:

```json
{
  "schema_version": 1,
  "campaign_id": "catboost_tabm_blend_v1",
  "review_only": true,
  "submission_package": false,
  "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
  "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
  "stage_c": {
    "delivery_sha256": "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a",
    "review_sha256": "461c4422cebdc7383577ad6c53b044bfe3d9d15b667383e454591135e4242d2c",
    "resume_sha256": "25aa43a846357f0ac97bcb18dd1826727d92e0850cb51091bec002ba6e2ed838",
    "stage_state_sha256": "290b3b10854aca9131e19f3fd1b0620aed17e9b9d40e25f0cb41f67bb2fcba76",
    "selected_predictor": "single_s3407",
    "prediction_members": {
      "2022->2023": "predictions/c_final__a__p2__piecewise_linear__bce__plateau__s42__s3407__tr2022__va2023.csv",
      "2023->2024": "predictions/c_final__a__p2__piecewise_linear__bce__plateau__s42__s3407__tr2023__va2024.csv"
    }
  },
  "folds": [
    {"train_end_year": 2022, "valid_year": 2023},
    {"train_end_year": 2023, "valid_year": 2024}
  ],
  "preprocessing": {"profile": "tree_native", "components": ["hand_matchup"]},
  "catboost": {
    "iterations": 400,
    "depth": 7,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "border_count": 128,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "l2_leaf_reg": 3.0,
    "model_size_reg": 0.5,
    "max_ctr_complexity": 1,
    "random_seed": 42,
    "task_type": "GPU",
    "od_type": "Iter",
    "od_wait": 50
  },
  "tabm_weights": [0.9, 0.8, 0.7],
  "gates": {"minimum_weighted_gain": 0.00003, "maximum_fold_regression": 0.00003},
  "budget": {
    "wall_seconds": 10800,
    "new_job_guard_seconds": 900,
    "snapshot_interval_seconds": 300,
    "download_interval_seconds": 1200
  }
}
```

- [ ] **Step 3: Implement frozen types and exact-key validation**

Use these types:

```python
@dataclass(frozen=True)
class BlendJob:
    job_id: str
    train_end_year: int
    valid_year: int
    seed: int


@dataclass(frozen=True)
class BlendContract:
    campaign_id: str
    review_only: bool
    submission_package: bool
    official_train_sha256: str
    official_history_sha256: str
    stage_c: Mapping[str, object]
    folds: tuple[tuple[int, int], ...]
    preprocessing_components: tuple[str, ...]
    catboost_parameters: Mapping[str, object]
    tabm_weights: tuple[float, ...]
    minimum_weighted_gain: float
    maximum_fold_regression: float
    wall_seconds: int
    new_job_guard_seconds: int
    snapshot_interval_seconds: int
    download_interval_seconds: int
```

`load_contract(path=None)` requires exact key sets and exact values. `contract_sha256()` hashes the checked-in bytes. `build_jobs()` returns exactly:

```python
(
    BlendJob("catboost__hand_matchup__tr2022__va2023__s42", 2022, 2023, 42),
    BlendJob("catboost__hand_matchup__tr2023__va2024__s42", 2023, 2024, 42),
)
```

- [ ] **Step 4: Run focused tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_tabm_blend_contracts.py -q
git diff --check
git add experiments/catboost_tabm_blend/__init__.py \
  experiments/catboost_tabm_blend/contract.json \
  experiments/catboost_tabm_blend/contracts.py \
  tests/test_catboost_tabm_blend_contracts.py
git commit -m "feat: seal CatBoost TabM blend contract"
```

## Task 2: Implement aligned metrics and the promotion decision

**Files:**

- Create: `experiments/catboost_tabm_blend/metrics.py`
- Create: `tests/test_catboost_tabm_blend_metrics.py`

- [ ] **Step 1: Write alignment and exact-boundary tests**

Use two synthetic folds with the exact Stage C diagnostic columns `row_id`, `target`, `probability`, `game_type`, `game_month`, `pitcher_id_known`, and `batter_id_known`:

```python
def fold_frame(ids, targets, probabilities):
    return pd.DataFrame(
        {
            "row_id": ids,
            "target": targets,
            "probability": probabilities,
            "game_type": ["R"] * len(ids),
            "game_month": [4] * len(ids),
            "pitcher_id_known": ["known"] * len(ids),
            "batter_id_known": ["known"] * len(ids),
        }
    )


def test_fixed_weights_are_evaluated_without_continuous_search() -> None:
    result = evaluate_blends(tabm_by_fold, catboost_by_fold, load_contract())
    assert tuple(item.tabm_weight for item in result.candidates) == (1.0, 0.9, 0.8, 0.7)
```

Reject duplicate IDs, reordered IDs, target or diagnostic-label mismatch, missing rows, extra columns, NaN/inf, values outside `[0, 1]`, missing fold, and unexpected fold. Add `np.nextafter` cases immediately below and above both `0.00003` boundaries. Test that a metric tie selects the larger TabM weight. For each fold/model/weight, require row count and Brier grouped separately by `game_type`, `game_month`, `pitcher_id_known`, and `batter_id_known`; these are diagnostics and never change pass/fail.

Run and expect the new module import to fail.

- [ ] **Step 2: Implement exact alignment and scalar metrics**

Expose:

```python
@dataclass(frozen=True)
class BlendCandidateMetric:
    tabm_weight: float
    fold_brier: Mapping[str, float]
    weighted_brier: float
    weighted_gain: float
    fold_regression: Mapping[str, float]
    passed: bool


@dataclass(frozen=True)
class BlendDecision:
    baseline_fold_brier: Mapping[str, float]
    baseline_weighted_brier: float
    prediction_correlation: Mapping[str, float | None]
    residual_correlation: Mapping[str, float | None]
    segment_diagnostics: Mapping[str, object]
    candidates: tuple[BlendCandidateMetric, ...]
    selected_tabm_weight: float | None
    reason: str
```

The exact callable signature is `evaluate_blends(tabm_by_fold: Mapping[str, pd.DataFrame], catboost_by_fold: Mapping[str, pd.DataFrame], contract: BlendContract) -> BlendDecision`.

Require exact sequence equality of `row_id` and exact equality of target plus all four diagnostic labels before arithmetic. Compute Brier in float64, validation-row-count weighted mean, and `tabm_weight * tabm + (1 - tabm_weight) * catboost`. Use `Decimal(str(value))` only for gate comparisons and never round before a gate. Pearson correlation is `None` if either vector has zero variance. Residual is `target - probability`. Select passing candidates by `(weighted_brier, -tabm_weight)`. Segment payload keys are sorted string values with explicit `row_count` and `brier`; no segment threshold is evaluated.

- [ ] **Step 3: Seal deterministic decision JSON**

Implement `decision_payload()` and `canonical_json()` with sorted keys, compact separators, UTF-8 and `allow_nan=False`. Serialize unavailable correlations as null. Test identical bytes across two calls and reclassify parsed metrics to ensure serialization does not cross a gate.

- [ ] **Step 4: Run tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_tabm_blend_contracts.py \
  tests/test_catboost_tabm_blend_metrics.py -q
git diff --check
git add experiments/catboost_tabm_blend/metrics.py \
  tests/test_catboost_tabm_blend_metrics.py
git commit -m "feat: decide fixed CatBoost TabM blends"
```

## Task 3: Prepare and verify the only accepted inputs

**Files:**

- Create: `experiments/catboost_tabm_blend/inputs.py`
- Create: `tools/prepare_catboost_tabm_blend_input.py`
- Create: `tests/test_catboost_tabm_blend_inputs.py`

- [ ] **Step 1: Test the local training archive contract**

Require exactly:

```text
input_manifest.json
trackman_history.csv
train.csv
```

Preparation finds exactly one top-level copy of both CSVs, rejects nested copies and symlinks, verifies both official hashes, excludes test/submission/arbitrary extras, publishes atomically, refuses overwrite without `replace=True`, and returns identical bytes for identical sources.

The manifest has exact keys:

```json
{
  "schema_version": 1,
  "artifact_kind": "catboost_tabm_blend_input",
  "campaign_config_sha256": "<contract SHA-256>",
  "members": {
    "trackman_history.csv": {"size": 123, "sha256": "<sha256>"},
    "train.csv": {"size": 456, "sha256": "<sha256>"}
  }
}
```

Fixture sizes use their actual byte lengths.

- [ ] **Step 2: Test extraction and Stage C verification**

Training ZIP negatives cover traversal, absolute/backslash names, duplicate, symlink, directory, extra member, manifest over 1 MiB, member/total limit, ratio over 200, wrong hash, truncated content and changed contract binding.

Build a small Stage C-shaped fixture with:

```text
colab_stage_C.log
delivery_manifest.json
tabm_search_stage_C_resume_bundle.zip
tabm_search_stage_C_review_bundle.zip
```

Verify the fixed outer/review/resume/state hashes, all manifest sizes/hashes, generic Stage C review bundle, `stage_complete=true`, `selected_predictor=single_s3407`, the single final member with seed 3407 and temporal best epochs `[3, 0]`, and both exact prediction members. Reject one-at-a-time changes to any binding, selected identity, seven-column row schema, duplicate ID, target or probability. Expose no prediction after failure.

- [ ] **Step 3: Implement streaming input primitives**

Expose:

```python
@dataclass(frozen=True)
class VerifiedTrainingInput:
    data_dir: Path
    manifest_sha256: str
    train_sha256: str
    history_sha256: str


@dataclass(frozen=True)
class VerifiedStageC:
    delivery_sha256: str
    review_sha256: str
    resume_sha256: str
    stage_state_sha256: str
    prediction_paths: Mapping[str, Path]
```

Exact callable signatures:

- `prepare_input_archive(data_dir: Path, output: Path, contract: BlendContract, *, replace: bool = False) -> Path`
- `verify_and_extract_training_input(source: Path, destination: Path, contract: BlendContract) -> VerifiedTrainingInput`
- `verify_and_extract_stage_c(source: Path, destination: Path, contract: BlendContract) -> VerifiedStageC`

Use descriptor-based regular-file opens where practical, 1 MiB chunks, deterministic ZIP timestamps, exclusive temporary paths and a second source digest before publication. Reuse only `experiments.tabm_campaign.artifacts.verify_review_bundle`; do not import Version D training/inference code.

- [ ] **Step 4: Implement the local preparation CLI**

Insert repository root before importing `experiments`:

```python
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

Success line:

```text
CATBOOST_BLEND_INPUT_READY path=<absolute-path> sha256=<sha256> size_bytes=<n>
```

- [ ] **Step 5: Run tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_tabm_blend_inputs.py \
  tests/test_tabm_campaign_artifacts.py -q
git diff --check
git add experiments/catboost_tabm_blend/inputs.py \
  tools/prepare_catboost_tabm_blend_input.py \
  tests/test_catboost_tabm_blend_inputs.py
git commit -m "feat: verify CatBoost blend inputs"
```

## Task 4: Train one CatBoost fold with native snapshot recovery

**Files:**

- Create: `experiments/catboost_tabm_blend/training.py`
- Create: `experiments/catboost_tabm_blend/requirements-colab.txt`
- Create: `tests/test_catboost_tabm_blend_training.py`

- [ ] **Step 1: Write fake-model fold tests**

Use preprocessing fixtures and an injected model factory. The fake records constructor and `fit` arguments, writes deterministic model/snapshot bytes, and predicts a supplied vector. Assert that one job:

- trains only `season <= train_end_year` and validates only `season == valid_year`;
- calls `fit_catboost_features(..., components=("hand_matchup",))`;
- never passes validation target into fitting;
- passes the sealed constructor parameters and an isolated `train_dir`;
- fits with `save_snapshot=True`, `snapshot_interval=300`, the job snapshot path, validation `eval_set`, `use_best_model=True`, and `early_stopping_rounds=50`;
- atomically writes `model.cbm`, `predictions.csv`, `metrics.json`, `job.json`, `worker_result.json`, and `worker.log`;
- records raw prediction min/max, clips to `[0, 1]`, then calculates Brier from the written values.

- [ ] **Step 2: Test restart and failure boundaries**

Cover an existing snapshot, changed job/contract/input/code identity, empty fold, duplicate `row_id`, non-binary target, non-finite or wrong-length prediction, model-save failure and expired deadline. A snapshot reaches CatBoost only when all identity fields match. A failed run never creates a completed result.

Test conversion of CatBoost iteration text to:

```text
CATBOOST_PROGRESS job=<job_id> iteration=<n> elapsed_seconds=<seconds>
```

Run and expect the new module import to fail.

- [ ] **Step 3: Implement the fold API**

```python
@dataclass(frozen=True)
class FoldResult:
    job_id: str
    status: str
    train_rows: int
    valid_rows: int
    brier: float | None
    model_path: Path | None
    predictions_path: Path | None
    snapshot_path: Path | None
    elapsed_seconds: float
    failure: str | None
```

The exact callable signature is `run_fold(*, job: BlendJob, contract: BlendContract, data_dir: Path, output_dir: Path, contract_sha256: str, input_manifest_sha256: str, code_sha256: str, absolute_deadline: float, model_factory: Callable[..., object] | None = None) -> FoldResult`.

Lazily import `CatBoostRegressor`. Set `allow_writing_files=True` only for snapshotting and set `train_dir=output_dir / "catboost_info"`. Pass snapshot settings to `fit`. Prediction columns are exactly:

```text
row_id,target,probability,game_type,game_month,pitcher_id_known,batter_id_known
```

Known/OOV flags come only from the fold-fitted category values.

- [ ] **Step 4: Add the worker CLI and version pin**

`python -m experiments.catboost_tabm_blend.training --job-json <path>` reads one exact job, streams markers, tees `worker.log`, and exits nonzero on failure. `requirements-colab.txt` contains exactly:

```text
catboost==1.2.10
```

Local tests use the fake and do not install CatBoost.

- [ ] **Step 5: Run tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_tabm_blend_training.py \
  tests/test_catboost_preprocessing.py \
  tests/test_preprocessing_profiles.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/catboost_tabm_blend/training.py
git diff --check
git add experiments/catboost_tabm_blend/training.py \
  experiments/catboost_tabm_blend/requirements-colab.txt \
  tests/test_catboost_tabm_blend_training.py
git commit -m "feat: train resumable CatBoost blend folds"
```

## Task 5: Write deterministic review and resume evidence

**Files:**

- Create: `experiments/catboost_tabm_blend/artifacts.py`
- Create: `tests/test_catboost_tabm_blend_artifacts.py`

- [ ] **Step 1: Test exact member sets**

Completed review:

```text
contract/contract.json
decision/blend_decision.json
logs/campaign.log
metrics/fold_results.json
predictions/catboost__hand_matchup__tr2022__va2023__s42.csv
predictions/catboost__hand_matchup__tr2023__va2024__s42.csv
state/stage_state.json
manifest.json
```

Resume contains contract/state/manifest and, for each completed job, exact `job.json`, `metrics.json`, `model.cbm`, `predictions.csv`, `worker.log`, and `worker_result.json`. An active job may add `experiment.cbsnapshot`; it cannot contribute predictions to the decision.

- [ ] **Step 2: Test corruption, safety and streaming**

Reject duplicate/traversal/symlink members, ratio/size limits, wrong manifest hash, wrong contract/code/input/Stage C binding, result/prediction Brier mismatch, model/snapshot hash mismatch, missing completed artifacts, and a complete stage with one fold. Use a 64 MiB path member and monkeypatch `Path.read_bytes`/`ZipFile.read` to prove streaming. Reopen and verify before atomic publication; on failure preserve the previous valid bundle.

- [ ] **Step 3: Implement exact artifact APIs**

```python
@dataclass(frozen=True)
class BundlePaths:
    review: Path | None
    resume: Path
    review_sha256: str | None
    resume_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class VerifiedBlendResume:
    path: Path
    manifest_sha256: str
    bindings: Mapping[str, str]
    stage_complete: bool
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None
```

Exact callable signatures:

- `write_bundles(*, output_dir: Path, contract_path: Path, bindings: Mapping[str, str], stage_state_path: Path, campaign_log_path: Path, job_directories: Mapping[str, Path], decision_path: Path | None, check_deadline: Callable[[], None] | None = None) -> BundlePaths`
- `verify_review_bundle(path: Path, *, expected_bindings: Mapping[str, str]) -> None`
- `verify_resume_bundle(path: Path, *, expected_bindings: Mapping[str, str]) -> VerifiedBlendResume`

Every manifest has exact keys, `schema_version=1`, `campaign_id=catboost_tabm_blend_v1`, `review_only=true`, `submission_package=false`, immutable bindings, and member size/SHA-256.

- [ ] **Step 4: Run tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_tabm_blend_artifacts.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/catboost_tabm_blend/artifacts.py
git diff --check
git add experiments/catboost_tabm_blend/artifacts.py \
  tests/test_catboost_tabm_blend_artifacts.py
git commit -m "feat: seal CatBoost blend evidence bundles"
```

## Task 6: Orchestrate two folds, resume, and final selection

**Files:**

- Create: `experiments/catboost_tabm_blend/runner.py`
- Create: `tests/test_catboost_tabm_blend_runner.py`

- [ ] **Step 1: Write sequential scheduling tests**

With an injected runtime, verify:

```text
verify inputs
restore a matching resume
reuse completed 2022->2023
resume or start 2023->2024
publish verified resume after each completed job
evaluate blends only after both complete
publish final review and resume
```

Never allow more than one active job. A 900-second guard blocks starting the second fold while preserving a valid first-fold resume.

- [ ] **Step 2: Test state and identity transitions**

Cover fresh, first complete, second active, complete, corrupt resume, contract/input/Stage C/code drift. Allow only:

```text
not_started -> running -> completed
not_started -> running -> failed
not_started -> budget_inconclusive
running -> budget_inconclusive
```

Only completed evidence is reusable. A provided corrupt resume fails before GPU work; omitting it starts an independent run.

- [ ] **Step 3: Implement the runner**

```python
@dataclass(frozen=True)
class BlendRun:
    stage_complete: bool
    bundles: BundlePaths
    decision: BlendDecision | None
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None
```

Exact callable signatures:

- `code_sha256() -> str`
- `run_campaign(*, verified_input: VerifiedTrainingInput, verified_stage_c: VerifiedStageC, output_dir: Path, resume_bundle: Path | None, absolute_deadline: float, on_verified_resume: Callable[[Path], None] | None = None, runtime: Callable[..., FoldResult] = run_fold) -> BlendRun`

Hash the exact runtime inventory plus requirements. Restore into a fresh directory, validate every restored hash, and schedule sequentially. After both folds, call `evaluate_blends` with the selected Stage C predictions and persist canonical decision JSON.

- [ ] **Step 4: Stream worker output and bound termination**

Print:

```text
CATBOOST_JOB_START job=<job_id>
CATBOOST_JOB_REUSED job=<job_id>
CATBOOST_JOB_END job=<job_id> status=<status>
CATBOOST_FOLD_RESULT job=<job_id> brier=<value>
BLEND_WEIGHT_RESULT tabm_weight=<value> weighted_brier=<value> passed=<bool>
BLEND_DECISION selected_tabm_weight=<value-or-none> reason=<reason>
```

On deadline: terminate, bounded join, then kill a resistant child. Package only a snapshot stable over two polls whose copied SHA matches state.

- [ ] **Step 5: Run tests and commit**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_tabm_blend_contracts.py \
  tests/test_catboost_tabm_blend_metrics.py \
  tests/test_catboost_tabm_blend_inputs.py \
  tests/test_catboost_tabm_blend_training.py \
  tests/test_catboost_tabm_blend_artifacts.py \
  tests/test_catboost_tabm_blend_runner.py -q
git diff --check
git add experiments/catboost_tabm_blend/runner.py \
  tests/test_catboost_tabm_blend_runner.py
git commit -m "feat: orchestrate CatBoost TabM blend OOF"
```

## Task 7: Build the Colab supervisor and one-cell handoff

**Files:**

- Create: `experiments/catboost_tabm_blend/colab.py`
- Create: `experiments/catboost_tabm_blend/runtime_inventory.py`
- Create: `experiments/catboost_tabm_blend/COLAB_CATBOOST_TABM_BLEND_CELL.py`
- Create: `tools/build_catboost_tabm_blend_colab_cell.py`
- Create: `tests/test_catboost_tabm_blend_colab.py`
- Create: `tests/test_catboost_tabm_blend_colab_cell.py`

- [ ] **Step 1: Test content-based upload classification**

Accept exactly two ZIPs fresh or three with resume. Identify one training input, one exact Stage C delivery, and at most one matching blend resume by verified content, not filename. Reject duplicate/unknown kinds, a Stage P resume, a standalone Stage C resume, and any test/submission archive.

- [ ] **Step 2: Test 20-minute emergency cadence**

With fake time/worker, require native snapshot every 300 seconds, browser download only after 1,200 seconds and a changed snapshot hash, immediate download after each fold, no repeated download for a stalled snapshot, preservation of the prior verified resume until replacement verifies, and latest-resume download before re-raising an error.

- [ ] **Step 3: Implement supervisor and recursive delivery**

Exact callable signatures:

- `classify_and_verify_uploads(paths: Sequence[Path], *, run_root: Path, contract: BlendContract, expected_contract_sha256: str, expected_code_sha256: str) -> tuple[VerifiedTrainingInput, VerifiedStageC, Path | None]`
- `run_supervised_campaign(*, verified_input: VerifiedTrainingInput, verified_stage_c: VerifiedStageC, output_dir: Path, snapshot_dir: Path, resume_bundle: Path | None, wall_deadline: float, on_verified_resume: Callable[[Path], None], log_path: Path) -> BlendRun`

Final delivery has exactly `blend_campaign.log`, `catboost_tabm_blend_review.zip`, `catboost_tabm_blend_resume.zip`, and `delivery_manifest.json`. Bind input, Stage C, contract, code, embedded runtime, nested ZIPs, and log. Recursively verify under the current run root before atomic publication.

- [ ] **Step 4: Declare the embedded inventory**

Allow only:

```text
experiments/catboost_preprocessing/features.py
experiments/independent_dl/preprocessing.py
experiments/independent_dl/row_features.py
experiments/tabm_campaign/artifacts.py
experiments/catboost_tabm_blend/contract.json
experiments/catboost_tabm_blend/contracts.py
experiments/catboost_tabm_blend/metrics.py
experiments/catboost_tabm_blend/inputs.py
experiments/catboost_tabm_blend/training.py
experiments/catboost_tabm_blend/artifacts.py
experiments/catboost_tabm_blend/runner.py
experiments/catboost_tabm_blend/colab.py
experiments/catboost_tabm_blend/runtime_inventory.py
experiments/catboost_tabm_blend/requirements-colab.txt
```

Empty package markers may be included only when isolated import requires them. Exclude notebooks, final training, inference, submission, Version D and Stage P cells.

- [ ] **Step 5: Build the deterministic renderer**

The cell sets `SESSION_DEADLINE = time.time() + 10800` before upload/install, creates an exclusive run root, safely extracts its runtime, calls `google.colab.files.upload()` once, pin-installs CatBoost 1.2.10 within the deadline, requires one `(7, 5)` or better CUDA device, streams logs, downloads emergency resume at most every 20 minutes, downloads final delivery once, and stays below 1,000,000 bytes.

Required markers:

```text
BLEND_CODE_READY
BLEND_INPUTS_VERIFIED
BLEND_DEPENDENCIES_READY
BLEND_GPU_READY
BLEND_RESUME_READY
CATBOOST_JOB_START
CATBOOST_PROGRESS
CATBOOST_SNAPSHOT_READY
CATBOOST_FOLD_RESULT
BLEND_WEIGHT_RESULT
BLEND_DECISION
BLEND_BUNDLE_SUCCESS
BLEND_DELIVERY_READY
BLEND_DOWNLOAD_REQUESTED
BLEND_ERROR stage=<stage> type=<type> message=<message>
```

- [ ] **Step 6: Test isolation, build, and commit**

Extract the embedded archive to a temp directory, import with only that path on `PYTHONPATH`, and run a fake two-job seam. Require deterministic archive/render bytes, checked-in cell equality, compile success, deadline before upload, size below 1 MB, and no Drive/GitHub/full-inference/submission code.

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_tabm_blend_colab.py \
  tests/test_catboost_tabm_blend_colab_cell.py -q
artifacts/tabm_submission_python311/bin/python \
  tools/build_catboost_tabm_blend_colab_cell.py
git diff --check
git add experiments/catboost_tabm_blend/colab.py \
  experiments/catboost_tabm_blend/runtime_inventory.py \
  experiments/catboost_tabm_blend/COLAB_CATBOOST_TABM_BLEND_CELL.py \
  tools/build_catboost_tabm_blend_colab_cell.py \
  tests/test_catboost_tabm_blend_colab.py \
  tests/test_catboost_tabm_blend_colab_cell.py
git commit -m "feat: add Colab CatBoost blend handoff"
```

Expected builder line:

```text
CATBOOST_BLEND_CELL_READY path=<absolute-path> sha256=<sha256> size_bytes=<1000000
```

## Task 8: Document and verify without an official-data run

**Files:**

- Create: `docs/CATBOOST_TABM_BLEND_COLAB.md`
- Modify: `README.md`

- [ ] **Step 1: Write the Korean runbook**

Explain the fixed folds/config/weights/gates, Stage C reuse, direct uploads, markers, 45-minute-to-2-hour estimate, 3-hour cap, 20-minute emergency downloads, rerun safety, outputs, and what the user returns. Include:

```bash
cd /Users/yonghyun/Documents/lg-aimers-9th-competition

artifacts/tabm_submission_python311/bin/python \
  tools/prepare_catboost_tabm_blend_input.py \
  --data-dir /Users/yonghyun/Documents/kaggle-lg-aimers-9th-data-upload \
  --output artifacts/catboost_tabm_blend_input.zip
```

State that the delivery is research-only and not submittable.

- [ ] **Step 2: Run focused and repository-wide verification**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_tabm_blend_contracts.py \
  tests/test_catboost_tabm_blend_metrics.py \
  tests/test_catboost_tabm_blend_inputs.py \
  tests/test_catboost_tabm_blend_training.py \
  tests/test_catboost_tabm_blend_artifacts.py \
  tests/test_catboost_tabm_blend_runner.py \
  tests/test_catboost_tabm_blend_colab.py \
  tests/test_catboost_tabm_blend_colab_cell.py \
  tests/test_catboost_preprocessing.py \
  tests/test_preprocessing_profiles.py \
  tests/test_tabm_campaign_artifacts.py -q

artifacts/tabm_submission_python311/bin/python -m pytest -q

artifacts/tabm_submission_python311/bin/python -m compileall -q \
  experiments/catboost_tabm_blend \
  experiments/catboost_preprocessing \
  experiments/independent_dl \
  experiments/tabm_campaign/artifacts.py \
  tools/prepare_catboost_tabm_blend_input.py \
  tools/build_catboost_tabm_blend_colab_cell.py

rg -n "test\.csv|sample_submission|submission\.zip|drive\.mount|github" \
  experiments/catboost_tabm_blend \
  tools/prepare_catboost_tabm_blend_input.py \
  tools/build_catboost_tabm_blend_colab_cell.py \
  docs/CATBOOST_TABM_BLEND_COLAB.md

git diff --check
git status --short
```

Expected: all tests/compile checks pass; no official CSV or GPU run occurs; no ZIP is tracked; unrelated dirty files remain unstaged.

- [ ] **Step 3: Commit documentation only**

```bash
git add README.md docs/CATBOOST_TABM_BLEND_COLAB.md
git commit -m "docs: explain CatBoost TabM blend run"
```

## Task 9: Hand off the official run to the user

This is user-run after Tasks 1-8 pass.

- [ ] **Step 1: Give the checked-in cell and exact uploads**

```text
catboost_tabm_blend_input.zip
tabm_colab_stage_C_delivery.zip
catboost_tabm_blend_resume.zip    # optional continuation only
```

Require one Colab T4 or better GPU. Success text:

```text
BLEND_DELIVERY_READY path=/content/catboost_tabm_blend/runs/<run_id>/catboost_tabm_blend_delivery.zip sha256=<sha256>
```

On failure request the full `BLEND_ERROR stage=<stage> type=<type> message=<message>` line and newest resume ZIP.

- [ ] **Step 2: Verify evidence before full-data work**

Recursively verify all returned hashes, then report exactly one state:

- `blend_promoted`: one fixed weight passes both gates;
- `blend_rejected`: both folds complete and none passes;
- `blend_incomplete`: return the newest resume for another session.

Only `blend_promoted` opens a separate full-data CatBoost and submission-candidate design. This plan never creates either artifact.
