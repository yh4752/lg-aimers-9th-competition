# TabM Row Feature Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rule-safe, resumable Colab Stage P campaign that compares six row-local feature bundles against the frozen Stage C TabM baseline on paired seed-42 and seed-3407 temporal proxies, and emits review/resume evidence only.

**Architecture:** Keep the existing TabM training, preprocessing-state fitting, deterministic sampling, checkpointing, and subprocess GPU runtime. Add one pure row-feature module, thread a sealed `feature_bundle` identity through jobs and caches, place proxy selection in a pure decision module, and wrap the campaign in a strict contract plus a direct-upload Colab entry point. The new path must never read test data and must not contain submission packaging or inference code.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch/TabM through the existing adapters, pytest, standard-library JSON/ZIP/hash/subprocess utilities, Google Colab T4.

---

## Fixed scope and acceptance boundary

- Implement Stage P only: 14 jobs, comprising one baseline and six single feature bundles for each of seeds `42` and `3407`.
- Use the frozen Stage C P2 model settings, a deterministic 400,000-row train sample through 2023, and the complete 2024 validation fold.
- Produce `tabm_row_feature_stage_P_review_bundle.zip`, `tabm_row_feature_stage_P_resume_bundle.zip`, and a final delivery ZIP containing both bundles and the session log.
- Do not read or package `test.csv`, do not produce evaluation predictions, and do not create a DACON submission ZIP.
- Codex runs fixture/static tests only. The user performs the official-data and GPU run.
- Defer full two-fold Stage V and greedy combinations until a real Stage P review bundle contains at least one survivor.

## Task 1: Implement pure row-local feature bundles

**Files:**

- Create: `experiments/independent_dl/row_features.py`
- Create: `tests/test_row_features.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add exact-value tests for all six bundles**

Use the existing `preprocessing_frame` fixture and extend it with `inning`, `outs_before`, and runner flags. Write one focused assertion set per bundle. The public surface is:

```python
from experiments.independent_dl.row_features import (
    ROW_FEATURE_BUNDLES,
    RowFeatureError,
    add_row_feature_bundle,
    row_segment_labels,
)

assert ROW_FEATURE_BUNDLES == (
    "count_context",
    "pressure_context",
    "hand_state_interactions",
    "pitcher_batter_gap",
    "recent_trend",
    "pitchmix_shape",
)
```

Test these exact derived names so the contract cannot silently drift:

```python
EXPECTED_COLUMNS = {
    "count_context": {
        "rf_count_state", "rf_count_out_state", "rf_base_out_state",
    },
    "pressure_context": {
        "rf_inning_bucket", "rf_pitcher_score_bucket", "rf_leverage_bucket",
        "rf_pressure_state", "rf_pitcher_team_win_expectancy",
    },
    "hand_state_interactions": {
        "rf_hand_count_state", "rf_hand_base_state", "rf_game_hand_matchup",
    },
    "pitcher_batter_gap": {
        "rf_success_gap", "rf_middle_gap", "rf_log_count_gap",
    },
    "recent_trend": {
        "rf_success_prev1_prev5", "rf_success_prev3_prev5",
        "rf_success_prev1_career", "rf_middle_prev1_prev5",
        "rf_middle_prev3_prev5", "rf_middle_prev1_career",
    },
    "pitchmix_shape": {
        "rf_pitchmix_max", "rf_pitchmix_min", "rf_pitchmix_top2_margin",
        "rf_pitchmix_entropy", "rf_fastball_breaking_gap",
        "rf_fastball_offspeed_gap", "rf_breaking_offspeed_gap",
    },
}
```

- [ ] **Step 2: Test independence and failure behavior before implementation**

Add tests proving that each output for an existing row is identical after row reversal, unrelated-row insertion, single-row transform, and batch splitting. Drop `control_success` before calling the feature function and assert the same output. Also test:

- missing source columns raise `RowFeatureError` naming the bundle and columns;
- an unknown bundle is rejected;
- negative `asof_pitcher_n`, `asof_batter_n`, or pitchmix rates are rejected;
- non-numeric values in required numeric columns are rejected;
- missing pitchmix inputs and a zero pitchmix total return all-NaN pitchmix derivatives;
- output names cannot overwrite existing columns;
- `row_segment_labels` uses only its row and explicitly supplied train-ID sets.

Run the tests and confirm they fail because the module does not exist:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_row_features.py -q
```

Expected: collection fails with `ModuleNotFoundError: experiments.independent_dl.row_features`.

- [ ] **Step 3: Implement the pure derivation module**

Keep validation and derivation in this module; it must not import model, cache, sampling, or target code. Use explicit source maps and a single numeric parser:

```python
class RowFeatureError(ValueError):
    """Raised when a row-local feature cannot be derived safely."""


ROW_FEATURE_BUNDLES = (
    "count_context",
    "pressure_context",
    "hand_state_interactions",
    "pitcher_batter_gap",
    "recent_trend",
    "pitchmix_shape",
)


def add_row_feature_bundle(frame: pd.DataFrame, bundle: str) -> pd.DataFrame:
    if bundle not in ROW_FEATURE_BUNDLES:
        raise RowFeatureError(f"unknown row feature bundle: {bundle}")
    result = frame.copy()
    _DERIVERS[bundle](result)
    if not result.index.equals(frame.index):
        raise RowFeatureError("row feature derivation changed row alignment")
    return result
```

Implementation rules:

- categorical joins use `astype("string").fillna("__MISSING__")`;
- `count_context` reads `balls_before`, `strikes_before`, `outs_before`, and `base_state`;
- `pressure_context` reads `inning`, `score_diff_pitcher_team`, `li`, `top_bottom`, `home_win_expectancy`, and `away_win_expectancy`;
- `hand_state_interactions` reads the already derived `hand_matchup` plus `balls_before`, `strikes_before`, `base_state`, and `game_type`;
- `pitcher_batter_gap` reads `asof_pitcher_success_rate`, `asof_batter_success_rate`, `asof_pitcher_middle_rate`, `asof_batter_middle_rate`, `asof_pitcher_n`, and `asof_batter_n`;
- `recent_trend` reads the official `asof_pitcher_prev{1,3,5}_game_{success,middle}_rate` fields plus pitcher career success/middle rate;
- `pitchmix_shape` reads `asof_pitcher_fastball_rate`, `asof_pitcher_breaking_rate`, and `asof_pitcher_offspeed_rate`;
- validate count states as integers and reject negative counts;
- pressure bins are exactly `1-3`, `4-6`, `7-9`, `10+`; `<=-4`, `-3:-2`, `-1:1`, `2:3`, `>=4`; and `<0.7`, `0.7:1.5`, `>=1.5`;
- `top_bottom == "T"` selects home win expectancy for the pitcher team and `"B"` selects away; other non-missing values are rejected instead of guessed;
- differences retain NaN when either operand is missing;
- pitchmix normalizes only three finite, nonnegative values with positive row sum, then uses `-sum(p*log(p))/log(3)` with zero terms contributing zero;
- segment labels are `li_high`, `late_inning`, and `runner_in_scoring_position`; known/OOV labels take frozen fold-train ID sets as arguments.

- [ ] **Step 4: Run focused tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_row_features.py -q
```

Expected: all row-feature tests pass.

- [ ] **Step 5: Commit the pure feature layer**

```bash
git add experiments/independent_dl/row_features.py tests/test_row_features.py tests/conftest.py
git commit -m "feat: add rule-safe TabM row features"
```

## Task 2: Integrate bundles into fold-fitted preprocessing

**Files:**

- Modify: `experiments/independent_dl/preprocessing.py`
- Modify: `experiments/tabm_campaign/cache.py`
- Modify: `tests/test_preprocessing_profiles.py`
- Modify: `tests/test_tabm_campaign_cache.py`

- [ ] **Step 1: Add preprocessing tests first**

Parametrize the six bundles with `PreprocessingSpec("dl_standard", ("hand_matchup", bundle))`. Assert:

- fit and transform have identical ordered output columns;
- derived categorical columns are learned from fit rows, with unseen validation combinations mapped through the existing OOV path;
- numeric medians/means/stds are fit from train only;
- changing validation rows does not change preprocessing state;
- `control_success` is absent from source and output feature columns;
- the baseline `("hand_matchup",)` remains byte-for-byte compatible with its existing test expectations.

Add cache tests that allow exactly the baseline or baseline plus one known row bundle, and reject two row bundles or any unrelated preprocessing component.

Run and expect failure at normalization/cache validation:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_preprocessing_profiles.py tests/test_tabm_campaign_cache.py -q
```

- [ ] **Step 2: Add a sealed preprocessing dispatch**

Import `ROW_FEATURE_BUNDLES`, `RowFeatureError`, and `add_row_feature_bundle`. Append the six names to `_SIMPLE_COMPONENTS`, and delegate each selected row bundle from `_add_components`:

```python
elif component in ROW_FEATURE_BUNDLES:
    try:
        result = add_row_feature_bundle(result, component)
    except RowFeatureError as exc:
        raise PreprocessingError(str(exc)) from exc
```

Retain `hand_matchup` before row bundles through the existing component ordering. This is required by `hand_state_interactions` and makes the normalized spec deterministic.

- [ ] **Step 3: Restrict campaign cache combinations**

Replace the hardcoded equality with a focused validator:

```python
def _validate_campaign_spec(spec: PreprocessingSpec) -> None:
    if spec.profile != "dl_standard" or not spec.components:
        raise CacheError("campaign cache requires dl_standard + hand_matchup")
    if spec.components[0] != "hand_matchup":
        raise CacheError("campaign cache requires hand_matchup first")
    extras = tuple(item for item in spec.components if item != "hand_matchup")
    if len(extras) > 1 or any(item not in ROW_FEATURE_BUNDLES for item in extras):
        raise CacheError("campaign cache accepts at most one sealed row feature bundle")
```

Add `row_feature_code_sha256` to `CacheIdentity` and set it to the SHA-256 of `row_features.py`. Update cache manifest tests to require that explicit field.

- [ ] **Step 4: Run preprocessing and cache tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_row_features.py tests/test_preprocessing_profiles.py \
  tests/test_tabm_campaign_cache.py -q
```

Expected: all selected tests pass and the original baseline cache test still passes.

- [ ] **Step 5: Commit preprocessing integration**

```bash
git add experiments/independent_dl/preprocessing.py experiments/tabm_campaign/cache.py \
  tests/test_preprocessing_profiles.py tests/test_tabm_campaign_cache.py
git commit -m "feat: bind row features into TabM preprocessing"
```

## Task 3: Thread feature identity through jobs, workers, and evidence

**Files:**

- Modify: `experiments/tabm_campaign/runner.py`
- Modify: `experiments/tabm_campaign/worker.py`
- Modify: `tests/test_tabm_campaign_runner.py`
- Create: `tests/test_tabm_campaign_worker.py`

- [ ] **Step 1: Write backward-compatibility and provenance tests**

Add `feature_bundle: str | None = None` as the final `CampaignJob` field in test fixtures. Verify that:

- old JSON without the field loads as baseline through the dataclass default;
- baseline creates `PreprocessingSpec("dl_standard", ("hand_matchup",))`;
- a feature job creates `("hand_matchup", feature_bundle)`;
- unknown feature bundles fail before data materialization;
- changing only `feature_bundle` changes `_job_sha` and cache identity;
- prediction evidence contains segment columns and exact row alignment;
- a deadline result writes `worker_result.json` with `inconclusive` instead of being silently retried.

Run and capture the expected failures:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tabm_campaign_runner.py tests/test_tabm_campaign_worker.py -q
```

- [ ] **Step 2: Extend the job schema without changing existing campaigns**

```python
@dataclass(frozen=True)
class CampaignJob:
    # existing fields remain in their current order
    feature_bundle: str | None = None
```

Keep `_job_from_json` as `CampaignJob(**raw)`: the dataclass default supplies `None` for old JSON. Do not add this field to the existing champion campaign JSON; existing A-D behavior must remain unchanged.

- [ ] **Step 3: Build the preprocessing spec once in the worker**

```python
components = ("hand_matchup",)
if job.feature_bundle is not None:
    if job.feature_bundle not in ROW_FEATURE_BUNDLES:
        raise RuntimeError(f"unknown row feature bundle: {job.feature_bundle}")
    components += (job.feature_bundle,)
spec = PreprocessingSpec("dl_standard", components)
```

Pass `spec` to cache materialization. Append `feature_bundle`, `preprocessing_spec`, and cache/code hashes to `resource_evidence`.

- [ ] **Step 4: Add only predeclared segment labels to validation predictions**

Call `row_segment_labels` on `valid_rows` with fit-only pitcher and batter ID sets. Keep current `game_type`, `game_month`, and known/OOV columns. Assert `row_id`, target, and probability alignment before writing. Never aggregate validation/test rows to form a feature.

- [ ] **Step 5: Make deadline outcomes persist**

Ensure the worker CLI catches the training deadline result, serializes `CampaignJobResult(status="inconclusive", ...)`, and atomically writes `worker_result.json`. Preserve the best checkpoint and its binding. Unexpected exceptions still produce an explicit failed result and a nonzero CLI exit.

- [ ] **Step 6: Run worker/runner regression tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tabm_campaign_runner.py tests/test_tabm_campaign_worker.py \
  tests/test_tabm_campaign_training.py -q
```

Expected: all selected tests pass, including existing A-D worker fixtures.

- [ ] **Step 7: Commit job and worker integration**

```bash
git add experiments/tabm_campaign/runner.py experiments/tabm_campaign/worker.py \
  tests/test_tabm_campaign_runner.py tests/test_tabm_campaign_worker.py
git commit -m "feat: carry row feature identity through TabM jobs"
```

## Task 4: Add the immutable Stage P contract

**Files:**

- Create: `experiments/tabm_campaign/configs/row_feature_proxy_v1.json`
- Create: `experiments/tabm_campaign/row_feature_contracts.py`
- Create: `tests/test_tabm_row_feature_contracts.py`

- [ ] **Step 1: Write strict contract tests**

Test exact parsing plus rejection of duplicate JSON keys, extra/missing fields, wrong type, nonfinite numbers, shuffled/duplicate bundle lists, seeds other than `[42, 3407]`, non-proxy sample mode, and changed source hashes. The loader must reject symlinks and return immutable tuples/dataclasses.

- [ ] **Step 2: Check in the sealed JSON**

The JSON must contain these values, with no runtime defaults hidden in code:

```json
{
  "schema_version": 1,
  "stage": "P",
  "review_only": true,
  "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
  "baseline": {
    "capacity": "p2",
    "k": 32,
    "width": 512,
    "blocks": 4,
    "dropout": 0.1,
    "num_embedding": "piecewise_linear",
    "loss": "bce",
    "scheduler": "plateau",
    "learning_rate": 0.0006,
    "weight_decay": 0.0001,
    "effective_batch_size": 4096,
    "micro_batch_size": 512
  },
  "seeds": [42, 3407],
  "feature_bundles": [
    "count_context", "pressure_context", "hand_state_interactions",
    "pitcher_batter_gap", "recent_trend", "pitchmix_shape"
  ],
  "fold": {"train_end_year": 2023, "valid_year": 2024},
  "sample": {"mode": "proxy", "max_rows": 400000},
  "training": {"max_epochs": 8, "min_epochs": 3, "patience": 3},
  "budget": {"wall_seconds": 10800, "new_job_guard_seconds": 900},
  "proxy_gate": {"mean_delta_max": -0.00003, "worst_seed_delta_max": 0.00005}
}
```

The values above match candidate `a__p2__piecewise_linear__bce__plateau__s42` in the accepted Stage C config. The contract test must cross-check those fields against `champion_v1.json` and pin the resulting contract SHA-256.

- [ ] **Step 3: Implement a strict loader**

Use `json.loads(..., object_pairs_hook=...)` to detect duplicate keys. Validate the complete key set at every nesting level, `allow_nan=False` on canonical serialization, and exact equality to `ROW_FEATURE_BUNDLES`. Expose `load_row_feature_proxy_contract()` and `row_feature_contract_sha256()`.

- [ ] **Step 4: Run contract tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tabm_row_feature_contracts.py -q
```

Expected: all contract mutation tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add experiments/tabm_campaign/configs/row_feature_proxy_v1.json \
  experiments/tabm_campaign/row_feature_contracts.py \
  tests/test_tabm_row_feature_contracts.py
git commit -m "feat: seal TabM row feature proxy contract"
```

## Task 5: Implement paired proxy decisions

**Files:**

- Create: `experiments/tabm_campaign/row_feature_decisions.py`
- Create: `tests/test_tabm_row_feature_decisions.py`

- [ ] **Step 1: Write decision-table tests**

Cover these cases with small explicit Brier values:

- strong survivor: mean delta `<= -0.00003` and both seed deltas `<= +0.00005`;
- equality at both boundaries passes;
- a mean improvement with one seed worse than `+0.00005` is not strong;
- all strong survivors are retained without a count cap;
- among remaining bundles, exactly the best negative-mean bundle is the safety survivor;
- no safety survivor is added when all remaining means are nonnegative;
- missing/inconclusive baseline blocks the decision;
- an inconclusive candidate keeps the stage incomplete;
- a failed candidate is excluded without blocking independent candidates;
- duplicate candidate/seed pairs, nonfinite Brier, or values outside `[0, 1]` are rejected.

- [ ] **Step 2: Implement pure dataclasses and selection**

```python
@dataclass(frozen=True)
class ProxyMetric:
    bundle: str | None
    seed: int
    status: str
    brier: float | None


@dataclass(frozen=True)
class BundleDecision:
    bundle: str
    seed_delta: Mapping[int, float]
    mean_delta: float
    worst_seed_delta: float
    classification: str


@dataclass(frozen=True)
class ProxyDecision:
    status: str
    strong_survivors: tuple[str, ...]
    safety_survivors: tuple[str, ...]
    rows: tuple[BundleDecision, ...]
```

Sort ties by `ROW_FEATURE_BUNDLES` order. Serialize explicit decimal floats with canonical JSON; never rank by rounded display values.

- [ ] **Step 3: Run decision tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tabm_row_feature_decisions.py -q
```

Expected: all paired-decision tests pass.

- [ ] **Step 4: Commit decision logic**

```bash
git add experiments/tabm_campaign/row_feature_decisions.py \
  tests/test_tabm_row_feature_decisions.py
git commit -m "feat: decide TabM row feature proxy survivors"
```

## Task 6: Build resumable Stage P orchestration and evidence bundles

**Files:**

- Modify: `experiments/tabm_campaign/artifacts.py`
- Create: `experiments/tabm_campaign/row_feature_proxy.py`
- Create: `tests/test_tabm_row_feature_proxy.py`
- Modify: `tests/test_tabm_campaign_artifacts.py`

- [ ] **Step 1: Test exact job generation and budget ordering**

Assert the 14 deterministic IDs and order:

```text
rfp__baseline__s42
rfp__baseline__s3407
rfp__count_context__s42
rfp__count_context__s3407
rfp__pressure_context__s42
rfp__pressure_context__s3407
rfp__hand_state_interactions__s42
rfp__hand_state_interactions__s3407
rfp__pitcher_batter_gap__s42
rfp__pitcher_batter_gap__s3407
rfp__recent_trend__s42
rfp__recent_trend__s3407
rfp__pitchmix_shape__s42
rfp__pitchmix_shape__s3407
```

The runner must start both baselines before feature jobs. On a single GPU it then alternates matching bundles across seeds so paired evidence accumulates early. It must stop starting jobs at `wall_deadline - 900`, checkpoint the active job, and publish a resume bundle.

- [ ] **Step 2: Extend artifact support without changing A-D bytes**

Add independent review-only version `P`. Treat `P` like `A` for `prior_manifest_sha256=None`, but require a resume bundle. Add an optional `bundle_prefix` argument to `write_stage_bundles`, defaulting to `tabm_search_stage` so existing A-D output bytes/names remain unchanged:

```python
write_stage_bundles(
    output_dir,
    evidence,
    bundle_prefix="tabm_row_feature_stage",
)
```

Pin tests proving old A-D filenames and manifest bytes are unchanged and P emits the two required names.

- [ ] **Step 3: Test resume trust boundaries**

Fixture tests must cover:

- completed job reuse only when job/config/cache/prediction/checkpoint hashes match;
- in-progress checkpoint reuse only when model, optimizer, scheduler, epoch, job, config, and cache bindings match;
- corrupt/missing ZIP members, duplicate names, traversal, symlinks, and hash mismatches are rejected;
- an incomplete candidate remains `inconclusive`, never `failed`;
- failed candidates do not block valid completed candidates or resume creation;
- an incomplete decision omits `decisions/proxy_decision.json` and records why;
- no member name contains `submission`, `test_predictions`, or evaluation rows.

- [ ] **Step 4: Implement the focused proxy orchestrator**

Expose:

```python
def build_proxy_jobs(contract: RowFeatureProxyContract) -> tuple[CampaignJob, ...]:
    """Return the baseline-first, bundle-paired 14-job proxy schedule."""

def run_row_feature_proxy(
    *,
    data_dir: Path,
    output_dir: Path,
    resume_bundle: Path | None,
    runtime: CampaignRuntime,
    gpu_count: int,
    wall_deadline: float,
) -> RowFeatureProxyRun:
    """Run or resume Stage P and publish only verified research evidence."""
```

Reuse `SubprocessCampaignRuntime`. Keep resume extraction under a temporary directory, validate all members before publication, then copy only trusted job directories into the current output root. Write stage state atomically after every result. Review members are:

```text
config/row_feature_proxy_v1.json
metrics/job_results.json
decisions/proxy_decision.json          # complete stage only
predictions/<candidate_id>.csv         # completed candidates only
logs/stage.log
state/stage_state.json
```

Resume members contain the same state/config plus completed predictions, `worker_result.json`, `job.json`, logs, and all files needed by the current training checkpoint. Include SHA-256 for every prediction and checkpoint in `stage_state.json`.

- [ ] **Step 5: Run orchestration/artifact tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tabm_campaign_artifacts.py tests/test_tabm_row_feature_proxy.py -q
```

Expected: deterministic bundle bytes, correct 14-job graph, and all resume corruption tests pass.

- [ ] **Step 6: Commit proxy orchestration**

```bash
git add experiments/tabm_campaign/artifacts.py \
  experiments/tabm_campaign/row_feature_proxy.py \
  tests/test_tabm_campaign_artifacts.py tests/test_tabm_row_feature_proxy.py
git commit -m "feat: add resumable TabM row feature proxy stage"
```

## Task 7: Create the direct-upload, one-cell Colab handoff

**Files:**

- Create: `experiments/tabm_campaign/row_feature_colab.py`
- Create: `experiments/tabm_campaign/COLAB_ROW_FEATURE_PROXY_CELL.py`
- Create: `tools/build_tabm_row_feature_colab_cell.py`
- Create: `tools/prepare_tabm_row_feature_colab_input.py`
- Create: `tests/test_tabm_row_feature_colab.py`
- Create: `tests/test_tabm_row_feature_colab_cell.py`

- [ ] **Step 1: Test local input preparation safety**

The local preparation tool accepts a directory and writes one ZIP containing exactly:

```text
train.csv
trackman_history.csv
input_manifest.json
```

It verifies the frozen train hash, rejects duplicate candidates/symlinks/unexpected nesting, and deliberately excludes `test.csv` and `sample_submission.csv`. The manifest records size and SHA-256 for both included CSVs. It refuses to overwrite an existing output unless `--replace` is explicit.

- [ ] **Step 2: Test Colab archive and resume verification**

Add fixture archives for valid data, wrong train hash, extra test data, traversal, symlink, duplicate member, compression bomb, and truncated ZIP. Resume verification must accept only the exact Stage P contract/code/input binding. A resume from Stage A-D or another contract must be rejected before GPU work.

- [ ] **Step 3: Test one-cell and recovery behavior**

Static tests must require these markers in the generated cell:

```text
ROW_FEATURE_CODE_READY
ROW_FEATURE_INPUTS_VERIFIED
ROW_FEATURE_DEPENDENCIES_READY
ROW_FEATURE_GPU_READY
ROW_FEATURE_STAGE_SELECTED version=P
JOB_START
TRAINING_PROGRESS
EPOCH_CHECKPOINTED
ROW_FEATURE_BUNDLE_SUCCESS
ROW_FEATURE_DELIVERY_READY
ROW_FEATURE_ERROR stage=<stage> type=<type> message=<message>
```

Also assert that the cell:

- uses `google.colab.files.upload()` and never mounts Drive;
- does not access GitHub or any remote source archive;
- requires exactly one data ZIP and zero or one Stage P resume ZIP;
- streams child stdout without buffering;
- creates a new downloadable emergency resume after each completed candidate and at most every 600 seconds while a job is active;
- retains the latest valid snapshot until the next is fully verified;
- on clean completion downloads `tabm_row_feature_stage_P_delivery.zip`;
- on error attempts to publish/download the latest verified resume before raising;
- contains no test-data, inference, or submission package path.

- [ ] **Step 4: Implement the Colab supervisor**

Set the absolute 10,800-second session deadline at cell start, before upload and dependency setup, and keep a 900-second new-job guard. Verify at least one CUDA device and report its name. Setup time therefore reduces training time instead of allowing the cell to exceed the session budget. Any setup failure prints the exact stage marker. The final delivery ZIP contains only:

```text
tabm_row_feature_stage_P_review_bundle.zip
tabm_row_feature_stage_P_resume_bundle.zip
row_feature_proxy.log
delivery_manifest.json
```

`delivery_manifest.json` binds the SHA-256 of both nested bundles, the input manifest, embedded runtime, contract, and log. Verification must recursively verify both nested bundles before publication.

- [ ] **Step 5: Build an embedded cell under Kaggle/Colab source limits**

`tools/build_tabm_row_feature_colab_cell.py` creates a deterministic tar archive containing only the required runtime modules and embeds it as base64 in the one-cell file. It must reject a generated cell at or above 1,000,000 bytes and print:

```text
ROW_FEATURE_CELL_READY path=<absolute-path> sha256=<sha256> size_bytes=<n>
```

- [ ] **Step 6: Run all handoff tests and build the cell**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tabm_row_feature_colab.py tests/test_tabm_row_feature_colab_cell.py -q

artifacts/tabm_submission_python311/bin/python \
  tools/build_tabm_row_feature_colab_cell.py
```

Expected: all tests pass, then one `ROW_FEATURE_CELL_READY ... size_bytes=<1000000>` line.

- [ ] **Step 7: Commit the Colab handoff**

```bash
git add experiments/tabm_campaign/row_feature_colab.py \
  experiments/tabm_campaign/COLAB_ROW_FEATURE_PROXY_CELL.py \
  tools/build_tabm_row_feature_colab_cell.py \
  tools/prepare_tabm_row_feature_colab_input.py \
  tests/test_tabm_row_feature_colab.py tests/test_tabm_row_feature_colab_cell.py
git commit -m "feat: add Colab handoff for TabM row feature proxy"
```

## Task 8: Verify the complete implementation without full-data training

**Files:**

- Modify: `README.md`
- Create: `docs/TABM_ROW_FEATURE_PROXY_RUNBOOK.md`
- Inspect: `.gitignore` (leave unchanged because the repository already ignores ZIP/artifact outputs)

- [ ] **Step 1: Write a human-readable runbook**

Explain purpose, the six bundles, why they comply with row independence, 14-job budget, exact inputs, interruption behavior, output meaning, and what the user should return. Include one local preparation command, but no submission command:

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tabm_row_feature_colab_input.py \
  --data-dir /Users/yonghyun/Documents/kaggle-lg-aimers-9th-data-upload \
  --output artifacts/tabm_row_feature_input.zip
```

The full-data/GPU operation must be described as user-run, with an approximate runtime of one to several 3-hour sessions depending on early stopping. Rerunning with the latest valid resume ZIP must reuse completed jobs and continue an in-progress epoch checkpoint where bindings match.

- [ ] **Step 2: Run focused and repository-wide verification**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_row_features.py \
  tests/test_preprocessing_profiles.py \
  tests/test_tabm_campaign_cache.py \
  tests/test_tabm_campaign_runner.py \
  tests/test_tabm_campaign_worker.py \
  tests/test_tabm_campaign_training.py \
  tests/test_tabm_campaign_artifacts.py \
  tests/test_tabm_row_feature_contracts.py \
  tests/test_tabm_row_feature_decisions.py \
  tests/test_tabm_row_feature_proxy.py \
  tests/test_tabm_row_feature_colab.py \
  tests/test_tabm_row_feature_colab_cell.py -q

artifacts/tabm_submission_python311/bin/python -m pytest -q

artifacts/tabm_submission_python311/bin/python -m compileall -q \
  experiments/independent_dl experiments/tabm_campaign tools
```

Expected: all tests and compile checks exit zero. Do not run the official CSVs or a GPU job.

- [ ] **Step 3: Audit scope, rules, and accidental artifacts**

```bash
rg -n "test\.csv|sample_submission|submission\.zip|files\.download|drive\.mount|github" \
  experiments/tabm_campaign/row_feature_* \
  experiments/tabm_campaign/COLAB_ROW_FEATURE_PROXY_CELL.py \
  tools/prepare_tabm_row_feature_colab_input.py \
  docs/TABM_ROW_FEATURE_PROXY_RUNBOOK.md

git status --short
git diff --check
```

Expected: `test.csv`/`sample_submission` appear only in explicit rejection checks or documentation, `files.download` only in the intended review/resume delivery path, no Drive/GitHub access, no generated ZIP tracked, and no unrelated dirty file staged.

- [ ] **Step 4: Commit documentation and final verification record**

```bash
git add README.md docs/TABM_ROW_FEATURE_PROXY_RUNBOOK.md
git commit -m "docs: explain TabM row feature proxy run"
```

## Task 9: Hand off the official-data run to the user

This task is not executed by Codex. It starts only after Tasks 1-8 pass.

- [ ] **Step 1: Provide one complete copyable Colab cell**

The handoff message must state:

- purpose: compare six rule-safe row feature bundles with the frozen TabM baseline;
- required uploads: `tabm_row_feature_input.zip`, plus the latest Stage P resume ZIP when continuing;
- accelerator: one T4 GPU;
- expected output: `tabm_row_feature_stage_P_delivery.zip`;
- approximate runtime: up to 3 hours per session, possibly multiple sessions;
- rerun safety: completed jobs are reused and a binding-matched checkpoint resumes;
- success text: `ROW_FEATURE_DELIVERY_READY path=...`;
- error text to return: the full `ROW_FEATURE_ERROR ...` line and latest emergency resume ZIP.

- [ ] **Step 2: Review returned evidence before any next campaign**

Verify the delivery and nested bundle hashes locally, inspect all 14 statuses and the paired decision, and report one of:

- `proxy_survivors_found`: write a separate Stage V implementation plan;
- `proxy_incomplete`: return the newest resume for another 3-hour session;
- `proxy_no_survivor`: stop this feature direction and preserve the current seed-3407 candidate.

No submission artifact or full-data final training may be produced from Stage P evidence alone.
