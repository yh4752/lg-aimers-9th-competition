# Independent DL Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restartable, T4-oriented campaign that explores at least 64 full-scale independent TabM, MLP/ResNet, FT-Transformer, and TabR candidates without using smoke runs as performance evidence.

**Architecture:** A strict JSON contract expands four model families, four feature views, and four capacity profiles into a deterministic candidate registry. Fold-fitted feature caches feed lazy model adapters through one shared checkpointing trainer; a campaign runner resumes candidates from Drive and writes standalone, diversity, and aligned-blend diagnostics. Official full-data and GPU execution remains a user-owned Colab action.

**Tech Stack:** Python 3.11+, NumPy, pandas, PyTorch on Colab, `tabm==0.0.3`, `rtdl-revisiting-models==0.0.2`, `rtdl-num-embeddings==0.0.12`, official TabR source pinned to commit `17baa9082506f8e7a0f8d11bb1e08212926a1507`, pytest for small fake-runtime tests.

---

## File map

- `experiments/independent_dl/configs/campaign_v1.json`: sealed folds, feature views, model/training profiles, seeds, blend grid, and boundary expansion rules.
- `experiments/independent_dl/contracts.py`: strict config loading and deterministic candidate expansion.
- `experiments/independent_dl/feature_sources/`: selected cutoff-safe Trackman and seasonal feature code copied from the existing R9 implementation without redesign.
- `experiments/independent_dl/features.py`: fold-fitted typed/engineered/entity/Trackman views and memory-mapped cache artifacts.
- `experiments/independent_dl/models/common.py`: model adapter protocol, batch representation, lazy dependency errors, and shared categorical embedding.
- `experiments/independent_dl/models/{tabm,mlp_resnet,ft_transformer,tabr}.py`: family-specific official-model construction and loss/probability behavior.
- `experiments/independent_dl/training.py`: AMP training, effective-batch preservation, OOM retry, checkpoint, and resume.
- `experiments/independent_dl/evaluation.py`: Brier/fold/segment metrics, Pareto survival, error diversity, and aligned blend grids.
- `experiments/independent_dl/campaign.py`: sequential state machine and Drive artifact layout.
- `experiments/independent_dl/run_campaign.py`: one CLI entry point used by Colab.
- `experiments/independent_dl/requirements-colab.txt`: user-installed runtime dependencies only.
- `experiments/independent_dl/COLAB.md`: one complete copyable user-run cell; no notebook file changes.
- `tests/test_independent_dl_*.py`: small synthetic and fake-runtime tests only.

## Task 1: Seal and expand the campaign contract

**Files:**
- Create: `experiments/independent_dl/__init__.py`
- Create: `experiments/independent_dl/contracts.py`
- Create: `experiments/independent_dl/configs/campaign_v1.json`
- Create: `tests/test_independent_dl_contract.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from pathlib import Path

import pytest

from experiments.independent_dl.contracts import CampaignContractError, load_campaign


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/independent_dl/configs/campaign_v1.json"


def test_campaign_expands_four_families_four_views_and_64_candidates():
    campaign = load_campaign(CONFIG)
    assert campaign.campaign_id == "independent_dl_campaign_v1"
    assert {candidate.family for candidate in campaign.candidates} == {
        "tabm", "mlp_resnet", "ft_transformer", "tabr"
    }
    assert {candidate.feature_view for candidate in campaign.candidates} == {
        "raw_typed", "engineered", "entity_context", "trackman_augmented"
    }
    assert len(campaign.candidates) == 64
    assert len({candidate.candidate_id for candidate in campaign.candidates}) == 64


def test_campaign_contains_full_scale_and_boundary_expansion_contracts():
    campaign = load_campaign(CONFIG)
    assert campaign.exploration_fold == (2023, 2024)
    assert campaign.oof_folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert max(c.epochs for c in campaign.candidates) == 400
    assert campaign.confirmation_seeds == (2026, 3407)
    assert campaign.boundary_expansion["tabm"]["k"] == (96, 128)
    assert campaign.boundary_expansion["mlp_resnet"]["width"] == (3072, 4096)
    assert campaign.boundary_expansion["ft_transformer"]["layers"] == (16, 20)
    assert campaign.boundary_expansion["tabr"]["retrieval"] == (384, 512)
    assert {
        candidate.model["architecture"]
        for candidate in campaign.candidates
        if candidate.family == "mlp_resnet"
    } == {"mlp", "resnet"}


@pytest.mark.parametrize("payload", ['{"campaign_id":NaN}', '{"a":1,"a":2}'])
def test_campaign_rejects_nonfinite_and_duplicate_json(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(CampaignContractError):
        load_campaign(path)
```

- [ ] **Step 2: Run the focused test and observe the missing module failure**

Run: `python -m pytest tests/test_independent_dl_contract.py -q`

Expected: collection fails with `ModuleNotFoundError: experiments.independent_dl`.

- [ ] **Step 3: Implement strict config dataclasses and expansion**

Define immutable `CandidateSpec` and `CampaignSpec` dataclasses. `load_campaign(path)` must use `object_pairs_hook` to reject duplicate keys and `parse_constant` to reject `NaN`/`Infinity`, then validate exact top-level keys before producing candidates in family → feature view → capacity profile order.

```python
@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    feature_view: str
    seed: int
    epochs: int
    model: Mapping[str, object]
    training: Mapping[str, object]


@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str
    protocol: str
    exploration_fold: tuple[int, int]
    oof_folds: tuple[tuple[int, int], ...]
    confirmation_seeds: tuple[int, ...]
    blend_weights: tuple[float, ...]
    boundary_expansion: Mapping[str, Mapping[str, tuple[int, ...]]]
    candidates: tuple[CandidateSpec, ...]
```

The JSON must define exactly four capacity profiles per family. Their increasing profiles are:

```text
tabm:          (k,width,blocks,epochs) = (16,256,3,100), (32,512,4,160), (64,768,6,240), (64,1024,8,400)
mlp_resnet:    (architecture,width,blocks,epochs) = (mlp,512,4,100), (resnet,1024,8,160), (mlp,1536,12,240), (resnet,2048,16,400)
ft_transformer:(dim,layers,epochs)     = (128,3,100), (256,6,160), (384,9,240), (512,12,400)
tabr:          (retrieval,width,blocks,epochs) = (32,256,3,100), (64,512,4,160), (128,768,6,240), (256,1024,8,400)
```

All 64 initial candidates use exploration seed `42`; `2026` and `3407` are confirmation seeds, not grounds for dropping a family before confirmation. Training profiles must include AdamW/cosine, AdamW/plateau, AdamW/one-cycle, and AdamW/cosine-with-warmup across the four capacity levels, effective batch `4096`, AMP enabled, and early-stopping patience `30/40/60/80` respectively.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_independent_dl_contract.py -q`

Expected: all contract tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add experiments/independent_dl/__init__.py experiments/independent_dl/contracts.py experiments/independent_dl/configs/campaign_v1.json tests/test_independent_dl_contract.py
git commit -m "feat: define independent dl campaign"
```

## Task 2: Port only the proven cutoff-safe feature sources

**Files:**
- Create: `experiments/independent_dl/feature_sources/__init__.py`
- Create: `experiments/independent_dl/feature_sources/trackman.py`
- Create: `experiments/independent_dl/feature_sources/seasonal.py`
- Create: `experiments/independent_dl/feature_sources/hierarchical.py`
- Create: `experiments/independent_dl/feature_sources/context.py`
- Create: `experiments/independent_dl/feature_sources/interactions.py`
- Create: `experiments/independent_dl/feature_sources/profiles.py`
- Create: `tests/test_independent_dl_feature_sources.py`

- [ ] **Step 1: Write one focused cutoff and row-preservation test**

Create tiny 2021–2024 train/history frames. Give 2024 Trackman rows extreme values and build a 2023-cutoff lookup. Assert that changing only the 2024 history does not change the lookup, transformed 2024 row count/index remains unchanged, and the result contains the established `tm_` columns.

```python
def test_trackman_and_seasonal_sources_are_cutoff_bound(tiny_train, tiny_history):
    first = build_trackman_lookup(tiny_train, tiny_history, cutoff_year=2023)
    changed = tiny_history.copy()
    changed.loc[changed["season"].eq(2024), "rel_speed"] = 999.0
    second = build_trackman_lookup(tiny_train, changed, cutoff_year=2023)
    pd.testing.assert_frame_equal(first.lookup, second.lookup)
    assert first.cutoff_year == 2023
    assert first.lookup["pitcher_id"].is_unique
```

- [ ] **Step 2: Run the test and observe the missing source package**

Run: `python -m pytest tests/test_independent_dl_feature_sources.py -q`

Expected: collection fails because `feature_sources` is absent.

- [ ] **Step 3: Copy the already verified R9 implementations without behavioral redesign**

Copy the required behavior from these paths in legacy repository commit
`9454d68b93971627e3d3f613ce30be690cb5dce2`:

```text
experiments/kyh/high_score/src/kyh_high_score/r9_replay/trackman.py
experiments/kyh/high_score/src/kyh_high_score/r9_replay/seasonal.py
experiments/kyh/high_score/src/kyh_high_score/r9_replay/hierarchical.py
experiments/kyh/high_score/src/kyh_high_score/r9_replay/context.py
experiments/kyh/high_score/src/kyh_high_score/r9_replay/interactions.py
experiments/kyh/high_score/src/kyh_high_score/r9_replay/profiles.py
```

Change only package-relative imports and error text that names Round 9. Preserve cutoff filtering, pitcher matching, fixed schemas, lookup fingerprint, and SciPy lazy import. Add a module comment naming source commit `9454d68b93971627e3d3f613ce30be690cb5dce2`.

- [ ] **Step 4: Run the one focused feature-source test**

Run: `python -m pytest tests/test_independent_dl_feature_sources.py -q`

Expected: pass without reading official CSV files.

- [ ] **Step 5: Commit Task 2**

```bash
git add experiments/independent_dl/feature_sources tests/test_independent_dl_feature_sources.py
git commit -m "feat: port cutoff safe dl feature sources"
```

## Task 3: Build four fold-fitted memory-mapped feature views

**Files:**
- Create: `experiments/independent_dl/features.py`
- Create: `tests/test_independent_dl_features.py`

- [ ] **Step 1: Write the feature-state and cache tests**

```python
def test_all_views_fit_on_train_and_preserve_validation_rows(tiny_train, tiny_valid, tiny_history):
    for view in ("raw_typed", "engineered", "entity_context", "trackman_augmented"):
        state, train = fit_feature_view(tiny_train, tiny_history, view=view, cutoff_year=2023)
        valid = transform_feature_view(tiny_valid, state)
        assert train.x_num.shape[0] == len(tiny_train)
        assert valid.x_num.shape[0] == len(tiny_valid)
        assert valid.row_id.tolist() == tiny_valid["row_id"].astype(str).tolist()
        assert "control_success" not in state.numeric_columns
        assert "row_id" not in state.numeric_columns


def test_unseen_categories_use_reserved_index(tiny_train, tiny_valid, tiny_history):
    state, _ = fit_feature_view(tiny_train, tiny_history, view="entity_context", cutoff_year=2023)
    changed = tiny_valid.assign(pitcher_id=999999)
    transformed = transform_feature_view(changed, state)
    pitcher_position = state.categorical_columns.index("pitcher_id")
    assert transformed.x_cat[:, pitcher_position].tolist() == [0] * len(changed)


def test_cache_reuse_requires_identity_match(tmp_path, tiny_train, tiny_valid, tiny_history):
    first = materialize_fold_cache(tmp_path, tiny_train, tiny_valid, tiny_history, "raw_typed", 2023, 2024)
    second = materialize_fold_cache(tmp_path, tiny_train, tiny_valid, tiny_history, "raw_typed", 2023, 2024)
    assert second.reused is True
    changed = tiny_valid.iloc[::-1].reset_index(drop=True)
    with pytest.raises(FeatureContractError, match="row identity"):
        materialize_fold_cache(tmp_path, tiny_train, changed, tiny_history, "raw_typed", 2023, 2024)
```

- [ ] **Step 2: Run and observe missing `features`**

Run: `python -m pytest tests/test_independent_dl_features.py -q`

Expected: collection fails with missing `experiments.independent_dl.features`.

- [ ] **Step 3: Implement the typed feature bundle and train-only state**

```python
@dataclass(frozen=True)
class FeatureState:
    view: str
    cutoff_year: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_maps: Mapping[str, Mapping[str, int]]
    numeric_mean: tuple[float, ...]
    numeric_std: tuple[float, ...]
    engineered_payload: Mapping[str, object]
    trackman_result: TrackmanBuildResult | None
    trackman_lookup_sha256: str | None


@dataclass(frozen=True)
class FeatureBatch:
    row_id: np.ndarray
    season: np.ndarray
    game_type: np.ndarray
    x_num: np.ndarray
    x_cat: np.ndarray
    y: np.ndarray | None


@dataclass(frozen=True)
class FoldCache:
    root: Path
    train: FeatureBatch
    valid: FeatureBatch
    state: FeatureState
    reused: bool
```

`raw_typed` uses the official columns by dtype role. `engineered` adds `count_state`, pitcher-team win expectancy, hand matchup, smoothed pitcher success, entity frequency/log-frequency, score/runners interactions, and row-local rate differences copied from the successful XGBoost feature sources. `entity_context` adds deterministic categorical interaction tokens. `trackman_augmented` adds the cutoff-bound lookup from Task 2.

Fit category maps, means, standard deviations, frequencies, Trackman matching, and all target-derived snapshots on training rows only. Unknown category index is `0`; known categories start at `1`. Constant numeric columns receive scale `1.0` and are retained rather than causing a smoke failure.

- [ ] **Step 4: Implement cache persistence**

Write `x_num.npy`, `x_cat.npy`, `row_id.npy`, `season.npy`, `game_type.npy`, optional `y.npy`, `state.json`, and `identity.json` below `{cache_root}/{fold}/{view}/`. For `trackman_augmented`, write the lookup separately as `trackman_lookup.csv`; `state.json` contains only its cutoff, schema, and lowercase SHA-256, never the DataFrame itself. Reconstruct the in-memory `TrackmanBuildResult` from that bound lookup when a later transform needs it. Load arrays with `numpy.load(..., mmap_mode="r")`. Identity must contain lowercase SHA-256 for train row IDs, validation row IDs, cutoff, feature code, state payload, Trackman lookup when present, and exact schema. Write to a sibling temporary directory and publish with `os.replace`; no hard links.

- [ ] **Step 5: Run the focused feature tests**

Run: `python -m pytest tests/test_independent_dl_features.py tests/test_independent_dl_feature_sources.py -q`

Expected: all feature tests pass on tiny frames.

- [ ] **Step 6: Commit Task 3**

```bash
git add experiments/independent_dl/features.py tests/test_independent_dl_features.py
git commit -m "feat: add fold fitted dl feature caches"
```

## Task 4: Add the shared trainer and official MLP/ResNet, FT-Transformer, and TabM adapters

**Files:**
- Create: `experiments/independent_dl/models/__init__.py`
- Create: `experiments/independent_dl/models/common.py`
- Create: `experiments/independent_dl/models/mlp_resnet.py`
- Create: `experiments/independent_dl/models/ft_transformer.py`
- Create: `experiments/independent_dl/models/tabm.py`
- Create: `experiments/independent_dl/training.py`
- Create: `tests/test_independent_dl_training.py`

- [ ] **Step 1: Write fake-adapter tests for effective batch, resume, and OOM adaptation**

Use a fake adapter that records calls and raises `RuntimeError("CUDA out of memory")` once. Assert:

```python
def test_oom_reduces_microbatch_without_changing_model_or_effective_batch(tmp_path, fake_request):
    result = fit_candidate(fake_request, FakeOOMOnceAdapter(), tmp_path)
    assert result.model_config == fake_request.model_config
    assert result.attempted_micro_batches == (512, 256)
    assert result.effective_batch_size == 4096


def test_resume_starts_after_last_complete_checkpoint(tmp_path, fake_request):
    write_fake_checkpoint(tmp_path, epoch=7, candidate_id=fake_request.candidate_id)
    result = fit_candidate(fake_request, FakeRecordingAdapter(), tmp_path)
    assert result.started_epoch == 8
```

- [ ] **Step 2: Run and observe missing trainer failure**

Run: `python -m pytest tests/test_independent_dl_training.py -q`

Expected: collection fails with missing `training`.

- [ ] **Step 3: Implement the common adapter protocol and trainer**

```python
class ModelAdapter(Protocol):
    def build(self, spec: CandidateSpec, metadata: FeatureMetadata, device: str) -> object:
        raise NotImplementedError

    def loss(self, model: object, x_num: object, x_cat: object, y: object) -> object:
        raise NotImplementedError

    def probabilities(self, model: object, x_num: object, x_cat: object) -> object:
        raise NotImplementedError

    def optimizer(self, model: object, spec: CandidateSpec) -> object:
        raise NotImplementedError


@dataclass(frozen=True)
class TrainRequest:
    candidate_id: str
    model_config: Mapping[str, object]
    training_config: Mapping[str, object]
    train: FeatureBatch
    valid: FeatureBatch


@dataclass(frozen=True)
class TrainResult:
    candidate_id: str
    best_epoch: int
    best_brier: float
    checkpoint: Path
    predictions: np.ndarray
    attempted_micro_batches: tuple[int, ...]
    effective_batch_size: int
    model_config: Mapping[str, object]
    started_epoch: int
```

`fit_candidate` must use FP16 autocast and `GradScaler` on CUDA, accumulate gradients to the fixed effective batch, evaluate Brier at each epoch, and save model/optimizer/scheduler/scaler/RNG/epoch/best metric atomically. OOM retry order is micro-batch halving, gradient accumulation increase, then activation checkpoint flag; it must rebuild the same model config and never reduce width, blocks, k, token dimension, or retrieval count.

- [ ] **Step 4: Implement the three official adapters with lazy imports**

- MLP/ResNet: use `rtdl_revisiting_models.MLP` and `ResNet`; prepend a shared embedding layer for high-cardinality categorical columns.
- FT-Transformer: use `rtdl_revisiting_models.FTTransformer` and `model.make_parameter_groups()` for AdamW.
- TabM: use `tabm.TabM.make`; train the `k` logits independently with mean member BCE, then average member probabilities at inference as required by the official package.
- Numerical embedding profiles use `rtdl_num_embeddings` only when enabled by the candidate.

All imports occur inside adapter construction. A missing dependency raises `DLRuntimeDependencyError` with the exact `requirements-colab.txt` install command; contract and campaign inspection must work without Torch.

- [ ] **Step 5: Run only the fake-runtime and contract tests**

Run: `python -m pytest tests/test_independent_dl_training.py tests/test_independent_dl_contract.py -q`

Expected: pass without downloading PyTorch or the model packages.

- [ ] **Step 6: Commit Task 4**

```bash
git add experiments/independent_dl/models experiments/independent_dl/training.py tests/test_independent_dl_training.py
git commit -m "feat: add resumable dl training adapters"
```

## Task 5: Add the fold-safe official TabR adapter

**Files:**
- Create: `experiments/independent_dl/models/tabr.py`
- Create: `experiments/independent_dl/models/tabr_official/NOTICE.md`
- Create: `tests/test_independent_dl_tabr.py`

- [ ] **Step 1: Write fake retrieval-boundary tests**

```python
def test_tabr_context_contains_only_training_rows(fake_tabr_runtime, train_batch, valid_batch):
    adapter = TabRAdapter(runtime=fake_tabr_runtime)
    adapter.fit_context(train_batch)
    adapter.probabilities(object(), valid_batch.x_num, valid_batch.x_cat)
    assert fake_tabr_runtime.context_row_ids == train_batch.row_id.tolist()
    assert not set(valid_batch.row_id).intersection(fake_tabr_runtime.context_row_ids)


def test_tabr_never_uses_query_as_its_own_neighbor(fake_tabr_runtime, train_batch):
    adapter = TabRAdapter(runtime=fake_tabr_runtime)
    adapter.fit_context(train_batch)
    adapter.loss(object(), train_batch.x_num, train_batch.x_cat, train_batch.y)
    assert fake_tabr_runtime.self_neighbor_masked is True
```

- [ ] **Step 2: Run and observe missing TabR adapter**

Run: `python -m pytest tests/test_independent_dl_tabr.py -q`

Expected: collection fails because `models.tabr` is absent.

- [ ] **Step 3: Adapt the official implementation at the pinned commit**

Use only model/retrieval behavior from `https://github.com/yandex-research/tabular-dl-tabr` commit `17baa9082506f8e7a0f8d11bb1e08212926a1507`. Record the upstream URL, commit, paper title, and upstream MIT license in `NOTICE.md`. Keep the context candidate pool strictly equal to the current fold's training rows. Validation and test rows are queries only. During training, mask each query's own row before top-k selection.

Expose the same `ModelAdapter` methods as Task 4. Retrieve in fixed query and candidate chunks so retrieval `256` and boundary-expanded `512` do not require a full query-by-1.47M distance matrix. Candidate/context embeddings may be refreshed and checkpointed, but no label or row from a later fold can enter the context.

- [ ] **Step 4: Run the fake retrieval tests**

Run: `python -m pytest tests/test_independent_dl_tabr.py tests/test_independent_dl_training.py -q`

Expected: pass without cloning the official repository or running GPU retrieval.

- [ ] **Step 5: Commit Task 5**

```bash
git add experiments/independent_dl/models/tabr.py experiments/independent_dl/models/tabr_official/NOTICE.md tests/test_independent_dl_tabr.py
git commit -m "feat: add fold safe tabr adapter"
```

## Task 6: Evaluate standalone performance, diversity, and aligned blends

**Files:**
- Create: `experiments/independent_dl/evaluation.py`
- Create: `tests/test_independent_dl_evaluation.py`

- [ ] **Step 1: Write metric, alignment, and survival tests**

```python
def test_blend_requires_exact_row_and_fold_alignment(oof_frame, ml_frame):
    with pytest.raises(EvaluationError, match="row alignment"):
        evaluate_aligned_blends(oof_frame, ml_frame.iloc[::-1], weights=(0.1, 0.5, 0.9))


def test_survivors_include_absolute_blend_segment_diversity_and_family_top(candidate_table):
    survivors = select_survivors(
        candidate_table,
        ml_brier=0.2480,
        proximity_delta=0.0020,
        max_error_correlation=0.98,
        top_k_per_family=2,
    )
    assert {"absolute", "blend", "segment", "diversity", "family_top"}.issubset(
        set().union(*(row["reasons"] for row in survivors))
    )
```

- [ ] **Step 2: Run and observe missing evaluation module**

Run: `python -m pytest tests/test_independent_dl_evaluation.py -q`

Expected: collection fails with missing `evaluation`.

- [ ] **Step 3: Implement deterministic diagnostics**

Compute Brier, logloss, base rate, prediction mean, calibration gap, fold/game-type metrics, residual Pearson correlation, and fixed probability/logit blend grids. Reject duplicate/null row IDs, nonfinite probabilities, values outside `[0,1]`, unequal row sets, and fold/season mismatch before blending.

`select_survivors` uses the sealed values `proximity_delta=0.002`, `max_error_correlation=0.98`, and `top_k_per_family=2`. It retains a candidate when any approved reason applies; it does not close a family. Emit `standalone.csv`, `diversity.csv`, `blend_contribution.csv`, and `survivors.json` in deterministic candidate order.

- [ ] **Step 4: Run the focused evaluation tests**

Run: `python -m pytest tests/test_independent_dl_evaluation.py -q`

Expected: pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add experiments/independent_dl/evaluation.py tests/test_independent_dl_evaluation.py
git commit -m "feat: evaluate independent dl candidates"
```

## Task 7: Orchestrate sequential execution, resume, and boundary expansion

**Files:**
- Create: `experiments/independent_dl/campaign.py`
- Create: `experiments/independent_dl/run_campaign.py`
- Create: `tests/test_independent_dl_campaign.py`

- [ ] **Step 1: Write a two-candidate interruption/resume test**

```python
def test_campaign_resumes_without_repeating_completed_candidate(tmp_path, tiny_campaign):
    first_runtime = FakeRuntime(interrupt_after="candidate_0001")
    with pytest.raises(FakeInterruption):
        run_campaign(tiny_campaign, tmp_path, first_runtime)
    second_runtime = FakeRuntime()
    summary = run_campaign(tiny_campaign, tmp_path, second_runtime)
    assert "candidate_0001" not in second_runtime.started
    assert summary.completed == ("candidate_0001", "candidate_0002")


def test_boundary_winner_creates_larger_candidate_not_smaller_model(tmp_path, boundary_campaign):
    summary = run_campaign(boundary_campaign, tmp_path, FakeRuntime())
    expanded = [item for item in summary.registered if item.parent_candidate_id]
    assert expanded
    assert expanded[0].model["width"] > expanded[0].parent_model["width"]
```

- [ ] **Step 2: Run and observe missing campaign runner**

Run: `python -m pytest tests/test_independent_dl_campaign.py -q`

Expected: collection fails with missing `campaign`.

- [ ] **Step 3: Implement the atomic campaign state**

```python
@dataclass(frozen=True)
class CampaignSummary:
    campaign_id: str
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    pending: tuple[str, ...]
    registered: tuple[CandidateSpec, ...]
    output_root: Path
```

Maintain `campaign_manifest.json` with candidate states `pending/running/completed/failed`, attempts, checkpoint, metric path, prediction path, timestamps, and failure reason. Before running a candidate marked complete, verify its config and output hashes; otherwise return it to pending. Write candidate metrics and predictions first, then atomically mark complete. A failed candidate does not stop unrelated candidates.

After the initial 64 candidates, register boundary expansions only for winning boundary axes. After exploration, register confirmation seeds and three temporal folds only for the survivor set. `run_campaign.py` exposes:

```text
python -m experiments.independent_dl.run_campaign run --config experiments/independent_dl/configs/campaign_v1.json --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR/independent_dl_campaign_v1"
python -m experiments.independent_dl.run_campaign status --output-dir "$OUTPUT_DIR/independent_dl_campaign_v1"
python -m experiments.independent_dl.run_campaign summarize --output-dir "$OUTPUT_DIR/independent_dl_campaign_v1" --aligned-ml-oof "$ALIGNED_ML_OOF"
```

`--aligned-ml-oof` is optional. Without it, standalone/diversity diagnostics still complete and blend diagnostics are explicitly marked unavailable rather than fabricated.

- [ ] **Step 4: Run the campaign and related focused tests**

Run: `python -m pytest tests/test_independent_dl_campaign.py tests/test_independent_dl_evaluation.py tests/test_independent_dl_contract.py -q`

Expected: pass using only tiny data and fake runtimes.

- [ ] **Step 5: Commit Task 7**

```bash
git add experiments/independent_dl/campaign.py experiments/independent_dl/run_campaign.py tests/test_independent_dl_campaign.py
git commit -m "feat: orchestrate resumable dl campaign"
```

## Task 8: Add the user-owned Colab handoff without editing a notebook

**Files:**
- Create: `experiments/independent_dl/requirements-colab.txt`
- Create: `experiments/independent_dl/COLAB.md`
- Modify: `docs/ROADMAP.md`
- Create: `tests/test_independent_dl_handoff.py`

- [ ] **Step 1: Write static handoff contract tests**

```python
def test_colab_handoff_is_one_cell_and_does_not_package_submission():
    text = read_text("experiments/independent_dl/COLAB.md")
    assert text.count("```python") == 1
    for phrase in ("예상 시간", "재실행", "성공 시", "오류 시", "T4"):
        assert phrase in text
    for forbidden in ("submit.zip", "submission.zip", "build-submission", "git push"):
        assert forbidden not in text


def test_runtime_requirements_pin_official_packages():
    text = read_text("experiments/independent_dl/requirements-colab.txt")
    assert "tabm==0.0.3" in text
    assert "rtdl-revisiting-models==0.0.2" in text
    assert "rtdl-num-embeddings==0.0.12" in text
```

- [ ] **Step 2: Run and observe missing handoff files**

Run: `python -m pytest tests/test_independent_dl_handoff.py -q`

Expected: fail because the files do not exist.

- [ ] **Step 3: Write the pinned runtime requirements**

Do not pin or reinstall Colab's CUDA-enabled Torch. Use exactly
`numpy==1.26.4`, `pandas==2.2.3`, `scipy==1.16.3`,
`scikit-learn==1.8.0`, `tabm==0.0.3`,
`rtdl-revisiting-models==0.0.2`, and `rtdl-num-embeddings==0.0.12`.
The cell installs them into a child-process target directory and invokes the
campaign in a fresh child process, avoiding the already-imported NumPy/pandas
binary mismatch seen in prior sessions.

- [ ] **Step 4: Write one complete Colab cell and ownership instructions**

The cell must:

1. mount Drive;
2. validate user-provided `DATA_DIR` and `OUTPUT_DIR` under Drive;
3. clone/fetch the exact repository commit using the existing Colab secret, without printing the token;
4. verify a T4 CUDA device;
5. install the pinned requirements for the child process;
6. run `run_campaign.py run`, which creates or resumes `independent_dl_campaign_v1`;
7. print the manifest, status command, latest completed candidate, and output root;
8. print unredacted child stderr/traceback on failure.

State that the first 64-candidate T4 campaign can require many Colab sessions and days; completion time is not a rejection gate. Rerunning the same cell with the same output root resumes safely. The exact success text is `INDEPENDENT_DL_CAMPAIGN_CHECKPOINTED`; the user returns `campaign_manifest.json`, `campaign_summary.json` when available, and the full traceback on failure.

- [ ] **Step 5: Update only the current roadmap state**

Add one short note under independent DL: `code_ready` after implementation, then `waiting_for_user_run` when the single handoff cell is ready. Do not add a ledger result because no full experiment has run.

- [ ] **Step 6: Run the handoff and focused campaign tests**

Run: `python -m pytest tests/test_independent_dl_handoff.py tests/test_independent_dl_campaign.py -q`

Expected: pass without installing packages, mounting Drive, or running a model.

- [ ] **Step 7: Run proportional final verification**

Run:

```bash
python -m pytest tests/test_independent_dl_contract.py tests/test_independent_dl_feature_sources.py tests/test_independent_dl_features.py tests/test_independent_dl_training.py tests/test_independent_dl_tabr.py tests/test_independent_dl_evaluation.py tests/test_independent_dl_campaign.py tests/test_independent_dl_handoff.py -q
python -m compileall -q experiments/independent_dl
git diff --check
```

Expected: focused synthetic/fake-runtime tests pass; compile and diff checks exit `0`. Do not install CUDA packages, read official CSV files, run repository-wide unrelated tests, edit a notebook, train a model, or create a submission package.

- [ ] **Step 8: Commit Task 8**

```bash
git add experiments/independent_dl/requirements-colab.txt experiments/independent_dl/COLAB.md docs/ROADMAP.md tests/test_independent_dl_handoff.py
git commit -m "docs: hand off independent dl campaign"
```

## Execution boundary after implementation

Implementation ends at `code_ready`/`waiting_for_user_run`. The user then runs the one Colab cell on official data. Codex analyzes returned manifests and metrics and writes the next candidate expansion or OOF code. No submission ZIP is designed or created in this plan.
