# TabM Champion Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restartable four-version Kaggle campaign that searches TabM under the frozen `dl_standard + hand_matchup` preprocessing contract and emits a rules-compliant final review bundle without creating a submission archive.

**Architecture:** Add a focused `experiments.tabm_campaign` package around the existing fold-fitted preprocessing and TabM adapter. Pure contracts and decisions remain dependency-light; GPU training, artifact sealing, and the inference-only runtime are isolated. One generated Kaggle cell embeds the reviewed source, accepts only the official data plus an optional prior resume bundle, advances one version, and writes review/resume evidence.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch 2.7, TabM 0.0.3, rtdl-num-embeddings 0.0.12, pytest, existing `competition_rules` guards

---

## File map

- `experiments/tabm_campaign/configs/champion_v1.json`: sealed search space, budgets, gates, and stage definitions.
- `experiments/tabm_campaign/contracts.py`: strict config, candidate, stage, and status types.
- `experiments/tabm_campaign/sampling.py`: deterministic season-proportional proxy selection.
- `experiments/tabm_campaign/cache.py`: fixed-preprocessing cache identity and reuse.
- `experiments/tabm_campaign/training.py`: minimum-epoch stopping, loss selection, preflight, deadlines, and resource evidence.
- `experiments/tabm_campaign/metrics.py`: aligned overall, fold, segment, and ensemble evidence.
- `experiments/tabm_campaign/decisions.py`: survivor, temporal, refinement, and ensemble decisions.
- `experiments/tabm_campaign/artifacts.py`: canonical manifests and atomic review/resume bundles.
- `experiments/tabm_campaign/runner.py`: versions A-D orchestration.
- `experiments/tabm_campaign/inference_runtime.py`: inference-only frozen-state runtime used by Version D.
- `experiments/tabm_campaign/KAGGLE_CELL.py`: generated one-cell handoff under Kaggle's source-size limit.
- `tools/render_tabm_campaign_kaggle_cell.py`: deterministic cell renderer.
- `tests/test_tabm_campaign_*.py`: dependency-light and synthetic regression tests.

### Task 1: Seal the campaign contract

**Files:**
- Create: `experiments/tabm_campaign/__init__.py`
- Create: `experiments/tabm_campaign/contracts.py`
- Create: `experiments/tabm_campaign/configs/champion_v1.json`
- Test: `tests/test_tabm_campaign_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
from pathlib import Path

import pytest

from experiments.tabm_campaign.contracts import (
    CampaignContractError,
    load_campaign,
)


CONFIG = Path("experiments/tabm_campaign/configs/champion_v1.json")


def test_champion_contract_has_fixed_preprocessing_and_four_versions() -> None:
    campaign = load_campaign(CONFIG)
    assert campaign.preprocessing.profile == "dl_standard"
    assert campaign.preprocessing.components == ("hand_matchup",)
    assert campaign.wall_seconds == {"A": 7200, "B": 10800, "C": 10800, "D": 7200}
    assert len(campaign.version_a_candidates) == 24
    assert {candidate.loss for candidate in campaign.version_a_candidates} == {"bce", "brier"}
    assert {candidate.scheduler for candidate in campaign.version_a_candidates} == {"plateau", "one_cycle"}
    assert campaign.seeds == (42, 2026, 3407)
    assert campaign.max_final_weights == 3


def test_contract_rejects_test_fitted_or_non_tabm_configuration(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(text.replace('"fit_scope": "official_train_only"', '"fit_scope": "test"'), encoding="utf-8")
    with pytest.raises(CampaignContractError, match="official_train_only"):
        load_campaign(bad)
```

- [ ] **Step 2: Run the tests and confirm the expected import failure**

Run: `python3 -m pytest tests/test_tabm_campaign_contracts.py -q`

Expected: collection fails because `experiments.tabm_campaign.contracts` does not exist.

- [ ] **Step 3: Add immutable contract types and strict JSON loading**

```python
@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    capacity: str
    k: int
    width: int
    blocks: int
    dropout: float
    num_embedding: str
    loss: str
    scheduler: str
    learning_rate: float
    seed: int


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    preprocessing: PreprocessingSpec
    wall_seconds: Mapping[str, int]
    proxy_max_rows: int
    seeds: tuple[int, ...]
    max_final_weights: int
    version_a_candidates: tuple[Candidate, ...]
    gates: Mapping[str, float]
```

The loader must reject duplicate JSON keys, non-finite numbers, unknown keys, any family other than `tabm`, any preprocessing other than `dl_standard + hand_matchup`, any `fit_scope` other than `official_train_only`, changed stage budgets, repeated candidate IDs, and a candidate cross-product other than 24 unique combinations.

- [ ] **Step 4: Add the sealed JSON configuration**

Encode the three capacities, two numerical embeddings, two losses, two schedulers, fixed AdamW and batch settings, two folds, nine refinement combinations, three seeds, all thresholds from the design, and stage budgets. Keep candidate IDs canonical:

```text
a__{capacity}__{embedding}__{loss}__{scheduler}__s42
```

- [ ] **Step 5: Run focused tests**

Run: `python3 -m pytest tests/test_tabm_campaign_contracts.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/tabm_campaign tests/test_tabm_campaign_contracts.py
git commit -m "feat: seal TabM champion campaign contract"
```

### Task 2: Add deterministic proxy sampling and fixed cache identity

**Files:**
- Create: `experiments/tabm_campaign/sampling.py`
- Create: `experiments/tabm_campaign/cache.py`
- Test: `tests/test_tabm_campaign_sampling.py`
- Test: `tests/test_tabm_campaign_cache.py`

- [ ] **Step 1: Write failing sampling tests**

```python
def test_proxy_sample_is_season_proportional_and_order_independent(frame):
    left = proxy_row_ids(frame, max_rows=40, seed=42)
    right = proxy_row_ids(frame.sample(frac=1.0, random_state=7), max_rows=40, seed=42)
    assert left == right
    assert len(left) == 40
    assert set(frame.loc[frame.row_id.isin(left), "season"]) == set(frame["season"])


def test_proxy_sample_does_not_read_target(frame):
    changed = frame.copy()
    changed["control_success"] = 1 - changed["control_success"]
    assert proxy_row_ids(frame, max_rows=40, seed=42) == proxy_row_ids(changed, max_rows=40, seed=42)
```

- [ ] **Step 2: Run sampling tests and verify failure**

Run: `python3 -m pytest tests/test_tabm_campaign_sampling.py -q`

Expected: import fails because `sampling.py` is absent.

- [ ] **Step 3: Implement deterministic season allocation**

Reuse the tested allocation rules from `experiments.preprocessing_campaign.budgeted_contracts`, but expose only row IDs. Hash `f"{seed}:{row_id}"` with SHA-256, allocate `max_rows` proportionally with deterministic remainder assignment, and return IDs sorted by `(season, hash, row_id)`.

- [ ] **Step 4: Write failing cache tests**

```python
def test_cache_key_changes_for_any_semantic_input(tmp_path, preprocessing_frame):
    identity = CacheIdentity.from_frames(
        preprocessing_frame, train_end_year=2023, valid_year=2024,
        spec=PreprocessingSpec("dl_standard", ("hand_matchup",)), sample_ids=("r1",)
    )
    changed = replace(identity, preprocessing_code_sha256="0" * 64)
    assert identity.digest() != changed.digest()


def test_cache_reuse_is_read_only_and_hash_checked(tmp_path, tiny_fold):
    first = materialize_fixed_cache(tmp_path, **tiny_fold)
    second = materialize_fixed_cache(tmp_path, **tiny_fold)
    assert first.reused is False
    assert second.reused is True
    assert first.array_sha256 == second.array_sha256
```

- [ ] **Step 5: Implement cache materialization around existing preprocessing**

Call `materialize_preprocessed_fold_cache` for the full fold state, derive the model-training subset by the sampled row-ID mask, and write arrays atomically. Bind reuse to raw frame hashes, row hashes, sample-ID hash, preprocessing source hash, categorical-map hash, and array hashes. Reject rather than overwrite a mismatching cache.

- [ ] **Step 6: Run focused tests**

Run: `python3 -m pytest tests/test_tabm_campaign_sampling.py tests/test_tabm_campaign_cache.py tests/test_preprocessing_profiles.py tests/test_preprocessing_feature_cache.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tabm_campaign/sampling.py experiments/tabm_campaign/cache.py tests/test_tabm_campaign_sampling.py tests/test_tabm_campaign_cache.py
git commit -m "feat: add deterministic TabM campaign caches"
```

### Task 3: Support campaign losses and reproducible embedding state

**Files:**
- Modify: `experiments/independent_dl/models/common.py`
- Modify: `experiments/independent_dl/models/tabm.py`
- Create: `experiments/tabm_campaign/model_state.py`
- Test: `tests/test_tabm_campaign_model.py`
- Test: `tests/test_independent_dl_training.py`

- [ ] **Step 1: Write failing loss and round-trip tests**

```python
def test_tabm_brier_loss_uses_member_mean_probability(fake_torch, model, x_num, x_cat, y, rows):
    adapter = TabMAdapter(loss_name="brier")
    loss = adapter.loss(model, x_num, x_cat, y, row_indices=rows)
    expected = ((model(x_num, x_cat).squeeze(-1).sigmoid().mean(1) - y) ** 2).mean()
    assert fake_torch.allclose(loss, expected)


def test_piecewise_edges_round_trip_without_training_matrix(tmp_path, train_x_num):
    state = fit_numeric_embedding_state("piecewise_linear", train_x_num)
    save_numeric_embedding_state(state, tmp_path / "numeric.json")
    restored = load_numeric_embedding_state(tmp_path / "numeric.json")
    assert restored == state
    assert build_inference_metadata(restored).train_x_num is None
```

- [ ] **Step 2: Run tests and verify the missing API failures**

Run: `python3 -m pytest tests/test_tabm_campaign_model.py -q`

Expected: tests fail because the campaign loss and serialized embedding-state APIs are absent.

- [ ] **Step 3: Extend TabM loss without changing existing callers**

Keep `TabMAdapter()` defaulting to BCE. Accept `loss_name` in the constructor and implement only `bce` and `brier`; reject every other value. BCE remains per-member logits. Brier uses the row-wise mean probability.

- [ ] **Step 4: Separate fit-time numerical state from model construction**

Add a frozen numeric embedding state containing mode, dimension, periodic settings, or piecewise edge arrays. Change `make_numeric_embeddings` to accept already-fitted edges at inference. Preserve the existing training-matrix path for older experiments and test that both paths construct identical piecewise modules.

- [ ] **Step 5: Run focused and regression tests**

Run: `python3 -m pytest tests/test_tabm_campaign_model.py tests/test_independent_dl_training.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/independent_dl/models experiments/tabm_campaign/model_state.py tests/test_tabm_campaign_model.py tests/test_independent_dl_training.py
git commit -m "feat: add reproducible TabM campaign model state"
```

### Task 4: Build restartable training with explicit deadline outcomes

**Files:**
- Create: `experiments/tabm_campaign/training.py`
- Modify: `experiments/independent_dl/training.py`
- Test: `tests/test_tabm_campaign_training.py`
- Test: `tests/test_independent_dl_training.py`

- [ ] **Step 1: Write failing deadline and stopping tests**

```python
def test_deadline_after_checkpoint_returns_inconclusive_not_pending(fake_backend, request, tmp_path):
    result = train_restartable(request, tmp_path, clock=fake_backend.clock)
    assert result.status == "inconclusive"
    assert result.completed_epochs == 3
    assert result.checkpoint_sha256


def test_patience_cannot_stop_before_minimum_epochs(curve):
    stop = EarlyStopper(min_epochs=3, patience=2)
    assert [stop.update(x) for x in curve[:2]] == [False, False]


def test_preflight_failure_rejects_only_candidate(fake_oom, candidate):
    result = preflight(candidate, fake_oom)
    assert result.status == "failed"
    assert result.failure_type == "cuda_oom"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python3 -m pytest tests/test_tabm_campaign_training.py -q`

Expected: import fails because campaign training is absent.

- [ ] **Step 3: Add minimum epochs and typed deadline completion**

Extend `TrainRequest` with `min_epochs: int = 1`. Do not raise out of the worker after a completed checkpoint. Return a typed result with `completed`, `inconclusive`, or `failed`; a deadline before the first validation checkpoint is `inconclusive` with no promotable metric. Keep legacy `fit_candidate` behavior compatible for existing callers.

- [ ] **Step 4: Add architecture preflight and resource sampling**

Build the model, execute one forward/backward micro-batch, and record parameter count, elapsed seconds, allocated/reserved GPU bytes, process RSS, device name, and dependency versions. Classify CUDA OOM and non-finite loss without terminating other candidates.

- [ ] **Step 5: Validate checkpoint identity on resume**

Checkpoint metadata must include candidate config hash, source hash, cache hash, epoch, best epoch, validation curve, optimizer/scheduler/scaler states, RNG states, and checkpoint SHA-256. Refuse resume if any binding differs.

- [ ] **Step 6: Run focused and existing scheduler tests**

Run: `python3 -m pytest tests/test_tabm_campaign_training.py tests/test_independent_dl_training.py tests/test_budgeted_preprocessing_scheduler.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tabm_campaign/training.py experiments/independent_dl/training.py tests/test_tabm_campaign_training.py tests/test_independent_dl_training.py
git commit -m "feat: add restartable TabM campaign training"
```

### Task 5: Encode all promotion and ensemble decisions as pure functions

**Files:**
- Create: `experiments/tabm_campaign/metrics.py`
- Create: `experiments/tabm_campaign/decisions.py`
- Test: `tests/test_tabm_campaign_decisions.py`

- [ ] **Step 1: Write failing decision tests**

```python
def test_version_a_keeps_best_and_axis_diversity(proxy_rows):
    survivors = choose_version_a_survivors(proxy_rows)
    assert len(survivors) == 4
    assert survivors[0].reason == "best_overall"
    assert "P2" in {row.capacity for row in survivors}
    assert any(row.capacity.startswith("P3") for row in survivors)


def test_temporal_gate_blocks_primary_regression():
    evidence = TemporalEvidence(
        candidate_id="candidate", delta_2024=0.000051, delta_2023=-0.001
    )
    assert temporal_verdict(evidence).accepted is False


def test_incomplete_refinement_retains_version_b_champion(evidence):
    evidence.pop(("refined", 2023))
    assert choose_refined_champion(evidence).source == "version_b"


def test_ensemble_needs_material_gain_and_older_fold_confirmation(ensemble_rows):
    verdict = ensemble_verdict(ensemble_rows, primary_gain=0.000029)
    assert verdict.accepted is False
    assert verdict.reason == "primary_gain_below_0_00003"
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `python3 -m pytest tests/test_tabm_campaign_decisions.py -q`

Expected: import fails because `decisions.py` is absent.

- [ ] **Step 3: Implement deterministic selection**

Define immutable evidence and verdict records:

```python
@dataclass(frozen=True)
class TemporalEvidence:
    candidate_id: str
    delta_2024: float
    delta_2023: float


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reason: str
    candidate_id: str
```

Implement stable ordering by metric, resource cost, then candidate ID. Version A selects the four declared diversity slots. Version B applies `0.70 * delta_2024 + 0.30 * delta_2023` plus per-fold gates. Version C applies both-fold refinement gates and the exact ensemble thresholds. Segment rows below 1,000 remain advisory; larger regressions above 0.00050 block promotion.

- [ ] **Step 4: Build aligned metric and segment evidence**

Require unique one-to-one `row_id` bindings and finite binary targets and probabilities. Compute overall Brier plus `game_type`, `game_month`, pitcher-known/OOV, and batter-known/OOV rows containing segment size, Brier, P2 delta, and hard-gate eligibility. Derive OOV only from the frozen training category maps.

- [ ] **Step 5: Add pairwise ensemble diagnostics**

Align predictions one-to-one by `row_id`, calculate equal means only, Brier, probability correlation, squared-error correlation, component deltas, segment deltas, member count, and measured inference sum. Reject missing, duplicate, or differently targeted rows.

- [ ] **Step 6: Run focused tests**

Run: `python3 -m pytest tests/test_tabm_campaign_decisions.py tests/test_independent_dl_evaluation.py tests/test_preprocessing_evaluation.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tabm_campaign/metrics.py experiments/tabm_campaign/decisions.py tests/test_tabm_campaign_decisions.py
git commit -m "feat: add deterministic TabM campaign decisions"
```

### Task 6: Seal canonical review and resume bundles

**Files:**
- Create: `experiments/tabm_campaign/artifacts.py`
- Test: `tests/test_tabm_campaign_artifacts.py`

- [ ] **Step 1: Write failing canonical bundle tests**

```python
def test_bundle_is_reproducible_and_hash_validated(tmp_path, stage_evidence):
    first = write_stage_bundles(tmp_path / "a", stage_evidence)
    second = write_stage_bundles(tmp_path / "b", stage_evidence)
    assert sha256(first.review.read_bytes()).hexdigest() == sha256(second.review.read_bytes()).hexdigest()
    verify_resume_bundle(first.resume)


def test_resume_rejects_modified_member(tmp_path, stage_evidence):
    bundle = write_stage_bundles(tmp_path, stage_evidence).resume
    tamper_zip_member(bundle, "stage_state.json")
    with pytest.raises(ArtifactError, match="SHA-256"):
        verify_resume_bundle(bundle)
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `python3 -m pytest tests/test_tabm_campaign_artifacts.py -q`

Expected: import fails because artifact code is absent.

- [ ] **Step 3: Implement canonical ZIP publication**

Sort member names, use one fixed ZIP timestamp, canonical compact JSON, SHA-256 every member, write to a temporary path, verify by reopening, then atomically publish. Review bundles contain metrics, curves, predictions, resources, decisions, policy evidence, and logs. Resume bundles contain only stage state, completed evidence, and required checkpoints.

- [ ] **Step 4: Enforce immutable stage transitions**

Allow `none -> A -> B -> C -> D` only. Bind each next stage to the prior manifest hash and campaign config hash. A failed candidate does not block independent jobs; pending or inconclusive evidence cannot win. Version D has no resume bundle.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m pytest tests/test_tabm_campaign_artifacts.py tests/test_budgeted_preprocessing_artifacts.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/tabm_campaign/artifacts.py tests/test_tabm_campaign_artifacts.py
git commit -m "feat: seal TabM campaign evidence bundles"
```

### Task 7: Orchestrate versions A-C on two independent GPUs

**Files:**
- Create: `experiments/tabm_campaign/runner.py`
- Test: `tests/test_tabm_campaign_runner.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_runner_advances_exactly_one_version(fake_runtime, data_dir, tmp_path):
    result = run_one_version(data_dir, tmp_path, resume_bundle=None, runtime=fake_runtime)
    assert result.version == "A"
    assert fake_runtime.started_versions == ["A"]


def test_two_gpus_run_one_independent_job_each(fake_runtime, data_dir, tmp_path):
    run_one_version(data_dir, tmp_path, runtime=fake_runtime)
    assert set(fake_runtime.concurrent_gpu_assignments()) == {0, 1}
    assert fake_runtime.shared_model_processes == 0


def test_finalization_reserve_stops_new_jobs_and_writes_bundles(fake_clock, data_dir, tmp_path):
    result = run_one_version(data_dir, tmp_path, runtime=fake_clock.runtime)
    assert result.bundles_written
    assert fake_clock.jobs_started_inside_last_600_seconds == 0
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `python3 -m pytest tests/test_tabm_campaign_runner.py -q`

Expected: import fails because the runner is absent.

- [ ] **Step 3: Implement deterministic job scheduling**

Spawn one process per GPU with `CUDA_VISIBLE_DEVICES` set to a single index. Assign jobs by expected cost and candidate ID. Stream worker logs with `WORKER[gpu:candidate]`, enforce absolute deadlines, and collect typed worker results. Never use DDP or place one model across two devices.

- [ ] **Step 4: Implement version-specific job creation**

Version A creates the sealed 24 proxy candidates. Version B creates full primary jobs for four survivors and older-fold jobs for the selected two plus P2 when needed. Version C creates nine proxy refinements, both-fold confirmations for two, primary multi-seed jobs, and only the older-fold seed jobs necessary for a verdict. Reuse exact completed evidence by hash.

- [ ] **Step 5: Emit required progress milestones**

Print the exact design milestones and a heartbeat at least every 5% of an epoch. Finish with:

```text
BUNDLE_SUCCESS version=A review=... resume=...
```

or:

```text
TABM_CAMPAIGN_ERROR version=... candidate=... type=... message=...
```

- [ ] **Step 6: Run focused tests**

Run: `python3 -m pytest tests/test_tabm_campaign_runner.py tests/test_budgeted_preprocessing_orchestration.py tests/test_budgeted_preprocessing_scheduler.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/tabm_campaign/runner.py tests/test_tabm_campaign_runner.py
git commit -m "feat: orchestrate staged TabM champion search"
```

### Task 8: Add Version D final fit and inference-only review runtime

**Files:**
- Create: `experiments/tabm_campaign/inference_runtime.py`
- Create: `experiments/tabm_campaign/final_review.py`
- Create: `experiments/tabm_campaign/dependency_probe.py`
- Modify: `experiments/tabm_campaign/runner.py`
- Test: `tests/test_tabm_campaign_inference.py`
- Test: `tests/test_tabm_campaign_final_review.py`

- [ ] **Step 1: Write failing inference isolation tests**

```python
def test_inference_module_has_no_fit_surface():
    source = Path("experiments/tabm_campaign/inference_runtime.py").read_text(encoding="utf-8")
    assert "fit_preprocessor" not in source
    assert "optimizer" not in source
    assert "backward(" not in source


def test_row_independence_with_fp32_tolerance(frozen_predictor, evaluation_frame):
    report = audit_frozen_predictor(frozen_predictor, evaluation_frame)
    assert report.max_abs_probability_delta <= 1e-6
    assert report.features_exact
    assert report.state_digest_before == report.state_digest_after
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `python3 -m pytest tests/test_tabm_campaign_inference.py tests/test_tabm_campaign_final_review.py -q`

Expected: import fails because the Version D modules are absent.

- [ ] **Step 3: Implement frozen FP32 inference**

Load a canonical manifest, verify all hashes before deserialization, reconstruct frozen preprocessing and categorical maps, load serialized numerical embedding state and one to three TabM weights, call `eval()` and `torch.inference_mode()`, disable autocast, average members row by row, and return probabilities bound to input `row_id`. The module accepts no training frame or fit option.

- [ ] **Step 4: Implement fixed-epoch final training**

For each selected member, calculate `round(median(best_epoch + 1))`, clip to `[2, 40]`, fit preprocessing on all 2019-2024 official training rows, train exactly that many epochs without validation/test access, and serialize only model state plus frozen inference state. Never serialize optimizer state into the final review artifact.

- [ ] **Step 5: Implement row-independence and scale audits**

Check singleton, full, reverse, deterministic shuffle, and batch sizes 1, 257, 2048, and production size. Require exact encoded features and maximum FP32 probability delta `1e-6`. Build the 245,789-row scale frame by repeating only public sample features and replacing each repeated `row_id` with `synthetic_{index:06d}`. Record elapsed time, peak RSS, peak allocated/reserved VRAM, output schema, and state hashes.

- [ ] **Step 6: Prohibit package creation**

Version D writes only `tabm_hand_matchup_final_review_bundle.zip`. Assert no `submit.zip`, no `script.py` at an archive root, and no call to `submission.package` is reachable from the campaign.

- [ ] **Step 7: Add clean dependency and resource gates**

Create a temporary Python 3.11 virtual environment with `--system-site-packages` to mirror the evaluation server's base packages, install exactly the two candidate requirements while timing the operation, import a copied frozen inference runtime, and run the five-row sample. Fail the Version D acceptance record if installation exceeds 480 seconds, synthetic inference exceeds 480 seconds, peak GPU allocation exceeds 20 GiB, peak RSS exceeds 22 GB, or projected compressed artifacts exceed 2 GB. Save commands, versions, return codes, elapsed times, and output hashes in the review evidence.

- [ ] **Step 8: Bind Version D to the reviewed rules policy**

Call the existing `competition_rules` contract and code gates before final fitting and again against the inference-only source. Store the policy version and SHA-256 from `reports/rules/2026-08-13-policy-review.json`. Mark the artifact `review_only` and require a new same-day live official-policy review in the later packaging task.

- [ ] **Step 9: Run focused rules tests**

Run: `python3 -m pytest tests/test_tabm_campaign_inference.py tests/test_tabm_campaign_final_review.py tests/test_rules_code_gate.py tests/test_row_independence_evidence.py tests/test_submission_runtime.py -q`

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add experiments/tabm_campaign/inference_runtime.py experiments/tabm_campaign/final_review.py experiments/tabm_campaign/dependency_probe.py experiments/tabm_campaign/runner.py tests/test_tabm_campaign_inference.py tests/test_tabm_campaign_final_review.py
git commit -m "feat: add rules-safe TabM final review runtime"
```

### Task 9: Generate the single Kaggle cell

**Files:**
- Create: `experiments/tabm_campaign/requirements-kaggle.txt`
- Create: `tools/render_tabm_campaign_kaggle_cell.py`
- Generate: `experiments/tabm_campaign/KAGGLE_CELL.py`
- Test: `tests/test_tabm_campaign_kaggle_cell.py`

- [ ] **Step 1: Write failing renderer tests**

```python
def test_generated_cell_is_self_contained_and_under_kaggle_limit():
    path = Path("experiments/tabm_campaign/KAGGLE_CELL.py")
    assert path.stat().st_size < 950_000
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_generated_cell_requires_only_official_data_and_optional_resume():
    text = Path("experiments/tabm_campaign/KAGGLE_CELL.py").read_text(encoding="utf-8")
    assert "lg-aimers-9th-data" in text
    assert "tabm_search_stage_" in text
    assert "git clone" not in text
    assert "github.com" not in text
    assert "submit.zip" not in text
```

- [ ] **Step 2: Run tests and verify missing generated cell**

Run: `python3 -m pytest tests/test_tabm_campaign_kaggle_cell.py -q`

Expected: tests fail because the renderer and cell are absent.

- [ ] **Step 3: Add minimal Kaggle requirements**

```text
tabm==0.0.3
rtdl-num-embeddings==0.0.12
```

- [ ] **Step 4: Implement deterministic embedded-source rendering**

Create a gzip-compressed tar containing only the campaign package plus required existing local modules. Base64-embed it, verify its SHA-256 in the cell, extract under `/kaggle/working/tabm_campaign_runtime`, install the two requirements, locate exactly one official data root, locate zero or one compatible resume bundle, and call `run_one_version`. Use no GitHub or runtime download.

- [ ] **Step 5: Add clear startup and completion logs**

The cell must print data path, code hash, dependency versions, two-GPU readiness, selected version, wall deadline, every job milestone, and bundle paths. Input ambiguity must end with `TABM_CAMPAIGN_ERROR stage=input ...`.

- [ ] **Step 6: Render twice and verify byte identity**

Run: `python3 tools/render_tabm_campaign_kaggle_cell.py && shasum -a 256 experiments/tabm_campaign/KAGGLE_CELL.py > /tmp/tabm-cell-1 && python3 tools/render_tabm_campaign_kaggle_cell.py && shasum -a 256 -c /tmp/tabm-cell-1`

Expected: `experiments/tabm_campaign/KAGGLE_CELL.py: OK`.

- [ ] **Step 7: Run cell and repository contract tests**

Run: `python3 -m pytest tests/test_tabm_campaign_kaggle_cell.py tests/test_rules_repository_enforcement.py tests/test_experiment_contract_gate.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add experiments/tabm_campaign/requirements-kaggle.txt experiments/tabm_campaign/KAGGLE_CELL.py tools/render_tabm_campaign_kaggle_cell.py tests/test_tabm_campaign_kaggle_cell.py
git commit -m "feat: generate TabM champion Kaggle cell"
```

### Task 10: Run static verification and write the user handoff

**Files:**
- Create: `docs/TABM_CHAMPION_KAGGLE.md`
- Modify: `README.md`

- [ ] **Step 1: Write the execution handoff**

Document:

- purpose: advance exactly one of versions A-D;
- required input: official `lg-aimers-9th-data` and, for B-D, the immediately preceding resume bundle as a private Kaggle dataset;
- accelerator: T4 x2;
- expected maximum runtime: A 2h, B 3h, C 3h, D 2h;
- rerun safety: same hashes resume completed epochs and never overwrite immutable bundles;
- outputs: A-C review/resume ZIPs, D final review ZIP only;
- success text: `BUNDLE_SUCCESS version=...`;
- error text to return: the complete `TABM_CAMPAIGN_ERROR ...` line plus the final 100 log lines;
- explicit warning: do not submit or rename any review bundle as `submit.zip`.

- [ ] **Step 2: Run syntax, generated-artifact, and dependency-light tests**

Run:

```bash
python3 -m compileall -q experiments/tabm_campaign tools/render_tabm_campaign_kaggle_cell.py
python3 -m pytest \
  tests/test_tabm_campaign_contracts.py \
  tests/test_tabm_campaign_sampling.py \
  tests/test_tabm_campaign_cache.py \
  tests/test_tabm_campaign_model.py \
  tests/test_tabm_campaign_training.py \
  tests/test_tabm_campaign_decisions.py \
  tests/test_tabm_campaign_artifacts.py \
  tests/test_tabm_campaign_runner.py \
  tests/test_tabm_campaign_inference.py \
  tests/test_tabm_campaign_final_review.py \
  tests/test_tabm_campaign_kaggle_cell.py -q
```

Expected: compile exits 0 and all campaign tests pass.

- [ ] **Step 3: Run affected regression and rules tests**

Run:

```bash
python3 -m pytest \
  tests/test_independent_dl_training.py \
  tests/test_preprocessing_profiles.py \
  tests/test_preprocessing_feature_cache.py \
  tests/test_budgeted_preprocessing_scheduler.py \
  tests/test_rules_policy.py \
  tests/test_rules_code_gate.py \
  tests/test_rules_repository_enforcement.py \
  tests/test_row_independence_evidence.py \
  tests/test_submission_runtime.py \
  tests/test_submission_package.py -q
```

Expected: all affected regression and rules tests pass.

- [ ] **Step 4: Confirm the repository cannot create a campaign submission archive**

Run: `rg -n "submit\\.zip|create_submission|package_submission" experiments/tabm_campaign tools/render_tabm_campaign_kaggle_cell.py`

Expected: no executable campaign path creates or packages `submit.zip`; the documentation warning and negative tests may mention the name.

- [ ] **Step 5: Confirm the working tree scope**

Run: `git diff --check && git status --short`

Expected: no whitespace error and only the intended campaign, test, tool, and documentation files are modified.

- [ ] **Step 6: Commit**

```bash
git add docs/TABM_CHAMPION_KAGGLE.md README.md
git commit -m "docs: add TabM champion Kaggle handoff"
```

## Full-data execution boundary

Codex does not run the full official data, GPU training, clean remote install,
245,789-row benchmark, or final ZIP evaluation. After implementation and local
synthetic verification, the user runs each Kaggle Save Version and returns both
bundles for A-C and the final review bundle for D. Submission packaging remains
a separate explicit task after candidate acceptance, every evidence gate, and
reviewed artifact hashes pass.
