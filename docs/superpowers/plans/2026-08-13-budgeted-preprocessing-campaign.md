# Budgeted Preprocessing Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one self-contained Kaggle cell that advances one of five restartable T4×2 preprocessing stages per Save Version and produces a cumulative final review bundle without running a submission workflow.

**Architecture:** Keep the existing 760-job campaign intact. Add a separate budgeted campaign whose pure contracts and decisions are independent from execution, whose two GPU workers write isolated job directories, and whose parent alone validates and merges the manifest. Reuse the existing fold-fitted preprocessing and DL/CatBoost runtimes, adding only deterministic proxy sampling, TabNet support, capped-budget evidence, stage orchestration, and sealed Kaggle-cell rendering.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch, TabM, RTDL FT-Transformer, pytorch-tabnet, CatBoost, pytest, subprocess-based GPU isolation, ZIP/CSV/JSON artifacts.

---

## File map

- Create `experiments/preprocessing_campaign/budgeted_contracts.py`: immutable job/stage/config contracts, deterministic sample IDs, stage registration.
- Create `experiments/preprocessing_campaign/budgeted_decisions.py`: model promotion, preprocessing promotion, final status decisions.
- Create `experiments/independent_dl/models/tabnet.py`: TabNet adapter compatible with the shared PyTorch trainer.
- Modify `experiments/independent_dl/models/__init__.py`: export TabNet adapter.
- Modify `experiments/independent_dl/campaign.py`: allow `tabnet` adapter lookup without changing existing campaign registration.
- Modify `experiments/independent_dl/preprocessing.py`: add a new ID frequency plus explicit OOV component without changing the old frequency component.
- Create `experiments/preprocessing_campaign/budgeted_runtime.py`: proxy/full fold materialization and one-job DL/CatBoost execution.
- Create `experiments/preprocessing_campaign/budgeted_scheduler.py`: deadline-aware two-process scheduling and parent-owned manifest.
- Create `experiments/preprocessing_campaign/budgeted_artifacts.py`: resume, per-stage review, and cumulative final review bundles.
- Create `experiments/preprocessing_campaign/run_budgeted_campaign.py`: auto-stage CLI and five-stage orchestration.
- Create `experiments/preprocessing_campaign/configs/budgeted_campaign_v1.json`: sealed model, preprocessing, budget, and decision values.
- Create `experiments/preprocessing_campaign/requirements-kaggle-budgeted.txt`: pinned runtime additions including CatBoost and pytorch-tabnet.
- Create `tools/render_budgeted_preprocessing_kaggle_cell.py`: render sealed runtime into one copyable cell.
- Create generated `experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py`: user-facing Kaggle cell.
- Create focused tests under `tests/test_budgeted_*.py`; modify existing adapter/preprocessing tests only where the new contract directly extends them.

### Task 1: Budgeted configuration and deterministic temporal proxy

**Files:**
- Create: `experiments/preprocessing_campaign/budgeted_contracts.py`
- Create: `experiments/preprocessing_campaign/configs/budgeted_campaign_v1.json`
- Test: `tests/test_budgeted_preprocessing_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_budgeted_config_registers_only_predeclared_models_and_settings():
    campaign = load_budgeted_campaign(CONFIG)
    assert campaign.campaign_id == "budgeted_preprocessing_campaign_v1"
    assert campaign.session_seconds == 6300
    assert campaign.stop_new_jobs_seconds == 900
    assert campaign.archive_reserve_seconds == 600
    assert [job.family for job in campaign.stage_jobs(1)] == [
        "tabm", "ft_transformer", "tabnet", "catboost"
    ]
    assert {job.setting_id for job in campaign.stage_jobs(2)} == {
        "dl_standard", "selective_yeo_johnson",
        "pitcher_smoothing_k100", "batter_smoothing_k250",
    }


def test_temporal_proxy_is_deterministic_proportional_and_train_only():
    frame = fixture_seasons(rows_per_season=120_000, years=range(2019, 2025))
    left = deterministic_temporal_sample(
        frame, train_end_year=2023, max_rows=400_000, seed=42
    )
    right = deterministic_temporal_sample(
        frame.sample(frac=1.0, random_state=7),
        train_end_year=2023,
        max_rows=400_000,
        seed=42,
    )
    assert len(left) == 400_000
    assert left["row_id"].tolist() == right["row_id"].tolist()
    assert left["season"].max() == 2023
    assert left.groupby("season").size().to_dict() == {
        2019: 80_000, 2020: 80_000, 2021: 80_000, 2022: 80_000, 2023: 80_000
    }
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run: `pytest -q tests/test_budgeted_preprocessing_contracts.py`

Expected: import failure for `budgeted_contracts`.

- [ ] **Step 3: Implement immutable contracts and sampling**

Define `BudgetedJob`, `BudgetedStage`, and `BudgetedCampaign` frozen dataclasses. Validate exact JSON keys, unique job IDs, stage IDs 1–5, positive budgets, `archive_reserve_seconds < stop_new_jobs_seconds < session_seconds`, allowed families, allowed folds, and fixed seed 42. Implement sampling by allocating the 400,000-row cap proportionally with largest-remainder allocation and sorting each season by `sha256(f"{seed}:{row_id}")`; sort the selected output by hash so input row order cannot change it.

The JSON must pin:

```json
{
  "campaign_id": "budgeted_preprocessing_campaign_v1",
  "session_seconds": 6300,
  "stop_new_jobs_seconds": 900,
  "archive_reserve_seconds": 600,
  "proxy_max_rows": 400000,
  "seed": 42,
  "blend_weights": [0.10, 0.25, 0.50],
  "primary_fold": [2023, 2024],
  "stability_fold": [2022, 2023]
}
```

Use the existing p2 TabM/FT configs and `level_2` training settings verbatim. Pin TabNet and CatBoost values from the design spec. Register only stage-1 and fixed single-ablation jobs in JSON; dynamically promoted stage-4/5 jobs are derived by pure decision functions, not edited into the config after results exist.

- [ ] **Step 4: Run contract tests**

Run: `pytest -q tests/test_budgeted_preprocessing_contracts.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/preprocessing_campaign/budgeted_contracts.py \
  experiments/preprocessing_campaign/configs/budgeted_campaign_v1.json \
  tests/test_budgeted_preprocessing_contracts.py
git commit -m "feat: define budgeted preprocessing stages"
```

### Task 2: Pure promotion and final decision rules

**Files:**
- Create: `experiments/preprocessing_campaign/budgeted_decisions.py`
- Test: `tests/test_budgeted_preprocessing_decisions.py`

- [ ] **Step 1: Write failing threshold tests**

```python
def test_model_tie_chooses_faster_candidate_within_two_e_minus_four():
    rows = [
        metric("tabm", brier=0.24750, seconds=2500, valid_epochs=30),
        metric("ft_transformer", brier=0.24738, seconds=2700, valid_epochs=30),
    ]
    decision = choose_dl_representative(rows, blend_rows=[])
    assert decision.selected_family == "tabm"
    assert decision.reason == "brier_tie_faster"


def test_tabnet_under_minimum_training_is_inconclusive():
    result = decide_tabnet(
        metric("tabnet", brier=0.2470, valid_epochs=9, validation_points=3),
        best_dl_brier=0.2472,
        blend_gain=0.0,
        oov_gain=0.0,
        overall_delta=-0.0002,
    )
    assert result.status == "inconclusive"


def test_preprocessing_requires_preregistered_gain_or_segment_or_blend_rule():
    assert promote_single(delta=-0.00011, oov_delta=0.0, blend_gain=0.0)
    assert promote_single(delta=-0.00004, oov_delta=-0.00021, blend_gain=0.0)
    assert promote_single(delta=0.00004, oov_delta=0.0, blend_gain=0.00011)
    assert not promote_single(delta=0.00006, oov_delta=-0.001, blend_gain=0.001)


def test_final_recommendation_requires_two_fold_direction_full_gain_and_segments():
    decision = final_preprocessing_status(
        proxy_deltas={2023: -0.00012, 2024: -0.00014},
        full_2024_delta=-0.00003,
        segment_deltas={"pitcher_oov": 0.0001, "batter_oov": 0.0, "game_type": 0.00019},
        hashes_valid=True,
    )
    assert decision == "recommended"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_budgeted_preprocessing_decisions.py`

Expected: import failure for `budgeted_decisions`.

- [ ] **Step 3: Implement pure decision dataclasses and functions**

Implement `ModelDecision`, `CandidateDecision`, `choose_dl_representative`, `decide_tabnet`, `fixed_blend_metrics`, `promote_single`, `choose_top_two`, and `final_preprocessing_status`. Use lexicographic sorting, no learned thresholds, and only blend weights from the loaded config. Reject malformed, duplicate, non-finite, hash-invalid, or row-misaligned evidence before ranking.

- [ ] **Step 4: Run decision tests**

Run: `pytest -q tests/test_budgeted_preprocessing_decisions.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/preprocessing_campaign/budgeted_decisions.py \
  tests/test_budgeted_preprocessing_decisions.py
git commit -m "feat: fix preprocessing promotion rules"
```

### Task 3: Explicit ID/OOV component and TabNet adapter

**Files:**
- Modify: `experiments/independent_dl/preprocessing.py`
- Create: `experiments/independent_dl/models/tabnet.py`
- Modify: `experiments/independent_dl/models/__init__.py`
- Modify: `experiments/independent_dl/campaign.py`
- Create: `experiments/preprocessing_campaign/requirements-kaggle-budgeted.txt`
- Test: `tests/test_independent_dl_preprocessing.py`
- Test: `tests/test_independent_dl_tabnet.py`

- [ ] **Step 1: Add failing preprocessing and adapter tests**

```python
def test_entity_frequency_and_oov_keeps_ids_and_marks_unseen_values():
    state, transformed = fit_preprocessor(
        train_frame(),
        PreprocessingSpec("dl_standard", ("entity_frequency_and_oov",)),
    )
    valid = transform_preprocessor(valid_with_unseen_ids(), state)
    assert "pitcher_id" in transformed
    assert valid.loc[0, "pitcher_id_oov"] == 1.0
    assert valid.loc[0, "pitcher_id_frequency_log1p"] == 0.0
    assert valid.loc[1, "pitcher_id_oov"] == 0.0


def test_tabnet_adapter_builds_two_input_binary_model(fake_runtime_modules):
    adapter = TabNetAdapter()
    model = adapter.build(TABNET_CONFIG, metadata(), "cpu")
    probabilities = adapter.probabilities(model, x_num(), x_cat())
    assert tuple(probabilities.shape) == (len(x_num()),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `pytest -q tests/test_independent_dl_preprocessing.py tests/test_independent_dl_tabnet.py`

Expected: unknown preprocessing component and missing TabNet adapter.

- [ ] **Step 3: Add a non-breaking preprocessing component**

Add `entity_frequency_and_oov` without altering `entity_frequency_log1p`. For pitcher and batter IDs, retain the original categorical ID and add numeric `frequency`, `frequency_log1p`, and binary `oov`. Fit counts only on supplied training rows. Treat the binary OOV columns as passthrough indicators so standardization does not erase 0/1 semantics.

- [ ] **Step 4: Implement the TabNet adapter**

Wrap `pytorch_tabnet.tab_network.TabNet` behind the shared `ModelAdapter` contract. Concatenate numerical columns with learned categorical embeddings, pass the flat matrix to the TabNet network, extract its prediction tensor, add the sparse mask loss to BCE using the fixed `lambda_sparse`, and return sigmoid probabilities. Build with the fixed design parameters and AdamW from the shared training config. Export `TabNetAdapter` and add only the adapter lookup; do not register TabNet in the existing independent-DL grid.

- [ ] **Step 5: Run adapter and preprocessing tests**

Run: `pytest -q tests/test_independent_dl_preprocessing.py tests/test_independent_dl_tabnet.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/independent_dl/preprocessing.py \
  experiments/independent_dl/models/tabnet.py \
  experiments/independent_dl/models/__init__.py \
  experiments/independent_dl/campaign.py \
  experiments/preprocessing_campaign/requirements-kaggle-budgeted.txt \
  tests/test_independent_dl_preprocessing.py tests/test_independent_dl_tabnet.py
git commit -m "feat: add TabNet preprocessing sentinel"
```

### Task 4: Proxy-aware one-job runtime

**Files:**
- Create: `experiments/preprocessing_campaign/budgeted_runtime.py`
- Modify: `experiments/independent_dl/preprocessing_campaign.py`
- Modify: `experiments/catboost_preprocessing/campaign.py`
- Test: `tests/test_budgeted_preprocessing_runtime.py`

- [ ] **Step 1: Write failing runtime tests with injected fitters**

```python
def test_proxy_runtime_fits_preprocessor_only_on_deterministic_train_sample(tmp_path):
    runtime = BudgetedRuntime(data_dir=FIXTURE, cache_root=tmp_path / "cache", fit_function=fake_fit)
    result = runtime.run_job(proxy_job("tabm", 2023, 2024), tmp_path / "job")
    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["train_rows"] == 400_000
    assert metrics["valid_rows"] == 253_507
    assert metrics["sample_sha256"] == EXPECTED_SAMPLE_HASH


def test_full_runtime_uses_every_training_row_and_records_matched_epoch_evidence(tmp_path):
    runtime = BudgetedRuntime(
        data_dir=FIXTURE,
        cache_root=tmp_path / "cache",
        fit_function=fake_fit,
    )
    result = runtime.run_job(full_job(), tmp_path / "job")
    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["sample_mode"] == "full"
    assert metrics["completed_epochs"] >= 10
    assert "best_brier_by_epoch" in metrics
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_budgeted_preprocessing_runtime.py`

Expected: import failure for `budgeted_runtime`.

- [ ] **Step 3: Expose completed-epoch evidence from the shared trainer**

Extend `TrainResult` and the saved metrics with completed epochs, validation points, best epoch, best Brier, per-epoch Brier curve, actual visible device, and peak memory. Preserve current constructors with defaults so existing tests and campaign output remain compatible.

- [ ] **Step 4: Implement budgeted DL and CatBoost job execution**

Load official CSVs lazily. Select proxy or full train rows from the job contract, keep the full validation season, materialize fold-fitted preprocessing, create the correct adapter, and write aligned predictions containing row ID, target, probability, family, setting, fold, seed, game type, pitcher OOV, and batter OOV. Add `od_type=Iter`, `od_wait=50`, and `use_best_model=True` to budgeted CatBoost only. Clip probabilities to `[0,1]` and write metrics atomically.

- [ ] **Step 5: Run runtime and regression tests**

Run: `pytest -q tests/test_budgeted_preprocessing_runtime.py tests/test_preprocessing_campaign.py tests/test_catboost_preprocessing.py tests/test_independent_dl_training.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/preprocessing_campaign/budgeted_runtime.py \
  experiments/independent_dl/preprocessing_campaign.py \
  experiments/catboost_preprocessing/campaign.py \
  tests/test_budgeted_preprocessing_runtime.py
git commit -m "feat: run budgeted temporal proxy jobs"
```

### Task 5: Two-GPU scheduler and parent-owned manifest

**Files:**
- Create: `experiments/preprocessing_campaign/budgeted_scheduler.py`
- Test: `tests/test_budgeted_preprocessing_scheduler.py`

- [ ] **Step 1: Write failing scheduler tests**

```python
def test_scheduler_pins_one_worker_per_gpu_and_parent_merges_only_valid_results(tmp_path):
    scheduler = BudgetedScheduler(process_factory=FakeProcessFactory(success=True))
    summary = scheduler.run([job("a"), job("b")], tmp_path, deadline=10_000)
    assert scheduler.started_envs == [
        {"CUDA_VISIBLE_DEVICES": "0"}, {"CUDA_VISIBLE_DEVICES": "1"}
    ]
    manifest = read_manifest(tmp_path)
    assert set(manifest["jobs"]) == {"a", "b"}
    assert {entry["state"] for entry in manifest["jobs"].values()} == {"completed"}


def test_scheduler_does_not_start_job_inside_nine_hundred_second_guard(tmp_path):
    clock = FakeClock(now=9_200)
    scheduler = BudgetedScheduler(clock=clock, process_factory=NoStartFactory())
    summary = scheduler.run([job("a")], tmp_path, deadline=10_000)
    assert summary.pending == ("a",)


def test_worker_cannot_publish_tampered_artifact(tmp_path):
    scheduler = BudgetedScheduler(process_factory=FakeProcessFactory(tamper=True))
    with pytest.raises(ArtifactValidationError, match="sha256"):
        scheduler.run([job("a")], tmp_path, deadline=10_000)
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest -q tests/test_budgeted_preprocessing_scheduler.py`

Expected: import failure for the scheduler.

- [ ] **Step 3: Implement worker protocol and atomic manifest merge**

Launch at most two `python -m experiments.preprocessing_campaign.run_budgeted_campaign worker` processes. Set one physical GPU per child, `PYTHONUNBUFFERED=1`, a per-job deadline, and an isolated output directory. Stream prefixed child output immediately. Workers write `worker_result.json` containing artifact relative paths and SHA-256 hashes. The parent validates paths cannot escape the root, hashes all artifacts, moves a fully valid temporary directory into `jobs/{job_id}`, then atomically updates the common manifest.

At 60-second intervals print `HEARTBEAT`; query `nvidia-smi` for GPU name, utilization and memory when available. Treat absence of exactly two CUDA devices as `STAGE_NEEDS_REVIEW` before any heavy job starts.

- [ ] **Step 4: Run scheduler tests**

Run: `pytest -q tests/test_budgeted_preprocessing_scheduler.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/preprocessing_campaign/budgeted_scheduler.py \
  tests/test_budgeted_preprocessing_scheduler.py
git commit -m "feat: schedule isolated T4 preprocessing jobs"
```

### Task 6: Stage decisions and restartable orchestration

**Files:**
- Create: `experiments/preprocessing_campaign/run_budgeted_campaign.py`
- Test: `tests/test_budgeted_preprocessing_orchestration.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_auto_mode_starts_stage_one_without_resume_bundle(tmp_path):
    result = run_auto(config=CONFIG, input_root=tmp_path / "input", output_root=tmp_path / "out", runtime=FakeRuntime())
    assert result.stage_id == 1


def test_auto_mode_selects_highest_hash_valid_resume_bundle(tmp_path):
    write_resume(tmp_path, stage=1)
    write_resume(tmp_path, stage=2)
    result = inspect_resume_bundles(tmp_path)
    assert result.completed_stage == 2


def test_conflicting_same_stage_bundles_need_review(tmp_path):
    write_resume(tmp_path / "a", stage=2, manifest_hash="a" * 64)
    write_resume(tmp_path / "b", stage=2, manifest_hash="b" * 64)
    with pytest.raises(StageNeedsReview, match="conflicting"):
        inspect_resume_bundles(tmp_path)


def test_stage_five_never_requires_sixth_version_for_normal_deadline(tmp_path):
    result = finalize_stage_five(baseline=metric(valid_epochs=9), candidate=metric(valid_epochs=10))
    assert result.status == "inconclusive"
    assert result.campaign_terminal is True
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest -q tests/test_budgeted_preprocessing_orchestration.py`

Expected: import failure for the orchestration module.

- [ ] **Step 3: Implement CLI actions and stage state machine**

Provide `auto`, `worker`, and `status` actions. `auto` validates config and data, captures the absolute session deadline at process start, restores the highest valid resume bundle, computes the next jobs using only sealed decisions, invokes the scheduler, evaluates the stage, writes the next stage plan, and prints exactly one terminal state. Stage 5 compares the baseline and candidate over the common completed epoch range and terminates as either a final decision or `inconclusive`.

- [ ] **Step 4: Run orchestration tests**

Run: `pytest -q tests/test_budgeted_preprocessing_orchestration.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/preprocessing_campaign/run_budgeted_campaign.py \
  tests/test_budgeted_preprocessing_orchestration.py
git commit -m "feat: orchestrate five preprocessing stages"
```

### Task 7: Resume and review bundle contracts

**Files:**
- Create: `experiments/preprocessing_campaign/budgeted_artifacts.py`
- Test: `tests/test_budgeted_preprocessing_artifacts.py`

- [ ] **Step 1: Write failing bundle tests**

```python
def test_stage_writes_separate_resume_and_small_review_bundles(tmp_path):
    result = write_stage_bundles(campaign_root=fixture_campaign(tmp_path), stage_id=1)
    assert result.resume.name == "preprocessing_stage_01_resume_bundle.zip"
    assert result.review.name == "preprocessing_stage_01_review_bundle.zip"
    with ZipFile(result.review) as archive:
        assert REQUIRED_REVIEW_NAMES <= set(archive.namelist())
        assert not any(name.endswith(".pt") or name.endswith(".cbm") for name in archive.namelist())


def test_final_review_is_cumulative_and_contains_wide_aligned_predictions(tmp_path):
    final = write_final_review(fixture_five_stages(tmp_path))
    with ZipFile(final) as archive:
        assert "validation_predictions.csv.gz" in archive.namelist()
        assert "decision_table.csv" in archive.namelist()
        assert "artifact_manifest.json" in archive.namelist()
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_budgeted_preprocessing_artifacts.py`

Expected: import failure for `budgeted_artifacts`.

- [ ] **Step 3: Implement deterministic bundles**

Resume bundles contain the campaign manifest, stage state, checkpoints and completed job artifacts. Review bundles contain only `stage_summary.json`, `decision_table.csv`, `model_metrics.csv`, `segment_metrics.csv`, `learning_curves.csv`, `resource_usage.csv`, `validation_predictions.csv.gz`, `artifact_manifest.json`, `errors.json`, and `run.log`. Align predictions one row per `row_id/fold/target` with one probability column per candidate. Record file size and SHA-256 for every member and use stable ZIP timestamps and sorted names.

- [ ] **Step 4: Run artifact tests**

Run: `pytest -q tests/test_budgeted_preprocessing_artifacts.py tests/test_preprocessing_review.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/preprocessing_campaign/budgeted_artifacts.py \
  tests/test_budgeted_preprocessing_artifacts.py
git commit -m "feat: package restartable preprocessing evidence"
```

### Task 8: Self-contained Kaggle cell

**Files:**
- Create: `tools/render_budgeted_preprocessing_kaggle_cell.py`
- Create: `experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py`
- Create: `tests/test_budgeted_preprocessing_kaggle_cell.py`

- [ ] **Step 1: Write failing rendered-cell contract tests**

```python
def test_budgeted_cell_is_one_self_contained_kaggle_cell_without_github_access():
    text = CELL.read_text(encoding="utf-8")
    assert "EMBEDDED_RUNTIME_B64" in text
    assert "MAX_SESSION_SECONDS = 6300" in text
    assert "run_budgeted_campaign" in text
    assert "github.com" not in text
    assert "git clone" not in text
    assert "preprocessing_campaign_final_review_bundle.zip" in text


def test_embedded_archive_matches_pinned_commit_and_imports_budgeted_cli(tmp_path):
    payload = decode_embedded_archive(CELL)
    assert sha256(payload).hexdigest() == embedded_sha(CELL)
    extract_safely(payload, tmp_path)
    assert (tmp_path / "experiments/preprocessing_campaign/run_budgeted_campaign.py").is_file()
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_budgeted_preprocessing_kaggle_cell.py`

Expected: generated cell does not exist.

- [ ] **Step 3: Implement renderer and cell template**

Render a gzip-compressed `git archive` from an explicit runtime commit, Base64-embed it, verify SHA-256 before extraction, reject links and path traversal, install the pinned requirements into `/kaggle/working/preprocessing_budgeted_runtime`, locate exactly one official `train.csv` and co-located `trackman_history.csv`, inspect resume bundles under `/kaggle/input`, and run the `auto` CLI with unbuffered streamed output. Always attempt a partial review bundle after an exception and print `STAGE_ERROR` with the original traceback.

- [ ] **Step 4: Generate and test the cell**

Run:

```bash
python tools/render_budgeted_preprocessing_kaggle_cell.py
python -m py_compile experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py
pytest -q tests/test_budgeted_preprocessing_kaggle_cell.py
```

Expected: compilation succeeds and all cell contract tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/render_budgeted_preprocessing_kaggle_cell.py \
  experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py \
  tests/test_budgeted_preprocessing_kaggle_cell.py
git commit -m "feat: render budgeted Kaggle preprocessing cell"
```

### Task 9: Full regression and handoff verification

**Files:**
- Modify only if verification exposes a defect in files changed by Tasks 1–8.

- [ ] **Step 1: Run syntax and complete fixture suite**

Run:

```bash
python -m compileall -q experiments tools
pytest -q
```

Expected: compilation succeeds and the entire test suite passes.

- [ ] **Step 2: Run sealed dry contract checks**

Run:

```bash
python -m experiments.preprocessing_campaign.run_budgeted_campaign status \
  --config experiments/preprocessing_campaign/configs/budgeted_campaign_v1.json \
  --dry-contract
python tools/render_budgeted_preprocessing_kaggle_cell.py
git diff --exit-code -- experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py
```

Expected: status reports five stages, two GPU workers, 6,300-second limit, no submission action; renderer is reproducible with no diff.

- [ ] **Step 3: Inspect repository scope**

Run:

```bash
git status --short
git diff --check
git log --oneline -12
```

Expected: only plan-scoped files are changed, no whitespace errors, and no notebook or submission packaging file was modified.

If verification exposes a defect, return to the task that owns that file, add a failing
regression test, make the smallest correction, rerun that task's focused tests and the complete
suite, and commit only the exact files named in that task's commit step.
