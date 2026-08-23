# Temporal Portfolio Platform and Final Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one-cell Kaggle and Colab launchers, safe recovery, T5 full fitting, frozen row-independent inference, and a review-only training delivery without creating `submit.zip`.

**Architecture:** Generated cells embed a deterministic source archive and call the common runner. T5-B is a separate full-fit state transition; frozen inference loads only approved train-fitted state; a final delivery verifier enforces hashes, runtime evidence, and explicit non-submission policy.

**Tech Stack:** Python 3.11, PyTorch, TabM, CatBoost, base64/tar runtime embedding, Kaggle T4x2, Colab T4, pytest.

---

### Task 1: Final epoch/prefix resolution and T5-B full fit

**Files:**
- Create: `experiments/temporal_portfolio/final_training.py`
- Create: `tests/test_temporal_portfolio_final.py`

- [ ] **Step 1: Write failing final-training tests**

```python
def test_final_epoch_is_temporally_weighted_median() -> None:
    assert resolve_final_epoch({2022: 8, 2023: 5, 2024: 3}) == 3
    with pytest.raises(FinalTrainingError, match="unstable"):
        resolve_final_epoch({2022: 2, 2023: 3, 2024: 40})


def test_full_fit_uses_2024_recent_and_2021_2024_multi(tiny_train, approved_t4_decision) -> None:
    plan = build_full_fit_plan(tiny_train, approved_t4_decision)
    assert plan.recent_years == (2024,)
    assert plan.multi_years == (2021, 2022, 2023, 2024)
    np.testing.assert_allclose(plan.multi_weight_by_year.values(), [0.55**3, 0.55**2, 0.55, 1.0])


def test_rejected_or_unconfirmed_candidate_cannot_full_fit(tiny_train, rejected_decision) -> None:
    with pytest.raises(FinalTrainingError, match="acceptance"):
        build_full_fit_plan(tiny_train, rejected_decision)


def test_full_test_inference_must_finish_inside_eight_minute_safety_gate(frozen_predictor, tiny_test, fake_clock) -> None:
    report = benchmark_inference(frozen_predictor, tiny_test, clock=fake_clock(elapsed=479.0))
    assert report.accepted is True
    with pytest.raises(FinalTrainingError, match="inference budget"):
        benchmark_inference(frozen_predictor, tiny_test, clock=fake_clock(elapsed=481.0))
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_final.py -q
```

Expected: FAIL on missing final training module.

- [ ] **Step 3: Implement stable epoch and approved full fit**

```python
def resolve_final_epoch(best_epoch_by_year: Mapping[int, int]) -> int:
    if set(best_epoch_by_year) != {2022, 2023, 2024}:
        raise FinalTrainingError("final epoch evidence differs")
    values = {year: int(value) for year, value in best_epoch_by_year.items()}
    if min(values.values()) < 1 or max(values.values()) > max(12, 4 * min(values.values())):
        raise FinalTrainingError("fold epochs are unstable")
    ordered = sorted((values[year], weight) for year, weight in {2022: 0.2, 2023: 0.3, 2024: 0.5}.items())
    cumulative = 0.0
    for epoch, weight in ordered:
        cumulative += weight
        if cumulative >= 0.5:
            return max(2, epoch)
    raise AssertionError("unreachable weighted median")


def build_full_fit_plan(train: pd.DataFrame, decision: T4Decision) -> FullFitPlan:
    verify_t4_acceptance(decision)
    return FullFitPlan(
        recent_years=(2024,), multi_years=(2021, 2022, 2023, 2024),
        decay=decision.decay, recent_weight=decision.recent_weight,
        epochs=resolve_final_epoch(decision.best_epoch_by_year),
        feature_spec=decision.feature_spec, model_spec=decision.model_spec,
    )
```

`run_full_fit` fits the recent and multi preprocessing states independently, trains only the components named in the sealed T4 decision, saves epoch-level atomic checkpoints, and never opens `test.csv`. CatBoost full fit is allowed only when one stable prefix was approved on all folds.

After fitting, mark copied feature states with `dataclasses.replace(state, inference_mode=True)`. `benchmark_inference` runs the full official test rows on T4 without fitting state and enforces 480 seconds, leaving two minutes of margin under the official 10-minute L4 limit.

- [ ] **Step 4: Run final-training tests**

```bash
pytest tests/test_temporal_portfolio_final.py -q
```

Expected: PASS with fixture backends only.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/final_training.py tests/test_temporal_portfolio_final.py
git commit -m "feat: guard temporal portfolio full fitting"
```

### Task 2: Frozen inference and row-independence audit

**Files:**
- Create: `experiments/temporal_portfolio/inference.py`
- Create: `experiments/temporal_portfolio/row_independence.py`
- Modify: `tests/test_temporal_portfolio_final.py`

- [ ] **Step 1: Add failing inference invariance tests**

```python
def test_frozen_prediction_is_invariant_to_order_batch_and_subset(frozen_predictor, tiny_test) -> None:
    expected = predict_by_row_id(frozen_predictor, tiny_test, batch_size=32)
    assert expected == predict_by_row_id(frozen_predictor, tiny_test.sample(frac=1, random_state=3), batch_size=1)
    assert expected == predict_by_row_id(frozen_predictor, tiny_test, batch_size=512)
    subset = tiny_test.iloc[[0, 2]]
    assert {key: expected[key] for key in subset.row_id.astype(str)} == predict_by_row_id(frozen_predictor, subset, batch_size=32)


def test_inference_rejects_target_and_unfitted_state(frozen_predictor, tiny_test) -> None:
    with pytest.raises(InferenceRuleError, match="target"):
        frozen_predictor.predict(tiny_test.assign(control_success=0))
    with pytest.raises(InferenceRuleError, match="frozen"):
        FrozenPredictor(unfitted_manifest()).predict(tiny_test)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_final.py -q
```

Expected: FAIL on missing inference modules.

- [ ] **Step 3: Implement frozen row-local inference**

```python
class FrozenPredictor:
    def __init__(self, manifest: FrozenManifest, models: Mapping[str, object], states: Mapping[str, PortfolioFeatureState]) -> None:
        verify_frozen_manifest(manifest, models, states)
        self.manifest, self.models, self.states = manifest, models, states

    def predict(self, rows: pd.DataFrame, *, batch_size: int = 512) -> np.ndarray:
        if "control_success" in rows:
            raise InferenceRuleError("evaluation rows contain target")
        if rows["row_id"].isna().any() or rows["row_id"].duplicated().any():
            raise InferenceRuleError("evaluation row identity differs")
        streams = {
            name: predict_batches(self.models[name], transform_portfolio_features(rows, self.states[name]), batch_size)
            for name in self.manifest.stream_order
        }
        probability = apply_frozen_recipe(streams, self.manifest.recipe)
        return validate_probability(probability, expected_rows=len(rows))
```

`audit_row_independence` compares single-row, shuffled, batch sizes 1/32/512, and subsets using `np.testing.assert_allclose(rtol=0, atol=1e-7)`. It writes hashes and maximum absolute differences, never test aggregate features.

- [ ] **Step 4: Run inference and legacy submission-runtime tests**

```bash
pytest tests/test_temporal_portfolio_final.py tests/test_submission_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/inference.py experiments/temporal_portfolio/row_independence.py tests/test_temporal_portfolio_final.py
git commit -m "feat: freeze row-independent temporal inference"
```

### Task 3: Review-only training delivery and fail-closed packaging policy

**Files:**
- Create: `experiments/temporal_portfolio/final_delivery.py`
- Modify: `tests/test_temporal_portfolio_final.py`

- [ ] **Step 1: Add failing delivery tests**

```python
def test_training_delivery_contains_frozen_evidence_without_submission(tmp_path: Path, accepted_full_fit) -> None:
    delivery = write_training_delivery(tmp_path, accepted_full_fit)
    verified = verify_training_delivery(delivery)
    assert verified.policy["submission_package"] is False
    assert "submit.zip" not in verified.members
    assert {"frozen/manifest.json", "policy/policy.json", "review/acceptance.json"}.issubset(verified.members)


def test_packaging_is_not_authorized_by_training_delivery(accepted_full_fit) -> None:
    with pytest.raises(SubmissionNotAuthorized, match="separate reviewed step"):
        assert_submission_packaging_authorized(accepted_full_fit)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_final.py -q
```

Expected: FAIL on missing delivery module.

- [ ] **Step 3: Implement recursive verified delivery**

```python
def write_training_delivery(root: Path, full_fit: FullFitResult) -> Path:
    verify_full_fit_result(full_fit)
    members = collect_frozen_members(full_fit)
    members["policy/policy.json"] = canonical_json({
        "campaign_id": "temporal_portfolio_v1",
        "submission_package": False,
        "evaluation_row_independent": True,
        "external_data": False,
    })
    members["review/acceptance.json"] = canonical_json(full_fit.acceptance)
    path = Path(root) / "temporal_portfolio_training_delivery.zip"
    atomic_deterministic_zip(path, artifact_kind="temporal_training_delivery_v1", members=members, bindings=full_fit.bindings)
    verify_training_delivery(path)
    return path


def assert_submission_packaging_authorized(_value: object) -> None:
    raise SubmissionNotAuthorized("submission packaging requires a separate reviewed step")
```

Verifier requirements include model/state hashes, approved T4 decision hash, T5A confirmation hash, row-independence report, dependency versions, fixture prediction hash, and measured inference resources. Review artifacts cannot contain restart optimizer state; resume artifacts cannot claim submission approval.

- [ ] **Step 4: Run delivery and existing packaging tests**

```bash
pytest tests/test_temporal_portfolio_final.py tests/test_submission_package.py -q
```

Expected: PASS and no `submit.zip` written.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/final_delivery.py tests/test_temporal_portfolio_final.py
git commit -m "feat: seal temporal training delivery"
```

### Task 4: Generated Kaggle T4x2 cell

**Files:**
- Create: `experiments/temporal_portfolio/KAGGLE_CELL.py`
- Create: `tools/build_temporal_portfolio_kaggle_cell.py`
- Create: `tests/test_temporal_portfolio_platform.py`

- [ ] **Step 1: Write failing Kaggle cell tests**

```python
def test_generated_kaggle_cell_is_under_commit_limit_and_offline(tmp_path: Path) -> None:
    cell = build_kaggle_cell(tmp_path / "KAGGLE_CELL.py")
    source = cell.read_text()
    assert cell.stat().st_size < 1_000_000
    assert "github.com" not in source and "requests.get" not in source and "git clone" not in source
    compile(source, str(cell), "exec")


def test_kaggle_discovery_accepts_extracted_handoff_and_requires_t4x2(kaggle_tree) -> None:
    inputs = discover_kaggle_inputs(kaggle_tree)
    assert inputs.data_root.name == "kaggle-lg-aimers-9th-data-upload"
    assert inputs.handoff.stage == "T2A"
    with pytest.raises(PlatformError, match="two Tesla T4"):
        require_kaggle_gpus(lambda: ["Tesla T4"])
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_platform.py -q
```

Expected: FAIL on missing platform files.

- [ ] **Step 3: Implement deterministic runtime embedding and launcher**

```python
RUNTIME_MEMBERS = (
    "experiments/temporal_portfolio", "experiments/independent_dl",
    "experiments/preprocessing_campaign/budgeted_scheduler.py",
)


def build_kaggle_cell(output: Path) -> Path:
    archive = deterministic_runtime_tar(RUNTIME_MEMBERS)
    rendered = TEMPLATE.replace("__RUNTIME_B64__", base64.b64encode(archive).decode("ascii"))
    if len(rendered.encode()) >= 1_000_000:
        raise PlatformError("generated Kaggle cell exceeds one megabyte")
    return atomic_text(Path(output), rendered)
```

The generated one-cell source must scan `/kaggle/input` recursively for official headers and artifact manifests, require exactly two Tesla T4 devices, derive the next stage only from the verified handoff, start the common runner, stream 30–60 second logs, and leave exactly one handoff under `/kaggle/working/temporal_portfolio_output/`.

- [ ] **Step 4: Render and test the cell**

```bash
python tools/build_temporal_portfolio_kaggle_cell.py --output experiments/temporal_portfolio/KAGGLE_CELL.py
pytest tests/test_temporal_portfolio_platform.py -q
```

Expected: PASS; generated SHA and byte size are pinned in the test.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/KAGGLE_CELL.py tools/build_temporal_portfolio_kaggle_cell.py tests/test_temporal_portfolio_platform.py
git commit -m "feat: add temporal Kaggle launcher"
```

### Task 5: Generated Colab T4 recovery cell with one terminal download

**Files:**
- Create: `experiments/temporal_portfolio/colab.py`
- Create: `experiments/temporal_portfolio/COLAB_RECOVERY_CELL.py`
- Create: `tools/build_temporal_portfolio_colab_cell.py`
- Modify: `tests/test_temporal_portfolio_platform.py`

- [ ] **Step 1: Add failing Colab supervision tests**

```python
def test_colab_reuses_upload_cache_and_downloads_one_terminal_handoff(upload_cache, fixture_run) -> None:
    events = []
    first = run_colab(upload=upload_cache.upload, download=events.append, campaign=fixture_run)
    second = run_colab(upload=upload_cache.upload, download=events.append, campaign=fixture_run)
    assert upload_cache.calls == 1
    assert len(events) == 2
    assert all(event.path.name.endswith("_handoff.zip") for event in events)


def test_colab_error_downloads_only_latest_emergency_handoff(failing_campaign, upload_cache) -> None:
    events = []
    with pytest.raises(RuntimeError, match="fixture"):
        run_colab(upload=upload_cache.upload, download=events.append, campaign=failing_campaign)
    assert [event.phase for event in events] == ["emergency_handoff"]
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_platform.py -q
```

Expected: FAIL on missing Colab implementation.

- [ ] **Step 3: Implement upload classification and single-download supervision**

```python
def run_colab(*, upload, download, campaign) -> StageRun:
    paths = cached_uploads(upload)
    verified_input, handoff = classify_uploads_by_manifest(paths)
    latest = {"path": handoff}
    monitor = LocalSnapshotMonitor(on_verified=lambda path: latest.__setitem__("path", path))
    monitor.start()
    try:
        result = campaign(
            verified_input=verified_input,
            prior_handoff=handoff,
            scheduler=SingleGpuScheduler(
                stop_new_jobs_seconds=900,
                archive_reserve_seconds=600,
                heartbeat_seconds=60,
            ),
        )
        download(DownloadEvent("terminal_handoff", result.handoff.path))
        return result
    except Exception:
        emergency = monitor.force_handoff() or latest["path"]
        if emergency is not None:
            download(DownloadEvent("emergency_handoff", emergency))
        raise
    finally:
        monitor.stop()
```

`SingleGpuScheduler` uses the same worker-result hash validation and completed-identity reuse as `BudgetedScheduler`, but starts at most one pending job on CUDA device 0. It stops safely at the Colab deadline and leaves remaining jobs pending instead of reclassifying them as rejected. Periodic snapshots stay on `/content`; they replace the prior stable snapshot and never call `files.download`. The generated cell requires one Tesla T4, supports one input ZIP plus one optional handoff, clears stale `experiments.temporal_portfolio` modules before loading an embedded runtime, and prints `HANDOFF_READY` or `PORTFOLIO_ERROR` followed by `EMERGENCY_HANDOFF_READY`.

- [ ] **Step 4: Render and run platform tests**

```bash
python tools/build_temporal_portfolio_colab_cell.py --output experiments/temporal_portfolio/COLAB_RECOVERY_CELL.py
pytest tests/test_temporal_portfolio_platform.py -q
```

Expected: PASS; no test opens a browser download.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/colab.py experiments/temporal_portfolio/COLAB_RECOVERY_CELL.py tools/build_temporal_portfolio_colab_cell.py tests/test_temporal_portfolio_platform.py
git commit -m "feat: add temporal Colab recovery launcher"
```

### Task 6: Human runbook and final static verification

**Files:**
- Create: `docs/TEMPORAL_PORTFOLIO_RUNBOOK.md`
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `tests/test_temporal_portfolio_platform.py`

- [ ] **Step 1: Write failing documentation contract test**

```python
def test_runbook_lists_inputs_runtime_outputs_and_return_text() -> None:
    text = Path("docs/TEMPORAL_PORTFOLIO_RUNBOOK.md").read_text()
    for required in (
        "lg-aimers-9th-data", "T4 x2", "예상 시간", "재실행",
        "HANDOFF_READY", "PORTFOLIO_ERROR", "전달할 파일", "제출 ZIP이 아닙니다",
    ):
        assert required in text
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_platform.py::test_runbook_lists_inputs_runtime_outputs_and_return_text -q
```

Expected: FAIL because the runbook is missing.

- [ ] **Step 3: Write the runbook and update status docs**

The runbook must give one complete path for every stage:

```text
목적 → 필요한 Kaggle Input → 실행 셀 → 예상 시간 → 정상 로그
→ Output에 생기는 handoff 하나 → 다음 Version 입력 방법 → 오류 시 전달할 두 로그
```

State explicitly that Codex runs only fixture/static checks and the user runs all full-data/GPU work. Record every stage version name, T4x2 requirement, Colab recovery rule, rerun safety, and the single file to return. README and ROADMAP link the runbook and label the campaign `code_ready` only after the complete focused suite passes.

- [ ] **Step 4: Run focused and regression suites**

```bash
pytest -q tests/test_temporal_portfolio_*.py
pytest -q tests/test_independent_dl_training.py tests/test_independent_dl_features.py tests/test_budgeted_preprocessing_scheduler.py tests/test_tabm_campaign_artifacts.py tests/test_submission_package.py
git diff --check
```

Expected: all tests pass and `git diff --check` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add docs/TEMPORAL_PORTFOLIO_RUNBOOK.md README.md docs/ROADMAP.md tests/test_temporal_portfolio_platform.py
git commit -m "docs: add temporal portfolio runbook"
```
