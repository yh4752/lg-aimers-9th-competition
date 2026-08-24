# Temporal Portfolio Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact loss weighting, restartable temporal TabM jobs, CatBoost prefix evidence, and cross-fitted TrackMan LUPI students while preserving the existing trainer’s behavior.

**Architecture:** The existing independent-DL trainer remains the execution engine. One backward-compatible window-loss hook enables exact sample weighting; a new TabM adapter owns decay and teacher arrays; CatBoost and LUPI use separate focused modules and publish the same worker result contract.

**Tech Stack:** PyTorch, TabM, CatBoost, pandas, NumPy, scikit-learn group folds, pytest with fake backends.

---

### Task 1: Backward-compatible exact window-loss hook

**Files:**
- Modify: `experiments/independent_dl/training.py`
- Modify: `tests/test_independent_dl_training.py`

- [ ] **Step 1: Write failing loss-hook tests**

```python
def test_optional_window_loss_receives_complete_window_indices() -> None:
    adapter = _WindowAwareAdapter()
    fit_candidate(_request(rows=5, effective_batch=4, micro_batch=2), adapter, output_dir, backend=TorchTrainingBackend())
    assert adapter.windows == [(0, 1, 2, 3), (4,)]


def test_legacy_adapter_keeps_original_microbatch_scaling() -> None:
    adapter = _FakeAdapter()
    result = call_adapter_window_loss(
        adapter, model=None, x_num=None, x_cat=None, y=None,
        row_indices=np.array([0, 1]), window_indices=np.array([0, 1, 2, 3]),
        microbatch_count=2,
    )
    assert result == adapter.loss_value / 2
```

- [ ] **Step 2: Run the tests and verify failure**

```bash
pytest tests/test_independent_dl_training.py -q
```

Expected: FAIL because `call_adapter_window_loss` is undefined.

- [ ] **Step 3: Add the optional hook and use it in the training loop**

```python
def call_adapter_window_loss(adapter, model, x_num, x_cat, y, *, row_indices, window_indices, microbatch_count):
    method = getattr(adapter, "loss_for_window", None)
    if method is not None:
        return method(
            model, x_num, x_cat, y,
            row_indices=row_indices,
            window_indices=window_indices,
        )
    return adapter.loss(model, x_num, x_cat, y, row_indices=row_indices) / microbatch_count
```

In `TorchTrainingBackend.run_attempt`, create one GPU `window_indices` tensor before the microbatch loop and replace the current `call_adapter_loss(...)/n_micro` expression with `call_adapter_window_loss(...)`. Do not change checkpoint keys, validation, scheduler, RNG, or progress output.

- [ ] **Step 4: Run the full independent trainer tests**

```bash
pytest tests/test_independent_dl_training.py -q
```

Expected: PASS, including the new hook and every legacy adapter test.

- [ ] **Step 5: Commit**

```bash
git add experiments/independent_dl/training.py tests/test_independent_dl_training.py
git commit -m "feat: support exact window-normalized losses"
```

### Task 2: Weighted and teacher-aware TabM adapter

**Files:**
- Create: `experiments/temporal_portfolio/tabm_training.py`
- Create: `tests/test_temporal_portfolio_training.py`

- [ ] **Step 1: Write failing adapter tests**

```python
def test_decay_weighted_bce_uses_window_denominator(fake_tabm_model) -> None:
    adapter = TemporalTabMAdapter(sample_weight=np.array([1.0, 0.5, 0.25]), loss_name="bce")
    adapter.bind_device("cpu")
    loss = adapter.loss_for_window(
        fake_tabm_model, x_num(), x_cat(), torch.tensor([1.0, 0.0]),
        row_indices=torch.tensor([0, 1]), window_indices=torch.tensor([0, 1, 2]),
    )
    expected = weighted_member_bce(fake_tabm_model.logits[:2], [1.0, 0.0], [1.0, 0.5]) / 1.75
    torch.testing.assert_close(loss, expected)


def test_lupi_changes_only_matched_row_targets(fake_tabm_model) -> None:
    adapter = TemporalTabMAdapter(
        sample_weight=np.ones(3), loss_name="bce",
        teacher_probability=np.array([0.9, np.nan, 0.1]), teacher_lambda=0.25,
    )
    adapter.bind_device("cpu")
    per_row = adapter.debug_per_row_loss(fake_tabm_model.logits, torch.tensor([1.0, 0.0, 0.0]))
    assert per_row[1] == pytest.approx(hard_label_bce(fake_tabm_model.logits[1], 0.0))
    assert per_row[0] != pytest.approx(hard_label_bce(fake_tabm_model.logits[0], 1.0))
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_training.py -q
```

Expected: FAIL on missing adapter.

- [ ] **Step 3: Implement the adapter over `TabMAdapter`**

```python
class TemporalTabMAdapter(TabMAdapter):
    def __init__(self, *, sample_weight: np.ndarray, loss_name: str, teacher_probability: np.ndarray | None = None, teacher_lambda: float = 0.0) -> None:
        super().__init__(loss_name)
        validate_loss_arrays(sample_weight, teacher_probability, teacher_lambda)
        self.sample_weight_np = np.asarray(sample_weight, dtype="float32")
        self.teacher_np = None if teacher_probability is None else np.asarray(teacher_probability, dtype="float32")
        self.teacher_lambda = float(teacher_lambda)
        self._weight = self._teacher = self._teacher_mask = None

    def build(self, model_config, metadata, device):
        model = super().build(model_config, metadata, device)
        self.bind_device(device)
        return model

    def bind_device(self, device: str) -> None:
        torch = import_runtime_module("torch")
        self._weight = torch.as_tensor(self.sample_weight_np, device=device)
        if self.teacher_np is not None:
            mask = np.isfinite(self.teacher_np)
            self._teacher_mask = torch.as_tensor(mask, device=device)
            self._teacher = torch.as_tensor(np.nan_to_num(self.teacher_np, nan=0.5), device=device)

    def loss_for_window(self, model, x_num, x_cat, y, *, row_indices, window_indices):
        torch = import_runtime_module("torch")
        logits = model(x_num, x_cat).squeeze(-1)
        hard = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y.float().unsqueeze(1).expand_as(logits), reduction="none"
        ).mean(dim=1)
        per_row = hard
        if self._teacher is not None:
            soft_target = self._teacher[row_indices].unsqueeze(1).expand_as(logits)
            soft = torch.nn.functional.binary_cross_entropy_with_logits(logits, soft_target, reduction="none").mean(dim=1)
            lam = self.teacher_lambda * self._teacher_mask[row_indices].float()
            per_row = (1.0 - lam) * hard + lam * soft
        numerator = (per_row * self._weight[row_indices]).sum()
        denominator = self._weight[window_indices].sum().clamp_min(1e-12)
        return numerator / denominator
```

For direct Brier loss, compute member-mean probability squared error before multiplying weights. LUPI is valid only with BCE and must reject Brier+teacher configuration.

- [ ] **Step 4: Run adapter and legacy model tests**

```bash
pytest tests/test_temporal_portfolio_training.py tests/test_tabm_campaign_model.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/tabm_training.py tests/test_temporal_portfolio_training.py
git commit -m "feat: train weighted temporal TabM experts"
```

### Task 3: Strict current-pitch TrackMan matching for LUPI

**Files:**
- Create: `experiments/temporal_portfolio/lupi_matching.py`
- Create: `tests/test_temporal_portfolio_lupi.py`

- [ ] **Step 1: Write failing matching tests**

```python
def test_lupi_matcher_accepts_only_unique_monotonic_alignment(tiny_main_game, tiny_trackman_game, id_maps) -> None:
    matched = match_current_pitch_rows(tiny_main_game, tiny_trackman_game, id_maps=id_maps)
    assert matched["row_id"].tolist() == tiny_main_game["row_id"].tolist()
    assert matched["trackman_id"].is_unique
    assert matched["lupi_match_accepted"].eq(1).all()


def test_ambiguous_candidate_games_remain_unmatched(tiny_main_game, tiny_trackman_game, id_maps) -> None:
    duplicated = pd.concat([tiny_trackman_game, tiny_trackman_game.assign(trackman_game_id="other")])
    matched = match_current_pitch_rows(tiny_main_game, duplicated, id_maps=id_maps)
    assert matched["lupi_match_accepted"].eq(0).all()


def test_future_trackman_mutation_cannot_change_lupi_training_matches(tiny_main_game, tiny_trackman_game, id_maps) -> None:
    future = tiny_trackman_game.assign(season=2024, rel_speed=9999)
    first = fit_lupi_matches(tiny_main_game, pd.concat([tiny_trackman_game, future]), cutoff_year=2023, id_maps=id_maps)
    future["rel_speed"] = -9999
    second = fit_lupi_matches(tiny_main_game, pd.concat([tiny_trackman_game, future]), cutoff_year=2023, id_maps=id_maps)
    pd.testing.assert_frame_equal(first, second)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_lupi.py -q
```

Expected: FAIL on missing matching module.

- [ ] **Step 3: Implement conservative game candidate and sequence alignment**

Main pseudo-games are segmented only in labeled training data, using stable source order and a boundary when season/month/day-of-week/team matchup changes or inning decreases. Candidate TrackMan games must match season, month, day-of-week, mapped teams, and side. Align pitch tokens monotonically:

```python
MAIN_TOKEN = ("inning", "top_bottom", "balls_before", "strikes_before", "outs_before", "pitcher_id", "batter_id")
TM_TOKEN = ("inning", "top_bottom", "balls_before", "strikes_before", "outs_before", "pitcher_trackman_id", "batter_trackman_id")


def match_current_pitch_rows(main: pd.DataFrame, history: pd.DataFrame, *, id_maps: EntityMaps) -> pd.DataFrame:
    mapped_main = attach_trackman_ids(main, id_maps)
    rows = []
    for pseudo_game in split_training_pseudo_games(mapped_main):
        candidates = candidate_trackman_games(pseudo_game, history)
        scored = sorted((sequence_alignment(pseudo_game, game), str(game.trackman_game_id), game) for game in candidates)
        if len(scored) != 1 or scored[0][0].coverage < 0.85 or scored[0][0].mean_cost > 0.10:
            rows.extend(unmatched_rows(pseudo_game))
            continue
        rows.extend(accepted_alignment_rows(pseudo_game, scored[0][2], scored[0][0]))
    return validate_one_to_one_alignment(pd.DataFrame(rows))
```

If multiple candidate games have scores within `0.02`, reject the entire pseudo-game. No heuristic may force a match. Record coverage, cost, margin, and exact token agreement.

- [ ] **Step 4: Run LUPI matching tests**

```bash
pytest tests/test_temporal_portfolio_lupi.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/lupi_matching.py tests/test_temporal_portfolio_lupi.py
git commit -m "feat: match train pitches for TrackMan LUPI"
```

### Task 4: Cross-fitted teacher probabilities

**Files:**
- Create: `experiments/temporal_portfolio/lupi_teacher.py`
- Modify: `tests/test_temporal_portfolio_lupi.py`

- [ ] **Step 1: Add failing cross-fit tests**

```python
def test_teacher_probability_is_never_in_sample(matched_training_rows) -> None:
    result = crossfit_teacher(matched_training_rows, seed=3407, folds=5, backend=RecordingTeacherBackend())
    assert result.probability.notna().sum() == len(matched_training_rows)
    assert all(result.fold_by_row_id[row_id] == held_out for row_id, held_out in result.predicted_by_fold.items())
    assert result.probability.between(0, 1).all()


def test_teacher_returns_nan_for_unmatched_student_rows(matched_training_rows, all_row_ids) -> None:
    result = build_teacher_vector(all_row_ids, crossfit_teacher(matched_training_rows, seed=3407, folds=5, backend=FixtureTeacherBackend()))
    assert result.shape == (len(all_row_ids),)
    assert np.isnan(result[all_row_ids == "unmatched"]).all()
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_lupi.py -q
```

Expected: FAIL on missing teacher module.

- [ ] **Step 3: Implement group-hash cross-fitting**

```python
def teacher_fold(pitcher_id: object, seed: int, folds: int) -> int:
    digest = sha256(f"{seed}|{pitcher_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def crossfit_teacher(rows: pd.DataFrame, *, seed: int, folds: int, backend: TeacherBackend) -> TeacherOOF:
    assignments = rows["pitcher_id"].map(lambda value: teacher_fold(value, seed, folds))
    probability = pd.Series(np.nan, index=rows.index, dtype="float64")
    fold_by_row_id = {str(row_id): int(fold) for row_id, fold in zip(rows["row_id"], assignments)}
    predicted_by_fold = {}
    for fold in range(folds):
        train = rows.loc[assignments.ne(fold)]
        valid = rows.loc[assignments.eq(fold)]
        model = backend.fit(train.loc[:, TEACHER_FEATURES], train["control_success"], seed=seed + fold)
        probability.loc[valid.index] = backend.predict(model, valid.loc[:, TEACHER_FEATURES])
        predicted_by_fold.update({row_id: fold for row_id in valid["row_id"].astype(str)})
    return validate_teacher_oof(rows, probability, fold_by_row_id, predicted_by_fold)
```

The production backend is CatBoost with a fixed contract, no early-stopping on the student validation fold, and CPU/GPU choice recorded. Save match evidence and teacher OOF; do not save teacher bytes into the final student delivery.

- [ ] **Step 4: Run LUPI tests**

```bash
pytest tests/test_temporal_portfolio_lupi.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/lupi_teacher.py tests/test_temporal_portfolio_lupi.py
git commit -m "feat: crossfit TrackMan teacher labels"
```

### Task 5: CatBoost prefix evidence and common worker output

**Files:**
- Create: `experiments/temporal_portfolio/catboost_training.py`
- Create: `experiments/temporal_portfolio/worker.py`
- Modify: `tests/test_temporal_portfolio_training.py`

- [ ] **Step 1: Add failing CatBoost and worker tests**

```python
def test_catboost_job_saves_all_preregistered_prefix_predictions(fake_catboost_backend, temporal_job) -> None:
    result = run_catboost_job(temporal_job, backend=fake_catboost_backend)
    assert set(result.prefix_predictions) == {16, 64, 192, 384}
    assert all(values.shape == (temporal_job.valid_rows,) for values in result.prefix_predictions.values())


def test_worker_result_binds_every_published_file(tmp_path: Path, fixture_worker_job) -> None:
    result = run_worker(fixture_worker_job, tmp_path, backend=FixtureBackend())
    payload = json.loads((tmp_path / "worker_result.json").read_text())
    assert payload["training_identity_sha256"] == fixture_worker_job.identity.sha256
    assert {item["path"] for item in payload["artifacts"]} >= {"metrics.json", "predictions.csv", "checkpoint_meta.json"}
    assert all(file_sha256(tmp_path / item["path"]) == item["sha256"] for item in payload["artifacts"])
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_training.py -q
```

Expected: FAIL on missing CatBoost and worker modules.

- [ ] **Step 3: Implement prefix evaluation and worker publication**

```python
CATBOOST_PREFIXES = (16, 64, 192, 384)


def run_catboost_job(job: TemporalTrainingJob, *, backend: CatBoostBackend) -> CatBoostResult:
    model = backend.fit(job.train_frame, job.target, sample_weight=job.sample_weight, iterations=max(CATBOOST_PREFIXES), seed=job.seed)
    predictions = {
        trees: np.asarray(backend.predict(model, job.valid_frame, ntree_end=trees), dtype="float64")
        for trees in CATBOOST_PREFIXES
    }
    return validate_catboost_result(model, predictions, job.valid_row_id)


def publish_worker_result(root: Path, job: TemporalTrainingJob, artifacts: Sequence[Path], status: str) -> Path:
    payload = {
        "schema_version": 1,
        "job_id": job.job_id,
        "status": status,
        "training_identity_sha256": job.identity.sha256,
        "artifacts": [bound_member(path, root) for path in artifacts],
    }
    return atomic_json(root / "worker_result.json", payload)
```

TabM workers call `fit_candidate` with `TemporalTabMAdapter`; CatBoost workers call `run_catboost_job`; LUPI workers first verify teacher OOF hash and then call the weighted TabM path. A completed worker must publish row-aligned target, probability, pitcher ID, batter ID, game type, OOV, TrackMan availability, and segment columns.

- [ ] **Step 4: Run training and scheduler regressions**

```bash
pytest tests/test_temporal_portfolio_training.py tests/test_independent_dl_training.py tests/test_budgeted_preprocessing_scheduler.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/catboost_training.py experiments/temporal_portfolio/worker.py tests/test_temporal_portfolio_training.py
git commit -m "feat: publish temporal model worker evidence"
```
