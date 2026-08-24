# Temporal Portfolio Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn row-aligned OOF evidence into preregistered stage promotions, calibrated ensembles, score tiers, and a restartable T1–T5A campaign.

**Architecture:** Pure metrics and decisions are separated from orchestration. A stage planner emits immutable jobs from prior review evidence; the existing two-GPU scheduler runs workers; the runner never trains a job whose semantic identity already completed and never produces a submission package.

**Tech Stack:** pandas, NumPy, SciPy optimization, scikit-learn logistic regression, existing budgeted scheduler, pytest.

---

### Task 1: Aligned OOF, Brier, score tiers, and expert recipes

**Files:**
- Create: `experiments/temporal_portfolio/metrics.py`
- Create: `tests/test_temporal_portfolio_metrics.py`

- [ ] **Step 1: Write failing metric tests**

```python
def test_align_predictions_requires_identical_row_target_and_fold() -> None:
    aligned = align_oof([prediction_frame("a"), prediction_frame("b")])
    assert aligned.row_id.is_unique
    with pytest.raises(PortfolioMetricError, match="target"):
        align_oof([prediction_frame("a"), prediction_frame("b", flip_target=True)])


def test_probability_logit_anchor_and_score_tier() -> None:
    recent = np.array([0.2, 0.8])
    multi = np.array([0.4, 0.6])
    np.testing.assert_allclose(blend_probability(recent, multi, Decimal("0.50")), [0.3, 0.7])
    assert np.isfinite(blend_logit(recent, multi, Decimal("0.50"))).all()
    anchored = apply_anchor(recent, anchor_rate=0.55, beta=Decimal("0.05"))
    assert np.all((anchored > 0) & (anchored < 1))
    assert score_tier(Decimal("0.00046")) == "breakthrough"


def test_brier_improvement_maps_monotonically_to_skill_score() -> None:
    assert score_gain_from_brier_gain(Decimal("0.00025"), Decimal("0.25")) == Decimal("100")


def test_t1_recipe_grid_contains_every_preregistered_anchor_recipe() -> None:
    recipes = build_t1_recipes(load_contract())
    assert len(recipes) == 160
    assert len({recipe.recipe_id for recipe in recipes}) == 160
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_metrics.py -q
```

Expected: FAIL on missing metrics module.

- [ ] **Step 3: Implement pure aligned metrics**

```python
EPS = np.finfo("float64").eps


def brier(target: np.ndarray, probability: np.ndarray) -> float:
    y, p = validate_binary_probability(target, probability)
    return float(np.mean(np.square(p - y)))


def blend_logit(left: np.ndarray, right: np.ndarray, left_weight: Decimal) -> np.ndarray:
    w = float(left_weight)
    a = np.log(np.clip(left, EPS, 1 - EPS) / np.clip(1 - left, EPS, 1 - EPS))
    b = np.log(np.clip(right, EPS, 1 - EPS) / np.clip(1 - right, EPS, 1 - EPS))
    z = w * a + (1.0 - w) * b
    return 1.0 / (1.0 + np.exp(-z))


def apply_anchor(probability: np.ndarray, *, anchor_rate: float, beta: Decimal) -> np.ndarray:
    return blend_logit(probability, np.full(len(probability), anchor_rate), Decimal("1") - beta)


def score_tier(gain: Decimal) -> str:
    if gain >= Decimal("0.00045"): return "breakthrough"
    if gain >= Decimal("0.00025"): return "competitive"
    if gain >= Decimal("0.00005"): return "incremental"
    return "below_incremental"


def build_t1_recipes(contract: PortfolioContract) -> tuple[T1Recipe, ...]:
    return tuple(
        T1Recipe(decay, recent_weight, mode, beta)
        for decay in contract.decays
        for recent_weight in contract.recent_weights
        for mode in ("probability", "logit")
        for beta in contract.anchor_betas
    )
```

`align_oof` must inner-join nothing: it requires exactly equal sorted `(row_id, valid_year)` keys, equal targets and segment values, finite probabilities, and one prediction column per candidate.

- [ ] **Step 4: Run metric tests**

```bash
pytest tests/test_temporal_portfolio_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/metrics.py tests/test_temporal_portfolio_metrics.py
git commit -m "feat: evaluate temporal OOF recipes"
```

### Task 2: Block bootstrap and segment gates

**Files:**
- Create: `experiments/temporal_portfolio/uncertainty.py`
- Modify: `tests/test_temporal_portfolio_metrics.py`

- [ ] **Step 1: Add failing uncertainty tests**

```python
def test_pitcher_block_bootstrap_is_deterministic(oof_frame) -> None:
    first = pitcher_block_bootstrap(oof_frame, repeats=1000, seed=3407)
    second = pitcher_block_bootstrap(oof_frame, repeats=1000, seed=3407)
    assert first == second
    assert first.repeats == 1000


def test_small_segments_are_diagnostic_not_eligible(oof_frame) -> None:
    result = segment_regressions(oof_frame, minimum_rows=5000)
    small = next(item for item in result if item.rows < 5000)
    assert small.eligible is False
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_metrics.py -q
```

Expected: FAIL on missing uncertainty functions.

- [ ] **Step 3: Implement paired pitcher resampling and fixed segments**

```python
def pitcher_block_bootstrap(frame: pd.DataFrame, *, repeats: int, seed: int) -> BootstrapResult:
    work = frame.reset_index(drop=True)
    grouped = {key: group.index.to_numpy() for key, group in work.groupby("pitcher_id", sort=True)}
    keys = np.asarray(sorted(grouped, key=str), dtype=object)
    rng = np.random.default_rng(seed)
    gains = np.empty(repeats, dtype="float64")
    for position in range(repeats):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([grouped[key] for key in sampled])
        gains[position] = brier(work.target.to_numpy()[indices], work.baseline.to_numpy()[indices]) - brier(work.target.to_numpy()[indices], work.candidate.to_numpy()[indices])
    return BootstrapResult(repeats, float(np.quantile(gains, 0.025)), float(np.median(gains)), float(np.quantile(gains, 0.975)))
```

Segment definitions are fixed for game type, hand matchup, pitcher/batter OOV, TrackMan availability, history-count bucket, runner state, and leverage bucket. Compare paired candidate-minus-baseline Brier and mark only groups with at least 5,000 rows eligible.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_temporal_portfolio_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/uncertainty.py tests/test_temporal_portfolio_metrics.py
git commit -m "feat: add temporal uncertainty gates"
```

### Task 3: Forward calibration and bounded ensemble search

**Files:**
- Create: `experiments/temporal_portfolio/ensembles.py`
- Create: `tests/test_temporal_portfolio_ensembles.py`

- [ ] **Step 1: Write failing ensemble tests**

```python
def test_forward_calibration_never_fits_the_scored_fold(three_fold_oof) -> None:
    result = evaluate_forward_calibration(three_fold_oof, method="platt")
    assert result.fit_years_by_score_year == {2023: (2022,), 2024: (2022, 2023)}
    assert 2022 not in result.calibrated_probability_by_year


def test_recipe_grid_is_bounded_and_anchor_never_stacks_with_calibration(streams) -> None:
    recipes = build_raw_recipes(streams, primary="C0")
    assert len(recipes) <= 24
    corrected = build_correction_recipes(top_raw=recipes[:5], approved_anchor=anchor())
    assert len(corrected) <= 20
    assert all(not (item.anchor and item.calibration) for item in corrected)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_ensembles.py -q
```

Expected: FAIL on missing ensemble module.

- [ ] **Step 3: Implement the preregistered grids**

```python
PAIR_WEIGHTS = (Decimal("0.90"), Decimal("0.80"), Decimal("0.70"))


def build_raw_recipes(streams: Mapping[str, OOFStream], *, primary: str) -> tuple[Recipe, ...]:
    output = [Recipe.single(primary)]
    for secondary in sorted(set(streams) - {primary}):
        for weight in PAIR_WEIGHTS:
            output.append(Recipe.pair(primary, secondary, weight, "probability"))
            output.append(Recipe.pair(primary, secondary, weight, "logit"))
    pair_survivors = pair_gate_survivors(output, streams)
    if len(pair_survivors) >= 2:
        output.extend(Recipe.triple(primary, pair_survivors[0], pair_survivors[1], mode) for mode in ("probability", "logit"))
    if len(output) > 24:
        raise EnsembleError("raw recipe grid exceeded its contract")
    return tuple(output)
```

Fit intercept-only and L2 Platt models only on prior OOF years. Reject Platt when slope is outside `[0.8, 1.2]`, intercept outside `[-0.15, 0.15]`, either forward year regresses, or weighted gain is below `0.00003`. Final coefficients fit on 2022–2024 OOF only after the method is approved.

- [ ] **Step 4: Run ensemble tests**

```bash
pytest tests/test_temporal_portfolio_ensembles.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/ensembles.py tests/test_temporal_portfolio_ensembles.py
git commit -m "feat: search bounded temporal ensembles"
```

### Task 4: Candidate decisions for T1–T4

**Files:**
- Create: `experiments/temporal_portfolio/decisions.py`
- Create: `tests/test_temporal_portfolio_decisions.py`

- [ ] **Step 1: Write failing decision tests**

```python
def test_champion_exploratory_and_rejected_are_distinct(metric_fixture) -> None:
    assert decide_candidate(metric_fixture(champion=True)).status == "champion"
    assert decide_candidate(metric_fixture(exploratory=True)).status == "exploratory"
    assert decide_candidate(metric_fixture()).status == "rejected"


def test_t2a_keeps_two_best_and_one_structural_wildcard(t2a_candidates) -> None:
    promoted = select_t2a_survivors(t2a_candidates)
    assert len(promoted) <= 3
    assert any(item.family in {"B1", "M1"} for item in promoted)


def test_unstable_catboost_prefix_cannot_reach_delivery(catboost_fold_metrics) -> None:
    decision = decide_catboost_prefix(catboost_fold_metrics(best_prefixes=(16, 384, 16)))
    assert decision.status == "unstable_for_deployment"
    assert decision.selected_prefix is None
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_decisions.py -q
```

Expected: FAIL on missing decisions module.

- [ ] **Step 3: Implement stage-specific pure decisions**

```python
def decide_candidate(metrics: CandidateMetrics) -> CandidateDecision:
    champion = (
        metrics.weighted_gain >= Decimal("0.00005")
        and metrics.latest_gain >= Decimal("0.00003")
        and metrics.bootstrap_lower > 0
        and metrics.max_segment_regression <= Decimal("0.00050")
        and all(regression <= Decimal("0.00003") for regression in metrics.fold_regressions)
    )
    if champion:
        return CandidateDecision("champion", score_tier(metrics.temporal_gain), metrics.candidate_id)
    exploratory = (
        metrics.improved_fold_count >= 2
        and metrics.weighted_gain >= Decimal("0.00003")
        and metrics.worst_fold_regression <= Decimal("0.00015")
        and metrics.latest_regression <= Decimal("0.00005")
        and metrics.max_segment_regression <= Decimal("0.00100")
    )
    return CandidateDecision("exploratory" if exploratory else "rejected", score_tier(metrics.temporal_gain), metrics.candidate_id)
```

Implement separate selectors for T1, T2-A, T2-B, T3, and T4. A candidate with missing mapping evidence returns `insufficient_mapping`; a completed CatBoost with non-adjacent fold prefixes returns `unstable_for_deployment`; incomplete minimum evidence returns `budget_inconclusive`, not rejected.

- [ ] **Step 4: Run decision tests**

```bash
pytest tests/test_temporal_portfolio_decisions.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/decisions.py tests/test_temporal_portfolio_decisions.py
git commit -m "feat: decide temporal portfolio stages"
```

### Task 5: Stage planner and restartable two-GPU runner

**Files:**
- Create: `experiments/temporal_portfolio/planner.py`
- Create: `experiments/temporal_portfolio/runner.py`
- Create: `experiments/temporal_portfolio/run_campaign.py`
- Create: `tests/test_temporal_portfolio_runner.py`

- [ ] **Step 1: Write failing planner and runner tests**

```python
def test_stage_planner_uses_prior_review_and_never_requeues_completed_identity(t1_review, completed_catalog) -> None:
    plan = plan_stage("T2A", prior_review=t1_review, completed=completed_catalog)
    assert len(plan.jobs) <= 10
    assert not set(job.identity.sha256 for job in plan.jobs) & set(completed_catalog)


def test_runner_stops_new_jobs_and_reserves_handoff_time(tmp_path: Path, fake_scheduler) -> None:
    result = run_stage(stage="T1", verified=verified_data(), output_root=tmp_path, deadline=10_000, scheduler=fake_scheduler(clock=9_200))
    assert result.status == "budget_inconclusive"
    assert result.handoff.path.is_file()
    assert fake_scheduler.started_jobs == []


def test_runner_reuses_valid_job_and_rejects_tampered_job(tmp_path: Path, completed_job) -> None:
    assert verify_completed_job(completed_job).status == "completed"
    completed_job.metrics.write_text("tampered")
    with pytest.raises(PortfolioRunnerError, match="artifact"):
        verify_completed_job(completed_job)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_runner.py -q
```

Expected: FAIL on missing planner and runner.

- [ ] **Step 3: Implement deterministic planning and scheduler composition**

```python
def run_stage(*, stage: str, verified: VerifiedOfficialData, output_root: Path, deadline: float, scheduler=None) -> StageRun:
    contract = load_contract()
    restored = restore_latest_verified_handoff(output_root, verified, contract)
    plan = plan_stage(stage, prior_review=restored.review, completed=restored.completed_identities)
    write_stage_plan(output_root, plan)
    runtime = scheduler or BudgetedScheduler(
        campaign_identity=plan.scheduler_identity,
        stop_new_jobs_seconds=900,
        archive_reserve_seconds=600,
        heartbeat_seconds=60,
        required_gpu_name="Tesla T4",
    )
    summary = runtime.run(plan.jobs, output_root / "campaign", deadline=deadline)
    review = evaluate_stage(stage, output_root / "campaign", prior_review=restored.review)
    state = advance_state(restored.state, stage=stage, summary=summary, review=review)
    handoff = write_handoff(output_root / "handoff", stage_evidence(state, review, summary))
    return StageRun(state.status, handoff, review.decision)
```

`plan_stage` accepts physical stages `T1`, `T2A`, `T2B`, `T3A`, `T3BT4`, `T5A`, and `T5B`. `T3BT4` first finishes T3 confirmations and then performs the GPU-free T4 decision; it must stop before its sealed 20,700-second limit. T5A emits only missing seed/fold confirmation jobs. The CLI accepts explicit `--stage`, `--data-root`, `--input-root`, `--output-root`, and `--deadline-unix`; it prints the exact standard logs from the design.

- [ ] **Step 4: Run runner and scheduler regressions**

```bash
pytest tests/test_temporal_portfolio_runner.py tests/test_budgeted_preprocessing_scheduler.py -q
python -m experiments.temporal_portfolio.run_campaign --help >/dev/null
```

Expected: PASS and exit code 0.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/planner.py experiments/temporal_portfolio/runner.py experiments/temporal_portfolio/run_campaign.py tests/test_temporal_portfolio_runner.py
git commit -m "feat: orchestrate temporal portfolio stages"
```
