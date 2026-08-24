# Temporal Portfolio Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize cutoff-safe recent/multi-season folds and independently selectable S1, pitcher TrackMan, batter exposure, and matchup feature bundles.

**Architecture:** A portfolio feature state composes existing `dl_standard` preprocessing, the existing seasonal snapshot, and the existing pitcher TrackMan lookup. New code adds explicit bundle selection, batter matching, frozen lookup serialization, and row-invariance checks without changing legacy feature views.

**Tech Stack:** pandas, NumPy, SciPy Hungarian matching, existing `experiments.independent_dl` preprocessing and feature sources, pytest.

---

### Task 1: Temporal row selection and decay weights

**Files:**
- Create: `experiments/temporal_portfolio/folds.py`
- Create: `tests/test_temporal_portfolio_folds.py`

- [ ] **Step 1: Write failing fold tests**

```python
def test_recent_and_multi_rows_follow_the_contract() -> None:
    frame = pd.DataFrame({"season": [2019, 2020, 2021, 2022], "row_id": list("abcd")})
    fold = TemporalFold(2021, 2019, 2021, 2022)
    recent = select_training_rows(frame, fold, expert="recent", decay=None)
    multi = select_training_rows(frame, fold, expert="multi", decay=Decimal("0.55"))
    assert recent.frame["row_id"].tolist() == ["c"]
    assert recent.sample_weight.tolist() == [1.0]
    assert multi.frame["row_id"].tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(multi.sample_weight, [0.55**2, 0.55, 1.0])


def test_validation_and_future_rows_are_rejected() -> None:
    frame = pd.DataFrame({"season": [2021, 2022], "row_id": ["a", "b"]})
    with pytest.raises(FoldError, match="training rows"):
        select_training_rows(frame, TemporalFold(2021, 2019, 2021, 2022), expert="multi", decay=Decimal("0.55"), prefiltered=False, allow_extra=False)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_folds.py -q
```

Expected: FAIL because `folds.py` is missing.

- [ ] **Step 3: Implement exact season selection and weights**

```python
@dataclass(frozen=True)
class WeightedRows:
    frame: pd.DataFrame
    sample_weight: np.ndarray
    row_sha256: str


def select_training_rows(frame: pd.DataFrame, fold: TemporalFold, *, expert: str, decay: Decimal | None, prefiltered: bool = False, allow_extra: bool = False) -> WeightedRows:
    if expert == "recent":
        years, decay_value = (fold.recent_year,), None
    elif expert == "multi" and decay is not None:
        years, decay_value = tuple(range(fold.multi_start, fold.multi_end + 1)), float(decay)
    else:
        raise FoldError("expert and decay differ")
    if not allow_extra and frame["season"].gt(fold.multi_end).any():
        raise FoldError("training rows exceed fold cutoff")
    selected = frame.loc[frame["season"].isin(years)].copy()
    if selected.empty or selected["row_id"].duplicated().any():
        raise FoldError("training rows are empty or duplicated")
    weights = np.ones(len(selected), dtype="float32") if decay_value is None else np.power(decay_value, fold.multi_end - selected["season"].to_numpy()).astype("float32")
    return WeightedRows(selected, weights, row_id_sha256(selected["row_id"]))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_temporal_portfolio_folds.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/folds.py tests/test_temporal_portfolio_folds.py
git commit -m "feat: add temporal expert row selection"
```

### Task 2: Connect the existing S1 seasonal implementation

**Files:**
- Create: `experiments/temporal_portfolio/seasonal_features.py`
- Modify: `tests/test_independent_dl_feature_sources.py`
- Create: `tests/test_temporal_portfolio_features.py`

- [ ] **Step 1: Add failing S1 audit tests**

```python
def test_s1_uses_only_the_previous_season_snapshot(tiny_train: pd.DataFrame) -> None:
    state = fit_s1_state(tiny_train.loc[tiny_train.season.le(2023)], valid_year=2024)
    valid = tiny_train.loc[tiny_train.season.eq(2024)].drop(columns="control_success")
    first = transform_s1(valid, state)
    mutated = tiny_train.copy()
    mutated.loc[mutated.season.eq(2024), "control_success"] = 1 - mutated.loc[mutated.season.eq(2024), "control_success"]
    second = transform_s1(valid, fit_s1_state(mutated.loc[mutated.season.le(2023)], valid_year=2024))
    pd.testing.assert_frame_equal(first, second)
    assert {"season_pitcher_n", "season_batter_n", "season_vs_career_success"}.issubset(first)


def test_s1_transform_is_row_separable(tiny_train: pd.DataFrame) -> None:
    state = fit_s1_state(tiny_train.loc[tiny_train.season.le(2023)], valid_year=2024)
    rows = tiny_train.loc[tiny_train.season.eq(2024)].drop(columns="control_success")
    whole = transform_s1(rows, state)
    one = transform_s1(rows.iloc[[0]], state)
    pd.testing.assert_series_equal(whole.iloc[0], one.iloc[0], check_names=False)
```

- [ ] **Step 2: Verify the new tests fail**

```bash
pytest tests/test_temporal_portfolio_features.py -q
```

Expected: FAIL on missing wrapper module.

- [ ] **Step 3: Wrap rather than rewrite the existing seasonal functions**

```python
@dataclass(frozen=True)
class S1State:
    valid_year: int
    prior_rate: float
    snapshot: SeasonalSnapshot


def fit_s1_state(train: pd.DataFrame, *, valid_year: int) -> S1State:
    if train["season"].ge(valid_year).any():
        raise SeasonalFeatureError("S1 fit rows reach validation season")
    snapshot = build_seasonal_snapshot(train, cutoff_year=valid_year - 1)
    prior = float(pd.to_numeric(train["control_success"], errors="raise").mean())
    return S1State(valid_year, prior, snapshot)


def transform_s1(rows: pd.DataFrame, state: S1State) -> pd.DataFrame:
    if rows["season"].ne(state.valid_year).any():
        raise SeasonalFeatureError("S1 transform season differs")
    transformed, categorical = attach_seasonal_features(rows, state.snapshot, prior_rate=state.prior_rate)
    added = [column for column in transformed if column not in rows]
    result = transformed.loc[:, added].copy()
    result.attrs["categorical_columns"] = tuple(categorical)
    return result
```

Add a regression test to `test_independent_dl_feature_sources.py` that pins the existing snapshot convention (`asof_n + current labeled row`) so future refactors cannot shift the season boundary.

- [ ] **Step 4: Run S1 and legacy tests**

```bash
pytest tests/test_temporal_portfolio_features.py tests/test_independent_dl_feature_sources.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/seasonal_features.py tests/test_temporal_portfolio_features.py tests/test_independent_dl_feature_sources.py
git commit -m "feat: expose cutoff-safe S1 features"
```

### Task 3: Split the existing pitcher TrackMan lookup into ablation bundles

**Files:**
- Create: `experiments/temporal_portfolio/trackman_pitcher.py`
- Create: `tests/test_temporal_portfolio_trackman.py`

- [ ] **Step 1: Write failing pitcher bundle tests**

```python
def test_pitcher_trackman_bundles_are_disjoint_and_cutoff_bound(tiny_main, tiny_history) -> None:
    state = fit_pitcher_trackman(tiny_main, tiny_history, cutoff_year=2023)
    names = {bundle: set(frame.columns) for bundle, frame in state.bundles.items()}
    assert set(names) == {"P0", "P1", "P2", "P3"}
    assert all("pitcher_id" in columns for columns in names.values())
    assert not (names["P1"] - {"pitcher_id"}) & (names["P3"] - {"pitcher_id"})
    changed = tiny_history.copy()
    changed.loc[changed.season.gt(2023), "rel_speed"] = 9999
    replay = fit_pitcher_trackman(tiny_main, changed, cutoff_year=2023)
    for bundle in names:
        pd.testing.assert_frame_equal(state.bundles[bundle], replay.bundles[bundle])
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_trackman.py::test_pitcher_trackman_bundles_are_disjoint_and_cutoff_bound -q
```

Expected: FAIL on missing module.

- [ ] **Step 3: Implement deterministic column selectors over the proven lookup**

```python
P0 = ("tm_history_n", "tm_match_cost", "tm_match_margin", "tm_match_confidence", "tm_match_accepted")
P1_PREFIXES = ("tm_career_", "tm_recent_")
P2_PREFIXES = ("tm_fastball_", "tm_breaking_", "tm_offspeed_", "tm_other_", "tm_history_fastball_", "tm_history_breaking_", "tm_history_offspeed_", "tm_history_other_")
P2_EXACT = ("tm_fastball_breaking_speed_gap", "tm_fastball_offspeed_speed_gap")
P3_PREFIXES = ("tm_trend_",)


def fit_pitcher_trackman(train: pd.DataFrame, history: pd.DataFrame, *, cutoff_year: int) -> PitcherTrackmanState:
    built = build_trackman_lookup(train, history, cutoff_year)
    lookup = validate_trackman_build_result(built, expected_cutoff_year=cutoff_year)
    bundles = {
        "P0": select_columns(lookup, exact=P0),
        "P1": select_columns(lookup, prefixes=P1_PREFIXES, exclude_prefixes=P3_PREFIXES),
        "P2": select_columns(lookup, prefixes=P2_PREFIXES, exact=P2_EXACT),
        "P3": select_columns(lookup, prefixes=P3_PREFIXES),
    }
    validate_disjoint_non_key_columns(bundles, allowed_overlap={"tm_history_n"})
    return PitcherTrackmanState(cutoff_year, built, MappingProxyType(bundles))
```

- [ ] **Step 4: Run TrackMan tests**

```bash
pytest tests/test_temporal_portfolio_trackman.py tests/test_independent_dl_feature_sources.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/trackman_pitcher.py tests/test_temporal_portfolio_trackman.py
git commit -m "feat: split pitcher TrackMan bundles"
```

### Task 4: Batter ID mapping and exposure bundles

**Files:**
- Create: `experiments/temporal_portfolio/trackman_batter.py`
- Modify: `tests/test_temporal_portfolio_trackman.py`

- [ ] **Step 1: Add failing batter mapping tests**

```python
def test_batter_mapping_is_one_to_one_and_leaves_ambiguous_rows_unmatched(tiny_batter_main, tiny_batter_history) -> None:
    state = fit_batter_trackman(tiny_batter_main, tiny_batter_history, cutoff_year=2023)
    accepted = state.mapping.loc[state.mapping.tm_batter_match_accepted.eq(1)]
    assert accepted["batter_id"].is_unique
    assert accepted["batter_trackman_id"].is_unique
    assert state.coverage == pytest.approx(len(accepted) / tiny_batter_main.batter_id.nunique())
    assert state.mapping.loc[state.mapping.batter_id.eq("ambiguous"), "tm_batter_match_accepted"].item() == 0


def test_batter_exposure_and_matchup_use_only_frozen_lookup(tiny_batter_main, tiny_batter_history, tiny_rows) -> None:
    state = fit_batter_trackman(tiny_batter_main, tiny_batter_history, cutoff_year=2023)
    b1 = attach_batter_exposure(tiny_rows, state)
    m1 = build_matchup_features(tiny_rows, pitcher_state=tiny_pitcher_state(), batter_state=state)
    assert len(b1) == len(m1) == len(tiny_rows)
    assert any(column.startswith("tm_batter_seen_") for column in b1)
    assert any(column.startswith("tm_matchup_") for column in m1)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_trackman.py -q
```

Expected: FAIL on missing batter functions.

- [ ] **Step 3: Implement cutoff-bound mapping and exposure**

Use per-season maximum `asof_batter_n`, hand, team, and official history pitch counts. Construct a rectangular cost matrix with exact hand gating, normalized log-count error, and team-season overlap penalty; solve one-to-one assignment with `scipy.optimize.linear_sum_assignment`.

```python
def accept_mapping(cost: float, margin: float, hand_equal: bool) -> bool:
    return hand_equal and cost <= 1.5 and margin >= 0.15


def fit_batter_trackman(train: pd.DataFrame, history: pd.DataFrame, *, cutoff_year: int) -> BatterTrackmanState:
    main = build_main_batter_signatures(train.loc[train.season.le(cutoff_year)])
    tm = build_trackman_batter_signatures(history.loc[history.season.le(cutoff_year)])
    mapping = solve_batter_assignment(main, tm, accept=accept_mapping)
    exposure = aggregate_batter_exposure(history.loc[history.season.le(cutoff_year)], mapping)
    coverage = float(mapping["tm_batter_match_accepted"].mean()) if len(mapping) else 0.0
    return BatterTrackmanState(cutoff_year, mapping, exposure, coverage, lookup_sha256(exposure))
```

`aggregate_batter_exposure` must output pitch-group rates, mean/std speed, spin, vertical/horizontal break, history count, recent count, mapping confidence, and missing flags. `build_matchup_features` computes differences only between frozen pitcher arsenal columns and frozen batter exposure columns; it never reads another prediction row.

- [ ] **Step 4: Run TrackMan tests**

```bash
pytest tests/test_temporal_portfolio_trackman.py -q
```

Expected: PASS, including a future-history mutation test and coverage below 30% status test.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/trackman_batter.py tests/test_temporal_portfolio_trackman.py
git commit -m "feat: add batter TrackMan exposure mapping"
```

### Task 5: Portfolio feature state, preprocessing, and cache

**Files:**
- Create: `experiments/temporal_portfolio/features.py`
- Create: `experiments/temporal_portfolio/feature_cache.py`
- Modify: `tests/test_temporal_portfolio_features.py`

- [ ] **Step 1: Add failing composition and invariance tests**

```python
def test_feature_bundle_composition_preserves_rows_and_fits_dl_state(tiny_train, tiny_history) -> None:
    spec = PortfolioFeatureSpec(("base", "S1", "P2", "B1", "M1"), "dl_standard")
    state, batch = fit_portfolio_features(tiny_train.query("season <= 2023"), tiny_history, spec=spec, valid_year=2024)
    assert batch.row_id.tolist() == tiny_train.query("season <= 2023").row_id.astype(str).tolist()
    assert state.spec == spec
    assert state.history_cutoff_year == 2023
    assert np.isfinite(batch.x_num).all()


def test_transform_does_not_fit_on_or_aggregate_evaluation_rows(tiny_train, tiny_history, tiny_test) -> None:
    state, _ = fit_portfolio_features(tiny_train, tiny_history, spec=PortfolioFeatureSpec(("base", "S1"), "dl_standard"), valid_year=2025)
    whole = transform_portfolio_features(tiny_test, state)
    shuffled = transform_portfolio_features(tiny_test.sample(frac=1, random_state=7), state)
    assert prediction_inputs_by_row_id(whole) == prediction_inputs_by_row_id(shuffled)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_features.py -q
```

Expected: FAIL on missing composition functions.

- [ ] **Step 3: Implement fit/transform composition**

```python
VALID_BUNDLES = ("base", "S1", "P0", "P1", "P2", "P3", "B1", "M1")


@dataclass(frozen=True)
class PortfolioFeatureSpec:
    bundles: tuple[str, ...]
    profile: str


@dataclass(frozen=True)
class PortfolioFeatureState:
    spec: PortfolioFeatureSpec
    history_cutoff_year: int
    preprocessing_state: PreprocessingState
    category_maps: Mapping[str, Mapping[str, int]]
    fitted_sources: Mapping[str, object]
    schema: tuple[str, ...]
    inference_mode: bool = False


def fit_portfolio_features(train: pd.DataFrame, history: pd.DataFrame, *, spec: PortfolioFeatureSpec, valid_year: int) -> tuple[PortfolioFeatureState, FeatureBatch]:
    normalized = normalize_bundle_spec(spec, VALID_BUNDLES)
    raw, fitted_sources = attach_fit_only_bundles(train, history, normalized, valid_year=valid_year)
    preprocessing_state, prepared = fit_preprocessor(raw, PreprocessingSpec(normalized.profile, ("hand_matchup",)))
    category_maps, batch = encode_preprocessed_frame(train, prepared, fitted_maps=None)
    state = PortfolioFeatureState(normalized, valid_year - 1, preprocessing_state, category_maps, fitted_sources, feature_schema(prepared), False)
    return state, batch


def transform_portfolio_features(rows: pd.DataFrame, state: PortfolioFeatureState) -> FeatureBatch:
    if "control_success" in rows and state.inference_mode:
        raise PortfolioFeatureError("evaluation transform contains target")
    raw = attach_frozen_bundles(rows, state.fitted_sources)
    prepared = transform_preprocessor(raw, state.preprocessing_state)
    return encode_preprocessed_frame(rows, prepared, fitted_maps=state.category_maps)[1]
```

The cache identity must include train/validation row hashes, feature spec, cutoff, serialized preprocessing state, every lookup hash, and feature code hash. Write arrays and state to a sibling temporary directory, fsync, then publish with `os.replace`; load arrays read-only with `mmap_mode="r"`.

- [ ] **Step 4: Run feature and preprocessing regressions**

```bash
pytest tests/test_temporal_portfolio_features.py tests/test_independent_dl_features.py tests/test_preprocessing_profiles.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/features.py experiments/temporal_portfolio/feature_cache.py tests/test_temporal_portfolio_features.py
git commit -m "feat: materialize temporal feature bundles"
```
