# EDA-Informed Preprocessing Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restartable five-fold preprocessing campaign that compares 19 DL preprocessing settings across eight medium/large anchors, promotes stable candidates through seed and combination waves, and independently evaluates 17 settings on four provenance-pinned CatBoost structures.

**Architecture:** Keep the existing independent-DL campaign unchanged. A new strict campaign contract reads and hash-pins its p3/p4 anchors, a fold-fitted preprocessing engine prepares DL or CatBoost frames, and a separate state machine schedules atomic model-anchor/preprocessing/fold/seed jobs. Deterministic promotion code generates seed confirmation, pairwise/beam combinations, feature-view confirmation, and CatBoost jobs only from hashed completed evidence.

**Tech Stack:** Python 3.11+, NumPy 1.26.4, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.8.0, PyTorch and the existing DL adapters, CatBoost 1.2.10, pytest 8.4.1.

---

## File map

- `experiments/independent_dl/preprocessing.py`: immutable preprocessing specs/states, train-only fit, transform, selective Yeo-Johnson, grouped missingness, ID frequency, smoothing, and CatBoost-native frame output.
- `experiments/independent_dl/features.py`: preserve the legacy feature path and add preprocessing-aware DL batches plus `raw_plus_trackman`.
- `experiments/independent_dl/preprocessing_contracts.py`: strict campaign JSON loader, source-config hash verification, and deterministic Wave-A job expansion.
- `experiments/independent_dl/configs/preprocessing_ablation_v1.json`: five folds, eight exact p3/p4 anchors, 19 DL settings, seeds, promotion gates, combination search, and CatBoost structures.
- `experiments/independent_dl/preprocessing_evaluation.py`: paired fold/seed/OOV metrics, promotion booleans, pairwise generation, beam expansion, and adoption verdicts.
- `experiments/independent_dl/preprocessing_campaign.py`: restartable DL job runtime, manifest, artifacts, resource accounting, and Wave B-D registration.
- `experiments/catboost_preprocessing/{__init__,features,campaign}.py`: provenance-pinned CatBoost structures and Wave-E runtime.
- `experiments/preprocessing_campaign/run_campaign.py`: unified `run`, `promote`, `status`, and `summarize` CLI.
- `experiments/preprocessing_campaign/requirements-colab.txt`: existing DL requirements plus `catboost==1.2.10`.
- `experiments/preprocessing_campaign/COLAB.md`: one complete user-run Colab cell; no notebook file is created or changed.
- `tests/test_preprocessing_*.py`: small synthetic and fake-runtime tests only.

## Task 1: Seal the expanded campaign contract

**Files:**
- Create: `experiments/independent_dl/preprocessing_contracts.py`
- Create: `experiments/independent_dl/configs/preprocessing_ablation_v1.json`
- Create: `tests/test_preprocessing_contracts.py`

- [ ] **Step 1: Write failing expansion and source-binding tests**

```python
from pathlib import Path

import pytest

from experiments.independent_dl.preprocessing_contracts import (
    PreprocessingContractError,
    load_preprocessing_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/independent_dl/configs/preprocessing_ablation_v1.json"


def test_wave_a_expands_eight_anchors_nineteen_settings_and_five_folds():
    campaign = load_preprocessing_campaign(CONFIG)
    assert campaign.campaign_id == "preprocessing_campaign_v1"
    assert campaign.folds == (
        (2019, 2020), (2020, 2021), (2021, 2022),
        (2022, 2023), (2023, 2024),
    )
    assert len(campaign.anchors) == 8
    assert len(campaign.dl_settings) == 19
    assert len(campaign.wave_a_jobs) == 760
    assert {job.seed for job in campaign.wave_a_jobs} == {42}
    assert {job.profile_id for job in campaign.wave_a_jobs} == {"p3", "p4"}
    assert len({job.job_id for job in campaign.wave_a_jobs}) == 760


def test_contract_pins_the_existing_campaign_bytes(tmp_path):
    config = CONFIG.read_text(encoding="utf-8").replace(
        "bd8116773b3f77255315262277461cd27764b9fc37d73a5e7c1f72b2a7f98b53",
        "0" * 64,
    )
    changed = tmp_path / "preprocessing.json"
    changed.write_text(config, encoding="utf-8")
    with pytest.raises(PreprocessingContractError, match="source campaign SHA-256"):
        load_preprocessing_campaign(changed)


def test_settings_keep_full_smoothing_ranges_and_three_seeds():
    campaign = load_preprocessing_campaign(CONFIG)
    assert campaign.seeds == (42, 2026, 3407)
    assert campaign.pitcher_smoothing_k == (25, 50, 100, 250)
    assert campaign.batter_smoothing_k == (10, 25, 100, 250, 500, 1000, 2500)
    assert len(campaign.catboost_settings) == 17
    assert tuple(campaign.catboost_structures) == (
        "champion", "depth5", "depth8", "lr003"
    )
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python3 -m pytest tests/test_preprocessing_contracts.py -q`

Expected: collection fails with `ModuleNotFoundError` for `preprocessing_contracts`.

- [ ] **Step 3: Add the strict JSON and immutable job contract**

Implement these public types and loader. JSON loading must reject duplicate keys, `NaN`, and `Infinity`; exact keys are validated before expansion.

```python
@dataclass(frozen=True)
class PreprocessingSetting:
    setting_id: str
    profile: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessingJob:
    job_id: str
    wave: str
    anchor_id: str
    family: str
    profile_id: str
    model: Mapping[str, object]
    training: Mapping[str, object]
    feature_view: str
    setting: PreprocessingSetting
    train_end_year: int
    valid_year: int
    seed: int


@dataclass(frozen=True)
class PreprocessingCampaignSpec:
    campaign_id: str
    folds: tuple[tuple[int, int], ...]
    seeds: tuple[int, ...]
    anchors: Mapping[str, CandidateSpec]
    dl_settings: tuple[PreprocessingSetting, ...]
    pitcher_smoothing_k: tuple[int, ...]
    batter_smoothing_k: tuple[int, ...]
    catboost_structures: Mapping[str, Mapping[str, object]]
    catboost_settings: tuple[PreprocessingSetting, ...]
    promotion: Mapping[str, object]
    wave_a_jobs: tuple[PreprocessingJob, ...]
```

The JSON must reference `campaign_v1.json` and its exact SHA-256
`bd8116773b3f77255315262277461cd27764b9fc37d73a5e7c1f72b2a7f98b53`.
Anchor IDs are the exact eight `raw_typed` p3/p4 candidate IDs. The loader resolves them through `load_campaign`, never copies model parameters into the new JSON, and expands anchors → settings → folds with seed 42.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `python3 -m pytest tests/test_preprocessing_contracts.py tests/test_independent_dl_contract.py -q`

Expected: all tests pass and the old 64-candidate contract is unchanged.

- [ ] **Step 5: Commit Task 1**

```bash
git add experiments/independent_dl/preprocessing_contracts.py experiments/independent_dl/configs/preprocessing_ablation_v1.json tests/test_preprocessing_contracts.py
git commit -m "feat: define preprocessing campaign contract"
```

## Task 2: Implement fold-fitted preprocessing profiles

**Files:**
- Create: `experiments/independent_dl/preprocessing.py`
- Create: `tests/conftest.py`
- Create: `tests/test_preprocessing_profiles.py`

- [ ] **Step 1: Add one shared synthetic official-schema fixture**

Add `preprocessing_frame`, `preprocessing_train`, `preprocessing_valid`, and
`preprocessing_history` fixtures to `tests/conftest.py`. Use five 2023 rows for the base frame,
change the valid fixture to two 2024 rows with different row IDs, and include every source column
used by the three missing groups, smoothing, count state, hand matchup, win expectancy,
Yeo-Johnson, and duplicate-count check. The exact common values are:

```python
@pytest.fixture
def preprocessing_frame() -> pd.DataFrame:
    n = 5
    return pd.DataFrame({
        "row_id": [f"r-{i}" for i in range(n)],
        "season": [2023] * n,
        "game_type": ["R", "F", "R", "F", "R"],
        "top_bottom": ["T", "B", "T", "B", "T"],
        "base_state": ["000", "100", "010", "001", "110"],
        "pitcher_id": [11, 12, 11, 13, 14],
        "batter_id": [21, 22, 23, 21, 24],
        "pitcher_hand": [1, 2, 1, 2, 1],
        "batter_hand": [2, 1, 2, 1, 1],
        "pitcher_team_id": [1, 1, 2, 2, 3],
        "batter_team_id": [2, 2, 1, 1, 4],
        "balls_before": [0, 1, 2, 3, 1],
        "strikes_before": [0, 1, 2, 1, 2],
        "home_win_expectancy": [.5, .6, .4, .7, .3],
        "away_win_expectancy": [.5, .4, .6, .3, .7],
        "li": [.8, 1.0, 1.5, 2.0, 3.0],
        "run_top_before": [0, 1, 2, 3, 4],
        "run_bot_before": [0, 0, 1, 1, 2],
        "run_total_before": [0, 1, 3, 4, 6],
        "score_diff_home": [0, -1, 1, -2, 2],
        "score_diff_pitcher_team": [0, 1, -1, 2, -2],
        "asof_pitcher_n": [100, 50, 25, 10, 5],
        "asof_pitcher_pitchmix_n": [100, 50, 25, 10, 5],
        "asof_pitcher_success_rate": [.55, .45, .60, .40, .50],
        "asof_batter_n": [80, 40, 20, 10, 5],
        "asof_batter_success_rate": [.52, .48, .58, .42, .50],
        "asof_batter_middle_rate": [.14, .13, .15, .12, .16],
        **{f"asof_pitcher_prev{k}_game_{kind}_rate": [.5, .4, .6, .3, .5]
           for k in (1, 3, 5) for kind in ("success", "middle")},
        **{f"asof_pitcher_{kind}_rate": [.2, .3, .4, .5, .6]
           for kind in ("ball", "breaking", "fastball", "middle", "offspeed", "reverse", "strike")},
        "control_success": [1, 0, 1, 0, 1],
    })


@pytest.fixture
def preprocessing_train(preprocessing_frame: pd.DataFrame) -> pd.DataFrame:
    return preprocessing_frame.copy()


@pytest.fixture
def preprocessing_valid(preprocessing_frame: pd.DataFrame) -> pd.DataFrame:
    return preprocessing_frame.iloc[:2].assign(
        row_id=["v-0", "v-1"], season=2024, pitcher_id=[11, 999], batter_id=[999, 22]
    ).reset_index(drop=True)


@pytest.fixture
def preprocessing_history() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "season", "pitcher_trackman_id", "pitch_type_group", "pitcher_hand",
        "pitcher_team", "rel_speed", "spin_rate", "induced_vert_break",
        "horz_break", "extension", "rel_height", "rel_side", "zone_speed",
    ])
```

- [ ] **Step 2: Write failing tests for state isolation, missingness, smoothing, and native NaNs**

```python
import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.preprocessing import (
    PreprocessingError,
    PreprocessingSpec,
    fit_preprocessor,
    transform_preprocessor,
)


def test_dl_standard_uses_train_median_and_never_refits_on_validation(preprocessing_frame):
    train = preprocessing_frame.iloc[:3].copy()
    valid = preprocessing_frame.iloc[3:].copy()
    train.loc[1, "li"] = np.nan
    valid["li"] = [999.0, np.nan]
    state, fitted = fit_preprocessor(train, PreprocessingSpec("dl_standard", ()))
    transformed = transform_preprocessor(valid, state)
    assert state.numeric_median["li"] == pytest.approx(train["li"].median())
    expected = (state.numeric_median["li"] - state.numeric_mean["li"]) / state.numeric_std["li"]
    assert transformed.loc[valid["li"].isna(), "li"].iloc[0] == pytest.approx(expected)


def test_grouped_missing_indicators_are_three_group_flags(preprocessing_frame):
    train = preprocessing_frame.copy()
    train.loc[0, "asof_pitcher_prev3_game_success_rate"] = np.nan
    train.loc[1, "asof_pitcher_ball_rate"] = np.nan
    train.loc[2, "asof_batter_middle_rate"] = np.nan
    _, result = fit_preprocessor(
        train,
        PreprocessingSpec("dl_standard", ("grouped_missing_indicators",)),
    )
    assert result[[
        "pitcher_recent_missing", "pitcher_career_missing", "batter_career_missing"
    ]].to_numpy().tolist()[:3] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@pytest.mark.parametrize(
    ("component", "rate", "count", "k"),
    [
        ("pitcher_smooth_k100", "asof_pitcher_success_rate", "asof_pitcher_n", 100.0),
        ("batter_smooth_k250", "asof_batter_success_rate", "asof_batter_n", 250.0),
    ],
)
def test_smoothing_uses_fold_target_prior(preprocessing_frame, component, rate, count, k):
    state, result = fit_preprocessor(
        preprocessing_frame,
        PreprocessingSpec("dl_standard", (component,)),
    )
    prior = preprocessing_frame["control_success"].mean()
    expected = (preprocessing_frame[count].iloc[0] * preprocessing_frame[rate].iloc[0] + k * prior) / (
        preprocessing_frame[count].iloc[0] + k
    )
    assert state.target_prior == pytest.approx(prior)
    output = "pitcher_success_smooth_100" if "pitcher" in component else "batter_success_smooth_250"
    assert result[output].iloc[0] == pytest.approx(expected)


def test_tree_native_preserves_numeric_nan(preprocessing_frame):
    frame = preprocessing_frame.copy()
    frame.loc[0, "li"] = np.nan
    _, result = fit_preprocessor(frame, PreprocessingSpec("tree_native", ()))
    assert np.isnan(result.loc[0, "li"])


def test_duplicate_pitchmix_count_must_match_in_every_transformed_frame(preprocessing_frame):
    state, _ = fit_preprocessor(preprocessing_frame, PreprocessingSpec("dl_standard", ()))
    changed = preprocessing_frame.copy()
    changed.loc[0, "asof_pitcher_pitchmix_n"] += 1
    with pytest.raises(PreprocessingError, match="pitchmix_n"):
        transform_preprocessor(changed, state)
```

- [ ] **Step 3: Run and confirm RED**

Run: `python3 -m pytest tests/test_preprocessing_profiles.py -q`

Expected: collection fails because `preprocessing.py` does not exist.

- [ ] **Step 4: Implement the immutable spec and state**

```python
@dataclass(frozen=True)
class PreprocessingSpec:
    profile: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessingState:
    spec: PreprocessingSpec
    source_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    numeric_median: Mapping[str, float]
    numeric_mean: Mapping[str, float]
    numeric_std: Mapping[str, float]
    yeo_johnson_lambda: Mapping[str, float]
    entity_frequency: Mapping[str, Mapping[str, int]]
    target_prior: float
```

Validate profiles against `tree_native`, `dl_standard`, and `dl_selective_transform`. Validate every component, reject two pitcher K values or two batter K values in one spec, and canonicalize components in a fixed order. Drop `row_id`, target, and the verified duplicate count only after exact equality checks on both fit and transform frames.

Implement Yeo-Johnson using `scipy.stats.yeojohnson(values)` at fit and the returned lambda with `scipy.stats.yeojohnson(values, lmbda)` at transform. Only the eight design columns are eligible. DL order is median imputation → optional Yeo-Johnson → train mean/std scaling. Tree-native output keeps numeric NaNs.

- [ ] **Step 5: Implement all independent components**

Use explicit source-column tuples for three missing groups. ID frequencies are fitted only from train. Counts use `clip(lower=0)` before `log1p`. Smoothing uses:

```python
valid = np.isfinite(rate) & np.isfinite(count) & (count > 0)
smoothed = np.full(len(frame), state.target_prior, dtype="float64")
smoothed[valid] = (
    count[valid] * rate[valid] + k * state.target_prior
) / (count[valid] + k)
```

`hand_matchup` and `count_state` are categorical strings. `pitcher_team_win_expectancy` is numeric and uses home expectancy for top innings and away expectancy for bottom innings. Preserve row index and row count exactly.

- [ ] **Step 6: Run profile tests and confirm GREEN**

Run: `python3 -m pytest tests/test_preprocessing_profiles.py -q`

Expected: all profile tests pass without official data.

- [ ] **Step 7: Commit Task 2**

```bash
git add experiments/independent_dl/preprocessing.py tests/conftest.py tests/test_preprocessing_profiles.py
git commit -m "feat: add fold fitted preprocessing profiles"
```

## Task 3: Integrate preprocessing-aware DL feature caches

**Files:**
- Modify: `experiments/independent_dl/features.py`
- Create: `tests/test_preprocessing_feature_cache.py`

- [ ] **Step 1: Write failing compatibility and cache-identity tests**

```python
from experiments.independent_dl.features import (
    materialize_preprocessed_fold_cache,
    materialize_fold_cache,
)
from experiments.independent_dl.preprocessing import PreprocessingSpec


def test_legacy_feature_cache_still_uses_the_existing_api(preprocessing_train, preprocessing_valid, preprocessing_history, tmp_path):
    cache = materialize_fold_cache(
        tmp_path / "legacy", preprocessing_train, preprocessing_valid, preprocessing_history,
        "raw_typed", 2023, 2024,
    )
    assert cache.train.x_num.shape[0] == len(preprocessing_train)


def test_preprocessing_identity_changes_cache_namespace(preprocessing_train, preprocessing_valid, preprocessing_history, tmp_path):
    base = materialize_preprocessed_fold_cache(
        tmp_path, preprocessing_train, preprocessing_valid, preprocessing_history, "raw_typed",
        PreprocessingSpec("dl_standard", ()), 2023, 2024,
    )
    smooth = materialize_preprocessed_fold_cache(
        tmp_path, preprocessing_train, preprocessing_valid, preprocessing_history, "raw_typed",
        PreprocessingSpec("dl_standard", ("pitcher_smooth_k100",)), 2023, 2024,
    )
    assert base.root != smooth.root
    assert base.state.preprocessing.spec.components == ()
    assert smooth.state.preprocessing.spec.components == ("pitcher_smooth_k100",)


def test_raw_plus_trackman_adds_lookup_without_engineered_context(preprocessing_train, preprocessing_valid, preprocessing_history, tmp_path):
    cache = materialize_preprocessed_fold_cache(
        tmp_path, preprocessing_train, preprocessing_valid, preprocessing_history, "raw_plus_trackman",
        PreprocessingSpec("dl_standard", ()), 2023, 2024,
    )
    assert any(name.startswith("tm_") for name in cache.state.numeric_columns)
    assert not any(name.startswith("ctx_") for name in cache.state.categorical_columns)
```

- [ ] **Step 2: Run and confirm RED**

Run: `python3 -m pytest tests/test_preprocessing_feature_cache.py -q`

Expected: import fails for `materialize_preprocessed_fold_cache`.

- [ ] **Step 3: Add a separate preprocessing-aware path without changing legacy behavior**

Add `PreprocessedFeatureState` containing base view metadata, `PreprocessingState`, category maps, and Trackman hash. `materialize_preprocessed_fold_cache(...)` must:

1. build `raw_typed` or `raw_plus_trackman` from train rows only;
2. fit `PreprocessingState` on the train base frame plus the train target;
3. transform valid with frozen state;
4. fit category maps on transformed train only, with OOV index 0;
5. emit the existing `FeatureBatch` type.

Do not route existing `fit_feature_view` or `materialize_fold_cache` through the new engine. This preserves the current campaign semantics and tests.

- [ ] **Step 4: Bind cache reuse to the complete preprocessing contract**

The namespace is `{fold}/{view}/{preprocessing_id}`. `identity.json` includes source frame hashes, row hashes, cutoff, view, serialized preprocessing state hash, Trackman lookup hash, and SHA-256 of both `features.py` and `preprocessing.py`. Atomic directory publication and memory-mapped loads reuse the existing implementation.

- [ ] **Step 5: Run focused and legacy tests**

Run: `python3 -m pytest tests/test_preprocessing_feature_cache.py tests/test_independent_dl_features.py tests/test_independent_dl_training.py -q`

Expected: all pass; no existing notebook changes.

- [ ] **Step 6: Commit Task 3**

```bash
git add experiments/independent_dl/features.py tests/test_preprocessing_feature_cache.py
git commit -m "feat: cache preprocessing aware dl features"
```

## Task 4: Implement paired metrics and deterministic promotion

**Files:**
- Create: `experiments/independent_dl/preprocessing_evaluation.py`
- Create: `tests/test_preprocessing_evaluation.py`

- [ ] **Step 1: Write failing tests for pairing, adoption gates, pairwise combinations, and beam search**

```python
import itertools

import pandas as pd
import pytest

from experiments.independent_dl.preprocessing_evaluation import (
    evaluate_paired_setting,
    generate_pairwise_settings,
    next_beam,
)


def _metric_rows(delta=-0.0002):
    rows = []
    for seed, fold in itertools.product((42, 2026, 3407), range(2020, 2025)):
        for setting_id, brier in (("baseline", 0.2500), ("candidate", 0.2500 + delta)):
            rows.append({
                "anchor_id": "tabm-p3", "preprocessing_id": setting_id,
                "seed": seed, "fold": f"valid_{fold}", "valid_rows": 100,
                "brier": brier, "pitcher_oov_brier": brier,
                "batter_oov_brier": brier,
            })
    return pd.DataFrame(rows)


def test_paired_evaluation_requires_same_anchor_fold_and_seed():
    changed = _metric_rows().drop(index=0)
    with pytest.raises(ValueError, match="paired baseline"):
        evaluate_paired_setting(changed, candidate_id="candidate", baseline_id="baseline")


def test_global_adoption_requires_all_fixed_gates():
    result = evaluate_paired_setting(
        _metric_rows(), candidate_id="candidate", baseline_id="baseline"
    )
    assert result["gates"] == {
        "weighted_mean_improved": True,
        "four_of_five_folds_improved": True,
        "two_of_three_seeds_improved": True,
        "worst_fold_delta_lte_0_0001": True,
        "pitcher_oov_delta_lte_0_0002": True,
        "batter_oov_delta_lte_0_0002": True,
    }
    assert result["adopt_global"] is True


def test_pairwise_generation_keeps_all_nine_components():
    settings = generate_pairwise_settings(
        pitcher_component="pitcher_smooth_k100",
        batter_component="batter_smooth_k250",
    )
    assert len(settings) == 36
    assert len({setting.components for setting in settings}) == 36


def test_beam_keeps_eight_and_adds_one_component():
    components = (
        "pitcher_smooth_k100", "batter_smooth_k250", "dl_selective_transform",
        "asof_count_log1p", "entity_frequency_log1p",
        "grouped_missing_indicators", "hand_matchup", "count_state",
        "pitcher_team_win_expectancy",
    )
    ranked_depth_two_rows = pd.DataFrame([
        {
            "preprocessing_id": f"pair-{index}",
            "components": pair,
            "weighted_delta": -0.001 + index * 0.00001,
            "improved_folds": 5,
            "worst_fold_delta": -0.0001,
            "max_oov_delta": 0.0,
        }
        for index, pair in enumerate(itertools.islice(itertools.combinations(components, 2), 10))
    ])
    expanded = next_beam(ranked_depth_two_rows, beam_width=8, max_components=5)
    assert len({item.parent_id for item in expanded}) <= 8
    assert {len(item.components) for item in expanded} == {3}
```

- [ ] **Step 2: Run and confirm RED**

Run: `python3 -m pytest tests/test_preprocessing_evaluation.py -q`

Expected: missing evaluation module.

- [ ] **Step 3: Implement strict paired aggregation**

`build_metric_rows` requires `row_id`, anchor, preprocessing ID, fold, season, seed, target,
probability, game type, pitcher OOV, and batter OOV and converts aligned predictions into the
aggregate schema used in the tests. Duplicate/missing rows or target mismatches are errors.
`evaluate_paired_setting` then pairs every candidate aggregate against the same anchor/fold/seed
baseline and writes row-weighted mean, seed means/std, improvement counts, worst-fold delta, and
OOV deltas.

Wave-B promotion booleans exactly implement the design: mean improvement plus 3/5 folds, top two per anchor, best per family, or 3/5 improvement in a named segment. Tie-break with preprocessing ID.

- [ ] **Step 4: Implement fixed combination expansion**

`generate_pairwise_settings` uses exactly nine components after selecting one pitcher and one batter K per anchor. `next_beam` sorts by `(weighted_delta, -improved_folds, worst_fold_delta, max_oov_delta, preprocessing_id)`, retains eight parents, adds one absent component, canonicalizes/deduplicates, and stops after five components or when all retained children are worse than the preceding depth.

- [ ] **Step 5: Run evaluation tests and confirm GREEN**

Run: `python3 -m pytest tests/test_preprocessing_evaluation.py tests/test_independent_dl_evaluation.py -q`

Expected: all pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add experiments/independent_dl/preprocessing_evaluation.py tests/test_preprocessing_evaluation.py
git commit -m "feat: evaluate preprocessing promotions"
```

## Task 5: Build the restartable DL preprocessing state machine

**Files:**
- Create: `experiments/independent_dl/preprocessing_campaign.py`
- Create: `tests/test_preprocessing_campaign.py`

- [ ] **Step 1: Write failing fake-runtime tests**

```python
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from experiments.independent_dl.preprocessing_campaign import (
    CampaignInterrupted,
    PreprocessingJobResult,
    estimate_remaining_resources,
    run_preprocessing_campaign,
)


def _tiny_campaign():
    jobs = tuple(
        SimpleNamespace(job_id=f"job-{index}", family="tabm", wave="a")
        for index in (1, 2)
    )
    return SimpleNamespace(campaign_id="tiny-preprocessing", wave_a_jobs=jobs)


class FakeRuntime:
    def __init__(self, *, interrupt_on=None, fail_on=None):
        self.interrupt_on = interrupt_on
        self.fail_on = fail_on
        self.started = []

    def run_job(self, job, output_dir: Path):
        self.started.append(job.job_id)
        if job.job_id == self.interrupt_on:
            raise CampaignInterrupted("interruption")
        if job.job_id == self.fail_on:
            raise ValueError("candidate-local failure")
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = output_dir / "predictions.csv"
        metrics = output_dir / "metrics.json"
        predictions.write_text("row_id,probability\na,0.5\n", encoding="utf-8")
        metrics.write_text(json.dumps({"brier": 0.25}), encoding="utf-8")
        return PreprocessingJobResult(metrics, predictions, 0.25, 120.0, 1.0, 2.0)


def test_campaign_resumes_at_job_granularity_without_repeating_completed(tmp_path):
    tiny_campaign = _tiny_campaign()
    first = FakeRuntime(interrupt_on="job-2")
    with pytest.raises(CampaignInterrupted, match="interruption"):
        run_preprocessing_campaign(tiny_campaign, tmp_path, first)
    second = FakeRuntime()
    summary = run_preprocessing_campaign(tiny_campaign, tmp_path, second)
    assert "job-1" not in second.started
    assert summary.completed == ("job-1", "job-2")


def test_one_failed_job_does_not_block_independent_jobs(tmp_path):
    tiny_campaign = _tiny_campaign()
    runtime = FakeRuntime(fail_on="job-1")
    summary = run_preprocessing_campaign(tiny_campaign, tmp_path, runtime)
    assert summary.failed == ("job-1",)
    assert "job-2" in summary.completed


def test_completed_artifact_hash_change_forces_only_that_job_to_rerun(tmp_path):
    tiny_campaign = _tiny_campaign()
    run_preprocessing_campaign(tiny_campaign, tmp_path, FakeRuntime())
    (tmp_path / "jobs/job-1/predictions.csv").write_text("changed", encoding="utf-8")
    runtime = FakeRuntime()
    run_preprocessing_campaign(tiny_campaign, tmp_path, runtime)
    assert runtime.started == ["job-1"]


def test_resource_summary_uses_completed_same_family_median():
    resource_rows = [
        {"family": "tabm", "elapsed_seconds": 60.0},
        {"family": "tabm", "elapsed_seconds": 120.0},
        {"family": "tabm", "elapsed_seconds": 180.0},
    ]
    estimate = estimate_remaining_resources(resource_rows, pending_same_family=10)
    assert estimate["estimated_remaining_seconds"] == pytest.approx(10 * 120.0)
```

- [ ] **Step 2: Run and confirm RED**

Run: `python3 -m pytest tests/test_preprocessing_campaign.py -q`

Expected: missing preprocessing campaign module.

- [ ] **Step 3: Implement manifest and DL runtime**

Use a manifest entry per job with immutable serialized job, config SHA, state, attempts, timestamps, checkpoint, prediction/metric hashes, failure reason, elapsed seconds, peak RAM, and peak GPU memory. `OfficialPreprocessingDLRuntime` uses `materialize_preprocessed_fold_cache`, the existing adapter factory, and `fit_candidate`. Predictions contain:

```python
@dataclass(frozen=True)
class PreprocessingJobResult:
    metrics_path: Path
    predictions_path: Path
    best_brier: float
    elapsed_seconds: float
    peak_ram_gb: float
    peak_gpu_gb: float
```

```text
row_id,fold,season,game_type,target,probability,anchor_id,
preprocessing_id,seed,pitcher_oov,batter_oov
```

Job-local exceptions preserve the full traceback, mark only that job failed, and continue. Keyboard interrupt and Colab termination leave the current job resumable from its training checkpoint.

- [ ] **Step 4: Implement atomic aggregate artifacts and promotion registration**

After each completed job, atomically refresh `fold_metrics.csv`, `segment_metrics.csv`, `paired_deltas.csv`, and `resource_usage.csv`. A `promote` entry point reads only hash-valid completed jobs, writes the fixed gate booleans and source hashes to `promotion_decisions.json`, and registers Wave-B/C/D jobs without mutating existing entries. Re-running promotion is idempotent.

- [ ] **Step 5: Run focused campaign tests**

Run: `python3 -m pytest tests/test_preprocessing_campaign.py tests/test_independent_dl_campaign.py tests/test_independent_dl_training.py -q`

Expected: all pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add experiments/independent_dl/preprocessing_campaign.py tests/test_preprocessing_campaign.py
git commit -m "feat: orchestrate preprocessing dl campaign"
```

## Task 6: Add provenance-pinned CatBoost preprocessing jobs

**Files:**
- Create: `experiments/catboost_preprocessing/__init__.py`
- Create: `experiments/catboost_preprocessing/features.py`
- Create: `experiments/catboost_preprocessing/campaign.py`
- Create: `tests/test_catboost_preprocessing.py`

- [ ] **Step 1: Write failing feature and parameter tests**

```python
from experiments.catboost_preprocessing.campaign import CATBOOST_STRUCTURES
from experiments.catboost_preprocessing.features import fit_catboost_features


def test_catboost_structures_match_pinned_source():
    assert CATBOOST_STRUCTURES == {
        "champion": {"depth": 7, "iterations": 400},
        "depth5": {"depth": 5, "iterations": 600},
        "depth8": {"depth": 8, "iterations": 300},
        "lr003": {"learning_rate": 0.03, "iterations": 700},
    }


def test_catboost_native_keeps_nan_and_categorical_ids(preprocessing_frame):
    train = preprocessing_frame.copy()
    train.loc[0, "li"] = np.nan
    state, features = fit_catboost_features(train, components=())
    assert np.isnan(features.loc[0, "li"])
    assert features["pitcher_id"].dtype == object
    assert "control_success" not in features
```

- [ ] **Step 2: Run and confirm RED**

Run: `python3 -m pytest tests/test_catboost_preprocessing.py -q`

Expected: missing CatBoost preprocessing package.

- [ ] **Step 3: Port only the verified contract and four structures**

Source commit is `9454d68b93971627e3d3f613ce30be690cb5dce2`. Port categorical normalization and the common base parameters from:

```text
competition/src/aimers/features/catboost_smooth.py
competition/experiments/catboost_smooth_v1/recipe.py
competition/experiments/catboost_multifold_v2/selection.py
```

Record the source commit in module docstrings. Use `fit_preprocessor(..., tree_native)` for all optional features; do not retain the source's fixed 0.5 smoothing or K=150 because this campaign uses fold target prior and the sealed K grid. Lazy-import `catboost` only in execution.

- [ ] **Step 4: Implement Wave-E runner**

Expand four structures × 17 settings × five folds × three seeds = 1,020 jobs. Fit `CatBoostRegressor` with RMSE, the pinned base parameters, selected structure overrides, exact categorical columns, and job seed. Save model, predictions, metrics, resource usage, and hashes through the same manifest protocol. CatBoost job failures do not change DL state.

- [ ] **Step 5: Run tests without fitting CatBoost**

Run: `python3 -m pytest tests/test_catboost_preprocessing.py tests/test_preprocessing_profiles.py -q`

Expected: all tests pass with a fake model factory; no official data or GPU is used.

- [ ] **Step 6: Commit Task 6**

```bash
git add experiments/catboost_preprocessing tests/test_catboost_preprocessing.py
git commit -m "feat: add catboost preprocessing campaign"
```

## Task 7: Add CLI, Colab handoff, and output-contract checks

**Files:**
- Create: `experiments/preprocessing_campaign/__init__.py`
- Create: `experiments/preprocessing_campaign/run_campaign.py`
- Create: `experiments/preprocessing_campaign/requirements-colab.txt`
- Create: `experiments/preprocessing_campaign/COLAB.md`
- Create: `tests/test_preprocessing_handoff.py`

- [ ] **Step 1: Write failing CLI and handoff tests**

```python
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
requirements_path = ROOT / "experiments/preprocessing_campaign/requirements-colab.txt"


def test_cli_exposes_run_promote_status_and_summarize():
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.preprocessing_campaign.run_campaign", "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert all(name in completed.stdout for name in ("run", "promote", "status", "summarize"))


def test_colab_handoff_is_one_complete_python_block():
    text = (ROOT / "experiments/preprocessing_campaign/COLAB.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert len(blocks) == 1
    compile(blocks[0], "<preprocessing-colab>", "exec")
    assert "PREPROCESSING_CAMPAIGN_CHECKPOINTED" in blocks[0]
    assert "GITHUB_TOKEN" in blocks[0]
    assert "submission" not in blocks[0].lower()


def test_requirements_pin_catboost_and_keep_existing_dl_runtime():
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    assert "catboost==1.2.10" in lines
    assert "tabm==0.0.3" in lines
    assert "scikit-learn==1.8.0" in lines
```

- [ ] **Step 2: Run and confirm RED**

Run: `python3 -m pytest tests/test_preprocessing_handoff.py -q`

Expected: CLI/package/COLAB files are missing.

- [ ] **Step 3: Implement the unified CLI**

`run --wave {a,b,c,d,e}` loads the sealed contract and resumes only registered jobs in that wave. `promote --from-wave` validates hashes and registers the next wave. `status` prints completed/failed/pending by wave plus current throughput and ETA. `summarize` requires all registered jobs terminal and writes `campaign_summary.json`; it never creates predictions, models, submissions, or ZIP files.

- [ ] **Step 4: Write one complete Colab cell**

The cell must mount Drive, validate `train.csv` and `trackman_history.csv`, clone/fetch the repository through Colab secret `GITHUB_TOKEN`, detach at a temporary all-zero `REQUIRED_CODE_COMMIT` sentinel that is replaced after final implementation commit, hash-install requirements into `/content/preprocessing_runtime_v1`, check CUDA, and invoke Wave A. Re-running the entire cell uses the same Drive output and resumes. Exact terminal messages:

```text
PREPROCESSING_CAMPAIGN_CHECKPOINTED
PREPROCESSING_CAMPAIGN_ERROR type=<ExceptionClass> message=<message>
```

The documentation states: purpose, inputs, outputs, initial runtime unknown until five baseline folds, rerun safety, and exactly which traceback/manifest/summary the user should return.

- [ ] **Step 5: Run handoff tests and confirm GREEN**

Run: `python3 -m pytest tests/test_preprocessing_handoff.py -q`

Expected: all pass without package installation or network access.

- [ ] **Step 6: Commit Task 7**

```bash
git add experiments/preprocessing_campaign tests/test_preprocessing_handoff.py
git commit -m "feat: hand off preprocessing campaign"
```

## Task 8: Verify the whole implementation and pin the final Colab commit

**Files:**
- Modify: `experiments/preprocessing_campaign/COLAB.md`
- Modify only if contract documentation needs exact file links: `docs/superpowers/specs/2026-08-12-eda-informed-preprocessing-design.md`

- [ ] **Step 1: Run formatting-independent static checks**

Run:

```bash
python3 -m compileall -q experiments
git diff --check
```

Expected: exit code 0 with no output from `git diff --check`.

- [ ] **Step 2: Run the complete small test suite**

Run: `python3 -m pytest -q`

Expected: all repository tests pass. No official CSV, GPU training, package installation, notebook, or submission artifact is used.

- [ ] **Step 3: Inspect generated contract counts through the CLI**

Run:

```bash
python3 -m experiments.preprocessing_campaign.run_campaign status \
  --config experiments/independent_dl/configs/preprocessing_ablation_v1.json \
  --dry-contract
```

Expected JSON fields:

```json
{
  "wave_a_jobs": 760,
  "catboost_wave_e_jobs": 1020,
  "dl_settings": 19,
  "catboost_settings": 17,
  "folds": 5,
  "seeds": 3
}
```

- [ ] **Step 4: Commit verified implementation**

```bash
git add experiments tests docs/superpowers/specs/2026-08-12-eda-informed-preprocessing-design.md
git commit -m "feat: complete preprocessing performance campaign"
```

- [ ] **Step 5: Pin the handoff to the verified commit**

Resolve `git rev-parse HEAD`, replace only `REQUIRED_CODE_COMMIT` in `COLAB.md`, run
`python3 -m pytest tests/test_preprocessing_handoff.py -q`, then commit:

```bash
git add experiments/preprocessing_campaign/COLAB.md
git commit -m "docs: pin preprocessing colab handoff"
```

- [ ] **Step 6: Final verification after the pin commit**

Run:

```bash
python3 -m compileall -q experiments
python3 -m pytest -q
git diff --check
git status --short
```

Expected: compile and tests pass, `git diff --check` is empty, and the worktree is clean. Do not push, regenerate either notebook, run official data, install Colab packages, train a model, or create any submission package.
