# Privileged TrackMan Profile Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restartable Kaggle T4x2 campaign that compares E2 against CatBoost residual students trained with cross-fitted current-pitch TrackMan soft labels and cutoff-safe hierarchical profiles, and emits a delivery only for a gate-passing candidate.

**Architecture:** Keep the accepted E2 handoff immutable and create a new `experiments/tree_privileged` package around it. Reuse the audited pitcher-held-out teacher implementation, add an R/F-aware training-only pitch matcher and time-safe profile encoder, then screen five trained candidates plus one R/F blend before confirming at most two candidates across seeds. A two-worker runner owns state, atomic checkpoints, artifacts, and a generated single Kaggle cell; no submission ZIP is in scope.

**Tech Stack:** Python 3.12/3.13, pandas, NumPy, CatBoost 1.2.10, SciPy 1.16.3, PyTorch only for T4 discovery, pytest, deterministic ZIP/JSON/SHA-256 artifacts.

---

## File map

Create one focused package rather than extending the already large E2 and S4 modules.

- `experiments/tree_privileged/contract.json`: sealed candidates, folds, seeds, gates, runtime and source hashes.
- `experiments/tree_privileged/contracts.py`: strict contract parser and immutable dataclasses.
- `experiments/tree_privileged/inputs.py`: accepted E2 handoff compaction, verification and official-data binding.
- `experiments/tree_privileged/matching.py`: R/F-aware current-pitch TrackMan matching for labeled training rows only.
- `experiments/tree_privileged/teacher.py`: adapter from verified matches to the existing pitcher-held-out teacher OOF.
- `experiments/tree_privileged/profiles.py`: rolling hierarchical target profiles and frozen inference state.
- `experiments/tree_privileged/features.py`: composition of E2 tree features, profiles and soft targets.
- `experiments/tree_privileged/training.py`: one fold/seed CatBoost residual job.
- `experiments/tree_privileged/decisions.py`: structure screen, seed confirmation, R/F blend and acceptance gates.
- `experiments/tree_privileged/full_fit.py`: accepted full-data students and inference state.
- `experiments/tree_privileged/inference.py`: row-independent E2/new-model blending.
- `experiments/tree_privileged/artifacts.py`: resume, review, delivery and handoff bundles.
- `experiments/tree_privileged/runner.py`: two-GPU scheduling, resume, deadline and full-fit boundary.
- `experiments/tree_privileged/kaggle.py`: Kaggle input discovery and single-cell generation.
- `experiments/tree_privileged/requirements-kaggle.txt`: exact runtime packages.
- `experiments/tree_privileged/KAGGLE_CELL.py`: generated copyable cell.
- `tools/prepare_tree_privileged_input.py`: local E2 handoff compactor.
- `tools/build_tree_privileged_kaggle_cell.py`: deterministic cell builder.
- `docs/TREE_PRIVILEGED_KAGGLE.md`: user runbook and return contract.
- `tests/test_tree_privileged_*.py`: focused contract, matching, teacher, profile, training, decision, artifact, runner and Kaggle tests.

Do not modify the existing E2 submission builder. The new campaign may import its verifier and inference classes, but packaging remains blocked until a later user request and a verified accepted delivery.

### Task 1: Seal the campaign contract

**Files:**
- Create: `experiments/tree_privileged/__init__.py`
- Create: `experiments/tree_privileged/contract.json`
- Create: `experiments/tree_privileged/contracts.py`
- Create: `tests/test_tree_privileged_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from pathlib import Path

import pytest

from experiments.tree_privileged.contracts import (
    PrivilegedContractError,
    load_contract,
)


def test_contract_seals_budget_candidates_and_submission_boundary() -> None:
    contract = load_contract()
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.screen_seed == 3407
    assert contract.confirm_seeds == (42, 2026)
    assert contract.candidates == ("P", "D15", "D35", "PD15", "PD35")
    assert contract.teacher_lambdas == (0.15, 0.35)
    assert contract.wall_seconds == 37_800
    assert contract.submission_package is False
    assert contract.maximum_confirmed_candidates == 2


def test_contract_rejects_any_local_change(tmp_path: Path) -> None:
    source = Path("experiments/tree_privileged/contract.json")
    changed = tmp_path / "contract.json"
    changed.write_text(source.read_text().replace('"0.00005"', '"0.00004"', 1))
    with pytest.raises(PrivilegedContractError, match="gates differ"):
        load_contract(changed)
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_contracts.py
```

Expected: FAIL because `experiments.tree_privileged` does not exist.

- [ ] **Step 3: Add the exact JSON contract**

```json
{
  "schema_version": 1,
  "campaign_id": "tree_privileged_profile_v1",
  "review_only": true,
  "submission_package": false,
  "inputs": {
    "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
    "e2_handoff_sha256": "4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f"
  },
  "folds": [[2021, 2022], [2022, 2023], [2023, 2024]],
  "candidates": ["P", "D15", "D35", "PD15", "PD35"],
  "screen_seed": 3407,
  "confirm_seeds": [42, 2026],
  "teacher": {
    "folds": 5,
    "lambdas": ["0.15", "0.35"],
    "minimum_total_coverage": "0.30",
    "minimum_latest_coverage": "0.20"
  },
  "profiles": {
    "identity_strengths": [25, 75, 200],
    "interaction_strengths": [50, 150, 400],
    "matchup_strengths": [100, 300, 800],
    "minimum_rows": {"identity": 20, "interaction": 50, "matchup": 100}
  },
  "rf_alphas": ["0.35", "0.60", "0.80", "1.00"],
  "catboost": {
    "iterations": 1200,
    "depth": 8,
    "learning_rate": "0.04",
    "l2_leaf_reg": "5.0",
    "random_strength": "0.5",
    "bagging_temperature": "0.5",
    "border_count": 128,
    "max_ctr_complexity": 2,
    "od_wait": 60
  },
  "gates": {
    "weighted_gain": "0.00005",
    "latest_min_gain": "-0.00002",
    "maximum_segment_regression": "0.00050",
    "minimum_improved_folds": 2,
    "minimum_non_worse_seeds": 2,
    "maximum_screen_correlation": "0.998",
    "ensemble_incremental_gain": "0.00002",
    "probability_tolerance": "0.000001"
  },
  "runtime": {
    "wall_seconds": 37800,
    "new_job_guard_seconds": 7200,
    "artifact_reserve_seconds": 4500,
    "snapshot_interval_seconds": 600,
    "maximum_confirmed_candidates": 2,
    "gpu_count": 2
  }
}
```

- [ ] **Step 4: Implement strict immutable parsing**

In `contracts.py`, define frozen dataclasses `TeacherContract`, `ProfileContract`, and
`PrivilegedContract`. Parse decimals through `Decimal(str(value))`, require the exact top-level and nested key sets, compare every sealed grid to the JSON above, and expose `contract_sha256()` over the original bytes. Follow `experiments/tree_expert/s4_contracts.py`; do not add a permissive default or environment override.

- [ ] **Step 5: Run the tests**

Run the command from Step 2. Expected: `2 passed`.

- [ ] **Step 6: Commit the contract**

```bash
git add experiments/tree_privileged/__init__.py experiments/tree_privileged/contract.json experiments/tree_privileged/contracts.py tests/test_tree_privileged_contracts.py
git commit -m "feat: seal privileged tree campaign"
```

### Task 2: Bind the accepted E2 handoff as the only campaign input

**Files:**
- Create: `experiments/tree_privileged/inputs.py`
- Create: `tools/prepare_tree_privileged_input.py`
- Create: `tests/test_tree_privileged_inputs.py`

- [ ] **Step 1: Write failing input round-trip and tamper tests**

```python
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from experiments.tree_privileged.inputs import (
    PrivilegedInputError,
    prepare_input,
    verify_and_extract_input,
)


def test_prepare_and_verify_input_keeps_only_accepted_e2(valid_e2_handoff, tmp_path: Path) -> None:
    archive = prepare_input(valid_e2_handoff, tmp_path / "tree_privileged_input.zip")
    verified = verify_and_extract_input(archive, tmp_path / "verified")
    assert verified.e2_handoff_sha256 == "4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f"
    assert verified.e2_delivery.is_file()
    assert verified.e2_oof_root.is_dir()


def test_input_rejects_an_extra_or_modified_member(valid_e2_handoff, tmp_path: Path) -> None:
    source = prepare_input(valid_e2_handoff, tmp_path / "source.zip")
    changed = tmp_path / "changed.zip"
    with ZipFile(source) as reader, ZipFile(changed, "w", compression=ZIP_DEFLATED) as writer:
        for info in reader.infolist():
            writer.writestr(info, reader.read(info.filename))
        writer.writestr("extra.bin", b"not authorized")
    with pytest.raises(PrivilegedInputError, match="member set differs"):
        verify_and_extract_input(changed, tmp_path / "changed")
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_inputs.py
```

Expected: FAIL on the missing `inputs` module.

- [ ] **Step 3: Implement deterministic compaction and verification**

`prepare_input()` must first call `experiments.tree_expert.e2_artifacts.verify_e2_handoff()` and
require `status == "accepted"` and `delivery is True`. Copy only:

```text
manifest.json
e2/model_delivery.zip
e2/oof/2022.csv
e2/oof/2023.csv
e2/oof/2024.csv
```

The new manifest stores artifact kind `tree_privileged_input_v1`, contract SHA-256, source E2
handoff SHA-256 and each member's size/SHA-256. Use fixed ZIP timestamp
`(2026, 1, 1, 0, 0, 0)`, reject paths with absolute or `..` components, symlinks, duplicates,
unlisted members and compression ratios above `250`.

- [ ] **Step 4: Add the repository-independent CLI bootstrap**

At the top of `tools/prepare_tree_privileged_input.py`, insert the repository root before imports:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tree_privileged.inputs import file_sha256, prepare_input
```

Arguments are exactly `--e2-handoff` and `--output`. Print:

```text
TREE_PRIV_INPUT_READY path=<absolute path> sha256=<64 hex> size_bytes=<int>
```

- [ ] **Step 5: Run input and CLI tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_inputs.py
```

Expected: all tests pass, including running the CLI with a working directory outside the repository.

- [ ] **Step 6: Commit input binding**

```bash
git add experiments/tree_privileged/inputs.py tools/prepare_tree_privileged_input.py tests/test_tree_privileged_inputs.py
git commit -m "feat: bind privileged campaign input"
```

### Task 3: Match training pitches and build teacher OOF safely

**Files:**
- Create: `experiments/tree_privileged/matching.py`
- Create: `experiments/tree_privileged/teacher.py`
- Create: `tests/test_tree_privileged_matching.py`
- Create: `tests/test_tree_privileged_teacher.py`

- [ ] **Step 1: Write failing R/F-aware matching tests**

```python
def test_same_franchise_regular_and_futures_games_do_not_share_team_code() -> None:
    main, history = rf_training_fixture()
    result = match_training_pitches(main, history, cutoff_year=2024)
    accepted = result.loc[result["lupi_match_accepted"].eq(1)]
    assert accepted["row_id"].tolist() == main["row_id"].tolist()
    assert accepted["trackman_id"].is_unique
    assert result.loc[main["game_type"].eq("R"), "matched_game_type"].eq("R").all()
    assert result.loc[main["game_type"].eq("F"), "matched_game_type"].eq("F").all()


def test_ambiguous_candidate_games_are_rejected_not_first_selected() -> None:
    main, history = ambiguous_game_fixture()
    result = match_training_pitches(main, history, cutoff_year=2024)
    assert result["lupi_match_accepted"].eq(0).all()
    assert result["trackman_id"].isna().all()
```

- [ ] **Step 2: Run matching tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_matching.py
```

Expected: FAIL because `match_training_pitches` is missing.

- [ ] **Step 3: Implement the training-only matcher**

Use `fit_pitcher_trackman()` and `fit_batter_trackman()` to keep only accepted one-to-one entity
maps. Split official rows into pseudo-games in source order. For each pseudo-game, search TrackMan
games with the same season, month, weekday, top/bottom and inning scope, then require mapped pitcher
and batter overlap. Infer `R` versus `F` from TrackMan team codes: any participating team beginning
with `MIN_` is `F`; all other accepted official KBO team codes are `R`. Run the dynamic-programming
token alignment from `experiments.temporal_portfolio.lupi_matching`, but expose it through this new
module rather than importing private functions.

Return exactly:

```python
MATCH_COLUMNS = (
    "row_id", "trackman_id", "lupi_match_accepted", "matched_game_type",
    "lupi_match_coverage", "lupi_match_mean_cost",
    "lupi_match_candidate_margin", "lupi_match_exact_token_agreement",
    "lupi_match_exact_token_evidence",
)
```

Accept only unique exact token rows inside a game with game coverage at least `0.85`, mean edit cost
at most `0.10`, and candidate margin greater than `0.02`. Reject any globally reused TrackMan ID.

- [ ] **Step 4: Write failing teacher adapter tests**

```python
def test_teacher_vector_is_pitcher_held_out_and_nan_for_unmatched() -> None:
    train, history = teacher_fixture()
    result = build_teacher_oof(train, history, cutoff_year=2024, backend=RecordingBackend())
    assert result.probability.shape == (len(train),)
    assert np.isnan(result.probability[train["row_id"].eq("unmatched")]).all()
    for train_pitchers, valid_pitchers in result.split_evidence:
        assert set(train_pitchers).isdisjoint(valid_pitchers)


def test_teacher_coverage_gate_skips_only_distillation() -> None:
    train, history = low_coverage_fixture()
    result = build_teacher_oof(train, history, cutoff_year=2024, backend=RecordingBackend())
    assert result.status == "insufficient_coverage"
    assert result.distillation_allowed is False
```

- [ ] **Step 5: Implement the teacher adapter**

Join accepted matches to the exact eight `TEACHER_FEATURES`, call the existing
`crossfit_teacher(..., folds=5, seed=3407)` and align with `build_teacher_vector()`. Store coverage
overall and by season/game type plus match, mapping and teacher OOF SHA-256 values. Do not serialize
teacher model bytes into a delivery.

`build_teacher_oof()` returns a frozen `TeacherEvidence` with `probability`, `accepted_mask`,
`coverage`, `latest_coverage`, `status`, `distillation_allowed`, and `split_evidence`. Set
`distillation_allowed=False` when total coverage is below `0.30` or latest-season coverage is below
`0.20`; this must not stop profile candidates.

- [ ] **Step 6: Run matching and teacher tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_matching.py tests/test_tree_privileged_teacher.py tests/test_temporal_portfolio_lupi.py
```

Expected: all tests pass and the existing LUPI suite remains green.

- [ ] **Step 7: Commit matching and teacher support**

```bash
git add experiments/tree_privileged/matching.py experiments/tree_privileged/teacher.py tests/test_tree_privileged_matching.py tests/test_tree_privileged_teacher.py
git commit -m "feat: build privileged TrackMan teacher"
```

### Task 4: Build cutoff-safe hierarchical profiles

**Files:**
- Create: `experiments/tree_privileged/profiles.py`
- Create: `tests/test_tree_privileged_profiles.py`

- [ ] **Step 1: Write failing shrinkage and cutoff tests**

```python
def test_profile_rate_shrinks_to_parent_and_unknown_falls_back() -> None:
    train = profile_fixture(seasons=(2021, 2022))
    state = fit_profiles(train, cutoff_year=2022, strengths=fixture_strengths())
    known = transform_profiles(profile_rows(known=True), state)
    unknown = transform_profiles(profile_rows(known=False), state)
    assert known.loc[0, "profile_pitcher_count_known"] == 1.0
    assert known.loc[0, "profile_pitcher_count_rate"] != known.loc[0, "profile_pitcher_rate"]
    assert unknown.loc[0, "profile_pitcher_count_known"] == 0.0
    assert unknown.loc[0, "profile_pitcher_count_rate"] == unknown.loc[0, "profile_pitcher_rate"]


def test_training_profiles_never_use_same_or_future_season_targets() -> None:
    rows = profile_fixture(seasons=(2021, 2022, 2023))
    expected = build_training_profiles(rows, valid_year=2024, strengths=fixture_strengths())
    changed = rows.copy()
    changed.loc[changed["season"].eq(2023), "control_success"] ^= 1
    replay = build_training_profiles(changed, valid_year=2024, strengths=fixture_strengths())
    mask = rows["season"].le(2023)
    assert_frame_equal(
        expected.loc[rows["season"].eq(2023)],
        replay.loc[rows["season"].eq(2023)],
    )
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_profiles.py
```

Expected: FAIL because `profiles.py` is missing.

- [ ] **Step 3: Implement profile levels and immutable state**

Define levels with exact parentage:

```python
LEVELS = (
    ProfileLevel("pitcher", ("pitcher_id",), "global", "identity"),
    ProfileLevel("batter", ("batter_id",), "global", "identity"),
    ProfileLevel("pitcher_count", ("pitcher_id", "balls_before", "strikes_before"), "pitcher", "interaction"),
    ProfileLevel("pitcher_batter_hand", ("pitcher_id", "batter_hand"), "pitcher", "interaction"),
    ProfileLevel("pitcher_base", ("pitcher_id", "base_state"), "pitcher", "interaction"),
    ProfileLevel("pitcher_game", ("pitcher_id", "game_type"), "pitcher", "interaction"),
    ProfileLevel("batter_pitcher_hand", ("batter_id", "pitcher_hand"), "batter", "interaction"),
    ProfileLevel("batter_count", ("batter_id", "balls_before", "strikes_before"), "batter", "interaction"),
    ProfileLevel("team_count", ("pitcher_team_id", "balls_before", "strikes_before"), "global", "interaction"),
    ProfileLevel("direct_matchup", ("pitcher_id", "batter_id"), "global", "matchup"),
)
```

For every level output count, raw rate, parent rate, shrunk rate, known flag, delta and clipped-logit
delta. `build_training_profiles()` must fit each season's rows from strictly earlier seasons. The
earliest season uses neutral `0.5`, zero count and known `0`. `fit_profiles()` rejects any row after
the cutoff and `transform_profiles()` rejects targets.

- [ ] **Step 4: Add strength-grid selection without extra model jobs**

Implement `select_strengths(train, folds)` by computing profile-only Brier on the first two
validation seasons for the 27 Cartesian combinations from the sealed grids. Pick lowest weighted
Brier, then larger strengths, then lexical order. Do not inspect 2024 to choose strengths. Return the
selected tuple and all scores for review evidence.

- [ ] **Step 5: Run profile tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 6: Commit profiles**

```bash
git add experiments/tree_privileged/profiles.py tests/test_tree_privileged_profiles.py
git commit -m "feat: add rolling target profiles"
```

### Task 5: Compose features and train one candidate job

**Files:**
- Create: `experiments/tree_privileged/features.py`
- Create: `experiments/tree_privileged/training.py`
- Create: `tests/test_tree_privileged_features.py`
- Create: `tests/test_tree_privileged_training.py`

- [ ] **Step 1: Write failing feature-composition tests**

```python
def test_candidate_feature_and_target_boundaries() -> None:
    fit_rows, valid_rows, history = training_fixture(valid_year=2024)
    state, batch = fit_candidate_features(fit_rows, history, valid_year=2024, candidate_id="PD35")
    assert all(name.startswith("profile_") for name in state.profile_columns)
    expected = fit_rows["control_success"].to_numpy(dtype="float64")
    mask = np.isfinite(batch.teacher_probability)
    expected[mask] = 0.65 * expected[mask] + 0.35 * batch.teacher_probability[mask]
    assert_allclose(batch.soft_target, expected)

    transformed = transform_candidate_features(valid_rows.drop(columns="control_success"), state)
    assert transformed.soft_target is None
    assert transformed.teacher_probability is None
```

- [ ] **Step 2: Run feature tests and verify failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_features.py
```

Expected: FAIL on missing feature composition.

- [ ] **Step 3: Implement exact candidate composition**

`P` adds profiles and uses the hard target. `D15` and `D35` use base E2 features and the indicated
teacher lambda. `PD15` and `PD35` add profiles and use the teacher lambda. A profile-only campaign
must still work when teacher coverage fails; distillation candidates return `skipped` with reason
`teacher_coverage_gate` before training.

Freeze `CandidateFeatureState` with the original `TreeFeatureState`, optional `ProfileState`, exact
feature order, categorical columns, selected strengths and teacher evidence hashes. Inference
transformation must not accept `control_success`, TrackMan current-pitch columns, match columns or a
teacher probability.

- [ ] **Step 4: Write failing residual-training tests**

```python
def test_residual_student_fits_soft_target_minus_anchor(tmp_path: Path) -> None:
    factory = RecordingRegressorFactory(prediction=np.array([0.02, -0.01]))
    result = run_candidate_job(
        job=fixture_job("PD35"), data=fixture_data(), baseline=fixture_e2_oof(),
        output_dir=tmp_path, absolute_deadline=time.time() + 60,
        gpu_id=0, model_factory=factory,
    )
    recorded = factory.fit_target
    assert_allclose(recorded, factory.batch.soft_target - factory.batch.anchor)
    assert result.status == "completed"
    assert result.predictions.is_file()


def test_distillation_job_skips_before_gpu_fit_when_coverage_fails(tmp_path: Path) -> None:
    factory = FailingIfCalledFactory()
    result = run_candidate_job(
        job=fixture_job("D15"), data=low_coverage_data(), baseline=fixture_e2_oof(),
        output_dir=tmp_path, absolute_deadline=time.time() + 60,
        gpu_id=0, model_factory=factory,
    )
    assert result.status == "skipped"
    assert result.failure == "teacher_coverage_gate"
```

- [ ] **Step 5: Implement CatBoost residual jobs**

Use `CatBoostRegressor` with RMSE on `soft_target - anchor`, while the validation set remains the
hard target residual `control_success - anchor`. Save `model.cbm`, `feature_state.zip`,
`predictions.csv`, `teacher_evidence.json`, `profile_evidence.json`, `metrics.json` and
`worker_result.json` atomically. Use CatBoost snapshots and the sealed parameters; include `gpu_id`
in `job.json` but not in the model identity.

- [ ] **Step 6: Run feature and training tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_features.py tests/test_tree_privileged_training.py tests/test_tree_expert_training.py
```

Expected: all tests pass and E2 training remains green.

- [ ] **Step 7: Commit candidate training**

```bash
git add experiments/tree_privileged/features.py experiments/tree_privileged/training.py tests/test_tree_privileged_features.py tests/test_tree_privileged_training.py
git commit -m "feat: train privileged residual students"
```

### Task 6: Decide structures, R/F blending and acceptance

**Files:**
- Create: `experiments/tree_privileged/decisions.py`
- Create: `tests/test_tree_privileged_decisions.py`

- [ ] **Step 1: Write failing screen and acceptance tests**

```python
def test_screen_keeps_at_most_two_diverse_candidates() -> None:
    evidence = screen_fixture(
        gains={"P": 0.00008, "D15": 0.00007, "D35": -0.00001, "PD15": 0.00012, "PD35": 0.00011},
        correlations={("PD15", "PD35"): 0.999, ("PD15", "P"): 0.991},
    )
    assert select_confirmation_candidates(evidence) == ("PD15", "P")


def test_acceptance_requires_weighted_recent_fold_seed_and_segment_gates() -> None:
    accepted = decide_candidate(confirmation_fixture(weighted_gain=0.00008, latest_gain=0.00001))
    rejected = decide_candidate(confirmation_fixture(weighted_gain=0.00008, latest_gain=-0.00003))
    assert accepted.status == "accepted"
    assert rejected.status == "rejected"
    assert "latest_min_gain" in rejected.failed_gates


def test_rf_alpha_is_chosen_on_2022_2023_only() -> None:
    source = rf_blend_fixture()
    first = select_rf_blend(source)
    changed = source.copy()
    changed.loc[changed["valid_year"].eq(2024), "target"] ^= 1
    assert select_rf_blend(changed) == first
```

- [ ] **Step 2: Run and verify the missing-module failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_decisions.py
```

Expected: FAIL because decision functions are missing.

- [ ] **Step 3: Implement screening and R/F blend selection**

Aggregate row-aligned E2 and candidate OOF. Rank on weighted gain, latest gain, worst fold gain,
lower E2 residual correlation and simplicity. Remove candidates whose first two folds are both
worse. Keep at most two and reject the lower-ranked one when pairwise prediction correlation exceeds
`0.998`.

For the best `PD` candidate only, evaluate the sealed R/F alpha grid on 2022 and 2023, holding the
other segment at E2. Fix `alpha_R` and `alpha_F` independently, then evaluate 2024 once. Treat the
result as virtual candidate `PD_RF`; no extra model job is scheduled.

- [ ] **Step 4: Implement confirmation and ensemble gates**

Require weighted gain `0.00005`, at least two improved folds, latest gain at least `-0.00002`, maximum
predeclared segment regression `0.00050`, at least two non-worse seeds and finite bounded
probabilities. If two candidates pass, average them only when the average improves on the best by
`0.00002`. Store every gate's observed value and pass/fail result.

- [ ] **Step 5: Run decision tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 6: Commit decisions**

```bash
git add experiments/tree_privileged/decisions.py tests/test_tree_privileged_decisions.py
git commit -m "feat: gate privileged tree candidates"
```

### Task 7: Full fit and row-independent inference

**Files:**
- Create: `experiments/tree_privileged/full_fit.py`
- Create: `experiments/tree_privileged/inference.py`
- Create: `tests/test_tree_privileged_full_fit.py`
- Create: `tests/test_tree_privileged_inference.py`

- [ ] **Step 1: Write failing full-fit gate test**

```python
def test_full_fit_refuses_rejected_or_unbound_decision(tmp_path: Path) -> None:
    with pytest.raises(PrivilegedFullFitError, match="accepted decision"):
        fit_accepted_candidate(rejected_decision(), fixture_sources(), tmp_path)
    with pytest.raises(PrivilegedFullFitError, match="bindings differ"):
        fit_accepted_candidate(accepted_decision(e2_sha256="0" * 64), fixture_sources(), tmp_path)
```

- [ ] **Step 2: Implement accepted-only full fit**

For each accepted seed, fit features through 2024 with `valid_year=2025`, build teacher OOF only for
the training target, and save student models plus the frozen profile/E2 feature states. Use the median
best iteration from the three OOF jobs for each candidate and disable early stopping. Embed the
verified E2 model delivery because final blending must reproduce the accepted E2 probability.

- [ ] **Step 3: Write failing independence audit**

```python
def test_inference_is_identical_singleton_shuffle_reverse_and_batches() -> None:
    predictor = fixture_predictor()
    rows = evaluation_rows(37)
    expected = predictor.predict(rows)
    assert_allclose(predict_singletons(predictor, rows), expected, atol=1e-6, rtol=0)
    assert_allclose(unshuffle(predictor.predict(rows.sample(frac=1, random_state=42))), expected, atol=1e-6, rtol=0)
    assert_allclose(predictor.predict(rows.iloc[::-1])[::-1], expected, atol=1e-6, rtol=0)
    for size in (1, 2, 7, 32):
        assert_allclose(predict_batches(predictor, rows, size), expected, atol=1e-6, rtol=0)
```

- [ ] **Step 4: Implement inference with a hard evaluation boundary**

`PrivilegedPredictor.predict(rows)` validates one DataFrame, transforms each row with frozen E2 and
profile state, averages accepted seed students, reproduces fixed R/F alphas from the decision, and
blends with the nested E2 predictor. It must reject target columns and any current-pitch TrackMan or
match-evidence columns. It must never call `groupby`, `rolling`, `rank`, `shift`, `diff`, `sort_values`
or `value_counts` on evaluation rows.

- [ ] **Step 5: Run full-fit and inference tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_full_fit.py tests/test_tree_privileged_inference.py tests/test_tree_expert_e2_inference.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit full fit and inference**

```bash
git add experiments/tree_privileged/full_fit.py experiments/tree_privileged/inference.py tests/test_tree_privileged_full_fit.py tests/test_tree_privileged_inference.py
git commit -m "feat: fit and audit privileged delivery"
```

### Task 8: Add restartable artifacts and the two-GPU runner

**Files:**
- Create: `experiments/tree_privileged/artifacts.py`
- Create: `experiments/tree_privileged/runner.py`
- Create: `tests/test_tree_privileged_artifacts.py`
- Create: `tests/test_tree_privileged_runner.py`

- [ ] **Step 1: Write failing artifact tests**

```python
def test_rejected_campaign_has_review_resume_handoff_but_no_delivery(tmp_path: Path) -> None:
    result = write_campaign_bundles(rejected_campaign_state(), tmp_path)
    assert result.review.is_file()
    assert result.resume.is_file()
    assert result.handoff.is_file()
    assert result.delivery is None


def test_accepted_campaign_delivery_is_hash_bound(tmp_path: Path) -> None:
    result = write_campaign_bundles(accepted_campaign_state(), tmp_path)
    verified = verify_delivery(result.delivery, accepted_campaign_state().bindings)
    assert verified["artifact_kind"] == "tree_privileged_delivery_v1"
```

- [ ] **Step 2: Implement deterministic artifacts**

Use artifact kinds `tree_privileged_resume_v1`, `tree_privileged_review_v1`,
`tree_privileged_delivery_v1`, and `tree_privileged_handoff_v1`. Bind contract, code inventory,
official train/history, E2 handoff, selected-strength, teacher-match and decision SHA-256 values.
Resume contains only completed atomic jobs and selected caches; review contains no model bytes;
delivery is impossible unless the decision is accepted and full fit finished.

- [ ] **Step 3: Write failing runner scheduling tests**

```python
def test_runner_never_starts_more_than_two_jobs_and_reuses_completed(tmp_path: Path) -> None:
    launcher = RecordingLauncher()
    state = run_campaign(fixture_verified(), tmp_path, launcher=launcher, gpu_ids=(0, 1), clock=FakeClock())
    assert launcher.maximum_concurrency == 2
    replay = run_campaign(fixture_verified(), tmp_path, launcher=FailingIfCalledLauncher(), gpu_ids=(0, 1), clock=FakeClock())
    assert replay.completed_jobs == state.completed_jobs


def test_runner_stops_before_reserve_and_emits_one_handoff(tmp_path: Path) -> None:
    result = run_campaign(fixture_verified(), tmp_path, launcher=SlowLauncher(), gpu_ids=(0, 1), clock=DeadlineClock())
    assert result.status == "paused"
    assert result.handoff.is_file()
    assert len(list((tmp_path / "bundles").glob("*handoff*.zip"))) == 1
```

- [ ] **Step 4: Implement the campaign state machine**

Phases are `cache`, `teacher`, `screen`, `confirm`, `full_fit`, `audit`, `complete`. Acquire a file
lock before initializing shared cache/state, give each worker one GPU through `CUDA_VISIBLE_DEVICES`,
and publish job results with temp-file plus `os.replace`. Never start a job inside the two-hour
new-job guard; reserve the final 75 minutes for bundles. Retry failed jobs only when their identity
matches and their retry count is zero.

- [ ] **Step 5: Run artifact and runner tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_artifacts.py tests/test_tree_privileged_runner.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit runner and artifacts**

```bash
git add experiments/tree_privileged/artifacts.py experiments/tree_privileged/runner.py tests/test_tree_privileged_artifacts.py tests/test_tree_privileged_runner.py
git commit -m "feat: run restartable privileged campaign"
```

### Task 9: Generate the Kaggle cell and user runbook

**Files:**
- Create: `experiments/tree_privileged/kaggle.py`
- Create: `experiments/tree_privileged/requirements-kaggle.txt`
- Create: `experiments/tree_privileged/KAGGLE_CELL.py`
- Create: `tools/build_tree_privileged_kaggle_cell.py`
- Create: `tests/test_tree_privileged_kaggle.py`
- Create: `tests/test_tree_privileged_kaggle_cell.py`
- Create: `docs/TREE_PRIVILEGED_KAGGLE.md`

- [ ] **Step 1: Write failing input-discovery tests**

```python
def test_discovery_accepts_one_official_one_input_and_optional_handoff(tmp_path: Path) -> None:
    official, compact, handoff = kaggle_input_fixture(tmp_path)
    found = discover_inputs(tmp_path)
    assert found.official_data == official
    assert found.campaign_input == compact
    assert found.previous_handoff == handoff


def test_nested_resume_inside_handoff_is_not_a_second_resume(tmp_path: Path) -> None:
    handoff = expanded_handoff_fixture(tmp_path)
    found = discover_inputs(tmp_path)
    assert found.previous_handoff == handoff
```

- [ ] **Step 2: Implement exact discovery and dependencies**

Discover one official root with top-level `train.csv`, `test.csv`, `trackman_history.csv`, and
`sample_submission.csv`; one `tree_privileged_input_v1`; and zero or one independent resume/handoff.
Ignore only a resume that is a strict descendant of the selected outer handoff. Require two GPUs
before campaign start. Pin:

```text
catboost==1.2.10
scipy==1.16.3
```

- [ ] **Step 3: Write failing deterministic-cell tests**

```python
def test_generated_cell_is_deterministic_under_one_megabyte_and_has_one_download_loop(tmp_path: Path) -> None:
    first = build_kaggle_cell(tmp_path / "first.py")
    second = build_kaggle_cell(tmp_path / "second.py")
    assert first.read_bytes() == second.read_bytes()
    source = first.read_text()
    assert len(source.encode()) < 1_000_000
    assert source.count("files.download(") == 1
    assert "TREE_PRIV_CODE_READY" in source
    assert "TREE_PRIV_CAMPAIGN_SUCCESS" in source
```

- [ ] **Step 4: Implement runtime inventory and single cell**

Build a deterministic tar.gz containing only the new package plus explicitly imported E2,
temporal-portfolio and independent-DL modules. Verify runtime SHA-256 before extraction. The cell
installs missing exact dependencies, discovers inputs, verifies two T4 GPUs, runs with absolute
deadline `now + 37_800`, streams the required logs, and calls `files.download()` once for the final
handoff. The handoff contains review and, only when accepted, delivery.

The exact user-visible event prefixes are:

```text
TREE_PRIV_CODE_READY
TREE_PRIV_DEPENDENCIES_READY
TREE_PRIV_INPUTS_VERIFIED
TREE_PRIV_GPU_READY
TREE_PRIV_MATCH_AUDIT
TREE_PRIV_TEACHER_JOB_START
TREE_PRIV_TEACHER_JOB_END
TREE_PRIV_CANDIDATE_JOB_START
TREE_PRIV_CANDIDATE_JOB_END
TREE_PRIV_DECISION
TREE_PRIV_HANDOFF_READY
TREE_PRIV_CAMPAIGN_SUCCESS
TREE_PRIV_ERROR
```

`TREE_PRIV_MATCH_AUDIT` includes total, season, game-type and pitcher coverage plus duplicate and
ambiguous-match counts. Worker logs include fold, seed, epoch or iteration, current metric, best
metric, elapsed seconds and ETA.

- [ ] **Step 5: Generate the checked-in cell**

Run:

```bash
artifacts/tabm_submission_python311/bin/python tools/build_tree_privileged_kaggle_cell.py
```

Expected:

```text
TREE_PRIV_KAGGLE_CELL_READY path=<absolute path> sha256=<64 hex> size_bytes=<less than 1000000>
```

- [ ] **Step 6: Write the runbook**

Document exactly two Kaggle datasets for a fresh run: official `lg-aimers-9th-data` and generated
`tree_privileged_input`. For resume, add the latest handoff as the third dataset. State T4x2,
Internet off after dependencies are available, Save Version/Run All, expected 7–10.5 wall hours,
rerun safety, success log and the one file to return:

```text
TREE_PRIV_CAMPAIGN_SUCCESS status=<completed|completed_no_candidate|paused>
TREE_PRIV_HANDOFF_READY path=/kaggle/working/tree_privileged/bundles/tree_privileged_handoff.zip
```

- [ ] **Step 7: Run Kaggle tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_kaggle.py tests/test_tree_privileged_kaggle_cell.py
artifacts/tabm_submission_python311/bin/python -m py_compile experiments/tree_privileged/KAGGLE_CELL.py
```

Expected: all tests pass and compilation succeeds.

- [ ] **Step 8: Commit the Kaggle handoff**

```bash
git add experiments/tree_privileged/kaggle.py experiments/tree_privileged/requirements-kaggle.txt experiments/tree_privileged/KAGGLE_CELL.py tools/build_tree_privileged_kaggle_cell.py tests/test_tree_privileged_kaggle.py tests/test_tree_privileged_kaggle_cell.py docs/TREE_PRIVILEGED_KAGGLE.md
git commit -m "feat: launch privileged campaign on Kaggle"
```

### Task 10: Run focused and repository regression verification

**Files:**
- Modify only if a failing regression demonstrates a defect in files created by Tasks 1–9.

- [ ] **Step 1: Run the complete new campaign suite**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q tests/test_tree_privileged_*.py
```

Expected: all privileged campaign tests pass.

- [ ] **Step 2: Run reused LUPI and E2/S4 regression suites**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q \
  tests/test_temporal_portfolio_lupi.py \
  tests/test_tree_expert_features.py \
  tests/test_tree_expert_e2_*.py \
  tests/test_tree_expert_s4_*.py
```

Expected: all selected tests pass. Do not modify unrelated dirty TabM campaign files to fix a failure.

- [ ] **Step 3: Verify cell, contract and no-submission boundary statically**

```bash
git diff --check
rg -n "submission_package|files.download|test.csv|groupby|rolling|rank|shift|diff" \
  experiments/tree_privileged tools/prepare_tree_privileged_input.py tools/build_tree_privileged_kaggle_cell.py
```

Expected: `submission_package` is false in the campaign contract and artifacts; evaluation-facing
code contains no cross-row aggregate operation; the generated cell has one download call.

- [ ] **Step 4: Record exact verification evidence**

Add a short section to `docs/TREE_PRIVILEGED_KAGGLE.md` containing the generated cell SHA-256,
contract SHA-256, focused test count and regression test count from this implementation run. Do not
claim the ML candidate is accepted; only a user-run Kaggle handoff can establish that.

- [ ] **Step 5: Commit verification evidence**

```bash
git add docs/TREE_PRIVILEGED_KAGGLE.md
git commit -m "docs: verify privileged campaign handoff"
```

## Completion boundary

Implementation is complete when all Tasks 1–10 pass locally and the generated cell is under 1 MB.
That means only that the experiment is ready for the user to run. It does not mean a candidate is
accepted and does not authorize a DACON submission package. The user must return
`tree_privileged_handoff.zip`; only its verified OOF evidence and accepted delivery can open a later
submission-packaging task.
