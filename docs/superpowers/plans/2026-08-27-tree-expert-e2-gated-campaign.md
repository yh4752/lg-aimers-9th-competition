# Tree Expert E2 Gated Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one restartable Kaggle T4x2 campaign that validates the two E1 CatBoost residual structures over three temporal folds and three seeds, conditionally checks a causal TabM blend, and emits a full-fit model delivery only after every performance, rule, and runtime gate passes.

**Architecture:** Add an E2-only layer beside the existing E1 modules. E2 reuses E1 feature fitting and CatBoost fold training, reads exact E1 and Stage C prediction evidence from a small sealed input archive, and coordinates B0–B3 plus final acceptance through a deterministic state machine. The full-fit and delivery paths are physically separated from review/resume publication, so rejected evidence cannot create a deployable artifact and no component creates a DACON submission package.

**Tech Stack:** Python 3.11/3.13, pandas, NumPy, CatBoost 1.2.10, PyTorch, TabM 0.0.3, rtdl-num-embeddings 0.0.12, pytest, deterministic ZIP/JSON artifacts.

---

## Scope and file map

Create only E2-specific files and generated artifacts. Do not modify the user's dirty files under `experiments/tabm_campaign`, their tests/tools, or `notebooks/INDEPENDENT_DL_RECOVERY_AND_TABR.ipynb`.

| File | Responsibility |
|---|---|
| `experiments/tree_expert/e2_contract.json` | Exact hashes, folds, candidates, gates, budgets, and fixed blend grid |
| `experiments/tree_expert/e2_contracts.py` | Strict typed parser and deterministic E2 job builders |
| `experiments/tree_expert/e2_inputs.py` | Verify E1/Stage C/872 artifacts, create and extract the compact E2 input |
| `experiments/tree_expert/e2_baseline.py` | Reuse F2/F3 Stage C OOF and train the missing F1 TabM baseline |
| `experiments/tree_expert/e2_decisions.py` | Fold metrics, structure gate, seed gate, causal blend selection, final acceptance |
| `experiments/tree_expert/e2_full_fit.py` | Full-data CatBoost fitting and portable frozen feature-state serialization |
| `experiments/tree_expert/e2_inference.py` | Row-independent inference and invariance/resource audits |
| `experiments/tree_expert/e2_artifacts.py` | Deterministic review/resume/handoff and conditional model-delivery ZIPs |
| `experiments/tree_expert/e2_runner.py` | Restartable B0–B3 state machine and two-GPU scheduling |
| `experiments/tree_expert/e2_kaggle.py` | Kaggle input discovery, runtime verification, logging, and final orchestration |
| `experiments/tree_expert/KAGGLE_E2_CELL.py` | Generated single copyable Kaggle cell, under 1 MB |
| `tools/prepare_tree_expert_e2_input.py` | Local CLI that creates `tree_expert_e2_input.zip` |
| `tools/build_tree_expert_e2_kaggle_cell.py` | Deterministically embeds the E2 runtime in one cell |
| `docs/TREE_EXPERT_E2_KAGGLE.md` | User runbook and exact return artifact |
| `tests/test_tree_expert_e2_*.py` | Contract, input, decisions, baseline, runner, artifacts, inference, Kaggle tests |

The E2 package must not contain a function named `build_submission`, must not write `submission.csv`, and must always set `submission_package` to `false`.

### Task 1: Seal the E2 contract

**Files:**
- Create: `experiments/tree_expert/e2_contract.json`
- Create: `experiments/tree_expert/e2_contracts.py`
- Create: `tests/test_tree_expert_e2_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
from dataclasses import replace
import json

import pytest

from experiments.tree_expert.e2_contracts import (
    E2ContractError,
    build_structure_jobs,
    load_e2_contract,
)


def test_e2_contract_is_exact_and_never_a_submission():
    contract = load_e2_contract()
    assert contract.campaign_id == "tree_expert_e2_v1"
    assert contract.folds == ((2021, 2022), (2022, 2023), (2023, 2024))
    assert contract.seeds == (42, 2026, 3407)
    assert contract.structures == ("c1_anchor_residual", "c2_trackman_residual")
    assert contract.review_only is False
    assert contract.submission_package is False
    assert contract.wall_seconds == 21600


def test_structure_jobs_are_deterministic():
    jobs = build_structure_jobs(load_e2_contract(), seed=3407, folds=((2021, 2022),))
    assert [job.job_id for job in jobs] == [
        "e2__c1_anchor_residual__tr2021__va2022__s3407",
        "e2__c2_trackman_residual__tr2021__va2022__s3407",
    ]


def test_changed_hash_fails_closed(tmp_path):
    source = json.loads(load_e2_contract().source_path.read_text())
    source["inputs"]["e1_handoff_sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(source))
    with pytest.raises(E2ContractError, match="input hashes differ"):
        load_e2_contract(path)
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_contracts.py -q
```

Expected: collection fails because `experiments.tree_expert.e2_contracts` does not exist.

- [ ] **Step 3: Add the exact JSON contract**

Use these immutable values in `e2_contract.json`:

```json
{
  "schema_version": 1,
  "campaign_id": "tree_expert_e2_v1",
  "review_only": false,
  "submission_package": false,
  "inputs": {
    "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
    "e1_handoff_sha256": "9e5b55d88715b7a76051d6e8497ffd19617a7a4b25fdafeb7f5f5af16b72fffc",
    "e1_review_sha256": "85a2dcf1a01a4c665c46d075571a4ffb8f463a981f49c22a2f3d73824ae4c2d2",
    "e1_resume_sha256": "66065feae86c0c252fdce1b33cfb0fb56c5f86811a7fc9a40b8c0e2a17af8d3c",
    "stage_c_delivery_sha256": "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a",
    "stage_c_review_sha256": "461c4422cebdc7383577ad6c53b044bfe3d9d15b667383e454591135e4242d2c",
    "tabm_submission_sha256": "ee4d6324eb8d1afec08157526db473834620aabac9647a865e9ff6a6cd1345be",
    "tabm_weight_sha256": "940c358c7e4af258ffec957a8ea42a438f3e639ffc6b5b6f777f77f45db9c945"
  },
  "promoted_structures": ["c1_anchor_residual", "c2_trackman_residual"],
  "folds": [[2021, 2022], [2022, 2023], [2023, 2024]],
  "seeds": [42, 2026, 3407],
  "tabm": {
    "capacity": "p2", "k": 32, "width": 512, "blocks": 4,
    "dropout": 0.1, "num_embedding": "piecewise_linear",
    "loss": "bce", "scheduler": "plateau", "learning_rate": 0.0006,
    "max_epochs": 40, "min_epochs": 3, "patience": 10,
    "effective_batch_size": 4096, "micro_batch_size": 512
  },
  "catboost": {
    "iterations": 800, "depth": 8, "learning_rate": 0.04,
    "l2_leaf_reg": 5.0, "random_strength": 0.5,
    "bootstrap_type": "Bayesian", "bagging_temperature": 0.5,
    "border_count": 128, "max_ctr_complexity": 2,
    "one_hot_max_size": 16, "od_type": "Iter", "od_wait": 60,
    "task_type": "GPU", "allow_writing_files": true
  },
  "blend_candidates": [
    ["catboost", 1.0], ["probability", 0.1], ["probability", 0.2],
    ["probability", 0.3], ["logit", 0.1], ["logit", 0.2], ["logit", 0.3]
  ],
  "gates": {
    "structure_weighted_gain": 0.00010,
    "structure_f3_gain": 0.00010,
    "structure_max_fold_regression": 0.00010,
    "seed_max_weighted_regression": 0.00010,
    "blend_min_sequential_gain": 0.00001,
    "blend_max_fold_regression": 0.00005,
    "accept_weighted_gain": 0.00015,
    "accept_f3_gain": 0.00010,
    "accept_max_fold_regression": 0.00005,
    "accept_max_segment_regression": 0.00050,
    "bootstrap_repeats": 1000,
    "bootstrap_seed": 3407,
    "minimum_segment_rows": 5000
  },
  "runtime": {
    "wall_seconds": 21600,
    "new_job_guard_seconds": 600,
    "snapshot_interval_seconds": 600,
    "inference_rows": 245789,
    "inference_max_seconds": 480,
    "rss_max_bytes": 23622320128,
    "gpu_max_bytes": 21474836480,
    "probability_tolerance": 0.000001
  }
}
```

- [ ] **Step 4: Implement strict typed parsing and job builders**

Define frozen dataclasses `E2Contract`, `E2Job`, and `BlendCandidate`. Reject unknown/missing keys, booleans where integers are expected, non-finite numbers, changed hashes, duplicate folds/seeds/candidates, `review_only=True`, or `submission_package=True`. `build_structure_jobs()` must map `c1` to `objective="residual", use_trackman=False` and `c2` to `objective="residual", use_trackman=True` without importing or changing the E1 contract.

- [ ] **Step 5: Run contract tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the contract**

```bash
git add experiments/tree_expert/e2_contract.json experiments/tree_expert/e2_contracts.py tests/test_tree_expert_e2_contracts.py
git commit -m "feat: seal tree expert E2 contract"
```

### Task 2: Build the compact, lineage-bound E2 input

**Files:**
- Create: `experiments/tree_expert/e2_inputs.py`
- Create: `tools/prepare_tree_expert_e2_input.py`
- Create: `tests/test_tree_expert_e2_inputs.py`

- [ ] **Step 1: Write failing input tests**

Cover these exact cases with tiny deterministic ZIP fixtures:

```python
def test_prepared_input_contains_only_required_evidence(prepared_archive):
    with ZipFile(prepared_archive) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "e1/decision.json",
            "e1/c1_f3_predictions.csv",
            "e1/c2_f3_predictions.csv",
            "stage_c/tabm_f2_predictions.csv",
            "stage_c/tabm_f3_predictions.csv",
            "tabm/script.py",
            "tabm/requirements.txt",
            "tabm/model/inference_manifest.json",
            "tabm/model/numeric_embedding_0.json",
            "tabm/model/preprocessing_state.json",
            "tabm/model/tabm_member_0_seed_3407.pt",
        }


def test_changed_e1_decision_is_rejected(valid_input, tmp_path):
    changed = rewrite_member(valid_input, "e1/decision.json", b'{"status":"rejected"}')
    with pytest.raises(E2InputError):
        verify_and_extract_e2_input(changed, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["duplicate", "traversal", "symlink", "ratio_bomb"])
def test_archive_rejects_unsafe_members(valid_input, mutation, tmp_path):
    changed = mutate_zip_structure(valid_input, mutation)
    with pytest.raises(E2InputError, match="unsafe|duplicate|ratio"):
        verify_and_extract_e2_input(changed, tmp_path / mutation)


@pytest.mark.parametrize("mutation", ["duplicate_row", "reverse_order", "wrong_target"])
def test_stage_c_fold_rows_must_be_aligned(valid_input, mutation, tmp_path):
    changed = mutate_prediction_member(valid_input, "stage_c/tabm_f2_predictions.csv", mutation)
    with pytest.raises(E2InputError, match="row_id|target|alignment"):
        verify_and_extract_e2_input(changed, tmp_path / mutation)


def test_tabm_member_hash_must_match_bound_weight(valid_input, tmp_path):
    changed = flip_member_byte(valid_input, "tabm/model/tabm_member_0_seed_3407.pt")
    with pytest.raises(E2InputError, match="TabM.*SHA-256"):
        verify_and_extract_e2_input(changed, tmp_path / "weight")
```

- [ ] **Step 2: Verify the tests fail because the module is absent**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_inputs.py -q
```

Expected: import failure for `e2_inputs`.

- [ ] **Step 3: Implement original artifact verification**

`prepare_e2_input()` must:

1. SHA-check E1 handoff before opening it.
2. Use `verify_e1_handoff()`, then verify the nested review/resume member hashes.
3. Extract E1 `decision.json` only if `status="completed"` and promoted IDs are exactly `c1_anchor_residual`, `c2_trackman_residual`.
4. Extract both E1 F3 prediction CSVs and validate the seven-column E1 prediction schema.
5. SHA-check and verify Stage C delivery/review, then extract only seed-3407 final predictions for folds `2022->2023` and `2023->2024`.
6. SHA-check the existing 872 package and every one of its six members; separately enforce the bound `.pt` hash.
7. Write a deterministic compact ZIP whose manifest binds every original artifact SHA, nested manifest SHA, member size, and member SHA.

Use `ZipFile.open()` streaming for model weights. Do not copy the original E1 or Stage C ZIP into the compact input.

- [ ] **Step 4: Implement safe extraction and official-data verification**

Return:

```python
@dataclass(frozen=True)
class VerifiedE2Input:
    root: Path
    archive_sha256: str
    manifest_sha256: str
    e1_predictions: Mapping[str, Path]
    tabm_predictions: Mapping[str, Path]
    tabm_runtime_root: Path
    lineage: Mapping[str, str]
```

Extraction must require an empty destination, exact member set, regular non-symlink files, no duplicate/path traversal entries, a 2 GiB member cap, an 8 GiB total cap, and a 250x compression-ratio cap. Reuse the E1 `VerifiedOfficialData` shape but verify it against E2 contract hashes without calling dirty recovery modules.

- [ ] **Step 5: Add a local CLI with robust repository import**

At the top of `tools/prepare_tree_expert_e2_input.py`, use:

```python
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
```

Arguments must be `--e1-handoff`, `--stage-c-delivery`, `--tabm-submission`, and `--output`. End with:

```text
TREE_E2_INPUT_READY path=<absolute> sha256=<sha256> size_bytes=<integer>
```

- [ ] **Step 6: Run input tests and CLI help**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_inputs.py -q
artifacts/tabm_submission_python311/bin/python tools/prepare_tree_expert_e2_input.py --help
```

Expected: tests pass and help lists all four required arguments.

- [ ] **Step 7: Commit input preparation**

```bash
git add experiments/tree_expert/e2_inputs.py tools/prepare_tree_expert_e2_input.py tests/test_tree_expert_e2_inputs.py
git commit -m "feat: seal compact tree expert E2 input"
```

### Task 3: Implement the missing F1 TabM baseline adapter

**Files:**
- Create: `experiments/tree_expert/e2_baseline.py`
- Create: `tests/test_tree_expert_e2_baseline.py`

- [ ] **Step 1: Write failing tests around an injected baseline runtime**

```python
def test_baseline_plan_reuses_f2_f3_and_runs_only_f1(contract, evidence):
    plan = build_baseline_plan(contract, evidence)
    assert [(item.fold, item.action) for item in plan] == [
        ((2021, 2022), "train"),
        ((2022, 2023), "reuse"),
        ((2023, 2024), "reuse"),
    ]


def test_f1_request_is_exact_stage_c_configuration(fake_cache, contract):
    request = make_f1_request(fake_cache, contract)
    assert request.seed == 3407
    assert request.model_config == {
        "architecture": "tabm", "k": 32, "width": 512,
        "blocks": 4, "dropout": 0.1, "num_embedding": "piecewise_linear",
    }
    assert request.training_config["scheduler"] == "plateau"
    assert request.training_config["learning_rate"] == 0.0006


@pytest.mark.parametrize("mutation", ["reverse_order", "wrong_target", "wrong_fold"])
def test_baseline_alignment_rejects_changed_evidence(official_rows, baseline, mutation):
    changed = mutate_baseline(baseline, mutation)
    with pytest.raises(E2BaselineError, match="alignment|target|fold"):
        align_baseline(changed, official_rows, fold=(2021, 2022))


def test_inconclusive_f1_blocks_structure_jobs(fake_baseline_runtime, contract):
    fake_baseline_runtime.result_status = "inconclusive"
    result = prepare_baselines(contract, fake_baseline_runtime)
    assert result.status == "inconclusive"
    assert result.ready_for_structure is False
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_baseline.py -q
```

Expected: import failure for `e2_baseline`.

- [ ] **Step 3: Implement F2/F3 evidence loading**

`load_reused_baselines()` must align each extracted Stage C prediction to official validation rows using `row_id` and target, never positional guessing. It must preserve the diagnostic columns needed by E2 metrics and fail if any probability is non-finite or outside `[0,1]`.

- [ ] **Step 4: Implement F1 training by composing stable APIs**

Use `experiments.tabm_campaign.cache.materialize_fixed_cache`, `CampaignJob`, `TabMAdapter`, `TrainRequest`, and `fit_candidate`. Do not call `run_worker()` because it hardcodes proxy sampling choices and broad runtime provenance. The F1 adapter must use all official rows through 2021 (`sample_mode="full"`), validation 2022, the `dl_standard + hand_matchup` preprocessing spec, and an injected backend in tests. Persist `best_checkpoint.pt`, `checkpoint.pt`, `checkpoint_meta.json`, `predictions.csv`, and `baseline_result.json` atomically.

Use candidate ID:

```text
e2_baseline__tabm_p2_piecewise_linear_bce_plateau__tr2021__va2022__s3407
```

If the deadline stops training before a terminal best checkpoint, return `status="inconclusive"`; never synthesize a baseline from another fold.

- [ ] **Step 5: Run the baseline tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_baseline.py -q
```

Expected: all tests pass using fake caches/backends; no GPU work occurs.

- [ ] **Step 6: Commit the baseline adapter**

```bash
git add experiments/tree_expert/e2_baseline.py tests/test_tree_expert_e2_baseline.py
git commit -m "feat: add exact E2 temporal TabM baseline"
```

### Task 4: Add deterministic structure, seed, blend, and acceptance decisions

**Files:**
- Create: `experiments/tree_expert/e2_decisions.py`
- Create: `tests/test_tree_expert_e2_decisions.py`

- [ ] **Step 1: Write failing gate tests with boundary values**

Test pass/fail at one representable float on each side of every threshold. Include:

```python
def test_structure_tie_prefers_c1():
    c1 = structure_evidence("c1_anchor_residual", gains=(.00010, .00010, .00010))
    c2 = structure_evidence("c2_trackman_residual", gains=(.00010, .00010, .00010))
    decision = decide_structure((c2, c1), contract())
    assert decision.status == "passed"
    assert decision.selected == "c1_anchor_residual"


def test_seed_gate_requires_two_non_worse_seeds_per_fold():
    evidence = seed_evidence(non_worse_counts=(2, 1, 2), weighted_gains=(.0002, .0002, .0002))
    assert decide_seeds(evidence, contract()).status == "rejected"


def test_seed_gate_rejects_one_weighted_regression_over_limit():
    evidence = seed_evidence(non_worse_counts=(3, 3, 3), weighted_gains=(.0002, -.0001001, .0002))
    assert decide_seeds(evidence, contract()).reason == "seed_weighted_regression"


def test_blend_does_not_retune_on_f3():
    evidence = blend_evidence(f1_best=("probability", .2), pooled_f1_f2_best=("logit", .1), f3_best=("probability", .3))
    decision = decide_blend(evidence, contract())
    assert decision.f2_applied == ("probability", .2)
    assert decision.f3_applied == ("logit", .1)


def test_blend_tie_order_is_deterministic():
    candidates = tied_blend_scores()
    assert rank_blends(candidates)[0].method == "catboost"


def test_failed_blend_does_not_reject_standalone_catboost():
    acceptance = decide_acceptance(accepted_catboost_evidence(), rejected_blend(), contract())
    assert acceptance.status == "accepted"
    assert acceptance.predictor == "catboost"


def test_bootstrap_clusters_pitcher_across_folds():
    evidence = pooled_pitcher_fixture()
    result = pitcher_block_bootstrap(evidence, repeats=10, seed=3407)
    assert result.cluster_count == evidence["pitcher_id"].nunique()


def test_max_segment_regression_checks_fold_and_pool():
    metrics = segment_fixture(fold_max=.00020, pooled_max=.00040)
    assert maximum_segment_regression(metrics) == pytest.approx(.00040)
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_decisions.py -q
```

Expected: import failure for `e2_decisions`.

- [ ] **Step 3: Implement aligned fold evidence**

Define frozen `FoldPrediction`, `FoldScore`, `StructureDecision`, `SeedDecision`, `BlendDecision`, and `AcceptanceDecision`. All prediction operations must join on `(fold, row_id, target)` with one-to-one validation. The weighted score denominator is the total number of validation rows, not the number of folds.

- [ ] **Step 4: Implement structure and seed gates**

Structure rank key must be:

```python
(
    min(fold_gains),
    weighted_gain,
    f3_gain,
    candidate_id == "c1_anchor_residual",
)
```

Apply the three preliminary thresholds after ranking. For seeds, explicitly validate all finite probabilities/Briers/best iterations, the per-seed weighted regression cap, sign agreement between seed-3407 and ensemble weighted gain, and at least two non-worse seeds on each fold.

- [ ] **Step 5: Implement causal blend and final acceptance**

Clip to `[1e-5, 1-1e-5]` before logit transforms. Select a fixed candidate on F1, evaluate it on F2, then select from pooled F1+F2 and evaluate that exact candidate on F3. Record both selection stages. The standalone three-seed CatBoost ensemble determines acceptance; blend failure falls back to CatBoost and cannot overturn standalone acceptance.

Bootstrap 1,000 times by pooled `pitcher_id`; compute fold and pooled segments for `game_type`, `game_month`, `pitcher_id_known`, and `batter_id_known`, skipping only groups below 5,000 rows.

- [ ] **Step 6: Run decision tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_decisions.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit decisions**

```bash
git add experiments/tree_expert/e2_decisions.py tests/test_tree_expert_e2_decisions.py
git commit -m "feat: add gated E2 model decisions"
```

### Task 5: Reuse E1 CatBoost training safely across E2 folds and seeds

**Files:**
- Create: `experiments/tree_expert/e2_training.py`
- Create: `tests/test_tree_expert_e2_training.py`

- [ ] **Step 1: Write failing wrapper tests**

```python
def test_e2_job_maps_to_e1_without_changing_feature_semantics(e2_job, verified_data):
    captured = {}
    result = run_e2_fold_job(
        job=e2_job,
        data=verified_data,
        baseline=baseline_frame(),
        output_dir=tmp_path,
        absolute_deadline=10**12,
        gpu_id=0,
        fold_runner=lambda **kwargs: captured.update(kwargs) or completed_result(),
    )
    mapped = captured["job"]
    assert mapped.train_end_year == e2_job.train_end_year
    assert mapped.valid_year == e2_job.valid_year
    assert mapped.seed == e2_job.seed
    assert mapped.objective == "residual"
    assert mapped.use_trackman is e2_job.use_trackman


@pytest.mark.parametrize("field", ["candidate_id", "fold", "seed", "model_path"])
def test_e2_result_rejects_changed_identity(completed_e2_result, field):
    changed = mutate_result(completed_e2_result, field)
    with pytest.raises(E2TrainingError, match="identity|artifact"):
        verify_fold_result(changed, completed_e2_result.binding)


def test_completed_job_reuse_requires_terminal_files(completed_job_dir):
    (completed_job_dir / "predictions.csv").unlink()
    assert reusable_completed_job(completed_job_dir, expected_binding()) is None
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_training.py -q
```

Expected: import failure for `e2_training`.

- [ ] **Step 3: Implement the narrow wrapper**

Map each `E2Job` to an `E1Job`, pass an E2-derived `E1Contract` clone containing the same CatBoost/TrackMan settings, and call `run_e1_job()`. The wrapper adds `e2_job_binding.json` with contract, code, data, input, candidate, fold, and seed hashes. Reuse a completed job only if `worker_result.json`, `metrics.json`, `predictions.csv`, `model.cbm`, and the binding all verify.

Do not alter `experiments/tree_expert/training.py`; this preserves the already measured E1 behavior.

- [ ] **Step 4: Run wrapper tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_training.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the training wrapper**

```bash
git add experiments/tree_expert/e2_training.py tests/test_tree_expert_e2_training.py
git commit -m "feat: extend E1 tree training across E2 folds"
```

### Task 6: Freeze full-fit feature state and fit accepted CatBoost models

**Files:**
- Create: `experiments/tree_expert/e2_full_fit.py`
- Create: `tests/test_tree_expert_e2_full_fit.py`

- [ ] **Step 1: Write failing serialization and gating tests**

```python
def test_full_fit_iterations_use_median_plus_one_and_clip():
    assert full_fit_iterations((3, 0, 8)) == 50
    assert full_fit_iterations((90, 120, 100)) == 101
    assert full_fit_iterations((500, 410, 430)) == 400


def test_feature_state_round_trip_is_prediction_equivalent(fitted_state, rows, tmp_path):
    export_frozen_tree_state(fitted_state, tmp_path)
    restored = load_frozen_tree_state(tmp_path)
    assert_frame_equal(
        transform_tree_features(rows, fitted_state).frame,
        transform_tree_features(rows, restored).frame,
        check_exact=True,
    )


def test_rejected_acceptance_cannot_call_full_fit():
    with pytest.raises(E2FullFitError, match="not accepted"):
        accepted_full_fit_token(rejected_acceptance())


def test_c1_omits_trackman_and_c2_requires_cutoff_2024_lookup(fitted_c1, fitted_c2, tmp_path):
    export_frozen_tree_state(fitted_c1, tmp_path / "c1")
    assert not (tmp_path / "c1" / "trackman_lookup.csv").exists()
    export_frozen_tree_state(fitted_c2, tmp_path / "c2")
    payload = json.loads((tmp_path / "c2" / "feature_state.json").read_text())
    assert payload["trackman_cutoff_year"] == 2024
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_full_fit.py -q
```

Expected: import failure for `e2_full_fit`.

- [ ] **Step 3: Implement portable state serialization**

Write canonical JSON plus CSV, not pickle. Store:

- `feature_state.json`: valid year 2025, prior, feature/categorical order, source hashes, anchor formula ID `e1_anchor_v1`, clip bounds.
- `s1_pitcher.csv` and `s1_batter.csv`: sorted snapshots with exact dtypes recorded in JSON.
- `trackman_lookup.csv` only for c2, with cutoff 2024, lookup SHA, bundle SHAs, columns, and dtypes.

Reconstruct immutable states through `S1State._from_snapshot()` and `PitcherTrackmanState._from_lookup()` after validating all hashes and schemas. These constructors are existing controlled classmethods; never mutate their private fields.

- [ ] **Step 4: Implement full fitting behind a typed accepted token**

Define:

```python
@dataclass(frozen=True)
class AcceptedForFullFit:
    candidate_id: str
    predictor: str
    seeds: tuple[int, ...]
    iterations: Mapping[int, int]
    decision_sha256: str
```

Only `accepted_full_fit_token(AcceptanceDecision)` may create this value, and it must reject any status other than `accepted`. Fit 2019–2024 rows for seeds 42, 2026, 3407 with no eval set and no evaluation data. Save each `.cbm` atomically and record model SHA, seed, iteration count, feature order, and CatBoost version.

- [ ] **Step 5: Run full-fit tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_full_fit.py -q
```

Expected: all tests pass with fake models and tiny feature fixtures; no full-data training occurs.

- [ ] **Step 6: Commit full-fit code**

```bash
git add experiments/tree_expert/e2_full_fit.py tests/test_tree_expert_e2_full_fit.py
git commit -m "feat: add gated E2 full-fit model state"
```

### Task 7: Implement row-independent inference and its audit

**Files:**
- Create: `experiments/tree_expert/e2_inference.py`
- Create: `tests/test_tree_expert_e2_inference.py`

- [ ] **Step 1: Write failing inference tests**

```python
def test_predictions_are_invariant_to_order_batch_and_companions(runtime, rows):
    expected = runtime.predict(rows)
    assert_allclose(runtime.predict(rows.iloc[::-1])[::-1], expected, atol=1e-6, rtol=0)
    assert_allclose(concat_batches(runtime, rows, 1), expected, atol=1e-6, rtol=0)
    assert_allclose(concat_batches(runtime, rows, 37), expected, atol=1e-6, rtol=0)
    assert_allclose(runtime.predict(rows.iloc[[3]])[0], expected[3], atol=1e-6, rtol=0)


@pytest.mark.parametrize("mutation", ["target", "duplicate_row_id", "missing_row_id"])
def test_inference_rejects_invalid_rows(runtime, rows, mutation):
    changed = mutate_inference_rows(rows, mutation)
    with pytest.raises(E2InferenceError):
        runtime.predict(changed)


def test_prediction_error_has_no_fallback(runtime, rows):
    runtime.models[0].raise_on_predict = True
    with pytest.raises(RuntimeError, match="model failure"):
        runtime.predict(rows)


def test_probability_and_logit_blends_match_formula():
    tabm = np.array([.2, .8])
    cat = np.array([.4, .6])
    assert_allclose(blend_probabilities(tabm, cat, "probability", .3), .7 * tabm + .3 * cat)
    expected = expit(.7 * logit(tabm) + .3 * logit(cat))
    assert_allclose(blend_probabilities(tabm, cat, "logit", .3), expected)


def test_static_source_has_no_cross_row_eval_operations():
    source = Path("experiments/tree_expert/e2_inference.py").read_text()
    for token in (".groupby(", ".rolling(", ".shift(", ".expanding(", ".cumsum("):
        assert token not in source
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_inference.py -q
```

Expected: import failure for `e2_inference`.

- [ ] **Step 3: Implement inference runtime**

Load the frozen state and three CatBoost models once. Transform each input row using only row-local operations plus saved S1/TrackMan lookups, average three CatBoost probabilities, and optionally combine the exact packaged TabM probability using the selected fixed method/weight. Enforce exact input/output `row_id`, finite values, and clip to `[1e-5,1-1e-5]`. Raise on any model or preprocessing error; there is no sample-submission fallback.

- [ ] **Step 4: Implement audit runner**

`audit_inference(runtime, mock_rows, contract)` must record singleton, reverse, seeded shuffle, batch sizes 1/257/4096, and companion-row checks. Measure wall time, peak process RSS, and peak CUDA allocation. Return a failed typed audit if 245,789 rows exceed 480 seconds, RSS exceeds 22 GiB, GPU allocation exceeds 20 GiB, or invariance exceeds `1e-6`.

- [ ] **Step 5: Run inference tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_inference.py -q
```

Expected: all tests pass using a deterministic fake predictor.

- [ ] **Step 6: Commit inference**

```bash
git add experiments/tree_expert/e2_inference.py tests/test_tree_expert_e2_inference.py
git commit -m "feat: audit row-independent E2 inference"
```

### Task 8: Create deterministic review, resume, handoff, and conditional delivery artifacts

**Files:**
- Create: `experiments/tree_expert/e2_artifacts.py`
- Create: `tests/test_tree_expert_e2_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

```python
def test_rejected_handoff_has_review_resume_and_no_delivery(rejected_state, tmp_path):
    paths = write_fixture_bundles(state=rejected_state, output_dir=tmp_path)
    with ZipFile(paths.handoff) as archive:
        assert set(archive.namelist()) == {
            "handoff_manifest.json", "tree_expert_e2_review.zip",
            "tree_expert_e2_resume.zip", "tree_expert_e2.log",
        }
    assert paths.delivery is None


def test_accepted_delivery_is_model_only_not_submission(accepted_state, tmp_path):
    paths = write_fixture_bundles(state=accepted_state, output_dir=tmp_path)
    with ZipFile(paths.delivery) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["review_only"] is False
        assert manifest["submission_package"] is False
        assert "script.py" not in archive.namelist()
        assert "submission.csv" not in archive.namelist()


def test_delivery_writer_requires_accepted_token(delivery_sources, tmp_path):
    with pytest.raises(TypeError):
        write_e2_delivery(rejected_acceptance(), delivery_sources, tmp_path / "delivery.zip")


def test_bundle_is_reproducible_and_tamper_evident(accepted_state, tmp_path):
    first = write_fixture_bundles(accepted_state, tmp_path / "first")
    second = write_fixture_bundles(accepted_state, tmp_path / "second")
    assert file_sha256(first.handoff) == file_sha256(second.handoff)
    changed = flip_member_byte(first.handoff, "tree_expert_e2_review.zip")
    with pytest.raises(E2ArtifactError):
        verify_e2_handoff(changed)


@pytest.mark.parametrize("binding", ["code_sha256", "contract_sha256", "train_sha256", "input_sha256"])
def test_resume_rejects_changed_binding(valid_resume, binding):
    changed = dict(expected_bindings())
    changed[binding] = "0" * 64
    with pytest.raises(E2ArtifactError, match="binding"):
        verify_e2_resume(valid_resume, changed)
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_artifacts.py -q
```

Expected: import failure for `e2_artifacts`.

- [ ] **Step 3: Implement artifact safety primitives**

Use fixed ZIP timestamps, sorted member names, atomic rename, explicit file modes, exact member manifests, SHA/size verification, duplicate/path traversal/symlink/ratio/size rejection, and strict binding keys. Review includes decisions, prediction evidence, segment/bootstrap/resource/rule audits, and logs. Resume additionally includes completed job models/checkpoints/snapshots/state.

- [ ] **Step 4: Implement the conditional delivery boundary**

`write_e2_delivery(token, delivery_sources, output)` is the only delivery writer. It must verify all three models, frozen feature state, full-fit manifest, acceptance decision SHA, inference audit pass, and optional exact TabM hashes before writing. It never writes executable submission code.

- [ ] **Step 5: Implement one final handoff**

Always include review, resume, and log. Add delivery only when it exists; do not include an absent-delivery placeholder in the handoff manifest. The final line is:

```text
TREE_E2_HANDOFF_READY path=<absolute-path> sha256=<sha256> status=<status> delivery=yes|no
```

- [ ] **Step 6: Run artifact tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_artifacts.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit artifacts**

```bash
git add experiments/tree_expert/e2_artifacts.py tests/test_tree_expert_e2_artifacts.py
git commit -m "feat: seal E2 evidence and conditional delivery"
```

### Task 9: Build the restartable B0–B3 campaign state machine

**Files:**
- Create: `experiments/tree_expert/e2_runner.py`
- Create: `tests/test_tree_expert_e2_runner.py`

- [ ] **Step 1: Write failing state-machine tests with a fake two-GPU runtime**

```python
def test_pass_path_runs_exact_phase_order(fake_runtime, accepted_scores):
    result = run_fixture_campaign(fake_runtime, accepted_scores)
    assert fake_runtime.phase_order == ["B0", "B1", "B2", "B3", "FULL_FIT", "AUDIT"]
    assert result.status == "accepted"
    assert result.bundles.delivery is not None


def test_structure_rejection_stops_later_phases(fake_runtime):
    result = run_fixture_campaign(fake_runtime, rejected_structure_scores())
    assert result.status == "rejected_structure"
    assert fake_runtime.phase_order == ["B0", "B1"]


def test_seed_rejection_stops_before_blend(fake_runtime):
    result = run_fixture_campaign(fake_runtime, rejected_seed_scores())
    assert result.status == "rejected_seed_instability"
    assert "B3" not in fake_runtime.phase_order


def test_blend_rejection_falls_back_to_catboost(fake_runtime):
    result = run_fixture_campaign(fake_runtime, accepted_catboost_rejected_blend_scores())
    assert result.status == "accepted"
    assert result.predictor == "catboost"


def test_baseline_failure_is_fail_closed(fake_runtime):
    fake_runtime.baseline_status = "failed"
    result = run_fixture_campaign(fake_runtime, accepted_scores())
    assert result.status == "failed"
    assert fake_runtime.phase_order == ["B0"]


def test_only_exact_completed_jobs_are_reused(fake_runtime, resume_with_one_completed_job):
    result = run_fixture_campaign(fake_runtime, accepted_scores(), resume=resume_with_one_completed_job)
    assert result.reused == (resume_with_one_completed_job.job_id,)


def test_deadline_publishes_resume_without_new_job(fake_runtime):
    result = run_fixture_campaign(fake_runtime, accepted_scores(), seconds_remaining=599)
    assert result.status == "budget_inconclusive"
    assert result.bundles.resume.is_file()
    assert fake_runtime.started_jobs == []


def test_independent_failure_preserves_sibling_evidence(fake_runtime):
    fake_runtime.fail_job = "e2__c2_trackman_residual__tr2021__va2022__s3407"
    result = run_fixture_campaign(fake_runtime, accepted_scores())
    assert result.status == "failed"
    assert "e2__c1_anchor_residual__tr2021__va2022__s3407" in result.completed
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_runner.py -q
```

Expected: import failure for `e2_runner`.

- [ ] **Step 3: Implement explicit phase state**

Use this closed phase set:

```python
PHASES = ("B0", "B1", "B2", "B3", "ACCEPTANCE", "FULL_FIT", "AUDIT", "TERMINAL")
TERMINAL_STATUSES = {
    "accepted", "rejected_structure", "rejected_seed_instability",
    "rejected_acceptance", "budget_inconclusive", "failed",
}
```

Persist `stage_state.json` after each job and at most every 600 seconds. State contains exact bindings, phase, terminal status, completed/skipped/failed/active jobs, decisions, and artifact paths. Transition validation must reject jumping over phases.

- [ ] **Step 4: Implement two-GPU scheduling**

Use two spawned processes only for independent GPU jobs, set `CUDA_VISIBLE_DEVICES` per worker, and schedule at most one job per GPU. B1 jobs are c1/c2 × F1/F2; F3 is restored from E1. B2 jobs are selected structure × seeds 42/2026 × F1/F2/F3; seed 3407 is reused. B0 precedes all CatBoost jobs. Never start a job inside the final 600 seconds.

- [ ] **Step 5: Wire decisions and gated full fit**

Call structure decision after B1, seed decision after B2, blend decision after B3, and standalone acceptance before full fit. Only an accepted token may enter `FULL_FIT`. After full fit, run the inference audit; an audit failure produces `failed`, preserves review/resume evidence, and writes no delivery.

- [ ] **Step 6: Run runner tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_runner.py -q
```

Expected: all tests pass and fake runtime records no GPU/full-data work on rejected paths.

- [ ] **Step 7: Commit runner**

```bash
git add experiments/tree_expert/e2_runner.py tests/test_tree_expert_e2_runner.py
git commit -m "feat: orchestrate restartable E2 gates"
```

### Task 10: Package a single Kaggle T4x2 cell

**Files:**
- Create: `experiments/tree_expert/e2_kaggle.py`
- Create: `tools/build_tree_expert_e2_kaggle_cell.py`
- Create: `experiments/tree_expert/KAGGLE_E2_CELL.py`
- Create: `tests/test_tree_expert_e2_kaggle.py`
- Create: `tests/test_tree_expert_e2_kaggle_cell.py`

- [ ] **Step 1: Write failing Kaggle wrapper tests**

```python
def test_discovers_one_official_data_and_one_e2_input(kaggle_input_root):
    discovered = discover_inputs(kaggle_input_root)
    assert discovered.official_data.name == "official"
    assert discovered.e2_input.name == "tree_expert_e2_input.zip"


def test_optional_resume_rejects_ambiguous_sources(kaggle_input_root):
    add_valid_resume(kaggle_input_root, "first")
    add_valid_resume(kaggle_input_root, "second")
    with pytest.raises(E2KaggleError, match="resume count"):
        discover_inputs(kaggle_input_root)


def test_requires_two_cuda_devices(fake_torch):
    fake_torch.cuda.device_count_value = 1
    with pytest.raises(E2KaggleError, match="two CUDA"):
        verify_gpu(fake_torch)


def test_runtime_inventory_hash_changes(tmp_path):
    write_runtime_fixture(tmp_path)
    before = runtime_identity_sha256(tmp_path)
    (tmp_path / runtime_member_names()[0]).write_text("changed")
    assert runtime_identity_sha256(tmp_path) != before


def test_generated_cell_is_small_and_has_no_auto_download():
    path = Path("experiments/tree_expert/KAGGLE_E2_CELL.py")
    source = path.read_text()
    assert path.stat().st_size < 1_000_000
    assert "files.download" not in source


def test_generated_cell_has_terminal_markers():
    source = Path("experiments/tree_expert/KAGGLE_E2_CELL.py").read_text()
    assert "TREE_E2_HANDOFF_READY" in source
    assert "TREE_EXPERT_ERROR stage=" in source
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_kaggle.py tests/test_tree_expert_e2_kaggle_cell.py -q
```

Expected: import/file failures for the E2 Kaggle runtime and cell.

- [ ] **Step 3: Implement input discovery and dependency/GPU checks**

Search `/kaggle/input` for manifests, not dataset folder names. Require exactly one `tree_expert_e2_input_v1`, one official root containing the two exact-hash CSVs, and zero or one E2 resume source. Verify `catboost==1.2.10`, `tabm==0.0.3`, `rtdl_num_embeddings==0.0.12`, and exactly two CUDA devices before campaign work.

- [ ] **Step 4: Implement deterministic runtime embedding**

The builder must include only required E2 files plus stable E1/temporal/independent-DL dependencies, `e2_contract.json`, and `requirements-kaggle.txt`. Compute a canonical runtime inventory SHA and verify it after extraction. The generated cell must:

- be valid Python and under 1,000,000 source bytes;
- use `/kaggle/working/tree_expert_e2`;
- preserve output/resume on rerun;
- stream logs to stdout and a file;
- never call `files.download`, browser APIs, GitHub, or network APIs except package installation;
- catch top-level errors as `TREE_EXPERT_ERROR stage=<stage> type=<type> message=<message>` and re-raise.

- [ ] **Step 5: Generate the cell and run tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python tools/build_tree_expert_e2_kaggle_cell.py
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_kaggle.py tests/test_tree_expert_e2_kaggle_cell.py -q
```

Expected: builder prints `TREE_E2_CELL_READY sha256=<sha256> size_bytes=<less-than-1000000>` and tests pass.

- [ ] **Step 6: Commit Kaggle runtime and generated cell**

```bash
git add experiments/tree_expert/e2_kaggle.py experiments/tree_expert/KAGGLE_E2_CELL.py tools/build_tree_expert_e2_kaggle_cell.py tests/test_tree_expert_e2_kaggle.py tests/test_tree_expert_e2_kaggle_cell.py
git commit -m "feat: add one-cell Kaggle E2 campaign"
```

### Task 11: Write the user runbook

**Files:**
- Create: `docs/TREE_EXPERT_E2_KAGGLE.md`
- Test: `tests/test_tree_expert_e2_kaggle_cell.py`

- [ ] **Step 1: Write exact local input preparation instructions**

Document this copyable command without escaped underscores:

```bash
cd /path/to/lg-aimers-9th-competition

artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tree_expert_e2_input.py \
  --e1-handoff /path/to/Downloads/tree_expert_e1_handoff.zip \
  --stage-c-delivery /path/to/Downloads/tabm_colab_stage_C_delivery.zip \
  --tabm-submission artifacts/tabm_submission_version_d_superseded_data_only/submit.zip \
  --output artifacts/tree_expert_e2_input.zip
```

Explain that this is a local verification/repack step, takes roughly 1–3 minutes, uses no GPU, and is safe to rerun because it atomically replaces only the explicit output file.

- [ ] **Step 2: Document the Kaggle run**

Required datasets:

1. `lg-aimers-9th-data`
2. a Kaggle dataset created from `artifacts/tree_expert_e2_input.zip` (Kaggle may show its members unpacked)
3. only on rerun: the previous E2 handoff or resume dataset

Instructions: select `GPU T4 x2`, internet on only if dependencies are not already present, paste all of `KAGGLE_E2_CELL.py` into one cell, run normally or Save Version. State expected runtime 45–90 minutes, hard cap 6 hours, and explain that Output Data may remain empty until the first stable publication.

- [ ] **Step 3: Document logs and return file**

List the exact phase logs from the design spec and tell the user to return only:

```text
/kaggle/working/tree_expert_e2/bundles/tree_expert_e2_handoff.zip
```

If the final marker is an error, request the full log plus the latest resume ZIP. Explicitly state that E2 does not create a DACON submission and that an accepted delivery still requires a separate local submission audit.

- [ ] **Step 4: Add a documentation assertion and run it**

Extend the cell test to assert the document contains `45–90`, `6시간`, `T4 x2`, `TREE_E2_HANDOFF_READY`, and `제출 파일을 만들지 않는다`.

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_kaggle_cell.py -q
```

Expected: pass.

- [ ] **Step 5: Commit the runbook**

```bash
git add docs/TREE_EXPERT_E2_KAGGLE.md tests/test_tree_expert_e2_kaggle_cell.py
git commit -m "docs: add tree expert E2 Kaggle runbook"
```

### Task 12: Run the complete static and fixture verification

**Files:**
- Modify only if a failure exposes an E2 defect: files created in Tasks 1–11

- [ ] **Step 1: Confirm user-owned changes remain untouched**

Run:

```bash
git status --short
git diff -- experiments/tabm_campaign tests/test_tabm_campaign_colab_cell.py tests/test_tabm_campaign_colab_recovery.py tests/test_tabm_row_feature_colab.py tests/test_tabm_row_feature_proxy.py tools/prepare_tabm_colab_stage_c_handoff.py tools/prepare_tabm_row_feature_colab_input.py notebooks/INDEPENDENT_DL_RECOVERY_AND_TABR.ipynb
```

Expected: the pre-existing changes are still present and no E2 commit contains them.

- [ ] **Step 2: Run all E2 tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_e2_*.py -q
```

Expected: all E2 tests pass.

- [ ] **Step 3: Run the existing E1 regression suite**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_tree_expert_contracts.py \
  tests/test_tree_expert_inputs.py \
  tests/test_tree_expert_features.py \
  tests/test_tree_expert_training.py \
  tests/test_tree_expert_metrics.py \
  tests/test_tree_expert_artifacts.py \
  tests/test_tree_expert_runner.py \
  tests/test_tree_expert_kaggle.py \
  tests/test_tree_expert_kaggle_cell.py -q
```

Expected: all existing E1 tests pass.

- [ ] **Step 4: Compile and scan the generated runtime**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/tree_expert/e2_*.py \
  experiments/tree_expert/KAGGLE_E2_CELL.py \
  tools/prepare_tree_expert_e2_input.py \
  tools/build_tree_expert_e2_kaggle_cell.py

rg -n "submission\.csv|files\.download|requests\.|urllib|github\.com|groupby\(|rolling\(|expanding\(|cumsum\(|shift\(" \
  experiments/tree_expert/e2_*.py experiments/tree_expert/KAGGLE_E2_CELL.py
```

Expected: compile succeeds. Scan output may contain only deliberate static-audit token lists or training-side pooled metrics; it must contain no submission creation, auto-download, external fetch, or evaluation-path cross-row operation.

- [ ] **Step 5: Rebuild twice and prove the cell is deterministic**

Run:

```bash
artifacts/tabm_submission_python311/bin/python tools/build_tree_expert_e2_kaggle_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_E2_CELL.py
artifacts/tabm_submission_python311/bin/python tools/build_tree_expert_e2_kaggle_cell.py
shasum -a 256 experiments/tree_expert/KAGGLE_E2_CELL.py
wc -c experiments/tree_expert/KAGGLE_E2_CELL.py
```

Expected: both SHA-256 values are identical and size is below 1,000,000 bytes.

- [ ] **Step 6: Inspect the final E2-only diff**

Run:

```bash
git diff --stat HEAD~1..HEAD
git log --oneline --max-count=12
git status --short
```

Expected: only E2 files and its docs/tests are in the E2 commits; unrelated dirty files remain unstaged. Do not push.

## Completion boundary

Implementation is complete only when Tasks 1–12 pass locally. That does not mean the candidate is accepted and does not authorize a submission package. The user must run the generated cell on Kaggle T4x2 and return `tree_expert_e2_handoff.zip`; only the evidence inside that artifact can determine whether a later submission-building task is allowed.
