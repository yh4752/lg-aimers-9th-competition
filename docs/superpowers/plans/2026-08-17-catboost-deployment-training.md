# CatBoost Deployment Training Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a review-only, resumable Colab campaign that validates one fixed CatBoost tree count on two temporal folds and trains a frozen full-data CatBoost model only when the deployment-alignment gate passes.

**Architecture:** Add an isolated `experiments/catboost_deployment` package that consumes the already verified training input, Stage C delivery, and promoted blend delivery. It trains two 400-tree alignment models, evaluates seven preregistered tree prefixes at the already fixed 70:30 blend, and conditionally trains one full-data model. The campaign emits review/resume evidence only; it contains no evaluation-data inference or submission writer.

**Tech Stack:** Python 3.11, pandas, NumPy, CatBoost 1.2.10 on Colab T4, standard-library JSON/ZIP/hash/subprocess utilities, pytest with injected fake CatBoost models.

---

## Scope boundary

This plan stops after the user returns `catboost_full_training_delivery.zip`.
It does not modify `submission/package.py`, render a dual-model `script.py`, read
`test.csv`, or create `submit.zip`. The returned frozen model hash is required
before a separate inference-validation implementation plan can be written.

## File map

```text
experiments/catboost_deployment/
├── __init__.py                         # package marker only
├── contract.json                       # immutable inputs, grid, gates and jobs
├── contracts.py                        # strict contract parser and job definitions
├── inputs.py                           # three/four upload trust boundary
├── metrics.py                          # fixed-prefix 70:30 OOF decision
├── state.py                            # strict CatBoost preprocessing-state JSON
├── training.py                         # alignment/full CatBoost worker APIs and CLI
├── artifacts.py                        # review/resume/final-delivery evidence
├── runner.py                           # sequential state machine and gate
├── colab.py                            # direct-upload supervisor and recovery
├── runtime_inventory.py                # exact embedded source closure
├── requirements-colab.txt              # catboost==1.2.10
└── COLAB_CATBOOST_DEPLOYMENT_CELL.py   # generated one-cell handoff

tools/build_catboost_deployment_colab_cell.py
docs/CATBOOST_DEPLOYMENT_TRAINING_COLAB.md
tests/test_catboost_deployment_*.py
```

## Task 1: Seal the deployment contract and job identities

**Files:**
- Create: `experiments/catboost_deployment/__init__.py`
- Create: `experiments/catboost_deployment/contract.json`
- Create: `experiments/catboost_deployment/contracts.py`
- Create: `tests/test_catboost_deployment_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

Test exact source identities, prefix grid, one blend weight, CatBoost parameters,
gate thresholds and job order:

```python
def test_contract_is_preregistered() -> None:
    contract = load_contract()
    assert contract.source_blend_delivery_sha256 == (
        "ab7ca41e98e2b94b997369d7c777f8293c64acf61114f314110d17bf743edcfa"
    )
    assert contract.tabm_weight == 0.70
    assert contract.tree_prefixes == (4, 32, 64, 128, 192, 296, 400)
    assert contract.minimum_weighted_gain == 0.00003
    assert contract.maximum_fold_regression == 0.00003
    assert contract.catboost_parameters["iterations"] == 400
    assert contract.catboost_parameters["random_seed"] == 42
    assert contract.catboost_parameters["task_type"] == "GPU"


def test_jobs_are_two_alignment_folds_then_full_fit() -> None:
    jobs = build_jobs(load_contract())
    assert [(job.kind, job.train_end_year, job.valid_year) for job in jobs] == [
        ("alignment", 2022, 2023),
        ("alignment", 2023, 2024),
        ("full_fit", 2024, None),
    ]
```

Also mutate every fixed value and require `DeploymentContractError` rather than
silent defaults.

- [ ] **Step 2: Run the tests and confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_contracts.py -q
```

Expected: collection fails with
`ModuleNotFoundError: experiments.catboost_deployment`.

- [ ] **Step 3: Implement strict dataclasses and parser**

Use these public types and signatures:

```python
@dataclass(frozen=True)
class DeploymentJob:
    job_id: str
    kind: str
    train_end_year: int
    valid_year: int | None
    seed: int


@dataclass(frozen=True)
class DeploymentContract:
    schema_version: int
    campaign_id: str
    source_blend_delivery_sha256: str
    source_stage_c_delivery_sha256: str
    tabm_weight: float
    tree_prefixes: tuple[int, ...]
    minimum_weighted_gain: float
    maximum_fold_regression: float
    catboost_parameters: Mapping[str, object]
    snapshot_interval_seconds: int
    emergency_interval_seconds: int
    session_seconds: int
    new_job_guard_seconds: int


def load_contract(path: Path = DEFAULT_CONTRACT) -> DeploymentContract:
    return _validate_contract(_strict_json_object(path.read_bytes()))


def contract_sha256(path: Path = DEFAULT_CONTRACT) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_jobs(contract: DeploymentContract) -> tuple[DeploymentJob, ...]:
    return (
        DeploymentJob("align_2022_2023", "alignment", 2022, 2023, 42),
        DeploymentJob("align_2023_2024", "alignment", 2023, 2024, 42),
        DeploymentJob("full_2024", "full_fit", 2024, None, 42),
    )
```

The JSON must use exact keys, reject booleans as integers, require lowercase
64-character SHA-256 values, require strictly increasing prefixes ending in
400, and return `MappingProxyType` for parameters.

- [ ] **Step 4: Run GREEN verification**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_contracts.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/catboost_deployment/contracts.py
git diff --check
```

- [ ] **Step 5: Commit Task 1**

```bash
git add experiments/catboost_deployment/__init__.py \
  experiments/catboost_deployment/contract.json \
  experiments/catboost_deployment/contracts.py \
  tests/test_catboost_deployment_contracts.py
git commit -m "feat: seal CatBoost deployment contract"
```

## Task 2: Verify the three source archives by content

**Files:**
- Create: `experiments/catboost_deployment/inputs.py`
- Create: `tests/test_catboost_deployment_inputs.py`

- [ ] **Step 1: Write failing content-classification tests**

Create small ZIP fixtures and require classification independent of filenames:

```python
def test_classifies_three_required_sources_by_content(tmp_path: Path) -> None:
    paths = make_renamed_source_fixtures(tmp_path)
    kinds = {classify_source(path) for path in paths}
    assert kinds == {"training_input", "stage_c_delivery", "blend_delivery"}


def test_rejects_test_or_submission_archive(tmp_path: Path) -> None:
    source = zip_members(tmp_path / "x.zip", {"test.csv": b"x"})
    with pytest.raises(DeploymentInputError, match="unknown source"):
        classify_source(source)
```

Cover duplicate/traversal/symlink members, high compression ratio, oversized
manifest, duplicate kinds, missing source, wrong outer SHA, altered blend
decision, unverified Stage C binding, and a resume whose identity differs.

- [ ] **Step 2: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_inputs.py -q
```

Expected: import fails because `inputs.py` does not exist.

- [ ] **Step 3: Implement the trust boundary**

Public API:

```python
@dataclass(frozen=True)
class VerifiedDeploymentInputs:
    training: VerifiedTrainingInput
    stage_c: VerifiedStageC
    source_blend_path: Path
    source_blend_sha256: str
    source_blend_manifest_sha256: str
    source_decision_sha256: str
    source_selected_tabm_weight: float


def classify_source(path: Path) -> str:
    names, manifest = _safe_member_inventory(path)
    return _kind_from_exact_members(names, manifest)


def classify_and_verify_sources(
    paths: Sequence[Path],
    *,
    run_root: Path,
    contract: DeploymentContract,
    expected_code_sha256: str,
) -> tuple[VerifiedDeploymentInputs, Path | None]:
    grouped = _classify_unique_sources(paths)
    training = _verify_training(grouped, run_root, contract)
    stage_c = _verify_stage_c(grouped, run_root, contract)
    verified = _verify_promoted_blend(grouped, run_root, contract, training, stage_c)
    resume = _verify_optional_resume(grouped, verified, expected_code_sha256)
    return verified, resume
```

Reuse `verify_and_extract_training_input`, `verify_and_extract_stage_c` and
`catboost_tabm_blend.colab.verify_delivery`. Recompute the expected old blend
bindings from the verified training and Stage C sources. Extract only the old
review decision needed for identity; do not trust its filename. Require
`selected_tabm_weight == 0.7`, `reason == fixed_blend_passed`, both completed
folds and the exact outer SHA from the new contract.

Accept exactly three ZIPs fresh or four with a matching new deployment resume.
Copy verified sources into the exclusive run root before returning.

- [ ] **Step 4: Run GREEN and legacy source regressions**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_inputs.py \
  tests/test_catboost_tabm_blend_inputs.py \
  tests/test_catboost_tabm_blend_colab.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/catboost_deployment/inputs.py
git diff --check
```

- [ ] **Step 5: Commit Task 2**

```bash
git add experiments/catboost_deployment/inputs.py \
  tests/test_catboost_deployment_inputs.py
git commit -m "feat: verify CatBoost deployment sources"
```

## Task 3: Decide one deployable tree prefix

**Files:**
- Create: `experiments/catboost_deployment/metrics.py`
- Create: `tests/test_catboost_deployment_metrics.py`

- [ ] **Step 1: Write failing pure-metric tests**

Use aligned seven-column Stage C frames and alignment frames with one probability
column per tree prefix:

```python
def test_selects_lowest_passing_fixed_prefix() -> None:
    decision = evaluate_prefixes(tabm_by_fold, catboost_by_fold, contract)
    assert decision.status == "deployment_aligned"
    assert decision.selected_tree_count == 128


def test_blocks_when_fixed_prefix_regresses_one_fold() -> None:
    decision = evaluate_prefixes(tabm_by_fold, regressing_catboost, contract)
    assert decision.status == "deployment_blocked"
    assert decision.selected_tree_count is None
```

Cover row ID order/uniqueness, target equality, finite `[0,1]` probability,
empty fold, exact weighted row counts, `0.00003` boundaries, all seven prefixes,
lowest-Brier selection, `1e-12` smaller-tree tie break and permutation
independence.

- [ ] **Step 2: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_metrics.py -q
```

- [ ] **Step 3: Implement the pure decision API**

```python
@dataclass(frozen=True)
class PrefixCandidate:
    tree_count: int
    fold_brier: Mapping[str, float]
    fold_regression: Mapping[str, float]
    weighted_brier: float
    weighted_gain: float
    passed: bool


@dataclass(frozen=True)
class DeploymentDecision:
    status: str
    selected_tree_count: int | None
    baseline_weighted_brier: float
    candidates: tuple[PrefixCandidate, ...]
    reason: str


def evaluate_prefixes(
    tabm_by_fold: Mapping[str, pd.DataFrame],
    catboost_by_fold: Mapping[str, pd.DataFrame],
    contract: DeploymentContract,
) -> DeploymentDecision:
    aligned = _validate_and_align_folds(tabm_by_fold, catboost_by_fold, contract)
    candidates = tuple(_score_prefix(aligned, prefix, contract) for prefix in contract.tree_prefixes)
    passing = tuple(candidate for candidate in candidates if candidate.passed)
    return _select_decision(candidates, passing, contract)


def decision_payload(decision: DeploymentDecision) -> dict[str, object]:
    return _finite_canonical_mapping(asdict(decision))
```

Each CatBoost frame has exact columns
`row_id,target,p_4,p_32,p_64,p_128,p_192,p_296,p_400` plus the four existing
diagnostic segment columns. Clip each CatBoost prefix before blending. Use
float64 and canonical JSON with `allow_nan=False`.

- [ ] **Step 4: Run GREEN**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_metrics.py \
  tests/test_catboost_tabm_blend_metrics.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/catboost_deployment/metrics.py
git diff --check
```

- [ ] **Step 5: Commit Task 3**

```bash
git add experiments/catboost_deployment/metrics.py \
  tests/test_catboost_deployment_metrics.py
git commit -m "feat: decide deployable CatBoost tree count"
```

## Task 4: Serialize fitted state and run both worker kinds

**Files:**
- Create: `experiments/catboost_deployment/state.py`
- Create: `experiments/catboost_deployment/training.py`
- Create: `experiments/catboost_deployment/requirements-colab.txt`
- Create: `tests/test_catboost_deployment_state.py`
- Create: `tests/test_catboost_deployment_training.py`

- [ ] **Step 1: Write RED tests for exact state round trips**

```python
def test_preprocessing_state_round_trip_is_exact(training_frame: pd.DataFrame) -> None:
    fitted, expected = fit_catboost_features(training_frame, components=("hand_matchup",))
    payload = serialize_feature_state(fitted)
    restored = deserialize_feature_state(payload)
    actual = transform_catboost_features(training_frame, restored)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
```

Reject extra/missing keys, reordered source/output columns, non-finite values,
unknown profile/component, category-value drift and target-bearing input fields
outside the training-only prior.

- [ ] **Step 2: Write RED tests for alignment and full workers**

Use a fake CatBoost model that records fit arguments and returns deterministic
prefix predictions:

```python
def test_alignment_worker_fits_once_and_predicts_all_prefixes(tmp_path: Path) -> None:
    result = run_alignment_job(**alignment_kwargs, model_factory=fake_factory)
    assert result.status == "completed"
    assert fake_model.fit_calls == 1
    assert fake_model.ntree_ends == [4, 32, 64, 128, 192, 296, 400]


def test_full_worker_uses_selected_tree_count_and_freezes_state(tmp_path: Path) -> None:
    result = run_full_fit_job(
        **full_fit_kwargs, selected_tree_count=128, model_factory=fake_factory
    )
    assert result.status == "completed"
    assert fake_model.parameters["iterations"] == 128
    assert result.model_path.name == "model.cbm"
    assert result.preprocessing_path.name == "preprocessing_state.json"
```

Cover fold leakage, empty folds, duplicate row IDs, nonbinary target, no early
stopping, `use_best_model=False`, model-save failure, changed snapshot identity,
deadline before fit and non-finite/wrong-length predictions.

- [ ] **Step 3: Confirm both modules are RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_state.py \
  tests/test_catboost_deployment_training.py -q
```

- [ ] **Step 4: Implement state APIs and worker results**

```python
def serialize_feature_state(state: CatBoostFeatureState) -> bytes:
    return canonical_json(_state_to_exact_payload(state))


def deserialize_feature_state(payload: bytes) -> CatBoostFeatureState:
    return _exact_payload_to_state(strict_json_object(payload))


@dataclass(frozen=True)
class DeploymentJobResult:
    job_id: str
    kind: str
    status: str
    train_rows: int
    valid_rows: int | None
    predictions_path: Path | None
    model_path: Path | None
    preprocessing_path: Path | None
    snapshot_path: Path | None
    elapsed_seconds: float
    failure: str | None


def run_alignment_job(*, job: DeploymentJob, contract: DeploymentContract,
    data_dir: Path, output_dir: Path, contract_sha256: str,
    input_manifest_sha256: str, code_sha256: str, absolute_deadline: float,
    model_factory: Callable[..., object] | None = None) -> DeploymentJobResult:
    return _run_job(_alignment_spec(job, contract), locals())


def run_full_fit_job(*, job: DeploymentJob, contract: DeploymentContract,
    selected_tree_count: int, alignment_decision_sha256: str,
    data_dir: Path, output_dir: Path, contract_sha256: str,
    input_manifest_sha256: str, code_sha256: str, absolute_deadline: float,
    model_factory: Callable[..., object] | None = None) -> DeploymentJobResult:
    return _run_job(
        _full_fit_spec(job, contract, selected_tree_count, alignment_decision_sha256),
        locals(),
    )
```

Both fits use `save_snapshot=True`, 300-second snapshot interval and an isolated
`train_dir`. Alignment fits exactly 400 trees without eval-set early stopping.
Full fit uses the selected prefix and all 1,475,092 official training rows.
Write job/result/metrics files atomically and stream `CATBOOST_DEPLOY_PROGRESS`.

`requirements-colab.txt` contains exactly:

```text
catboost==1.2.10
```

- [ ] **Step 5: Run GREEN and preprocessing regressions**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_state.py \
  tests/test_catboost_deployment_training.py \
  tests/test_catboost_preprocessing.py \
  tests/test_preprocessing_profiles.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/catboost_deployment/state.py \
  experiments/catboost_deployment/training.py
git diff --check
```

- [ ] **Step 6: Commit Task 4**

```bash
git add experiments/catboost_deployment/state.py \
  experiments/catboost_deployment/training.py \
  experiments/catboost_deployment/requirements-colab.txt \
  tests/test_catboost_deployment_state.py \
  tests/test_catboost_deployment_training.py
git commit -m "feat: train fixed CatBoost deployment models"
```

## Task 5: Seal review, resume and final full-training evidence

**Files:**
- Create: `experiments/catboost_deployment/artifacts.py`
- Create: `tests/test_catboost_deployment_artifacts.py`

- [ ] **Step 1: Write RED tests for exact bundle states**

Require these states:

```text
fresh/active alignment       -> resume only
one alignment complete       -> resume only
two alignment complete       -> alignment review + resume
deployment_blocked           -> blocked review + resume, no frozen model
full fit active              -> alignment review + resume with snapshot
full fit complete            -> final review + resume + outer delivery
```

The final outer delivery has exactly:

```text
catboost_deployment.log
catboost_deployment_review.zip
catboost_deployment_resume.zip
delivery_manifest.json
```

The final review contains exact contract, alignment decision and fold metrics,
two alignment prediction files, frozen `model.cbm`, strict
`preprocessing_state.json`, `inference_manifest.json`, state, log and manifest.

- [ ] **Step 2: Write RED safety and integrity cases**

Reject traversal, duplicate/symlink members, ratio/size bombs, incorrect
source/input/code bindings, prediction/Brier mismatch, selected-prefix mismatch,
model/preprocessing SHA mismatch, a full fit after `deployment_blocked`, and a
complete stage missing one fold. Stream a 64 MiB fixture while monkeypatching
`Path.read_bytes` and `ZipFile.read` for large members.

- [ ] **Step 3: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_artifacts.py -q
```

- [ ] **Step 4: Implement deterministic bundle APIs**

```python
@dataclass(frozen=True)
class DeploymentBundlePaths:
    review: Path | None
    resume: Path
    delivery: Path | None
    review_sha256: str | None
    resume_sha256: str
    delivery_sha256: str | None


def write_deployment_bundles(*, output_dir: Path, bindings: Mapping[str, str],
    contract_path: Path, stage_state_path: Path, campaign_log_path: Path,
    job_directories: Mapping[str, Path], decision_path: Path | None,
    check_deadline: Callable[[], None] | None = None) -> DeploymentBundlePaths:
    evidence = _validated_stage_evidence(
        bindings, stage_state_path, job_directories, decision_path
    )
    return _publish_verified_bundles(
        output_dir, contract_path, campaign_log_path, evidence, check_deadline
    )


def verify_deployment_resume(path: Path, *,
    expected_bindings: Mapping[str, str]) -> VerifiedDeploymentResume:
    return _verify_resume_archive(path, expected_bindings)


def verify_deployment_review(path: Path, *,
    expected_bindings: Mapping[str, str]) -> None:
    _verify_review_archive(path, expected_bindings)


def verify_full_training_delivery(path: Path, *,
    expected_bindings: Mapping[str, str]) -> None:
    _verify_outer_delivery_and_nested_archives(path, expected_bindings)
```

Build and verify temporary archives before atomic `os.replace`; preserve the
prior valid destination on any failure. Manifest keys include
`review_only=true` and `submission_package=false`.

- [ ] **Step 5: Run GREEN**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_artifacts.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/catboost_deployment/artifacts.py
git diff --check
```

- [ ] **Step 6: Commit Task 5**

```bash
git add experiments/catboost_deployment/artifacts.py \
  tests/test_catboost_deployment_artifacts.py
git commit -m "feat: seal CatBoost deployment evidence"
```

## Task 6: Orchestrate the alignment gate and conditional full fit

**Files:**
- Create: `experiments/catboost_deployment/runner.py`
- Create: `tests/test_catboost_deployment_runner.py`

- [ ] **Step 1: Write RED state-machine tests**

With injected workers, verify exact scheduling:

```python
def test_full_fit_runs_only_after_alignment_passes() -> None:
    result = run_deployment_campaign(
        **campaign_kwargs,
        alignment_runtime=fake_passing_runtime,
        full_runtime=fake_passing_runtime,
    )
    assert result.status == "full_training_complete"
    assert fake_passing_runtime.calls == ["align_2022_2023", "align_2023_2024", "full_2024"]


def test_blocked_alignment_never_calls_full_fit() -> None:
    result = run_deployment_campaign(
        **campaign_kwargs,
        alignment_runtime=fake_blocked_runtime,
        full_runtime=fake_blocked_runtime,
    )
    assert result.status == "deployment_blocked"
    assert all(call != "full_2024" for call in fake_blocked_runtime.calls)
    assert result.bundles.delivery is None
```

Cover fresh run, resume after each job, active snapshot, completed reuse,
900-second new-job guard, corrupt resume before GPU work, decision hash drift,
callback failure preserving latest resume and exact state transitions.

- [ ] **Step 2: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_runner.py -q
```

- [ ] **Step 3: Implement the sequential runner**

```python
@dataclass(frozen=True)
class DeploymentRun:
    status: str
    bundles: DeploymentBundlePaths
    decision: DeploymentDecision | None
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None


def code_sha256() -> str:
    return code_identity_sha256(repository_root())


def run_deployment_campaign(*, verified: VerifiedDeploymentInputs,
    output_dir: Path, resume_bundle: Path | None, absolute_deadline: float,
    on_verified_resume: Callable[[Path], None] | None = None,
    alignment_runtime: Callable[..., DeploymentJobResult] = run_alignment_job,
    full_runtime: Callable[..., DeploymentJobResult] = run_full_fit_job,
) -> DeploymentRun:
    state = _restore_or_initialize(verified, output_dir, resume_bundle)
    state = _run_missing_alignment_jobs(state, alignment_runtime, absolute_deadline)
    decision = _evaluate_and_publish_alignment(state, verified)
    if decision.status == "deployment_blocked":
        return _blocked_run(state, decision)
    state = _run_full_fit(state, decision, full_runtime, absolute_deadline)
    return _finalize_run(state, decision, on_verified_resume)
```

After two alignment jobs, load Stage C TabM predictions and the seven-prefix
CatBoost frames, call `evaluate_prefixes`, persist canonical decision, and
publish a verified resume. Return immediately on `deployment_blocked`. Pass the
selected tree count and decision SHA into the full worker. Never run jobs in
parallel.

- [ ] **Step 4: Run GREEN and combined regressions**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_runner.py \
  tests/test_catboost_deployment_artifacts.py \
  tests/test_catboost_deployment_metrics.py \
  tests/test_catboost_tabm_blend_runner.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/catboost_deployment/runner.py
git diff --check
```

- [ ] **Step 5: Commit Task 6**

```bash
git add experiments/catboost_deployment/runner.py \
  tests/test_catboost_deployment_runner.py
git commit -m "feat: gate CatBoost full training on deployment OOF"
```

## Task 7: Build the direct-upload Colab supervisor and one cell

**Files:**
- Create: `experiments/catboost_deployment/colab.py`
- Create: `experiments/catboost_deployment/runtime_inventory.py`
- Create: `experiments/catboost_deployment/COLAB_CATBOOST_DEPLOYMENT_CELL.py`
- Create: `tools/build_catboost_deployment_colab_cell.py`
- Create: `tests/test_catboost_deployment_colab.py`
- Create: `tests/test_catboost_deployment_colab_cell.py`

- [ ] **Step 1: Write RED tests for upload and recovery behavior**

Require exactly three fresh or four resume uploads by content. Reject old
Stage P/Stage C resumes, standalone old review, test/sample/submission archives,
duplicate kinds and mismatched code/contract/source hashes.

With fake time and subprocess workers, require native snapshot interval 300,
browser resume interval 1200 only on changed snapshot, immediate fold/full-fit
completion download, prior verified resume preservation and deadline termination
using terminate then kill with bounded waits.

- [ ] **Step 2: Write RED tests for the deterministic cell**

```python
def test_checked_in_cell_matches_renderer() -> None:
    rendered = render_colab_cell(ROOT)
    assert CELL.read_bytes() == rendered
    assert len(rendered) < 1_000_000
    text = rendered.decode()
    assert text.index("SESSION_DEADLINE = time.time() + 10800") < text.index(
        "google.colab.files.upload()"
    )
    assert "drive.mount" not in text
    assert "github.com" not in text
```

Extract the embedded archive under an isolated `PYTHONPATH`, import every
runtime module and run a fake three-job seam. Assert exact deterministic archive
members and that requirements participate in `code_sha256()`.

- [ ] **Step 3: Confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_colab.py \
  tests/test_catboost_deployment_colab_cell.py -q
```

- [ ] **Step 4: Implement supervisor and explicit inventory**

Public supervisor API:

```python
def run_supervised_deployment(*, verified: VerifiedDeploymentInputs,
    output_dir: Path, snapshot_dir: Path, resume_bundle: Path | None,
    wall_deadline: float, on_verified_resume: Callable[[Path], None],
    log_path: Path) -> DeploymentRun:
    runtime = _SubprocessRuntime(wall_deadline, snapshot_dir, on_verified_resume, log_path)
    return run_deployment_campaign(
        verified=verified,
        output_dir=output_dir,
        resume_bundle=resume_bundle,
        absolute_deadline=wall_deadline,
        on_verified_resume=runtime.publish,
        alignment_runtime=runtime.run_alignment,
        full_runtime=runtime.run_full_fit,
    )
```

Use a subprocess for each CatBoost worker, relay logs, poll snapshots, terminate
at the absolute deadline, recursively verify every resume before callback and
download final delivery once. The embedded inventory is an explicit tuple of
only required preprocessing, old artifact verification and new deployment
modules plus `requirements-colab.txt`. Exclude submission, inference runtime,
notebooks, Stage P/D cells, official data and generated result files.

Required markers:

```text
DEPLOY_CODE_READY
DEPLOY_INPUTS_VERIFIED
DEPLOY_DEPENDENCIES_READY
DEPLOY_GPU_READY
DEPLOY_RESUME_READY
DEPLOY_JOB_START
CATBOOST_DEPLOY_PROGRESS
DEPLOY_PREFIX_RESULT
DEPLOY_ALIGNMENT_DECISION
DEPLOY_FULL_FIT_RESULT
DEPLOY_BUNDLE_SUCCESS
DEPLOY_DELIVERY_READY
DEPLOY_DOWNLOAD_REQUESTED
DEPLOY_ERROR stage=<stage> type=<type> message=<message>
```

- [ ] **Step 5: Build and run GREEN**

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/build_catboost_deployment_colab_cell.py
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_colab.py \
  tests/test_catboost_deployment_colab_cell.py -q
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/catboost_deployment/colab.py \
  experiments/catboost_deployment/runtime_inventory.py \
  experiments/catboost_deployment/COLAB_CATBOOST_DEPLOYMENT_CELL.py \
  tools/build_catboost_deployment_colab_cell.py
git diff --check
```

Expected builder marker:

```text
CATBOOST_DEPLOYMENT_CELL_READY path=<absolute> sha256=<sha256> size_bytes=<1000000
```

- [ ] **Step 6: Commit Task 7**

```bash
git add experiments/catboost_deployment/colab.py \
  experiments/catboost_deployment/runtime_inventory.py \
  experiments/catboost_deployment/COLAB_CATBOOST_DEPLOYMENT_CELL.py \
  tools/build_catboost_deployment_colab_cell.py \
  tests/test_catboost_deployment_colab.py \
  tests/test_catboost_deployment_colab_cell.py
git commit -m "feat: add Colab CatBoost deployment campaign"
```

## Task 8: Document and verify the code-only handoff

**Files:**
- Create: `docs/CATBOOST_DEPLOYMENT_TRAINING_COLAB.md`
- Modify: `README.md`

- [ ] **Step 1: Write the Korean runbook**

Document the three fresh uploads, optional newest deployment resume, one T4 or
better GPU, fixed prefix grid, unchanged 70:30 weight, expected 10–30 minute
runtime with 3-hour cap, 20-minute emergency downloads, exact markers and the
file to return. State plainly that no `test.csv`, full inference or submission
package is used or created.

- [ ] **Step 2: Run focused verification**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_catboost_deployment_contracts.py \
  tests/test_catboost_deployment_inputs.py \
  tests/test_catboost_deployment_metrics.py \
  tests/test_catboost_deployment_state.py \
  tests/test_catboost_deployment_training.py \
  tests/test_catboost_deployment_artifacts.py \
  tests/test_catboost_deployment_runner.py \
  tests/test_catboost_deployment_colab.py \
  tests/test_catboost_deployment_colab_cell.py \
  tests/test_catboost_tabm_blend_colab.py \
  tests/test_catboost_preprocessing.py \
  tests/test_preprocessing_profiles.py -q
```

- [ ] **Step 3: Run repository-wide static and fixture verification**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest -q
artifacts/tabm_submission_python311/bin/python -m compileall -q \
  experiments/catboost_deployment \
  experiments/catboost_tabm_blend \
  experiments/catboost_preprocessing \
  experiments/independent_dl \
  tools/build_catboost_deployment_colab_cell.py
rg -n -i "test\\.csv|sample_submission|submit\\.zip|drive\\.mount|github" \
  experiments/catboost_deployment \
  tools/build_catboost_deployment_colab_cell.py \
  docs/CATBOOST_DEPLOYMENT_TRAINING_COLAB.md
git diff --check
git ls-files '*.zip' '*.cbm' '*.cbsnapshot'
git status --short
```

Expected: all tests and compile checks pass; grep hits only explicit runbook
rejections; no tracked model/archive; existing unrelated main changes remain
unstaged.

- [ ] **Step 4: Commit documentation only**

```bash
git add README.md docs/CATBOOST_DEPLOYMENT_TRAINING_COLAB.md
git commit -m "docs: explain CatBoost deployment training run"
```

## Task 9: User runs the official-data campaign

This task is a user handoff and must not be executed by Codex.

- [ ] **Step 1: Upload exact fresh inputs in one Colab dialog**

```text
catboost_tabm_blend_input.zip
tabm_colab_stage_C_delivery.zip
catboost_tabm_blend_delivery.zip
```

Use one T4 or better GPU and the checked-in
`COLAB_CATBOOST_DEPLOYMENT_CELL.py`. On continuation, add only the newest
`catboost_deployment_resume.zip` as the fourth upload.

- [ ] **Step 2: Return the result by state**

Success marker:

```text
DEPLOY_DELIVERY_READY path=/content/catboost_deployment/runs/<run_id>/catboost_full_training_delivery.zip sha256=<sha256>
```

Return that delivery unchanged. On `deployment_blocked`, return
`catboost_deployment_alignment_review.zip` and the matching resume; no outer
full-training delivery exists in that state. On failure, return the complete
`DEPLOY_ERROR stage=<stage> type=<type> message=<message>` line and newest
deployment resume.

- [ ] **Step 3: Stop before dual-model inference work**

Codex recursively verifies the returned delivery and reports one of:

```text
full_training_ready
deployment_blocked
deployment_incomplete
```

Only `full_training_ready` with current hashes permits writing the separate
TabM+CatBoost inference-validation implementation plan. No submission package
is created by this plan.
