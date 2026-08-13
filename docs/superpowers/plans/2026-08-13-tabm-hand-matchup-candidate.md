# TabM Hand-Matchup Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one rules-compliant candidate using only the approved `dl_standard + hand_matchup` preprocessing and the validated TabM P2 model, stopping at a review bundle until all submission gates pass.

**Architecture:** Reuse the existing fold-fitted preprocessing and TabM adapter instead of creating a second feature implementation. Seal the selected preprocessing, model configuration, seed, and three-epoch full-data fit in one candidate contract. The Kaggle runner fits all mutable preprocessing and numerical bin edges from official training rows only, trains the model, performs row-independence canaries on test rows, and exports a review bundle; submission packaging remains fail-closed.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch, TabM 0.0.3, pytest, existing `competition_rules` and `submission` modules.

---

### Task 1: Seal the single candidate contract

**Files:**
- Create: `experiments/tabm_candidate/__init__.py`
- Create: `experiments/tabm_candidate/contract.py`
- Create: `experiments/tabm_candidate/experiment_contract.json`
- Test: `tests/test_tabm_candidate_contract.py`

- [ ] **Step 1: Write the failing contract test**

```python
from experiments.tabm_candidate.contract import CANDIDATE


def test_candidate_is_the_approved_single_tabm_configuration() -> None:
    assert CANDIDATE.candidate_id == "tabm_hand_matchup_v1"
    assert CANDIDATE.preprocessing_profile == "dl_standard"
    assert CANDIDATE.preprocessing_components == ("hand_matchup",)
    assert CANDIDATE.seed == 42
    assert CANDIDATE.epochs == 3
    assert dict(CANDIDATE.model) == {
        "architecture": "tabm",
        "blocks": 4,
        "dropout": 0.1,
        "k": 32,
        "num_embedding": "piecewise_linear",
        "width": 512,
    }
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run: `python -m pytest tests/test_tabm_candidate_contract.py -q`
Expected: FAIL because `experiments.tabm_candidate` does not exist.

- [ ] **Step 3: Implement the immutable candidate contract**

```python
@dataclass(frozen=True)
class CandidateContract:
    candidate_id: str
    preprocessing_profile: str
    preprocessing_components: tuple[str, ...]
    seed: int
    epochs: int
    model: Mapping[str, object]
    training: Mapping[str, object]


CANDIDATE = CandidateContract(
    candidate_id="tabm_hand_matchup_v1",
    preprocessing_profile="dl_standard",
    preprocessing_components=("hand_matchup",),
    seed=42,
    epochs=3,
    model=MappingProxyType({
        "architecture": "tabm",
        "blocks": 4,
        "dropout": 0.1,
        "k": 32,
        "num_embedding": "piecewise_linear",
        "width": 512,
    }),
    training=MappingProxyType({
        "amp": True,
        "effective_batch_size": 4096,
        "learning_rate": 0.0006,
        "micro_batch_size": 512,
        "weight_decay": 0.0001,
    }),
)
```

- [ ] **Step 4: Add a rules contract that allows only official train data, row-local `hand_matchup`, and frozen-state test inference**

The JSON must declare one candidate, `fit_scope: "official_train_only"`, `evaluation_scope: "row_independent"`, no external API, no retrieval corpus, and no pretrained model.

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest tests/test_tabm_candidate_contract.py tests/test_experiment_contract_gate.py -q`
Expected: PASS.

### Task 2: Add a serializable full-training feature state

**Files:**
- Modify: `experiments/independent_dl/features.py`
- Test: `tests/test_tabm_candidate_features.py`

- [ ] **Step 1: Write failing tests for train-only fit and row separability**

```python
def test_test_companions_do_not_change_a_rows_features(train_frame, test_frame):
    state, _ = fit_preprocessed_training_batch(train_frame, APPROVED_SPEC)
    alone = transform_preprocessed_batch(test_frame.iloc[[0]], state)
    together = transform_preprocessed_batch(test_frame, state)
    np.testing.assert_array_equal(alone.x_num[0], together.x_num[0])
    np.testing.assert_array_equal(alone.x_cat[0], together.x_cat[0])


def test_test_values_do_not_change_fitted_numeric_state(train_frame, test_frame):
    state, _ = fit_preprocessed_training_batch(train_frame, APPROVED_SPEC)
    before = preprocessing_state_digest(state)
    transform_preprocessed_batch(test_frame, state)
    assert preprocessing_state_digest(state) == before
```

- [ ] **Step 2: Verify both tests fail because the public helpers are absent**

Run: `python -m pytest tests/test_tabm_candidate_features.py -q`
Expected: FAIL on missing helper imports.

- [ ] **Step 3: Expose minimal wrappers around the existing preprocessing implementation**

```python
def fit_preprocessed_training_batch(
    train: pd.DataFrame, spec: PreprocessingSpec
) -> tuple[PreprocessedFeatureState, FeatureBatch]:
    preprocessing, prepared = fit_preprocessor(train, spec)
    category_maps = _fit_categories(prepared, preprocessing.categorical_columns)
    state = PreprocessedFeatureState(
        view="raw_typed",
        cutoff_year=int(train["season"].max()),
        numeric_columns=preprocessing.numeric_columns,
        categorical_columns=preprocessing.categorical_columns,
        category_maps=category_maps,
        preprocessing=preprocessing,
        trackman_result=None,
        trackman_lookup_sha256=None,
    )
    return state, _preprocessed_batch(train, prepared, state)


def transform_preprocessed_batch(
    frame: pd.DataFrame, state: PreprocessedFeatureState
) -> FeatureBatch:
    prepared = transform_preprocessor(frame, state.preprocessing)
    return _preprocessed_batch(frame, prepared, state)
```

- [ ] **Step 4: Add canonical JSON save/load and SHA-256 helpers for the preprocessing and category state**

State loading must reject missing keys, extra keys, non-finite numbers, schema drift, and a mismatched stored digest.

- [ ] **Step 5: Run the focused feature tests**

Run: `python -m pytest tests/test_tabm_candidate_features.py tests/test_preprocessing_profiles.py tests/test_preprocessing_feature_cache.py -q`
Expected: PASS.

### Task 3: Make TabM numerical bins reproducible at inference

**Files:**
- Modify: `experiments/independent_dl/models/common.py`
- Modify: `experiments/independent_dl/models/tabm.py`
- Test: `tests/test_tabm_candidate_model_state.py`

- [ ] **Step 1: Write a failing round-trip test for saved piecewise-linear bin edges**

```python
def test_saved_bin_edges_rebuild_the_same_embedding_definition(train_batch):
    metadata = metadata_from_train(train_batch)
    payload = numerical_bin_edges_payload(metadata)
    restored = metadata_from_payload(payload)
    for expected, actual in zip(metadata.numerical_bin_edges, restored.numerical_bin_edges):
        np.testing.assert_array_equal(expected, actual)
```

- [ ] **Step 2: Verify the test fails because metadata has no persisted bin edges**

Run: `python -m pytest tests/test_tabm_candidate_model_state.py -q`
Expected: FAIL on the missing `numerical_bin_edges` field.

- [ ] **Step 3: Store train-fitted bin edges in `ModelMetadata` and make TabM use them**

```python
@dataclass(frozen=True)
class ModelMetadata:
    n_num_features: int
    categorical_cardinalities: tuple[int, ...]
    train_x_num: np.ndarray
    numerical_bin_edges: tuple[np.ndarray, ...] | None = None


def metadata_from_train(batch: FeatureBatch) -> ModelMetadata:
    return ModelMetadata(
        n_num_features=batch.x_num.shape[1],
        categorical_cardinalities=_cardinalities(batch.x_cat),
        train_x_num=batch.x_num,
        numerical_bin_edges=quantile_bin_edges(batch.x_num),
    )
```

The inference loader supplies the saved edges and an empty `(0, n_num_features)`
training matrix. `make_numeric_embeddings` uses saved edges when present and falls
back to `train_x_num` only during training; it must never compute edges from test rows.

- [ ] **Step 4: Run model contract regressions**

Run: `python -m pytest tests/test_tabm_candidate_model_state.py tests/test_independent_dl_training.py -q`
Expected: PASS.

### Task 4: Train three fixed epochs and export a review artifact

**Files:**
- Create: `experiments/tabm_candidate/train.py`
- Create: `experiments/tabm_candidate/artifact.py`
- Test: `tests/test_tabm_candidate_training.py`

- [ ] **Step 1: Write failing tests for exact epoch count, artifact inventory, and hash validation**

```python
def test_training_plan_is_fixed_to_three_epochs() -> None:
    plan = build_training_plan(CANDIDATE)
    assert plan.epochs == 3
    assert plan.validation_data is None


def test_artifact_requires_all_hashed_members(tmp_path) -> None:
    with pytest.raises(CandidateArtifactError, match="missing artifact"):
        load_candidate_artifact(tmp_path)
```

- [ ] **Step 2: Verify the tests fail on the missing implementation**

Run: `python -m pytest tests/test_tabm_candidate_training.py -q`
Expected: FAIL on missing functions.

- [ ] **Step 3: Implement the fixed-epoch trainer**

The loop must reuse `TabMAdapter`, seed Python/NumPy/PyTorch, train on all official training rows for exactly three completed epochs, use constant AdamW settings from the sealed contract, log batch and epoch progress, and atomically save only the final model state.

- [ ] **Step 4: Write the review artifact**

Required files are `candidate.json`, `preprocessing_state.json`, `model_metadata.json`, `model.pt`, `artifact_manifest.json`, and `run.log`. The manifest records size and SHA-256 for every other member and rejects unexpected files.

- [ ] **Step 5: Run focused artifact tests**

Run: `python -m pytest tests/test_tabm_candidate_training.py -q`
Expected: PASS.

### Task 5: Add a row-independent inference adapter without enabling packaging

**Files:**
- Create: `experiments/tabm_candidate/inference.py`
- Modify: `submission/adapters.py`
- Test: `tests/test_tabm_candidate_inference.py`

- [ ] **Step 1: Write failing inference invariance tests**

```python
def test_predictions_are_invariant_to_companion_rows(adapter, test_frame):
    full = adapter.predict_batch(test_frame)
    reversed_predictions = adapter.predict_batch(test_frame.iloc[::-1])[::-1]
    singleton = adapter.predict_batch(test_frame.iloc[[0]])
    np.testing.assert_array_equal(full, reversed_predictions)
    np.testing.assert_array_equal(full[:1], singleton)


def test_inference_does_not_change_adapter_state(adapter, test_frame):
    before = adapter.state_digest()
    adapter.predict_batch(test_frame)
    assert adapter.state_digest() == before
```

- [ ] **Step 2: Verify the tests fail before adapter registration**

Run: `python -m pytest tests/test_tabm_candidate_inference.py -q`
Expected: FAIL because `tabm_hand_matchup_v1` is not registered.

- [ ] **Step 3: Implement and register the closed adapter**

The adapter loads only the hashed review artifact, calls `model.eval()`, transforms each batch with the frozen state, and returns TabM member-mean probabilities. It cannot accept training data, fit methods, evaluation-wide statistics, remote paths, or network URLs.

- [ ] **Step 4: Keep packaging fail-closed**

Do not add a script template to `submission.runtime.render_script` and do not create a submission ZIP. Registration permits review-time inference only; packaging remains blocked until the returned artifact passes all gates.

- [ ] **Step 5: Run inference and rules regressions**

Run: `python -m pytest tests/test_tabm_candidate_inference.py tests/test_submission_runtime.py tests/test_rules_code_gate.py tests/test_rules_repository_enforcement.py -q`
Expected: PASS.

### Task 6: Render one Kaggle cell that creates the review bundle

**Files:**
- Create: `tools/render_tabm_candidate_kaggle_cell.py`
- Create: `experiments/tabm_candidate/KAGGLE_TABM_CANDIDATE_CELL.py`
- Test: `tests/test_tabm_candidate_kaggle_cell.py`

- [ ] **Step 1: Write failing renderer and embedded-code integrity tests**

```python
def test_rendered_cell_is_self_contained_and_review_only(rendered_text):
    assert "TABM_CANDIDATE_SUCCESS" in rendered_text
    assert "tabm_hand_matchup_v1_review_bundle.zip" in rendered_text
    assert "submission.zip" not in rendered_text
    assert "REQUIRED_CODE_COMMIT" in rendered_text
    assert "EMBEDDED_RUNTIME_SHA256" in rendered_text
```

- [ ] **Step 2: Verify the tests fail because the renderer is absent**

Run: `python -m pytest tests/test_tabm_candidate_kaggle_cell.py -q`
Expected: FAIL on missing renderer.

- [ ] **Step 3: Implement the deterministic renderer**

The generated cell locates exactly one co-located `train.csv`, `test.csv`, and `sample_submission.csv`; verifies embedded code hashes; runs the rules contract; trains the sealed candidate; executes full-batch/reversed-batch/singleton canaries; and writes only `tabm_hand_matchup_v1_review_bundle.zip`.

- [ ] **Step 4: Define user-visible logs**

Required milestones are `DATA_FOUND`, `RULES_GATE_PASSED`, `PREPROCESSING_FIT_COMPLETE`, `TRAINING_PROGRESS`, `EPOCH_COMPLETE`, `ROW_INDEPENDENCE_PASSED`, and `TABM_CANDIDATE_SUCCESS`. Errors use `TABM_CANDIDATE_ERROR stage=<stage> type=<type> message=<message>`.

- [ ] **Step 5: Verify deterministic rendering and repository syntax**

Run: `python tools/render_tabm_candidate_kaggle_cell.py && git diff --check && python -m compileall -q competition_rules experiments submission tools`
Expected: the second render is byte-identical, no whitespace errors, and no syntax errors.

### Task 7: Final review before asking the user to run Kaggle

**Files:**
- Modify: `docs/EXPERIMENT_CONTRACT.md`
- Modify: `README.md`

- [ ] **Step 1: Run the complete non-GPU suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Run repository and rules gates directly**

Run: `python -m competition_rules.code_gate --project-root .` and the candidate experiment-contract validation command documented by `competition_rules`.
Expected: both report a passed verdict.

- [ ] **Step 3: Review the generated cell for prohibited test-wide operations**

Search the candidate inference path for `groupby`, `rolling`, `expanding`, `rank`, `value_counts`, `fit`, `partial_fit`, remote URLs, and test-time state writes. Every occurrence must either be absent or provably confined to official training data.

- [ ] **Step 4: Document the user handoff**

The handoff specifies the required Kaggle input, approximate runtime, rerun behavior, success/error log text, and the single review ZIP to return. It explicitly states that the ZIP is not yet a DACON submission package.

- [ ] **Step 5: Stop at the evidence gate**

After the user returns the review bundle, verify every artifact hash, the rules evidence, row-independence results, runtime, and candidate identity. Only a separately approved packaging task may create the DACON submission package.
