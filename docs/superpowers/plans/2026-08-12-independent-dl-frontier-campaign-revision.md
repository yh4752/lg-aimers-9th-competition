# Independent DL Frontier Campaign Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the existing full-scale independent-DL campaign for higher early information value, make TabR retrieval scalable, add a research-only TabICLv2 candidate, and hand the user a restartable Colab Pro notebook.

**Architecture:** Keep the existing feature cache, candidate artifacts, manifest, and training backends. Add ordered execution waves without changing the payload or hash of existing candidates, give TabR an epoch-scoped fold-local key cache, and route the single sklearn-style TabICLv2 family through a small dedicated adapter. The notebook remains a two-cell handoff generated from `COLAB.md`; full-data GPU execution stays user-owned.

**Tech Stack:** Python 3.12, NumPy, pandas, PyTorch, TabM/RTDL, TabICL 2.1.1, pytest, Colab Pro, Google Drive

---

### Task 1: Preserve Existing Candidates and Apply Explicit Execution Waves

**Files:**
- Modify: `experiments/independent_dl/configs/campaign_v1.json`
- Modify: `experiments/independent_dl/contracts.py`
- Modify: `experiments/independent_dl/campaign.py`
- Modify: `tests/test_independent_dl_contract.py`
- Modify: `tests/test_independent_dl_campaign.py`

- [ ] **Step 1: Write failing contract tests for priority order and hash compatibility**

Add tests that assert the first incomplete candidates follow this exact order while existing
`tabm__raw_typed__p1__s42` and `tabm__raw_typed__p2__s42` retain their original candidate
payload and config SHA:

```python
EXPECTED_PRIORITY = (
    "tabm__raw_typed__p1__s42",
    "tabm__raw_typed__p2__s42",
    "tabm__engineered__p2__s42",
    "tabm__entity_context__p2__s42",
    "tabm__trackman_augmented__p2__s42",
    "tabm__raw_typed__p3__s42",
    "tabm__engineered__p3__s42",
    "tabm__entity_context__p3__s42",
    "tabm__trackman_augmented__p3__s42",
    "mlp_resnet__raw_typed__p3__s42",
    "mlp_resnet__engineered__p3__s42",
    "mlp_resnet__entity_context__p3__s42",
    "mlp_resnet__trackman_augmented__p3__s42",
    "ft_transformer__raw_typed__p3__s42",
    "ft_transformer__engineered__p3__s42",
    "ft_transformer__entity_context__p3__s42",
    "ft_transformer__trackman_augmented__p3__s42",
    "tabicl_v2__raw_typed__frontier32__s42",
)

def test_campaign_uses_frontier_priority_before_remaining_grid() -> None:
    campaign = load_campaign(CONFIG)
    assert tuple(item.candidate_id for item in campaign.candidates[:18]) == EXPECTED_PRIORITY
    assert len({item.candidate_id for item in campaign.candidates}) == len(campaign.candidates)

def test_existing_candidate_payload_hash_is_not_changed_by_wave_metadata() -> None:
    campaign = load_campaign(CONFIG)
    candidate = campaign.candidates[0]
    assert candidate.candidate_id == "tabm__raw_typed__p1__s42"
    assert "wave" not in _candidate_payload(candidate)
    assert _config_sha256(candidate) == ORIGINAL_TABM_RAW_P1_SHA256
```

Add a resume fixture with the first two entries marked completed and assert the runtime begins
at `tabm__engineered__p2__s42` without rewriting those entries.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest tests/test_independent_dl_contract.py tests/test_independent_dl_campaign.py -q
```

Expected: failures because schema 1 has no execution waves or TabICLv2 family.

- [ ] **Step 3: Add schema 2 execution waves without adding wave fields to CandidateSpec**

Extend the JSON with `execution_waves` and one `frontier_candidates` entry. Parse the wave IDs,
expand the existing 64 candidates exactly as before, append the TabICLv2 candidate, then reorder
the tuple by the flattened wave IDs followed by all remaining grid candidates in their old order.
Reject duplicate, unknown, or omitted non-sentinel IDs. Use a final `"remaining_grid"` wave to
append every candidate not explicitly prioritized.

The TabICLv2 candidate is:

```json
{
  "candidate_id": "tabicl_v2__raw_typed__frontier32__s42",
  "family": "tabicl_v2",
  "feature_view": "raw_typed",
  "seed": 42,
  "epochs": 1,
  "stage": "research_only",
  "model": {
    "architecture": "tabicl_v2",
    "n_estimators": 32,
    "kv_cache": true,
    "offload_mode": "auto",
    "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt"
  },
  "training": {}
}
```

Do not add wave metadata to `_candidate_payload`; this preserves existing config hashes and Drive
resume compatibility.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same command. Expected: all focused tests pass and the fake resume starts at the third
priority candidate.

- [ ] **Step 5: Commit Task 1**

```bash
git add experiments/independent_dl/configs/campaign_v1.json \
  experiments/independent_dl/contracts.py experiments/independent_dl/campaign.py \
  tests/test_independent_dl_contract.py tests/test_independent_dl_campaign.py
git commit -m "feat: order independent dl frontier campaign"
```

### Task 2: Record the Hardware Actually Used

**Files:**
- Modify: `experiments/independent_dl/training.py`
- Modify: `experiments/independent_dl/campaign.py`
- Modify: `tests/test_independent_dl_training.py`
- Modify: `tests/test_independent_dl_campaign.py`

- [ ] **Step 1: Write failing pure tests for hardware metadata**

Use a fake torch CUDA surface and require an immutable result:

```python
def test_cuda_hardware_reports_every_visible_device_without_claiming_ddp() -> None:
    hardware = inspect_cuda_hardware(_FakeTorch(names=("NVIDIA A100", "NVIDIA A100")))
    assert hardware == {
        "device_count": 2,
        "devices": (
            {"index": 0, "name": "NVIDIA A100", "vram_bytes": 42_949_672_960},
            {"index": 1, "name": "NVIDIA A100", "vram_bytes": 42_949_672_960},
        ),
        "training_mode": "single_gpu",
        "training_device_indices": (0,),
    }
```

Require candidate metrics to contain the same `hardware` object returned by the training result.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python -m pytest tests/test_independent_dl_training.py tests/test_independent_dl_campaign.py -q
```

Expected: missing hardware helper/result fields.

- [ ] **Step 3: Implement the minimal hardware record**

Add `inspect_cuda_hardware(torch)` and include its result in `TrainResult`. Keep training on
`cuda:0`; do not claim multi-GPU use or add DDP in this task. Write the actual hardware object to
candidate `metrics.json`.

- [ ] **Step 4: Run focused tests and commit**

```bash
python -m pytest tests/test_independent_dl_training.py tests/test_independent_dl_campaign.py -q
git add experiments/independent_dl/training.py experiments/independent_dl/campaign.py \
  tests/test_independent_dl_training.py tests/test_independent_dl_campaign.py
git commit -m "feat: record independent dl gpu usage"
```

### Task 3: Replace Per-Batch TabR Full-Context Encoding with an Epoch Cache

**Files:**
- Modify: `experiments/independent_dl/models/common.py`
- Modify: `experiments/independent_dl/models/tabr.py`
- Modify: `experiments/independent_dl/training.py`
- Modify: `tests/test_independent_dl_tabr.py`
- Modify: `tests/test_independent_dl_training.py`

- [ ] **Step 1: Write failing cache and exactness tests**

Add a small deterministic model whose `encode` records calls. Require one full-context encoding
per refresh, no full-context encoding during two query calls, exact agreement with brute-force
top-k over the cached keys, and self-neighbor exclusion for training rows:

```python
runtime.refresh_keys(model, device="cpu")
first = runtime.search(model, query_num, query_cat, row_indices=np.array([0, 2]))
second = runtime.search(model, query_num, query_cat, row_indices=np.array([0, 2]))

assert model.full_context_encode_calls == 1
assert first.indices.tolist() == brute_force_indices.tolist()
assert second.indices.tolist() == first.indices.tolist()
assert all(row not in neighbors for row, neighbors in zip([0, 2], first.indices))
```

Add a backend hook test proving refresh occurs before every training epoch, before validation, and
before final prediction—not before every micro-batch.

- [ ] **Step 2: Run the TabR tests and verify RED**

```bash
python -m pytest tests/test_independent_dl_tabr.py tests/test_independent_dl_training.py -q
```

Expected: current runtime repeatedly encodes candidate chunks and has no refresh hook.

- [ ] **Step 3: Implement the fold-local CPU key cache**

Add `refresh_retrieval_cache(adapter, model, device)` as an optional adapter hook. In TabR,
encode context in fixed batches under `no_grad`, store contiguous CPU float32 keys, and search the
cached array in chunks transferred to the query device. Re-encode only the selected context rows
inside the differentiable value path. The cache must retain the context row IDs and reject use
with another context or model generation.

Call refresh at epoch start, again after the epoch updates before validation, and before final
prediction. Other adapters keep the no-op default.

- [ ] **Step 4: Run focused tests and commit**

```bash
python -m pytest tests/test_independent_dl_tabr.py tests/test_independent_dl_training.py -q
git add experiments/independent_dl/models/common.py experiments/independent_dl/models/tabr.py \
  experiments/independent_dl/training.py tests/test_independent_dl_tabr.py \
  tests/test_independent_dl_training.py
git commit -m "fix: cache tabr fold retrieval keys"
```

### Task 4: Add the Research-Only TabICLv2 Candidate

**Files:**
- Create: `experiments/independent_dl/models/tabicl_v2.py`
- Modify: `experiments/independent_dl/models/__init__.py`
- Modify: `experiments/independent_dl/campaign.py`
- Modify: `experiments/independent_dl/requirements-colab.txt`
- Create: `tests/test_independent_dl_tabicl_v2.py`
- Modify: `tests/test_independent_dl_campaign.py`
- Modify: `tests/test_independent_dl_handoff.py`

- [ ] **Step 1: Write failing adapter tests using an injected fake estimator**

The tests must not download weights or import TabICL. Require categorical columns to remain pandas
`category`, train and validation row order to remain unchanged, class-1 probabilities to be finite
and one-dimensional, and model metadata to record `research_only`:

```python
result = fit_predict_tabicl_v2(
    train,
    valid,
    config,
    output_dir,
    estimator_factory=_FakeTabICLClassifier,
)
assert result.predictions.shape == (len(valid.row_id),)
assert np.isfinite(result.predictions).all()
assert result.submission_eligibility == "research_only"
assert fake.fit_row_ids == train.row_id.tolist()
```

Also reject a probability matrix without exactly two classes or with nonfinite values.

- [ ] **Step 2: Run the new tests and verify RED**

```bash
python -m pytest tests/test_independent_dl_tabicl_v2.py \
  tests/test_independent_dl_campaign.py tests/test_independent_dl_handoff.py -q
```

Expected: missing module, dependency, and runtime route.

- [ ] **Step 3: Implement one direct sklearn-style adapter**

Create `TabICLv2Adapter.fit_predict` that lazily imports `TabICLClassifier`, constructs a pandas
frame from `x_num` plus categorical columns converted to `category`, passes the sealed
`n_estimators=32`, `kv_cache=True`, `offload_mode="auto"`, checkpoint, device, AMP, and seed, then
calls `fit` and `predict_proba`. Save only small estimator metadata and predictions; do not add a
generic foundation-model abstraction.

In `OfficialCampaignRuntime.run_candidate`, route only `family == "tabicl_v2"` through this
adapter and keep all trainable families on `fit_candidate`. Add `tabicl==2.1.1` to the Colab
requirements and keep the import lazy.

- [ ] **Step 4: Run focused tests and commit**

```bash
python -m pytest tests/test_independent_dl_tabicl_v2.py \
  tests/test_independent_dl_campaign.py tests/test_independent_dl_handoff.py -q
git add experiments/independent_dl/models/tabicl_v2.py \
  experiments/independent_dl/models/__init__.py experiments/independent_dl/campaign.py \
  experiments/independent_dl/requirements-colab.txt tests/test_independent_dl_tabicl_v2.py \
  tests/test_independent_dl_campaign.py tests/test_independent_dl_handoff.py
git commit -m "feat: add research-only tabicl v2 candidate"
```

### Task 5: Update the Restartable Colab Pro Handoff Notebook

**Files:**
- Modify: `experiments/independent_dl/COLAB.md`
- Modify: `notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb`
- Modify: `tests/test_independent_dl_handoff.py`
- Modify: `tests/test_independent_dl_notebook.py`

- [ ] **Step 1: Write failing handoff tests**

Require the one-cell code to print every visible GPU and VRAM, never assert `T4`, describe Colab
Pro as availability-dependent, print whether it is a new run or resume, show the next candidate,
and retain Drive checkpoint paths. Require the notebook code cell to remain byte-equal to the code
block in `COLAB.md`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python -m pytest tests/test_independent_dl_handoff.py tests/test_independent_dl_notebook.py -q
```

Expected: T4-only assertion and missing VRAM/next-candidate output.

- [ ] **Step 3: Replace the handoff cell and regenerate only the code cell**

Keep the existing two-cell notebook. The code cell must:

1. mount Drive and validate only the two user-editable paths;
2. fetch the exact final commit using the existing Colab secret;
3. install the sealed requirements in the child runtime;
4. print repository commit, Python, CUDA, package versions, GPU count/name/VRAM;
5. inspect the manifest and print new/resume plus the next pending candidate;
6. stream `run_campaign run` output;
7. print the manifest, results JSONL, summary path, completed count, and exact files to return.

Do not create a submission package, push, or automatically restart the runtime.

- [ ] **Step 4: Run focused tests and commit**

```bash
python -m pytest tests/test_independent_dl_handoff.py tests/test_independent_dl_notebook.py -q
git add experiments/independent_dl/COLAB.md notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb \
  tests/test_independent_dl_handoff.py tests/test_independent_dl_notebook.py
git commit -m "docs: hand off frontier dl colab campaign"
```

### Task 6: Focused Verification and Roadmap Alignment

**Files:**
- Modify: `docs/ROADMAP.md`
- Modify: `README.md`
- Test: existing independent-DL tests

- [ ] **Step 1: Update only current-state documentation**

Record that the trainable campaign order is revised, TabR awaits the scalable cache implementation
until this plan completes, TabICLv2 is research-only pending exact competition eligibility, and
Colab Pro + Drive is the primary long-run environment. Do not add completed performance claims or
ledger entries before user evidence exists.

- [ ] **Step 2: Run the complete small independent-DL suite**

```bash
python -m pytest tests/test_independent_dl_contract.py \
  tests/test_independent_dl_campaign.py tests/test_independent_dl_training.py \
  tests/test_independent_dl_tabr.py tests/test_independent_dl_tabicl_v2.py \
  tests/test_independent_dl_handoff.py tests/test_independent_dl_notebook.py -q
python -m compileall -q experiments/independent_dl
python -m json.tool experiments/independent_dl/configs/campaign_v1.json >/dev/null
git diff --check
```

Expected: all focused tests pass; compile, JSON, and diff checks exit 0. No official data or GPU
training is run.

- [ ] **Step 3: Confirm scope and commit**

```bash
git diff --name-only HEAD~5..HEAD
git status --short
git add docs/ROADMAP.md README.md
git commit -m "docs: align frontier dl campaign roadmap"
```

Verify that no submission package code, automatic push, unrelated experiment code, or large
artifact was added.
