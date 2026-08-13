# T4 TabR Retrieval and Independent DL Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TabR P1 practical on a Tesla T4 without shrinking the model, and persist meaningful progress for every independent-DL candidate.

**Architecture:** Add one dependency-light JSONL progress reporter to the shared trainer so TabM, MLP/ResNet, FT-Transformer, and TabR receive identical lifecycle, throughput, validation, GPU-memory, checkpoint, and failure events. Replace TabR's all-pairs PyTorch search with a fold-train-only FAISS GPU IVF-Flat index, exact PyTorch reranking of an oversampled shortlist, and post-warmup frozen neighbor IDs. Keep TabR search policy in code so the user's existing campaign manifest remains compatible; bind cached search state to candidate/model/context identities.

**Tech Stack:** Python 3.11/3.12, NumPy 1.26.4, PyTorch 2.7+/CUDA 12, `faiss-gpu-cu12==1.14.1.post1`, pytest 8.4.1.

---

### Task 1: Shared append-only progress reporter

**Files:**
- Create: `experiments/independent_dl/progress.py`
- Create: `tests/test_independent_dl_progress.py`

- [ ] **Step 1: Write failing reporter tests**

Add tests that instantiate `ProgressReporter(candidate_id, family, fold, output_dir, clock, process_id)` and assert:

```python
reporter.emit(
    "TRAINING_PROGRESS",
    completed_rows=4096,
    total_rows=1_221_585,
    elapsed_seconds=8.0,
    gpu={"allocated_bytes": 10, "reserved_bytes": 20, "peak_bytes": 30},
)
```

writes the same strict finite JSON object to stdout and `progress.jsonl`, includes UTC timestamp/run ID/candidate/family/fold/PID, computes `rows_per_second` and ETA, appends instead of overwriting, and flushes immediately. Add fake-clock cases proving `should_emit` returns true at 60 seconds, emits `TRAINING_STALL_WARNING` after five minutes without increased completed rows, and raises `ProgressTimeoutError` after ten minutes without any substantive progress. Reject booleans/nonfinite numbers/negative counts and unknown event names.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run --no-project --python python3.11 \
  --with pytest==8.4.1 --with numpy==1.26.4 \
  python -m pytest tests/test_independent_dl_progress.py -q
```

Expected: collection fails because `experiments.independent_dl.progress` does not exist.

- [ ] **Step 3: Implement the minimal reporter**

Create:

```python
class ProgressTimeoutError(RuntimeError): ...

class ProgressReporter:
    def emit(self, event: str, **fields: object) -> Mapping[str, object]: ...
    def progress(self, event: str, *, completed_rows: int, total_rows: int,
                 started_at: float, gpu: Mapping[str, int], **fields: object) -> None: ...
    def should_emit(self, *, completed_rows: int, now: float | None = None) -> bool: ...
```

Use a fixed allowlist of the events in the approved design. Serialize with `allow_nan=False`, one compact object per line, `flush=True` to stdout, and `open("a", encoding="utf-8")` for the candidate-local file. A heartbeat counts as substantive only when `completed_rows` increases. Do not start a background thread that can claim progress while GPU work is stuck.

- [ ] **Step 4: Verify GREEN and commit**

Run the Task 1 command and `python3 -m compileall -q experiments/independent_dl/progress.py`; expect all tests to pass. Commit:

```bash
git add experiments/independent_dl/progress.py tests/test_independent_dl_progress.py
git commit -m "feat: persist independent dl progress events"
```

### Task 2: Instrument every shared PyTorch candidate

**Files:**
- Modify: `experiments/independent_dl/training.py`
- Modify: `experiments/independent_dl/models/common.py`
- Modify: `tests/test_independent_dl_training.py`

- [ ] **Step 1: Write failing shared-backend tests**

Add a recording reporter and fake CUDA memory object. Assert one attempt emits:

```text
CANDIDATE_RUNTIME_READY
TRAINING_PROGRESS
VALIDATION_PROGRESS
EPOCH_CHECKPOINTED
CANDIDATE_COMPLETED
```

for `family` values `tabm`, `mlp_resnet`, `ft_transformer`, and `tabr`. Assert training progress contains epoch/batch/loss/completed rows/rows per second/ETA and allocated/reserved/peak bytes. Assert `_predict` reports validation chunks rather than staying silent until all validation rows finish. Add a failure case proving the original exception is logged as `CANDIDATE_FAILED` and re-raised without rewriting its type/message.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run --no-project --python python3.11 \
  --with pytest==8.4.1 --with pandas==2.2.3 --with numpy==1.26.4 \
  --with scipy==1.16.3 --with torch==2.7.1 \
  python -m pytest tests/test_independent_dl_training.py \
    -k 'progress or validation or candidate_runtime' -q
```

Expected: failures because the backend does not accept or emit through a reporter.

- [ ] **Step 3: Add reporter hooks without changing model math**

In `fit_candidate`, construct one reporter from the request and candidate output directory and pass it into `TorchTrainingBackend.run_attempt`. In the backend:

- emit runtime-ready after device/model/optimizer construction;
- accumulate detached microbatch loss only for reporting;
- synchronize CUDA only at a due heartbeat, then report throughput and memory;
- pass the reporter to `_predict` and report each due validation chunk;
- add `best_checkpoint` and validation metrics to epoch events;
- emit time-budget/failure/completion events at existing safe boundaries.

Add optional adapter hook:

```python
def attach_progress_reporter(adapter: object, reporter: ProgressReporter) -> None:
    setter = getattr(adapter, "set_progress_reporter", None)
    if setter is not None:
        setter(reporter)
```

Do not change optimizer, scheduler, seed, batch order, model capacity, early stopping, or probability calculation.

- [ ] **Step 4: Verify all shared trainer regressions and commit**

Run:

```bash
uv run --no-project --python python3.11 \
  --with pytest==8.4.1 --with pandas==2.2.3 --with numpy==1.26.4 \
  --with scipy==1.16.3 --with torch==2.7.1 \
  python -m pytest tests/test_independent_dl_progress.py \
    tests/test_independent_dl_training.py -q
```

Expected: all pass. Commit:

```bash
git add experiments/independent_dl/training.py \
  experiments/independent_dl/models/common.py \
  tests/test_independent_dl_training.py
git commit -m "feat: log every independent dl candidate"
```

### Task 3: Fold-safe FAISS GPU retrieval with exact reranking

**Files:**
- Create: `experiments/independent_dl/models/tabr_search.py`
- Modify: `experiments/independent_dl/models/tabr.py`
- Modify: `experiments/independent_dl/requirements-colab.txt`
- Modify: `tests/test_independent_dl_tabr.py`

- [ ] **Step 1: Write failing search-contract tests**

Use a fake FAISS module for index lifecycle tests and small CPU tensors for exact reranking. Cover:

- only the supplied fold-training keys/absolute indices enter the index;
- GPU APIs (`StandardGpuResources`, IVF-Flat GPU transfer/search) are mandatory;
- CPU-only or missing FAISS raises `DLRuntimeDependencyError` before training;
- shortlist size is `min(train_rows, retrieval * 4 + self_neighbor_margin)`;
- exact reranking equals brute-force top-k on a small fixture;
- self index is excluded even when FAISS returns it first;
- validation query order, singleton, reverse, and rebatching produce identical neighbors;
- nonfinite keys, duplicate absolute IDs, wrong dimensions, and cross-model index reuse fail closed.

- [ ] **Step 2: Run TabR tests and verify RED**

Run:

```bash
uv run --no-project --python python3.11 \
  --with pytest==8.4.1 --with numpy==1.26.4 --with torch==2.7.1 \
  python -m pytest tests/test_independent_dl_tabr.py -q
```

Expected: failures for the absent `tabr_search` module and the existing all-pairs search path.

- [ ] **Step 3: Implement the index wrapper and exact reranking**

Define an immutable code policy, without changing `campaign_v1.json` or existing candidate config hashes:

```python
TABR_SEARCH_POLICY = {
    "version": "tabr_t4_ivf_flat_v1",
    "nlist": 4096,
    "nprobe": 64,
    "oversample_factor": 4,
    "training_sample_rows": 262144,
    "key_batch_rows": 8192,
    "freeze_after_epochs": 1,
}
```

`FoldTrainFaissIndex.build(keys, absolute_indices, reporter)` must train IVF-Flat on a deterministic seed-42 subset, add every fold-training vector with its absolute ID, move the index to GPU 0, set `nprobe`, and emit encoding/index events. `search_and_rerank(query_keys, query_absolute_indices)` retrieves the oversampled IDs and reranks only those cached keys with the existing squared-L2 PyTorch expression. Return exactly `retrieval` absolute IDs in deterministic distance/index order.

Pin `faiss-gpu-cu12==1.14.1.post1`. Before lazy import, set `_FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1` so the wheel uses the Colab CUDA 12 runtime already loaded by PyTorch. Probe GPU index creation and a two-vector search before encoding the full fold; fail instead of falling back to CPU or full PyTorch search.

- [ ] **Step 4: Replace `_TorchTabRRuntime._search_keys`**

Keep fold context features/labels on CPU. Cache context keys once per refresh, build the fold index, use exact-reranked absolute IDs for context feature/label gathering, and preserve the current differentiable recomputation of selected context keys. Attach the shared reporter so TabR emits context encoding, index-ready, and search progress events.

- [ ] **Step 5: Verify search equivalence and commit**

Run the Task 3 test command and confirm no top-k contract regressions. Commit:

```bash
git add experiments/independent_dl/models/tabr_search.py \
  experiments/independent_dl/models/tabr.py \
  experiments/independent_dl/requirements-colab.txt \
  tests/test_independent_dl_tabr.py
git commit -m "feat: accelerate TabR retrieval on T4"
```

### Task 4: Freeze contexts, bind resume identity, and preserve the existing campaign

**Files:**
- Modify: `experiments/independent_dl/models/tabr.py`
- Modify: `experiments/independent_dl/training.py`
- Modify: `tests/test_independent_dl_tabr.py`
- Modify: `tests/test_independent_dl_training.py`
- Modify: `tests/test_independent_dl_campaign.py`

- [ ] **Step 1: Write failing freeze/resume tests**

Cover:

- dynamic FAISS neighbors during epoch 0;
- deterministic full fold-train neighbor materialization after epoch 0;
- `TABR_CONTEXTS_FROZEN` contains shape and SHA-256 but no labels or row data;
- frozen neighbors are reused from epoch 1 onward without FAISS search per batch;
- checkpoint stores search policy version, context row-ID hash, model-config hash, fold cutoff, and frozen-neighbor hash;
- mismatched identity rejects reuse and rebuilds only search state while retaining the last complete model epoch;
- the user's old interrupted TabR entry (same candidate config, no valid checkpoint) returns from `running` to `pending`;
- the failed FT-Transformer candidate remains eligible only through the explicit `--retry-candidate` path added in commit `1c264ab`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run --no-project --python python3.11 \
  --with pytest==8.4.1 --with pandas==2.2.3 --with numpy==1.26.4 \
  --with scipy==1.16.3 --with torch==2.7.1 \
  python -m pytest tests/test_independent_dl_tabr.py \
    tests/test_independent_dl_training.py \
    tests/test_independent_dl_campaign.py -q
```

Expected: new freeze and search-identity assertions fail.

- [ ] **Step 3: Implement freeze lifecycle and checkpoint identity**

Add optional adapter lifecycle hooks called by the shared trainer:

```python
adapter.on_epoch_start(model, epoch, device)
adapter.on_epoch_end(model, epoch, device)
adapter.checkpoint_state()
adapter.restore_checkpoint_state(payload, model, device)
```

TabR materializes and atomically stores `frozen_neighbors.npy` after epoch 0. Store `search_state` in `checkpoint.pt` and `checkpoint_meta.json`. On resume, validate identity before loading frozen IDs. Never put validation/test rows, targets, or predictions into the index/frozen-neighbor artifact.

- [ ] **Step 4: Verify resume behavior and commit**

Run the Task 4 command. Expected: all pass and the existing manifest schema/config remain unchanged. Commit:

```bash
git add experiments/independent_dl/models/tabr.py \
  experiments/independent_dl/training.py \
  tests/test_independent_dl_tabr.py \
  tests/test_independent_dl_training.py \
  tests/test_independent_dl_campaign.py
git commit -m "feat: resume frozen TabR contexts safely"
```

### Task 5: Static acceptance and user-run handoff

**Files:**
- Create: `docs/runs/INDEPENDENT_DL_T4_MONITORING.md`
- Test: all repository tests

- [ ] **Step 1: Write the concise monitoring handoff**

Document the exact normal markers, their meanings, the Drive `progress.jsonl` path, the five-minute stall warning, the ten-minute failure condition, safe rerun behavior, and the exact FT retry candidate ID. State that no submission CSV/ZIP is created.

- [ ] **Step 2: Run complete lightweight verification**

Run:

```bash
git diff --check
python3 -m compileall -q experiments/independent_dl
uv run --no-project --python python3.11 \
  --with pytest==8.4.1 --with pandas==2.2.3 --with numpy==1.26.4 \
  --with scipy==1.16.3 --with torch==2.7.1 \
  python -m pytest -q
```

Expected: zero failures. Do not run full data, install FAISS locally on macOS, start GPU training, regenerate notebooks, create submission artifacts, or push.

- [ ] **Step 3: Commit documentation and report the user-run boundary**

```bash
git add docs/runs/INDEPENDENT_DL_T4_MONITORING.md
git commit -m "docs: explain independent dl progress monitoring"
```

Report the verified commit and ask separately for push. After push, provide separate complete Colab cells for repository/runtime preparation, the FAISS GPU probe, the explicit retry of `ft_transformer__raw_typed__p3__s42`, and TabR P1. Each model-run cell must be independently resumable so the user can wait for GPU quota between runs. The user performs those GPU runs and returns the log markers plus the lightweight result ZIP.
