# Kaggle Independent DL Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale all-in-one Kaggle campaign notebook with GitHub-free family cells that run one candidate and export one verified candidate OOF handoff ZIP.

**Architecture:** Add one dependency-light ZIP writer that consumes the existing campaign manifest and completed candidate artifacts. Add one deterministic notebook renderer because the GitHub-free notebook must embed the current runtime archive; the renderer produces shared setup/status cells, five family cells, and a handoff cell without changing the campaign configuration or submission boundary.

**Tech Stack:** Python standard library, argparse, Jupyter JSON, pytest, Kaggle Notebook, embedded tar.gz runtime.

---

### Task 1: Verified candidate handoff ZIP

**Files:**
- Create: `experiments/independent_dl/handoff.py`
- Modify: `experiments/independent_dl/run_campaign.py`
- Create: `tests/test_independent_dl_handoff_bundle.py`

- [ ] **Step 1: Write failing handoff tests**

Create a tiny campaign root containing one completed candidate, `metrics.json`, and a small `predictions.csv`. Test this public API:

```python
result = write_candidate_handoff(
    campaign_root=campaign_root,
    candidate_id="tabm__raw_typed__p3__s42",
    result_path=tmp_path / "candidate_handoff.zip",
    runtime_sha256="a" * 64,
    requirements_path=requirements,
)
```

Assert that the ZIP contains exactly:

```python
{
    "handoff_manifest.json",
    "campaign_entry.json",
    "metrics.json",
    "predictions.csv",
    "requirements-colab.txt",
}
```

Assert `handoff_manifest.json` contains the candidate ID, campaign ID, protocol, runtime digest, requirements digest, metrics/predictions digests, uncompressed sizes, ZIP size, and metrics hardware. Add separate tests proving the writer rejects missing/non-completed candidates, path escape, symlinks, altered artifact hashes, malformed/duplicate/non-finite manifest JSON, invalid runtime digest, output aliasing an input, and an existing output before opening artifact contents.

- [ ] **Step 2: Run the handoff tests and confirm RED**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest tests/test_independent_dl_handoff_bundle.py -q
```

Expected: collection fails because `experiments.independent_dl.handoff` does not exist.

- [ ] **Step 3: Implement the minimal ZIP writer**

Implement `HandoffError` and `write_candidate_handoff`. Strictly parse JSON with duplicate-key and non-finite-number rejection. Resolve all input paths under `campaign_root`, reject symlinks and non-regular files, hash before packaging, and require exact equality with the manifest SHA-256 values. Write to a temporary regular file in the destination directory, use ZIP deflate, record member sizes and hashes, then publish without overwriting an existing destination. Do not include checkpoints, caches, source data, test predictions, or submission files.

Add a CLI action:

```text
handoff --output-dir ROOT --candidate-id ID --result ZIP
        --runtime-sha256 SHA256 --requirements FILE
```

Print JSON with `status: handoff_ready`, candidate ID, absolute ZIP path, byte size, and SHA-256. Do not import pandas, NumPy, Torch, or model packages in the handoff path.

- [ ] **Step 4: Run focused GREEN**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest tests/test_independent_dl_handoff_bundle.py \
  tests/test_independent_dl_cli.py -q
python3 -S -m experiments.independent_dl.run_campaign --help >/dev/null
```

Expected: all tests pass and the dependency-free CLI help exits zero.

- [ ] **Step 5: Commit Task 1**

```bash
git add experiments/independent_dl/handoff.py \
  experiments/independent_dl/run_campaign.py \
  tests/test_independent_dl_handoff_bundle.py
git commit -m "feat: export independent dl candidate handoff"
```

### Task 2: Deterministic split Kaggle notebook

**Files:**
- Create: `tools/render_independent_dl_kaggle_notebook.py`
- Modify: `notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb`
- Modify: `tests/test_independent_dl_kaggle_notebook.py`

- [ ] **Step 1: Write failing notebook contract tests**

Replace the stale two-cell assertions. Require ordered markdown/code sections for common setup, status, TabM, MLP/ResNet, FT-Transformer, TabR, TabICLv2, handoff ZIP, and result summary. Require each family code cell to call:

```text
run_one_family("<family>")
```

and require the shared helper to invoke `run --family FAMILY --max-candidates 1`. Require the handoff cell to use an explicit `HANDOFF_CANDIDATE_ID` and the CLI `handoff` action. Assert the notebook has no GitHub URL, token, clone, Colab path, submission CSV, or submission ZIP. Assert all four official CSV names, unique common-parent discovery, ambiguous checkpoint rejection, checkpoint preservation, P3/P4/expansion/confirmation wording, and `research_only` TabICLv2 wording.

Decode `EMBEDDED_RUNTIME_B64`, verify its SHA-256, inspect the tar members, and require the archive to contain `handoff.py`, the new CLI, campaign config, models, features, and requirements without links or traversal paths.

- [ ] **Step 2: Run notebook tests and confirm RED**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest tests/test_independent_dl_kaggle_notebook.py -q
```

Expected: failures because the notebook is still a two-cell full-campaign runner and its embedded archive predates the handoff code.

- [ ] **Step 3: Implement the small renderer and regenerate once**

The renderer must:

- archive tracked `experiments/independent_dl` files from the current committed HEAD using deterministic gzip metadata;
- embed the archive bytes, digest, and source commit in the notebook;
- discover exactly one directory containing all four official CSVs;
- restore at most one attached `independent_dl_campaign_v1` directory only when `/kaggle/working` has no campaign directory;
- prepare packages and print GPU/VRAM without starting training;
- define read-only `show_campaign_status`, streaming `run_one_family`, and `write_handoff` helpers;
- create five model-family cells, each running one candidate;
- create a handoff cell where the user sets only `HANDOFF_CANDIDATE_ID` and receives `/kaggle/working/codex_handoffs/<candidate_id>_handoff.zip` plus bytes and SHA-256;
- never create a submission artifact.

Each family markdown section must explain purpose, input, one-candidate scope, approximate runtime, checkpoint resume, exact success text, traceback handoff, and non-submission behavior in Korean.

- [ ] **Step 4: Prove deterministic generation and focused GREEN**

Run the renderer twice and compare notebook SHA-256, then run:

```bash
python3 -m json.tool notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb >/dev/null
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest tests/test_independent_dl_kaggle_notebook.py \
  tests/test_independent_dl_handoff_bundle.py \
  tests/test_independent_dl_cli.py -q
```

Expected: identical notebook digests and all focused tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add tools/render_independent_dl_kaggle_notebook.py \
  notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb \
  tests/test_independent_dl_kaggle_notebook.py
git commit -m "docs: split kaggle dl candidate handoff"
```

### Task 3: Final lightweight verification

**Files:**
- Verify only; modify a scoped file only if a test exposes a defect.

- [ ] **Step 1: Run the complete small suite**

```bash
uv run --no-project --python python3.11 \
  --with pytest==8.4.1 --with pandas==2.2.3 --with numpy==1.26.4 \
  --with scipy==1.16.3 --with torch==2.7.1 \
  python -m pytest -q
```

Expected: all tests pass. Do not run official data or GPU training.

- [ ] **Step 2: Run static and scope checks**

```bash
python3 -m compileall -q experiments/independent_dl \
  tools/render_independent_dl_kaggle_notebook.py
python3 -m json.tool notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb >/dev/null
python3 -m json.tool experiments/independent_dl/configs/campaign_v1.json >/dev/null
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Confirm the campaign was not narrowed**

Load `campaign_v1.json` and assert all four families retain four P1–P4 profiles, four feature views, maximum epochs of 400, three OOF folds, confirmation seeds 2026 and 3407, non-empty boundary expansion, and the TabICLv2 frontier candidate.

- [ ] **Step 4: Hand off without pushing**

Report the notebook path, exact Kaggle execution order, expected handoff ZIP path, and files the user should attach. Confirm the worktree is clean and do not push.
