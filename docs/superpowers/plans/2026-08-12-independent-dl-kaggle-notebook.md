# Independent DL Kaggle Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one clean Kaggle notebook that resumes the independent DL campaign from private Dataset inputs without changing training code or creating submissions.

**Architecture:** Keep the handoff self-contained in a two-cell notebook. The code cell discovers immutable Kaggle inputs, copies an optional checkpoint snapshot into `/kaggle/working`, checks out the sealed public repository commit, installs the existing requirements locally, and invokes the existing campaign entry point.

**Tech Stack:** Jupyter Notebook JSON (nbformat 4), Python standard library, PyTorch CUDA check, pytest static contract tests

---

### Task 1: Kaggle handoff notebook

**Files:**
- Create: `notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb`
- Create: `tests/test_independent_dl_kaggle_notebook.py`

- [ ] **Step 1: Write the failing notebook contract test**

Create a test that loads the notebook as JSON and requires exactly one Markdown
cell followed by one clean code cell. Require `/kaggle/input`,
`/kaggle/working`, the public repository URL, the sealed commit, CUDA checking,
the existing `experiments.independent_dl.run_campaign` entry point, input
ambiguity rejection, and non-destructive checkpoint copying. Reject
`google.colab`, `/content/drive`, submission CSV creation, and submission ZIP
creation.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest tests/test_independent_dl_kaggle_notebook.py -q
```

Expected: fail because `notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb` does
not exist.

- [ ] **Step 3: Add the two-cell Kaggle notebook**

The Markdown cell explains the required private Dataset contents, GPU setting,
expected runtime, resume behavior, `Save Version`, success text, and traceback
handoff. The code cell implements the exact runtime flow in the approved spec.
It discovers unique required files under `/kaggle/input`, derives their common
data directory, optionally discovers a unique `independent_dl_campaign_v1`
directory, uses `shutil.copytree(..., dirs_exist_ok=False)` only when the working
output does not already exist, and never deletes input or working artifacts.

- [ ] **Step 4: Run focused GREEN and static validation**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 \
  python -m pytest tests/test_independent_dl_kaggle_notebook.py \
  tests/test_independent_dl_handoff.py -q
python3 -m json.tool notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb >/dev/null
git diff --check
```

Expected: all tests pass; JSON and diff checks exit 0.

- [ ] **Step 5: Commit the notebook slice**

```bash
git add notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb \
  tests/test_independent_dl_kaggle_notebook.py
git commit -m "feat: add independent dl kaggle notebook"
```

Do not push. Do not execute full-data preprocessing, training, or Kaggle GPU
work locally.
