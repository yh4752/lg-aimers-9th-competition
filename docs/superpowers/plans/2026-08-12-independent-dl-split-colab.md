# Independent DL Split Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run exactly one pending candidate from a chosen independent-DL family per Colab cell while preserving the complete P1–P4, expansion, fold, and seed campaign.

**Architecture:** Add optional scheduling constraints to the existing campaign runner instead of creating another campaign or manifest. Expose them through the existing CLI, then replace the single Colab handoff cell with shared setup/status cells and one thin execution cell per family. Candidate generation, priority, capacity, checkpoints, expansion, confirmation, and artifacts remain unchanged.

**Tech Stack:** Python 3.11, argparse, pytest, Jupyter JSON, Google Colab, Google Drive.

---

### Task 1: Limit one campaign call without shrinking the campaign

**Files:**
- Modify: `experiments/independent_dl/campaign.py`
- Test: `tests/test_independent_dl_campaign.py`

- [ ] **Step 1: Write the failing scheduling tests**

Allow the existing `_candidate` helper to accept a `family`, then add these behaviors:

```python
def test_family_limit_runs_only_one_matching_candidate(tmp_path: Path) -> None:
    campaign = _campaign((
        _candidate("tabm_first", width=512, family="tabm"),
        _candidate("resnet_first", width=512, family="mlp_resnet"),
        _candidate("tabm_second", width=256, family="tabm"),
    ))
    runtime = _FakeRuntime()

    summary = run_campaign(
        campaign, tmp_path, runtime, family="tabm", max_candidates=1
    )

    assert runtime.started == ["tabm_first"]
    assert {"tabm_second", "resnet_first"} <= set(summary.pending)


def test_family_limit_resumes_with_next_candidate(tmp_path: Path) -> None:
    campaign = _campaign((
        _candidate("first", width=512, family="tabm"),
        _candidate("second", width=256, family="tabm"),
    ))
    run_campaign(campaign, tmp_path, _FakeRuntime(), family="tabm", max_candidates=1)
    runtime = _FakeRuntime()

    run_campaign(campaign, tmp_path, runtime, family="tabm", max_candidates=1)

    assert runtime.started == ["second"]
```

Add the failure and validation boundaries:

```python
def test_failed_candidate_consumes_one_attempt(tmp_path: Path) -> None:
    campaign = _campaign((
        _candidate("failed", width=512, family="tabm"),
        _candidate("next", width=256, family="tabm"),
    ))
    first = _FakeRuntime(candidate_error_on="failed")
    run_campaign(campaign, tmp_path, first, family="tabm", max_candidates=1)
    assert first.started == ["failed"]

    second = _FakeRuntime()
    run_campaign(campaign, tmp_path, second, family="tabm", max_candidates=1)
    assert second.started == ["next"]


@pytest.mark.parametrize("limit", [0, -1, True])
def test_nonpositive_or_boolean_limit_is_rejected(tmp_path: Path, limit) -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        run_campaign(_campaign((_candidate("one", width=512),)), tmp_path,
                     _FakeRuntime(), max_candidates=limit)


def test_unknown_family_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="family"):
        run_campaign(_campaign((_candidate("one", width=512),)), tmp_path,
                     _FakeRuntime(), family="unknown", max_candidates=1)
```

Add `test_limited_runs_still_register_boundary_expansion_and_confirmations` using a tiny campaign with expansion `(768,)`, OOF fold `((2022, 2023),)`, and confirmation seed `(2026,)`. Repeatedly call `run_campaign(campaign, tmp_path, runtime, family="mlp_resnet", max_candidates=1)` until no pending entries remain, then assert the registered IDs include `expand_width_768`, `confirm_f2022_2023`, and `s2026`. Use a maximum of 20 fixture calls and fail if the campaign does not terminate.

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `python3 -m pytest tests/test_independent_dl_campaign.py -q`

Expected: new tests fail because `run_campaign` does not accept `family` or `max_candidates`.

- [ ] **Step 3: Implement the minimal constraints**

Extend the public signature without changing default behavior:

```python
def run_campaign(
    campaign: CampaignSpec,
    output_root: str | Path,
    runtime: CampaignRuntime,
    *,
    family: str | None = None,
    max_candidates: int | None = None,
) -> CampaignSummary:
```

Validate `family` against complete campaign families and require a positive non-boolean integer limit. Maintain an attempted count, skip other families, and leave through the existing summary-writing path when the count is reached. Count both completion and normal `CandidateExecutionError`; propagate unexpected exceptions. Do not filter `CampaignSpec.candidates`, rewrite the registry, change epochs, or disable expansion and confirmation registration.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `python3 -m pytest tests/test_independent_dl_campaign.py -q`

Expected: all campaign tests pass, including existing resume, hash, priority, and expansion tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add experiments/independent_dl/campaign.py tests/test_independent_dl_campaign.py
git commit -m "feat: run one independent dl family candidate"
```

### Task 2: Expose scheduling and family status through the CLI

**Files:**
- Modify: `experiments/independent_dl/run_campaign.py`
- Create: `tests/test_independent_dl_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Use a temporary campaign and patch only the full-data runtime boundary:

```python
def test_run_cli_forwards_family_and_single_candidate_limit(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        module,
        "run_campaign",
        lambda *args, **kwargs: captured.update(kwargs) or summary,
    )
    assert module.main([
        "run", "--config", str(config), "--data-dir", str(data),
        "--output-dir", str(output), "--family", "tabm",
        "--max-candidates", "1",
    ]) == 0
    assert captured == {"family": "tabm", "max_candidates": 1}
```

For status, write a manifest with ordered `completed`, P3 `pending`, P4 `pending`, boundary `pending`, and confirmation `pending` entries for one family. Capture stdout from `main(["status", "--output-dir", str(output)])`, parse JSON, and assert `next_candidate` is the P3 ID, `remaining_count == 4`, and all four pending IDs remain in the projection.

- [ ] **Step 2: Run the CLI tests and confirm RED**

Run: `python3 -m pytest tests/test_independent_dl_cli.py -q`

Expected: the parser rejects the new options and status lacks the family projection.

- [ ] **Step 3: Implement minimal CLI wiring**

Add only:

```python
run.add_argument(
    "--family",
    choices=("tabm", "mlp_resnet", "ft_transformer", "tabr", "tabicl_v2"),
)
run.add_argument("--max-candidates", type=int)
```

Pass the values directly to `run_campaign`. For `status`, print the existing manifest plus a deterministic read-only `family_status` projection with `completed`, `failed`, `pending`, `next_candidate`, and `remaining_count` for each family.

- [ ] **Step 4: Run CLI and campaign tests**

Run:

```bash
python3 -m pytest tests/test_independent_dl_cli.py tests/test_independent_dl_campaign.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add experiments/independent_dl/run_campaign.py tests/test_independent_dl_cli.py
git commit -m "feat: expose independent dl candidate scheduling"
```

### Task 3: Split the Colab notebook into reusable execution cells

**Files:**
- Modify: `notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb`
- Modify: `experiments/independent_dl/COLAB.md`
- Modify: `tests/test_independent_dl_notebook.py`
- Modify: `tests/test_independent_dl_handoff.py`

- [ ] **Step 1: Write failing split-flow assertions**

Replace the old two-cell assertion with this conceptual order:

```python
expected_headings = [
    "공통 준비", "현재 상태 확인", "TabM 후보 1개",
    "MLP/ResNet 후보 1개", "FT-Transformer 후보 1개",
    "TabR 후보 1개", "TabICLv2 후보 1개", "결과 요약",
]
```

For each execution cell require `run --family <family> --max-candidates 1`. Assert setup contains Drive, repository, packages, data, and GPU preparation but no campaign `run`; status/summary do not train; every family section explains purpose, setup, one-candidate behavior, runtime, checkpoint rerun, success output, traceback handoff, and no submission in Korean. Require P3/P4/expansion/confirmation visibility and visible `research_only` for TabICLv2.

- [ ] **Step 2: Run notebook tests and confirm RED**

Run:

```bash
python3 -m pytest tests/test_independent_dl_notebook.py tests/test_independent_dl_handoff.py -q
```

Expected: failure because the notebook is still one markdown plus one code cell.

- [ ] **Step 3: Rebuild only the requested Colab handoff**

Preserve secure Git authentication, pinned code, package pins, Drive validation, hardware reporting, and output root in a shared setup cell. Define:

```python
def run_one_family(family: str) -> None:
    run_checked([
        sys.executable, "-m", "experiments.independent_dl.run_campaign", "run",
        "--config", str(CONFIG), "--data-dir", str(DATA_DIR),
        "--output-dir", str(CAMPAIGN_OUTPUT_DIR),
        "--family", family, "--max-candidates", "1",
    ], cwd=REPO_DIR, env=child_env)
```

Add one status cell, five execution cells, and a final summary/status cell. Put detailed Korean markdown immediately before each execution cell. Keep `COLAB.md` synchronized, but do not add a notebook generator, another state system, or per-family notebooks.

- [ ] **Step 4: Validate JSON and focused tests**

Run:

```bash
python3 -m json.tool notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb >/dev/null
python3 -m pytest tests/test_independent_dl_notebook.py tests/test_independent_dl_handoff.py tests/test_independent_dl_cli.py tests/test_independent_dl_campaign.py -q
```

Expected: JSON validation and all focused tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb experiments/independent_dl/COLAB.md tests/test_independent_dl_notebook.py tests/test_independent_dl_handoff.py
git commit -m "docs: split independent dl colab execution"
```

### Task 4: Final lightweight verification

**Files:**
- Verify only; modify only a scoped file if a test exposes a defect.

- [ ] **Step 1: Run the repository test suite**

Run: `python3 -m pytest -q`

Expected: all tests pass. Do not run official-data preprocessing or GPU training.

- [ ] **Step 2: Run static checks**

```bash
python3 -m compileall -q experiments/independent_dl
python3 -m json.tool experiments/independent_dl/configs/campaign_v1.json >/dev/null
python3 -m json.tool notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb >/dev/null
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Verify performance scope was not reduced**

Run a read-only probe containing:

```python
assert set(config["capacity_profiles"]) == {
    "tabm", "mlp_resnet", "ft_transformer", "tabr"
}
assert all(len(items) == 4 for items in config["capacity_profiles"].values())
assert config["oof_folds"] == [[2021, 2022], [2022, 2023], [2023, 2024]]
assert config["confirmation_seeds"] == [2026, 3407]
assert config["boundary_expansion"]
assert config["frontier_candidates"][0]["family"] == "tabicl_v2"
```

Expected: the original broad and deep campaign remains intact.

- [ ] **Step 4: Confirm scope and hand off without pushing**

```bash
git status --short --branch
git log --oneline --max-count=5
```

Expected: only planned commits, a clean worktree, and no push. Report the notebook path and explain that the user runs setup, status, then any one family cell.
