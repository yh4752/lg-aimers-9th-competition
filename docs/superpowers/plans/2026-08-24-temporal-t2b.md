# Temporal Portfolio T2-B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** T2-A 승격 피처를 두 과거 fold와 두 개의 조합에서 확인하는, 최대 10개 작업의 재시작 가능한 Kaggle T2-B 캠페인을 만든다.

**Architecture:** T1 review와 T2-A handoff를 로컬에서 검증해 최소 T2-B 입력 ZIP으로 줄인다. Kaggle 실행기는 공식 데이터와 이 ZIP만 받아 Phase F, C, H를 순차 실행하고, 완료 작업을 compact 형식으로 재사용하며 review/resume을 단일 handoff에 넣는다.

**Tech Stack:** Python 3.11+, pandas, NumPy, PyTorch, TabM 0.0.3, rtdl-num-embeddings 0.0.12, pytest, ZIP/TAR SHA-256 manifest

---

### Task 1: T2-B 입력 계약

**Files:**
- Create: `experiments/temporal_portfolio/t2b_input.py`
- Create: `tools/prepare_temporal_portfolio_t2b_input.py`
- Create: `tests/test_temporal_portfolio_t2b.py`

- [ ] **Step 1: Write the failing input tests**

```python
def test_t2b_input_contains_three_fixed_t1_folds_and_t2a_lineage(tmp_path):
    path = prepare_t2b_input(t1_review, t2a_handoff, tmp_path / "input.zip")
    verified = verify_t2b_input(path)
    assert verified.promoted == ("S1", "P3", "P2")
    assert verified.t1_decision_sha256 == T1_DECISION_SHA


def test_t2b_input_rejects_tampered_member(tmp_path):
    path = prepared_t2b_input(tmp_path)
    rewrite_zip_member(path, "t1_anchor_2022.csv", b"tampered")
    with pytest.raises(T2BInputError, match="member differs"):
        verify_t2b_input(path)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest -q tests/test_temporal_portfolio_t2b.py -k input`

Expected: import failure for the missing `t2b_input` module.

- [ ] **Step 3: Implement the minimal input API**

```python
@dataclass(frozen=True)
class VerifiedT2BInput:
    path: Path
    archive_sha256: str
    data_rows_sha256: str
    t1_decision_sha256: str
    t2a_review_sha256: str
    promoted: tuple[str, ...]


def prepare_t2b_input(t1_review: str | Path, t2a_handoff: str | Path, output: str | Path) -> Path:
    """Verify both parent artifacts and publish the nine-member T2-B input."""


def verify_t2b_input(path: str | Path) -> VerifiedT2BInput:
    """Fail closed on schema, hashes, lineage, rows, or promoted-candidate drift."""
```

Use the existing T1 review evaluator to reconstruct fixed T1 anchor/multi frames for 2022,
2023, and 2024. Accept only a completed T2-A result whose promoted list is exactly S1, P3,
P2 with the recorded variants. Write deterministic ZIP timestamps and sorted JSON.

- [ ] **Step 4: Run input tests and confirm GREEN**

Run: `python -m pytest -q tests/test_temporal_portfolio_t2b.py -k input`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/t2b_input.py tools/prepare_temporal_portfolio_t2b_input.py tests/test_temporal_portfolio_t2b.py
git commit -m "feat: bind temporal T2B inputs"
```

### Task 2: T2-B 작업과 판정

**Files:**
- Create: `experiments/temporal_portfolio/t2b.py`
- Modify: `tests/test_temporal_portfolio_t2b.py`

- [ ] **Step 1: Write failing schedule, cutoff, and decision tests**

```python
def test_t2b_schedule_is_six_single_two_latest_combo_then_two_history():
    assert len(build_phase_f_specs()) == 6
    assert [item.valid_year for item in build_phase_c_specs()] == [2024, 2024]
    assert len(build_phase_h_specs("S1+P3")) == 2


def test_t2b_materialization_uses_only_previous_season_for_recent_expert(
    tmp_path, monkeypatch, train, history, spec_va2022
):
    materialized = materialize_t2b_job(
        spec_va2022,
        data_rows_sha256="a" * 64,
        parent_sha256="b" * 64,
        train=train,
        history=history,
        cache_root=tmp_path / "cache",
    )
    assert tuple(materialized.training.identity.payload["train_seasons"]) == (2021,)


def test_t2b_combo_selection_requires_all_latest_fold_gates():
    assert select_combination(evidence(s1_p3_good=True)) == "S1+P3"
    assert select_combination(evidence()) is None
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest -q tests/test_temporal_portfolio_t2b.py -k 'schedule or materialization or combination'`

Expected: import failure for missing T2-B experiment functions.

- [ ] **Step 3: Implement sealed specs and materialization**

```python
@dataclass(frozen=True)
class T2BJobSpec:
    job_id: str
    phase: str
    bundles: tuple[str, ...]
    valid_year: int
    seed: int = 3407


PHASE_F_BUNDLES = ("S1", "P3", "P2")
PHASE_C_COMBINATIONS = (("S1", "P3"), ("S1", "P2"))
VALID_YEARS = (2022, 2023, 2024)
```

Implement `build_phase_f_specs()`, `build_phase_c_specs()`, `build_phase_h_specs(combo)`, and
`materialize_t2b_job(spec, *, data_rows_sha256, parent_sha256, train, history, cache_root)` with
those exact names and arguments. Each builder returns immutable `T2BJobSpec` tuples and rejects
any bundle, year, phase, seed, or canonical job ID outside this table.

Materialize `PortfolioFeatureSpec(("base", *spec.bundles), "dl_standard")`. The fitting
season is exactly `valid_year - 1`; feature context may include only seasons earlier than
`valid_year`. Reuse the T2-A TabM p2 training configuration verbatim.

- [ ] **Step 4: Implement deterministic evidence and decisions**

```python
COMBINATION_MIN_GAIN = 0.00003
COMBINATION_MAX_SEGMENT_REGRESSION = 0.00050
```

Implement `evaluate_t2b_fold`, `select_combination`, and `decide_t2b_candidates` with the names
and arguments used by the failing tests. `select_combination` sorts eligible entries by negative
gain and then the declared `PHASE_C_COMBINATIONS` order.

Use fixed 0.50 logit blending, strict row alignment, pitcher-block bootstrap, existing segment
gates, and `decide_candidate`. Tie-break `S1+P3` before `S1+P2`.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_temporal_portfolio_t2b.py`

Expected: all T2-B tests pass.

```bash
git add experiments/temporal_portfolio/t2b.py tests/test_temporal_portfolio_t2b.py
git commit -m "feat: define temporal T2B screening"
```

### Task 3: Restartable two-GPU runner and artifacts

**Files:**
- Create: `experiments/temporal_portfolio/t2b_runner.py`
- Create: `experiments/temporal_portfolio/t2b_artifacts.py`
- Modify: `tests/test_temporal_portfolio_t2b.py`

- [ ] **Step 1: Write failing orchestration and compact-resume tests**

```python
def test_t2b_runner_never_starts_more_than_ten_jobs(
    tmp_path, verified, t2b_input, fake_launcher
):
    result = run_t2b_stage(
        verified=verified,
        t2b_input=t2b_input,
        output_root=tmp_path / "output",
        deadline=20_000,
        launcher=fake_launcher,
    )
    assert len(fake_launcher.jobs) <= 10
    assert result.status == "completed"


def test_t2b_runner_reuses_compact_completed_jobs(tmp_path, completed_resume):
    output = tmp_path / "output"
    restore_t2b_resume_source(completed_resume, output)
    result = run_t2b_stage(
        verified=verified,
        t2b_input=t2b_input,
        output_root=output,
        deadline=20_000,
        launcher=launcher_that_must_not_start,
    )
    assert result.completed


def test_t2b_review_and_resume_are_verified_compact_bundles(tmp_path, completed_jobs):
    bundles = write_t2b_bundles(
        tmp_path / "bundles",
        jobs_root=completed_jobs,
        completed=("t2b__f__s1__va2022__s3407",),
        pending=(),
        failed=(),
        evidence={"promoted": []},
        parent_sha256="a" * 64,
    )
    assert restore_t2b_resume_source(bundles.resume, restored).is_dir()
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest -q tests/test_temporal_portfolio_t2b.py -k 'runner or resume or bundles'`

Expected: missing runner/artifact imports.

- [ ] **Step 3: Implement phase orchestration**

```python
@dataclass(frozen=True)
class T2BStageResult:
    status: str
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    evidence: Mapping[str, object]
    output_root: Path


MAXIMUM_JOB_COUNT = 10
```

Implement `run_t2b_stage` with keyword-only `verified`, `t2b_input`, `output_root`, `deadline`,
optional `launcher`, `frame_loader`, `clock`, and `sleeper` arguments. It returns the declared
`T2BStageResult` and writes `t2b_stage_result.json` before returning.

Run F, evaluate singles, run C, select one combination, optionally run H, then decide. Stop new
jobs 25 minutes before the deadline and reserve 10 minutes for artifacts. Verify semantic identity
before every reuse.

- [ ] **Step 4: Implement deterministic review/resume bundles**

```python
REVIEW_KIND = "temporal_t2b_review_v1"
RESUME_KIND = "temporal_t2b_resume_v1"
```

Implement `write_t2b_bundles` and `restore_t2b_resume_source` with the exact arguments used in
the test. The artifact kinds are the two constants above.

Reuse the T2-A compact format and safe ZIP extraction helpers. Preserve skipped phases and the
selected combination in both review and resume evidence.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_temporal_portfolio_t2b.py`

Expected: all T2-B tests pass.

```bash
git add experiments/temporal_portfolio/t2b_runner.py experiments/temporal_portfolio/t2b_artifacts.py tests/test_temporal_portfolio_t2b.py
git commit -m "feat: run restartable temporal T2B campaign"
```

### Task 4: Kaggle one-cell handoff

**Files:**
- Create: `experiments/temporal_portfolio/t2b_platform.py`
- Create: `tools/build_temporal_portfolio_t2b_kaggle_cell.py`
- Create: `experiments/temporal_portfolio/T2B_KAGGLE_CELL.py`
- Modify: `tests/test_temporal_portfolio_t2b.py`

- [ ] **Step 1: Write failing platform tests**

```python
def test_t2b_kaggle_cell_is_deterministic_small_and_single_download(tmp_path):
    first = build_t2b_kaggle_cell(tmp_path / "first.py")
    second = build_t2b_kaggle_cell(tmp_path / "second.py")
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 1_000_000
    assert first.read_text().count("files.download") == 1
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `python -m pytest -q tests/test_temporal_portfolio_t2b.py -k kaggle_cell`

Expected: missing platform builder import.

- [ ] **Step 3: Implement the cell template and builder**

The cell must discover exactly one official data root and one T2-B input, optionally restore one
T2-B resume/handoff, require two Tesla T4 devices, pin dependencies, tee logs, emit the declared
T2B log events, and request only the final normal or emergency handoff ZIP.

- [ ] **Step 4: Generate and verify the cell**

Run: `python tools/build_temporal_portfolio_t2b_kaggle_cell.py`

Expected: `T2B_KAGGLE_CELL_READY path=<generated-file> size_bytes=<number-below-1000000>`.

Run: `python -m py_compile experiments/temporal_portfolio/T2B_KAGGLE_CELL.py`

Expected: exit code 0.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/t2b_platform.py tools/build_temporal_portfolio_t2b_kaggle_cell.py experiments/temporal_portfolio/T2B_KAGGLE_CELL.py tests/test_temporal_portfolio_t2b.py
git commit -m "feat: launch temporal T2B on Kaggle"
```

### Task 5: Final synthetic verification and handoff instructions

**Files:**
- Modify only if verification reveals a T2-B defect.

- [ ] **Step 1: Run focused and full temporal tests**

Run: `python -m pytest -q tests/test_temporal_portfolio_t2b.py`

Run: `python -m pytest -q tests/test_temporal_portfolio_*.py`

Expected: all tests pass with no warning or collection error.

- [ ] **Step 2: Verify source quality and generated runtime**

Run: `git diff --check`

Run: `python -m py_compile experiments/temporal_portfolio/t2b_*.py tools/*t2b*.py`

Decode the generated cell runtime in a fresh temporary directory and import
`experiments.temporal_portfolio.t2b_runner` from that directory. Expected: import succeeds.

- [ ] **Step 3: Provide the user-run preparation command**

The command must consume:

- `/Users/yonghyun/Downloads/temporal_t1_review.zip`
- `/Users/yonghyun/Downloads/temporal_t2a_handoff (1).zip`

It must produce `artifacts/temporal_portfolio_t2b_handoff/temporal_portfolio_t2b_input.zip`.
Do not run this full OOF preparation inside Codex without separate approval.

- [ ] **Step 4: Report exact Kaggle inputs and success logs**

Kaggle inputs are the official data and `temporal_portfolio_t2b_input.zip`. The expected final log
is `T2B_HANDOFF_READY status=completed path=/kaggle/working/temporal_t2b_handoff.zip`.
