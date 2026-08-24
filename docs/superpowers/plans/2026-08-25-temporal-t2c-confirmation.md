# Temporal T2-C Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** T2-B에서 사후 정의한 S1 `game_type=F` fallback 후보를 두 새 seed와 세 temporal fold에서 재현성 확인하는, 최대 6개 작업의 재시작 가능한 Kaggle T2-C 캠페인을 만든다.

**Architecture:** 로컬 입력 생성기가 검증된 T2-B input과 handoff에서 T1 기준 예측, seed 3407 S1 예측, T2-B 계보만 추출해 최소 T2-C 입력을 만든다. Kaggle 실행기는 seed 42와 2026의 6개 recent S1 작업을 학습하고, 행 단위 안전 게이트와 3-seed logit ensemble을 평가한 뒤 review/resume을 단일 handoff에 담는다. 이 계획은 T2-C 검증까지만 구현하며 full-data 학습, test 추론, 제출 패키징은 만들지 않는다.

**Tech Stack:** Python 3.11+, pandas, NumPy, PyTorch, TabM 0.0.3, rtdl-num-embeddings 0.0.12, pytest, deterministic ZIP/TAR SHA-256 manifests

---

### Task 1: T2-C 입력 계약

**Files:**
- Create: `experiments/temporal_portfolio/t2c_input.py`
- Create: `tools/prepare_temporal_portfolio_t2c_input.py`
- Create: `tests/test_temporal_portfolio_t2c.py`

- [ ] **Step 1: Write failing lineage and tamper tests**

```python
def test_t2c_input_keeps_only_s1_references_and_t2b_lineage(tmp_path):
    path = prepare_t2c_input(t2b_input, t2b_handoff, tmp_path / "input.zip")
    verified = verify_t2c_input(path)
    assert verified.candidate_id == "s1_game_type_f_fallback_v1"
    assert verified.t2b_handoff_sha256 == T2B_HANDOFF_SHA256
    with ZipFile(path) as archive:
        assert set(archive.namelist()) == EXPECTED_T2C_MEMBERS


def test_t2c_input_rejects_tampered_seed_3407_prediction(tmp_path):
    path = prepared_t2c_input(tmp_path)
    rewrite_zip_member(path, "s1_recent_s3407_2023.csv", b"tampered")
    with pytest.raises(T2CInputError, match="member differs"):
        verify_t2c_input(path)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py -k input
```

Expected: collection fails because `experiments.temporal_portfolio.t2c_input` does not exist.

- [ ] **Step 3: Implement the sealed input API**

```python
@dataclass(frozen=True)
class VerifiedT2CInput:
    path: Path
    archive_sha256: str
    data_rows_sha256: str
    t2b_input_sha256: str
    t2b_handoff_sha256: str
    t2b_review_sha256: str
    decision_sha256: str
    candidate_id: str


def prepare_t2c_input(
    t2b_input: str | Path,
    t2b_handoff: str | Path,
    output: str | Path,
) -> Path:
    """Verify T2-B parents and publish the minimal deterministic T2-C input."""


def verify_t2c_input(path: str | Path) -> VerifiedT2CInput:
    """Reject schema, hash, lineage, row alignment, or candidate drift."""


def load_t2c_references(
    verified: VerifiedT2CInput,
) -> tuple[
    Mapping[int, pd.DataFrame],
    Mapping[int, pd.DataFrame],
    Mapping[int, pd.DataFrame],
    Mapping[str, object],
]:
    """Return anchor, fixed multi, seed-3407 S1, and sealed decision."""
```

The archive contains `manifest.json`, `decision.json`, three T1 anchors, three fixed T1 multi
predictions, and seed-3407 S1 predictions for 2022, 2023, and 2024. Extract 2022/2023 from the
T2-B review and 2024 from the verified T2-B input. Verify exact non-probability row alignment for
every year. The decision records `posthoc_candidate=true`, exact F fallback, seeds
`[3407,42,2026]`, and the parent hashes from the approved design.

```python
EXPECTED_T2C_MEMBERS = {
    "manifest.json",
    "decision.json",
    *(f"t1_anchor_{year}.csv" for year in (2022, 2023, 2024)),
    *(f"t1_multi_{year}.csv" for year in (2022, 2023, 2024)),
    *(f"s1_recent_s3407_{year}.csv" for year in (2022, 2023, 2024)),
}
```

- [ ] **Step 4: Implement the thin preparation CLI**

```python
def main() -> int:
    args = parser.parse_args()
    output = prepare_t2c_input(args.t2b_input, args.t2b_handoff, args.output)
    verified = verify_t2c_input(output)
    print(
        f"T2C_INPUT_READY path={output} sha256={verified.archive_sha256} "
        f"candidate={verified.candidate_id}",
        flush=True,
    )
    return 0
```

The CLI arguments are exactly `--t2b-input`, `--t2b-handoff`, and `--output`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py -k input
python -m py_compile experiments/temporal_portfolio/t2c_input.py tools/prepare_temporal_portfolio_t2c_input.py
```

Expected: selected tests pass and compilation exits zero.

```bash
git add experiments/temporal_portfolio/t2c_input.py tools/prepare_temporal_portfolio_t2c_input.py tests/test_temporal_portfolio_t2c.py
git commit -m "feat: bind temporal T2C inputs"
```

### Task 2: Fixed safety gate and 3-seed ensemble

**Files:**
- Create: `experiments/temporal_portfolio/t2c.py`
- Modify: `tests/test_temporal_portfolio_t2c.py`

- [ ] **Step 1: Write failing schedule, cutoff, and gate tests**

```python
def test_t2c_schedule_is_exactly_two_new_seeds_by_three_folds():
    specs = build_t2c_specs()
    assert [(item.seed, item.valid_year) for item in specs] == [
        (42, 2022), (2026, 2022),
        (42, 2023), (2026, 2023),
        (42, 2024), (2026, 2024),
    ]


def test_t2c_materialization_uses_previous_season_and_cutoff_prefix(
    tmp_path, monkeypatch
):
    job = materialize_t2c_job(
        T2CJobSpec("t2c__s1__va2022__s42", 42, 2022),
        data_rows_sha256="a" * 64,
        parent_sha256="b" * 64,
        train=training_rows(),
        history=pd.DataFrame(),
        cache_root=tmp_path / "cache",
    )
    assert tuple(job.training.identity.payload["train_seasons"]) == (2021,)
    assert tuple(job.training.identity.payload["features"]) == ("base", "S1")


def test_safety_gate_uses_anchor_only_for_exact_game_type_f():
    oof = build_gated_s1_oof(anchor, multi, (recent,), valid_year=2024)
    assert oof.loc[oof["game_type"].eq("F"), "candidate"].equals(
        oof.loc[oof["game_type"].eq("F"), "baseline"]
    )
    assert oof.loc[oof["game_type"].ne("F"), "candidate"].ne(
        oof.loc[oof["game_type"].ne("F"), "baseline"]
    ).any()


def test_recent_seed_ensemble_averages_logits_before_multi_blend():
    actual = build_gated_s1_oof(anchor, multi, (recent_a, recent_b), valid_year=2024)
    expected_recent = expit((logit(recent_a_prob) + logit(recent_b_prob)) / 2)
    expected = blend_logit(expected_recent, multi_prob, Decimal("0.50"))
    assert_allclose(actual.loc[non_f, "candidate"], expected[non_f])
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py -k 'schedule or cutoff or safety_gate or seed_ensemble'
```

Expected: import failure for the missing T2-C experiment module.

- [ ] **Step 3: Implement immutable specs and materialization**

```python
@dataclass(frozen=True)
class T2CJobSpec:
    job_id: str
    seed: int
    valid_year: int


NEW_SEEDS = (42, 2026)
ALL_SEEDS = (3407, 42, 2026)
VALID_YEARS = (2022, 2023, 2024)
MAXIMUM_JOB_COUNT = 6


def build_t2c_specs() -> tuple[T2CJobSpec, ...]:
    return tuple(
        T2CJobSpec(f"t2c__s1__va{year}__s{seed}", seed, year)
        for year in VALID_YEARS
        for seed in NEW_SEEDS
    )
```

Implement `materialize_t2c_job` with the same TabM p2 model and training settings as T2-B:
piecewise-linear embeddings, BCE, AdamW, plateau scheduler, learning rate `0.0006`, weight decay
`0.0001`, effective batch `4096`, micro batch `512`, AMP, 12 epochs, minimum 3 epochs, patience
3. Training rows are exactly `valid_year - 1`; feature context contains only years less than
`valid_year`; features are exactly `("base", "S1")`.

- [ ] **Step 4: Implement row-aligned gated OOF**

```python
def build_gated_s1_oof(
    anchor: pd.DataFrame,
    fixed_multi: pd.DataFrame,
    recent_predictions: tuple[pd.DataFrame, ...],
    *,
    valid_year: int,
) -> pd.DataFrame:
    """Align rows, average recent logits, blend fixed multi, then fall back on F."""


def evaluate_gated_s1_oof(
    oof: pd.DataFrame,
    *,
    bootstrap_repeats: int = 1_000,
) -> Mapping[str, float | int | str]:
    """Return Brier, paired bootstrap, and preregistered segment diagnostics."""
```

Use strict typed row IDs and `align_oof`. Refuse duplicate rows, different targets, different
stable segments, missing probability columns, non-finite values, or validation-year drift. Clip
probabilities using the same epsilon as `blend_logit` before logits. Set `candidate=baseline` only
where the aligned `game_type` value is the exact string `F`.

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py -k 'schedule or cutoff or safety_gate or seed_ensemble'
git add experiments/temporal_portfolio/t2c.py tests/test_temporal_portfolio_t2c.py
git commit -m "feat: define temporal T2C safety candidate"
```

### Task 3: Fail-closed T2-C decision

**Files:**
- Modify: `experiments/temporal_portfolio/t2c.py`
- Modify: `tests/test_temporal_portfolio_t2c.py`

- [ ] **Step 1: Write failing decision tests**

```python
def test_t2c_promotes_only_complete_reproducible_candidate():
    decision = decide_t2c(seed_evidence=passing_seeds(), ensemble=passing_ensemble())
    assert decision.status == "promoted"


def test_t2c_rejects_when_one_new_seed_has_nonpositive_weighted_gain():
    evidence = passing_seeds()
    evidence[42]["weighted_gain"] = 0.0
    assert decide_t2c(seed_evidence=evidence, ensemble=passing_ensemble()).status == "rejected"


def test_t2c_marks_incomplete_jobs_budget_inconclusive():
    evidence = passing_seeds()
    evidence[2026][2023] = {"status": "pending"}
    assert decide_t2c(seed_evidence=evidence, ensemble={}).status == "budget_inconclusive"
```

- [ ] **Step 2: Run decision tests and confirm RED**

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py -k decision
```

Expected: failure because `decide_t2c` and `T2CDecision` are missing.

- [ ] **Step 3: Implement the exact decision API**

```python
@dataclass(frozen=True)
class T2CDecision:
    candidate_id: str
    status: str
    reason: str
    new_seed_weighted_gains: Mapping[int, float]
    ensemble_weighted_gain: float | None
    ensemble_latest_gain: float | None
    ensemble_bootstrap_lower: float | None
    ensemble_max_segment_regression: float | None


def decide_t2c(
    *,
    seed_evidence: Mapping[int, Mapping[int | str, Mapping[str, object] | float]],
    ensemble: Mapping[str, object],
) -> T2CDecision:
    """Return promoted, rejected, or budget_inconclusive from sealed evidence."""
```

Calculate row-weighted seed gains from the three exact years. Promotion requires both new seed
weighted gains `> 0`, ensemble weighted gain `>= 0.00005`, latest gain `>= 0.00003`, combined
bootstrap lower `> 0`, maximum across combined and per-fold eligible segment regressions
`<= 0.00050`, and every ensemble fold gain `>= -0.00003`. Reject non-finite, extra, duplicated,
or contradictory evidence rather than coercing it.

- [ ] **Step 4: Run all core tests and commit**

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py -k 'decision or safety_gate or seed_ensemble'
git add experiments/temporal_portfolio/t2c.py tests/test_temporal_portfolio_t2c.py
git commit -m "feat: decide temporal T2C confirmation"
```

### Task 4: Compact artifacts and restartable two-GPU runner

**Files:**
- Create: `experiments/temporal_portfolio/t2c_artifacts.py`
- Create: `experiments/temporal_portfolio/t2c_runner.py`
- Modify: `tests/test_temporal_portfolio_t2c.py`

- [ ] **Step 1: Write failing runner and artifact tests**

```python
def test_t2c_runner_starts_exactly_six_authorized_jobs(tmp_path, fake_launcher):
    result = run_t2c_stage(
        verified=verified,
        t2c_input=t2c_input,
        output_root=tmp_path / "output",
        deadline=20_000,
        launcher=fake_launcher,
        clock=lambda: 1_000.0,
        sleeper=lambda _: None,
    )
    assert len(fake_launcher.jobs) == 6
    assert result.status == "completed"


def test_t2c_runner_reuses_compact_completed_job_without_launch(tmp_path):
    restored = restore_t2c_resume_source(resume, tmp_path / "output")
    result = run_t2c_stage(
        verified=verified,
        t2c_input=t2c_input,
        output_root=restored,
        deadline=20_000,
        launcher=launcher_that_must_not_start,
        clock=lambda: 1_000.0,
        sleeper=lambda _: None,
    )
    assert result.completed == ALL_JOB_IDS


def test_t2c_artifacts_bind_parent_and_compact_identities(tmp_path):
    bundles = write_t2c_bundles(
        tmp_path / "bundles",
        jobs_root=completed_jobs,
        completed=ALL_JOB_IDS,
        pending=(),
        failed=(),
        evidence=passing_evidence,
        parent_sha256="a" * 64,
    )
    assert restore_t2c_resume_source(bundles.resume, tmp_path / "restored").is_dir()
```

- [ ] **Step 2: Run orchestration tests and confirm RED**

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py -k 'runner or artifact or resume'
```

Expected: import failures for missing T2-C runner and artifact modules.

- [ ] **Step 3: Implement compact artifacts**

```python
REVIEW_KIND = "temporal_t2c_review_v1"
RESUME_KIND = "temporal_t2c_resume_v1"


@dataclass(frozen=True)
class T2CBundles:
    review: Path
    resume: Path


def write_t2c_bundles(
    output_dir: str | Path,
    *,
    jobs_root: str | Path,
    completed: tuple[str, ...],
    pending: tuple[str, ...],
    failed: tuple[str, ...],
    evidence: Mapping[str, object],
    parent_sha256: str,
) -> T2CBundles:
    """Write deterministic review and compact restart bundles."""


def restore_t2c_resume_source(source: str | Path, output_root: str | Path) -> Path:
    """Verify bundle identity before restoring worker files."""
```

Review stores completed predictions, metrics, checkpoint metadata, decision, and evidence. Resume
stores the same compact completed results plus restart files for active jobs. Both manifests bind
the T2-C input decision and all training identities.

- [ ] **Step 4: Implement the runner**

```python
@dataclass(frozen=True)
class T2CStageResult:
    status: str
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    evidence: Mapping[str, object]
    output_root: Path


def run_t2c_stage(
    *,
    verified: VerifiedOfficialData,
    t2c_input: str | Path,
    output_root: str | Path,
    deadline: float,
    launcher: Launcher | None = None,
    frame_loader: Callable[[Path], pd.DataFrame] = pd.read_csv,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> T2CStageResult:
    """Train or reuse six jobs, evaluate seeds and ensemble, and persist state."""
```

Use a two-GPU queue and a hard 7,200-second stage cap. Stop launching when the remaining time is
1,500 seconds, including a 600-second artifact reserve. Write `t2c_stage_result.json` atomically
before returning. A nonzero worker exit is failed, exit 75 or deadline termination is pending,
and incomplete evidence is `budget_inconclusive`.

- [ ] **Step 5: Run runner tests and commit**

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py -k 'runner or artifact or resume'
git add experiments/temporal_portfolio/t2c_artifacts.py experiments/temporal_portfolio/t2c_runner.py tests/test_temporal_portfolio_t2c.py
git commit -m "feat: run restartable temporal T2C campaign"
```

### Task 5: Kaggle one-cell handoff

**Files:**
- Create: `experiments/temporal_portfolio/t2c_platform.py`
- Create: `tools/build_temporal_portfolio_t2c_kaggle_cell.py`
- Create: `experiments/temporal_portfolio/T2C_KAGGLE_CELL.py`
- Modify: `experiments/temporal_portfolio/platform.py`
- Modify: `experiments/temporal_portfolio/t1_runner.py`
- Modify: `tests/test_temporal_portfolio_t2c.py`

- [ ] **Step 1: Write failing platform tests**

```python
def test_t2c_kaggle_cell_is_deterministic_small_and_single_handoff(tmp_path):
    first = build_t2c_kaggle_cell(tmp_path / "first.py")
    second = build_t2c_kaggle_cell(tmp_path / "second.py")
    text = first.read_text(encoding="utf-8")
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 1_000_000
    assert "files.download" not in text
    assert "temporal_t2c_handoff.zip" in text
    assert "temporal_t2c_emergency_handoff.zip" in text
    compile(text, str(first), "exec")


def test_gpu_probe_accepts_t2c_log_prefix():
    assert require_two_t4_gpus(
        lambda: ("Tesla T4", "Tesla T4"), log_prefix="T2C"
    ) == ("Tesla T4", "Tesla T4")
```

- [ ] **Step 2: Run platform tests and confirm RED**

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py -k 'kaggle_cell or gpu_probe'
```

Expected: missing T2-C platform module or rejected `T2C` log prefix.

- [ ] **Step 3: Implement the self-contained builder**

```python
def build_t2c_kaggle_cell(output: str | Path) -> Path:
    archive = _runtime_archive()
    rendered = TEMPLATE.replace(
        "__RUNTIME_B64__", base64.b64encode(archive).decode("ascii")
    )
    data = rendered.encode("utf-8")
    if len(data) >= 1_000_000:
        raise T2CPlatformError("generated T2-C Kaggle cell exceeds one megabyte")
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    return destination
```

The cell discovers exactly one official data root and one `temporal_t2c_input_v1`, optionally one
T2-C resume or handoff, pins dependency versions, verifies two Tesla T4 devices, tees logs, and
leaves exactly one normal or emergency handoff in `/kaggle/working`. Add
`T2C_KAGGLE_CELL.py` to the embedded-runtime exclusion list so repeated generation remains
deterministic. Permit `T2C` as a GPU log prefix.

- [ ] **Step 4: Generate and verify the canonical cell**

```bash
python tools/build_temporal_portfolio_t2c_kaggle_cell.py
python -m py_compile experiments/temporal_portfolio/T2C_KAGGLE_CELL.py
```

Expected log:

```text
T2C_KAGGLE_CELL_READY path=<absolute-path> size_bytes=<value-below-1000000>
```

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/t2c_platform.py experiments/temporal_portfolio/T2C_KAGGLE_CELL.py experiments/temporal_portfolio/platform.py experiments/temporal_portfolio/t1_runner.py tools/build_temporal_portfolio_t2c_kaggle_cell.py tests/test_temporal_portfolio_t2c.py
git commit -m "feat: launch temporal T2C on Kaggle"
```

### Task 6: Final synthetic verification and user handoff

**Files:**
- Modify only if verification reveals a T2-C defect.

- [ ] **Step 1: Run focused and full temporal tests**

```bash
python -m pytest -q tests/test_temporal_portfolio_t2c.py
python -m pytest -q tests/test_temporal_portfolio_*.py
```

Expected: all tests pass with zero failures and no collection errors.

- [ ] **Step 2: Verify source and generated runtime**

```bash
git diff --check
python -m py_compile experiments/temporal_portfolio/t2c*.py tools/*t2c*.py
python tools/prepare_temporal_portfolio_t2c_input.py --help
python tools/build_temporal_portfolio_t2c_kaggle_cell.py --help
```

Decode `RUNTIME_B64` from the generated cell into a fresh temporary directory and import
`experiments.temporal_portfolio.t2c_runner` from that directory. Assert the embedded constants
report six maximum jobs and a 7,200-second stage cap.

- [ ] **Step 3: Confirm prohibited outputs do not exist**

```bash
rg -n "sample_submission|submission.zip|test inference|full-data" \
  experiments/temporal_portfolio/t2c*.py \
  tools/*t2c*.py
```

Expected: no T2-C code path creates predictions for official test rows, full-data checkpoints, or
submission packages. Documentation text and explicit rejection messages are allowed.

- [ ] **Step 4: Provide one exact user-run preparation command**

Inputs:

- `/Users/yonghyun/Documents/lg-aimers-9th-competition/artifacts/temporal_portfolio_t2b_handoff/temporal_portfolio_t2b_input.zip`
- `/Users/yonghyun/Downloads/temporal_t2b_handoff.zip`

Output:

- `artifacts/temporal_portfolio_t2c_handoff/temporal_portfolio_t2c_input.zip`

Success text:

```text
T2C_INPUT_READY path=<path> sha256=<sha256> candidate=s1_game_type_f_fallback_v1
```

The user uploads the official data and generated T2-C input to Kaggle, pastes
`experiments/temporal_portfolio/T2C_KAGGLE_CELL.py` into one cell, enables T4×2, and runs Save
Version. Expected final text is:

```text
T2C_HANDOFF_READY status=completed path=/kaggle/working/temporal_t2c_handoff.zip
```

- [ ] **Step 5: Stop before final training or submission code**

After the user returns the verified T2-C handoff, inspect its decision. Only a `promoted` result
with matching artifact hashes authorizes a separate implementation plan for the nine full-data
models. Do not create a final training cell, test inference path, or submission packaging entry
point as part of this plan.
