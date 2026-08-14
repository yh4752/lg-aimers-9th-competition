# Colab TabM Stage C Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify one self-contained Colab T4 cell that safely finishes the single pending TabM Stage C job from direct uploads, survives ordinary interruptions through verified epoch snapshots, and returns review/resume evidence without creating a submission package.

**Architecture:** Keep the existing Kaggle T4×2 campaign path unchanged by adding explicit runner concurrency with a default of two. Put archive trust, Stage C resume sanitization, emergency snapshot verification, process identity, and delivery packaging in a focused `colab_recovery.py` module; generate a thin one-cell Colab wrapper around the sealed runtime. A local handoff builder deterministically packages the four official files and copies the exact Stage C resume and rendered cell into an ignored artifact directory.

**Tech Stack:** Python 3.11+, standard library (`zipfile`, `hashlib`, `subprocess`, `signal`, `threading`, `json`), pytest, PyTorch/TabM only inside the user-run Colab GPU process, Google Colab `files` API only in the generated cell.

---

## File Structure

- Create `experiments/tabm_campaign/colab_stage_c_contract.json`: immutable source-file hashes, base-resume identity, target candidate, and archive limits.
- Create `experiments/tabm_campaign/colab_recovery.py`: pure trust/recovery primitives plus the Colab subprocess supervisor CLI.
- Modify `experiments/tabm_campaign/runner.py`: accept an explicit `gpu_count`, defaulting to two.
- Modify `experiments/independent_dl/training.py`: honor an optional stop-after-epoch marker only after an atomic epoch checkpoint.
- Create `tools/render_tabm_colab_stage_c_recovery_cell.py`: deterministically embed the runtime and render the one-cell Colab program.
- Create `tools/prepare_tabm_colab_stage_c_handoff.py`: verify the local official files and base resume, then build the deterministic upload ZIP and handoff directory.
- Create `experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py`: generated copy-paste cell.
- Create `tests/test_tabm_campaign_colab_recovery.py`: archive, resume, snapshot, environment, lock, and delivery tests.
- Create `tests/test_tabm_campaign_colab_cell.py`: renderer determinism and static one-cell contract tests.
- Modify `tests/test_tabm_campaign_runner.py`: one-GPU scheduling regression tests.
- Modify `tests/test_independent_dl_training.py`: epoch-boundary stop tests.
- Create `docs/experiments/tabm-colab-stage-c-recovery.md`: exact user-run and recovery instructions.

### Task 1: Make Campaign Concurrency Explicit Without Changing Kaggle

**Files:**
- Modify: `experiments/tabm_campaign/runner.py` (`_run_with_completed_reuse`, `_run_a`, `_run_b`, `_run_c`, `run_one_version`)
- Modify: `tests/test_tabm_campaign_runner.py`

- [ ] **Step 1: Write the failing single-GPU runner test**

Add this test beside `test_two_gpus_run_one_independent_job_each`:

```python
def test_explicit_one_gpu_never_assigns_gpu_one(tmp_path: Path) -> None:
    runtime = _Runtime()

    run_one_version(
        tmp_path / "official",
        tmp_path / "out",
        runtime=runtime,
        gpu_count=1,
        now=lambda: 1000.0,
    )

    assert runtime.assignments
    assert set(runtime.assignments) == {0}
```

- [ ] **Step 2: Run the test and verify the API is missing**

Run: `python3 -m pytest tests/test_tabm_campaign_runner.py::test_explicit_one_gpu_never_assigns_gpu_one -v`

Expected: `FAILED` with `TypeError: run_one_version() got an unexpected keyword argument 'gpu_count'`.

- [ ] **Step 3: Thread `gpu_count` through the runner**

Use this validation helper and signatures; every `_run_with_completed_reuse(...)` call in A/B/C must pass the received value:

```python
def _validated_gpu_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CampaignRunnerError("gpu_count must be a positive integer")
    return value


def _run_with_completed_reuse(
    runtime: CampaignRuntime,
    version: str,
    jobs: tuple[CampaignJob, ...],
    output_dir: Path,
    deadline: float,
    prior: dict[str, object],
    gpu_count: int,
) -> tuple[CampaignJobResult, ...]:
    completed = _prior_completed(prior)
    pending = tuple(job for job in jobs if job.candidate_id not in completed)
    prior_rows = {
        str(row.get("candidate_id")): row
        for row in prior.get("results", [])
        if isinstance(row, dict)
    }
    for job in pending:
        training_dir = prior_rows.get(job.candidate_id, {}).get("training_dir")
        if training_dir is None:
            continue
        source = Path(str(training_dir))
        if source.is_dir():
            target = output_dir / job.candidate_id
            target.mkdir(parents=True, exist_ok=True)
            for path in source.iterdir():
                if path.is_file():
                    shutil.copy2(path, target / path.name)
    fresh = (
        runtime.run_jobs(
            version,
            pending,
            output_dir,
            gpu_count=gpu_count,
            job_deadline=deadline,
        )
        if pending
        else ()
    )
    by_id = {**completed, **{row.candidate_id: row for row in fresh}}
    return tuple(by_id[job.candidate_id] for job in jobs)
```

Add `gpu_count: int = 2` to `run_one_version`, validate it once, and add a required `gpu_count: int` argument to `_run_a`, `_run_b`, and `_run_c`. This preserves all callers that rely on Kaggle's two-GPU default.

- [ ] **Step 4: Run focused runner tests**

Run: `python3 -m pytest tests/test_tabm_campaign_runner.py -v`

Expected: all tests pass, including `test_two_gpus_run_one_independent_job_each` and the new one-GPU test.

- [ ] **Step 5: Commit the concurrency change**

```bash
git add experiments/tabm_campaign/runner.py tests/test_tabm_campaign_runner.py
git commit -m "feat: allow single GPU TabM campaign runs"
```

### Task 2: Stop Only at a Complete Epoch Boundary

**Files:**
- Modify: `experiments/independent_dl/training.py`
- Modify: `tests/test_independent_dl_training.py`

- [ ] **Step 1: Write failing marker-contract tests**

Add imports for `os` and these tests using `monkeypatch` and `tmp_path`:

```python
def test_stop_after_epoch_marker_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABM_STOP_AFTER_EPOCH_FILE", raising=False)
    assert stop_after_epoch_requested() is False


def test_stop_after_epoch_marker_requires_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TABM_STOP_AFTER_EPOCH_FILE", "relative.stop")
    with pytest.raises(TrainingContractError, match="absolute"):
        stop_after_epoch_requested()


def test_stop_after_epoch_marker_is_observed_only_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "stop-after-epoch"
    monkeypatch.setenv("TABM_STOP_AFTER_EPOCH_FILE", str(marker))
    assert stop_after_epoch_requested() is False
    marker.write_text("stop\n", encoding="utf-8")
    assert stop_after_epoch_requested() is True
```

- [ ] **Step 2: Run the marker tests and verify the helper is absent**

Run: `python3 -m pytest tests/test_independent_dl_training.py -k stop_after_epoch -v`

Expected: collection fails with `ImportError` or tests fail with `NameError` for `stop_after_epoch_requested`.

- [ ] **Step 3: Implement the marker reader and checkpoint-boundary check**

Add this helper near `_session_deadline`:

```python
def stop_after_epoch_requested() -> bool:
    raw = os.environ.get("TABM_STOP_AFTER_EPOCH_FILE")
    if raw is None or not raw.strip():
        return False
    marker = Path(raw)
    if not marker.is_absolute():
        raise TrainingContractError("TABM_STOP_AFTER_EPOCH_FILE must be absolute")
    return marker.is_file()
```

Immediately after the `EPOCH_CHECKPOINTED` event and before patience handling, add:

```python
            if stop_after_epoch_requested():
                budget_reached = True
                reporter.emit(
                    "TRAINING_STOP_AFTER_EPOCH_REQUESTED",
                    completed_epoch=epoch,
                    completed_epochs=len(validation_curve),
                )
                break
```

This location guarantees `checkpoint.pt`, `checkpoint_meta.json`, and any new `best_checkpoint.pt` are already atomically published before stopping.

- [ ] **Step 4: Run training tests**

Run: `python3 -m pytest tests/test_independent_dl_training.py -v`

Expected: all tests pass and the existing deadline behavior remains unchanged.

- [ ] **Step 5: Commit the epoch-boundary stop contract**

```bash
git add experiments/independent_dl/training.py tests/test_independent_dl_training.py
git commit -m "feat: support checkpoint-boundary training stops"
```

### Task 3: Seal the Colab Input Contract and Secure Archive Handling

**Files:**
- Create: `experiments/tabm_campaign/colab_stage_c_contract.json`
- Create: `experiments/tabm_campaign/colab_recovery.py`
- Create: `tests/test_tabm_campaign_colab_recovery.py`

- [ ] **Step 1: Add the immutable contract**

Create the JSON with these exact source identities:

```json
{
  "schema_version": 1,
  "base_resume": {
    "filename": "tabm_search_stage_C_resume_bundle.zip",
    "sha256": "6f7cc5b3c8280551f266767f63e71667db05473846e262371803a1fe1272d806",
    "version": "C"
  },
  "data_archive": {
    "filename": "lg-aimers-9th-data.zip",
    "max_member_count": 4,
    "max_uncompressed_bytes": 722352760,
    "members": {
      "sample_submission.csv": {"size": 112, "sha256": "b2cf6ba6745c74c46db23e620d74705f9b327efb8c8a14dc0dd2afab0e898775"},
      "test.csv": {"size": 1894, "sha256": "478d10b20c00443f6fe8270ab7348de49d14395526130f0b2d915da496821a19"},
      "trackman_history.csv": {"size": 353823031, "sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9"},
      "train.csv": {"size": 368527723, "sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"}
    }
  },
  "target_candidate_id": "c_final__a__p2__piecewise_linear__bce__plateau__s42__s3407__tr2022__va2023",
  "expected_completed_jobs": 20,
  "snapshot_interval_seconds": 1200,
  "interrupt_grace_seconds": 180
}
```

The uncompressed limit is the exact total `722352760`, not a loose bomb allowance.

- [ ] **Step 2: Write failing secure-archive tests**

Create fixtures that build four tiny members with a test contract, then add:

```python
def test_verify_and_extract_data_archive_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("../train.csv", b"escape")
    with pytest.raises(ColabRecoveryError, match="unsafe"):
        verify_and_extract_data_archive(archive, tmp_path / "data", TEST_CONTRACT)


def test_verify_and_extract_data_archive_is_atomic_and_hash_checked(
    tmp_path: Path,
) -> None:
    archive = build_fixture_data_zip(tmp_path, TEST_CONTRACT)
    published = verify_and_extract_data_archive(
        archive, tmp_path / "published", TEST_CONTRACT
    )
    assert {path.name for path in published.iterdir()} == set(
        TEST_CONTRACT["data_archive"]["members"]
    )
    assert not (tmp_path / ".published.extracting").exists()


def test_data_archive_rejects_symlink_duplicate_extra_and_wrong_hash(
    tmp_path: Path,
) -> None:
    for mutation in ("symlink", "duplicate", "extra", "wrong_hash"):
        archive = build_mutated_fixture_zip(tmp_path / mutation, mutation)
        with pytest.raises(ColabRecoveryError):
            verify_and_extract_data_archive(
                archive, tmp_path / f"out-{mutation}", TEST_CONTRACT
            )
```

- [ ] **Step 3: Run the archive tests and verify imports fail**

Run: `python3 -m pytest tests/test_tabm_campaign_colab_recovery.py -k data_archive -v`

Expected: collection fails because `experiments.tabm_campaign.colab_recovery` does not exist.

- [ ] **Step 4: Implement contract loading, hashing, and atomic extraction**

Implement these public interfaces:

```python
class ColabRecoveryError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_colab_contract(path: Path = Path(__file__).with_name(
    "colab_stage_c_contract.json"
)) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ColabRecoveryError("unsupported Colab recovery contract")
    return value
```

`verify_and_extract_data_archive(archive, destination, contract)` must inspect every `ZipInfo` before reading bytes, reject absolute/parent/backslash/directory/symlink/duplicate/unexpected members, require the exact declared member set and sizes, stream each member to `.destination.extracting`, verify SHA-256 while writing, `fsync`, and atomically rename only after all members pass. If the final destination already exists, verify its live file hashes before reusing it.

- [ ] **Step 5: Run archive tests**

Run: `python3 -m pytest tests/test_tabm_campaign_colab_recovery.py -k data_archive -v`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the input trust boundary**

```bash
git add experiments/tabm_campaign/colab_stage_c_contract.json experiments/tabm_campaign/colab_recovery.py tests/test_tabm_campaign_colab_recovery.py
git commit -m "feat: seal Colab Stage C input archives"
```

### Task 4: Sanitize the Kaggle Checkpoint and Verify Colab Snapshots

**Files:**
- Modify: `experiments/tabm_campaign/colab_recovery.py`
- Modify: `tests/test_tabm_campaign_colab_recovery.py`

- [ ] **Step 1: Write failing resume-sanitization tests**

Use a fixture Stage C resume with 20 completed rows and the target incomplete row, then add:

```python
def test_sanitize_stage_c_resume_preserves_completed_bytes_and_drops_only_target_training(
    stage_c_resume: Path,
    tmp_path: Path,
) -> None:
    sanitized = sanitize_stage_c_resume(
        stage_c_resume,
        tmp_path / "sanitized.zip",
        BASE_CONTRACT,
    )
    source = read_zip_members(stage_c_resume)
    output = read_zip_members(sanitized.path)
    target_prefix = f"training/{TARGET}/"
    for name, value in source.items():
        if name == "manifest.json" or name == "stage_state.json" or name.startswith(target_prefix):
            continue
        assert output[name] == value
    state = json.loads(output["stage_state.json"])
    target = next(row for row in state["results"] if row["candidate_id"] == TARGET)
    assert target["status"] == "inconclusive"
    assert target["brier"] is None
    assert target["completed_epochs"] == 0
    assert TARGET not in state["resume_artifacts"]
    assert sum(row["status"] == "completed" for row in state["results"]) == 20
    assert verify_resume_bundle(sanitized.path).version == "C"
```

Also test wrong base SHA, wrong stage/reason, a second incomplete job, and a changed completed prediction hash; each must raise `ColabRecoveryError` before writing the sanitized output.

- [ ] **Step 2: Run the sanitization tests and verify failure**

Run: `python3 -m pytest tests/test_tabm_campaign_colab_recovery.py -k sanitize -v`

Expected: tests fail because `sanitize_stage_c_resume` is absent.

- [ ] **Step 3: Implement deterministic resume repackaging**

Add these interfaces:

```python
@dataclass(frozen=True)
class SanitizedResume:
    path: Path
    source_sha256: str
    source_manifest_sha256: str
    sanitized_sha256: str
    sanitized_manifest_sha256: str


def sanitize_stage_c_resume(
    source: Path,
    destination: Path,
    contract: Mapping[str, object],
) -> SanitizedResume:
    expected = contract["base_resume"]
    source_sha = file_sha256(source)
    if source_sha != expected["sha256"]:
        raise ColabRecoveryError("base Stage C resume SHA-256 differs")
    verified = verify_resume_bundle(source)
    if verified.version != "C":
        raise ColabRecoveryError("base resume is not Stage C")
    with ZipFile(source) as archive:
        members = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "manifest.json"
        }
    state = json.loads(members["stage_state.json"])
    rows = state.get("results")
    if (
        state.get("stage_complete") is not False
        or state.get("reason") != "older_fold_seed_confirmation_incomplete"
        or not isinstance(rows, list)
    ):
        raise ColabRecoveryError("base Stage C state is not the expected incomplete state")
    target_id = str(contract["target_candidate_id"])
    completed = [row for row in rows if row.get("status") == "completed"]
    incomplete = [row for row in rows if row.get("status") != "completed"]
    if (
        len(completed) != int(contract["expected_completed_jobs"])
        or len(incomplete) != 1
        or incomplete[0].get("candidate_id") != target_id
    ):
        raise ColabRecoveryError("Stage C completed/pending job identity differs")
    target = incomplete[0]
    target.update({
        "status": "inconclusive",
        "brier": None,
        "best_epoch": None,
        "completed_epochs": 0,
        "checkpoint": None,
        "predictions": None,
        "resource_evidence": {},
        "failure": "cross_runtime_restart_required",
    })
    state["resume_artifacts"].pop(target_id, None)
    prefix = f"training/{target_id}/"
    members = {name: value for name, value in members.items() if not name.startswith(prefix)}
    members["stage_state.json"] = canonical_json(state)
    path, manifest_sha = write_verified_resume_members(
        destination,
        version="C",
        campaign_config_sha256=verified.campaign_config_sha256,
        prior_manifest_sha256=verified.prior_manifest_sha256,
        members=members,
    )
    return SanitizedResume(
        path=path,
        source_sha256=source_sha,
        source_manifest_sha256=verified.manifest_sha256,
        sanitized_sha256=file_sha256(path),
        sanitized_manifest_sha256=manifest_sha,
    )
```

Implement `write_verified_resume_members(...)` with the fixed `(2026, 1, 1, 0, 0, 0)` ZIP timestamp, sorted member order, canonical manifest JSON, temporary-file publication, and a final `verify_resume_bundle` call. Do not mutate or overwrite the uploaded source ZIP.

- [ ] **Step 4: Write failing emergency snapshot tests**

Add tests for a stable snapshot and every rejection gate:

```python
def test_snapshot_round_trip_restores_next_epoch(
    sanitized_resume: SanitizedResume,
    stable_training_dir: Path,
    tmp_path: Path,
) -> None:
    snapshot = create_emergency_snapshot(
        training_dir=stable_training_dir,
        log_path=tmp_path / "colab.log",
        output_dir=tmp_path / "snapshots",
        identity=FIXTURE_IDENTITY,
    )
    verified = verify_emergency_snapshots(
        [snapshot.path], sanitized_resume, FIXTURE_IDENTITY
    )
    merged = merge_emergency_snapshot(
        sanitized_resume,
        verified,
        tmp_path / "merged.zip",
    )
    state, files = extract_resume_state_for_test(merged)
    assert state["resume_artifacts"][TARGET]["training_files"]
    assert json.loads(files[f"training/{TARGET}/checkpoint_meta.json"])["epoch"] == 4
```

Parameterized rejection tests must cover: tampered member, wrong base archive SHA, wrong base manifest SHA, wrong candidate, wrong environment, wrong campaign config, wrong cache binding, lower epoch selected over a higher compatible epoch, and two different snapshots claiming the same epoch.

- [ ] **Step 5: Implement snapshot creation, verification, selection, and merge**

Use these data contracts:

```python
@dataclass(frozen=True)
class RuntimeIdentity:
    base_resume_sha256: str
    base_manifest_sha256: str
    sanitized_resume_sha256: str
    campaign_config_sha256: str
    runtime_sha256: str
    training_source_sha256: str
    cache_sha256: str | None
    python: str
    torch: str
    cuda_runtime: str
    numpy: str
    pandas: str
    tabm: str
    rtdl_num_embeddings: str
    gpu_name: str


@dataclass(frozen=True)
class EmergencySnapshot:
    path: Path
    epoch: int
    sha256: str
    manifest: Mapping[str, object]
```

`create_emergency_snapshot` must read `checkpoint_meta.json` twice around hashing/copying, require identical bytes and candidate ID, require `checkpoint.pt` and `best_checkpoint.pt`, and obtain `cache_sha256` from the metadata's checkpoint binding. Use unique `tabm_colab_emergency_epoch_{epoch:03d}_{sha12}.zip` names, write through a temporary file, verify member hashes, and atomically rename. On a fresh VM, `verify_emergency_snapshots` validates that the manifest cache hash equals the checkpoint metadata binding; after cache materialization, the existing `_resume_epoch` binding check rejects a live cache mismatch before loading model/optimizer state. The verifier returns the highest compatible epoch and rejects same-epoch hash conflicts. `merge_emergency_snapshot` adds only the three training members to the sanitized resume, updates only the target row and target `resume_artifacts`, then calls `verify_resume_bundle`.

- [ ] **Step 6: Run all recovery-core tests**

Run: `python3 -m pytest tests/test_tabm_campaign_colab_recovery.py -v`

Expected: all tests pass without importing torch or running full training.

- [ ] **Step 7: Commit resume and snapshot recovery**

```bash
git add experiments/tabm_campaign/colab_recovery.py tests/test_tabm_campaign_colab_recovery.py
git commit -m "feat: add verified Stage C checkpoint recovery"
```

### Task 5: Add Process Identity, Interrupt Supervision, and Final Delivery

**Files:**
- Modify: `experiments/tabm_campaign/colab_recovery.py`
- Modify: `tests/test_tabm_campaign_colab_recovery.py`

- [ ] **Step 1: Write failing process-lock tests**

Add a fixture subprocess that sleeps and these cases:

```python
def test_live_matching_lock_refuses_duplicate_run(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        receipt = write_process_receipt(tmp_path / "run.lock", process.pid, "run-a")
        with pytest.raises(ColabRecoveryError, match="already running"):
            acquire_process_lock(tmp_path / "run.lock", "run-b")
        assert receipt.pid == process.pid
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_stale_or_pid_reused_lock_is_archived(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    lock.write_bytes(canonical_json({
        "pid": os.getpid(),
        "process_start_identity": "different-start",
        "command_sha256": "0" * 64,
        "run_uuid": "old",
    }))
    acquired = acquire_process_lock(lock, "new")
    assert acquired.run_uuid == "new"
    assert list(tmp_path.glob("run.lock.stale-*"))
```

- [ ] **Step 2: Run lock tests and verify failure**

Run: `python3 -m pytest tests/test_tabm_campaign_colab_recovery.py -k lock -v`

Expected: tests fail because the receipt/lock functions are absent.

- [ ] **Step 3: Implement Linux process identity and safe termination**

Use `/proc/<pid>/stat` field 22 plus `/proc/<pid>/cmdline` hash:

```python
@dataclass(frozen=True)
class ProcessReceipt:
    pid: int
    process_start_identity: str
    command_sha256: str
    run_uuid: str


def linux_process_identity(pid: int) -> tuple[str, str] | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
        command = cmdline_path.read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return fields[21], sha256(command).hexdigest()
```

`terminate_matching_process_group(receipt)` must re-read both identities before `os.killpg`, send `SIGTERM`, wait up to 10 seconds, then send `SIGKILL` only if the same identity is still alive. It must never act on PID alone.

- [ ] **Step 4: Write failing supervisor and delivery tests**

Use a fixture campaign command that writes epoch metadata, reacts to the stop marker, and emits fake verified Stage C bundles. Assert:

```python
def test_supervisor_interrupt_requests_epoch_boundary_and_exports_snapshot(
    fixture_campaign: FixtureCampaign,
    tmp_path: Path,
) -> None:
    result = supervise_campaign(
        command=fixture_campaign.command,
        work_root=tmp_path / "work",
        identity=FIXTURE_IDENTITY,
        snapshot_interval_seconds=0,
        interrupt_after_seconds=0.1,
        interrupt_grace_seconds=3,
    )
    assert result.interrupted is True
    assert result.latest_snapshot is not None
    assert result.latest_snapshot.epoch == 0
    assert fixture_campaign.stop_marker.is_file()
    assert linux_process_identity(result.receipt.pid) is None


def test_delivery_verifies_nested_bundles_and_binds_hashes(
    completed_stage_c_bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    delivery = write_delivery_bundle(
        review=completed_stage_c_bundles[0],
        resume=completed_stage_c_bundles[1],
        log=tmp_path / "colab.log",
        identity=FIXTURE_IDENTITY,
        run_uuid="run-a",
        destination=tmp_path / "tabm_colab_stage_C_delivery.zip",
    )
    verified = verify_delivery_bundle(delivery)
    assert verified.final_stage_complete is True
    assert set(verified.member_sha256) == {
        "tabm_search_stage_C_review_bundle.zip",
        "tabm_search_stage_C_resume_bundle.zip",
        "colab_stage_C.log",
    }
```

- [ ] **Step 5: Implement the supervisor**

`supervise_campaign(...)` must:

1. acquire a lock before launching;
2. call `Popen(..., start_new_session=True, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)`;
3. stream each line to stdout and an append-only UTF-8 log with immediate flush;
4. poll the stable checkpoint every two seconds;
5. create a snapshot only after a new epoch and 1,200 elapsed seconds, plus on error/interrupt;
6. on `KeyboardInterrupt`, atomically create the stop marker and wait at most 180 seconds for a newer complete epoch;
7. export the newest stable checkpoint, safely terminate only the matching process group if still alive, and re-raise a typed interrupted result;
8. preserve every completed snapshot and remove only its own lock after verifying the child identity is gone.

The `interrupt_after_seconds` parameter exists only for fixture tests and defaults to `None`.

- [ ] **Step 6: Implement deterministic final delivery verification**

`write_delivery_bundle(...)` must call `verify_review_bundle` and `verify_resume_bundle`, require both version `C`, read both `stage_state.json` files, require identical bytes and `stage_complete is True`, and write the three payload members plus `delivery_manifest.json`. The manifest records source/runtime identity, run UUID, member sizes/hashes, and final state. `verify_delivery_bundle` must independently re-hash every member. Neither function may contain `submission`, `submit.zip`, or a prediction-only packaging path.

- [ ] **Step 7: Run supervisor and delivery tests**

Run: `python3 -m pytest tests/test_tabm_campaign_colab_recovery.py -k 'lock or supervisor or delivery' -v`

Expected: all selected tests pass; no fixture child remains alive.

- [ ] **Step 8: Commit supervision and delivery**

```bash
git add experiments/tabm_campaign/colab_recovery.py tests/test_tabm_campaign_colab_recovery.py
git commit -m "feat: supervise resumable Colab Stage C runs"
```

### Task 6: Render the Self-Contained Colab Cell

**Files:**
- Create: `tools/render_tabm_colab_stage_c_recovery_cell.py`
- Create: `experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py`
- Create: `tests/test_tabm_campaign_colab_cell.py`

- [ ] **Step 1: Write failing renderer contract tests**

```python
CELL = Path("experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py")
RENDERER = Path("tools/render_tabm_colab_stage_c_recovery_cell.py")


def test_colab_cell_is_self_contained_and_small() -> None:
    text = CELL.read_text(encoding="utf-8")
    compile(text, str(CELL), "exec")
    assert CELL.stat().st_size < 950_000
    assert "google.colab" in text
    assert "files.upload()" in text
    assert "files.download(" in text
    assert "/content/tabm_stage_c_recovery" in text
    assert "gpu_count=1" in text


def test_colab_cell_has_no_drive_github_or_submission_path() -> None:
    text = CELL.read_text(encoding="utf-8").lower()
    assert "drive.mount" not in text
    assert "git clone" not in text
    assert "github.com" not in text
    assert "submit.zip" not in text
    assert "submission package" not in text


def test_colab_renderer_is_byte_deterministic() -> None:
    subprocess.run([sys.executable, str(RENDERER)], check=True)
    first = sha256(CELL.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(RENDERER)], check=True)
    assert sha256(CELL.read_bytes()).hexdigest() == first
```

- [ ] **Step 2: Run renderer tests and verify missing files**

Run: `python3 -m pytest tests/test_tabm_campaign_colab_cell.py -v`

Expected: tests fail because the renderer and generated cell do not exist.

- [ ] **Step 3: Implement the deterministic runtime renderer**

Follow the existing Kaggle renderer's deterministic tar/gzip rules, but render a Colab wrapper with fixed roots:

```python
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py"
WORK_ROOT = Path("/content/tabm_stage_c_recovery")
```

The embedded source list must contain the full `experiments/tabm_campaign`, the required `experiments/independent_dl` modules, `competition_rules`, and the policy review JSON, while excluding both generated cell files. The wrapper must verify the embedded gzip SHA before secure extraction and print `COLAB_CODE_READY`.

- [ ] **Step 4: Implement sequential uploads and same-VM reuse**

The rendered cell must use this one-file-at-a-time helper so the 700 MB data payload and 93 MB resume are not retained together in the upload mapping:

```python
def upload_one(expected_name: str, destination: Path) -> Path:
    print(f"COLAB_UPLOAD_REQUIRED filename={expected_name}", flush=True)
    uploaded = files.upload()
    if set(uploaded) != {expected_name}:
        raise RuntimeError(
            f"expected exactly {expected_name}; received={sorted(uploaded)}"
        )
    source = Path(expected_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    uploaded.clear()
    del uploaded
    gc.collect()
    return destination
```

Before prompting, the cell must verify and reuse existing `/content/tabm_stage_c_recovery/inputs` receipts. It prompts separately for the data ZIP, base resume ZIP, and any optional emergency ZIPs. Emergency upload is skipped when no recovery snapshot is needed.

- [ ] **Step 5: Implement GPU/runtime checks and campaign command**

The wrapper must require `torch.cuda.device_count() == 1`, exact device name containing `T4`, record dependency versions, then launch the embedded recovery CLI with the equivalent arguments:

```python
command = [
    sys.executable,
    "-m",
    "experiments.tabm_campaign.colab_recovery",
    "run",
    "--data-dir", str(DATA_ROOT),
    "--base-resume", str(BASE_RESUME),
    "--work-root", str(WORK_ROOT),
    "--runtime-sha256", EXPECTED_RUNTIME_SHA256,
    "--gpu-count", "1",
]
```

The module CLI sanitizes the base resume on first use, restores the highest compatible snapshot when supplied, and invokes `run_one_version(..., gpu_count=1)`. It must stream `STAGE_SELECTED`, `JOB_START`, `TRAINING_PROGRESS`, and `EPOCH_CHECKPOINTED` lines unchanged. The generated cell launches this CLI with `Popen(..., start_new_session=True)` and reads stdout line by line. When a line matches `EMERGENCY_SNAPSHOT_READY`, the cell verifies the printed path is under its snapshot root and invokes the browser download block in Step 6 immediately. On `KeyboardInterrupt`, it sends `SIGINT` to the matching CLI process group and allows the CLI's bounded supervisor cleanup to finish.

- [ ] **Step 6: Implement browser download requests without false guarantees**

For every completed emergency or final artifact, print and request exactly:

```python
print(f"EMERGENCY_SNAPSHOT_READY epoch={epoch} path={path} sha256={digest}", flush=True)
display(FileLink(str(path)))
files.download(str(path))
print(f"EMERGENCY_DOWNLOAD_REQUESTED path={path}", flush=True)
```

For final success, use `COLAB_DELIVERY_READY` and `COLAB_DOWNLOAD_REQUESTED`. Do not print that the browser saved a file. On every exception print `COLAB_STAGE_C_ERROR stage=<stage> type=<type> message=<message>` and re-raise.

- [ ] **Step 7: Render and run static cell tests**

Run:

```bash
python3 tools/render_tabm_colab_stage_c_recovery_cell.py
python3 -m pytest tests/test_tabm_campaign_colab_cell.py -v
```

Expected: renderer prints `rendered=... size_bytes=<n> sha256=<sha>` and all cell tests pass.

- [ ] **Step 8: Commit the renderer and generated cell**

```bash
git add tools/render_tabm_colab_stage_c_recovery_cell.py experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py tests/test_tabm_campaign_colab_cell.py
git commit -m "feat: render one-cell Colab Stage C recovery"
```

### Task 7: Build and Verify the Local Upload Handoff

**Files:**
- Create: `tools/prepare_tabm_colab_stage_c_handoff.py`
- Modify: `tests/test_tabm_campaign_colab_recovery.py`
- Create: `docs/experiments/tabm-colab-stage-c-recovery.md`

- [ ] **Step 1: Write failing deterministic handoff tests**

```python
def test_prepare_handoff_is_deterministic(
    fixture_official_data: Path,
    stage_c_resume: Path,
    tmp_path: Path,
) -> None:
    first = prepare_handoff(
        fixture_official_data, stage_c_resume, tmp_path / "first", TEST_CONTRACT
    )
    second = prepare_handoff(
        fixture_official_data, stage_c_resume, tmp_path / "second", TEST_CONTRACT
    )
    assert file_sha256(first.data_zip) == file_sha256(second.data_zip)
    assert json.loads(first.manifest.read_text()) == json.loads(
        second.manifest.read_text()
    )


def test_prepare_handoff_refuses_changed_source_file(
    fixture_official_data: Path,
    stage_c_resume: Path,
    tmp_path: Path,
) -> None:
    (fixture_official_data / "train.csv").write_bytes(b"changed")
    with pytest.raises(ColabRecoveryError, match="train.csv"):
        prepare_handoff(
            fixture_official_data, stage_c_resume, tmp_path / "out", TEST_CONTRACT
        )
```

- [ ] **Step 2: Run handoff tests and verify the builder is absent**

Run: `python3 -m pytest tests/test_tabm_campaign_colab_recovery.py -k handoff -v`

Expected: tests fail because `prepare_handoff` is absent.

- [ ] **Step 3: Implement deterministic ZIP creation and CLI**

The builder writes fixed timestamps/modes and streams source files into `ZipFile.open`:

```python
@dataclass(frozen=True)
class HandoffPaths:
    root: Path
    data_zip: Path
    resume_zip: Path
    cell: Path
    manifest: Path


def prepare_handoff(
    data_dir: Path,
    base_resume: Path,
    output_dir: Path,
    contract: Mapping[str, object],
) -> HandoffPaths:
    declared = contract["data_archive"]["members"]
    actual_names = {path.name for path in data_dir.iterdir() if path.is_file()}
    if actual_names != set(declared):
        raise ColabRecoveryError("official data member names differ")
    for name, evidence in declared.items():
        source = data_dir / name
        if source.stat().st_size != evidence["size"] or file_sha256(source) != evidence["sha256"]:
            raise ColabRecoveryError(f"official data file differs: {name}")
    if file_sha256(base_resume) != contract["base_resume"]["sha256"]:
        raise ColabRecoveryError("base Stage C resume SHA-256 differs")
    verify_resume_bundle(base_resume)
    cell_source = Path("experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_zip = output_dir / contract["data_archive"]["filename"]
    write_deterministic_data_zip(data_dir, data_zip, declared)
    with tempfile.TemporaryDirectory() as temporary:
        verify_and_extract_data_archive(
            data_zip, Path(temporary) / "verified", contract
        )
    resume_zip = output_dir / contract["base_resume"]["filename"]
    cell = output_dir / cell_source.name
    shutil.copyfile(base_resume, resume_zip)
    shutil.copyfile(cell_source, cell)
    payloads = (data_zip, resume_zip, cell)
    manifest_value = {
        "schema_version": 1,
        "files": {
            path.name: {"size": path.stat().st_size, "sha256": file_sha256(path)}
            for path in payloads
        },
    }
    manifest = output_dir / "handoff_manifest.json"
    atomic_write(manifest, canonical_json(manifest_value))
    return HandoffPaths(output_dir, data_zip, resume_zip, cell, manifest)
```

Implement `write_deterministic_data_zip(...)` with sorted declared names, fixed timestamp/mode, `ZIP_DEFLATED`, and `shutil.copyfileobj` into `ZipFile.open`; implement `atomic_write(...)` using `mkstemp`, flush, `fsync`, and `os.replace`. The CLI accepts `--data-dir`, `--base-resume`, and `--output-dir`; it prints `COLAB_HANDOFF_READY root=<path> data_sha256=<sha> resume_sha256=<sha> cell_sha256=<sha>` only after re-verification.

- [ ] **Step 4: Add exact user instructions**

Document:

1. choose Colab `T4 GPU` and verify one GPU;
2. copy all of `COLAB_STAGE_C_RECOVERY_CELL.py` into one cell;
3. upload `lg-aimers-9th-data.zip`, then `tabm_search_stage_C_resume_bundle.zip` only when each prompt appears;
4. confirm each `EMERGENCY_SNAPSHOT_READY` file is present in the computer's Downloads folder;
5. after a VM reset, rerun the same cell and additionally upload the newest emergency ZIP when prompted;
6. return only `tabm_colab_stage_C_delivery.zip` to Codex on final success;
7. return the full log from `COLAB_STAGE_C_ERROR` onward on failure.

State an expected first-run duration of 20–70 minutes plus upload/cache time, same-VM rerun safety, and that no submission package is produced.

- [ ] **Step 5: Run builder tests and build the real ignored handoff**

Run:

```bash
python3 -m pytest tests/test_tabm_campaign_colab_recovery.py -k handoff -v
python3 tools/prepare_tabm_colab_stage_c_handoff.py \
  --data-dir /Users/yonghyun/Documents/kaggle-lg-aimers-9th-data-upload \
  --base-resume /Users/yonghyun/Downloads/tabm_search_stage_C_resume_bundle.zip \
  --output-dir artifacts/tabm_colab_stage_c_handoff
```

Expected: `COLAB_HANDOFF_READY` and these four files:

```text
artifacts/tabm_colab_stage_c_handoff/
├── COLAB_STAGE_C_RECOVERY_CELL.py
├── handoff_manifest.json
├── lg-aimers-9th-data.zip
└── tabm_search_stage_C_resume_bundle.zip
```

- [ ] **Step 6: Commit the builder and guide, not the ignored artifacts**

```bash
git add tools/prepare_tabm_colab_stage_c_handoff.py tests/test_tabm_campaign_colab_recovery.py docs/experiments/tabm-colab-stage-c-recovery.md
git commit -m "docs: add Colab Stage C recovery handoff"
```

### Task 8: Complete Static and Synthetic Verification

**Files:**
- Verify all files changed in Tasks 1–7

- [ ] **Step 1: Run focused tests**

Run:

```bash
python3 -m pytest \
  tests/test_tabm_campaign_colab_recovery.py \
  tests/test_tabm_campaign_colab_cell.py \
  tests/test_tabm_campaign_runner.py \
  tests/test_independent_dl_training.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Regenerate every checked-in generated cell and verify determinism**

Run:

```bash
python3 tools/render_tabm_campaign_kaggle_cell.py
python3 tools/render_tabm_colab_stage_c_recovery_cell.py
git diff --check
```

Expected: both renderers succeed and `git diff --check` prints nothing.

- [ ] **Step 3: Run the full repository suite**

Run: `python3 -m pytest -q`

Expected: exit code 0. This remains a static/synthetic verification run; it does not train on full data.

- [ ] **Step 4: Audit prohibited paths and package boundaries**

Run:

```bash
rg -n "drive\.mount|git clone|github\.com|submit\.zip|submission package" \
  experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py \
  experiments/tabm_campaign/colab_recovery.py
rg -n "test\.csv|sample_submission\.csv" experiments/tabm_campaign/worker.py
```

Expected: the first command finds nothing. The second command finds no training/preprocessing use of evaluation rows; the fixed workflow continues to fit only official training folds and `trackman_history.csv`.

- [ ] **Step 5: Verify the real handoff manifest and archive twice**

Run:

```bash
python3 tools/prepare_tabm_colab_stage_c_handoff.py \
  --data-dir /Users/yonghyun/Documents/kaggle-lg-aimers-9th-data-upload \
  --base-resume /Users/yonghyun/Downloads/tabm_search_stage_C_resume_bundle.zip \
  --output-dir artifacts/tabm_colab_stage_c_handoff_second
shasum -a 256 \
  artifacts/tabm_colab_stage_c_handoff/lg-aimers-9th-data.zip \
  artifacts/tabm_colab_stage_c_handoff_second/lg-aimers-9th-data.zip
```

Expected: both data ZIP hashes are identical.

- [ ] **Step 6: Review repository state and commit only intended source changes**

Run:

```bash
git status --short
git diff --check
git log --oneline -8
```

Expected: ignored ZIP/data artifacts are absent from `git status`; pre-existing unrelated user changes remain untouched. If a final source-only adjustment is needed, stage only the listed Task 1–7 files and commit it with `fix: finalize Colab Stage C recovery`.

## Success Criteria

- Kaggle still defaults to two workers; the Colab entry point proves it schedules only GPU 0.
- The uploaded base resume hash is exact, all 20 completed jobs and their prediction bytes survive, and only the cross-runtime incomplete checkpoint is discarded.
- A Colab checkpoint is restored only when base, source, cache, environment, GPU, candidate, epoch, and member hashes match.
- Interrupt handling waits no more than 180 seconds and never kills a process using PID alone.
- Logs are flushed to the cell and file; browser downloads are requested but never falsely reported as saved.
- Final output is a verified research delivery ZIP containing Stage C review/resume/log evidence and no submission package.
- Codex performs no full-data GPU training; the user runs the final one-cell Colab operation.
