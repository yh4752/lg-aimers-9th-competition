# S4 Compact Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the verified interrupted S4 handoff into a compact recovery input and resume only pending S4 work on Kaggle T4 ×2 without another disk-exhaustion failure.

**Architecture:** A sealed recovery contract authorizes one exact predecessor artifact and runtime migration. A local streaming compactor verifies and filters the nested resume, while the Kaggle runtime verifies the compact recovery input, restores current-bound state, prunes obsolete candidates after selection, and writes compact snapshots with free-space guards.

**Tech Stack:** Python 3.11/3.12, `zipfile`, SHA-256 manifests, pathlib, pytest, Kaggle Tesla T4 ×2.

---

### Task 1: Seal the Recovery Identity

**Files:**
- Create: `experiments/tree_expert/s4_recovery_contract.json`
- Create: `experiments/tree_expert/s4_recovery.py`
- Create: `tests/test_tree_expert_s4_recovery.py`

- [ ] **Step 1: Write the failing contract tests**

```python
def test_recovery_contract_seals_source_and_predecessor():
    contract = load_recovery_contract()
    assert contract.source_handoff_sha256 == "c8becb4037e511ff5d28d0fd146e1a9ec73923125cb4fb9c4467ba3cab6f8e9d"
    assert contract.predecessor_code_sha256 == "5e73e15269723679d421f88ddbb04141941b5760a9d7257c550b42183acd35c8"

def test_recovery_contract_rejects_unknown_keys(tmp_path):
    payload = valid_contract_payload()
    payload["unexpected"] = True
    with pytest.raises(S4RecoveryError, match="contract keys differ"):
        parse_recovery_contract(payload)
```

- [ ] **Step 2: Run the tests and confirm they fail because the recovery module is absent**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_recovery.py -q`

Expected: collection failure for `experiments.tree_expert.s4_recovery`.

- [ ] **Step 3: Add the exact recovery contract and strict parser**

The JSON records schema version, artifact kind, source handoff SHA-256, predecessor code SHA-256, S4 contract SHA-256, and the immutable official/input/E2 binding values. `S4RecoveryContract` exposes those exact fields through a frozen dataclass. The parser rejects missing, extra, malformed, or non-lowercase SHA-256 values.

- [ ] **Step 4: Run the recovery contract tests**

Expected: contract tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/s4_recovery_contract.json experiments/tree_expert/s4_recovery.py tests/test_tree_expert_s4_recovery.py
git commit -m "feat: seal S4 recovery identity"
```

### Task 2: Compact and Rebind the Interrupted Resume

**Files:**
- Modify: `experiments/tree_expert/s4_recovery.py`
- Modify: `tests/test_tree_expert_s4_recovery.py`

- [ ] **Step 1: Write failing compaction tests**

Use a fixture handoff containing `jobs/`, `anchor_basis/`, `verified_e2/`, anchors, residual predictions, full-chain outputs, decisions, diagnostics, log, and `state.phase="full_chains"`. Assert:

```python
result = compact_recovery_handoff(source, output, destination_code_sha256="d" * 64)
verified = verify_recovery_input(result.path, expected_code_sha256="d" * 64)
assert verified.state_phase == "full_chains"
assert verified.completed_full_chains == tuple(range(14))
assert not any(name.startswith(("jobs/", "anchor_basis/", "verified_e2/")) for name in verified.resume_members)
assert "full_chains/c13/config.json" in verified.resume_members
```

Add separate failures for a changed outer hash, predecessor code, nested member digest, and missing output required by a completed full-chain job.

- [ ] **Step 2: Run the compaction tests and confirm the missing behavior fails**

- [ ] **Step 3: Implement streaming verification and compaction**

Add these public boundaries:

```python
@dataclass(frozen=True)
class RecoveryResult:
    path: Path
    sha256: str
    source_sha256: str
    retained_bytes: int
    dropped_bytes: int

def compact_recovery_handoff(
    source_handoff: Path,
    output: Path,
    *,
    destination_code_sha256: str,
) -> RecoveryResult: ...

def verify_recovery_input(
    source: Path,
    *,
    expected_code_sha256: str,
) -> VerifiedRecoveryInput: ...
```

The implementation streams the 2.5 GB nested resume to a temporary file, verifies the source handoff and resume manifests, filters members by the saved phase, writes a deterministic current-bound compact resume, then writes `tree_s4_recovery_input_v1`. Existing output paths and symlinks are rejected. Temporary files are removed in `finally` blocks.

- [ ] **Step 4: Run all recovery tests**

Expected: tampering and binding tests fail closed; valid compaction passes.

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/s4_recovery.py tests/test_tree_expert_s4_recovery.py
git commit -m "feat: compact interrupted S4 resume"
```

### Task 3: Restore Recovery Inputs on Kaggle

**Files:**
- Modify: `experiments/tree_expert/s4_kaggle.py`
- Modify: `tests/test_tree_expert_s4_kaggle.py`

- [ ] **Step 1: Write failing discovery and restore tests**

Test both a ZIP and a recursively expanded `tree_s4_recovery_input_v1` directory. Assert exactly one S4 base input, zero or one recovery input, and no ambiguity with a previous ordinary handoff. Test that restored resume bindings use the generated runtime identity.

- [ ] **Step 2: Confirm the tests fail with the current two-input discovery contract**

- [ ] **Step 3: Add recovery discovery and materialization**

Extend `DiscoveredS4Inputs` with `recovery_input`. Discovery rejects simultaneous recovery input and ordinary previous handoff. Materialization reconstructs only the compact nested resume, validates its manifest and current bindings, and streams it to `/kaggle/working/tree_s4_resume.zip` without rebuilding a multi-gigabyte outer handoff.

Add `s4_recovery_contract.json` and `s4_recovery.py` to `_RUNTIME_MEMBERS`.

- [ ] **Step 4: Run Kaggle and recovery tests**

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/s4_kaggle.py tests/test_tree_expert_s4_kaggle.py
git commit -m "feat: restore compact S4 recovery inputs"
```

### Task 4: Prune Disposable Runtime Artifacts

**Files:**
- Modify: `experiments/tree_expert/s4_production.py`
- Modify: `experiments/tree_expert/s4_artifacts.py`
- Modify: `tests/test_tree_expert_s4_artifacts.py`
- Modify: `tests/test_tree_expert_s4_runner.py`
- Modify: `tests/test_tree_expert_s4_recovery.py`

- [ ] **Step 1: Write failing retention tests**

Create a full-chain fixture with selected indices `(2, 5)` and assert pruning keeps only their anchor dependencies, residual predictions, full-chain configurations, and later-phase evidence. Assert OOF model files are absent after a successful residual job and excluded from resumes even if a stale copy exists.

- [ ] **Step 2: Confirm the retention tests fail**

- [ ] **Step 3: Implement phase-aware pruning**

Add:

```python
def prune_after_full_chain_selection(root: Path, selected: tuple[int, ...], archetypes) -> PruneReport: ...
```

Call it only after `confirmation_selection.json` is atomically committed. It removes unselected large data, anchor-basis outputs, OOF model binaries, and reconstructed E2 copies. It never removes state, decisions, selected anchor rows, selected residual predictions, selected chain configs, or logs.

Stop persisting research-phase model binaries after their predictions and result metadata are safely written. Keep `_save_model` available for future accepted full-fit delivery only.

- [ ] **Step 4: Make resume and review retention defensive**

`create_s4_resume` excludes all research job model suffixes, prior bundles, temporary files, and reproducible E2 copies. `create_s4_review` retains state, decisions, diagnostics, selected configs, log, and confirmation evidence while excluding rejected-candidate raw tables.

- [ ] **Step 5: Run production, artifact, runner, and recovery tests**

- [ ] **Step 6: Commit**

```bash
git add experiments/tree_expert/s4_production.py experiments/tree_expert/s4_artifacts.py tests/test_tree_expert_s4_artifacts.py tests/test_tree_expert_s4_runner.py tests/test_tree_expert_s4_recovery.py
git commit -m "fix: bound S4 runtime artifact storage"
```

### Task 5: Make Snapshot Creation Disk-safe

**Files:**
- Modify: `experiments/tree_expert/s4_artifacts.py`
- Modify: `experiments/tree_expert/s4_runner.py`
- Modify: `tests/test_tree_expert_s4_artifacts.py`
- Modify: `tests/test_tree_expert_s4_runner.py`

- [ ] **Step 1: Write failing streaming and low-space tests**

Assert artifact writers accept file-backed payloads without calling `Path.read_bytes()` for large members. Simulate low free space and assert a periodic snapshot logs `S4_SNAPSHOT_SKIPPED reason=insufficient_space` while preserving the previous handoff and continuing jobs. Assert final failure attempts exactly one compact emergency handoff.

- [ ] **Step 2: Confirm the tests fail**

- [ ] **Step 3: Implement streaming writers and disk guards**

Change the payload writer to accept `bytes | Path`, calculate evidence by streaming, and copy file-backed members through `ZipFile.open`. Add a peak-space estimator using current review/resume sizes and `shutil.disk_usage`. Periodic snapshots below the reserve are skipped; finalization returns an explicit artifact error if no compact handoff can be written. Temporary nested bundles are deleted after a committed outer handoff.

Log:

```text
S4_DISK_STATUS free_bytes=... estimated_peak_bytes=...
S4_SNAPSHOT_READY path=... size_bytes=...
S4_SNAPSHOT_SKIPPED reason=insufficient_space
```

- [ ] **Step 4: Run artifact and runner tests**

- [ ] **Step 5: Commit**

```bash
git add experiments/tree_expert/s4_artifacts.py experiments/tree_expert/s4_runner.py tests/test_tree_expert_s4_artifacts.py tests/test_tree_expert_s4_runner.py
git commit -m "fix: make S4 snapshots disk safe"
```

### Task 6: Deliver the Local Tool and One-cell Recovery Runtime

**Files:**
- Create: `tools/prepare_tree_s4_recovery_input.py`
- Modify: `experiments/tree_expert/KAGGLE_S4_CELL.py`
- Modify: `experiments/tree_expert/s4_kaggle.py`
- Create: `docs/TREE_S4_RECOVERY.md`
- Create: `tests/test_tree_expert_s4_recovery_tool.py`
- Modify: `tests/test_tree_expert_s4_runbook.py`
- Modify: `tests/test_tree_expert_s4_kaggle.py`

- [ ] **Step 1: Write failing CLI, cell, and runbook tests**

Assert the CLI refuses overwrite, prints source verification and retained/dropped sizes, and writes no submission artifact. Assert the committed cell equals the renderer, remains below one megabyte, contains recovery discovery and disk logs, and contains no browser download loop or submission entry point.

- [ ] **Step 2: Confirm the delivery tests fail**

- [ ] **Step 3: Implement the CLI**

```python
output = compact_recovery_handoff(
    args.source_handoff,
    args.output,
    destination_code_sha256=runtime_identity_sha256(ROOT),
)
print(
    "TREE_S4_RECOVERY_INPUT_READY "
    f"path={output.path.resolve()} sha256={output.sha256} "
    f"retained_bytes={output.retained_bytes} dropped_bytes={output.dropped_bytes}"
)
```

- [ ] **Step 4: Update and regenerate the one-cell runtime**

The cell prints the new code identity, recovery source identity, restored phase and completed-job count before any GPU job starts. It uses the same final handoff path and the same acceptance gates as the original S4 campaign.

- [ ] **Step 5: Write the user runbook**

Document the exact local command, required free local space, expected duration, no-overwrite rerun behavior, Kaggle Add Input layout, T4 ×2 Save Version steps, expected first pending job, disk logs, success/error text, and which handoff to return.

- [ ] **Step 6: Run delivery tests and commit**

```bash
git add tools/prepare_tree_s4_recovery_input.py experiments/tree_expert/KAGGLE_S4_CELL.py experiments/tree_expert/s4_kaggle.py docs/TREE_S4_RECOVERY.md tests/test_tree_expert_s4_recovery_tool.py tests/test_tree_expert_s4_runbook.py tests/test_tree_expert_s4_kaggle.py
git commit -m "feat: deliver S4 compact recovery workflow"
```

### Task 7: Full Static and Synthetic Verification

**Files:**
- Modify only if a verification failure identifies an S4 recovery defect.

- [ ] **Step 1: Run the complete S4 suite**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert_s4_*.py -q`

- [ ] **Step 2: Run the complete Tree Expert suite**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_tree_expert*.py -q`

- [ ] **Step 3: Run competition-rule gates**

Run: `artifacts/tabm_submission_python311/bin/python -m pytest tests/test_repository_contract.py tests/test_rules_code_gate.py tests/test_rules_entrypoints.py tests/test_rules_policy.py tests/test_rules_repository_enforcement.py -q`

- [ ] **Step 4: Verify isolated runtime import and generated cell**

Extract `_runtime_archive` into a temporary directory and import `s4_recovery`, `s4_production`, `s4_runner`, and `s4_kaggle` under Python isolated mode. Confirm `KAGGLE_S4_CELL.py` is byte-identical to `build_s4_kaggle_cell`, below one megabyte, and contains no submission builder.

- [ ] **Step 5: Run source and repository checks**

Run compileall, `git diff --check`, and an S4-scoped forbidden-pattern scan. Confirm unrelated dirty TabM files remain byte-for-byte outside the staged diff.

- [ ] **Step 6: Commit any verification-only correction, then report the user-run handoff**

Do not run the 3.8 GB compaction or Kaggle GPU campaign. Provide one exact local command and one exact Kaggle cell/input procedure. The expected local success prefix is `TREE_S4_RECOVERY_INPUT_READY`; Kaggle must report `S4_RECOVERY_READY`, then begin with pending full-chain job `14`.
