# TabM Submission Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, user-run validation handoff for the frozen Version D TabM candidate without registering a production packager or creating `submit.zip`.

**Architecture:** Refresh the official policy, verify and import the immutable review delivery, render one standalone evaluator script, and validate that exact script with phased row-independence and official-environment checks. A small Kaggle cell consumes the uploaded handoff and official data, emits logs and review/resume ZIPs, then stops before acceptance or leaderboard packaging.

**Tech Stack:** Python 3.11, pytest, pandas, NumPy, PyTorch, `tabm==0.0.3`, `rtdl-num-embeddings==0.0.12`, deterministic ZIP/JSON, Kaggle T4.

---

## File map

- `competition_rules/policy.json`: current rules plus the official FAQ source.
- `competition_rules/contract.py`: strict policy identity and source contract.
- `competition_rules/evidence_gate.py`: schema-2 phased independence evidence.
- `submission/tabm_candidate.py`: nested-delivery verification, immutable import, and deterministic script rendering.
- `submission/tabm_version_d_script.py`: self-contained evaluator source with one metadata sentinel.
- `submission/tabm_validation.py`: environment, full-scale output, capacity, and independence validation.
- `tools/prepare_tabm_submission_validation_handoff.py`: non-submission upload handoff builder.
- `experiments/tabm_campaign/KAGGLE_SUBMISSION_VALIDATION_CELL.py`: one copyable Kaggle cell.
- `docs/TABM_SUBMISSION_VALIDATION.md`: user-run instructions and return contract.
- `tests/test_rules_policy.py`: refreshed-policy tests.
- `tests/test_tabm_submission_candidate.py`: delivery and renderer tests.
- `tests/test_tabm_submission_script.py`: evaluator fixture tests.
- `tests/test_tabm_submission_validation.py`: phased evidence and review-bundle tests.
- `tests/test_tabm_submission_handoff.py`: handoff and cell tests.

The production adapter registry remains empty throughout this plan. No task
calls `build_submission_package`, creates `submit.zip`, or writes a passed
acceptance record.

### Task 1: Refresh the official policy contract

**Files:**
- Modify: `competition_rules/policy.json`
- Modify: `competition_rules/contract.py`
- Modify: `tests/test_rules_policy.py`

- [ ] **Step 1: Write failing policy tests**

```python
def test_policy_includes_current_faq_and_no_final_selection() -> None:
    policy = load_policy(ROOT / "competition_rules/policy.json", project_root=ROOT)
    assert policy["policy_version"] == "dacon-236743-2026-08-15"
    assert policy["leaderboard_selection"] == "highest_compliant_submission"
    assert any(
        source["url"].endswith("/talkboard/417082?page=1&dtype=recent")
        for source in policy["official_sources"]
    )


def test_old_policy_version_is_rejected(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "competition_rules/policy.json").read_text())
    payload["policy_version"] = "dacon-236743-2026-08-13"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RulesContractError, match="unknown policy_version"):
        load_policy(path, project_root=tmp_path)
```

- [ ] **Step 2: Run the test and verify failure**

Run: `.venv/bin/pytest tests/test_rules_policy.py -q`

Expected: FAIL because the existing contract knows only the older policy and
six official sources.

- [ ] **Step 3: Implement the new strict policy**

Add `leaderboard_selection` to `_POLICY_KEYS`; require version
`dacon-236743-2026-08-15`, seven official sources, and this exact value:

```python
if policy["leaderboard_selection"] != "highest_compliant_submission":
    raise RulesContractError("leaderboard_selection is invalid")
```

Add the FAQ URL to `policy.json` and update the policy version. Keep all
runtime, size, data-source, and independence limits unchanged.

- [ ] **Step 4: Verify and commit**

Run:

```bash
.venv/bin/pytest tests/test_rules_policy.py tests/test_rules_repository_enforcement.py -q
git add competition_rules/policy.json competition_rules/contract.py tests/test_rules_policy.py
git commit -m "rules: refresh DACON submission policy"
```

Expected: tests PASS and only the three listed files are committed.

### Task 2: Verify and import the frozen delivery

**Files:**
- Create: `submission/tabm_candidate.py`
- Create: `tests/test_tabm_submission_candidate.py`

- [ ] **Step 1: Write delivery fixture and rejection tests**

```python
def test_import_is_hash_bound_and_exclusive(tmp_path: Path) -> None:
    delivery = make_delivery_fixture(tmp_path)
    result = import_review_delivery(delivery, tmp_path / "candidate")
    assert result.candidate_id == "tabm_hand_matchup_version_d_seed3407_v1"
    assert sorted(result.member_sha256) == [
        "inference_manifest.json",
        "numeric_embedding_0.json",
        "preprocessing_state.json",
        "tabm_member_0_seed_3407.pt",
    ]
    with pytest.raises(TabMCandidateError, match="destination already exists"):
        import_review_delivery(delivery, tmp_path / "candidate")


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("outer_extra", "outer delivery member set"),
        ("inner_hash", "review bundle member SHA-256"),
        ("status", "review_ready"),
        ("epochs", "three epochs"),
        ("seed", "seed 3407"),
        ("fit_scope", "fit scope"),
        ("optimizer", "training state"),
    ],
)
def test_import_rejects_untrusted_delivery(tmp_path, mutation, message) -> None:
    delivery = make_delivery_fixture(tmp_path, mutation=mutation)
    with pytest.raises(TabMCandidateError, match=message):
        import_review_delivery(delivery, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()
```

- [ ] **Step 2: Run the test and verify failure**

Run: `.venv/bin/pytest tests/test_tabm_submission_candidate.py -q`

Expected: collection FAIL because the candidate module does not exist.

- [ ] **Step 3: Implement immutable import contracts**

```python
@dataclass(frozen=True)
class ImportedTabMCandidate:
    candidate_id: str
    root: Path
    model_dir: Path
    delivery_sha256: str
    review_bundle_sha256: str
    model_sha256: str
    member_sha256: Mapping[str, str]


def import_review_delivery(
    delivery_path: str | Path,
    destination: str | Path,
) -> ImportedTabMCandidate:
    """Verify two ZIP layers, then atomically publish four frozen files."""
```

Reject duplicate or unsafe names, symlinks, unexpected members, non-finite or
duplicate-key JSON, member-hash differences, expanded content above 32 GB,
status other than `review_ready`, version other than D, epochs other than three,
seed other than 3407, non-constant scheduler, wrong fit scope, and training
state. Compute the model digest exactly like `submission.audit._model_snapshot`:

```python
digest = sha256()
for name, value in sorted(model_members.items()):
    digest.update(name.encode("utf-8") + b"\0" + sha256(value).digest())
```

Publish a verified temporary sibling directory with `os.replace`; never
overwrite the destination.

- [ ] **Step 4: Verify and commit**

```bash
.venv/bin/pytest tests/test_tabm_submission_candidate.py -q
git add submission/tabm_candidate.py tests/test_tabm_submission_candidate.py
git commit -m "feat: verify frozen TabM review delivery"
```

Expected: tests PASS.

### Task 3: Render a standalone evaluator without enabling packaging

**Files:**
- Create: `submission/tabm_version_d_script.py`
- Modify: `submission/tabm_candidate.py`
- Create: `tests/test_tabm_submission_script.py`

- [ ] **Step 1: Write renderer, registry, and source-gate tests**

```python
def test_script_is_deterministic_parseable_and_unregistered(candidate) -> None:
    first = render_validation_script(candidate)
    assert first == render_validation_script(candidate)
    ast.parse(first.decode("utf-8"))
    assert read_embedded_metadata(first)["model_sha256"] == candidate.model_sha256
    with pytest.raises(AdapterRegistryError, match="not registered"):
        resolve_adapter_factory(candidate.candidate_id)


def test_script_passes_source_gate(tmp_path: Path, candidate) -> None:
    source = tmp_path / "script.py"
    source.write_bytes(render_validation_script(candidate))
    assert inspect_inference_source([source], project_root=tmp_path)["status"] == "passed"
```

- [ ] **Step 2: Run the test and verify failure**

Run: `.venv/bin/pytest tests/test_tabm_submission_script.py -q`

Expected: FAIL because the source and renderer do not exist.

- [ ] **Step 3: Implement deterministic rendering**

The standalone source contains exactly one valid sentinel:

```python
EMBEDDED_METADATA = None
```

The renderer replaces it once with canonical JSON and compiles the result:

```python
def render_validation_script(candidate: ImportedTabMCandidate) -> bytes:
    source = SCRIPT_SOURCE.read_bytes()
    sentinel = b"EMBEDDED_METADATA = None"
    if source.count(sentinel) != 1:
        raise TabMCandidateError("script metadata sentinel differs")
    payload = canonical_json({
        "candidate_id": candidate.candidate_id,
        "model_sha256": candidate.model_sha256,
        "members": dict(candidate.member_sha256),
    }).decode("utf-8").strip()
    replacement = ("EMBEDDED_METADATA = json.loads(" + repr(payload) + ")").encode()
    rendered = source.replace(sentinel, replacement)
    compile(rendered, "script.py", "exec")
    return rendered
```

- [ ] **Step 4: Implement only the accepted frozen runtime**

The standalone source exports exactly these runtime names:

```python
__all__ = (
    "FrozenTabMPredictor",
    "load_frozen_predictor",
    "validate_inputs",
    "predict_frame",
    "write_submission",
    "main",
)
```

Implement the bodies directly rather than importing repository code. Require:

```python
if state["view"] != "raw_typed" or state["trackman"] is not None:
    raise EvaluatorError("frozen feature view differs")
if state["preprocessing"]["spec"] != {
    "profile": "dl_standard",
    "components": ["hand_matchup"],
}:
    raise EvaluatorError("frozen preprocessing spec differs")
```

Transform the state-declared schema, add only `hand_matchup`, use frozen
imputation/standardization/category maps, and map OOV categories to zero. Build
PiecewiseLinearEmbeddings and TabM from the frozen manifest, use the same
`model` wrapper attribute as training, load `weights_only=True`, and average the
internal-k sigmoid probabilities. Do not import SciPy or carry unused training
components. Normal `main()` execution requires CUDA. The validation runner may
call `load_frozen_predictor` with its `device` parameter set to `"cpu"` only for
the isolated official-version compatibility probe; submitted `main()` never
selects it.

- [ ] **Step 5: Add fail-closed fixture execution tests**

```python
@pytest.mark.parametrize(
    "case,message",
    [
        ("duplicate_id", "unique"),
        ("missing_id", "exactly match"),
        ("schema", "source schema"),
        ("nan_probability", "finite"),
        ("out_of_range", "inside"),
        ("existing_output", "already exists"),
    ],
)
def test_failure_never_publishes_partial_output(tmp_path, case, message) -> None:
    sandbox = make_script_sandbox(tmp_path, mutation=case)
    completed = run_script_fixture(sandbox)
    assert completed.returncode != 0
    assert message in completed.stderr
    assert not (sandbox / "output/submission.csv").exists()
```

Also test exact `data/` input paths, exact sample order, eight-decimal finite
probabilities, artifact hashes, and atomic output publication. The fake model is
injected only by the test harness and cannot activate in normal execution.

- [ ] **Step 6: Verify and commit**

```bash
.venv/bin/pytest tests/test_tabm_submission_script.py tests/test_submission_runtime.py tests/test_rules_code_gate.py -q
git add submission/tabm_candidate.py submission/tabm_version_d_script.py tests/test_tabm_submission_script.py
git commit -m "feat: render frozen TabM validation runtime"
```

Expected: tests PASS and the production adapter registry is still empty.

### Task 4: Add bounded phased independence evidence

**Files:**
- Modify: `competition_rules/evidence_gate.py`
- Create: `submission/tabm_validation.py`
- Create: `tests/test_tabm_submission_validation.py`

- [ ] **Step 1: Write schema-2 resume and rejection tests**

```python
def test_phased_audit_resumes_verified_phases(tmp_path: Path) -> None:
    first = run_phased_independence_audit(
        frame=fixture_frame(19), output_dir=tmp_path / "audit",
        identity=fixture_identity(), load_predictor=fixture_predictor,
        singleton_count=5, stop_after_phases=2,
    )
    assert first.status == "incomplete"
    second = run_phased_independence_audit(
        frame=fixture_frame(19), output_dir=tmp_path / "audit",
        identity=fixture_identity(), load_predictor=fixture_predictor,
        singleton_count=5,
    )
    assert second.status == "passed" and second.reused_phases == 2
    assert validate_full_audit(
        second.manifest_path, expected_identity=fixture_identity()
    )["schema_version"] == 2


def test_phased_audit_rejects_batch_dependence(tmp_path: Path) -> None:
    with pytest.raises(RulesEvidenceError, match="row independence mismatch"):
        run_phased_independence_audit(
            frame=fixture_frame(19), output_dir=tmp_path / "audit",
            identity=fixture_identity(), load_predictor=batch_dependent_predictor,
            singleton_count=5,
        )
```

- [ ] **Step 2: Run the test and verify failure**

Run: `.venv/bin/pytest tests/test_tabm_submission_validation.py -q`

Expected: FAIL because schema-2 audit functions do not exist.

- [ ] **Step 3: Implement six fixed phases**

```python
PHASES = (
    "baseline", "reverse", "shuffle",
    "batch_257", "batch_2048", "singleton_canaries",
)
```

Run the production phase audit on all five locally supplied official sample
rows and use every one of them as a singleton. Each exclusive canonical phase
JSON records identity, phase, selected
ID digest, feature digest, probability digest, state digest before/after,
elapsed seconds, status, and its own hash. Reuse requires all fields and hashes
to match. Compare all row-aligned values against baseline before publishing a
schema-2 `full_audit_manifest.json`. Preserve schema-1 validation unchanged.

- [ ] **Step 4: Verify and commit**

```bash
.venv/bin/pytest tests/test_row_independence_evidence.py tests/test_tabm_submission_validation.py -q
git add competition_rules/evidence_gate.py submission/tabm_validation.py tests/test_tabm_submission_validation.py
git commit -m "feat: add bounded TabM independence evidence"
```

Expected: both evidence schemas PASS.

### Task 5: Validate official-environment compatibility and capacity

**Files:**
- Modify: `submission/tabm_validation.py`
- Modify: `tests/test_tabm_submission_validation.py`

- [ ] **Step 1: Write exact environment and threshold tests**

```python
def test_exact_probe_requires_python311_and_torch271() -> None:
    with pytest.raises(TabMValidationError, match="Python 3.11"):
        validate_exact_probe(fixture_probe(python="3.12.13"))
    with pytest.raises(TabMValidationError, match="PyTorch 2.7.1"):
        validate_exact_probe(
            fixture_probe(python="3.11.15", torch="2.11.0+cpu")
        )


def test_exact_and_gpu_probe_predictions_must_agree() -> None:
    exact = fixture_probe(probabilities=[0.4, 0.5])
    gpu = fixture_probe(probabilities=[0.4, 0.5001])
    with pytest.raises(TabMValidationError, match="probe predictions"):
        compare_probe_predictions(exact, gpu, tolerance=1e-6)


def test_runtime_above_safety_threshold_is_rejected(tmp_path: Path) -> None:
    report = make_validation_report(tmp_path, inference_seconds=481.0)
    with pytest.raises(TabMValidationError, match="runtime safety"):
        validate_report(report)
```

- [ ] **Step 2: Run the test and verify failure**

Run: `.venv/bin/pytest tests/test_tabm_submission_validation.py -q`

Expected: FAIL because environment and review contracts are incomplete.

- [ ] **Step 3: Implement the validation CLI**

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return run_validation(parser.parse_args(argv))
```

Create an isolated Python 3.11.15 environment and install CPU PyTorch 2.7.1,
pandas 2.0.3, NumPy 1.26.4, `tabm==0.0.3`, and
`rtdl-num-embeddings==0.0.12`. In that environment, deserialize the actual
checkpoint and record a deterministic five-row CPU prediction. This base setup
is validation-only and is not counted as submitted-requirements install time.

Record the actual GPU host versions separately. Render and bind the exact
script, run the same five-row probe on the T4, and require probabilities to
match the exact CPU probe within `1e-6`. Then run the phased audit and execute
the script in an isolated sandbox with `data/` paths. Verify the five-row CSV
against sample schema, IDs, order, row count, and probability bounds.

Build a deterministic 245,789-row capacity frame by repeating the five official
sample rows and replacing only `row_id` with unique `SCALE_000001`-style IDs.
Run the exact script against a matching synthetic sample submission to measure
time, RAM, and VRAM. Record `audit_scope="official_sample_plus_synthetic_scale"`
and never call this hidden-evaluation evidence. Measure the two submitted
requirements separately after the exact base environment exists.

Construct one `AuditIdentity` before either probe. Bind the refreshed policy
version, SHA-256 of the official test and sample inputs, delivery lineage,
validation source/config, frozen preprocessing, combined model directory,
standalone adapter source, and rendered script into `data_sha256`,
`code_sha256`, `config_sha256`, `preprocessing_sha256`, `model_sha256`,
`adapter_sha256`, and `runtime_sha256`. Every phase, probe, report, and bundle
manifest must carry this exact mapping.

Measure monotonic wall time, `resource.getrusage` peak RSS, and
`torch.cuda.max_memory_allocated`. Reject installation or inference over 480
seconds, RAM at or above 22 GiB, allocated VRAM at or above 20 GiB, and size
projections above official limits.

- [ ] **Step 4: Implement deterministic review and resume bundles**

Review bundle members are `validation_report.json`, six phase JSON files and
their manifest, `environment.json`, `install.log`, `validation.log`, and
`manifest.json`. The resume bundle additionally contains hash-bound NumPy
prediction shards under `audit_state/` so interrupted later phases can compare
against the exact baseline without recomputing completed work. Resume shards
are evidence-only: the handoff verifier rejects them from `candidate/model/`,
the production registry stays closed, and the eventual packager never receives
the resume directory. The resume contains no model or official input CSV. All
ZIP timestamps are fixed and every member hash is bound.

Required logs are:

```text
VALIDATION_CODE_READY candidate=<id> runtime_sha256=<sha256>
VALIDATION_INPUTS_VERIFIED rows=245789 model_sha256=<sha256>
VALIDATION_ENVIRONMENT_READY python=3.11.15 torch=2.7.1+cu128 gpu=<name>
VALIDATION_PROGRESS phase=<phase> status=<started|completed> elapsed_seconds=<seconds>
VALIDATION_OUTPUT_VERIFIED rows=245789 output_sha256=<sha256>
VALIDATION_SUCCESS review=<path> resume=<path>
VALIDATION_ERROR stage=<stage> type=<type> message=<message>
```

- [ ] **Step 5: Verify and commit**

```bash
.venv/bin/pytest tests/test_tabm_submission_validation.py -q
git add submission/tabm_validation.py tests/test_tabm_submission_validation.py
git commit -m "feat: validate TabM evaluator compatibility"
```

Expected: fixture tests PASS; no local package installation or full GPU run.

### Task 6: Build the upload handoff and copyable Kaggle cell

**Files:**
- Create: `tools/prepare_tabm_submission_validation_handoff.py`
- Create: `experiments/tabm_campaign/KAGGLE_SUBMISSION_VALIDATION_CELL.py`
- Create: `tests/test_tabm_submission_handoff.py`

- [ ] **Step 1: Write handoff and cell tests**

```python
def test_handoff_is_validation_only(tmp_path: Path) -> None:
    result = prepare_handoff(make_delivery_fixture(tmp_path), tmp_path / "handoff")
    assert (result.root / "handoff_manifest.json").is_file()
    assert (result.root / "candidate/model/inference_manifest.json").is_file()
    assert not list(result.root.rglob("submit.zip"))


def test_cell_is_one_offline_copyable_program() -> None:
    path = ROOT / "experiments/tabm_campaign/KAGGLE_SUBMISSION_VALIDATION_CELL.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    assert "github.com" not in source
    assert "VALIDATION_SUCCESS" in source
    assert "VALIDATION_ERROR" in source
```

- [ ] **Step 2: Run the test and verify failure**

Run: `.venv/bin/pytest tests/test_tabm_submission_handoff.py -q`

Expected: FAIL because the tool and cell do not exist.

- [ ] **Step 3: Implement handoff preparation**

CLI arguments are `--delivery` and `--output-dir`. Import the delivery into
`candidate/`; copy `competition_rules/contract.py`, `code_gate.py`,
`evidence_gate.py`, `policy.json`, and the three TabM validation modules; write
the two exact requirements; and hash every member in `handoff_manifest.json`.
Do not include official data or create a ZIP. Print:

```text
TABM_SUBMISSION_VALIDATION_HANDOFF_READY root=<path> manifest_sha256=<sha256> model_sha256=<sha256>
```

- [ ] **Step 4: Implement the Kaggle cell**

Find exactly one verified handoff manifest under `/kaggle/input` and exactly one
official-data directory containing `test.csv` plus `sample_submission.csv`.
With Kaggle Internet enabled, install an isolated CPython 3.11.15 runtime, the
official-version CPU compatibility dependencies, and the two submitted
requirements; do not replace or restart the GPU kernel. Stream validation logs
and request downloads of review and resume ZIPs. Do not access GitHub, mount
Drive, restart the runtime, or create a leaderboard archive. Preserve a
completed resume ZIP when an error occurs.

- [ ] **Step 5: Verify and commit**

```bash
.venv/bin/pytest tests/test_tabm_submission_handoff.py tests/test_rules_entrypoints.py -q
git add tools/prepare_tabm_submission_validation_handoff.py experiments/tabm_campaign/KAGGLE_SUBMISSION_VALIDATION_CELL.py tests/test_tabm_submission_handoff.py
git commit -m "feat: hand off TabM submission validation"
```

Expected: tests PASS and repository enforcement finds no alternate package
writer.

### Task 7: Document and build the lightweight handoff

**Files:**
- Create: `docs/TABM_SUBMISSION_VALIDATION.md`
- Generate, do not commit: `artifacts/tabm_submission_validation_handoff/`

- [ ] **Step 1: Write the user-run contract**

Document purpose, required uploads, one T4 accelerator, approximate 15–40
minute runtime, phase-resume behavior, all success/error lines, and the exact
files to return. The user returns
`tabm_submission_validation_review_bundle.zip` plus the final log; on failure,
the user also returns the resume ZIP.

- [ ] **Step 2: Run all lightweight verification**

```bash
.venv/bin/pytest tests/test_rules_policy.py tests/test_rules_code_gate.py tests/test_row_independence_evidence.py tests/test_submission_runtime.py tests/test_submission_package.py tests/test_tabm_submission_candidate.py tests/test_tabm_submission_script.py tests/test_tabm_submission_validation.py tests/test_tabm_submission_handoff.py -q
python3 -m compileall -q competition_rules submission tools/prepare_tabm_submission_validation_handoff.py experiments/tabm_campaign/KAGGLE_SUBMISSION_VALIDATION_CELL.py
git diff --check
```

Expected: all tests PASS, compilation exits zero, and diff check prints nothing.

- [ ] **Step 3: Build and verify the upload directory**

```bash
.venv/bin/python tools/prepare_tabm_submission_validation_handoff.py \
  --delivery "/Users/yonghyun/Downloads/tabm_hand_matchup_stage_D_review_delivery (1).zip" \
  --output-dir artifacts/tabm_submission_validation_handoff
.venv/bin/python - <<'PY'
from pathlib import Path
from submission.adapters import ADAPTER_FACTORIES
root = Path("artifacts/tabm_submission_validation_handoff")
assert root.is_dir() and not list(root.rglob("submit.zip"))
assert "tabm_hand_matchup_version_d_seed3407_v1" not in ADAPTER_FACTORIES
print("TABM_VALIDATION_HANDOFF_VERIFIED")
PY
```

Expected: the handoff-ready line followed by
`TABM_VALIDATION_HANDOFF_VERIFIED`. This is only file verification/copying; it
does not install packages, use the GPU, or run full data.

- [ ] **Step 4: Commit documentation only**

```bash
git add docs/TABM_SUBMISSION_VALIDATION.md
git commit -m "docs: explain TabM submission validation"
```

## Stop condition

Stop after giving the user the upload directory and copyable cell. The user runs
the full Kaggle GPU validation and returns the review ZIP and log. Only after
those exact artifacts pass review may a new plan create acceptance evidence,
register the production adapter, connect the renderer to the sole package
writer, or create `submit.zip`.
