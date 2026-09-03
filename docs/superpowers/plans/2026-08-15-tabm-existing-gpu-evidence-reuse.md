# TabM Existing GPU Evidence Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the hash-matched Stage C/D Tesla T4 evidence, validate the final evaluator on the official five-row sample under Python 3.11, and create one fail-closed DACON `submit.zip` without another GPU run.

**Architecture:** A verifier reads the existing Stage C and Stage D deliveries, validates their nested manifests and binds temporal performance, model files, T4 capacity, and row-independence evidence to the current candidate. The final evaluator is registered with the existing sole packager, a Python 3.11 CPU probe proves five-row parity with the Stage D GPU probe, and the existing audit/package gates create the archive only after all identities match.

**Tech Stack:** Python 3.11, pytest, pandas, NumPy, PyTorch, `tabm==0.0.3`, `rtdl-num-embeddings==0.0.12`, standard-library JSON/ZIP/SHA-256, existing DACON policy and package gates.

---

## File map

- `submission/tabm_candidate.py`: expose strictly verified Stage D review evidence in addition to the frozen model.
- `submission/tabm_existing_evidence.py`: combine Stage C performance, Stage D T4 capacity, and current-runtime sample evidence.
- `submission/adapters.py`: register only the reviewed Version D adapter.
- `submission/runtime.py`: dispatch final script rendering for that registered adapter.
- `submission/package.py`: bind model-member hashes into the rendered script.
- `tools/build_tabm_submission_from_existing_evidence.py`: one fail-closed local entry point.
- `tests/test_tabm_existing_evidence.py`: evidence parsing, parity, and rejection tests.
- `tests/test_tabm_submission_script.py`: registered adapter and rendered-package tests.
- `tests/test_submission_package.py`: production TabM package layout and tamper rejection.
- `docs/TABM_SUBMISSION_VALIDATION.md`: document the no-new-GPU route and exact output contract.

### Task 1: Verify Stage C and Stage D evidence as typed inputs

**Files:**
- Modify: `submission/tabm_candidate.py`
- Create: `submission/tabm_existing_evidence.py`
- Create: `tests/test_tabm_existing_evidence.py`

- [ ] **Step 1: Write failing tests for trusted evidence**

```python
def test_verified_evidence_binds_stage_c_d_model_and_t4(tmp_path: Path) -> None:
    stage_c, stage_d, candidate = make_evidence_fixture(tmp_path)
    evidence = verify_existing_gpu_evidence(
        stage_c_delivery=stage_c,
        stage_d_delivery=stage_d,
        candidate=candidate,
    )
    assert evidence.candidate_id == CANDIDATE_ID
    assert evidence.model_sha256 == candidate.model_sha256
    assert evidence.gpu_name == "Tesla T4"
    assert evidence.capacity_rows == 245_789
    assert evidence.inference_seconds == pytest.approx(11.938824568999735)
    assert evidence.primary_brier == pytest.approx(0.24811080225115084)
    assert evidence.older_brier == pytest.approx(0.25086573594721084)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("stage_c_sha", "Stage C delivery SHA-256"),
        ("model_member", "model SHA-256"),
        ("gpu", "Tesla T4"),
        ("rows", "245789"),
        ("independence", "row independence"),
        ("predictor", "single_s3407"),
    ],
)
def test_existing_evidence_rejects_mismatch(
    tmp_path: Path, mutation: str, message: str
) -> None:
    stage_c, stage_d, candidate = make_evidence_fixture(tmp_path, mutation=mutation)
    with pytest.raises(ExistingEvidenceError, match=message):
        verify_existing_gpu_evidence(
            stage_c_delivery=stage_c,
            stage_d_delivery=stage_d,
            candidate=candidate,
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_tabm_existing_evidence.py -q
```

Expected: collection fails because `submission.tabm_existing_evidence` does not exist.

- [ ] **Step 3: Add the typed evidence contract**

```python
@dataclass(frozen=True)
class ExistingGpuEvidence:
    candidate_id: str
    stage_c_sha256: str
    stage_d_sha256: str
    model_sha256: str
    model_members: Mapping[str, str]
    gpu_name: str
    capacity_rows: int
    inference_seconds: float
    peak_ram_bytes: int
    peak_vram_bytes: int
    independence_delta: float
    install_seconds: float
    primary_brier: float
    older_brier: float
    gpu_sample_probabilities: tuple[float, ...]


def verify_existing_gpu_evidence(
    *,
    stage_c_delivery: str | Path,
    stage_d_delivery: str | Path,
    candidate: ImportedTabMCandidate,
) -> ExistingGpuEvidence:
    """Verify both deliveries and return only hash-bound accepted fields."""
```

The verifier must reject duplicate ZIP names, paths containing `..`, symlinks,
undeclared members, member-size/hash differences, non-finite JSON, and duplicate
JSON keys. Require these exact conditions:

```python
assert sha256(stage_c_bytes).hexdigest() == stage_d_manifest["stage_c_delivery_sha256"]
assert final_review["acceptance_status"] == "review_ready"
assert final_review["fit_report"]["status"] == "completed"
assert stage_c_state["predictor_evidence"][0]["predictor_id"] == "single_s3407"
assert stage_c_state["predictor_evidence"][0]["accepted"] is True
assert stage_d_log contains "VERSION_D_GPU_READY device_count=1 name=Tesla T4"
assert stage_d_log contains "VERSION_D_INDEPENDENCE_PASSED"
assert stage_d_log contains "VERSION_D_SCALE_GATE_PASSED rows=245789"
assert scale["rows"] == 245_789
assert independence["max_abs_probability_delta"] <= 1e-6
assert final_review["artifact_sha256"] == dict(candidate.member_sha256)
assert candidate.model_sha256 == recomputed_model_sha256
```

Parse the five comma-separated values in
`final_review["frozen_sample_probe"]["stdout_tail"]`; require exactly five finite
probabilities in `[0, 1]`.

- [ ] **Step 4: Run GREEN tests and commit**

```bash
.venv/bin/pytest tests/test_tabm_existing_evidence.py tests/test_tabm_submission_candidate.py -q
git add submission/tabm_candidate.py submission/tabm_existing_evidence.py tests/test_tabm_existing_evidence.py
git commit -m "feat: verify existing TabM GPU evidence"
```

Expected: all selected tests pass.

### Task 2: Register and render the reviewed final evaluator

**Files:**
- Modify: `submission/adapters.py`
- Modify: `submission/runtime.py`
- Modify: `submission/package.py`
- Modify: `submission/tabm_candidate.py`
- Modify: `tests/test_tabm_submission_script.py`
- Modify: `tests/test_submission_package.py`

- [ ] **Step 1: Write failing registry and render-parity tests**

```python
def test_version_d_adapter_is_the_only_registered_real_adapter() -> None:
    factory = resolve_adapter_factory(CANDIDATE_ID)
    assert callable(factory)
    assert set(ADAPTER_FACTORIES) == {CANDIDATE_ID}


def test_package_renderer_equals_validated_renderer(candidate) -> None:
    metadata = {
        "candidate_id": candidate.candidate_id,
        "delivery_sha256": candidate.delivery_sha256,
        "review_bundle_sha256": candidate.review_bundle_sha256,
        "model_sha256": candidate.model_sha256,
        "members": dict(candidate.member_sha256),
    }
    assert render_script(adapter_id=CANDIDATE_ID, artifact_metadata=metadata) == (
        render_validation_script(candidate)
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_tabm_submission_script.py tests/test_submission_package.py -q
```

Expected: the registry test fails because the production adapter registry is empty.

- [ ] **Step 3: Add one reviewed adapter and one renderer**

Expose a metadata renderer that shares the same source and sentinel as validation:

```python
def render_bound_script(metadata: Mapping[str, object]) -> bytes:
    expected = {
        "candidate_id", "delivery_sha256", "review_bundle_sha256",
        "model_sha256", "members",
    }
    if set(metadata) != expected or metadata["candidate_id"] != CANDIDATE_ID:
        raise TabMCandidateError("script metadata contract differs")
    members = metadata["members"]
    if not isinstance(members, dict) or set(members) != _MODEL_NAMES:
        raise TabMCandidateError("script model member contract differs")
    for label, value in {
        "delivery_sha256": metadata["delivery_sha256"],
        "review_bundle_sha256": metadata["review_bundle_sha256"],
        "model_sha256": metadata["model_sha256"],
        **members,
    }.items():
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise TabMCandidateError(f"invalid SHA-256: {label}")
    source = _SCRIPT_SOURCE.read_bytes()
    sentinel = b"EMBEDDED_METADATA = None"
    if source.count(sentinel) != 1:
        raise TabMCandidateError("script metadata sentinel differs")
    payload = _canonical_json(dict(metadata)).decode("utf-8").strip()
    rendered = source.replace(
        sentinel,
        ("EMBEDDED_METADATA = json.loads(" + repr(payload) + ")").encode("utf-8"),
    )
    compile(rendered, "script.py", "exec")
    return rendered


def render_validation_script(candidate: ImportedTabMCandidate) -> bytes:
    return render_bound_script(candidate_metadata(candidate))
```

Register the real factory without importing PyTorch at module import time:

```python
def _load_version_d(model_dir: Path, metadata: Mapping[str, object]) -> object:
    del metadata
    from .tabm_version_d_script import load_frozen_predictor
    return load_frozen_predictor(model_dir)


ADAPTER_FACTORIES = MappingProxyType({CANDIDATE_ID: _load_version_d})
```

Dispatch `submission.runtime.render_script` only for `CANDIDATE_ID`, and reject
all unknown adapter IDs. Extend the package metadata with exact model-member
hashes, delivery SHA-256, and review-bundle SHA-256 so the packaged script bytes
are identical to the validated script bytes.

- [ ] **Step 4: Verify GREEN and commit**

```bash
.venv/bin/pytest tests/test_tabm_submission_script.py tests/test_submission_package.py -q
git add submission/adapters.py submission/runtime.py submission/package.py submission/tabm_candidate.py tests/test_tabm_submission_script.py tests/test_submission_package.py
git commit -m "feat: register reviewed TabM evaluator"
```

Expected: selected tests pass, and `resolve_adapter_factory("unknown")` still fails.

### Task 3: Compose current-runtime evidence without a GPU run

**Files:**
- Modify: `submission/tabm_existing_evidence.py`
- Modify: `tests/test_tabm_existing_evidence.py`

- [ ] **Step 1: Write failing parity and fail-closed tests**

```python
def test_build_acceptance_combines_current_sample_and_existing_t4(tmp_path, fixture) -> None:
    result = build_reused_gpu_acceptance(
        candidate=fixture.candidate,
        gpu_evidence=fixture.gpu_evidence,
        test_frame=fixture.test,
        sample_frame=fixture.sample,
        runtime_bytes=fixture.runtime,
        output_dir=tmp_path / "evidence",
        load_predictor=fixture.predictor,
        python_probe=fixture.python_probe,
        package_bytes=11_000_000,
        extracted_bytes=11_000_000,
    )
    assert result.acceptance["status"] == "passed"
    assert all(result.acceptance["gates"].values())
    assert result.benchmark["inference_seconds"] == pytest.approx(11.938824568999735)
    validate_full_audit(result.audit_manifest, expected_identity=result.identity)


def test_build_acceptance_rejects_gpu_cpu_probability_drift(tmp_path, fixture) -> None:
    fixture.gpu_evidence = replace(
        fixture.gpu_evidence,
        gpu_sample_probabilities=(0.9, 0.9, 0.9, 0.9, 0.9),
    )
    with pytest.raises(ExistingEvidenceError, match="sample prediction parity"):
        build_reused_gpu_acceptance(
            candidate=fixture.candidate,
            gpu_evidence=fixture.gpu_evidence,
            test_frame=fixture.test,
            sample_frame=fixture.sample,
            runtime_bytes=fixture.runtime,
            output_dir=tmp_path / "evidence",
            load_predictor=fixture.predictor,
            python_probe=fixture.python_probe,
            package_bytes=11_000_000,
            extracted_bytes=11_000_000,
        )
    assert not (tmp_path / "evidence/acceptance.json").exists()
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
.venv/bin/pytest tests/test_tabm_existing_evidence.py -q
```

Expected: failure because `build_reused_gpu_acceptance` is missing.

- [ ] **Step 3: Implement current-runtime audit and evidence publication**

```python
@dataclass(frozen=True)
class ReusedGpuAcceptance:
    identity: AuditIdentity
    audit_manifest: Path
    acceptance_path: Path
    benchmark_path: Path
    acceptance: Mapping[str, object]
    benchmark: Mapping[str, object]


def build_reused_gpu_acceptance(
    *,
    candidate: ImportedTabMCandidate,
    gpu_evidence: ExistingGpuEvidence,
    test_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
    runtime_bytes: bytes,
    output_dir: str | Path,
    load_predictor: Callable[[], object],
    python_probe: Mapping[str, object],
    package_bytes: int,
    extracted_bytes: int,
) -> ReusedGpuAcceptance:
    """Publish acceptance only after sample parity and all existing gates pass."""
```

Compute the identity from canonical hashes of official `test.csv` and
`sample_submission.csv`, final script bytes, fixed preprocessing state, model
directory, adapter source, and requirements. Run
`run_phased_independence_audit` over all five official sample rows with the
current final predictor. Compare baseline predictions to the Stage D GPU sample
values with absolute tolerance `1e-6`.

Require the Python probe to report a Python version beginning with `3.11.`,
`tabm==0.0.3`, and `rtdl-num-embeddings==0.0.12`. Build the benchmark only from
verified Stage D T4 values:

```python
benchmark = {
    "schema_version": 1,
    "status": "passed",
    "candidate_id": identity.candidate_id,
    "adapter_id": CANDIDATE_ID,
    "identity": asdict(identity),
    "install_seconds": gpu_evidence.install_seconds,
    "inference_seconds": gpu_evidence.inference_seconds,
    "peak_ram_bytes": gpu_evidence.peak_ram_bytes,
    "peak_vram_bytes": gpu_evidence.peak_vram_bytes,
    "extracted_bytes": extracted_bytes,
    "package_bytes": package_bytes,
}
```

Set every existing acceptance gate to true only after its evidence is checked:

```python
gates = {
    "temporal_validation": gpu_evidence.primary_brier < 0.25,
    "performance": gpu_evidence.primary_brier < gpu_evidence.older_brier,
    "provenance": model_and_delivery_hashes_match,
    "row_independence": phased_audit_passed and sample_parity_passed,
    "evaluator_runtime": runtime_and_capacity_passed,
    "pretrained_license": True,  # no external pretrained artifact
    "current_rules": policy_review_is_current,
}
```

Write JSON with exclusive creation and `fsync`; do not publish a partially
passing directory. Store `gpu_evidence.json` beside the acceptance so the reused
source hashes and measured values remain reviewable.

- [ ] **Step 4: Verify GREEN and commit**

```bash
.venv/bin/pytest tests/test_tabm_existing_evidence.py tests/test_tabm_submission_validation.py -q
git add submission/tabm_existing_evidence.py tests/test_tabm_existing_evidence.py
git commit -m "feat: bind current TabM runtime to T4 evidence"
```

Expected: selected tests pass, including all tamper and probability-drift cases.

### Task 4: Add the single final build command

**Files:**
- Create: `tools/build_tabm_submission_from_existing_evidence.py`
- Create: `tests/test_tabm_submission_build_tool.py`

- [ ] **Step 1: Write failing end-to-end fixture tests**

```python
def test_tool_creates_exact_submission_and_receipt(tmp_path, fixture) -> None:
    result = run_build(fixture.args(tmp_path))
    assert result.archive_path.name == "submit.zip"
    with ZipFile(result.archive_path) as archive:
        names = archive.namelist()
    assert names[0:2] == ["script.py", "requirements.txt"]
    assert set(name.split("/", 1)[0] for name in names) == {
        "script.py", "requirements.txt", "model"
    }
    assert json.loads(result.receipt_path.read_text())["status"] == "packaged"


def test_tool_never_packages_when_probe_or_hash_fails(tmp_path, fixture) -> None:
    fixture.stage_d.write_bytes(fixture.stage_d.read_bytes() + b"changed")
    with pytest.raises(ExistingEvidenceError):
        run_build(fixture.args(tmp_path))
    assert not (tmp_path / "submit.zip").exists()
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
.venv/bin/pytest tests/test_tabm_submission_build_tool.py -q
```

Expected: collection fails because the tool module does not exist.

- [ ] **Step 3: Implement the fail-closed orchestration**

The command accepts explicit paths and never searches Downloads heuristically:

```text
--stage-c-delivery
--stage-d-delivery
--candidate-root
--official-data
--output-dir
```

`run_build` must perform this order:

```python
candidate = load_imported_candidate(args.candidate_root)
gpu = verify_existing_gpu_evidence(
    stage_c_delivery=args.stage_c_delivery,
    stage_d_delivery=args.stage_d_delivery,
    candidate=candidate,
)
runtime = render_validation_script(candidate)
probe = require_python311_environment()
runtime_module = load_rendered_runtime(runtime)
test_frame, sample_frame = load_official_sample(args.official_data)
acceptance = build_reused_gpu_acceptance(
    candidate=candidate,
    gpu_evidence=gpu,
    test_frame=test_frame,
    sample_frame=sample_frame,
    runtime_bytes=runtime,
    output_dir=output / "evidence",
    load_predictor=lambda: runtime_module.load_frozen_predictor(
        candidate.model_dir, device="cpu"
    ),
    python_probe=probe,
    package_bytes=projected_package_bytes,
    extracted_bytes=projected_extracted_bytes,
)
request = PackageRequest(
    project_root=project_root,
    policy_path=project_root / "competition_rules/policy.json",
    policy_review_path=project_root / "reports/rules/2026-08-15-policy-review.json",
    acceptance_path=acceptance.acceptance_path,
    full_audit_manifest_path=acceptance.audit_manifest,
    runtime_benchmark_path=acceptance.benchmark_path,
    model_dir=candidate.model_dir,
    requirements_path=project_root / "artifacts/tabm_submission_validation_handoff_v2/requirements.txt",
    adapter_id=CANDIDATE_ID,
    archive_path=output / "submit.zip",
    receipt_path=output / "submission_receipt.json",
    package_time=datetime.now(ZoneInfo("Asia/Seoul")),
)
result = build_submission_package(request)
verify_created_package(result.archive_path, expected_runtime=runtime)
return result
```

The command itself must run under Python 3.11. Reject any other Python minor
version and any dependency versions other than `tabm==0.0.3` and
`rtdl-num-embeddings==0.0.12`. It loads the final script with `device="cpu"` and
audits the five official rows. Never install packages inside this tool.

- [ ] **Step 4: Verify GREEN and commit**

```bash
.venv/bin/pytest tests/test_tabm_submission_build_tool.py tests/test_submission_package.py -q
git add tools/build_tabm_submission_from_existing_evidence.py tests/test_tabm_submission_build_tool.py
git commit -m "feat: build hash-gated TabM submission"
```

Expected: the fixture package passes and every injected failure leaves no ZIP.

### Task 5: Run the real lightweight probe and final verification

**Files:**
- Modify: `docs/TABM_SUBMISSION_VALIDATION.md`
- Create at runtime: `artifacts/tabm_submission_version_d/submit.zip`
- Create at runtime: `artifacts/tabm_submission_version_d/submission_receipt.json`

- [ ] **Step 1: Prepare a Python 3.11 CPU environment after explicit approval**

This downloads a CPU PyTorch wheel and is the only nontrivial local setup run.
Before executing, report its expected duration (about 5–15 minutes), network and
disk use (roughly 1–2GB), and that skipping it leaves Python 3.11 compatibility
unverified. After approval, run:

```bash
uv venv --python 3.11.14 artifacts/tabm_submission_python311
uv pip install --python artifacts/tabm_submission_python311/bin/python \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python artifacts/tabm_submission_python311/bin/python \
  pandas==2.0.3 numpy==1.26.4 tabm==0.0.3 rtdl-num-embeddings==0.0.12
```

Expected: all packages install successfully. No training or full-scale inference
runs in this environment.

- [ ] **Step 2: Run the real final build**

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/build_tabm_submission_from_existing_evidence.py \
  --stage-c-delivery /path/to/Downloads/tabm_colab_stage_C_delivery.zip \
  --stage-d-delivery '/path/to/Downloads/tabm_hand_matchup_stage_D_review_delivery.zip' \
  --candidate-root artifacts/tabm_submission_validation_handoff_v2/candidate \
  --official-data /path/to/kaggle-lg-aimers-9th-data-upload \
  --output-dir artifacts/tabm_submission_version_d
```

Expected terminal marker:

```text
SUBMISSION_PACKAGE_READY archive=.../submit.zip receipt=.../submission_receipt.json archive_sha256=<64 hex>
```

- [ ] **Step 3: Run fresh repository and archive verification**

```bash
.venv/bin/pytest -q
.venv/bin/python -m compileall -q competition_rules submission tools tests
git diff --check
unzip -t artifacts/tabm_submission_version_d/submit.zip
unzip -l artifacts/tabm_submission_version_d/submit.zip
```

Expected: all tests pass; compile, diff, and ZIP integrity checks exit zero; the
ZIP contains only `script.py`, `requirements.txt`, and four files below `model/`.

- [ ] **Step 4: Document the final artifact and commit code/docs only**

Update the guide with the archive path, receipt fields, exact submission layout,
the reused Stage D evidence scope, and the fact that the hidden evaluation rows
were not inspected. Do not commit `submit.zip`, model weights, generated evidence,
or the Python environment.

```bash
git add docs/TABM_SUBMISSION_VALIDATION.md
git commit -m "docs: explain final TabM submission artifact"
```

Expected: only source, tests, plans, and documentation are tracked; generated
submission artifacts remain local and ignored.
