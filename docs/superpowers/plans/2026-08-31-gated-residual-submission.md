# Gated Residual G0 Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one hash-gated DACON `submit.zip` from accepted candidate `G0_7628291922`, with a standalone row-independent runtime and complete local packaging evidence.

**Architecture:** A strict importer converts the exact delivery, review, and handoff artifacts into one frozen candidate directory. A package-local inference runtime loads the reviewed E2 anchor, six CatBoost residual models, and frozen feature state; the repository's central packager remains the only ZIP writer. A dedicated evidence builder and command-line tool verify current rules, exact hashes, five-row prediction parity, row independence, environment, runtime projection, and final archive contents before publication.

**Tech Stack:** Python 3.11.15, CatBoost 1.2.10, pandas 2.0.3, NumPy 1.26.4, pytest, standard-library ZIP/JSON/hash utilities.

---

## File map

- Create `reports/rules/2026-08-31-final-policy-review.json`: same-day official-rules evidence.
- Create `submission/gated_residual_candidate.py`: exact artifact verification, safe extraction, frozen candidate manifest, runtime binding.
- Create `submission/gated_residual_script.py`: evaluator entry point and package-local predictor loader.
- Create `submission/gated_residual_existing_evidence.py`: current-rule acceptance, source scan, row-independence audit, benchmark evidence.
- Create `tools/build_gated_residual_submission.py`: final build coordinator and post-build verifier.
- Modify `submission/adapters.py`: register the reviewed G0 adapter.
- Modify `submission/runtime.py`: render the reviewed G0 script template.
- Modify `submission/audit.py`: bind the G0 candidate manifest to live model members.
- Create `tests/test_gated_residual_submission_candidate.py`: importer and tamper tests.
- Create `tests/test_gated_residual_submission_script.py`: runtime identity and independence tests.
- Create `tests/test_gated_residual_submission_evidence.py`: acceptance and gate-failure tests.
- Create `tests/test_gated_residual_submission_build_tool.py`: build orchestration and final-layout tests.

The existing training and campaign modules are not restructured. The final candidate copies only the reviewed inference dependency closure required to restore `feature_state.pkl` and transform a row; those files are scanned as part of the final runtime.

### Task 1: Record the same-day DACON policy review

**Files:**
- Create: `reports/rules/2026-08-31-final-policy-review.json`

- [ ] **Step 1: Write the current review record**

Use the exact current policy digest and the seven sources already registered in
`competition_rules/policy.json`:

```json
{
  "schema_version": 1,
  "policy_version": "dacon-236743-2026-08-15",
  "policy_sha256": "7f118c46b81dec21a0667f2591ab39d536bad01102f8330b43d1e0291d4f78ec",
  "reviewed_at": "2026-08-31T15:08:32+09:00",
  "sources": [
    {"title": "대회 규칙", "url": "https://dacon.io/competitions/official/236743/overview/rules"},
    {"title": "평가 및 코드 제출 안내", "url": "https://dacon.io/competitions/official/236743/overview/evaluation"},
    {"title": "평가 데이터 독립 예측 원칙", "url": "https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent"},
    {"title": "대회 FAQ 및 운영진 답변", "url": "https://dacon.io/competitions/official/236743/talkboard/417082?page=1&dtype=recent"},
    {"title": "데이터 설명", "url": "https://dacon.io/competitions/official/236743/data"},
    {"title": "대회 설명", "url": "https://dacon.io/competitions/official/236743/overview/description"},
    {"title": "부정 제출 및 치팅 행위에 관하여", "url": "https://dacon.io/notice/notice/13"}
  ],
  "verdict": "unchanged"
}
```

- [ ] **Step 2: Validate the review against the policy**

Run:

```bash
artifacts/tabm_submission_python311/bin/python - <<'PY'
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from competition_rules.contract import load_policy, load_policy_review
root = Path.cwd()
policy = load_policy(root / "competition_rules/policy.json", project_root=root)
load_policy_review(
    root / "reports/rules/2026-08-31-final-policy-review.json",
    policy=policy,
    package_time=datetime.now(ZoneInfo("Asia/Seoul")),
)
print("POLICY_REVIEW_OK")
PY
```

Expected: `POLICY_REVIEW_OK`.

- [ ] **Step 3: Commit only the review record**

```bash
git add reports/rules/2026-08-31-final-policy-review.json
git commit -m "docs: record final dacon policy review"
```

### Task 2: Import the accepted G0 artifacts as one frozen candidate

**Files:**
- Create: `submission/gated_residual_candidate.py`
- Create: `tests/test_gated_residual_submission_candidate.py`

- [ ] **Step 1: Write failing importer tests**

The fixture uses small valid bundles and monkeypatches only the three registered
outer hashes. Cover successful import, one-byte tampering, rejected status,
binding disagreement, duplicate/unsafe ZIP names, changed candidate ID, changed
alpha, changed model set, and pre-existing output.

```python
def test_imports_only_exact_accepted_g0_artifacts(tmp_path, monkeypatch):
    delivery, review, handoff = make_g0_artifacts(tmp_path)
    register_fixture_hashes(monkeypatch, delivery, review, handoff)
    candidate = import_gated_residual_candidate(
        delivery=delivery,
        review=review,
        handoff=handoff,
        destination=tmp_path / "candidate",
    )
    assert candidate.candidate_id == "G0_7628291922"
    assert candidate.config == {"alpha": 0.3, "archetype": "G0", "beta": None, "k": None, "ridge": None}
    assert set(candidate.direct_models) == {
        "D0_s42", "D0_s2026", "D0_s3407",
        "D5_s42", "D5_s2026", "D5_s3407",
    }
    assert candidate.root.joinpath("candidate_manifest.json").is_file()

def test_tampered_or_rejected_artifact_publishes_nothing(tmp_path, monkeypatch):
    delivery, review, handoff = make_g0_artifacts(tmp_path, status="rejected")
    register_fixture_hashes(monkeypatch, delivery, review, handoff)
    output = tmp_path / "candidate"
    with pytest.raises(GatedResidualCandidateError):
        import_gated_residual_candidate(delivery, review, handoff, output)
    assert not output.exists()
```

- [ ] **Step 2: Run the tests and confirm the intended failure**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_gated_residual_submission_candidate.py -q
```

Expected: collection fails because `submission.gated_residual_candidate` does not exist.

- [ ] **Step 3: Implement the strict candidate importer**

Define the immutable identity and result contract:

```python
GATED_RESIDUAL_CANDIDATE_ID = "G0_7628291922"
GATED_RESIDUAL_ADAPTER_ID = "gated_residual_g0_v1"
DELIVERY_SHA256 = "dcb4a602318cf787a247653349de0daf777351cdf296b1a78c10380c5b577568"
REVIEW_SHA256 = "e7a651c3c8b96aed13ed9270ce7f78c2f719c42baa3e7ee42ef09edef9a7193a"
HANDOFF_SHA256 = "e14a5558810cd8e5be9ecb76e0cf95abf0abb3d255dcac56bce89e7e852a055c"

@dataclass(frozen=True)
class ImportedGatedResidualCandidate:
    candidate_id: str
    adapter_id: str
    root: Path
    model_dir: Path
    artifact_sha256: Mapping[str, str]
    member_sha256: Mapping[str, str]
    model_sha256: str
    config: Mapping[str, object]
    iterations: Mapping[str, int]
    direct_models: Mapping[str, Path]
    decision: Mapping[str, object]
    evidence: Mapping[str, object]
    audit: Mapping[str, object]
    bindings: Mapping[str, str]
```

`import_gated_residual_candidate()` must:

1. resolve three regular non-symlink files and verify the exact registered SHA-256 values;
2. call the existing artifact verifier with the registered bindings;
3. require the exact delivery/review/handoff kinds and identical bindings;
4. require accepted evidence, `G0_7628291922`, `G0`, `alpha=0.3`, no failed gates, iterations `D0=143`, `D5=78`, and the six exact model roles;
5. verify every delivery member against its manifest before reading it;
6. expand the embedded E2 package into `model/e2/`, copy the six direct models and frozen state, and copy the reviewed inference dependency closure into `model/runtime/`;
7. omit logs, OOF frames, training inputs, campaign checkpoints, and non-runtime files;
8. create `candidate/candidate_manifest.json` beside `candidate/model/`, binding every copied model member by hash without introducing a self-referential digest;
9. publish with an exclusive temporary directory plus `os.replace`.

Use a maximum of 256 archive members and 4 GiB expanded input per artifact,
reject encrypted members, duplicate names, absolute paths, `..`, backslashes,
directories declared as files, and symlinks.

- [ ] **Step 4: Run importer tests**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_gated_residual_submission_candidate.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit importer and tests**

```bash
git add submission/gated_residual_candidate.py \
  tests/test_gated_residual_submission_candidate.py
git commit -m "feat: import accepted gated residual candidate"
```

### Task 3: Add the standalone G0 evaluator runtime

**Files:**
- Create: `submission/gated_residual_script.py`
- Create: `tests/test_gated_residual_submission_script.py`
- Modify: `submission/adapters.py`
- Modify: `submission/runtime.py`

- [ ] **Step 1: Write failing runtime tests**

```python
def test_g0_runtime_is_order_batch_and_singleton_independent(frozen_candidate):
    predictor = load_frozen_predictor(frozen_candidate.model_dir)
    rows = official_five_rows()
    baseline = predictor.predict_batch(rows)
    reverse = rows.iloc[::-1].reset_index(drop=True)
    mapped = dict(zip(reverse.row_id, predictor.predict_batch(reverse), strict=True))
    np.testing.assert_array_equal(baseline, [mapped[row_id] for row_id in rows.row_id])
    for index in range(len(rows)):
        np.testing.assert_array_equal(
            predictor.predict_batch(rows.iloc[[index]]), baseline[[index]]
        )

def test_g0_adapter_is_registered_and_rendered(frozen_candidate):
    assert callable(resolve_adapter_factory("gated_residual_g0_v1"))
    source = render_script(
        adapter_id="gated_residual_g0_v1",
        artifact_metadata=candidate_metadata(frozen_candidate),
    )
    assert b"G0_7628291922" in source
    compile(source, "script.py", "exec")
```

Also test changed model bytes, changed metadata, missing E2 files, wrong sample
schema, duplicate IDs, non-finite output, pre-existing output, and package-local
imports in a subprocess with the repository removed from `PYTHONPATH`.

- [ ] **Step 2: Run the tests and confirm failure**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_gated_residual_submission_script.py -q
```

Expected: failure because the adapter and script template are absent.

- [ ] **Step 3: Implement the runtime template**

The rendered module exposes the fixed names `EMBEDDED_METADATA`,
`GatedResidualPredictor`, `load_frozen_predictor`, `validate_inputs`, and
`main`. `GatedResidualPredictor` exposes `adapter_id`, `state_digest()`, and
`predict_batch(frame, batch_size=4096)`.

`load_frozen_predictor()` verifies the embedded candidate metadata and every
live member hash, inserts only `model/runtime` into `sys.path`, restores the direct
feature state, loads six CatBoost models on CPU, and loads the embedded E2
predictor from `model/e2/script.py`. `predict_batch()` reproduces the reviewed
flow exactly:

```python
direct = np.where(game_type == "R", d5_mean, d0_mean)
strength = (game_type == "R").astype("float64")
result = np.clip(anchor + 0.3 * strength * (direct - anchor), 0.0, 1.0)
```

The final `main()` resolves `open/` first and `data/` second, reads only
`test.csv` and `sample_submission.csv`, validates exact row IDs, invokes the
generic row-independence canaries, and writes only `output/submission.csv`.
It sets CatBoost thread count to at most six, performs no fitting, network
access, subprocess launch, or evaluation-set aggregation.

- [ ] **Step 4: Register and render the reviewed adapter**

Add the adapter ID to `submission/adapters.py`:

```python
def _load_gated_residual(model_dir, metadata):
    from .gated_residual_script import load_frozen_predictor
    return load_frozen_predictor(model_dir, metadata=metadata)

ADAPTER_FACTORIES = MappingProxyType({
    **existing_factories,
    GATED_RESIDUAL_ADAPTER_ID: _load_gated_residual,
})
```

Add one explicit branch to `submission/runtime.py` that calls
`gated_residual_candidate.render_bound_script()` and translates validation
errors into `SubmissionRuntimeError`.

- [ ] **Step 5: Run runtime and existing registry tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_gated_residual_submission_script.py \
  tests/test_submission_runtime.py \
  tests/test_tree_expert_e2_submission_script.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit runtime changes**

```bash
git add submission/gated_residual_script.py submission/gated_residual_candidate.py \
  submission/adapters.py submission/runtime.py \
  tests/test_gated_residual_submission_script.py
git commit -m "feat: add gated residual submission runtime"
```

### Task 4: Bind campaign evidence to the central packager

**Files:**
- Create: `submission/gated_residual_existing_evidence.py`
- Create: `tests/test_gated_residual_submission_evidence.py`
- Modify: `submission/audit.py`

- [ ] **Step 1: Write failing evidence tests**

```python
def test_builds_acceptance_for_exact_g0_candidate(candidate, official_frames, tmp_path):
    result = build_gated_residual_acceptance(
        project_root=Path.cwd(),
        candidate=candidate,
        test_frame=official_frames.test,
        sample_frame=official_frames.sample,
        runtime_bytes=render_validation_script(candidate),
        output_dir=tmp_path / "evidence",
        load_predictor=lambda: load_frozen_predictor(candidate.model_dir),
        python_probe=OFFICIAL_ENVIRONMENT,
        package_bytes=250_000_000,
        extracted_bytes=700_000_000,
        projected_inference_seconds=300.0,
        peak_ram_bytes=4_000_000_000,
        policy_path=Path("competition_rules/policy.json"),
        policy_review_path=Path("reports/rules/2026-08-31-final-policy-review.json"),
        package_time=datetime(2026, 8, 31, 15, 0, tzinfo=ZoneInfo("Asia/Seoul")),
    )
    assert result.acceptance["status"] == "passed"
    assert all(result.acceptance["gates"].values())
```

Add rejection tests for each exact artifact hash, code/contract/input/history
binding, candidate ID, failed decision gate, non-positive weighted/latest/
bootstrap gains, nonzero audit delta, stale rules review, source-gate failure,
environment mismatch, runtime above 480 seconds, RAM above 28 GiB, and package
size above policy.

- [ ] **Step 2: Run tests and confirm failure**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_gated_residual_submission_evidence.py -q
```

Expected: failure because the evidence builder is absent.

- [ ] **Step 3: Implement evidence creation**

Define `GatedResidualAcceptance` with `identity`, `audit_manifest`,
`acceptance_path`, and `benchmark_path`. `build_gated_residual_acceptance()`
must verify the immutable identities from the design, scan the rendered
`script.py`, E2 script, and every package-local runtime `.py`, and call
`run_phased_independence_audit()` with all five official rows.

Build the standard seven gates:

```python
gates = {
    "temporal_validation": weighted_gain > 0 and latest_gain > 0,
    "performance": bootstrap_lower > 0 and not failed_gates,
    "provenance": artifact_hashes_and_bindings_match,
    "row_independence": current_audit.status == "passed",
    "evaluator_runtime": projected_inference_seconds <= 480.0,
    "pretrained_license": True,
    "current_rules": True,
}
```

Use the exact candidate model digest for `AuditIdentity.model_sha256`, the
rendered script digest for `runtime_sha256`, the complete scanned source digest
for `adapter_sha256`, and the official five-row test/sample digest for
`data_sha256`. Write all outputs through a temporary directory and publish only
after every gate passes.

- [ ] **Step 4: Extend central audit metadata for G0**

In `submission/audit.py`, recognize `GATED_RESIDUAL_ADAPTER_ID`, require the
exact candidate-manifest key set, recompute every live member hash, require
candidate `G0_7628291922`, exact three artifact hashes, exact config and
iterations, and return that metadata to the renderer. Do not loosen the TabM
or E2 branches.

- [ ] **Step 5: Run evidence and central-packager regression tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_gated_residual_submission_evidence.py \
  tests/test_submission_package.py \
  tests/test_tree_expert_e2_submission_build_tool.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit evidence integration**

```bash
git add submission/gated_residual_existing_evidence.py submission/audit.py \
  tests/test_gated_residual_submission_evidence.py
git commit -m "feat: gate gated residual submission evidence"
```

### Task 5: Add the final build command and deterministic verification

**Files:**
- Create: `tools/build_gated_residual_submission.py`
- Create: `tests/test_gated_residual_submission_build_tool.py`

- [ ] **Step 1: Write failing build-tool tests**

```python
def test_final_builder_uses_sole_packager(monkeypatch, args):
    called = []
    monkeypatch.setattr(module, "build_submission_package", lambda request: called.append(request) or result)
    module.run_build(args)
    assert len(called) == 1
    assert called[0].adapter_id == "gated_residual_g0_v1"

def test_created_archive_has_exact_top_level_layout(submission_zip):
    with ZipFile(submission_zip) as archive:
        names = archive.namelist()
    assert "script.py" in names
    assert "requirements.txt" in names
    assert all(name in {"script.py", "requirements.txt"} or name.startswith("model/") for name in names)
```

Also cover wrong Python/package versions, non-five-row official sample, parity
mismatch, runtime projection over 480 seconds, existing output, changed final
member, and receipt/archive hash disagreement.

- [ ] **Step 2: Run tests and confirm failure**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_gated_residual_submission_build_tool.py -q
```

Expected: failure because the tool is absent.

- [ ] **Step 3: Implement `run_build()`**

The command accepts exactly:

```python
parser.add_argument("--delivery", type=Path, required=True)
parser.add_argument("--review", type=Path, required=True)
parser.add_argument("--handoff", type=Path, required=True)
parser.add_argument("--official-data", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
```

`run_build()` must:

1. require Python 3.11.15 and exact CatBoost/pandas/NumPy versions;
2. refuse outputs outside the repository or an existing output directory;
3. import the exact candidate and render the bound runtime;
4. load the official five-row `test.csv` and `sample_submission.csv`;
5. compare trusted delivery predictions and standalone predictions bit-for-bit;
6. benchmark a 4096-row local batch made by repeating only those five rows with
   synthetic unique `row_id` values, project to 245,789 rows with a 2x safety
   factor, and reject estimates above 480 seconds;
7. write `requirements.txt` as exactly `catboost==1.2.10\n`, leaving the
   official pandas and NumPy base versions untouched;
8. project compressed/extracted sizes and build current acceptance evidence;
9. call `build_submission_package()` exactly once;
10. reopen `submit.zip`, verify every byte against the audited runtime,
   requirements, and candidate model directory, and verify the receipt hash.

Print only after all checks succeed:

```python
print(
    "GATED_RESIDUAL_SUBMISSION_READY "
    f"archive={result.archive_path} sha256={result.archive_sha256} "
    f"bytes={result.archive_bytes} candidate={candidate.candidate_id} "
    f"delivery_sha256={candidate.artifact_sha256['delivery']} "
    f"review_sha256={candidate.artifact_sha256['review']} "
    f"handoff_sha256={candidate.artifact_sha256['handoff']}",
    flush=True,
)
```

- [ ] **Step 4: Run focused and full regression tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_gated_residual_submission_candidate.py \
  tests/test_gated_residual_submission_script.py \
  tests/test_gated_residual_submission_evidence.py \
  tests/test_gated_residual_submission_build_tool.py \
  tests/test_submission_package.py -q

artifacts/tabm_submission_python311/bin/python -m pytest -q
```

Expected: focused tests and the full repository suite pass.

- [ ] **Step 5: Commit the build tool**

```bash
git add tools/build_gated_residual_submission.py \
  tests/test_gated_residual_submission_build_tool.py
git commit -m "feat: build gated residual dacon submission"
```

### Task 6: Build and inspect the real submission

**Files:**
- Create at runtime: `artifacts/gated_residual_g0_submission_20260831/`

- [ ] **Step 1: Run pre-build static verification**

```bash
git diff --check
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_gated_residual_submission_candidate.py \
  tests/test_gated_residual_submission_script.py \
  tests/test_gated_residual_submission_evidence.py \
  tests/test_gated_residual_submission_build_tool.py \
  tests/test_submission_package.py -q
```

Expected: no diff errors and all focused tests pass.

- [ ] **Step 2: Build from the registered real artifacts**

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/build_gated_residual_submission.py \
  --delivery /Users/yonghyun/Downloads/gated_residual_final_delivery.zip \
  --review /Users/yonghyun/Downloads/gated_residual_final_review.zip \
  --handoff /Users/yonghyun/Downloads/gated_residual_final_handoff.zip \
  --official-data /Users/yonghyun/Documents/kaggle-lg-aimers-9th-data-upload \
  --output-dir artifacts/gated_residual_g0_submission_20260831
```

Expected: one `GATED_RESIDUAL_SUBMISSION_READY` line. This operation performs
no training and no full-test inference; expected local duration is roughly
3-10 minutes, dominated by hashing and recompressing the approximately 249 MB
delivery and its model files.

- [ ] **Step 3: Independently inspect the finished archive and receipt**

```bash
unzip -t artifacts/gated_residual_g0_submission_20260831/submit.zip
unzip -Z1 artifacts/gated_residual_g0_submission_20260831/submit.zip | \
  awk '!($0 == "script.py" || $0 == "requirements.txt" || $0 ~ /^model\//) {exit 1}'
shasum -a 256 artifacts/gated_residual_g0_submission_20260831/submit.zip
cat artifacts/gated_residual_g0_submission_20260831/submission_receipt.json
```

Expected: no corrupt member, no unexpected top-level path, and matching SHA-256
in command output and receipt.

- [ ] **Step 4: Record the handoff without uploading**

Report the archive path, SHA-256, compressed size, candidate logic, expected
runtime, and the exact one-line DACON memo. Do not push or upload automatically.
