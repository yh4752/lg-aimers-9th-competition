# Gated Residual Final Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one Kaggle T4 x2 experiment that searches only pre-declared, cutoff-safe residual/calibration candidates, blocks full training unless every OOF gate passes, and emits review/handoff plus a delivery only for an accepted candidate.

**Architecture:** A new `experiments/gated_residual_final` package reads a compact, hash-bound input containing E2 and Direct Expert evidence. Pure modules perform OOF alignment, reliability shrinkage, temporal calibration, selection, and compliance; a runner conditionally reuses Direct Expert full-fit/inference primitives after acceptance. A generated one-cell Kaggle launcher embeds the closed runtime dependency set and is tested against both zipped and Kaggle-expanded input layouts.

**Tech Stack:** Python 3.11/3.12, NumPy, pandas, CatBoost 1.2.10, pytest, ZIP/JSON/SHA-256 artifact contracts, Kaggle T4 x2.

---

## File map

- `experiments/gated_residual_final/contract.json`: immutable grids, hashes, gates, runtime limits, and output names.
- `experiments/gated_residual_final/contracts.py`: strict typed contract loader.
- `experiments/gated_residual_final/inputs.py`: compact-input creation and ZIP/expanded-layout verification.
- `experiments/gated_residual_final/oof.py`: semantic row alignment and extraction of E2/D0/D5 seed predictions.
- `experiments/gated_residual_final/residual.py`: logit residual and cutoff-safe reliability calculation.
- `experiments/gated_residual_final/calibration.py`: temporal hierarchical calibration fit/apply.
- `experiments/gated_residual_final/selection.py`: pre-2024 tuning, frozen 2024 confirmation, bootstrap/seed/segment gates.
- `experiments/gated_residual_final/state.py`: atomic resumable phase/job state.
- `experiments/gated_residual_final/artifacts.py`: bound review, handoff, and accepted-only delivery archives.
- `experiments/gated_residual_final/production.py`: conditional D0/D5 full fit and frozen inference assembly.
- `experiments/gated_residual_final/compliance.py`: temporal, row-independence, probability, and artifact audits.
- `experiments/gated_residual_final/runner.py`: P0-P5 campaign orchestration and deadline handling.
- `experiments/gated_residual_final/kaggle.py`: Kaggle discovery, preflight, resume, logs, and single final handoff download target.
- `experiments/gated_residual_final/runtime_inventory.py`: closed embedded-runtime member list and identity hash.
- `experiments/gated_residual_final/KAGGLE_CELL.py`: generated one-cell launcher under Kaggle's 1 MB source limit.
- `tools/prepare_gated_residual_final_input.py`: local compact-input builder.
- `tools/build_gated_residual_final_kaggle_cell.py`: deterministic cell generator.
- `tools/preflight_gated_residual_final.py`: no-training verification of ZIP/expanded layouts, embedded imports, and submission-package absence.
- `tests/test_gated_residual_*.py`: unit, integration, packaging, and Kaggle-layout regressions.

### Task 1: Lock the contract and runtime identity

**Files:**
- Create: `experiments/gated_residual_final/__init__.py`
- Create: `experiments/gated_residual_final/contract.json`
- Create: `experiments/gated_residual_final/contracts.py`
- Create: `experiments/gated_residual_final/runtime_inventory.py`
- Test: `tests/test_gated_residual_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

```python
def test_contract_locks_search_and_gates():
    contract = load_contract()
    assert contract.alpha_grid == (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    assert contract.k_grid == (25, 100, 400)
    assert contract.beta_grid == (0.05, 0.10, 0.20)
    assert contract.lambda_grid == (100, 500, 2000)
    assert contract.gates["weighted_gain"] == 0.00005
    assert contract.maximum_runtime_seconds == 28_800

def test_runtime_inventory_contains_every_imported_project_module():
    members = runtime_members()
    assert "experiments/gated_residual_final/runner.py" in members
    assert "experiments/direct_expert/features.py" in members
    assert len(members) == len(set(members))
```

- [ ] **Step 2: Run the contract tests and confirm the missing-package failure**

Run: `pytest -q tests/test_gated_residual_contracts.py`

Expected: FAIL with `ModuleNotFoundError: experiments.gated_residual_final`.

- [ ] **Step 3: Implement the strict contract loader and closed runtime inventory**

```python
@dataclass(frozen=True)
class FinalContract:
    alpha_grid: tuple[float, ...]
    k_grid: tuple[int, ...]
    beta_grid: tuple[float, ...]
    lambda_grid: tuple[int, ...]
    gates: Mapping[str, float]
    expected_hashes: Mapping[str, str]
    maximum_runtime_seconds: int

def load_contract() -> FinalContract:
    payload = json.loads(Path(__file__).with_name("contract.json").read_text())
    if set(payload) != {"version", "search", "gates", "expected_hashes", "runtime"}:
        raise FinalContractError("contract keys differ")
    return FinalContract(
        alpha_grid=tuple(float(x) for x in payload["search"]["alpha"]),
        k_grid=tuple(int(x) for x in payload["search"]["k"]),
        beta_grid=tuple(float(x) for x in payload["search"]["beta"]),
        lambda_grid=tuple(int(x) for x in payload["search"]["lambda"]),
        gates=MappingProxyType(dict(payload["gates"])),
        expected_hashes=MappingProxyType(dict(payload["expected_hashes"])),
        maximum_runtime_seconds=int(payload["runtime"]["maximum_seconds"]),
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `pytest -q tests/test_gated_residual_contracts.py`

Expected: `2 passed`.

- [ ] **Step 5: Commit the contract**

```bash
git add experiments/gated_residual_final tests/test_gated_residual_contracts.py
git commit -m "feat: lock gated residual final contract"
```

### Task 2: Build and verify the compact input

**Files:**
- Create: `experiments/gated_residual_final/inputs.py`
- Create: `tools/prepare_gated_residual_final_input.py`
- Test: `tests/test_gated_residual_inputs.py`
- Test: `tests/test_gated_residual_prepare_tool.py`

- [ ] **Step 1: Write failing tests for ZIP and expanded Kaggle layouts**

```python
def test_compact_input_round_trip_renames_nested_submission(tmp_path, source_artifacts):
    archive = prepare_final_input(**source_artifacts, output=tmp_path / "input.zip")
    with ZipFile(archive) as handle:
        assert "e2/submission.bin" in handle.namelist()
        assert not any(name.endswith(".zip") for name in handle.namelist())
    verified = verify_and_extract_final_input(archive, tmp_path / "verified")
    assert verified.e2_submission.name == "submission.zip"

def test_expanded_kaggle_dataset_has_same_identity(tmp_path, prepared_input):
    expanded = tmp_path / "dataset"
    with ZipFile(prepared_input) as handle:
        handle.extractall(expanded)
    left = verify_and_extract_final_input(prepared_input, tmp_path / "a")
    right = verify_and_extract_final_input(expanded, tmp_path / "b")
    assert left.manifest_sha256 == right.manifest_sha256
```

- [ ] **Step 2: Run tests and confirm missing API failures**

Run: `pytest -q tests/test_gated_residual_inputs.py tests/test_gated_residual_prepare_tool.py`

Expected: FAIL because `prepare_final_input` and `verify_and_extract_final_input` do not exist.

- [ ] **Step 3: Implement exact-member, hash-bound input creation and verification**

```python
INPUT_MEMBERS = frozenset({
    "e2/submission.bin", "e2/oof.csv", "direct/stage_a.bin",
    "direct/stage_b.bin", "sources.json", "manifest.json",
})

def verify_and_extract_final_input(source: Path, destination: Path) -> VerifiedFinalInput:
    payloads = source_payloads(source)
    if set(payloads) != INPUT_MEMBERS:
        raise FinalInputError("final input member set differs")
    manifest = strict_json(payloads["manifest.json"])
    verify_declared_members(payloads, manifest)
    destination.mkdir(parents=True, exist_ok=False)
    write_verified_members(payloads, destination)
    submission = destination / "e2" / "submission.zip"
    submission.write_bytes(payloads["e2/submission.bin"])
    return VerifiedFinalInput(destination, submission, sha256_bytes(payloads["manifest.json"]))
```

- [ ] **Step 4: Add a CLI that accepts the fixed source artifacts and prints one success marker**

```python
def main(argv=None) -> int:
    args = parse_args(argv)
    output = prepare_final_input(
        e2_input=args.e2_input,
        stage_a=args.stage_a,
        stage_b=args.stage_b,
        output=args.output,
    )
    print(f"GATED_RESIDUAL_INPUT_READY path={output.resolve()} sha256={file_sha256(output)}")
    return 0
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_gated_residual_inputs.py tests/test_gated_residual_prepare_tool.py`

Expected: all tests pass.

```bash
git add experiments/gated_residual_final/inputs.py tools/prepare_gated_residual_final_input.py tests/test_gated_residual_inputs.py tests/test_gated_residual_prepare_tool.py
git commit -m "feat: add bound gated residual input"
```

### Task 3: Align OOF evidence semantically

**Files:**
- Create: `experiments/gated_residual_final/oof.py`
- Test: `tests/test_gated_residual_oof.py`

- [ ] **Step 1: Write failing tests for dtype-insensitive alignment and D5 fallback**

```python
def test_alignment_accepts_equivalent_numeric_dtypes():
    left = frame(row_dtype="int64", target_dtype="int8")
    right = frame(row_dtype="int32", target_dtype="int64")
    aligned = align_predictions({"e2": left, "d0": right})
    assert aligned["e2"]["row_id"].tolist() == aligned["d0"]["row_id"].tolist()

def test_direct_probability_uses_d5_only_for_regular_season():
    out = direct_probability(game_type=np.array(["R", "F"]), d0=np.array([.2, .3]), d5=np.array([.8, .9]))
    np.testing.assert_allclose(out, [.8, .3])
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q tests/test_gated_residual_oof.py`

Expected: FAIL because the OOF module is absent.

- [ ] **Step 3: Implement canonical value comparison, never pandas dtype equality**

```python
def align_predictions(frames: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
    normalized = {name: normalize_frame(frame) for name, frame in frames.items()}
    anchor = next(iter(normalized.values()))
    for name, frame in normalized.items():
        if not np.array_equal(anchor["row_id"].to_numpy(), frame["row_id"].to_numpy()):
            raise OOFError(f"row identity differs: {name}")
        if not np.array_equal(anchor["target"].to_numpy(), frame["target"].to_numpy()):
            raise OOFError(f"target values differ: {name}")
    return MappingProxyType(normalized)

def direct_probability(*, game_type, d0, d5):
    return np.where(np.asarray(game_type) == "R", np.asarray(d5, dtype="float64"), np.asarray(d0, dtype="float64"))
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_gated_residual_oof.py`

Expected: all tests pass.

```bash
git add experiments/gated_residual_final/oof.py tests/test_gated_residual_oof.py
git commit -m "feat: align final candidate OOF evidence"
```

### Task 4: Implement cutoff-safe residual and reliability

**Files:**
- Create: `experiments/gated_residual_final/residual.py`
- Test: `tests/test_gated_residual_residual.py`

- [ ] **Step 1: Write failing mathematical and cutoff tests**

```python
def test_non_regular_rows_back_off_exactly_to_anchor():
    result = gated_residual(np.array([.2]), np.array([.9]), np.array([0.0]), alpha=.3)
    np.testing.assert_allclose(result, [.2], atol=1e-12)

def test_validation_counts_use_only_prior_years():
    counts = temporal_entity_counts(training_rows(), validation_year=2024)
    assert counts.pitcher["p1"] == 2
    assert "p_future" not in counts.pitcher

def test_unknown_entity_has_zero_reliability():
    value = reliability(0, 100, k=25)
    assert value == 0.0
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q tests/test_gated_residual_residual.py`

Expected: FAIL because the residual module is absent.

- [ ] **Step 3: Implement clipped logits and geometric count shrinkage**

```python
def gated_residual(anchor, direct, row_reliability, *, alpha):
    p0 = np.clip(np.asarray(anchor, dtype="float64"), 1e-6, 1 - 1e-6)
    pdirect = np.clip(np.asarray(direct, dtype="float64"), 1e-6, 1 - 1e-6)
    strength = float(alpha) * np.asarray(row_reliability, dtype="float64")
    return expit(logit(p0) + strength * (logit(pdirect) - logit(p0)))

def reliability(pitcher_count, batter_count, *, k):
    rp = pitcher_count / (pitcher_count + k)
    rb = batter_count / (batter_count + k)
    return float(np.sqrt(rp * rb))
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_gated_residual_residual.py`

Expected: all tests pass.

```bash
git add experiments/gated_residual_final/residual.py tests/test_gated_residual_residual.py
git commit -m "feat: add cutoff safe gated residual"
```

### Task 5: Implement temporal hierarchical calibration

**Files:**
- Create: `experiments/gated_residual_final/calibration.py`
- Test: `tests/test_gated_residual_calibration.py`

- [ ] **Step 1: Write failing tests for fit years, backoff, and row independence**

```python
def test_2024_calibrator_fits_only_2022_and_2023():
    fitted = fit_temporal_calibrator(oof_rows(), validation_year=2024, hierarchy="full", ridge=500)
    assert fitted.source_years == (2022, 2023)

def test_unknown_pitcher_and_batter_back_off_to_hand_effect():
    fitted = fixture_calibrator()
    effect = fitted.effect_for(game_type="R", hand_matchup="RL", pitcher_id="new", batter_id="new")
    assert effect == fitted.hand_effects[("R", "RL")]

def test_other_evaluation_rows_do_not_change_prediction():
    one = apply_calibration(fitted(), evaluation().iloc[[0]], beta=.1)
    many = apply_calibration(fitted(), evaluation(), beta=.1).iloc[[0]]
    np.testing.assert_allclose(one, many)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q tests/test_gated_residual_calibration.py`

Expected: FAIL because the calibration module is absent.

- [ ] **Step 3: Implement ridge effects and parent shrink/backoff**

```python
def _effect(group: pd.DataFrame, ridge: float, parent: float) -> float:
    numerator = float((group["target"] - group["probability"]).sum())
    information = float((group["probability"] * (1.0 - group["probability"])).sum())
    weight = information / (information + ridge)
    raw = numerator / (information + ridge)
    return weight * raw + (1.0 - weight) * parent

def fit_temporal_calibrator(frame, *, validation_year, hierarchy, ridge):
    source = frame.loc[frame["oof_year"] < validation_year].copy()
    if not set(source["oof_year"]).issubset(set(range(2022, validation_year))):
        raise CalibrationError("calibration cutoff differs")
    return build_tables(source, hierarchy=hierarchy, ridge=float(ridge))
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_gated_residual_calibration.py`

Expected: all tests pass.

```bash
git add experiments/gated_residual_final/calibration.py tests/test_gated_residual_calibration.py
git commit -m "feat: add temporal hierarchical calibration"
```

### Task 6: Freeze search before 2024 and enforce every gate

**Files:**
- Create: `experiments/gated_residual_final/selection.py`
- Test: `tests/test_gated_residual_selection.py`

- [ ] **Step 1: Write failing tests for frozen confirmation and hard rejection**

```python
def test_2024_is_not_passed_to_structure_search(monkeypatch, evidence):
    seen = []
    monkeypatch.setattr(selection, "score_structure", lambda frame, candidate: seen.extend(frame.oof_year.unique()) or 0.0)
    freeze_archetypes(evidence)
    assert 2024 not in seen

def test_one_failed_gate_blocks_acceptance():
    evidence = passing_evidence(latest_gain=-1e-12)
    decision = decide(evidence)
    assert decision.status == "rejected"
    assert "latest_gain" in decision.failed_gates

def test_seed_and_segment_gates_are_mandatory():
    evidence = passing_evidence(non_worse_seed_count=1, improving_latest_seed_count=1, maximum_segment_regression=.00031)
    assert set(decide(evidence).failed_gates) == {"non_worse_seed_count", "improving_latest_seed_count", "maximum_segment_regression"}
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_gated_residual_selection.py`

Expected: FAIL because the selection module is absent.

- [ ] **Step 3: Implement four-archetype tuning and immutable confirmation records**

```python
def freeze_archetypes(evidence: OOFSet) -> tuple[FrozenCandidate, ...]:
    tuning = evidence.frame.loc[evidence.frame["oof_year"].isin((2022, 2023))]
    frozen = []
    for archetype in ("G0", "G1", "G2", "G3"):
        candidates = enumerate_declared_candidates(archetype, load_contract())
        best = max(candidates, key=lambda item: tuning_objective(tuning, item))
        frozen.append(FrozenCandidate.from_config(best, locked_years=(2022, 2023)))
    return tuple(frozen)

def decide(evidence: CandidateEvidence) -> CandidateDecision:
    checks = ordered_gate_checks(evidence, load_contract().gates)
    failures = tuple(name for name, passed in checks if not passed)
    return CandidateDecision("accepted" if not failures else "rejected", failures)
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_gated_residual_selection.py`

Expected: all tests pass.

```bash
git add experiments/gated_residual_final/selection.py tests/test_gated_residual_selection.py
git commit -m "feat: gate frozen residual candidates"
```

### Task 7: Add atomic state and bound artifacts

**Files:**
- Create: `experiments/gated_residual_final/state.py`
- Create: `experiments/gated_residual_final/artifacts.py`
- Test: `tests/test_gated_residual_state.py`
- Test: `tests/test_gated_residual_artifacts.py`

- [ ] **Step 1: Write failing state and accepted-only delivery tests**

```python
def test_atomic_state_survives_reload(tmp_path):
    path = tmp_path / "state.json"
    save_state(path, CampaignState(phase="P2", status="running", completed_jobs=("d0_s42",)))
    assert load_state(path).completed_jobs == ("d0_s42",)

def test_rejected_campaign_cannot_create_delivery(tmp_path):
    with pytest.raises(FinalArtifactError, match="accepted evidence is required"):
        create_delivery(tmp_path / "delivery.zip", decision=rejected_decision(), payloads={})

def test_handoff_rejects_different_code_binding(tmp_path):
    handoff = create_handoff_fixture(tmp_path, code_sha256="a" * 64)
    with pytest.raises(FinalArtifactError, match="bindings differ"):
        restore_handoff(handoff, expected_bindings=bindings(code_sha256="b" * 64))
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q tests/test_gated_residual_state.py tests/test_gated_residual_artifacts.py`

Expected: FAIL because state/artifact APIs are absent.

- [ ] **Step 3: Implement fsync-plus-replace state and exact manifest verification**

```python
def save_state(path: Path, state: CampaignState) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(asdict(state)), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def create_delivery(path, *, decision, payloads):
    if decision.status != "accepted" or not decision.acceptance_sha256:
        raise FinalArtifactError("accepted evidence is required")
    return create_bound_archive(path, kind="gated_residual_final_delivery_v1", payloads=payloads, bindings=decision.bindings)
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_gated_residual_state.py tests/test_gated_residual_artifacts.py`

Expected: all tests pass.

```bash
git add experiments/gated_residual_final/state.py experiments/gated_residual_final/artifacts.py tests/test_gated_residual_state.py tests/test_gated_residual_artifacts.py
git commit -m "feat: add resumable final candidate artifacts"
```

### Task 8: Conditional full fit, inference, and compliance

**Files:**
- Create: `experiments/gated_residual_final/production.py`
- Create: `experiments/gated_residual_final/compliance.py`
- Test: `tests/test_gated_residual_production.py`
- Test: `tests/test_gated_residual_compliance.py`

- [ ] **Step 1: Write failing tests for the no-training gate and row independence**

```python
def test_rejected_decision_never_calls_full_fit(monkeypatch, fixtures):
    called = False
    monkeypatch.setattr(production, "fit_direct_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    result = build_production_candidate(decision=rejected_decision(), **fixtures)
    assert result is None

def test_accepted_candidate_trains_exactly_six_direct_models(monkeypatch, fixtures):
    calls = []
    monkeypatch.setattr(production, "fit_direct_model", lambda role, seed, **k: calls.append((role, seed)) or fake_model())
    build_production_candidate(decision=accepted_decision(), **fixtures)
    assert calls == [(role, seed) for role in ("D0", "D5") for seed in (42, 2026, 3407)]

def test_batch_context_cannot_change_one_row_prediction():
    audit = audit_row_independence(frozen_predictor(), evaluation_fixture())
    assert audit.passed
    assert audit.maximum_absolute_difference <= 1e-12
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_gated_residual_production.py tests/test_gated_residual_compliance.py`

Expected: FAIL because production/compliance APIs are absent.

- [ ] **Step 3: Implement accepted-only training and frozen lookup inference**

```python
def build_production_candidate(*, decision, train, feature_state, calibration_state):
    if decision.status != "accepted":
        return None
    models = {
        f"{role}_s{seed}": fit_direct_model(role=role, seed=seed, train=train, depth=10, iterations=decision.iterations[role])
        for role in ("D0", "D5") for seed in (42, 2026, 3407)
    }
    return ProductionCandidate(models=models, feature_state=feature_state, calibration_state=calibration_state, decision=decision)

def predict_rows(candidate, frame):
    direct = predict_fixed_models(candidate.models, frame)
    reliability = lookup_frozen_counts(candidate.feature_state.counts, frame)
    calibrated = apply_frozen_calibration(candidate.calibration_state, frame, direct, reliability)
    return validate_probabilities(calibrated)
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_gated_residual_production.py tests/test_gated_residual_compliance.py`

Expected: all tests pass.

```bash
git add experiments/gated_residual_final/production.py experiments/gated_residual_final/compliance.py tests/test_gated_residual_production.py tests/test_gated_residual_compliance.py
git commit -m "feat: add accepted final production candidate"
```

### Task 9: Orchestrate P0-P5 with resume and one download target

**Files:**
- Create: `experiments/gated_residual_final/runner.py`
- Create: `experiments/gated_residual_final/kaggle.py`
- Test: `tests/test_gated_residual_runner.py`
- Test: `tests/test_gated_residual_kaggle.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_rejected_campaign_stops_before_p3(monkeypatch, campaign_fixture):
    monkeypatch.setattr(runner, "select_candidate", lambda *a, **k: rejected_decision())
    monkeypatch.setattr(runner, "build_production_candidate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("P3 ran")))
    result = run_campaign(**campaign_fixture)
    assert result.status == "completed_no_candidate"
    assert result.delivery is None

def test_interrupted_run_reuses_completed_model(tmp_path, monkeypatch, accepted_fixture):
    first = run_until_one_model_then_interrupt(tmp_path, accepted_fixture)
    calls = capture_fit_calls(monkeypatch)
    result = run_campaign(resume=first.handoff, **accepted_fixture)
    assert "D0_s42" not in calls
    assert result.status == "completed"

def test_deadline_emits_one_latest_handoff(tmp_path, campaign_fixture):
    result = run_campaign(absolute_deadline=expired_clock(), **campaign_fixture)
    assert result.status == "paused"
    assert result.handoff.name == "gated_residual_final_handoff.zip"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_gated_residual_runner.py tests/test_gated_residual_kaggle.py`

Expected: FAIL because runner/Kaggle APIs are absent.

- [ ] **Step 3: Implement deterministic P0-P5 orchestration**

```python
def run_campaign(*, verified, official, output_dir, resume=None, absolute_deadline=None):
    state = restore_or_initialize(resume, verified.bindings, output_dir)
    evidence = phase_p1_evidence(state, verified, official)
    decision = phase_p2_gate(state, evidence)
    if decision.status != "accepted":
        review, handoff = finalize_no_candidate(state, decision)
        return CampaignResult("completed_no_candidate", review, handoff, None)
    production = phase_p3_full_fit(state, decision, official, absolute_deadline)
    audits = phase_p4_audit(state, production, official)
    return phase_p5_finalize(state, decision, production, audits)
```

- [ ] **Step 4: Log stable markers and expose only the latest final handoff**

```python
print(f"FINAL_CANDIDATE_STAGE phase={phase}", flush=True)
print(f"FINAL_CANDIDATE_DECISION status={decision.status} candidate={decision.candidate_id}", flush=True)
print(f"FINAL_CANDIDATE_HANDOFF_READY path={result.handoff}", flush=True)
if result.delivery is not None:
    print(f"FINAL_CANDIDATE_DELIVERY_READY path={result.delivery}", flush=True)
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_gated_residual_runner.py tests/test_gated_residual_kaggle.py`

Expected: all tests pass.

```bash
git add experiments/gated_residual_final/runner.py experiments/gated_residual_final/kaggle.py tests/test_gated_residual_runner.py tests/test_gated_residual_kaggle.py
git commit -m "feat: orchestrate gated residual final campaign"
```

### Task 10: Generate and preflight the one-cell Kaggle launcher

**Files:**
- Create: `tools/build_gated_residual_final_kaggle_cell.py`
- Create: `experiments/gated_residual_final/KAGGLE_CELL.py`
- Test: `tests/test_gated_residual_kaggle_cell.py`

- [ ] **Step 1: Write failing cell parity, size, and isolated-import tests**

```python
def test_checked_in_cell_matches_generator(tmp_path):
    generated = build_cell(tmp_path / "cell.py")
    assert generated.read_bytes() == Path("experiments/gated_residual_final/KAGGLE_CELL.py").read_bytes()

def test_cell_is_below_kaggle_source_limit():
    assert Path("experiments/gated_residual_final/KAGGLE_CELL.py").stat().st_size < 1_000_000

def test_embedded_runtime_imports_with_isolated_python(tmp_path):
    runtime = extract_embedded_runtime(Path("experiments/gated_residual_final/KAGGLE_CELL.py"), tmp_path)
    result = subprocess.run([sys.executable, "-I", "-c", isolated_import_script(runtime)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q tests/test_gated_residual_kaggle_cell.py`

Expected: FAIL because the builder and cell are absent.

- [ ] **Step 3: Implement deterministic tar/base64 runtime embedding and one-cell launcher**

```python
payload = base64.b64encode(build_runtime_tar(ROOT, runtime_members())).decode("ascii")
cell = TEMPLATE.replace("__RUNTIME_B64__", payload).replace("__CODE_SHA256__", code_identity_sha256(ROOT))
output.write_text(cell, encoding="utf-8")
if output.stat().st_size >= 1_000_000:
    raise CellBuildError("Kaggle cell exceeds one megabyte")
```

- [ ] **Step 4: Add preflight before dependency installation or training**

```python
found = discover_inputs(Path("/kaggle/input"))
verified = verify_and_extract_final_input(found.final_input, RUN_ROOT / "verified")
official = verify_official_data(found.official_data)
verify_gpu_count(minimum=2)
print(f"FINAL_CANDIDATE_INPUTS_VERIFIED input_sha256={verified.manifest_sha256}", flush=True)
```

- [ ] **Step 5: Generate, run tests, and commit**

Run: `python tools/build_gated_residual_final_kaggle_cell.py`

Expected: `GATED_RESIDUAL_CELL_READY ... size_bytes=<1000000`.

Run: `pytest -q tests/test_gated_residual_kaggle_cell.py`

Expected: all tests pass.

```bash
git add tools/build_gated_residual_final_kaggle_cell.py experiments/gated_residual_final/KAGGLE_CELL.py tests/test_gated_residual_kaggle_cell.py
git commit -m "feat: add final Kaggle experiment cell"
```

### Task 11: Run end-to-end synthetic and actual-layout preflight

**Files:**
- Create: `tests/test_gated_residual_end_to_end.py`
- Create: `tools/preflight_gated_residual_final.py`
- Modify: `README.md`

- [ ] **Step 1: Write a rejected-path end-to-end test**

```python
def test_end_to_end_rejection_emits_no_delivery(tmp_path, synthetic_sources):
    result = run_fixture_campaign(tmp_path, synthetic_sources, forced_gain=-0.001)
    assert result.status == "completed_no_candidate"
    assert result.review.is_file()
    assert result.handoff.is_file()
    assert result.delivery is None
    assert not any(path.name.endswith("delivery.zip") for path in tmp_path.rglob("*"))
```

- [ ] **Step 2: Write an accepted-path end-to-end test with stub models**

```python
def test_end_to_end_acceptance_emits_audited_delivery(tmp_path, synthetic_sources, stub_training):
    result = run_fixture_campaign(tmp_path, synthetic_sources, forced_gain=0.001)
    verified = verify_delivery(result.delivery, expected_bindings=result.bindings)
    assert result.status == "completed"
    assert verified.acceptance_evidence["status"] == "accepted"
    assert verified.compliance["row_independence"] == "passed"
```

- [ ] **Step 3: Run focused and full regression suites**

Run: `pytest -q tests/test_gated_residual_*.py`

Expected: all gated-residual tests pass.

Run: `pytest -q tests/test_direct_expert_*.py tests/test_gated_residual_*.py`

Expected: all Direct Expert and gated-residual tests pass.

- [ ] **Step 4: Build the real compact input and validate both Kaggle layouts without training**

Run:

```bash
python tools/prepare_gated_residual_final_input.py \
  --e2-input artifacts/direct_expert_input.zip \
  --stage-a /path/to/Downloads/direct_expert_stage_A_handoff.zip \
  --stage-b /path/to/Downloads/direct_expert_handoff.zip \
  --output artifacts/gated_residual_final_input.zip
```

Expected: `GATED_RESIDUAL_INPUT_READY` with a stable SHA-256.

Run: `python tools/preflight_gated_residual_final.py --input artifacts/gated_residual_final_input.zip --cell experiments/gated_residual_final/KAGGLE_CELL.py`

Expected: `GATED_RESIDUAL_PREFLIGHT_SUCCESS layouts=zip,expanded imports=isolated submission_created=false`.

The preflight entry point must execute both source layouts in fresh temporary directories and fail closed:

```python
def main(argv=None) -> int:
    args = parse_args(argv)
    with TemporaryDirectory(prefix="gated_residual_preflight_") as temporary:
        root = Path(temporary)
        verify_zip_and_expanded_layouts(args.input, root)
        verify_embedded_import(args.cell, root)
        if any(path.name == "submission.zip" for path in root.rglob("*")):
            raise PreflightError("submission package was created")
    print("GATED_RESIDUAL_PREFLIGHT_SUCCESS layouts=zip,expanded imports=isolated submission_created=false")
    return 0
```

- [ ] **Step 5: Document exact user handoff and commit**

```markdown
Inputs: official `lg-aimers-9th-data` plus `gated_residual_final_input` only.
Accelerator: GPU T4 x2, Internet off.
Expected runtime: rejected path 30-60 minutes; accepted path 5-8 hours.
Rerun safety: add the previous `gated_residual_final_handoff` Dataset to resume completed phases/jobs.
Return: `gated_residual_final_review.zip`, `gated_residual_final_handoff.zip`, and only if present `gated_residual_final_delivery.zip`.
```

```bash
git add README.md tests/test_gated_residual_end_to_end.py tools/preflight_gated_residual_final.py
git commit -m "docs: add final candidate Kaggle handoff"
```

The real compact input remains an ignored local artifact and is distributed to Kaggle separately; it is never added to Git.

### Task 12: Final verification before user handoff

**Files:**
- Verify only; no new files unless a discovered defect requires a focused fix and regression test.

- [ ] **Step 1: Verify repository status and generated parity**

Run: `git status --short && python tools/build_gated_residual_final_kaggle_cell.py --check`

Expected: clean status and `GATED_RESIDUAL_CELL_CHECK_SUCCESS`.

- [ ] **Step 2: Verify syntax and runtime imports**

Run: `python -m compileall -q experiments/gated_residual_final tools/prepare_gated_residual_final_input.py tools/build_gated_residual_final_kaggle_cell.py`

Expected: exit code 0.

- [ ] **Step 3: Run the complete relevant test suite**

Run: `pytest -q tests/test_direct_expert_*.py tests/test_gated_residual_*.py`

Expected: all tests pass with no failures.

- [ ] **Step 4: Inspect the real deliverables**

Run: `ls -lh artifacts/gated_residual_final_input.zip experiments/gated_residual_final/KAGGLE_CELL.py && shasum -a 256 artifacts/gated_residual_final_input.zip experiments/gated_residual_final/KAGGLE_CELL.py`

Expected: both files exist; cell size is under 1 MB; hashes match the final handoff message.

- [ ] **Step 5: Commit any verification-only documentation update**

```bash
git add README.md
git commit -m "docs: finalize gated residual run instructions"
```
