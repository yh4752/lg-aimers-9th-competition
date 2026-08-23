# Temporal Portfolio Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create sealed experiment contracts, immutable identities, verified input discovery, resumable state, and deterministic single-handoff artifacts for the temporal portfolio.

**Architecture:** The foundation package has no model imports. Contract bytes define the approved stages and grids; immutable training identities are separated from runtime build hashes; manifest-based artifact discovery works for ZIP files and Kaggle-extracted directories.

**Tech Stack:** Python 3.11 standard library, dataclasses, hashlib, zipfile, JSON, pytest.

---

### Task 1: Package skeleton and sealed contract

**Files:**
- Create: `experiments/temporal_portfolio/__init__.py`
- Create: `experiments/temporal_portfolio/contract.json`
- Create: `experiments/temporal_portfolio/contracts.py`
- Create: `tests/test_temporal_portfolio_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from dataclasses import replace
from decimal import Decimal

import pytest

from experiments.temporal_portfolio.contracts import (
    PortfolioContractError,
    build_stage_jobs,
    load_contract,
)


def test_contract_pins_budget_folds_and_t1_grid() -> None:
    contract = load_contract()
    assert contract.campaign_id == "temporal_portfolio_v1"
    assert sum(contract.stage_hours.values()) == Decimal("30")
    assert [(f.recent_year, f.multi_start, f.valid_year) for f in contract.folds] == [
        (2021, 2019, 2022), (2022, 2019, 2023), (2023, 2020, 2024)
    ]
    assert contract.decays == (Decimal("0.40"), Decimal("0.55"), Decimal("0.70"), Decimal("1.00"))
    assert contract.recent_weights == (
        Decimal("0.50"), Decimal("0.65"), Decimal("0.75"),
        Decimal("0.85"), Decimal("1.00"),
    )
    assert len(build_stage_jobs(contract, "T1")) == 15


def test_contract_is_fail_closed() -> None:
    contract = load_contract()
    with pytest.raises(PortfolioContractError, match="campaign identity"):
        build_stage_jobs(replace(contract, campaign_id="other"), "T1")
```

- [ ] **Step 2: Run the tests and verify import failure**

```bash
pytest tests/test_temporal_portfolio_contracts.py -q
```

Expected: FAIL with `ModuleNotFoundError: experiments.temporal_portfolio`.

- [ ] **Step 3: Add the approved JSON contract and strict parser**

`contract.json` must contain exact JSON strings for decimals and these stage budgets:

```json
{
  "schema_version": 1,
  "campaign_id": "temporal_portfolio_v1",
  "submission_package": false,
  "folds": [[2021, 2019, 2021, 2022], [2022, 2019, 2022, 2023], [2023, 2020, 2023, 2024]],
  "decays": ["0.40", "0.55", "0.70", "1.00"],
  "recent_weights": ["0.50", "0.65", "0.75", "0.85", "1.00"],
  "anchor_betas": ["0", "0.025", "0.05", "0.10"],
  "feature_bundles": ["S1", "P0", "P1", "P2", "P3", "B1", "M1"],
  "tabm_profiles": {
    "p2": {"k": 32, "width": 512, "blocks": 4, "dropout": "0.10"},
    "p3_lite": {"k": 32, "width": 768, "blocks": 6, "dropout": "0.15"}
  },
  "losses": ["bce", "brier"],
  "catboost_prefixes": [16, 64, 192, 384],
  "lupi_lambdas": ["0.10", "0.25"],
  "pair_primary_weights": ["0.90", "0.80", "0.70"],
  "bootstrap": {"repeats": 1000, "seed": 3407, "minimum_segment_rows": 5000},
  "gates": {
    "champion_weighted_gain": "0.00005",
    "champion_latest_gain": "0.00003",
    "champion_max_segment_regression": "0.00050",
    "exploratory_weighted_gain": "0.00003",
    "exploratory_worst_fold_regression": "0.00015",
    "exploratory_latest_regression": "0.00005",
    "exploratory_max_segment_regression": "0.00100"
  },
  "stage_hours": {"T1": "6", "T2A": "4", "T2B": "4", "T3": "6", "T4": "3", "T5": "5", "reserve": "2"},
  "physical_stage_seconds": {"T1": 21600, "T2A": 14400, "T2B": 14400, "T3A": 10800, "T3BT4": 20700, "T5A": 9000, "T5B": 9000},
  "seeds": {"screen": 3407, "confirm": 42},
  "score_tiers": {"incremental": "0.00005", "competitive": "0.00025", "breakthrough": "0.00045"}
}
```

Implement immutable types and duplicate-key/non-finite rejection:

```python
@dataclass(frozen=True)
class TemporalFold:
    recent_year: int
    multi_start: int
    multi_end: int
    valid_year: int


@dataclass(frozen=True)
class PortfolioContract:
    schema_version: int
    campaign_id: str
    folds: tuple[TemporalFold, ...]
    decays: tuple[Decimal, ...]
    recent_weights: tuple[Decimal, ...]
    anchor_betas: tuple[Decimal, ...]
    feature_bundles: tuple[str, ...]
    tabm_profiles: Mapping[str, Mapping[str, object]]
    losses: tuple[str, ...]
    catboost_prefixes: tuple[int, ...]
    lupi_lambdas: tuple[Decimal, ...]
    pair_primary_weights: tuple[Decimal, ...]
    bootstrap_repeats: int
    minimum_segment_rows: int
    gates: Mapping[str, Decimal]
    stage_hours: Mapping[str, Decimal]
    physical_stage_seconds: Mapping[str, int]
    screen_seed: int
    confirm_seed: int
    score_tiers: Mapping[str, Decimal]


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    expert: str
    fold: TemporalFold
    decay: Decimal | None
    seed: int


def build_stage_jobs(contract: PortfolioContract, stage: str) -> tuple[JobSpec, ...]:
    if contract.campaign_id != "temporal_portfolio_v1":
        raise PortfolioContractError("campaign identity differs")
    if stage != "T1":
        raise PortfolioContractError(f"stage jobs are not defined yet: {stage}")
    recent = [JobSpec(f"t1__recent__va{f.valid_year}__s3407", "recent", f, None, 3407) for f in contract.folds]
    multi = [
        JobSpec(f"t1__multi_d{str(d).replace('.', 'p')}__va{f.valid_year}__s3407", "multi", f, d, 3407)
        for d in contract.decays for f in contract.folds
    ]
    return tuple(recent + multi)
```

Seal the exact contract SHA-256 in `contracts.py`, matching the existing contract pattern.

- [ ] **Step 4: Run the focused tests**

```bash
pytest tests/test_temporal_portfolio_contracts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/__init__.py experiments/temporal_portfolio/contract.json experiments/temporal_portfolio/contracts.py tests/test_temporal_portfolio_contracts.py
git commit -m "feat: seal temporal portfolio contract"
```

### Task 2: Training identity and duplicate audit

**Files:**
- Create: `experiments/temporal_portfolio/identity.py`
- Create: `tests/test_temporal_portfolio_identity.py`

- [ ] **Step 1: Write failing identity tests**

```python
from experiments.temporal_portfolio.identity import TrainingIdentity, audit_duplicate


def test_training_identity_changes_only_for_semantic_changes() -> None:
    base = TrainingIdentity.from_payload({
        "data_rows": "a" * 64, "train_seasons": [2021], "valid_year": 2022,
        "decay": None, "features": ["base"], "model": {"profile": "p2"},
        "loss": "bce", "seed": 3407,
    })
    same = TrainingIdentity.from_payload(dict(base.payload))
    changed = TrainingIdentity.from_payload({**dict(base.payload), "decay": "0.55"})
    assert base.sha256 == same.sha256
    assert base.sha256 != changed.sha256
    assert audit_duplicate(base, {base.sha256: "jobs/old"}) == "jobs/old"
```

- [ ] **Step 2: Verify the tests fail**

```bash
pytest tests/test_temporal_portfolio_identity.py -q
```

Expected: FAIL because `identity.py` does not exist.

- [ ] **Step 3: Implement canonical semantic identities**

```python
@dataclass(frozen=True)
class TrainingIdentity:
    payload: Mapping[str, object]
    sha256: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TrainingIdentity":
        required = {"data_rows", "train_seasons", "valid_year", "decay", "features", "model", "loss", "seed"}
        if set(payload) != required:
            raise PortfolioIdentityError("training identity fields differ")
        encoded = canonical_json(dict(payload))
        return cls(MappingProxyType(json.loads(encoded)), sha256(encoded).hexdigest())


def audit_duplicate(identity: TrainingIdentity, completed: Mapping[str, str]) -> str | None:
    value = completed.get(identity.sha256)
    return None if value is None else str(value)
```

`canonical_json` must reject NaN, booleans where integers are expected, unordered sets, missing keys, and non-lowercase source hashes.

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_temporal_portfolio_identity.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/identity.py tests/test_temporal_portfolio_identity.py
git commit -m "feat: add semantic training identity"
```

### Task 3: Verified official data inputs

**Files:**
- Create: `experiments/temporal_portfolio/inputs.py`
- Create: `tools/prepare_temporal_portfolio_input.py`
- Create: `tests/test_temporal_portfolio_inputs.py`

- [ ] **Step 1: Write failing input tests**

```python
def test_verify_data_requires_exact_official_members(tiny_official_dir: Path) -> None:
    verified = verify_official_data(tiny_official_dir)
    assert verified.train.name == "train.csv"
    assert verified.test.name == "test.csv"
    assert verified.history.name == "trackman_history.csv"
    assert verified.train_rows > verified.test_rows > 0


def test_prepare_input_archive_contains_data_identity_not_submission(tmp_path: Path, tiny_official_dir: Path) -> None:
    result = prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")
    with ZipFile(result.path) as archive:
        assert set(archive.namelist()) == {"manifest.json", "data/train.csv", "data/test.csv", "data/trackman_history.csv", "data/sample_submission.csv"}
        assert "submission.csv" not in archive.namelist()
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_inputs.py -q
```

Expected: FAIL on missing input module.

- [ ] **Step 3: Implement strict verification and the local CLI**

```python
@dataclass(frozen=True)
class VerifiedOfficialData:
    root: Path
    train: Path
    test: Path
    history: Path
    sample_submission: Path
    member_sha256: Mapping[str, str]
    train_rows: int
    test_rows: int


def verify_official_data(root: Path) -> VerifiedOfficialData:
    required = ("train.csv", "test.csv", "trackman_history.csv", "sample_submission.csv")
    paths = {name: Path(root) / name for name in required}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise PortfolioInputError("official data members differ")
    train_header = pd.read_csv(paths["train.csv"], nrows=0)
    test_header = pd.read_csv(paths["test.csv"], nrows=0)
    if "control_success" not in train_header or "control_success" in test_header:
        raise PortfolioInputError("target boundary differs")
    train_rows = sum(1 for _ in paths["train.csv"].open("rb")) - 1
    test_rows = sum(1 for _ in paths["test.csv"].open("rb")) - 1
    return VerifiedOfficialData(
        Path(root), paths["train.csv"], paths["test.csv"],
        paths["trackman_history.csv"], paths["sample_submission.csv"],
        MappingProxyType({name: file_sha256(path) for name, path in paths.items()}),
        train_rows, test_rows,
    )
```

The CLI must insert the repository root into `sys.path`, accept `--data-dir` and `--output`, and print exactly:

```text
TEMPORAL_INPUT_READY path=<absolute> sha256=<64 hex> size_bytes=<integer>
```

- [ ] **Step 4: Run input and import tests**

```bash
pytest tests/test_temporal_portfolio_inputs.py -q
python tools/prepare_temporal_portfolio_input.py --help >/dev/null
```

Expected: PASS and exit code 0.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/inputs.py tools/prepare_temporal_portfolio_input.py tests/test_temporal_portfolio_inputs.py
git commit -m "feat: verify temporal portfolio inputs"
```

### Task 4: State, lineage, and runtime compatibility

**Files:**
- Create: `experiments/temporal_portfolio/state.py`
- Create: `experiments/temporal_portfolio/compatibility.py`
- Create: `tests/test_temporal_portfolio_state.py`

- [ ] **Step 1: Write failing state tests**

```python
def test_state_sequence_and_parent_are_monotonic(bindings: Bindings) -> None:
    fresh = CampaignState.fresh(bindings)
    next_state = fresh.advance(stage="T1", status="active", parent_manifest_sha256="1" * 64)
    assert next_state.sequence == 1
    with pytest.raises(PortfolioStateError, match="sequence"):
        validate_transition(next_state, replace(next_state, sequence=0))


def test_runtime_change_needs_explicit_compatibility(bindings: Bindings) -> None:
    checkpoint = CheckpointIdentity("1" * 64, "2" * 64, 1)
    with pytest.raises(CompatibilityError, match="runtime"):
        validate_runtime(checkpoint, current_runtime_sha256="3" * 64, migrations={})
    validate_runtime(checkpoint, current_runtime_sha256="3" * 64, migrations={("2" * 64, "3" * 64): 1})
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/test_temporal_portfolio_state.py -q
```

Expected: FAIL on missing modules.

- [ ] **Step 3: Implement immutable state and explicit compatibility**

```python
ALLOWED_STAGE_ORDER = ("fresh", "T1", "T2A", "T2B", "T3A", "T3BT4", "T5A", "T5B")
ALLOWED_STATUS = ("fresh", "active", "completed", "budget_inconclusive", "failed", "rule_blocked")


@dataclass(frozen=True)
class CampaignState:
    stage: str
    status: str
    sequence: int
    parent_manifest_sha256: str | None
    bindings: Bindings

    @classmethod
    def fresh(cls, bindings: Bindings) -> "CampaignState":
        return cls("fresh", "fresh", 0, None, bindings)

    def advance(self, *, stage: str, status: str, parent_manifest_sha256: str) -> "CampaignState":
        candidate = CampaignState(stage, status, self.sequence + 1, parent_manifest_sha256, self.bindings)
        validate_transition(self, candidate)
        return candidate


def validate_transition(old: CampaignState, new: CampaignState) -> None:
    if old.bindings != new.bindings:
        raise PortfolioStateError("state bindings differ")
    if new.sequence != old.sequence + 1:
        raise PortfolioStateError("state sequence differs")
    if ALLOWED_STAGE_ORDER.index(new.stage) < ALLOWED_STAGE_ORDER.index(old.stage):
        raise PortfolioStateError("stage moved backward")


def validate_runtime(checkpoint: CheckpointIdentity, *, current_runtime_sha256: str, migrations: Mapping[tuple[str, str], int]) -> None:
    if checkpoint.runtime_sha256 == current_runtime_sha256:
        return
    if migrations.get((checkpoint.runtime_sha256, current_runtime_sha256)) != checkpoint.state_schema_version:
        raise CompatibilityError("checkpoint runtime differs without a tested migration")
```

Keep `COMPATIBLE_RUNTIME_MIGRATIONS` empty in production. A future bug fix may add one pair only with model-key, optimizer, RNG, and fixture-prediction tests.

- [ ] **Step 4: Run state tests**

```bash
pytest tests/test_temporal_portfolio_state.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/state.py experiments/temporal_portfolio/compatibility.py tests/test_temporal_portfolio_state.py
git commit -m "feat: add temporal campaign state lineage"
```

### Task 5: Deterministic review, resume, and one-file handoff

**Files:**
- Create: `experiments/temporal_portfolio/artifacts.py`
- Create: `tests/test_temporal_portfolio_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

```python
def test_handoff_round_trip_is_deterministic(tmp_path: Path, evidence: StageEvidence) -> None:
    first = write_handoff(tmp_path / "a", evidence)
    second = write_handoff(tmp_path / "b", evidence)
    assert sha256(first.path.read_bytes()).hexdigest() == sha256(second.path.read_bytes()).hexdigest()
    verified = verify_handoff(first.path)
    assert verified.stage == "T1"
    assert set(verified.members) == {"review.zip", "resume.zip", "run.log", "stage_summary.json"}


@pytest.mark.parametrize("mode", ["zip", "directory"])
def test_discovery_reads_kaggle_zip_or_extracted_directory(tmp_path: Path, mode: str, evidence: StageEvidence) -> None:
    handoff = write_handoff(tmp_path / "source", evidence).path
    root = extract_if_requested(handoff, tmp_path / "input", mode)
    found = discover_handoffs([root])
    assert found[0].manifest_sha256 == verify_handoff(handoff).manifest_sha256
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_temporal_portfolio_artifacts.py -q
```

Expected: FAIL on missing artifact module.

- [ ] **Step 3: Implement deterministic nested artifacts**

```python
def write_handoff(root: Path, evidence: StageEvidence) -> HandoffPath:
    review = deterministic_zip_bytes("temporal_review_v1", evidence.review_members, evidence.bindings)
    resume = deterministic_zip_bytes("temporal_resume_v1", evidence.resume_members, evidence.bindings)
    members = {
        "review.zip": review,
        "resume.zip": resume,
        "run.log": evidence.run_log,
        "stage_summary.json": canonical_json(evidence.summary),
    }
    payload = deterministic_zip_bytes("temporal_handoff_v1", members, evidence.bindings, lineage=evidence.lineage)
    path = Path(root) / f"temporal_portfolio_stage_{evidence.stage}_handoff.zip"
    atomic_write(path, payload)
    verify_handoff(path)
    return HandoffPath(path, file_sha256(path))
```

Define the referenced immutable evidence types in the same module:

```python
@dataclass(frozen=True)
class StageEvidence:
    stage: str
    bindings: Bindings
    lineage: Lineage
    review_members: Mapping[str, bytes | BoundFile]
    resume_members: Mapping[str, bytes | BoundFile]
    run_log: bytes
    summary: Mapping[str, object]


@dataclass(frozen=True)
class HandoffPath:
    path: Path
    sha256: str
```

Before reading member bytes, reject duplicate names, absolute/traversal paths, symlinks, encrypted members, uncompressed size over the declared limit, and extreme compression ratio. Discovery must select the highest sequence only within one parent lineage; equal sequence with different manifest hashes is an error.

- [ ] **Step 4: Run artifact and legacy regression tests**

```bash
pytest tests/test_temporal_portfolio_artifacts.py tests/test_tabm_campaign_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/temporal_portfolio/artifacts.py tests/test_temporal_portfolio_artifacts.py
git commit -m "feat: add temporal portfolio handoffs"
```
