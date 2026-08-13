# DACON Rules Compliance Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with one agent. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make current DACON rule compliance a fail-closed prerequisite for candidate implementation, heavy user runs, acceptance, and the repository's sole submission-package path.

**Architecture:** A small `competition_rules` package validates versioned policy, experiment contracts, source/runtime behavior, and full-run evidence. Existing experiment entry points call the gate before official data access. A separate `submission` package accepts only hash-bound passing evidence, renders a reviewed adapter into one official `script.py`, runs evaluator-operability checks, and is the only code allowed to create a DACON-format ZIP.

**Tech Stack:** Python 3.11 standard library, NumPy/Pandas only in behavioral test fixtures and candidate adapters, pytest 8.4.1, JSON/JSONL evidence, SHA-256, existing notebook/cell renderers.

---

## File map

New policy and gate files:

- `competition_rules/__init__.py`: public gate API and error exports.
- `competition_rules/policy.json`: fixed official sources, limits, policy version, and digest input.
- `competition_rules/contract.py`: strict JSON, safe-path checks, policy loading, same-day review loading, and experiment-contract validation.
- `competition_rules/code_gate.py`: registered-scope source checks and lightweight behavioral invariance checks.
- `competition_rules/evidence_gate.py`: restartable full-audit chunks, final audit manifest, and acceptance evidence validation.
- `submission/__init__.py`: package API exports.
- `submission/contract.py`: strict package request, runtime receipt, and artifact binding.
- `submission/adapters.py`: closed reviewed adapter registry and adapter protocol.
- `submission/runtime.py`: row-separable execution, canaries, output validation, and deterministic `script.py` renderer.
- `submission/audit.py`: final same-day policy, evidence, benchmark, size, layout, and hash gate.
- `submission/package.py`: sole official archive writer.

New and modified evidence/configuration:

- `reports/rules/2026-08-13-policy-review.json`: initial human-reviewed official-policy receipt.
- `experiments/independent_dl/experiment_contract.json`: config-bound DL campaign contract.
- `experiments/preprocessing_campaign/experiment_contract.json`: config-bound preprocessing campaign contract.
- `experiments/catboost_preprocessing/experiment_contract.json`: component contract inherited only by the bound preprocessing campaign.
- `reports/acceptances/xgboost_aggressive_capacity_public_result.json`: historical result quarantined from reuse.

New tests:

- `tests/test_rules_policy.py`
- `tests/test_experiment_contract_gate.py`
- `tests/test_rules_code_gate.py`
- `tests/test_rules_entrypoints.py`
- `tests/test_row_independence_evidence.py`
- `tests/test_submission_runtime.py`
- `tests/test_submission_package.py`
- `tests/test_rules_repository_enforcement.py`

Existing entry points and generated runners to modify:

- `experiments/independent_dl/run_campaign.py`
- `experiments/preprocessing_campaign/run_campaign.py`
- `experiments/preprocessing_campaign/run_budgeted_campaign.py`
- `tools/render_preprocessing_kaggle_cell.py`
- `tools/render_budgeted_preprocessing_kaggle_cell.py`
- `tools/render_independent_dl_kaggle_notebook.py`
- `experiments/preprocessing_campaign/KAGGLE_CELL.py`
- `experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py`
- `notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb`

Documentation to modify:

- `AGENTS.md`
- `docs/EXPERIMENT_CONTRACT.md`
- `docs/ROADMAP.md`
- `reports/EXPERIMENT_LEDGER.md`
- `README.md`

No official CSV, model, OOF, GPU job, package installation, or real submission archive is used while implementing this plan. All archives and predictions in tests are synthetic fixtures.

---

### Task 1: Add the versioned official policy and strict loader

**Files:**

- Create: `competition_rules/__init__.py`
- Create: `competition_rules/policy.json`
- Create: `competition_rules/contract.py`
- Create: `reports/rules/2026-08-13-policy-review.json`
- Create: `tests/test_rules_policy.py`

- [ ] **Step 1: Write failing strict-policy tests**

Create tests that require exact source URLs, official limits, strict JSON, a canonical policy digest, safe regular files, and a same-KST-date review:

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from competition_rules.contract import (
    RulesContractError,
    load_policy,
    load_policy_review,
    policy_digest,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "competition_rules/policy.json"


def test_checked_in_policy_matches_official_rules() -> None:
    policy = load_policy(POLICY, project_root=ROOT)
    assert policy["policy_version"] == "dacon-236743-2026-08-13"
    assert policy["competition_id"] == "236743"
    assert policy["allowed_data_sources"] == [
        "official_train",
        "official_trackman",
    ]
    assert policy["limits"] == {
        "install_seconds": 600,
        "inference_seconds": 600,
        "package_bytes": 10_000_000_000,
        "extracted_bytes": 32_000_000_000,
        "evaluation_rows": 245789,
    }
    assert policy["manual_rules"] == {
        "team_max_members": 5,
        "duplicate_registration_allowed": False,
        "daily_submission_limit": 5,
    }
    assert len(policy["official_sources"]) == 6
    assert all(item["url"].startswith("https://dacon.io/") for item in policy["official_sources"])


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}'])
def test_policy_rejects_ambiguous_json(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "policy.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(RulesContractError):
        load_policy(path, project_root=tmp_path)


def test_review_must_match_policy_and_package_date(tmp_path: Path) -> None:
    policy = load_policy(POLICY, project_root=ROOT)
    now = datetime(2026, 8, 13, 23, 59, tzinfo=ZoneInfo("Asia/Seoul"))
    review = {
        "schema_version": 1,
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_digest(policy),
        "reviewed_at": "2026-08-13T15:20:00+09:00",
        "sources": policy["official_sources"],
        "verdict": "unchanged",
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    assert load_policy_review(path, policy=policy, package_time=now)["verdict"] == "unchanged"
    review["reviewed_at"] = "2026-08-12T23:59:59+09:00"
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(RulesContractError, match="same KST date"):
        load_policy_review(path, policy=policy, package_time=now)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_rules_policy.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'competition_rules'`.

- [ ] **Step 3: Implement the strict policy contract**

Implement `RulesContractError`, duplicate-key/non-finite JSON rejection, numeric overflow rejection, no-symlink project-bound reads, exact schema validation, canonical JSON hashing, and KST-date comparison. The public signatures are:

```python
class RulesContractError(ValueError):
    pass


def load_policy(path: str | Path, *, project_root: str | Path) -> dict[str, object]:
    """Load schema-1 policy from one project-contained regular file."""


def policy_digest(policy: Mapping[str, object]) -> str:
    """Hash UTF-8 canonical JSON with sorted keys and compact separators."""


def load_policy_review(
    path: str | Path,
    *,
    policy: Mapping[str, object],
    package_time: datetime,
) -> dict[str, object]:
    """Require exact policy identity and the same Asia/Seoul calendar date."""
```

`policy.json` must contain the six reviewed URLs from the design, the environment values, `decimal_places: 8`, `runtime_safety_seconds: 480`, official archive members, allowed data sources, prohibited external APIs, and the policy version. The review JSON records the 2026-08-13 titles and the policy digest; it is evidence for tests and history, not reusable on a later package date.

- [ ] **Step 4: Run policy tests and static checks**

Run:

```bash
.venv/bin/python -m pytest tests/test_rules_policy.py -q
python3 -m json.tool competition_rules/policy.json >/dev/null
python3 -m json.tool reports/rules/2026-08-13-policy-review.json >/dev/null
python3 -m compileall -q competition_rules
git diff --check
```

Expected: all tests pass and all static commands exit 0.

- [ ] **Step 5: Commit Task 1**

```bash
git add competition_rules reports/rules/2026-08-13-policy-review.json tests/test_rules_policy.py
git commit -m "feat: define dacon rules policy"
```

---

### Task 2: Require a contract for every experiment scope

**Files:**

- Modify: `competition_rules/contract.py`
- Create: `experiments/independent_dl/experiment_contract.json`
- Create: `experiments/preprocessing_campaign/experiment_contract.json`
- Create: `experiments/catboost_preprocessing/experiment_contract.json`
- Create: `tests/test_experiment_contract_gate.py`

- [ ] **Step 1: Write contract-schema and campaign-binding tests**

Tests must cover exact policy version, allowed sources, fit/evaluation/time scopes, API prohibition, pretraining provenance, explicit config hashes, candidate membership, component inheritance, strict booleans, and unlisted candidates:

```python
from pathlib import Path

import pytest

from competition_rules.contract import (
    RulesContractError,
    validate_experiment_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_active_experiment_contracts_are_current_and_hash_bound() -> None:
    expected = {
        "experiments/independent_dl/experiment_contract.json": {
            "experiments/independent_dl/configs/campaign_v1.json":
                "a3a7f82b6d20d59764f0c7e2964a3c2b9489b452ebd59f5a9c137cf25e6d5c55",
            "experiments/independent_dl/configs/preprocessing_ablation_v1.json":
                "e322a0d5b8e0759ad189528c24a353d2dfa9342d6115cc4b361a1cd77939c09f",
        },
        "experiments/preprocessing_campaign/experiment_contract.json": {
            "experiments/preprocessing_campaign/configs/budgeted_campaign_v1.json":
                "99deaa32e5d9616a0f50625ac38eda73df2d472dd54d75ff26fda00ef9f3958b",
        },
    }
    for relative, bindings in expected.items():
        contract = validate_experiment_contract(ROOT / relative, project_root=ROOT)
        assert contract["rules_version"] == "dacon-236743-2026-08-13"
        assert contract["config_sha256"] == bindings


def test_unlisted_candidate_cannot_inherit_campaign_contract() -> None:
    path = ROOT / "experiments/independent_dl/experiment_contract.json"
    with pytest.raises(RulesContractError, match="candidate is not covered"):
        validate_experiment_contract(
            path,
            project_root=ROOT,
            candidate_id="unknown__candidate",
            config_path=ROOT / "experiments/independent_dl/configs/campaign_v1.json",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("fit_scope", "evaluation_rows"),
        ("evaluation_scope", "whole_evaluation_set"),
        ("time_scope", "post_pitch"),
        ("external_api", True),
    ],
)
def test_rule_relevant_mutations_are_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    path, payload = write_valid_contract_fixture(tmp_path)
    assert validate_experiment_contract(path, project_root=tmp_path)
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RulesContractError, match=field):
        validate_experiment_contract(path, project_root=tmp_path)
```

`write_valid_contract_fixture` writes a minimal self-contained policy, config,
registered source file, and contract under `tmp_path`; it proves the unmodified
fixture passes before each mutation so a test cannot pass for an unrelated
missing path.

- [ ] **Step 2: Verify the focused suite is RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_experiment_contract_gate.py -q
```

Expected: imports fail for missing `validate_experiment_contract` and contract files.

- [ ] **Step 3: Implement exact experiment-contract validation**

Add an immutable return type or validated dictionary with this schema:

```json
{
  "schema_version": 1,
  "contract_id": "independent_dl_rules_v1",
  "rules_version": "dacon-236743-2026-08-13",
  "scope": "config_bound_campaign",
  "candidate_count": 64,
  "candidate_ids_sha256": "896866d9906e60f3361475a26445d3214461a87d730bff029018af68a0a8e51e",
  "allowed_derivations": ["confirmation_seed", "boundary_expansion"],
  "config_sha256": {"relative/config.json": "64 lowercase hex"},
  "inference_source_paths": ["experiments/independent_dl/models/*.py"],
  "data_sources": ["official_train", "official_trackman"],
  "fit_scope": "training_rows_only",
  "evaluation_scope": "current_row_only",
  "time_scope": "pre_pitch_only",
  "external_api": false,
  "pretrained_models": [],
  "retrieval_corpora": []
}
```

For large deterministic grids, store `candidate_ids_sha256` over canonical
UTF-8 JSON for the complete ordered candidate list plus the exact bound config,
and validate the list derived by the existing strict config parser against that
digest. Record the exact registered inference source paths; path globs are
expanded deterministically inside the project and cannot escape or traverse a
symlink. TabICLv2 remains excluded until its public checkpoint URL, version,
license, and weight hash are complete. The failure must affect only that
candidate, not other independent-DL candidates.

The initial independent-DL contract covers the 64 non-TabICLv2 candidates
derived from `campaign_v1.json`. The unproven TabICLv2 frontier candidate is
intentionally absent. A whole-campaign run filters uncontracted candidates
before official data access, reports each one as `blocked_by_rules`, and still
runs the covered candidates. It must never silently inherit the campaign
contract or stop unrelated candidates.

Confirmation and boundary-expansion children may inherit only through the
existing strict campaign parser and scheduler: require an approved parent ID,
an allowed derivation named in the contract, the exact config-listed seed or
capacity value, and an exact canonical child-spec hash. Tests reject a forged
parent, arbitrary suffix, out-of-config value, changed feature view, changed
pretrained checkpoint, and missing derivation chain. Apply the equivalent
strict parent/derivation rule to preprocessing-wave promotion.

- [ ] **Step 4: Run focused and repository contract tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_rules_policy.py tests/test_experiment_contract_gate.py tests/test_repository_contract.py -q
python3 -m json.tool experiments/independent_dl/experiment_contract.json >/dev/null
python3 -m json.tool experiments/preprocessing_campaign/experiment_contract.json >/dev/null
python3 -m json.tool experiments/catboost_preprocessing/experiment_contract.json >/dev/null
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add competition_rules/contract.py experiments/*/experiment_contract.json tests/test_experiment_contract_gate.py
git commit -m "feat: require experiment rules contracts"
```

---

### Task 3: Add lightweight source and row-independence code gates

**Files:**

- Create: `competition_rules/code_gate.py`
- Create: `tests/test_rules_code_gate.py`

- [ ] **Step 1: Write adversarial and compliant behavioral tests**

Use small synthetic frames to demonstrate that the gate rejects mean shift, rank normalization, cross-row retrieval, mutable state, batch-dependent prediction, and hard-coded row IDs, while accepting row-local vectorized arithmetic, missing values, unseen categories, frozen calibration, and deterministic per-row randomness:

```python
import numpy as np
import pandas as pd
import pytest

from competition_rules.code_gate import (
    RulesCodeGateError,
    assert_row_independent,
    inspect_inference_source,
)


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"row_id": ["a", "b", "c", "d"], "x": [0.1, np.nan, 0.7, 0.3]}
    )


def test_mean_shift_is_rejected() -> None:
    def predict(value: pd.DataFrame) -> np.ndarray:
        raw = value["x"].fillna(0.0).to_numpy(dtype="float64")
        return raw - raw.mean() + 0.5

    with pytest.raises(RulesCodeGateError, match="row independence"):
        assert_row_independent(frame(), load_predictor=lambda: predict)


def test_row_local_vectorized_transform_passes() -> None:
    def predict(value: pd.DataFrame) -> np.ndarray:
        raw = value["x"].fillna(-1.0).to_numpy(dtype="float64")
        return 1.0 / (1.0 + np.exp(-raw))

    report = assert_row_independent(frame(), load_predictor=lambda: predict)
    assert report["status"] == "passed"
    assert report["decimal_places"] == 8


def test_deterministic_row_seed_passes() -> None:
    def predict(value: pd.DataFrame) -> np.ndarray:
        return np.asarray([
            (int.from_bytes(str(row_id).encode(), "little") % 1000) / 1000
            for row_id in value["row_id"]
        ])

    assert assert_row_independent(frame(), load_predictor=lambda: predict)["status"] == "passed"
```

Add source fixtures that reject `requests`, URL/HTTP/socket clients, remote API
SDKs, subprocess or shell network downloads, direct reads of `test.csv`, output
lookup dictionaries keyed by known row IDs, and prediction arrays with
evaluation length. Reject branches keyed to the public five-row sample, the
official 245,789-row evaluation count, row position, or batch length. Merely
locating the official four-file dataset directory is allowed; opening
evaluation rows is reserved for the central runtime.
Training-only `groupby` is not scanned as inference code; every inference path
registered in the experiment contract is scanned.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_rules_code_gate.py -q
```

Expected: collection fails for missing `competition_rules.code_gate`.

- [ ] **Step 3: Implement the code gate**

Expose these APIs:

```python
def inspect_inference_source(
    paths: Sequence[str | Path], *, project_root: str | Path
) -> dict[str, object]:
    """AST/token defense-in-depth scan of registered inference source only."""


def canonical_probability(value: float, *, decimal_places: int = 8) -> str:
    """Reject bool/non-real/non-finite/out-of-range and return fixed decimal."""


def assert_row_independent(
    frame: "pd.DataFrame",
    *,
    load_predictor: Callable[[], Callable[["pd.DataFrame"], object]],
    batch_sizes: Sequence[int] = (1, 2, 3, 17),
    decimal_places: int = 8,
    state_digest: Callable[[object], str] | None = None,
) -> dict[str, object]:
    """Compare repeat, reverse, fixed shuffle, rebatch, and singleton outputs."""
```

Every pass reloads the predictor. Validate non-null unique string-normalized `row_id`, exact one-dimensional length, canonical probabilities, ID alignment, and state digest before/after. Return only hashes and counts, not prediction values. Normalize only expected policy failures; programmer exceptions remain visible.

- [ ] **Step 4: Run code-gate and existing feature tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_rules_code_gate.py tests/test_independent_dl_features.py tests/test_preprocessing_profiles.py -q
python3 -m compileall -q competition_rules
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add competition_rules/code_gate.py tests/test_rules_code_gate.py
git commit -m "feat: gate row-independent inference code"
```

---

### Task 4: Put the contract gate in every active heavy-run entry point

**Files:**

- Modify: `experiments/independent_dl/run_campaign.py`
- Modify: `experiments/preprocessing_campaign/run_campaign.py`
- Modify: `experiments/preprocessing_campaign/run_budgeted_campaign.py`
- Modify: `tools/render_preprocessing_kaggle_cell.py`
- Modify: `tools/render_budgeted_preprocessing_kaggle_cell.py`
- Modify: `tools/render_independent_dl_kaggle_notebook.py`
- Regenerate: `experiments/preprocessing_campaign/KAGGLE_CELL.py`
- Regenerate: `experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py`
- Regenerate: `notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb`
- Create: `tests/test_rules_entrypoints.py`
- Modify: existing notebook and Kaggle-cell tests only where the new gate contract changes expected source.

- [ ] **Step 1: Write tests proving gate-before-data-access ordering**

Monkeypatch the gate to raise and the runtime/data loader to fail if called.
Every action that starts/resumes training, registers or promotes candidates,
writes summaries, or creates a candidate handoff must gate first. Only a
strictly read-only `status` action may read existing small manifests without
official CSV access, and it cannot advance candidate state.

```python
def test_independent_dl_run_gates_before_campaign_load(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(module, "assert_experiment_runnable", lambda **_: (_ for _ in ()).throw(RulesContractError("blocked")))
    monkeypatch.setattr(module, "load_campaign", lambda *_: calls.append("load"))
    with pytest.raises(RulesContractError, match="blocked"):
        module.main([
            "run", "--config", "missing.json", "--data-dir", "missing",
            "--output-dir", str(tmp_path),
        ])
    assert calls == []
```

Add equivalent tests for both preprocessing runners and assert rendered standalone Kaggle/Colab sources contain the bound policy version, contract bytes, contract digest, and a gate invocation before `train.csv` is opened.

Add an independent-DL regression proving an uncontracted TabICLv2 candidate is
filtered and reported before runtime construction while a covered TabM
candidate in the same command remains runnable.

Add transition tests proving config-authorized confirmation/expansion and
preprocessing-wave children pass, while a hand-edited child with the same ID
but different model, feature, data, or inference fields is blocked before
state publication.

- [ ] **Step 2: Run the entry-point tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_rules_entrypoints.py -q
```

Expected: failures show missing `assert_experiment_runnable` calls and absent embedded contracts.

- [ ] **Step 3: Implement one public runnable assertion**

Add this API to `competition_rules.contract` and call it as the first mutating action:

```python
def assert_experiment_runnable(
    *,
    project_root: str | Path,
    contract_path: str | Path,
    config_path: str | Path,
    candidate_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate policy, contract, config/candidates, and registered inference source."""
```

The assertion runs the Task 3 source gate over every contract-registered
inference path before official data is opened. Its report includes the live
ordered source digest that later evidence must bind. The standalone renderers
must include `competition_rules/`, the relevant contract, and the bound config
in the embedded tarball. They execute the same loader after extraction and
before locating official data. Do not implement a weaker notebook-only
validator.

- [ ] **Step 4: Regenerate execution artifacts deterministically**

Run the repository's renderer scripts twice and compare hashes:

```bash
python3 tools/render_preprocessing_kaggle_cell.py
python3 tools/render_budgeted_preprocessing_kaggle_cell.py
python3 tools/render_independent_dl_kaggle_notebook.py
shasum -a 256 experiments/preprocessing_campaign/KAGGLE_CELL.py \
  experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py \
  notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb
```

Run again and require identical hashes. Update the two Colab builders or deterministic source cells using their existing repository mechanism; do not hand-edit notebook JSON.

- [ ] **Step 5: Run entry-point and notebook suites**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_rules_entrypoints.py \
  tests/test_independent_dl_cli.py \
  tests/test_preprocessing_campaign.py \
  tests/test_budgeted_preprocessing_kaggle_cell.py \
  tests/test_preprocessing_kaggle_cell.py \
  tests/test_independent_dl_kaggle_notebook.py \
  tests/test_independent_dl_notebook.py \
  tests/test_preprocessing_eda_colab.py -q
git diff --check
```

Expected: all pass without reading official data or starting training.

- [ ] **Step 6: Commit Task 4**

Stage only the entry points, renderers, generated artifacts, and their tests, then:

```bash
git commit -m "feat: gate all active experiment runners"
```

---

### Task 5: Implement restartable full row-independence evidence

**Files:**

- Create: `competition_rules/evidence_gate.py`
- Create: `tests/test_row_independence_evidence.py`

- [ ] **Step 1: Write chunk, resume, and forgery tests**

Use 11 synthetic rows and chunk size 4. Cover interruption after two chunks, safe resume, exact expected chunk set, duplicate/missing/reordered IDs, modified prediction bytes, mismatched policy/code/model/data hashes, partial temporary files, and final manifest publication.

```python
def test_interrupted_full_audit_resumes_verified_chunks(tmp_path):
    first = run_full_independence_audit(
        frame=_frame(11),
        output_dir=tmp_path,
        identity=_identity(),
        load_predictor=_loader,
        chunk_size=4,
        stop_after_chunks=2,
    )
    assert first.status == "incomplete"
    assert first.completed_chunks == 2
    second = run_full_independence_audit(
        frame=_frame(11),
        output_dir=tmp_path,
        identity=_identity(),
        load_predictor=_loader,
        chunk_size=4,
    )
    assert second.status == "passed"
    assert second.row_count == 11
    assert second.reused_chunks == 2
```

- [ ] **Step 2: Run the focused suite and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_row_independence_evidence.py -q
```

Expected: collection fails for missing `competition_rules.evidence_gate`.

- [ ] **Step 3: Implement immutable audit identity and chunk publication**

Use these public types and functions:

```python
@dataclass(frozen=True)
class AuditIdentity:
    policy_version: str
    candidate_id: str
    data_sha256: str
    code_sha256: str
    config_sha256: str
    preprocessing_sha256: str
    model_sha256: str
    adapter_sha256: str
    runtime_sha256: str


def run_full_independence_audit(
    *,
    frame: "pd.DataFrame",
    output_dir: str | Path,
    identity: AuditIdentity,
    load_predictor: Callable[[], object],
    chunk_size: int = 4096,
    stop_after_chunks: int | None = None,
) -> FullAuditResult:
    """Atomically create/reuse exact chunks and publish only a complete manifest."""


def validate_full_audit(
    manifest_path: str | Path,
    *,
    expected_identity: AuditIdentity,
) -> dict[str, object]:
    """Rehash every chunk and require complete singleton/batch equivalence."""
```

Each chunk stores canonical predictions for production batch, repeat, reverse, fixed shuffle, rebatch, and singleton for the exact ID slice. The final report stores counts and hashes only. Temporary files use exclusive creation and atomic replacement; an existing destination is never overwritten.

- [ ] **Step 4: Run evidence, code-gate, and compile tests**

```bash
.venv/bin/python -m pytest tests/test_row_independence_evidence.py tests/test_rules_code_gate.py -q
python3 -m compileall -q competition_rules
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add competition_rules/evidence_gate.py tests/test_row_independence_evidence.py
git commit -m "feat: record full row independence evidence"
```

---

### Task 6: Build the closed adapter registry and evaluator runtime

**Files:**

- Create: `submission/__init__.py`
- Create: `submission/adapters.py`
- Create: `submission/runtime.py`
- Create: `tests/test_submission_runtime.py`

- [ ] **Step 1: Write runtime tests for supported safe behavior**

Use fake NumPy adapters, not installed CatBoost/XGBoost/Torch. Test adapter allowlisting, arbitrary positive row counts, unique IDs, missing/unseen values, fixed per-row segmentation, fixed calibration/ensemble, deterministic row-seeded TTA, canary mismatch, state mutation, non-finite/out-of-range probabilities, sample-submission alignment, and atomic no-output-on-failure behavior.

```python
class LinearFixtureAdapter:
    adapter_id = "fixture_linear_v1"

    def state_digest(self) -> str:
        return "a" * 64

    def predict_batch(self, frame):
        x = frame["x"].fillna(0.0).to_numpy(dtype="float64")
        return 1.0 / (1.0 + np.exp(-x))


def test_runtime_supports_arbitrary_row_count_and_exact_output_order(tmp_path):
    test = pd.DataFrame({"row_id": ["c", "a", "b"], "x": [0.3, 0.1, 0.2]})
    sample = pd.DataFrame({"row_id": ["a", "b", "c"], "control_success": [0.0] * 3})
    report = run_evaluator(
        test_frame=test,
        sample_submission=sample,
        adapter=LinearFixtureAdapter(),
        output_path=tmp_path / "output/submission.csv",
        canary_count=3,
    )
    assert report["status"] == "passed"
    output = pd.read_csv(tmp_path / "output/submission.csv")
    assert output["row_id"].tolist() == ["a", "b", "c"]
```

- [ ] **Step 2: Run runtime tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_submission_runtime.py -q
```

Expected: collection fails because `submission` does not exist.

- [ ] **Step 3: Implement the protocol, closed registry, and runtime**

Define the adapter boundary:

```python
class SubmissionAdapter(Protocol):
    adapter_id: str

    def state_digest(self) -> str:
        raise NotImplementedError

    def predict_batch(self, frame: "pd.DataFrame") -> object:
        raise NotImplementedError


ADAPTER_FACTORIES: Mapping[str, Callable[[Path, Mapping[str, object]], SubmissionAdapter]]
```

The registry initially includes only fixture adapters in tests; production adapter IDs are added in separate reviewed candidate work. The generic runtime and packager exist now, but no real candidate can package until its adapter, acceptance, model, and benchmark all pass. This preserves the AGENTS rule forbidding premature submission creation.

`run_evaluator` must compute fixed canary indices from the immutable package seed and current row IDs, compare canonical batch/singleton strings, verify state before/after, predict all rows once, align to the sample IDs, write eight-decimal probabilities, fsync the temporary CSV, and atomically publish. No partial output remains on failure.

`render_script` receives one registered adapter ID and immutable artifact metadata, uses a fixed source template in `runtime.py`, and returns UTF-8 bytes. It must not accept arbitrary source or import paths.

- [ ] **Step 4: Run runtime and policy suites**

```bash
.venv/bin/python -m pytest tests/test_submission_runtime.py tests/test_rules_code_gate.py tests/test_rules_policy.py -q
python3 -m compileall -q submission
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add submission tests/test_submission_runtime.py
git commit -m "feat: add row-separable submission runtime"
```

---

### Task 7: Implement the final audit and sole packager

**Files:**

- Create: `submission/contract.py`
- Create: `submission/audit.py`
- Create: `submission/package.py`
- Create: `tests/test_submission_package.py`
- Create: `tests/test_rules_repository_enforcement.py`

- [ ] **Step 1: Write fail-closed package tests**

Build only tiny synthetic model files. Require rejection before archive creation for stale policy review, wrong policy version, incomplete acceptance, false gate, mismatched live hashes, missing/full-audit mismatch, unknown adapter, unlicensed pretrained weights, benchmark over 480 seconds, install over 600 seconds, memory/VRAM/size excess, unsafe paths, symlinks, duplicate archive names, wrong layout, existing output, and alternate package writer.

```python
def test_package_is_not_created_when_policy_review_is_stale(tmp_path):
    request = valid_request(tmp_path)
    request.policy_review_path.write_text(stale_review_json(), encoding="utf-8")
    with pytest.raises(SubmissionAuditError, match="same KST date"):
        build_submission_package(request)
    assert not request.archive_path.exists()


def test_valid_fixture_package_has_exact_top_level_layout(tmp_path):
    request = valid_request(tmp_path)
    result = build_submission_package(request)
    with zipfile.ZipFile(result.archive_path) as archive:
        names = archive.namelist()
    assert "script.py" in names
    assert "requirements.txt" in names
    assert any(name.startswith("model/") for name in names)
    assert all(name == "script.py" or name == "requirements.txt" or name.startswith("model/") for name in names)
    assert result.receipt_path.parent == result.archive_path.parent
```

- [ ] **Step 2: Run package tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_submission_package.py -q
```

Expected: collection fails for missing package modules.

- [ ] **Step 3: Implement package contracts and audit order**

Use immutable dataclasses:

```python
@dataclass(frozen=True)
class PackageRequest:
    project_root: Path
    policy_path: Path
    policy_review_path: Path
    acceptance_path: Path
    full_audit_manifest_path: Path
    runtime_benchmark_path: Path
    model_dir: Path
    requirements_path: Path
    adapter_id: str
    archive_path: Path
    receipt_path: Path
```

`audit_package_request` performs all read-only checks and returns an immutable snapshot containing bytes and SHA-256 for every input. `build_submission_package` accepts only that snapshot, renders `script.py`, copies model files, writes deterministic ZIP metadata, reopens and validates the ZIP, atomically publishes it, then writes the external receipt. If receipt publication fails, remove only the archive whose inode/identity and hash match this invocation.

The acceptance contract is strict and candidate-local: `status` must be
`passed`; all required temporal, performance, provenance, row-independence,
and evaluator gates must be exact booleans set to true; IDs and current policy
must match; and every data/code/config/preprocessing/model/adapter/runtime hash
must equal the live input snapshot. Unknown or extra gate names cannot replace
a required gate. A rejected candidate never blocks a different candidate's
request.

No CLI flag may weaken a check. The module must not upload or submit anything.

- [ ] **Step 4: Add repository-wide sole-packager detection**

In the package tests, scan tracked Python and notebook source for creation of the official `script.py` + `requirements.txt` + `model/` archive layout. Maintain one exact allowlist entry: `submission/package.py`. Generic research/checkpoint ZIP or tar creation is allowed when it cannot produce the official layout.

Also execute the rendered fixture `script.py` from an official-layout sandbox
containing `data/test.csv` and `data/sample_submission.csv`. Require it to write
only `output/submission.csv`, preserve sample-submission ID order and columns,
and succeed for both five rows and a different positive row count. This binds
the runtime to the baseline path contract without using official data.

- [ ] **Step 5: Run package, runtime, and source-boundary tests**

```bash
.venv/bin/python -m pytest \
  tests/test_submission_package.py \
  tests/test_submission_runtime.py \
  tests/test_rules_repository_enforcement.py -q
python3 -m compileall -q submission competition_rules
git diff --check
```

Expected: all pass and no real submission archive is created.

- [ ] **Step 6: Commit Task 7**

```bash
git add submission tests/test_submission_package.py tests/test_rules_repository_enforcement.py
git commit -m "feat: enforce sole dacon submission packager"
```

---

### Task 8: Quarantine unsafe history and make the rule contract visible

**Files:**

- Modify: `AGENTS.md`
- Modify: `docs/EXPERIMENT_CONTRACT.md`
- Modify: `docs/ROADMAP.md`
- Modify: `reports/EXPERIMENT_LEDGER.md`
- Modify: `README.md`
- Modify: `reports/acceptances/xgboost_aggressive_capacity_public_result.json`
- Modify: `tests/test_repository_contract.py`
- Modify: `tests/test_rules_repository_enforcement.py`

- [ ] **Step 1: Write documentation and quarantine tests first**

Require the concise core phrases, official links, four transition gates, current policy version, sole package path, user-owned heavy execution, and explicit XGBoost quarantine:

```python
def test_historical_xgboost_v3_is_visible_but_package_blocked() -> None:
    report = load_json("reports/acceptances/xgboost_aggressive_capacity_public_result.json")
    assert report["status"] == "public_scored"
    assert report["rules_review_required"] is True
    assert report["package_blocked"] is True
    assert report["blocked_reason"] == "evaluation_prediction_mean_shift"


def test_operational_docs_require_current_rules_gates() -> None:
    agents = read_text("AGENTS.md")
    contract = read_text("docs/EXPERIMENT_CONTRACT.md")
    for phrase in (
        "dacon-236743-2026-08-13",
        "experiment_contract.json",
        "current_row_only",
        "competition_rules",
        "submission/package.py",
        "규칙을 통과하지 못하면",
    ):
        assert phrase in agents + contract
```

- [ ] **Step 2: Run documentation tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_repository_contract.py tests/test_rules_repository_enforcement.py -q
```

Expected: failures for missing rule text and quarantine fields.

- [ ] **Step 3: Update concise operational documentation**

Keep `AGENTS.md` short: require reading policy/contract, contract-before-code, code-gate-before-user-run, evidence-before-passed, same-day review and final package gate, no alternate package route, and no heavy Codex run without approval. Keep broad/deep performance exploration unchanged after the safety gates.

Update the experiment contract with the four gates and permitted/prohibited examples. Update README/roadmap/ledger so XGBoost 820.9583 remains historical but is visibly not reusable. Do not delete or rewrite its score.

Document the non-automatable DACON checks separately: correct team/account,
no duplicate registration, remaining daily submission quota, open deadline,
and the exact archive selected in the upload UI. The repository never uploads
automatically and never records these platform checks as machine-passed.

- [ ] **Step 4: Run documentation and full static checks**

```bash
.venv/bin/python -m pytest tests/test_repository_contract.py tests/test_rules_repository_enforcement.py -q
python3 -m json.tool reports/acceptances/xgboost_aggressive_capacity_public_result.json >/dev/null
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit Task 8**

```bash
git add AGENTS.md README.md docs/EXPERIMENT_CONTRACT.md docs/ROADMAP.md \
  reports/EXPERIMENT_LEDGER.md \
  reports/acceptances/xgboost_aggressive_capacity_public_result.json \
  tests/test_repository_contract.py tests/test_rules_repository_enforcement.py
git commit -m "docs: make dacon rules gates mandatory"
```

---

### Task 9: Run the complete lightweight verification and prepare the user-run handoff

**Files:**

- Modify only confirmed defects found by verification in their owning files.
- Do not create a real submission ZIP.
- Do not run official data or GPU work.

- [ ] **Step 1: Run all focused guardrail suites**

```bash
.venv/bin/python -m pytest \
  tests/test_rules_policy.py \
  tests/test_experiment_contract_gate.py \
  tests/test_rules_code_gate.py \
  tests/test_rules_entrypoints.py \
  tests/test_row_independence_evidence.py \
  tests/test_submission_runtime.py \
  tests/test_submission_package.py \
  tests/test_rules_repository_enforcement.py -q
```

Expected: all pass.

- [ ] **Step 2: Run the full repository suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass; existing documented skips remain skips.

- [ ] **Step 3: Run static, deterministic-render, and diff checks**

```bash
python3 -m compileall -q competition_rules submission experiments tools tests
python3 -m json.tool competition_rules/policy.json >/dev/null
find experiments -name experiment_contract.json -print0 | xargs -0 -n1 python3 -m json.tool >/dev/null
git diff --check
git status --short
```

Regenerate each generated execution artifact once more and require no diff. Expected: static commands exit 0 and only intentional committed changes exist.

- [ ] **Step 4: Perform the final rule-to-test audit**

Create a read-only checklist in the implementation review notes mapping each official rule to a passing test:

```text
evaluation rows independent -> row independence + runtime canary tests
official data only -> experiment contract tests
pre-pitch information only -> contract/time-source tests
pretrained license/source -> provenance tests
no remote API -> source and contract tests
10-minute install/inference -> runtime receipt gate tests
10GB/32GB -> package size tests
official layout/output -> package/runtime format tests
reproducible code -> live hash/evidence tests
team/quota/deadline/upload selection -> explicit human pre-upload checklist
```

If any row lacks a test, add the smallest focused test before completion.

- [ ] **Step 5: Commit only necessary final fixes**

If verification required fixes, commit them with:

```bash
git commit -m "fix: close dacon guardrail verification gaps"
```

If there were no fixes, do not create an empty commit.

- [ ] **Step 6: Hand off the heavy user run separately**

After implementation is complete, provide one complete copyable Colab/Kaggle cell that runs only the full validation-season independence audit and writes its chunked manifest. State purpose, required paths, expected runtime, resume safety, success text, and exact error output to return. A different later cell may perform evaluator-equivalent install/runtime benchmarking. Neither cell creates a real submission ZIP.

The first real package request remains blocked until a candidate has new-policy performance acceptance, full independence evidence, same-day official-rule review, and evaluator-operability evidence with matching live hashes.
