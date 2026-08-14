# Colab TabM Version D Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one self-contained Colab T4 cell that verifies the accepted Stage C result, trains the fixed single-seed TabM on official train data only, survives epoch-boundary interruptions, and emits review-only Version D evidence without creating a submission package.

**Architecture:** A sealed JSON contract binds the official data archive, accepted Stage C delivery, model configuration, three-epoch fit policy, and resource gates. `version_d.py` verifies inputs and owns deterministic emergency, frozen-model, and delivery archives; `final_training.py` owns resumable train-only fitting; `final_review.py` owns post-freeze independence and scale gates; `colab_version_d.py` coordinates the state machine. A deterministic renderer embeds these sources in a direct-upload Colab cell smaller than 1 MB.

**Tech Stack:** Python 3.11-compatible code, PyTorch, TabM 0.0.3, rtdl-num-embeddings 0.0.12, pandas, NumPy, pytest, standard-library ZIP/tar/hash utilities, Google Colab `files.upload` and `files.download`.

---

## File map

- Create `experiments/tabm_campaign/version_d_contract.json`: immutable Stage C, model, data, epoch, resource, and output contract.
- Create `experiments/tabm_campaign/version_d.py`: input verification, two-phase extraction, snapshot verification, deterministic archives, and final delivery.
- Create `reports/rules/2026-08-14-policy-review.json`: same-day official-rules review bound to the existing policy digest.
- Modify `experiments/tabm_campaign/final_training.py`: single-member fixed-policy fitting with atomic epoch checkpoints and exact resume.
- Modify `experiments/tabm_campaign/final_review.py`: separate frozen-model review from fitting and enforce Version D gates.
- Create `experiments/tabm_campaign/colab_version_d.py`: Version D state machine and recovery selection.
- Create `tools/render_tabm_colab_version_d_cell.py`: deterministic one-cell renderer.
- Create `experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py`: generated user-facing Colab cell.
- Create `tests/test_tabm_campaign_version_d.py`: contract, lineage, extraction, and archive tests.
- Create `tests/test_tabm_campaign_final_training.py`: checkpoint identity, atomicity, and resume tests.
- Modify `tests/test_tabm_campaign_final_review.py`: fixed three-epoch and post-freeze review tests.
- Create `tests/test_tabm_campaign_colab_version_d.py`: state machine and generated-cell tests.
- Create `docs/TABM_VERSION_D_COLAB.md`: exact upload, recovery, expected-log, and return-artifact instructions.

Existing modified Stage C files and `notebooks/INDEPENDENT_DL_RECOVERY_AND_TABR.ipynb` are outside this plan. Every commit below uses path-specific `git add` and must leave those files untouched.

### Task 1: Seal and verify the Version D contract

**Files:**
- Create: `experiments/tabm_campaign/version_d_contract.json`
- Create: `experiments/tabm_campaign/version_d.py`
- Create: `reports/rules/2026-08-14-policy-review.json`
- Create: `tests/test_tabm_campaign_version_d.py`

- [ ] **Step 1: Write failing contract and input-verification tests**

Add tests that use a small, deterministic nested delivery fixture rather than the 65 MB real delivery:

```python
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from experiments.tabm_campaign.version_d import (
    VersionDError,
    extract_review_inputs,
    extract_training_input,
    load_version_d_contract,
    verify_stage_c_delivery,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)


def test_contract_seals_single_s3407_and_three_epochs() -> None:
    contract = load_version_d_contract()
    assert contract["predictor_id"] == "single_s3407"
    assert contract["final_fit"]["epochs"] == 3
    assert contract["selected_candidate"]["seed"] == 3407
    assert contract["preprocessing"] == {
        "profile": "dl_standard",
        "components": ["hand_matchup"],
        "fit_scope": "official_train_only",
    }


def test_same_day_policy_review_matches_policy_digest() -> None:
    from competition_rules.contract import load_policy, policy_digest

    policy = load_policy(Path("competition_rules/policy.json"), project_root=Path.cwd())
    review = json.loads(Path("reports/rules/2026-08-14-policy-review.json").read_text(encoding="utf-8"))
    assert review["verdict"] == "unchanged"
    assert review["policy_sha256"] == policy_digest(policy)


def test_training_extraction_does_not_materialize_test(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    _write_zip(
        archive,
        {
            "train.csv": b"season,target\n2024,1\n",
            "trackman_history.csv": b"x\n1\n",
            "test.csv": b"row_id\na\n",
            "sample_submission.csv": b"row_id,target\na,0.5\n",
        },
    )
    with ZipFile(archive) as source:
        members = {
            item.filename: {"size": item.file_size}
            for item in source.infolist()
        }
    contract = {
        "data_archive": {
            "sha256": sha256(archive.read_bytes()).hexdigest(),
            "members": members,
        }
    }
    train_root = tmp_path / "train_only"
    extract_training_input(archive, train_root, contract)
    assert (train_root / "train.csv").is_file()
    assert sorted(path.name for path in train_root.iterdir()) == ["train.csv"]

    review_root = tmp_path / "review"
    extract_review_inputs(archive, review_root, contract)
    assert sorted(path.name for path in review_root.iterdir()) == [
        "sample_submission.csv",
        "test.csv",
    ]


def test_stage_c_delivery_rejects_member_tampering(tmp_path: Path) -> None:
    delivery = tmp_path / "delivery.zip"
    _write_zip(delivery, {"delivery_manifest.json": json.dumps({"schema_version": 1}).encode()})
    with pytest.raises(VersionDError, match="Stage C delivery SHA-256 differs"):
        verify_stage_c_delivery(delivery, {"stage_c_delivery": {"sha256": "0" * 64}})
```

- [ ] **Step 2: Run the focused tests and confirm the expected import failure**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_version_d.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'experiments.tabm_campaign.version_d'`.

- [ ] **Step 3: Add the sealed JSON contract**

Create the contract with these exact decisions and limits. Copy the four official member sizes and SHA-256 values from `colab_stage_c_contract.json`; add the deterministic official archive SHA-256 and accepted delivery evidence shown below.

Create `reports/rules/2026-08-14-policy-review.json` with the following content. It records the
same-day browser review without changing the policy when no official rule changed.

```json
{
  "schema_version": 1,
  "policy_version": "dacon-236743-2026-08-13",
  "policy_sha256": "59b5988d4350955a0a95587d4d916221bdb51064b630028debc82ae66425d2b4",
  "reviewed_at": "2026-08-14T21:49:40+09:00",
  "sources": [
    {"title": "대회 규칙", "url": "https://dacon.io/competitions/official/236743/overview/rules"},
    {"title": "평가 및 코드 제출 안내", "url": "https://dacon.io/competitions/official/236743/overview/evaluation"},
    {"title": "평가 데이터 독립 예측 원칙", "url": "https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent"},
    {"title": "데이터 설명", "url": "https://dacon.io/competitions/official/236743/data"},
    {"title": "대회 설명", "url": "https://dacon.io/competitions/official/236743/overview/description"},
    {"title": "부정 제출 및 치팅 행위에 관하여", "url": "https://dacon.io/notice/notice/13"}
  ],
  "verdict": "unchanged"
}
```

```json
{
  "schema_version": 1,
  "version": "D",
  "review_only": true,
  "submission_package": false,
  "data_archive": {
    "filename": "lg-aimers-9th-data.zip",
    "sha256": "0a1df39e82ed6621629378650c7b938c21bee56f0d57e414c61882f6de409176",
    "members": {
      "sample_submission.csv": {"size": 112, "sha256": "b2cf6ba6745c74c46db23e620d74705f9b327efb8c8a14dc0dd2afab0e898775"},
      "test.csv": {"size": 1894, "sha256": "478d10b20c00443f6fe8270ab7348de49d14395526130f0b2d915da496821a19"},
      "trackman_history.csv": {"size": 353823031, "sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9"},
      "train.csv": {"size": 368527723, "sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"}
    }
  },
  "stage_c_delivery": {
    "filename": "tabm_colab_stage_C_delivery.zip",
    "sha256": "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a",
    "review_sha256": "461c4422cebdc7383577ad6c53b044bfe3d9d15b667383e454591135e4242d2c",
    "resume_sha256": "25aa43a846357f0ac97bcb18dd1826727d92e0850cb51091bec002ba6e2ed838",
    "stage_state_sha256": "290b3b10854aca9131e19f3fd1b0620aed17e9b9d40e25f0cb41f67bb2fcba76",
    "completed_jobs": 21
  },
  "predictor_id": "single_s3407",
  "preprocessing": {"profile": "dl_standard", "components": ["hand_matchup"], "fit_scope": "official_train_only"},
  "selected_candidate": {
    "family": "tabm",
    "capacity": "p2",
    "k": 32,
    "width": 512,
    "blocks": 4,
    "dropout": 0.1,
    "num_embedding": "piecewise_linear",
    "loss": "bce",
    "scheduler": "plateau",
    "learning_rate": 0.0006,
    "seed": 3407
  },
  "final_fit": {
    "epochs": 3,
    "scheduler": "constant",
    "learning_rate": 0.0006,
    "weight_decay": 0.0001,
    "effective_batch_size": 4096,
    "micro_batch_size": 512
  },
  "limits": {
    "install_seconds": 480,
    "inference_seconds": 480,
    "gpu_bytes": 21474836480,
    "rss_bytes": 22000000000,
    "artifact_bytes": 2000000000
  },
  "outputs": {
    "emergency_prefix": "tabm_version_D_emergency_epoch_",
    "frozen_model": "tabm_version_D_frozen_model.zip",
    "review_bundle": "tabm_hand_matchup_final_review_bundle.zip",
    "delivery": "tabm_hand_matchup_stage_D_review_delivery.zip"
  }
}
```

- [ ] **Step 4: Implement strict loading, delivery verification, and two-phase extraction**

Implement these public interfaces in `version_d.py`:

```python
class VersionDError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_version_d_contract(path: Path | None = None) -> dict[str, object]:
    source = Path(__file__).with_name("version_d_contract.json") if path is None else path
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("version") != "D" or value.get("review_only") is not True:
        raise VersionDError("Version D contract identity differs")
    final_fit = value.get("final_fit")
    if value.get("submission_package") is not False or not isinstance(final_fit, dict):
        raise VersionDError("Version D policy differs")
    if final_fit.get("epochs") != 3 or final_fit.get("scheduler") != "constant":
        raise VersionDError("Version D policy differs")
    return value
```

`verify_stage_c_delivery` must verify the outer delivery hash before opening it, require exactly the four declared outer members, verify every outer member hash and size from `delivery_manifest.json`, then call `verify_review_bundle` and `verify_resume_bundle` on securely materialized inner ZIPs. It must require Version C, `final_stage_complete=true`, the sealed stage-state hash, 21 completed results, `single_s3407`, seed 3407, temporal best epochs `[3, 0]`, and the sealed `selected_candidate` fields including `scheduler="plateau"`. It returns a frozen evidence mapping containing the prior manifest SHA-256, Stage C state, selected member, and all verified hashes.

`extract_training_input` must verify the whole archive hash plus member names and sizes, then extract and hash only `train.csv`. Reading the raw ZIP bytes for its sealed SHA-256 does not parse or expose any CSV member. `extract_review_inputs` runs only after a frozen artifact exists and extracts and hashes only `test.csv` and `sample_submission.csv`. Both use a temporary sibling directory followed by `os.replace`; neither uses `ZipFile.extractall`.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_version_d.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the contract unit**

```bash
git add experiments/tabm_campaign/version_d_contract.json experiments/tabm_campaign/version_d.py reports/rules/2026-08-14-policy-review.json tests/test_tabm_campaign_version_d.py
git commit -m "feat: seal TabM Version D inputs"
```

### Task 2: Make final fitting resumable at epoch boundaries

**Files:**
- Modify: `experiments/tabm_campaign/final_training.py:1-199`
- Create: `tests/test_tabm_campaign_final_training.py`

- [ ] **Step 1: Write failing checkpoint tests**

```python
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.tabm_campaign.final_training import (
    FinalTrainingError,
    load_epoch_checkpoint,
    save_epoch_checkpoint,
)


IDENTITY = {
    "contract_sha256": "1" * 64,
    "data_archive_sha256": "2" * 64,
    "train_sha256": "3" * 64,
    "runtime_sha256": "4" * 64,
    "training_source_sha256": "7" * 64,
}


def test_epoch_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "epoch.pt"
    payload = {
        "schema_version": 1,
        "identity": IDENTITY,
        "completed_epochs": 1,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {}, "param_groups": []},
        "scaler": {},
        "python_rng": None,
        "numpy_rng": None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [],
        "preprocessing_sha256": "5" * 64,
        "numeric_embedding_sha256": "6" * 64,
    }
    save_epoch_checkpoint(path, payload)
    restored = load_epoch_checkpoint(path, IDENTITY)
    assert restored["completed_epochs"] == 1
    assert restored["model"]["weight"].tolist() == [1.0]


def test_epoch_checkpoint_rejects_identity_change(tmp_path: Path) -> None:
    path = tmp_path / "epoch.pt"
    save_epoch_checkpoint(
        path,
        {
            "schema_version": 1,
            "identity": IDENTITY,
            "completed_epochs": 1,
            "model": {},
            "optimizer": {},
            "scaler": {},
            "python_rng": None,
            "numpy_rng": None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": [],
            "preprocessing_sha256": "5" * 64,
            "numeric_embedding_sha256": "6" * 64,
        },
    )
    changed = {**IDENTITY, "train_sha256": "9" * 64}
    with pytest.raises(FinalTrainingError, match="checkpoint identity differs"):
        load_epoch_checkpoint(path, changed)
```

- [ ] **Step 2: Run the checkpoint tests and confirm they fail**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_final_training.py -q
```

Expected: collection fails because `load_epoch_checkpoint` and `save_epoch_checkpoint` do not exist.

- [ ] **Step 3: Add atomic checkpoint serialization**

Add these interfaces and keep checkpoint loading CPU-safe:

```python
def save_epoch_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_epoch_checkpoint(path: Path, expected_identity: Mapping[str, str]) -> dict[str, object]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise FinalTrainingError("checkpoint schema differs")
    if value.get("identity") != dict(expected_identity):
        raise FinalTrainingError("checkpoint identity differs")
    completed = value.get("completed_epochs")
    if isinstance(completed, bool) or not isinstance(completed, int) or not 1 <= completed <= 3:
        raise FinalTrainingError("checkpoint completed epoch differs")
    return value
```

- [ ] **Step 4: Refactor the fit API to the sealed single-member policy**

Replace `fit_final_members` with this public signature:

```python
from typing import Callable, Mapping


def fit_final_member(
    *,
    data_dir: Path,
    artifact_root: Path,
    candidate: Mapping[str, object],
    seed: int,
    epochs: int,
    absolute_deadline: float,
    checkpoint_path: Path,
    checkpoint_identity: Mapping[str, str],
    on_epoch_checkpoint: Callable[[Path, int], None] | None = None,
) -> dict[str, object]:
```

Require seed 3407, epochs 3, and exactly one `train.csv`. The orchestrator copies the sealed `selected_candidate` and replaces only its scheduler with the sealed `final_fit.scheduler="constant"`; any other changed field is rejected. Keep the existing `dl_standard + hand_matchup` preprocessing and TabM construction. Use constant AdamW learning rate `0.0006`, weight decay `0.0001`, effective batch 4096, micro-batch 512, and AMP.

Before training, hash `preprocessing_state.json` and `numeric_embedding_0.json`. If a checkpoint exists, validate its identity and both state hashes, load model/optimizer/scaler, restore Python, NumPy, torch, and CUDA RNG states, and begin at `completed_epochs`. At each completed epoch, atomically save this exact payload before printing progress or invoking the callback:

```python
payload = {
    "schema_version": 1,
    "identity": dict(checkpoint_identity),
    "completed_epochs": epoch + 1,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
    "python_rng": random.getstate(),
    "numpy_rng": np.random.get_state(),
    "torch_rng": torch.get_rng_state(),
    "cuda_rng": torch.cuda.get_rng_state_all(),
    "preprocessing_sha256": file_sha256(state_path),
    "numeric_embedding_sha256": file_sha256(numeric_path),
}
save_epoch_checkpoint(checkpoint_path, payload)
print(f"FINAL_TRAINING_PROGRESS seed={seed} epoch={epoch + 1}/{epochs}", flush=True)
if on_epoch_checkpoint is not None:
    on_epoch_checkpoint(checkpoint_path, epoch + 1)
```

After epoch 3, save only the model state dict and immutable inference state in `artifact_root`; exclude optimizer, scaler, and RNG from `inference_manifest.json`. Record `fit_scope`, row count, three epochs, seed, constant scheduler, input and source hashes.

- [ ] **Step 5: Add a resume-control test without full model training**

Extract and test a pure helper:

```python
def test_resume_epoch_is_next_completed_epoch() -> None:
    from experiments.tabm_campaign.final_training import resume_start_epoch

    assert resume_start_epoch(None, 3) == 0
    assert resume_start_epoch({"completed_epochs": 1}, 3) == 1
    assert resume_start_epoch({"completed_epochs": 3}, 3) == 3
    with pytest.raises(FinalTrainingError, match="outside final epoch range"):
        resume_start_epoch({"completed_epochs": 4}, 3)
```

Implement:

```python
def resume_start_epoch(checkpoint: Mapping[str, object] | None, epochs: int) -> int:
    completed = 0 if checkpoint is None else checkpoint.get("completed_epochs")
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= epochs:
        raise FinalTrainingError("checkpoint epoch is outside final epoch range")
    return completed
```

- [ ] **Step 6: Run the focused tests**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_final_training.py -q
```

Expected: all tests pass without loading official data or using a GPU.

- [ ] **Step 7: Commit resumable training**

```bash
git add experiments/tabm_campaign/final_training.py tests/test_tabm_campaign_final_training.py
git commit -m "feat: resume final TabM fitting by epoch"
```

### Task 3: Add verified emergency and frozen-model snapshots

**Files:**
- Modify: `experiments/tabm_campaign/version_d.py`
- Modify: `tests/test_tabm_campaign_version_d.py`

- [ ] **Step 1: Write failing deterministic snapshot tests**

```python
def test_emergency_snapshot_is_deterministic_and_bound(tmp_path: Path) -> None:
    from experiments.tabm_campaign.version_d import verify_emergency_snapshot, write_emergency_snapshot

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    state = tmp_path / "preprocessing_state.json"
    state.write_bytes(b"{}")
    numeric = tmp_path / "numeric_embedding_0.json"
    numeric.write_bytes(b"{}")
    identity = {"contract_sha256": "1" * 64, "train_sha256": "2" * 64}
    first = write_emergency_snapshot(tmp_path / "one", checkpoint, state, numeric, 1, identity)
    second = write_emergency_snapshot(tmp_path / "two", checkpoint, state, numeric, 1, identity)
    assert first.read_bytes() == second.read_bytes()
    verified = verify_emergency_snapshot(first, identity)
    assert verified.epoch == 1


def test_frozen_snapshot_rejects_optimizer_state(tmp_path: Path) -> None:
    from experiments.tabm_campaign.version_d import VersionDError, verify_frozen_snapshot

    archive = tmp_path / "frozen.zip"
    _write_zip(archive, {"optimizer.pt": b"not allowed"})
    with pytest.raises(VersionDError, match="frozen snapshot member set differs"):
        verify_frozen_snapshot(archive, {"contract_sha256": "1" * 64})
```

- [ ] **Step 2: Run the focused tests and confirm missing APIs**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_version_d.py -q
```

Expected: fails because the snapshot functions are not defined.

- [ ] **Step 3: Implement deterministic ZIP helpers and snapshot manifests**

Use fixed ZIP timestamp `(2026, 1, 1, 0, 0, 0)`, sorted member order, POSIX mode `0o100644`, DEFLATE level 9, and atomic publication. Emergency archives contain only:

```text
checkpoint.pt
preprocessing_state.json
numeric_embedding_0.json
snapshot_manifest.json
```

The manifest has schema version, `artifact_kind="tabm_version_D_emergency"`, epoch, exact identity mapping, and SHA-256 for the first three members. Name it `tabm_version_D_emergency_epoch_001_<first12sha>.zip`, using epochs 001 through 003.

Frozen archives contain only the files declared by `inference_manifest.json` plus the manifest itself and `frozen_manifest.json`. The frozen manifest uses `artifact_kind="tabm_version_D_frozen_model"`, the exact identity mapping, and every member hash. Verification rejects path traversal, duplicate names, symlinks, undeclared members, missing members, hash mismatches, identity changes, optimizer/scaler/RNG members, and epochs other than three.

- [ ] **Step 4: Implement recovery selection rules**

Add:

```python
def select_recovery_snapshot(
    emergency_paths: Sequence[Path],
    frozen_paths: Sequence[Path],
    expected_identity: Mapping[str, str],
) -> RecoverySelection:
```

Verify all candidates. Return the verified frozen snapshot if exactly one exists. Otherwise return the unique highest-epoch emergency snapshot. If two different files claim the same highest epoch, raise `VersionDError("conflicting recovery snapshots")`. Return mode `"fresh"` only when neither kind exists.

- [ ] **Step 5: Run the snapshot tests**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_version_d.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit snapshot support**

```bash
git add experiments/tabm_campaign/version_d.py tests/test_tabm_campaign_version_d.py
git commit -m "feat: verify Version D recovery snapshots"
```

### Task 4: Separate frozen review from final fitting

**Files:**
- Modify: `experiments/tabm_campaign/final_review.py:1-212`
- Modify: `tests/test_tabm_campaign_final_review.py`

- [ ] **Step 1: Replace the median-epoch test with the sealed policy test**

```python
from experiments.tabm_campaign.final_review import final_epoch_count, synthetic_scale_frame


def test_final_epoch_count_is_sealed_at_three() -> None:
    assert final_epoch_count() == 3
```

Remove the old `fixed_epoch_count` assertions.

- [ ] **Step 2: Run the focused test and confirm the expected import failure**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_final_review.py::test_final_epoch_count_is_sealed_at_three -q
```

Expected: fails because `final_epoch_count` does not exist.

- [ ] **Step 3: Seal the epoch function and remove data-dependent rounding**

```python
def final_epoch_count() -> int:
    return 3
```

Remove the `statistics` import and all median-based epoch selection.

- [ ] **Step 4: Split post-freeze review into its own function**

Add this public API:

```python
def review_frozen_artifact(
    *,
    artifact_root: Path,
    review_data_dir: Path,
    output_dir: Path,
    prior_manifest_sha256: str,
    version_d_contract_sha256: str,
    fit_report: Mapping[str, object],
    runtime_versions: Mapping[str, str],
) -> StageRunResult:
```

Move the dependency probe, 5-row probe, frozen predictor construction, independence audit, synthetic 245,789-row scale test, source-before/source-after equality, resource gates, and Version D bundle writing into this function. It must reject an artifact manifest unless it says `fit_scope="official_train_2019_2024_only"`, `epochs=3`, `seeds=[3407]`, and `scheduler="constant"`.

Replace the hard-coded `reports/rules/2026-08-13-policy-review.json` lookup with
`reports/rules/2026-08-14-policy-review.json`; require its policy digest to match
`competition_rules/policy.json` and its verdict to be `unchanged`.

Hash every frozen artifact before the review and again afterward; fail if the mapping changes. Record Python, CUDA, torch, TabM, rtdl-num-embeddings, NumPy, and pandas versions. Emit these markers only after their corresponding checks pass:

```python
print("VERSION_D_INDEPENDENCE_PASSED", flush=True)
print(f"VERSION_D_SCALE_GATE_PASSED rows=245789 seconds={inference_seconds:.3f}", flush=True)
print(f"VERSION_D_REVIEW_READY path={final_path}", flush=True)
```

Keep `run_final_review` as a compatibility wrapper for `runner.py`: validate Stage C input, call `fit_final_member` with three epochs and a local checkpoint, then call `review_frozen_artifact`. This prevents a broad `runner.py` rewrite.

- [ ] **Step 5: Add failure tests for a non-frozen or wrongly scoped manifest**

```python
def test_review_rejects_wrong_fit_scope(tmp_path: Path) -> None:
    import json
    import pytest

    from experiments.tabm_campaign.final_review import FinalReviewError, validate_frozen_manifest

    root = tmp_path / "frozen"
    root.mkdir()
    (root / "inference_manifest.json").write_text(
        json.dumps({"fit_scope": "test_aware", "epochs": 3, "seeds": [3407], "scheduler": "constant"}),
        encoding="utf-8",
    )
    with pytest.raises(FinalReviewError, match="fit scope differs"):
        validate_frozen_manifest(root)
```

- [ ] **Step 6: Run final-review tests**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_final_review.py tests/test_tabm_campaign_inference.py -q
```

Expected: all tests pass without full-data inference or GPU training.

- [ ] **Step 7: Commit the review separation**

```bash
git add experiments/tabm_campaign/final_review.py tests/test_tabm_campaign_final_review.py
git commit -m "refactor: review only frozen TabM artifacts"
```

### Task 5: Orchestrate Version D and build review-only delivery

**Files:**
- Create: `experiments/tabm_campaign/colab_version_d.py`
- Modify: `experiments/tabm_campaign/version_d.py`
- Modify: `tests/test_tabm_campaign_colab_version_d.py`
- Modify: `tests/test_tabm_campaign_version_d.py`

- [ ] **Step 1: Write failing state-machine tests with injected lightweight functions**

```python
from __future__ import annotations

from pathlib import Path

from experiments.tabm_campaign.colab_version_d import run_version_d


def test_frozen_recovery_skips_training(tmp_path: Path, monkeypatch) -> None:
    import experiments.tabm_campaign.colab_version_d as runtime

    calls: list[str] = []
    monkeypatch.setattr(runtime, "verify_inputs", lambda *args: {"prior_manifest_sha256": "1" * 64})
    monkeypatch.setattr(runtime, "restore_recovery", lambda *args: "frozen")
    monkeypatch.setattr(runtime, "fit_final", lambda *args, **kwargs: calls.append("fit"))
    monkeypatch.setattr(runtime, "review_final", lambda *args, **kwargs: calls.append("review") or tmp_path / "review.zip")
    monkeypatch.setattr(runtime, "publish_delivery", lambda *args, **kwargs: tmp_path / "delivery.zip")

    result = runtime.run_version_d(
        data_archive=tmp_path / "data.zip",
        stage_c_delivery=tmp_path / "stage_c.zip",
        work_root=tmp_path / "work",
        recovery_archive=tmp_path / "frozen.zip",
        absolute_deadline=4_000_000_000.0,
        on_download=lambda path, kind: calls.append(kind),
    )
    assert "fit" not in calls
    assert calls == ["frozen", "review", "delivery"]
    assert result.name == "delivery.zip"
```

Add companion tests for fresh fitting, emergency resume, expired deadline, review failure, and ensuring no delivery is created when a gate fails.

- [ ] **Step 2: Run the state-machine tests and confirm the missing module**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_colab_version_d.py -q
```

Expected: collection fails because `colab_version_d` does not exist.

- [ ] **Step 3: Implement the explicit three-mode state machine**

`run_version_d` performs these operations in order:

```text
verify contract and Stage C delivery
verify official archive identity without extracting test data
select a same-VM or uploaded fresh, emergency, or frozen mode
extract train.csv only when no frozen model exists
fit or resume when no frozen model exists
publish and request the frozen-model download
extract test.csv and sample_submission.csv only after freeze
run frozen review gates
publish and request the final delivery download
```

Keep the production signature limited to input paths, deadline, and download callback. Tests replace module-level boundary functions with pytest `monkeypatch`; do not add test-only parameters, a framework, or a class hierarchy. Emit `VERSION_D_INPUTS_VERIFIED`, `VERSION_D_GPU_READY`, and `VERSION_D_FROZEN_MODEL_READY` only after successful checks.

- [ ] **Step 4: Implement final delivery publication**

Add `write_review_delivery` to `version_d.py`. It creates `tabm_hand_matchup_stage_D_review_delivery.zip` with exactly:

```text
delivery_manifest.json
tabm_hand_matchup_final_review_bundle.zip
version_d.log
```

The manifest records `artifact_kind="tabm_version_D_review_delivery"`, `review_only=true`, `submission_package=false`, contract hash, official archive hash, Stage C delivery hash, prior Stage C manifest hash, frozen manifest hash, runtime versions, and every included member hash and size. Flush the log immediately after `VERSION_D_REVIEW_READY`, package that immutable log snapshot, and then verify the completed delivery by reopening it before printing:

```python
print(f"VERSION_D_DELIVERY_READY path={delivery_path}", flush=True)
```

The delivery-ready and download-requested markers occur after the log snapshot by necessity; they remain in the visible Colab output. The sealed log inside the delivery contains the complete run through the final review decision.

The publication function rejects member names `script.py`, `requirements.txt`, `submit.zip`, and every path beginning with `model/`.

- [ ] **Step 5: Run orchestration and archive tests**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_colab_version_d.py tests/test_tabm_campaign_version_d.py -q
```

Expected: all tests pass using fixtures only.

- [ ] **Step 6: Commit the orchestrator**

```bash
git add experiments/tabm_campaign/colab_version_d.py experiments/tabm_campaign/version_d.py tests/test_tabm_campaign_colab_version_d.py tests/test_tabm_campaign_version_d.py
git commit -m "feat: orchestrate TabM Version D review"
```

### Task 6: Render the one-cell Colab handoff

**Files:**
- Create: `tools/render_tabm_colab_version_d_cell.py`
- Create: `experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py`
- Modify: `tests/test_tabm_campaign_colab_version_d.py`

- [ ] **Step 1: Write generated-cell contract tests**

```python
from hashlib import sha256
from pathlib import Path
import subprocess
import sys


CELL = Path("experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py")
RENDERER = Path("tools/render_tabm_colab_version_d_cell.py")


def test_version_d_cell_is_self_contained_small_and_review_only() -> None:
    text = CELL.read_text(encoding="utf-8")
    compile(text, str(CELL), "exec")
    assert CELL.stat().st_size < 950_000
    assert "files.upload()" in text
    assert "files.download(" in text
    assert "RECOVERY_MODE" in text
    assert "VERSION_D_DELIVERY_READY" in text
    lowered = text.lower()
    assert "drive.mount" not in lowered
    assert "git clone" not in lowered
    assert "submit.zip" not in lowered
    assert "script.py" not in lowered


def test_version_d_renderer_is_deterministic() -> None:
    subprocess.run([sys.executable, str(RENDERER)], check=True)
    first = sha256(CELL.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(RENDERER)], check=True)
    assert sha256(CELL.read_bytes()).hexdigest() == first
```

- [ ] **Step 2: Run the cell tests and confirm the files are absent**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_colab_version_d.py -q
```

Expected: fails because the renderer and generated cell do not exist.

- [ ] **Step 3: Implement deterministic runtime embedding**

Follow the existing Stage C renderer's deterministic tar+gzip+base64 pattern. Include only required `.py`, `.json`, and requirement files from `experiments/tabm_campaign`, `experiments/independent_dl`, `competition_rules`, and `reports/rules/2026-08-14-policy-review.json`. Exclude all generated cell files, caches, notebooks, tests, artifacts, and Git metadata.

The generated cell begins with:

```python
# Copy this entire file into one Colab cell and select a T4 GPU runtime.
from __future__ import annotations

RECOVERY_MODE = "none"  # Use "emergency" or "frozen" only after a VM reset.
WORK_ROOT = Path("/content/tabm_version_d")
DATA_ARCHIVE = WORK_ROOT / "inputs/lg-aimers-9th-data.zip"
STAGE_C_DELIVERY = WORK_ROOT / "inputs/tabm_colab_stage_C_delivery.zip"
```

It must:

1. Verify and atomically extract the embedded runtime.
2. Reuse already verified same-VM files; otherwise request exactly the expected upload.
3. Request one recovery ZIP only when `RECOVERY_MODE` is `emergency` or `frozen` and validate its filename before moving it.
4. Install pinned dependencies only when their exact versions are absent.
5. Require exactly one CUDA device and print its name and versions.
6. Tee stdout and stderr to `/content/tabm_version_d/version_d.log` with line flushing.
7. Invoke `run_version_d` with a download callback.
8. Download every completed epoch emergency snapshot, then the frozen model snapshot, then final delivery.
9. On an exception, print `VERSION_D_ERROR stage=<stage> type=<type> message=<message>`, expose the log, and request the latest already-complete emergency or frozen snapshot before re-raising.

Use these exact successful markers:

```text
VERSION_D_CODE_READY
VERSION_D_INPUTS_VERIFIED
VERSION_D_GPU_READY
FINAL_TRAINING_PROGRESS seed=3407 epoch=1/3
VERSION_D_EMERGENCY_SNAPSHOT_READY
VERSION_D_FROZEN_MODEL_READY
VERSION_D_INDEPENDENCE_PASSED
VERSION_D_SCALE_GATE_PASSED
VERSION_D_REVIEW_READY
VERSION_D_DELIVERY_READY
VERSION_D_DOWNLOAD_REQUESTED
```

- [ ] **Step 4: Render and inspect the cell**

Run:

```bash
.venv/bin/python tools/render_tabm_colab_version_d_cell.py
.venv/bin/python -m py_compile experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py
wc -c experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py
```

Expected: renderer prints the output path and SHA-256, compilation succeeds, and size is below 950,000 bytes.

- [ ] **Step 5: Run generated-cell tests**

Run:

```bash
.venv/bin/pytest tests/test_tabm_campaign_colab_version_d.py -q
```

Expected: all tests pass without importing `google.colab` locally or running training.

- [ ] **Step 6: Commit the Colab handoff**

```bash
git add tools/render_tabm_colab_version_d_cell.py experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py tests/test_tabm_campaign_colab_version_d.py
git commit -m "feat: add Colab TabM Version D cell"
```

### Task 7: Document execution and run final lightweight verification

**Files:**
- Create: `docs/TABM_VERSION_D_COLAB.md`
- Verify: all Version D source and tests

- [ ] **Step 1: Write the user handoff document**

Document these exact instructions:

```text
Purpose: train the sealed single_s3407 TabM and create review-only evidence.
Runtime: Google Colab, T4 GPU, about 20-45 minutes.
Required uploads: lg-aimers-9th-data.zip and tabm_colab_stage_C_delivery.zip.
Fresh run: leave RECOVERY_MODE="none".
VM-reset recovery: set RECOVERY_MODE="emergency" and upload the newest epoch ZIP.
Post-training recovery: set RECOVERY_MODE="frozen" and upload tabm_version_D_frozen_model.zip.
Rerun safety: completed epochs resume; a verified frozen model skips training; a partial epoch reruns.
Final success: VERSION_D_DELIVERY_READY followed by VERSION_D_DOWNLOAD_REQUESTED.
Return to Codex: tabm_hand_matchup_stage_D_review_delivery.zip and the final error log only if an error occurred.
Not produced: submit.zip, script.py, requirements.txt, or a submission model/ directory.
```

Also explain that browser downloads may appear three times for epoch snapshots plus once for the frozen model and once for final delivery. The newest emergency snapshot supersedes older ones, but users should keep the frozen model and final delivery.

- [ ] **Step 2: Run static and focused verification**

Run:

```bash
.venv/bin/python -m py_compile \
  experiments/tabm_campaign/version_d.py \
  experiments/tabm_campaign/final_training.py \
  experiments/tabm_campaign/final_review.py \
  experiments/tabm_campaign/colab_version_d.py \
  tools/render_tabm_colab_version_d_cell.py \
  experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py
.venv/bin/pytest \
  tests/test_tabm_campaign_version_d.py \
  tests/test_tabm_campaign_final_training.py \
  tests/test_tabm_campaign_final_review.py \
  tests/test_tabm_campaign_inference.py \
  tests/test_tabm_campaign_colab_version_d.py -q
```

Expected: compilation succeeds and all focused tests pass.

- [ ] **Step 3: Run the existing lightweight regression suite**

Run:

```bash
.venv/bin/pytest -q
```

Expected: all repository tests pass. Do not run official-data preprocessing, GPU fitting, scale inference, package installation, or submission evaluation locally.

- [ ] **Step 4: Check generated artifacts and forbidden paths**

Run:

```bash
git diff --check
rg -n "submit[.]zip|drive[.]mount|git clone" \
  experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py \
  experiments/tabm_campaign/colab_version_d.py
git status --short
```

Expected: `git diff --check` is silent; the search has no match; `git status` shows only the previously existing Stage C modifications, the user notebook, and any intentionally uncommitted Version D documentation before its commit.

- [ ] **Step 5: Commit the handoff documentation**

```bash
git add docs/TABM_VERSION_D_COLAB.md
git commit -m "docs: explain Colab Version D review"
```

- [ ] **Step 6: Report the user-run boundary**

Provide the generated cell path, SHA-256, required upload filenames, expected runtime, rerun modes, success marker, and exact delivery filename. State that Codex did not run the full-data or GPU job and that no submission package exists yet.
