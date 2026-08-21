# OOF Reset Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 규칙 준수 OOF와 격리된 과거 OOF를 동일 행에서 다시 계산해 시간 변화, calibration과 모델 다양성을 진단하는 CPU 전용 감사 도구를 만든다.

**Architecture:** `experiments/oof_reset_audit`가 기존 산출물 verifier를 통과한 ZIP만 읽어 공통 `PredictionSet`으로 정규화한다. 독립적인 metrics 계층이 단독·paired·segment·block bootstrap 지표를 계산하고, orchestration 계층은 비교 가능성 등급과 다음 실험 선택지를 작은 보고서 ZIP으로 발행한다. 실제 전체 OOF 실행은 사용자가 수행하며 구현 중에는 합성 fixture만 사용한다.

**Tech Stack:** Python 3.11, dataclasses, pathlib, zipfile, hashlib, json, NumPy, pandas, scikit-learn, pytest

---

## File map

- Create `experiments/oof_reset_audit/__init__.py`: 공개 타입과 실행 함수만 노출한다.
- Create `experiments/oof_reset_audit/types.py`: 신뢰 등급, fold와 예측 자료 타입을 정의한다.
- Create `experiments/oof_reset_audit/metrics.py`: 단독, paired, 구간, calibration과 block bootstrap 계산을 담당한다.
- Create `experiments/oof_reset_audit/artifacts.py`: 기존 ZIP verifier 호출, 산출물 판별과 `PredictionSet` 변환을 담당한다.
- Create `experiments/oof_reset_audit/run.py`: 전체 감사, 판정 근거, 표·JSON·Markdown·결과 ZIP 생성을 담당한다.
- Create `tools/run_oof_reset_audit.py`: 프로젝트 루트를 안전하게 import path에 넣는 얇은 CLI다.
- Create `tests/test_oof_reset_audit_metrics.py`: 수학과 행 정렬 계약을 검증한다.
- Create `tests/test_oof_reset_audit_artifacts.py`: 실제 산출물 형식별 adapter와 격리 경계를 검증한다.
- Create `tests/test_oof_reset_audit_run.py`: 보고서, missing evidence, 판정과 ZIP allowlist를 검증한다.
- Modify `experiments/hierarchical_tabm/calibration.py`: JSON round-trip 후 H3 effect 순서를 정규화한다.
- Modify `tests/test_hierarchical_tabm_calibration.py`: 실제 JSON 직렬화 round-trip 회귀 테스트를 추가한다.
- Create `docs/OOF_RESET_AUDIT.md`: 사용자 실행 입력, 예상 로그와 전달 파일을 설명한다.

## Task 1: Repair persisted H3 calibration loading

**Files:**
- Modify: `experiments/hierarchical_tabm/calibration.py:388-431`
- Modify: `tests/test_hierarchical_tabm_calibration.py`

- [ ] **Step 1: Write the failing JSON round-trip test**

Add this test beside the existing calibration payload round-trip test:

```python
def test_h3_calibration_survives_canonical_json_round_trip() -> None:
    state = CalibrationState(
        schema_version=1,
        kind="H3",
        regularization=0.01,
        clip=1e-6,
        bias=-0.1,
        slope=0.9,
        effects=MappingProxyType({
            "game_type": MappingProxyType({"F": -0.2, "R": 0.1}),
            "count_state": MappingProxyType({"0_0": 0.01}),
            "hand_matchup": MappingProxyType({"1_2": -0.03}),
            "base_out_state": MappingProxyType({"___0": 0.02}),
        }),
        fit_row_ids_sha256="a" * 64,
    )
    persisted = json.loads(canonical_state_json(state))

    restored = calibration_state_from_payload(persisted)

    assert calibration_state_payload(restored) == calibration_state_payload(state)
    assert tuple(restored.effects) == CALIBRATION_EFFECTS
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_calibration.py::test_h3_calibration_survives_canonical_json_round_trip -q
```

Expected: FAIL with `CalibrationError: calibration effects differ`.

- [ ] **Step 3: Normalize persisted effect order without weakening key validation**

Replace the order-sensitive H3 mapping check with exact-set validation followed by contract-order reconstruction:

```python
effects_payload = payload["effects"]
expected_effects = CALIBRATION_EFFECTS if payload["kind"] == "H3" else ()
if (
    not isinstance(effects_payload, Mapping)
    or set(effects_payload) != set(expected_effects)
):
    raise CalibrationError("calibration effects differ")
effects: dict[str, Mapping[str, float]] = {}
for column in expected_effects:
    mapping = effects_payload[column]
    if not isinstance(mapping, Mapping) or list(mapping) != sorted(mapping):
        raise CalibrationError("calibration effect levels differ")
    parsed: dict[str, float] = {}
    for level, value in mapping.items():
        if (
            not isinstance(level, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise CalibrationError("calibration effect value is invalid")
        parsed[level] = float(value)
    effects[column] = MappingProxyType(parsed)
```

Keep the final `calibration_state_payload(state) != dict(payload)` canonical equality check. It verifies values and digest after reconstruction.

- [ ] **Step 4: Run calibration tests and confirm GREEN**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_hierarchical_tabm_calibration.py -q
```

Expected: all tests pass, including the new JSON round-trip.

- [ ] **Step 5: Commit the isolated repair**

```bash
git add experiments/hierarchical_tabm/calibration.py \
  tests/test_hierarchical_tabm_calibration.py
git commit -m "fix: restore persisted H3 calibration"
```

## Task 2: Define normalized prediction evidence and metric primitives

**Files:**
- Create: `experiments/oof_reset_audit/__init__.py`
- Create: `experiments/oof_reset_audit/types.py`
- Create: `experiments/oof_reset_audit/metrics.py`
- Create: `tests/test_oof_reset_audit_metrics.py`

- [ ] **Step 1: Write failing tests for evidence validation and exact metrics**

Create `tests/test_oof_reset_audit_metrics.py` with small deterministic frames:

```python
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.oof_reset_audit.metrics import (
    block_bootstrap_interval,
    calibration_deciles,
    paired_metrics,
    single_model_metrics,
    validate_prediction_frame,
)
from experiments.oof_reset_audit.types import PredictionSet, TrustClass


def frame(probability: list[float], *, reverse: bool = False) -> pd.DataFrame:
    value = pd.DataFrame({
        "row_id": [f"r{i}" for i in range(8)],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
        "probability": probability,
        "game_month": [4, 4, 5, 5, 6, 6, 7, 7],
        "game_type": ["R", "R", "F", "F", "R", "R", "F", "F"],
    })
    return value.iloc[::-1].reset_index(drop=True) if reverse else value


def prediction(model_id: str, values: list[float], *, reverse: bool = False) -> PredictionSet:
    return PredictionSet(
        artifact_path=Path(f"{model_id}.zip"),
        artifact_sha256="a" * 64,
        source_member=f"predictions/{model_id}.csv",
        prediction_sha256="b" * 64,
        model_id=model_id,
        fold="2023->2024",
        trust=TrustClass.RULE_SAFE,
        frame=frame(values, reverse=reverse),
    )


def test_validation_rejects_duplicate_ids_targets_and_probability_range() -> None:
    duplicate = frame([0.1] * 8)
    duplicate.loc[1, "row_id"] = "r0"
    with pytest.raises(ValueError, match="row_id must be unique"):
        validate_prediction_frame(duplicate)
    invalid = frame([0.1] * 7 + [1.1])
    with pytest.raises(ValueError, match="probability"):
        validate_prediction_frame(invalid)


def test_single_and_paired_metrics_have_exact_values_and_ignore_row_order() -> None:
    anchor = prediction("anchor", [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55])
    candidate = prediction("candidate", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6], reverse=True)
    single = single_model_metrics(anchor)
    paired = paired_metrics(anchor, candidate)
    assert math.isclose(single["brier"], 0.123125, abs_tol=1e-12)
    assert math.isclose(single["local_bss"], 50_750.0, abs_tol=1e-9)
    assert math.isclose(paired["candidate_brier"], 0.075, abs_tol=1e-12)
    assert math.isclose(paired["gain_vs_anchor"], 0.048125, abs_tol=1e-12)


def test_paired_metrics_reject_different_row_sets_and_targets() -> None:
    anchor = prediction("anchor", [0.2] * 8)
    candidate = prediction("candidate", [0.2] * 8)
    candidate.frame.loc[7, "row_id"] = "different"
    with pytest.raises(ValueError, match="row_id set differs"):
        paired_metrics(anchor, candidate)


def test_calibration_and_block_interval_are_deterministic() -> None:
    evidence = prediction("anchor", [0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95])
    table = calibration_deciles(evidence)
    assert table["rows"].sum() == 8
    losses = pd.DataFrame({
        "block": [f"2024-{month}" for month in range(4, 10) for _ in range(2)],
        "loss_delta": np.arange(12, dtype="float64") / 1000,
    })
    first = block_bootstrap_interval(losses, repeats=2_000, seed=3407)
    second = block_bootstrap_interval(losses, repeats=2_000, seed=3407)
    assert first == second
    assert first["block_count"] == 6
```

- [ ] **Step 2: Run the metric test and confirm RED**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_metrics.py -q
```

Expected: collection fails because `experiments.oof_reset_audit` does not exist.

- [ ] **Step 3: Implement minimal types**

Create `experiments/oof_reset_audit/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd


class TrustClass(str, Enum):
    RULE_SAFE = "rule_safe"
    QUARANTINED_DIAGNOSTIC = "quarantined_diagnostic"


class ArtifactRole(str, Enum):
    STAGE_C_TABM = "stage_c_tabm"
    ROW_FEATURE = "row_feature"
    CATBOOST_BLEND = "catboost_blend"
    CATBOOST_DEPLOYMENT = "catboost_deployment"
    HIERARCHICAL = "hierarchical"
    QUARANTINED_XGBOOST = "quarantined_xgboost"


@dataclass(frozen=True)
class PredictionSet:
    artifact_path: Path
    artifact_sha256: str
    source_member: str
    prediction_sha256: str
    model_id: str
    fold: str
    trust: TrustClass
    frame: pd.DataFrame


@dataclass(frozen=True)
class ArtifactRecord:
    path: Path
    sha256: str
    role: ArtifactRole
    artifact_kind: str
    status: str
    detail: str | None
```

Create `experiments/oof_reset_audit/__init__.py` exporting only `PredictionSet`, `TrustClass`, and later `run_audit`.

- [ ] **Step 4: Implement validation and metric functions**

Create `experiments/oof_reset_audit/metrics.py` with these implementations:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .types import PredictionSet

SEGMENTS = (
    "game_month", "game_type", "count_state", "hand_matchup",
    "base_out_state", "pitcher_known", "batter_known",
)


def validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "target", "probability"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("prediction columns differ")
    result = frame.copy()
    if result["row_id"].isna().any():
        raise ValueError("row_id is missing")
    result["row_id"] = result["row_id"].astype(str)
    if result["row_id"].duplicated().any():
        raise ValueError("row_id must be unique")
    target = pd.to_numeric(result["target"], errors="coerce").to_numpy("float64")
    probability = pd.to_numeric(
        result["probability"], errors="coerce"
    ).to_numpy("float64")
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("target must contain only 0 and 1")
    if (
        not np.isfinite(probability).all()
        or ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise ValueError("probability must be finite and in [0, 1]")
    result["target"] = target.astype("int8")
    result["probability"] = probability
    return result


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(probability - target), dtype=np.float64))


def single_model_metrics(evidence: PredictionSet) -> dict[str, object]:
    frame = validate_prediction_frame(evidence.frame)
    target = frame["target"].to_numpy("float64")
    probability = frame["probability"].to_numpy("float64")
    brier = _brier(target, probability)
    rate = float(target.mean())
    prior_brier = rate * (1.0 - rate)
    if prior_brier <= 0.0:
        raise ValueError("local BSS requires both target classes")
    deciles = calibration_deciles(evidence)
    ece = float(
        (
            deciles["rows"] / len(frame)
            * (deciles["target_mean"] - deciles["prediction_mean"]).abs()
        ).sum()
    )
    return {
        "model_id": evidence.model_id,
        "fold": evidence.fold,
        "trust": evidence.trust.value,
        "rows": len(frame),
        "target_mean": rate,
        "prediction_mean": float(probability.mean()),
        "prediction_std": float(probability.std()),
        "prediction_min": float(probability.min()),
        "prediction_max": float(probability.max()),
        "brier": brier,
        "local_bss": 100_000.0 * (1.0 - brier / prior_brier),
        "roc_auc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "ece_10": ece,
    }


def align_pair(anchor: PredictionSet, candidate: PredictionSet) -> pd.DataFrame:
    if anchor.fold != candidate.fold:
        raise ValueError("fold differs")
    left = validate_prediction_frame(anchor.frame)
    right = validate_prediction_frame(candidate.frame)
    if set(left["row_id"]) != set(right["row_id"]):
        raise ValueError("row_id set differs")
    right = right.set_index("row_id", drop=False).loc[left["row_id"]].reset_index(drop=True)
    left = left.reset_index(drop=True)
    if not np.array_equal(left["target"].to_numpy(), right["target"].to_numpy()):
        raise ValueError("target differs")
    output = left.rename(columns={"probability": "anchor_probability"})
    output["candidate_probability"] = right["probability"].to_numpy("float64")
    for column in SEGMENTS:
        if column in left and column in right:
            matches = left[column].eq(right[column]) | (
                left[column].isna() & right[column].isna()
            )
            if not bool(matches.all()):
                raise ValueError(f"segment differs: {column}")
        elif column in right:
            output[column] = right[column].to_numpy()
    return output


def paired_metrics(anchor: PredictionSet, candidate: PredictionSet) -> dict[str, object]:
    paired = align_pair(anchor, candidate)
    target = paired["target"].to_numpy("float64")
    anchor_probability = paired["anchor_probability"].to_numpy("float64")
    candidate_probability = paired["candidate_probability"].to_numpy("float64")
    anchor_loss = np.square(anchor_probability - target)
    candidate_loss = np.square(candidate_probability - target)
    return {
        "anchor_model_id": anchor.model_id,
        "candidate_model_id": candidate.model_id,
        "fold": anchor.fold,
        "rows": len(paired),
        "anchor_brier": float(anchor_loss.mean()),
        "candidate_brier": float(candidate_loss.mean()),
        "gain_vs_anchor": float((anchor_loss - candidate_loss).mean()),
        "prediction_correlation": float(np.corrcoef(anchor_probability, candidate_probability)[0, 1]),
        "loss_correlation": float(np.corrcoef(anchor_loss, candidate_loss)[0, 1]),
        "residual_correlation": float(np.corrcoef(target - anchor_probability, target - candidate_probability)[0, 1]),
    }


def calibration_deciles(evidence: PredictionSet) -> pd.DataFrame:
    frame = validate_prediction_frame(evidence.frame)
    quantile = pd.qcut(frame["probability"], 10, labels=False, duplicates="drop")
    table = frame.assign(decile=quantile).groupby("decile", observed=True).agg(
        rows=("row_id", "size"),
        target_mean=("target", "mean"),
        prediction_mean=("probability", "mean"),
    ).reset_index()
    table.insert(0, "fold", evidence.fold)
    table.insert(0, "model_id", evidence.model_id)
    return table


def segment_metrics(anchor: PredictionSet, candidate: PredictionSet) -> pd.DataFrame:
    paired = align_pair(anchor, candidate)
    rows: list[dict[str, object]] = []
    for column in SEGMENTS:
        if column not in paired:
            continue
        for level, group in paired.groupby(column, dropna=False, sort=True):
            target = group["target"].to_numpy("float64")
            anchor_probability = group["anchor_probability"].to_numpy("float64")
            candidate_probability = group["candidate_probability"].to_numpy("float64")
            anchor_brier = _brier(target, anchor_probability)
            candidate_brier = _brier(target, candidate_probability)
            rows.append({
                "anchor_model_id": anchor.model_id,
                "candidate_model_id": candidate.model_id,
                "fold": anchor.fold,
                "segment": column,
                "level": "__MISSING__" if pd.isna(level) else str(level),
                "rows": len(group),
                "anchor_brier": anchor_brier,
                "candidate_brier": candidate_brier,
                "regression": candidate_brier - anchor_brier,
                "eligible": len(group) >= 5_000,
            })
    return pd.DataFrame(rows)


def block_bootstrap_interval(
    losses: pd.DataFrame, *, repeats: int = 2_000, seed: int = 3407
) -> dict[str, object]:
    if set(losses) != {"block", "loss_delta"}:
        raise ValueError("bootstrap columns differ")
    if repeats != 2_000 or seed != 3407:
        raise ValueError("bootstrap contract differs")
    grouped = tuple(
        group["loss_delta"].to_numpy("float64")
        for _, group in losses.groupby("block", sort=True)
    )
    if len(grouped) < 6:
        return {"status": "insufficient_blocks", "block_count": len(grouped)}
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype="float64")
    for index in range(repeats):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        values = np.concatenate([grouped[position] for position in selected])
        samples[index] = float(values.mean())
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "status": "completed",
        "block_count": len(grouped),
        "lower": float(lower),
        "upper": float(upper),
    }
```

Use these exact formulas:

```python
brier = float(np.mean(np.square(probability - target), dtype=np.float64))
prior_brier = float(target.mean() * (1.0 - target.mean()))
local_bss = 100_000.0 * (1.0 - brier / prior_brier)
loss_delta = np.square(candidate_probability - target) - np.square(anchor_probability - target)
gain_vs_anchor = -float(loss_delta.mean())
```

`align_pair` must require identical fold, trust-independent row sets, identical targets, and identical shared segment values. It must align with `set_index("row_id").loc[...]`; never use an inner merge that silently drops rows.

`block_bootstrap_interval` must group by the supplied `block`, require at least six distinct blocks, resample whole block labels with replacement 2,000 times, and return the 2.5th and 97.5th quantiles. If fewer than six blocks exist, return `{"status": "insufficient_blocks", "block_count": count}` without fabricated bounds.

- [ ] **Step 5: Run focused metric tests and confirm GREEN**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_metrics.py -q
```

Expected: all metric tests pass.

- [ ] **Step 6: Commit normalized evidence and metrics**

```bash
git add experiments/oof_reset_audit tests/test_oof_reset_audit_metrics.py
git commit -m "feat: add OOF reset audit metrics"
```

## Task 3: Add verified adapters for existing artifacts

**Files:**
- Create: `experiments/oof_reset_audit/artifacts.py`
- Create: `tests/test_oof_reset_audit_artifacts.py`

- [ ] **Step 1: Write failing adapter boundary tests**

Create fixture builders that call the existing production bundle publishers from:

- `experiments.tabm_campaign.artifacts`
- `experiments.catboost_tabm_blend.artifacts`
- `experiments.catboost_deployment.artifacts`
- `experiments.hierarchical_tabm.artifacts`

Then add these contract tests:

```python
def test_stage_c_adapter_loads_three_final_seeds_on_two_folds(stage_c_delivery: Path) -> None:
    loaded = load_artifacts([stage_c_delivery])
    assert {(row.model_id, row.fold) for row in loaded.predictions} == {
        (f"tabm_stage_c_seed_{seed}", fold)
        for seed in (42, 2026, 3407)
        for fold in ("2022->2023", "2023->2024")
    }
    assert {row.trust for row in loaded.predictions} == {TrustClass.RULE_SAFE}


def test_catboost_adapters_load_oof_and_prefixes(
    blend_delivery: Path, deployment_review: Path
) -> None:
    loaded = load_artifacts([blend_delivery, deployment_review])
    ids = {row.model_id for row in loaded.predictions}
    assert "catboost_hand_matchup_seed_42" in ids
    assert {f"catboost_prefix_{count}" for count in (4, 32, 64, 128, 192, 296, 400)} <= ids


def test_hierarchical_adapter_reconstructs_h2_and_h3(hierarchical_review: Path) -> None:
    loaded = load_artifacts([hierarchical_review])
    assert {(row.model_id, row.fold) for row in loaded.predictions} == {
        (candidate, fold)
        for candidate in ("H1", "H2", "H3")
        for fold in ("2022->2023", "2023->2024")
    }


def test_quarantined_generic_oof_cannot_become_rule_safe(quarantined_oof: Path) -> None:
    loaded = load_artifacts([quarantined_oof])
    assert {row.trust for row in loaded.predictions} == {
        TrustClass.QUARANTINED_DIAGNOSTIC
    }


def test_unknown_submission_zip_is_missing_evidence(submission_zip: Path) -> None:
    loaded = load_artifacts([submission_zip])
    assert loaded.predictions == ()
    assert loaded.inventory[0].status == "missing_evidence"
```

Also test duplicate ZIP members, traversal, manifest hash tampering, duplicate `(model_id, fold)` with differing prediction hashes, and a file swap between inventory hashing and reading.

- [ ] **Step 2: Run adapter tests and confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_artifacts.py -q
```

Expected: collection fails because `load_artifacts` is missing.

- [ ] **Step 3: Implement artifact detection through verified manifests**

Create these types and entry point in `artifacts.py`:

```python
@dataclass(frozen=True)
class LoadedArtifacts:
    predictions: tuple[PredictionSet, ...]
    inventory: tuple[ArtifactRecord, ...]


DEFAULT_AUDIT_ROLES = tuple(ArtifactRole)


def load_artifacts(
    paths: Sequence[Path],
    *,
    expected_roles: Sequence[ArtifactRole] = DEFAULT_AUDIT_ROLES,
) -> LoadedArtifacts:
    decoded = tuple(_decode_verified_artifact(Path(path)) for path in paths)
    predictions = tuple(
        prediction for item in decoded for prediction in item.predictions
    )
    inventory = [item.inventory for item in decoded]
    observed = {item.inventory.role for item in decoded}
    for role in expected_roles:
        if role not in observed:
            inventory.append(ArtifactRecord(
                path=Path("__MISSING__") / role.value,
                sha256="0" * 64,
                role=role,
                artifact_kind="missing",
                status="missing_evidence",
                detail=f"artifact role was not supplied: {role.value}",
            ))
    return LoadedArtifacts(
        predictions=_deduplicate_predictions(predictions),
        inventory=tuple(inventory),
    )
```

Detection order must be manifest-based:

1. `tabm_colab_stage_C_delivery`: call `verify_delivery_bundle`, open the verified nested review, load only `c_final` seeds `42`, `2026`, `3407` with both fold suffixes.
2. `tabm_row_feature_stage_P_delivery`: read its four expected bindings, call `row_feature_colab.verify_delivery`, open the verified nested review, and load every completed Stage P prediction member with its candidate ID and fold from `metrics/job_results.json`.
3. `catboost_tabm_blend_delivery`: read self-declared bindings, call `verify_delivery(path, expected_bindings=bindings)`, load the two `catboost__hand_matchup` OOF members.
4. `catboost_deployment_review_v1`: call `verify_deployment_review` with its self-declared bindings, expand `p_4` through `p_400` into seven `PredictionSet` instances per fold.
5. `hierarchical_tabm_review_v1`: call `verify_review_bundle`, load H1, parse canonical H2/H3 states through `calibration_state_from_payload`, and apply them to copies of H1 probabilities.
6. `oof_reset_quarantined_v1`: accept only a locally built diagnostic ZIP containing `manifest.json` plus OOF CSV members. Require `trust="quarantined_diagnostic"`, `submission_package=false`, member hashes, folds, model IDs and source provenance. Never infer this role from `submit_xgboost*.zip`.

Use the existing verifier before reading prediction bytes. Hash every consumed CSV again and bind it to `PredictionSet.prediction_sha256`. Re-stat and re-hash the outer file after decoding; if it changed, reject it.

Normalize source segment aliases while decoding: `pitcher_id_known` becomes `pitcher_known` and `batter_id_known` becomes `batter_known`. Reject a frame that contains both an alias and canonical column with different values.

Do not add a fallback that accepts arbitrary CSV or ZIP layouts. Unknown packages, including submission ZIPs with only `script.py` and model files, become `missing_evidence` inventory rows. After decoding all inputs, add one synthetic inventory row for every member of `expected_roles` that was not supplied. This is how absent row-feature and raw XGBoost OOF evidence remains visible without guessing a path.

- [ ] **Step 4: Enforce cross-artifact identity rules**

After all adapters return, key predictions by `(model_id, fold)`. Equal keys are allowed only when `prediction_sha256`, row count and target digest are identical. Otherwise raise:

```python
raise AuditArtifactError(
    f"conflicting prediction evidence: model={model_id} fold={fold}"
)
```

Keep rule-safe and quarantined candidates in separate sequences. A caller cannot override the trust class supplied by the verified artifact.

- [ ] **Step 5: Run adapter tests and existing artifact regression**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_artifacts.py \
  tests/test_tabm_campaign_artifacts.py \
  tests/test_catboost_tabm_blend_artifacts.py \
  tests/test_catboost_deployment_artifacts.py \
  tests/test_hierarchical_tabm_artifacts.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit verified artifact adapters**

```bash
git add experiments/oof_reset_audit/artifacts.py \
  tests/test_oof_reset_audit_artifacts.py
git commit -m "feat: load verified OOF audit artifacts"
```

## Task 4: Build paired audit tables and diagnosis evidence

**Files:**
- Create: `experiments/oof_reset_audit/run.py`
- Create: `tests/test_oof_reset_audit_run.py`
- Modify: `experiments/oof_reset_audit/__init__.py`

- [ ] **Step 1: Write failing end-to-end audit tests**

Build six-row synthetic `PredictionSet` fixtures for two folds and add:

```python
def test_audit_uses_seed_3407_anchor_and_never_blends_quarantined(tmp_path: Path) -> None:
    result = run_audit(
        predictions=(anchor_old, anchor_latest, safe_old, safe_latest,
                     quarantined_old, quarantined_latest),
        inventory=inventory,
        output_root=tmp_path,
    )
    paired = pd.read_csv(result.output_dir / "paired_comparison.csv")
    assert set(paired["anchor_model_id"]) == {"tabm_stage_c_seed_3407"}
    assert "quarantined" not in set(
        json.loads((result.output_dir / "next_experiment.json").read_text())["eligible_model_ids"]
    )


def test_audit_marks_unpaired_and_missing_evidence_without_fabricated_delta(tmp_path: Path) -> None:
    result = run_audit(
        predictions=(anchor_old, safe_latest),
        inventory=missing_inventory,
        output_root=tmp_path,
    )
    table = pd.read_csv(result.output_dir / "model_comparison.csv")
    assert set(table["comparison_class"]) == {"descriptive_only"}
    next_value = json.loads((result.output_dir / "next_experiment.json").read_text())
    assert next_value["missing_evidence"]
    assert next_value["automatic_acceptance"] is False


def test_audit_outputs_exact_allowlist_and_deterministic_tables(tmp_path: Path) -> None:
    first = run_audit(predictions=safe_predictions, inventory=inventory, output_root=tmp_path)
    second = run_audit(predictions=safe_predictions, inventory=inventory, output_root=tmp_path)
    assert first.output_dir != second.output_dir
    assert set(first.output_files) == {
        "artifact_inventory.json", "audit_summary.md", "calibration_deciles.csv",
        "correlation_matrix.csv", "model_comparison.csv", "next_experiment.json",
        "paired_comparison.csv", "segment_diagnostics.csv",
        "oof_reset_audit_results.zip",
    }
```

Add exact tests for these diagnoses:

- shared month-direction drift plus high residual correlation emits `RECENCY_WEIGHTING` evidence;
- stable safe-model fixed blends emit `DIVERSE_BLEND` evidence;
- no stable improvement emits `STOP_AND_REFRAME` evidence;
- overlapping evidence returns multiple `supported_directions` and no automatic winner.

- [ ] **Step 2: Run end-to-end tests and confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_run.py -q
```

Expected: collection fails because `run_audit` is missing.

- [ ] **Step 3: Implement table construction**

Add:

```python
@dataclass(frozen=True)
class AuditResult:
    output_dir: Path
    result_zip: Path
    output_files: tuple[str, ...]


def run_audit(
    *,
    predictions: Sequence[PredictionSet],
    inventory: Sequence[ArtifactRecord],
    output_root: Path,
) -> AuditResult:
    run_dir = _new_run_directory(Path(output_root))
    model_table, groups = _model_table_and_groups(tuple(predictions))
    paired_table, segment_table, correlation_table = _paired_tables(groups)
    calibration_table = _calibration_table(tuple(predictions))
    next_experiment = _diagnose(
        predictions=tuple(predictions),
        inventory=tuple(inventory),
        paired=paired_table,
        segments=segment_table,
        correlations=correlation_table,
    )
    files = _publish_reports(
        run_dir=run_dir,
        inventory=tuple(inventory),
        model_table=model_table,
        paired_table=paired_table,
        segment_table=segment_table,
        correlation_table=correlation_table,
        calibration_table=calibration_table,
        next_experiment=next_experiment,
    )
    result_zip = _publish_result_zip(run_dir, files)
    return AuditResult(run_dir, result_zip, tuple(sorted((*files, result_zip.name))))
```

Create the output directory as `oof_reset_audit_<UTC timestamp>_<8 hex nonce>` with mode-safe atomic file publication. Refuse symlink output roots and existing run directories.

For every `PredictionSet`, write one `model_comparison` row. Mark it `paired` only when seed 3407 or an explicitly configured anchor has the exact same fold and row set. Otherwise use `descriptive_only` or `quarantined_diagnostic`.

For each paired safe candidate, compute the three fixed blend weights. Quarantined candidates get correlations and descriptive metrics only; do not create blend rows for them.

- [ ] **Step 4: Implement evidence flags without automatic candidate approval**

Write `next_experiment.json` with this fixed shape:

```json
{
  "schema_version": 1,
  "automatic_acceptance": false,
  "eligible_model_ids": [],
  "quarantined_model_ids": [],
  "missing_evidence": [],
  "supported_directions": [],
  "evidence": {
    "recency_weighting": {},
    "diverse_blend": {},
    "stop_and_reframe": {}
  }
}
```

Populate `supported_directions` only from rule-safe comparisons. Keep the decision conditions from the design document exact. If conditions overlap, retain all supported directions and state `manual_review_required=true`; do not impose an arbitrary priority.

- [ ] **Step 5: Write deterministic small reports and ZIP**

Write CSV files with stable column order, UTF-8, `index=False`, and float format `%.12g`. Write JSON with `sort_keys=True`, compact separators and `allow_nan=False`. `audit_summary.md` must summarize:

- verified and missing inputs;
- paired comparison groups;
- best and worst fold deltas without mixing protocols;
- strongest month or segment drift;
- safe and quarantined correlation evidence in separate sections;
- supported next directions and why no model is automatically approved.

The ZIP must include the eight report files and exclude itself, original predictions, data, models, checkpoints and submission files. Verify its member names, CRC, uncompressed total size and SHA-256 immediately after atomic publication.

- [ ] **Step 6: Run end-to-end and combined focused tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_metrics.py \
  tests/test_oof_reset_audit_artifacts.py \
  tests/test_oof_reset_audit_run.py -q
```

Expected: all OOF reset audit tests pass.

- [ ] **Step 7: Commit orchestration and reports**

```bash
git add experiments/oof_reset_audit/run.py \
  experiments/oof_reset_audit/__init__.py \
  tests/test_oof_reset_audit_run.py
git commit -m "feat: produce OOF reset audit reports"
```

## Task 5: Add one-command user handoff

**Files:**
- Create: `tools/run_oof_reset_audit.py`
- Create: `docs/OOF_RESET_AUDIT.md`
- Modify: `tests/test_oof_reset_audit_run.py`

- [ ] **Step 1: Write failing CLI tests**

Add subprocess tests that run the script from outside the repository and verify project imports:

```python
def test_cli_help_works_outside_project_root(tmp_path: Path) -> None:
    completed = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "tools/run_oof_reset_audit.py"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--artifact" in completed.stdout


def test_cli_reports_exact_success_marker(tmp_path: Path, audit_inputs: tuple[Path, ...]) -> None:
    completed = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "tools/run_oof_reset_audit.py"),
         *(item for path in audit_inputs for item in ("--artifact", str(path))),
         "--output-root", str(tmp_path / "runs")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "OOF_RESET_AUDIT_SUCCESS" in completed.stdout
```

Also require `OOF_RESET_AUDIT_ERROR stage=<stage> type=<type> message=<message>` on failure and no result ZIP on failed verification.

- [ ] **Step 2: Run CLI tests and confirm RED**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_run.py -k cli -q
```

Expected: FAIL because the CLI script does not exist.

- [ ] **Step 3: Implement the thin CLI**

Start the script with the proven project-root bootstrap pattern:

```python
from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.oof_reset_audit.artifacts import load_artifacts
from experiments.oof_reset_audit.run import run_audit
```

Arguments:

```text
--artifact PATH       repeatable, at least one
--output-root PATH    required
```

Resolve every input path, reject directories and symlinks, call `load_artifacts`, then `run_audit`. Print only concise stage markers plus:

```text
OOF_RESET_AUDIT_SUCCESS output_dir=<absolute path> result_zip=<absolute path> sha256=<64 hex>
```

- [ ] **Step 4: Write the Korean runbook**

Document the current recommended command without claiming absent inputs exist:

```bash
cd /Users/yonghyun/Documents/lg-aimers-9th-competition

artifacts/tabm_submission_python311/bin/python \
  tools/run_oof_reset_audit.py \
  --artifact /Users/yonghyun/Downloads/tabm_colab_stage_C_delivery.zip \
  --artifact /Users/yonghyun/Downloads/catboost_tabm_blend_delivery.zip \
  --artifact /Users/yonghyun/Downloads/catboost_deployment_review.zip \
  --artifact /Users/yonghyun/Downloads/hierarchical_tabm_review.zip \
  --output-root artifacts/oof_reset_audit_runs
```

State that the locally absent row-feature delivery and raw XGBoost OOF will appear as missing evidence; `submit_xgboost_v3.zip` must not be supplied because it contains no OOF and is not diagnostic evidence. Include purpose, inputs, expected 5–15 minute CPU runtime, 2–6GB memory estimate, rerun safety, exact success marker, exact error marker, and the one result ZIP to return.

- [ ] **Step 5: Run CLI and focused regression tests**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_metrics.py \
  tests/test_oof_reset_audit_artifacts.py \
  tests/test_oof_reset_audit_run.py \
  tests/test_hierarchical_tabm_calibration.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit CLI and runbook**

```bash
git add tools/run_oof_reset_audit.py docs/OOF_RESET_AUDIT.md \
  tests/test_oof_reset_audit_run.py
git commit -m "docs: add OOF reset audit handoff"
```

## Task 6: Final scope and regression verification

**Files:**
- Verify only; modify a file only if a failing test identifies an in-scope defect.

- [ ] **Step 1: Run the complete targeted suite**

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_metrics.py \
  tests/test_oof_reset_audit_artifacts.py \
  tests/test_oof_reset_audit_run.py \
  tests/test_hierarchical_tabm_calibration.py \
  tests/test_hierarchical_tabm_artifacts.py \
  tests/test_tabm_campaign_artifacts.py \
  tests/test_tabm_campaign_colab_recovery.py \
  tests/test_catboost_tabm_blend_artifacts.py \
  tests/test_catboost_tabm_blend_colab.py \
  tests/test_catboost_deployment_artifacts.py -q
```

Expected: all selected tests pass with no warning converted to failure.

- [ ] **Step 2: Run static verification**

```bash
artifacts/tabm_submission_python311/bin/python -m compileall -q \
  experiments/oof_reset_audit \
  experiments/hierarchical_tabm/calibration.py \
  tools/run_oof_reset_audit.py
git diff --check
```

Expected: both commands exit 0 with no output.

- [ ] **Step 3: Audit rule and packaging boundaries**

```bash
rg -n "test\.csv|sample_submission|submission\.csv|submit_xgboost|files\.download|drive\.mount|requests\.|urllib" \
  experiments/oof_reset_audit tools/run_oof_reset_audit.py docs/OOF_RESET_AUDIT.md
```

Expected: only the runbook's explicit warning about `submit_xgboost_v3.zip` may match. Runtime code must contain no evaluation-file access, submission creation, network or Drive logic.

- [ ] **Step 4: Inspect exact change scope**

```bash
git status --short
git log --oneline --decorate -8
```

Expected: audit work is committed in small commits. Pre-existing user modifications remain untouched and are reported separately; no model, OOF, checkpoint, result ZIP or original data is tracked.

- [ ] **Step 5: Stop before the full-data audit**

Do not run `tools/run_oof_reset_audit.py` against the user's full ZIPs during implementation. Hand the exact command from `docs/OOF_RESET_AUDIT.md` to the user. The user returns only `oof_reset_audit_results.zip` for interpretation.
