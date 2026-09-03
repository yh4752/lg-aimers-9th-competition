# Experiment Evidence Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 실험을 검증 가능한 JSON 장부로 통합하고, 중복·누락·비교 불가능한 수치를 드러내는 감사 보고서와 다음 주력 캠페인 결정을 생성한다.

**Architecture:** 범용 ZIP 파서를 만들지 않는다. `reports/experiment_registry.json`을 유일한 정규화 원본으로 두고, 작은 Python 모듈이 스키마·상태·수치·출처를 검증한 뒤 Markdown 감사 보고서를 결정적으로 렌더링한다. 현재 확인 가능한 두 Privileged 번들은 명시적인 파일 인자로만 해시와 manifest를 교차 검증하며, 접근할 수 없는 과거 결과는 추정하지 않고 evidence gap으로 남긴다.

**Tech Stack:** Python 3.11 표준 라이브러리, pytest 8.4.1, JSON, Markdown, SHA-256

---

## 파일 구조

- Create: `experiments/experiment_registry.py` — registry 로딩, 스키마 검증, 비교 그룹 요약, 감사 Markdown 렌더링
- Create: `tools/build_experiment_audit.py` — 명령행 진입점, 선택적 번들 해시 검증, 원자적 보고서 생성
- Create: `tests/test_experiment_registry.py` — 상태·수치·출처·비교 그룹·렌더링 단위 테스트
- Create: `tests/test_build_experiment_audit.py` — CLI 성공·오류·결정적 출력 테스트
- Create: `reports/experiment_registry.schema.json` — 사람이 읽을 수 있는 필드 계약
- Create: `reports/experiment_registry.json` — 확인된 실험과 evidence gap의 정규화 원본
- Create: `reports/EXPERIMENT_RESET_AUDIT.md` — 생성된 감사 보고서
- Modify: `reports/EXPERIMENT_LEDGER.md` — S3 이후 확인 결과와 감사 보고서 링크 추가
- Modify: `README.md` — 현재 판정과 다음 방향을 감사 결과에 맞게 갱신

## Task 1: Registry 계약과 실패 검증부터 고정

**Files:**
- Create: `tests/test_experiment_registry.py`
- Create: `experiments/experiment_registry.py`
- Create: `reports/experiment_registry.schema.json`

- [ ] **Step 1: 최소 정상 레코드와 실패 조건 테스트 작성**

`tests/test_experiment_registry.py`에 다음 형태의 fixture와 테스트를 작성한다.

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.experiment_registry import RegistryError, load_registry, validate_registry


def record(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": "candidate_a",
        "family": "catboost",
        "variant": "anchor_residual",
        "completed_at": "2026-08-27",
        "status": "rejected",
        "status_reason": "weighted gain below gate",
        "evidence_grade": "B",
        "comparison_group": "e2_temporal_3fold",
        "folds": ["2021->2022", "2022->2023", "2023->2024"],
        "validation_rows": 245789,
        "baseline_id": "tree_expert_e2_c1_catboost",
        "baseline_brier": 0.248,
        "candidate_brier": 0.24796,
        "weighted_gain": 0.00004,
        "worst_fold_gain": 0.000001,
        "latest_fold_gain": 0.00002,
        "max_segment_regression": 0.0,
        "residual_correlation": 0.98,
        "seed_stability": "3/3_non_worse",
        "public_score": None,
        "submission_sha256": None,
        "artifact_paths": [],
        "artifact_sha256": [],
        "evidence_paths": ["reports/rejections/candidate_a.json"],
        "rule_audit_status": "passed",
        "row_independence_status": "passed",
        "failure_class": "performance",
        "lesson": "signal is positive but too small",
        "repeat_policy": "redefine",
    }
    base.update(changes)
    return base


def registry(*records: dict[str, object]) -> dict[str, object]:
    return {"schema_version": 1, "experiments": list(records), "evidence_gaps": []}


def test_minimal_registry_is_valid() -> None:
    validate_registry(registry(record()))


@pytest.mark.parametrize("status", ["failed", "rejected", "accepted", "diagnostic"])
def test_allowed_statuses(status: str) -> None:
    validate_registry(registry(record(status=status)))


def test_duplicate_experiment_id_is_rejected() -> None:
    with pytest.raises(RegistryError, match="duplicate experiment_id"):
        validate_registry(registry(record(), record()))


def test_gain_direction_and_probability_metric_ranges_are_checked() -> None:
    with pytest.raises(RegistryError, match="baseline_brier"):
        validate_registry(registry(record(baseline_brier=1.01)))
    with pytest.raises(RegistryError, match="residual_correlation"):
        validate_registry(registry(record(residual_correlation=1.01)))


def test_public_score_requires_score() -> None:
    with pytest.raises(RegistryError, match="public_score"):
        validate_registry(registry(record(status="public_scored", public_score=None)))


def test_new_verified_public_score_requires_submission_hash() -> None:
    with pytest.raises(RegistryError, match="submission_sha256"):
        validate_registry(registry(record(status="public_scored", public_score=977.38)))


def test_legacy_grade_c_public_score_can_lack_hash() -> None:
    validate_registry(registry(record(
        status="public_scored", public_score=828.99, evidence_grade="C",
        status_reason="legacy public score; submission hash was not retained",
    )))


def test_accepted_candidate_cannot_fail_rule_audit() -> None:
    with pytest.raises(RegistryError, match="rule_audit_status"):
        validate_registry(registry(record(status="accepted", rule_audit_status="failed")))
```

- [ ] **Step 2: 테스트가 구현 부재로 실패하는지 확인**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_experiment_registry.py -q
```

Expected: collection 단계에서 `ModuleNotFoundError: No module named 'experiments.experiment_registry'`.

- [ ] **Step 3: 최소 validator 구현**

`experiments/experiment_registry.py`에 다음 공개 인터페이스를 구현한다.

```python
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence


class RegistryError(ValueError):
    pass


STATUSES = frozenset({"failed", "rejected", "accepted", "public_scored", "diagnostic"})
EVIDENCE_GRADES = frozenset({"A", "B", "C"})
FAILURE_CLASSES = frozenset({
    "none", "performance", "instability", "diversity", "deployment_alignment",
    "data_eligibility", "runtime", "rule_quarantine", "evidence_missing",
})
REPEAT_POLICIES = frozenset({"closed", "redefine", "retain", "blocked", "reference_only"})
RULE_STATUSES = frozenset({"passed", "failed", "not_run", "quarantined", "unknown"})
REQUIRED_FIELDS = frozenset({
    "experiment_id", "family", "variant", "completed_at", "status", "status_reason",
    "evidence_grade", "comparison_group", "folds", "validation_rows", "baseline_id",
    "baseline_brier", "candidate_brier", "weighted_gain", "worst_fold_gain",
    "latest_fold_gain", "max_segment_regression", "residual_correlation",
    "seed_stability", "public_score", "submission_sha256", "artifact_paths",
    "artifact_sha256", "evidence_paths", "rule_audit_status",
    "row_independence_status", "failure_class", "lesson", "repeat_policy",
})


def _number_or_none(value: object, name: str, lower: float | None = None,
                    upper: float | None = None) -> None:
    if value is None:
        return
    if type(value) not in (int, float):
        raise RegistryError(f"{name} must be numeric or null")
    numeric = float(value)
    if lower is not None and numeric < lower:
        raise RegistryError(f"{name} is below {lower}")
    if upper is not None and numeric > upper:
        raise RegistryError(f"{name} is above {upper}")


def validate_registry(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != 1:
        raise RegistryError("schema_version must be 1")
    experiments = payload.get("experiments")
    gaps = payload.get("evidence_gaps")
    if type(experiments) is not list or type(gaps) is not list:
        raise RegistryError("experiments and evidence_gaps must be lists")
    seen: set[str] = set()
    for raw in experiments:
        if type(raw) is not dict or set(raw) != REQUIRED_FIELDS:
            raise RegistryError("experiment member set differs")
        experiment_id = raw["experiment_id"]
        if type(experiment_id) is not str or not experiment_id:
            raise RegistryError("experiment_id must be a non-empty string")
        if experiment_id in seen:
            raise RegistryError(f"duplicate experiment_id: {experiment_id}")
        seen.add(experiment_id)
        if raw["status"] not in STATUSES:
            raise RegistryError("status is invalid")
        if raw["evidence_grade"] not in EVIDENCE_GRADES:
            raise RegistryError("evidence_grade is invalid")
        if raw["failure_class"] not in FAILURE_CLASSES:
            raise RegistryError("failure_class is invalid")
        if raw["repeat_policy"] not in REPEAT_POLICIES:
            raise RegistryError("repeat_policy is invalid")
        if raw["rule_audit_status"] not in RULE_STATUSES:
            raise RegistryError("rule_audit_status is invalid")
        if raw["row_independence_status"] not in RULE_STATUSES:
            raise RegistryError("row_independence_status is invalid")
        _number_or_none(raw["baseline_brier"], "baseline_brier", 0.0, 1.0)
        _number_or_none(raw["candidate_brier"], "candidate_brier", 0.0, 1.0)
        _number_or_none(raw["residual_correlation"], "residual_correlation", -1.0, 1.0)
        _number_or_none(raw["public_score"], "public_score")
        if raw["status"] == "public_scored" and raw["public_score"] is None:
            raise RegistryError("public_score is required for public_scored")
        if (raw["status"] == "public_scored" and raw["evidence_grade"] in {"A", "B"}
                and raw["submission_sha256"] is None):
            raise RegistryError("submission_sha256 is required for verified public_scored")
        if raw["status"] == "accepted" and raw["rule_audit_status"] == "failed":
            raise RegistryError("accepted record cannot have failed rule_audit_status")


def load_registry(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise RegistryError("registry root must be an object")
    validate_registry(payload)
    return payload


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
```

`reports/experiment_registry.schema.json`은 JSON Schema draft 2020-12로 작성하고,
`schema_version=1`, 위 enum과 필수 필드, `additionalProperties=false`를 동일하게 고정한다.

- [ ] **Step 4: validator 테스트 통과 확인**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_experiment_registry.py -q
```

Expected: 모든 테스트 PASS.

- [ ] **Step 5: 첫 커밋**

```bash
git add experiments/experiment_registry.py tests/test_experiment_registry.py reports/experiment_registry.schema.json
git commit -m "feat: validate experiment evidence registry"
```

## Task 2: 기존 23개 실험과 Privileged 2개 결과 등록

**Files:**
- Create: `reports/experiment_registry.json`
- Modify: `tests/test_experiment_registry.py`

- [ ] **Step 1: 실제 registry 불변 조건 테스트 추가**

```python
def test_repository_registry_is_valid_and_contains_known_public_scores() -> None:
    payload = load_registry(Path("reports/experiment_registry.json"))
    experiments = {row["experiment_id"]: row for row in payload["experiments"]}
    assert len(experiments) == 25
    assert experiments["catboost_smooth_v1"]["public_score"] == 828.9963889533
    assert experiments["xgboost_aggressive_capacity_v1"]["public_score"] == 820.9583317093
    assert experiments["tabm_hand_matchup_version_d_seed3407_v1"]["public_score"] == 872.3920184667
    assert experiments["tree_expert_e2_c1_catboost"]["public_score"] == 977.3809532715
    assert experiments["tree_privileged_profile_p_only_v1"]["weighted_gain"] == pytest.approx(
        0.00002026775135556824
    )
    assert experiments["tree_privileged_profile_p_only_v1"]["status"] == "rejected"
```

- [ ] **Step 2: 실제 registry가 없어 실패하는지 확인**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_experiment_registry.py::test_repository_registry_is_valid_and_contains_known_public_scores -q
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: 25개 레코드를 명시적으로 이관**

`reports/EXPERIMENT_LEDGER.md`의 1~23번을 같은 `experiment_id`로 옮긴다. 값이 없는
metric은 `null`로 유지한다. 다음 두 레코드를 추가해 총 25개로 고정한다.

| experiment_id | status | evidence | 핵심 수치 | repeat_policy |
|---|---|---|---|---|
| `tree_privileged_trackman_teacher_v1` | `diagnostic` | A | exact match 전체 약 `0.0039`, latest 약 `0.0029`; 요구 coverage `0.30/0.20` 미달 | `closed` |
| `tree_privileged_profile_p_only_v1` | `rejected` | A | weighted gain `0.00002026775135556824`, latest `0.00002244958184524637`, 3 folds improved, max segment regression `0.000001293162079529786` | `redefine` |

TrackMan handoff SHA-256은
`5754124bb6534b2163953150a2b64d770877d05d446317ed0e5cb5b07723ddd2`, P-only
handoff SHA-256은
`b89c77570560422914d7fba99ff13c32733226e3ce91e7a5404d134a627bd8fc`로 등록한다.

P-only의 actual 3-seed 단순 평균 진단 `0.00004181384518595898`은 gate 판정값과
혼동하지 않도록 `lesson`에만 기록하고 `weighted_gain`은 공식 acceptance의
`0.00002026775135556824`로 둔다. 2023 fold gain `0.0000045861`도 `lesson`에
병목으로 기록한다.

`evidence_gaps`에는 현재 원본 review/handoff를 로컬에서 확인하지 못한 아래 여섯 항목을
정확히 기록한다.

```json
[
  {"experiment_id":"temporal_portfolio_t1","reason":"result bundle is not currently available","required_evidence":"review or handoff bundle"},
  {"experiment_id":"temporal_portfolio_t2a","reason":"result bundle is not currently available","required_evidence":"review or handoff bundle"},
  {"experiment_id":"temporal_portfolio_t2b","reason":"result bundle is not currently available","required_evidence":"review or handoff bundle"},
  {"experiment_id":"temporal_portfolio_t2c","reason":"result bundle is not currently available","required_evidence":"review or handoff bundle"},
  {"experiment_id":"tree_expert_rf","reason":"result bundle is not currently available","required_evidence":"review or handoff bundle"},
  {"experiment_id":"anchor_residual_hierarchical_s4","reason":"result bundle is not currently available","required_evidence":"review or handoff bundle"}
]
```

- [ ] **Step 4: 실제 registry 검증**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_experiment_registry.py -q
```

Expected: PASS, 실험 25건과 Public 네 점수 일치.

- [ ] **Step 5: registry 커밋**

```bash
git add reports/experiment_registry.json tests/test_experiment_registry.py
git commit -m "data: register verified competition experiments"
```

## Task 3: 비교 그룹을 보존하는 감사 분석과 Markdown 렌더링

**Files:**
- Modify: `experiments/experiment_registry.py`
- Modify: `tests/test_experiment_registry.py`

- [ ] **Step 1: 비교 그룹·기각 원인·다음 방향 테스트 작성**

```python
from experiments.experiment_registry import audit_registry, render_audit_markdown


def test_audit_never_ranks_different_comparison_groups_together() -> None:
    payload = registry(
        record(experiment_id="a", comparison_group="group_a", candidate_brier=0.20),
        record(experiment_id="b", comparison_group="group_b", candidate_brier=0.10),
    )
    audit = audit_registry(payload)
    assert set(audit["comparison_groups"]) == {"group_a", "group_b"}
    assert "global_brier_ranking" not in audit


def test_audit_counts_rejection_classes_and_evidence_gaps() -> None:
    payload = registry(
        record(experiment_id="a", failure_class="performance"),
        record(experiment_id="b", failure_class="diversity"),
    )
    payload["evidence_gaps"] = [{
        "experiment_id": "missing", "reason": "not available",
        "required_evidence": "review bundle",
    }]
    audit = audit_registry(payload)
    assert audit["failure_classes"] == {"diversity": 1, "performance": 1}
    assert audit["evidence_gap_count"] == 1


def test_rendered_audit_explains_deep_campaign_without_promising_score() -> None:
    markdown = render_audit_markdown(audit_registry(registry(record())))
    assert "구조적으로 깊은 캠페인" in markdown
    assert "점수를 보장" in markdown
    assert "comparison_group" in markdown
```

- [ ] **Step 2: 새 API 부재로 실패 확인**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_experiment_registry.py -q
```

Expected: `ImportError` for `audit_registry` 또는 `render_audit_markdown`.

- [ ] **Step 3: 분석·렌더링 구현**

`audit_registry()`는 다음 키만 반환하는 결정적 dict를 만든다.

```python
{
    "experiment_count": int,
    "status_counts": dict[str, int],
    "evidence_grade_counts": dict[str, int],
    "failure_classes": dict[str, int],
    "comparison_groups": dict[str, list[dict[str, object]]],
    "public_scores": list[dict[str, object]],
    "evidence_gap_count": int,
    "evidence_gaps": list[dict[str, str]],
    "closed_families": list[str],
    "redefine_families": list[str],
    "retained_experiments": list[str],
}
```

모든 dict key와 list는 알파벳 또는 `completed_at`, `experiment_id` 순으로 정렬한다.
`comparison_groups` 안에서만 candidate Brier와 gain을 표시하며 전역 Brier 순위를
만들지 않는다.

`render_audit_markdown()`은 다음 순서의 섹션을 생성한다.

1. `# 실험 증거 재감사`
2. `## 결론`
3. `## 실제 Public 제출`
4. `## 비교 가능한 OOF 그룹`
5. `## 반복 기각 원인`
6. `## 증거가 부족한 실행`
7. `## 닫을 계열과 다시 정의할 계열`
8. `## 다음 단일 캠페인`

마지막 섹션은 점수를 보장하지 않는다고 명시하고, 시즌 snapshot, R/F 전문가, 안정적으로
정의되는 상황 전문가, CatBoost 다중 seed, 오차 다양성이 확인된 이종 residual,
계층 calibration을 “구조적으로 깊은 캠페인”의 구성요소로 적는다.

- [ ] **Step 4: 분석·렌더링 테스트 통과 확인**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_experiment_registry.py -q
```

Expected: PASS.

- [ ] **Step 5: 분석 코드 커밋**

```bash
git add experiments/experiment_registry.py tests/test_experiment_registry.py
git commit -m "feat: summarize experiment evidence"
```

## Task 4: 명시적 번들 교차 검증과 감사 CLI

**Files:**
- Create: `tools/build_experiment_audit.py`
- Create: `tests/test_build_experiment_audit.py`

- [ ] **Step 1: CLI fixture 테스트 작성**

```python
from __future__ import annotations

import json
from pathlib import Path
import subprocess


PYTHON = Path("artifacts/tabm_submission_python311/bin/python")
TOOL = Path("tools/build_experiment_audit.py")


def test_cli_writes_deterministic_report(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "schema_version": 1,
        "experiments": [],
        "evidence_gaps": [],
    }), encoding="utf-8")
    output = tmp_path / "audit.md"
    result = subprocess.run(
        [str(PYTHON), str(TOOL), "--registry", str(registry), "--output", str(output)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "EXPERIMENT_AUDIT_SUCCESS" in result.stdout
    first = output.read_bytes()
    subprocess.run(
        [str(PYTHON), str(TOOL), "--registry", str(registry), "--output", str(output)],
        text=True, capture_output=True, check=True,
    )
    assert output.read_bytes() == first


def test_cli_rejects_wrong_declared_artifact_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "review.zip"
    artifact.write_bytes(b"fixture")
    result = subprocess.run(
        [str(PYTHON), str(TOOL), "--registry", "reports/experiment_registry.json",
         "--output", str(tmp_path / "audit.md"), "--artifact",
         f"tree_privileged_profile_p_only_v1={'0' * 64}:{artifact}"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1
    assert "EXPERIMENT_AUDIT_ERROR" in result.stdout
    assert "artifact_sha256_differs" in result.stdout
```

- [ ] **Step 2: CLI가 없어 실패 확인**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_build_experiment_audit.py -q
```

Expected: CLI 실행 return code가 2이고 파일을 찾을 수 없다는 오류.

- [ ] **Step 3: 작고 명시적인 CLI 구현**

`tools/build_experiment_audit.py`는 다음 인자를 받는다.

```text
--registry PATH               기본값 reports/experiment_registry.json
--output PATH                 필수
--artifact ID=SHA256:PATH     선택, 반복 가능
```

처리 순서는 registry 검증, 각 명시적 artifact의 일반 파일·비심볼릭 링크 검사, SHA-256
일치 검사, Markdown 렌더링, 같은 디렉터리의 임시 파일에 `fsync`, `Path.replace()`를
이용한 원자적 교체다. 성공 로그는 다음 형식으로 고정한다.

```text
EXPERIMENT_AUDIT_SUCCESS experiments=25 evidence_gaps=6 output=/absolute/path sha256=<64 hex>
```

오류 로그는 다음 형식이며 traceback을 stdout에 출력하지 않는다.

```text
EXPERIMENT_AUDIT_ERROR stage=<registry|artifacts|render> type=<class> message=<underscored message>
```

- [ ] **Step 4: CLI 테스트 통과 확인**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest tests/test_build_experiment_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: CLI 커밋**

```bash
git add tools/build_experiment_audit.py tests/test_build_experiment_audit.py
git commit -m "feat: build deterministic experiment audit"
```

## Task 5: 실제 감사 보고서 생성과 기존 문서 정합화

**Files:**
- Create: `reports/EXPERIMENT_RESET_AUDIT.md`
- Modify: `reports/EXPERIMENT_LEDGER.md`
- Modify: `README.md`

- [ ] **Step 1: 현재 접근 가능한 Privileged 번들의 해시 재확인**

Run:

```bash
shasum -a 256 \
  "/path/to/Downloads/tree_privileged_handoff.zip" \
  "/path/to/Downloads/tree_privileged_handoff (1).zip"
```

Expected: P-only 번들의 SHA-256이
`b89c77570560422914d7fba99ff13c32733226e3ce91e7a5404d134a627bd8fc`와 일치한다.
첫 TrackMan 번들의 SHA-256도
`5754124bb6534b2163953150a2b64d770877d05d446317ed0e5cb5b07723ddd2`와 일치한다.

- [ ] **Step 2: 실제 감사 보고서 생성**

Run:

```bash
artifacts/tabm_submission_python311/bin/python tools/build_experiment_audit.py \
  --registry reports/experiment_registry.json \
  --output reports/EXPERIMENT_RESET_AUDIT.md \
  --artifact tree_privileged_profile_p_only_v1=b89c77570560422914d7fba99ff13c32733226e3ce91e7a5404d134a627bd8fc:"/path/to/Downloads/tree_privileged_handoff (1).zip"
```

Expected: `EXPERIMENT_AUDIT_SUCCESS experiments=25 evidence_gaps=6`.

- [ ] **Step 3: 장부와 README를 감사 결과에 맞춰 수정**

`reports/EXPERIMENT_LEDGER.md`에 다음 두 행을 추가한다.

- TrackMan exact-match teacher: coverage 부족으로 `diagnostic`, 계열 종료
- P-only rolling profile: 공식 weighted gain `0.0000202678`, 최소 gain gate 미달로
  `rejected`, 2023 fold 병목

또한 evidence gap 여섯 건은 결과를 추정하지 않고 “원본 번들 재확보 전 미확정”으로
별도 표에 둔다.

`README.md`의 `최근 판정`을 P-only rejection으로 바꾸고, `다음 방향`을
`검증 장부 재감사 후 구조적으로 깊은 단일 캠페인`으로 바꾼다. 현재 최고 Public
`977.3809532715`는 그대로 유지한다.

- [ ] **Step 4: 문서·registry 정합성 테스트 추가 및 실행**

`tests/test_experiment_registry.py`에 다음 검사를 추가한다.

```python
def test_generated_audit_and_readme_match_registry_public_scores() -> None:
    payload = load_registry(Path("reports/experiment_registry.json"))
    scores = [str(row["public_score"]) for row in payload["experiments"]
              if row["public_score"] is not None]
    audit = Path("reports/EXPERIMENT_RESET_AUDIT.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    for score in scores:
        assert score in audit
    assert "977.3809532715" in readme
    assert "tree_privileged_profile_p_only_v1" in audit
```

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_experiment_registry.py tests/test_build_experiment_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: 보고서와 문서 커밋**

```bash
git add reports/EXPERIMENT_RESET_AUDIT.md reports/EXPERIMENT_LEDGER.md README.md \
  tests/test_experiment_registry.py
git commit -m "docs: publish experiment reset audit"
```

## Task 6: 전체 정적 검증과 다음 캠페인 설계 진입 조건 확인

**Files:**
- Verify only

- [ ] **Step 1: 새 코드 문법 검사**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m py_compile \
  experiments/experiment_registry.py tools/build_experiment_audit.py
```

Expected: 출력 없이 exit code 0.

- [ ] **Step 2: 새 테스트 전체 실행**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_experiment_registry.py tests/test_build_experiment_audit.py -q
```

Expected: 모든 테스트 PASS.

- [ ] **Step 3: 기존 감사 테스트 회귀 확인**

Run:

```bash
artifacts/tabm_submission_python311/bin/python -m pytest \
  tests/test_oof_reset_audit_artifacts.py \
  tests/test_oof_reset_audit_metrics.py \
  tests/test_oof_reset_audit_run.py -q
```

Expected: 모든 테스트 PASS.

- [ ] **Step 4: 문서와 Git 오염 검사**

Run:

```bash
git diff --check
git status --short
```

Expected: whitespace 오류 없음. 기존 사용자의 TabM 관련 미커밋 파일과 notebook은
그대로 남고, 이 계획에서 만든 파일만 해당 커밋들에 포함됨.

- [ ] **Step 5: 다음 단계 판단**

다음 조건이 모두 만족될 때만 별도의 “구조적으로 깊은 주력 캠페인” 설계를 시작한다.

- registry 검증 PASS
- Public 네 점수와 README·감사 보고서 일치
- P-only 공식 gate와 actual 3-seed 진단값을 구분해 기록
- evidence gap 여섯 건을 결과로 추정하지 않음
- 닫힌 계열과 재정의할 계열이 명시됨
- 새 GPU 실행이나 제출 패키지를 이번 작업에서 생성하지 않음

조건을 통과하면 다음 설계는 Kaggle T4 x2 예산을 대상으로 시즌 snapshot,
R/F 전문가, 상황 전문가, 고용량 다중 seed CatBoost, 선택적 이종 residual과 계층
calibration을 한 캠페인에 묶되, 저비용 적격성 검사와 중간 중단 기준을 먼저 고정한다.
