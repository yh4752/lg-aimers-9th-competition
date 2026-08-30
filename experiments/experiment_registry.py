from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Mapping


class RegistryError(ValueError):
    pass


STATUSES = frozenset({"failed", "rejected", "accepted", "public_scored", "diagnostic"})
EVIDENCE_GRADES = frozenset({"A", "B", "C"})
FAILURE_CLASSES = frozenset({
    "none",
    "performance",
    "instability",
    "diversity",
    "deployment_alignment",
    "data_eligibility",
    "runtime",
    "rule_quarantine",
    "evidence_missing",
})
REPEAT_POLICIES = frozenset({"closed", "redefine", "retain", "blocked", "reference_only"})
RULE_STATUSES = frozenset({"passed", "failed", "not_run", "quarantined", "unknown"})
REQUIRED_FIELDS = frozenset({
    "experiment_id",
    "family",
    "variant",
    "completed_at",
    "status",
    "status_reason",
    "evidence_grade",
    "comparison_group",
    "folds",
    "validation_rows",
    "baseline_id",
    "baseline_brier",
    "candidate_brier",
    "weighted_gain",
    "worst_fold_gain",
    "latest_fold_gain",
    "max_segment_regression",
    "residual_correlation",
    "seed_stability",
    "public_score",
    "submission_sha256",
    "artifact_paths",
    "artifact_sha256",
    "evidence_paths",
    "rule_audit_status",
    "row_independence_status",
    "failure_class",
    "lesson",
    "repeat_policy",
})
GAP_FIELDS = frozenset({"experiment_id", "reason", "required_evidence"})
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _number_or_none(
    value: object,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    if value is None:
        return
    if type(value) not in (int, float):
        raise RegistryError(f"{name} must be numeric or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RegistryError(f"{name} must be finite")
    if lower is not None and numeric < lower:
        raise RegistryError(f"{name} is below {lower}")
    if upper is not None and numeric > upper:
        raise RegistryError(f"{name} is above {upper}")


def _string(value: object, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not str or not value.strip():
        raise RegistryError(f"{name} must be a non-empty string")


def _string_list(value: object, name: str) -> None:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise RegistryError(f"{name} must be a list of non-empty strings")


def validate_registry(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != 1:
        raise RegistryError("schema_version must be 1")
    if set(payload) != {"schema_version", "experiments", "evidence_gaps"}:
        raise RegistryError("registry member set differs")
    experiments = payload.get("experiments")
    gaps = payload.get("evidence_gaps")
    if type(experiments) is not list or type(gaps) is not list:
        raise RegistryError("experiments and evidence_gaps must be lists")

    seen: set[str] = set()
    for raw in experiments:
        if type(raw) is not dict or set(raw) != REQUIRED_FIELDS:
            raise RegistryError("experiment member set differs")
        experiment_id = raw["experiment_id"]
        _string(experiment_id, "experiment_id")
        assert isinstance(experiment_id, str)
        if experiment_id in seen:
            raise RegistryError(f"duplicate experiment_id: {experiment_id}")
        seen.add(experiment_id)

        for name in (
            "family",
            "variant",
            "completed_at",
            "status_reason",
            "comparison_group",
            "seed_stability",
            "lesson",
        ):
            _string(raw[name], name)
        _string(raw["baseline_id"], "baseline_id", nullable=True)
        _string(raw["submission_sha256"], "submission_sha256", nullable=True)
        _string_list(raw["folds"], "folds")
        _string_list(raw["artifact_paths"], "artifact_paths")
        _string_list(raw["artifact_sha256"], "artifact_sha256")
        _string_list(raw["evidence_paths"], "evidence_paths")

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

        rows = raw["validation_rows"]
        if rows is not None and (type(rows) is not int or rows < 0):
            raise RegistryError("validation_rows must be a non-negative integer or null")
        _number_or_none(raw["baseline_brier"], "baseline_brier", 0.0, 1.0)
        _number_or_none(raw["candidate_brier"], "candidate_brier", 0.0, 1.0)
        for name in (
            "weighted_gain",
            "worst_fold_gain",
            "latest_fold_gain",
            "max_segment_regression",
        ):
            _number_or_none(raw[name], name)
        _number_or_none(raw["residual_correlation"], "residual_correlation", -1.0, 1.0)
        _number_or_none(raw["public_score"], "public_score")

        declared_hashes = raw["artifact_sha256"]
        assert isinstance(declared_hashes, list)
        if any(_HASH.fullmatch(value) is None for value in declared_hashes):
            raise RegistryError("artifact_sha256 contains an invalid digest")
        submission_hash = raw["submission_sha256"]
        if submission_hash is not None and _HASH.fullmatch(str(submission_hash)) is None:
            raise RegistryError("submission_sha256 is invalid")
        if len(raw["artifact_paths"]) != len(declared_hashes):
            raise RegistryError("artifact path and sha256 counts differ")

        if raw["status"] == "public_scored" and raw["public_score"] is None:
            raise RegistryError("public_score is required for public_scored")
        if (
            raw["status"] == "public_scored"
            and raw["evidence_grade"] in {"A", "B"}
            and submission_hash is None
        ):
            raise RegistryError("submission_sha256 is required for verified public_scored")
        if raw["status"] == "accepted" and raw["rule_audit_status"] == "failed":
            raise RegistryError("accepted record cannot have failed rule_audit_status")

    gap_ids: set[str] = set()
    for gap in gaps:
        if type(gap) is not dict or set(gap) != GAP_FIELDS:
            raise RegistryError("evidence gap member set differs")
        for name in GAP_FIELDS:
            _string(gap[name], f"evidence gap {name}")
        gap_id = str(gap["experiment_id"])
        if gap_id in gap_ids or gap_id in seen:
            raise RegistryError(f"duplicate evidence gap: {gap_id}")
        gap_ids.add(gap_id)


def load_registry(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot load registry: {error}") from error
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


def _sorted_counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def audit_registry(payload: Mapping[str, object]) -> dict[str, object]:
    validate_registry(payload)
    experiments = payload["experiments"]
    gaps = payload["evidence_gaps"]
    assert isinstance(experiments, list)
    assert isinstance(gaps, list)

    groups: dict[str, list[dict[str, object]]] = {}
    for raw in experiments:
        assert isinstance(raw, dict)
        group = str(raw["comparison_group"])
        groups.setdefault(group, []).append({
            "experiment_id": raw["experiment_id"],
            "completed_at": raw["completed_at"],
            "status": raw["status"],
            "baseline_id": raw["baseline_id"],
            "baseline_brier": raw["baseline_brier"],
            "candidate_brier": raw["candidate_brier"],
            "weighted_gain": raw["weighted_gain"],
            "worst_fold_gain": raw["worst_fold_gain"],
            "latest_fold_gain": raw["latest_fold_gain"],
            "residual_correlation": raw["residual_correlation"],
            "evidence_grade": raw["evidence_grade"],
        })
    ordered_groups = {
        name: sorted(rows, key=lambda item: (str(item["completed_at"]), str(item["experiment_id"])))
        for name, rows in sorted(groups.items())
    }

    public_scores = sorted(
        (
            {
                "experiment_id": raw["experiment_id"],
                "completed_at": raw["completed_at"],
                "public_score": raw["public_score"],
                "evidence_grade": raw["evidence_grade"],
                "rule_audit_status": raw["rule_audit_status"],
            }
            for raw in experiments
            if raw["public_score"] is not None
        ),
        key=lambda item: (str(item["completed_at"]), str(item["experiment_id"])),
    )
    failure_values = [
        str(raw["failure_class"])
        for raw in experiments
        if raw["failure_class"] != "none"
    ]
    return {
        "experiment_count": len(experiments),
        "status_counts": _sorted_counts([str(raw["status"]) for raw in experiments]),
        "evidence_grade_counts": _sorted_counts([
            str(raw["evidence_grade"]) for raw in experiments
        ]),
        "failure_classes": _sorted_counts(failure_values),
        "comparison_groups": ordered_groups,
        "public_scores": public_scores,
        "evidence_gap_count": len(gaps),
        "evidence_gaps": sorted(gaps, key=lambda item: str(item["experiment_id"])),
        "closed_families": sorted({
            str(raw["family"]) for raw in experiments if raw["repeat_policy"] == "closed"
        }),
        "redefine_families": sorted({
            str(raw["family"]) for raw in experiments if raw["repeat_policy"] == "redefine"
        }),
        "retained_experiments": sorted({
            str(raw["experiment_id"]) for raw in experiments if raw["repeat_policy"] == "retain"
        }),
    }


def _metric(value: object) -> str:
    if value is None:
        return "—"
    if type(value) is float:
        return f"{value:.12g}"
    return str(value)


def render_audit_markdown(audit: Mapping[str, object]) -> str:
    public_scores = audit["public_scores"]
    comparison_groups = audit["comparison_groups"]
    failure_classes = audit["failure_classes"]
    evidence_gaps = audit["evidence_gaps"]
    assert isinstance(public_scores, list)
    assert isinstance(comparison_groups, dict)
    assert isinstance(failure_classes, dict)
    assert isinstance(evidence_gaps, list)

    lines = [
        "# 실험 증거 재감사",
        "",
        "## 결론",
        "",
        f"확인된 실험은 **{audit['experiment_count']}건**, 원본 결과가 없어 보류한 실행은 "
        f"**{audit['evidence_gap_count']}건**이다. 서로 다른 `comparison_group`의 Brier를 "
        "한 순위로 합치지 않았으며, 실행 실패와 성능 기각도 분리했다.",
        "",
        "현재 최고 규칙 준수 Public 결과는 Tree Expert E2다. E2 이후 후보는 일부 양의 "
        "OOF 신호를 보였지만 최소 개선량, 시즌 안정성, 배포 정렬 또는 오차 다양성 중 "
        "하나 이상을 통과하지 못했다.",
        "",
        "## 실제 Public 제출",
        "",
        "| 실험 | Public | 증거 | 규칙 상태 |",
        "|---|---:|---|---|",
    ]
    if public_scores:
        for item in public_scores:
            lines.append(
                f"| `{item['experiment_id']}` | `{_metric(item['public_score'])}` | "
                f"{item['evidence_grade']} | {item['rule_audit_status']} |"
            )
    else:
        lines.append("| 확인된 제출 없음 | — | — | — |")

    lines.extend([
        "",
        "Public 점수는 OOF Brier와 다른 척도이며, 이 다섯 점으로 점수 환산식이나 "
        "사후 가중치를 맞추지 않는다.",
        "",
        "## 비교 가능한 OOF 그룹",
        "",
        "아래 표는 그룹 내부 결과만 비교하기 위한 것이다. 그룹 사이의 Brier 크기는 "
        "학습 행과 fold가 다를 수 있어 직접 순위를 의미하지 않는다.",
        "",
    ])
    for group, rows in comparison_groups.items():
        lines.extend([
            f"### `comparison_group={group}`",
            "",
            "| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for item in rows:
            lines.append(
                f"| `{item['experiment_id']}` | {item['status']} | "
                f"{_metric(item['candidate_brier'])} | {_metric(item['weighted_gain'])} | "
                f"{_metric(item['worst_fold_gain'])} | {_metric(item['latest_fold_gain'])} | "
                f"{_metric(item['residual_correlation'])} | {item['evidence_grade']} |"
            )
        lines.append("")

    lines.extend([
        "## 반복 기각 원인",
        "",
        "| 원인 | 건수 |",
        "|---|---:|",
    ])
    if failure_classes:
        for name, count in failure_classes.items():
            lines.append(f"| `{name}` | {count} |")
    else:
        lines.append("| 확인된 실패 원인 없음 | 0 |")

    lines.extend([
        "",
        "## 증거가 부족한 실행",
        "",
        "이 항목은 결과를 추정하지 않는다. 원본 review 또는 handoff를 다시 확보한 뒤 "
        "검증 장부에 추가한다.",
        "",
        "| 실험 | 부족한 증거 | 사유 |",
        "|---|---|---|",
    ])
    if evidence_gaps:
        for item in evidence_gaps:
            lines.append(
                f"| `{item['experiment_id']}` | {item['required_evidence']} | {item['reason']} |"
            )
    else:
        lines.append("| 없음 | — | — |")

    closed = ", ".join(f"`{item}`" for item in audit["closed_families"]) or "없음"
    redefine = ", ".join(f"`{item}`" for item in audit["redefine_families"]) or "없음"
    retained = ", ".join(f"`{item}`" for item in audit["retained_experiments"]) or "없음"
    lines.extend([
        "",
        "## 닫을 계열과 다시 정의할 계열",
        "",
        f"- 현재 정의를 반복하지 않을 계열: {closed}",
        f"- 입력이나 구조를 바꿔 다시 정의할 계열: {redefine}",
        f"- 다음 비교의 기준으로 유지할 실험: {retained}",
        "",
        "## 다음 단일 캠페인",
        "",
        "다음은 작은 확률 보정이나 단일 feature 추가가 아니라 **구조적으로 깊은 캠페인**으로 "
        "설계한다. 여기서 깊다는 말은 트리 depth만 크게 만든다는 뜻이 아니다.",
        "",
        "1. 학습 데이터에서 cutoff를 지켜 만든 시즌별 선수·상황 snapshot과 시간 감쇠 anchor",
        "2. 정규 시즌 `R`과 `F`를 나누는 전문가",
        "3. 공식 학습 데이터로 안정적으로 정의되는 상황 전문가",
        "4. 충분한 용량의 CatBoost 다중 seed",
        "5. E2와 실제 오차 다양성이 확인될 때만 추가하는 XGBoost·LightGBM residual",
        "6. 구조 fold에서 강도를 고정한 residual correction과 마지막 계층 calibration",
        "",
        "이 구성은 탐색 깊이를 높이기 위한 것이며 점수를 보장하지 않는다. 먼저 저비용 "
        "적격성 검사로 입력 신호와 2023 fold 병목을 확인하고, 통과한 구조에만 Kaggle "
        "T4 x2 장시간 예산을 배정한다.",
        "",
    ])
    return "\n".join(lines)
