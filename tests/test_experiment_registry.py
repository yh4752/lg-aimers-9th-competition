from __future__ import annotations

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
        status="public_scored",
        public_score=828.99,
        evidence_grade="C",
        status_reason="legacy public score; submission hash was not retained",
    )))


def test_accepted_candidate_cannot_fail_rule_audit() -> None:
    with pytest.raises(RegistryError, match="rule_audit_status"):
        validate_registry(registry(record(status="accepted", rule_audit_status="failed")))


def test_load_registry_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(RegistryError, match="root"):
        load_registry(path)


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
