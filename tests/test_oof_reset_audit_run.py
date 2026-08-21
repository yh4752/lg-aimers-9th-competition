from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pandas as pd

from experiments.oof_reset_audit.run import _diagnose, run_audit
from experiments.oof_reset_audit.types import ArtifactRecord, ArtifactRole, PredictionSet, TrustClass


FOLDS = ("2022->2023", "2023->2024")
REPORTS = {
    "artifact_inventory.json", "audit_summary.md", "calibration_deciles.csv",
    "correlation_matrix.csv", "model_comparison.csv", "next_experiment.json",
    "paired_comparison.csv", "segment_diagnostics.csv",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "tools/run_oof_reset_audit.py"


def _prediction(model: str, fold: str, shift: float, trust: TrustClass = TrustClass.RULE_SAFE) -> PredictionSet:
    target = [0, 1] * 6
    base = [0.25 if value == 0 else 0.75 for value in target]
    probability = [value - shift if truth == 0 else value + shift for value, truth in zip(base, target)]
    frame = pd.DataFrame({
        "row_id": [f"{fold}-r{i}" for i in range(12)],
        "target": target,
        "probability": probability,
        "game_month": [4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
        "game_type": ["R"] * 6 + ["F"] * 6,
        "count_state": ["0_0"] * 12,
        "hand_matchup": ["1_2"] * 12,
        "base_out_state": ["___0"] * 12,
        "pitcher_known": ["known"] * 12,
        "batter_known": ["known"] * 12,
    })
    return PredictionSet(Path(f"{model}.zip"), "a" * 64, f"{model}/{fold}.csv", "b" * 64,
                         model, fold, trust, frame)


def _inventory() -> tuple[ArtifactRecord, ...]:
    return (ArtifactRecord(Path("stage.zip"), "a" * 64, ArtifactRole.STAGE_C_TABM,
                           "tabm_colab_stage_C_delivery", "verified", None),)


def _with_eligible_segments(value: PredictionSet) -> PredictionSet:
    frame = pd.concat([value.frame] * 500, ignore_index=True)
    frame["row_id"] = [f"{value.fold}-expanded-{index}" for index in range(len(frame))]
    return replace(value, frame=frame)


def test_audit_uses_fixed_anchor_and_never_makes_quarantine_eligible(tmp_path: Path) -> None:
    values = []
    for fold in FOLDS:
        values.extend((
            _prediction("tabm_stage_c_seed_3407", fold, 0.0),
            _prediction("safe_model", fold, 0.05),
            _prediction("old_xgb", fold, 0.08, TrustClass.QUARANTINED_DIAGNOSTIC),
        ))
    result = run_audit(predictions=values, inventory=_inventory(), output_root=tmp_path)
    paired = pd.read_csv(result.output_dir / "paired_comparison.csv")
    assert set(paired["anchor_model_id"]) == {"tabm_stage_c_seed_3407"}
    decision = json.loads((result.output_dir / "next_experiment.json").read_text())
    assert decision["automatic_acceptance"] is False
    assert "old_xgb" not in decision["eligible_model_ids"]
    assert decision["eligible_model_ids"] == ["safe_model"]
    assert decision["quarantined_model_ids"] == ["old_xgb"]


def test_unpaired_and_missing_evidence_do_not_fabricate_delta(tmp_path: Path) -> None:
    inventory = _inventory() + (
        ArtifactRecord(Path("__MISSING__/row_feature"), "0" * 64, ArtifactRole.ROW_FEATURE,
                       "missing", "missing_evidence", None),
    )
    result = run_audit(
        predictions=[_prediction("tabm_stage_c_seed_3407", FOLDS[0], 0.0),
                     _prediction("safe_model", FOLDS[1], 0.05)],
        inventory=inventory,
        output_root=tmp_path,
    )
    model = pd.read_csv(result.output_dir / "model_comparison.csv")
    assert set(model["comparison_class"]) == {"descriptive_only"}
    paired = pd.read_csv(result.output_dir / "paired_comparison.csv")
    assert paired.empty
    decision = json.loads((result.output_dir / "next_experiment.json").read_text())
    assert decision["missing_evidence"] == ["row_feature"]


def test_reports_have_exact_allowlist_and_zip_is_small_review_only(tmp_path: Path) -> None:
    predictions = [_prediction("tabm_stage_c_seed_3407", fold, 0.0) for fold in FOLDS]
    first = run_audit(predictions=predictions, inventory=_inventory(), output_root=tmp_path)
    second = run_audit(predictions=predictions, inventory=_inventory(), output_root=tmp_path)
    assert first.output_dir != second.output_dir
    assert set(first.output_files) == REPORTS | {"oof_reset_audit_results.zip"}
    with ZipFile(first.result_zip) as archive:
        assert set(archive.namelist()) == REPORTS
        assert all(not name.endswith((".pt", ".cbm", ".pkl", ".npy")) for name in archive.namelist())


def test_stable_safe_fixed_blend_supports_diverse_blend_but_never_accepts_it(tmp_path: Path) -> None:
    values = []
    for fold in FOLDS:
        values.extend((
            _with_eligible_segments(_prediction("tabm_stage_c_seed_3407", fold, 0.0)),
            _with_eligible_segments(_prediction("safe_model", fold, 0.05)),
        ))
    result = run_audit(predictions=values, inventory=_inventory(), output_root=tmp_path)
    paired = pd.read_csv(result.output_dir / "paired_comparison.csv")
    blend = paired[(paired["comparison"] == "fixed_blend") & (paired["candidate_weight"] == 0.25)]
    assert set(blend["interval_status"]) == {"completed"}
    assert bool((blend["interval_lower"] > 0).all())
    segments = pd.read_csv(result.output_dir / "segment_diagnostics.csv")
    assert {"comparison", "candidate_weight"} <= set(segments)
    assert "fixed_blend" in set(segments["comparison"])
    decision = json.loads((result.output_dir / "next_experiment.json").read_text())
    assert decision["automatic_acceptance"] is False
    assert "DIVERSE_BLEND" in decision["supported_directions"]
    assert any(
        row["candidate_model_id"] == "safe_model" and row["candidate_weight"] == 0.25
        for row in decision["evidence"]["diverse_blend"]["stable_blends"]
    )
    assert set(decision["evidence"]) == {"recency_weighting", "diverse_blend", "stop_and_reframe"}


def test_diverse_blend_requires_nonnegative_latest_block_interval() -> None:
    predictions = tuple(
        item
        for fold in FOLDS
        for item in (
            _prediction("tabm_stage_c_seed_3407", fold, 0.0),
            _prediction("safe_model", fold, 0.05),
        )
    )
    model = pd.DataFrame([
        {"model_id": item.model_id, "fold": item.fold, "trust": item.trust.value,
         "comparison_class": "descriptive_only" if "3407" in item.model_id else "paired",
         "brier": 0.2, "prediction_mean": 0.5, "target_mean": 0.5}
        for item in predictions
    ])
    paired = pd.DataFrame([
        {"candidate_model_id": "safe_model", "fold": fold, "comparison": "fixed_blend",
         "candidate_weight": 0.25, "gain_vs_anchor": 0.001,
         "interval_status": "completed", "interval_lower": -0.00001 if fold == FOLDS[1] else 0.0001}
        for fold in FOLDS
    ])
    segments = pd.DataFrame([
        {"candidate_model_id": "safe_model", "fold": fold, "comparison": "fixed_blend",
         "candidate_weight": 0.25, "eligible": True, "regression": 0.0001}
        for fold in FOLDS
    ])
    correlations = pd.DataFrame(columns=["trust", "residual_correlation"])

    decision = _diagnose(predictions, _inventory(), model, paired, segments, correlations)

    assert decision["evidence"]["diverse_blend"]["stable_blends"] == []
    assert decision["supported_directions"] == ["STOP_AND_REFRAME"]

    paired.loc[paired["fold"] == FOLDS[1], "interval_lower"] = 0.00001
    segments.loc[segments["fold"] == FOLDS[1], "regression"] = 0.0008
    segment_rejected = _diagnose(
        predictions, _inventory(), model, paired, segments, correlations
    )
    assert segment_rejected["evidence"]["diverse_blend"]["stable_blends"] == []

    segments["regression"] = 0.0007
    accepted_for_manual_review = _diagnose(
        predictions, _inventory(), model, paired, segments, correlations
    )
    assert accepted_for_manual_review["supported_directions"] == ["DIVERSE_BLEND"]
    assert accepted_for_manual_review["automatic_acceptance"] is False


def test_fixed_blend_rows_report_blended_correlations(tmp_path: Path) -> None:
    anchor = _prediction("tabm_stage_c_seed_3407", FOLDS[1], 0.0)
    candidate = _prediction("safe_model", FOLDS[1], 0.0)
    candidate.frame["probability"] += pd.Series(
        [-0.10, 0.03, 0.08, -0.02, -0.04, 0.09, 0.02, -0.08, 0.06, -0.01, -0.07, 0.04]
    )
    result = run_audit(
        predictions=[anchor, candidate], inventory=_inventory(), output_root=tmp_path
    )
    paired = pd.read_csv(result.output_dir / "paired_comparison.csv")
    raw = paired[paired["comparison"] == "candidate"].iloc[0]
    blend = paired[
        (paired["comparison"] == "fixed_blend")
        & (paired["candidate_weight"] == 0.25)
    ].iloc[0]

    expected_probability = 0.75 * anchor.frame["probability"] + 0.25 * candidate.frame["probability"]
    expected = float(anchor.frame["probability"].corr(expected_probability))
    assert math.isclose(blend["prediction_correlation"], expected, abs_tol=1e-12)
    assert not math.isclose(
        blend["prediction_correlation"], raw["prediction_correlation"], abs_tol=1e-8
    )


def test_shared_fold_drift_with_high_residual_correlation_supports_recency(tmp_path: Path) -> None:
    values = [
        _prediction("tabm_stage_c_seed_3407", FOLDS[0], 0.05),
        _prediction("tabm_stage_c_seed_3407", FOLDS[1], 0.0),
        _prediction("safe_model", FOLDS[0], 0.07),
        _prediction("safe_model", FOLDS[1], 0.02),
    ]
    for item in values:
        if item.fold == FOLDS[0]:
            item.frame["probability"] += 0.02
    result = run_audit(predictions=values, inventory=_inventory(), output_root=tmp_path)
    decision = json.loads((result.output_dir / "next_experiment.json").read_text())
    assert "RECENCY_WEIGHTING" in decision["supported_directions"]
    assert decision["evidence"]["recency_weighting"]["common_calibration_shift"] is True
    assert decision["manual_review_required"] is True


def test_no_repeatable_safe_improvement_supports_stop_and_reframe(tmp_path: Path) -> None:
    values = []
    for fold in FOLDS:
        values.extend((_prediction("tabm_stage_c_seed_3407", fold, 0.05),
                       _prediction("safe_model", fold, -0.05)))
    result = run_audit(predictions=values, inventory=_inventory(), output_root=tmp_path)
    decision = json.loads((result.output_dir / "next_experiment.json").read_text())
    assert decision["supported_directions"] == ["STOP_AND_REFRAME"]


def test_cli_help_works_outside_project_root(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI), "--help"], cwd=tmp_path, text=True,
        capture_output=True, check=False,
    )
    assert completed.returncode == 0
    assert "--artifact" in completed.stdout


def test_cli_success_and_error_markers_are_exact(tmp_path: Path) -> None:
    from test_oof_reset_audit_artifacts import _quarantined

    source = _quarantined(tmp_path / "xgb.zip")
    completed = subprocess.run(
        [sys.executable, str(CLI), "--artifact", str(source), "--output-root", str(tmp_path / "runs")],
        cwd=tmp_path, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.startswith("OOF_RESET_AUDIT_INPUTS_VERIFIED")
    assert "OOF_RESET_AUDIT_SUCCESS output_dir=" in completed.stdout

    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    failed = subprocess.run(
        [sys.executable, str(CLI), "--artifact", str(broken), "--output-root", str(tmp_path / "failed")],
        cwd=tmp_path, text=True, capture_output=True, check=False,
    )
    assert failed.returncode == 1
    assert failed.stdout.startswith("OOF_RESET_AUDIT_ERROR stage=inputs type=AuditArtifactError message=")
    assert not list((tmp_path / "failed").rglob("*.zip")) if (tmp_path / "failed").exists() else True
