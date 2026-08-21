from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from experiments.oof_reset_audit.run import run_audit
from experiments.oof_reset_audit.types import ArtifactRecord, ArtifactRole, PredictionSet, TrustClass


FOLDS = ("2022->2023", "2023->2024")
REPORTS = {
    "artifact_inventory.json", "audit_summary.md", "calibration_deciles.csv",
    "correlation_matrix.csv", "model_comparison.csv", "next_experiment.json",
    "paired_comparison.csv", "segment_diagnostics.csv",
}


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
        values.extend((_prediction("tabm_stage_c_seed_3407", fold, 0.0),
                       _prediction("safe_model", fold, 0.05)))
    result = run_audit(predictions=values, inventory=_inventory(), output_root=tmp_path)
    decision = json.loads((result.output_dir / "next_experiment.json").read_text())
    assert decision["automatic_acceptance"] is False
    assert "DIVERSE_BLEND" in decision["supported_directions"]
    assert set(decision["evidence"]) == {"recency_weighting", "diverse_blend", "stop_and_reframe"}


def test_shared_fold_drift_with_high_residual_correlation_supports_recency(tmp_path: Path) -> None:
    values = [
        _prediction("tabm_stage_c_seed_3407", FOLDS[0], 0.05),
        _prediction("tabm_stage_c_seed_3407", FOLDS[1], 0.0),
        _prediction("safe_model", FOLDS[0], 0.07),
        _prediction("safe_model", FOLDS[1], 0.02),
    ]
    result = run_audit(predictions=values, inventory=_inventory(), output_root=tmp_path)
    decision = json.loads((result.output_dir / "next_experiment.json").read_text())
    assert "RECENCY_WEIGHTING" in decision["supported_directions"]
    assert decision["manual_review_required"] is True


def test_no_repeatable_safe_improvement_supports_stop_and_reframe(tmp_path: Path) -> None:
    values = []
    for fold in FOLDS:
        values.extend((_prediction("tabm_stage_c_seed_3407", fold, 0.05),
                       _prediction("safe_model", fold, -0.05)))
    result = run_audit(predictions=values, inventory=_inventory(), output_root=tmp_path)
    decision = json.loads((result.output_dir / "next_experiment.json").read_text())
    assert decision["supported_directions"] == ["STOP_AND_REFRAME"]
