from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
import pytest

from experiments.hierarchical_tabm.calibration import (
    CALIBRATION_EFFECTS,
    CalibrationState,
    calibration_state_payload,
)
from experiments.oof_reset_audit.artifacts import AuditArtifactError, load_artifacts
from experiments.oof_reset_audit.types import ArtifactRole, TrustClass


FOLDS = ("2022->2023", "2023->2024")


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, value in sorted(members.items()):
            info = ZipInfo(name, (2024, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return output.getvalue()


def _write(path: Path, members: dict[str, bytes]) -> Path:
    path.write_bytes(_zip_bytes(members))
    return path


def _frame(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{index}" for index in range(8)],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "probability": [
                0.2 + offset,
                0.8 + offset,
                0.3 + offset,
                0.7 + offset,
                0.4 + offset,
                0.6 + offset,
                0.45 + offset,
                0.55 + offset,
            ],
            "game_type": ["R", "R", "F", "F", "R", "R", "F", "F"],
            "game_month": [4, 4, 5, 5, 6, 6, 7, 7],
            "count_state": ["0_0", "0_0", "1_1", "1_1", "2_1", "2_1", "3_2", "3_2"],
            "hand_matchup": ["1_2"] * 8,
            "base_out_state": ["___0"] * 8,
            "pitcher_id_known": ["known"] * 8,
            "batter_id_known": ["known"] * 8,
        }
    )


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _stage_c_delivery(path: Path) -> Path:
    nested: dict[str, bytes] = {}
    for seed in (42, 2026, 3407):
        for train, valid in ((2022, 2023), (2023, 2024)):
            name = (
                "predictions/c_final__a__p2__piecewise_linear__bce__plateau"
                f"__s42__s{seed}__tr{train}__va{valid}.csv"
            )
            nested[name] = _csv(_frame(seed / 100_000))
    review = _zip_bytes(nested)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tabm_colab_stage_C_delivery",
        "final_stage_complete": True,
        "members": {},
    }
    return _write(
        path,
        {
            "delivery_manifest.json": _json(manifest),
            "tabm_search_stage_C_review_bundle.zip": review,
        },
    )


def _row_feature_delivery(path: Path) -> Path:
    candidate = "rfp__count_context__s42"
    jobs = [
        {
            "candidate_id": candidate,
            "fold": fold,
            "status": "completed",
            "predictions": f"predictions/{candidate}__{fold}.csv",
        }
        for fold in FOLDS
    ]
    nested = {"metrics/job_results.json": _json(jobs)}
    for row in jobs:
        nested[str(row["predictions"])] = _csv(_frame())
    bindings = {
        "input_manifest_sha256": "1" * 64,
        "embedded_runtime_sha256": "2" * 64,
        "campaign_config_sha256": "3" * 64,
        "code_sha256": "4" * 64,
    }
    return _write(
        path,
        {
            "delivery_manifest.json": _json(
                {
                    "schema_version": 1,
                    "artifact_kind": "tabm_row_feature_stage_P_delivery",
                    "bindings": bindings,
                }
            ),
            "tabm_row_feature_stage_P_review_bundle.zip": _zip_bytes(nested),
        },
    )


def _deployment_review(path: Path) -> Path:
    members: dict[str, bytes] = {}
    for train, valid in ((2022, 2023), (2023, 2024)):
        frame = _frame().drop(columns=["probability"])
        for count in (4, 32, 64, 128, 192, 296, 400):
            frame[f"p_{count}"] = _frame(count / 100_000)["probability"]
        members[f"predictions/align_{train}_{valid}.csv"] = _csv(frame)
    members["manifest.json"] = _json(
        {
            "schema_version": 1,
            "artifact_kind": "catboost_deployment_review",
            "bindings": {"contract_sha256": "5" * 64},
        }
    )
    return _write(path, members)


def _blend_delivery(path: Path) -> Path:
    bindings = {"contract_sha256": "7" * 64}
    nested = {
        f"predictions/catboost__hand_matchup__tr{train}__va{valid}__s42.csv": _csv(_frame())
        for train, valid in ((2022, 2023), (2023, 2024))
    }
    return _write(
        path,
        {
            "delivery_manifest.json": _json(
                {
                    "schema_version": 1,
                    "artifact_kind": "catboost_tabm_blend_delivery",
                    "bindings": bindings,
                }
            ),
            "catboost_tabm_blend_review.zip": _zip_bytes(nested),
        },
    )


def _calibration(kind: str) -> bytes:
    effects = MappingProxyType({})
    if kind == "H3":
        effects = MappingProxyType(
            {
                column: MappingProxyType({
                    "game_type": {"F": -0.01, "R": 0.01},
                    "count_state": {"0_0": 0.0, "1_1": 0.0, "2_1": 0.0, "3_2": 0.0},
                    "hand_matchup": {"1_2": 0.0},
                    "base_out_state": {"___0": 0.0},
                }[column])
                for column in CALIBRATION_EFFECTS
            }
        )
    return _json(
        calibration_state_payload(
            CalibrationState(1, kind, 0.01, 1e-6, 0.0, 0.95, effects, "a" * 64)
        )
    )


def _hierarchical_review(path: Path) -> Path:
    members = {
        "manifest.json": _json(
            {
                "schema_version": 1,
                "artifact_kind": "hierarchical_tabm_review_v1",
                "bindings": {"contract_sha256": "6" * 64},
            }
        ),
        "calibration/H2.json": _calibration("H2"),
        "calibration/H3.json": _calibration("H3"),
        "jobs/h1__tr2022__va2023__s3407/predictions.csv": _csv(_frame()),
        "jobs/h1__tr2023__va2024__s3407/predictions.csv": _csv(_frame()),
    }
    return _write(path, members)


def _quarantined(path: Path, *, probability: float = 0.4) -> Path:
    member = "predictions/xgb__tr2023__va2024.csv"
    value = _csv(_frame().assign(probability=probability))
    manifest = {
        "schema_version": 1,
        "artifact_kind": "oof_reset_quarantined_v1",
        "trust": "quarantined_diagnostic",
        "submission_package": False,
        "predictions": {
            member: {
                "model_id": "xgboost_v3_raw_oof",
                "fold": "2023->2024",
                "size": len(value),
                "sha256": sha256(value).hexdigest(),
            }
        },
    }
    return _write(path, {"manifest.json": _json(manifest), member: value})


def test_stage_c_and_row_feature_adapters_decode_verified_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "experiments.oof_reset_audit.artifacts.verify_stage_c_delivery", lambda path: None
    )
    monkeypatch.setattr(
        "experiments.oof_reset_audit.artifacts.verify_row_feature_delivery",
        lambda path, **bindings: None,
    )
    loaded = load_artifacts(
        [_stage_c_delivery(tmp_path / "stage_c.zip"), _row_feature_delivery(tmp_path / "row.zip")],
        expected_roles=(ArtifactRole.STAGE_C_TABM, ArtifactRole.ROW_FEATURE),
    )
    assert {(row.model_id, row.fold) for row in loaded.predictions} == {
        *((f"tabm_stage_c_seed_{seed}", fold) for seed in (42, 2026, 3407) for fold in FOLDS),
        *(("rfp__count_context__s42", fold) for fold in FOLDS),
    }
    assert {row.trust for row in loaded.predictions} == {TrustClass.RULE_SAFE}
    assert all("pitcher_known" in row.frame for row in loaded.predictions)


def test_deployment_and_hierarchical_adapters_expand_derived_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "experiments.oof_reset_audit.artifacts.verify_deployment_review",
        lambda path, expected_bindings: None,
    )
    monkeypatch.setattr(
        "experiments.oof_reset_audit.artifacts.verify_hierarchical_review",
        lambda path, expected_bindings: None,
    )
    loaded = load_artifacts(
        [_deployment_review(tmp_path / "deployment.zip"), _hierarchical_review(tmp_path / "hier.zip")],
        expected_roles=(ArtifactRole.CATBOOST_DEPLOYMENT, ArtifactRole.HIERARCHICAL),
    )
    ids = {row.model_id for row in loaded.predictions}
    assert {f"catboost_prefix_{count}" for count in (4, 32, 64, 128, 192, 296, 400)} <= ids
    assert {"H1", "H2", "H3"} <= ids
    assert len(loaded.predictions) == 20


def test_blend_adapter_loads_two_rule_safe_catboost_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "experiments.oof_reset_audit.artifacts.verify_blend_delivery",
        lambda path, expected_bindings: None,
    )
    loaded = load_artifacts(
        [_blend_delivery(tmp_path / "blend.zip")],
        expected_roles=(ArtifactRole.CATBOOST_BLEND,),
    )
    assert {(row.model_id, row.fold) for row in loaded.predictions} == {
        ("catboost_hand_matchup_seed_42", fold) for fold in FOLDS
    }
    assert all(row.trust is TrustClass.RULE_SAFE for row in loaded.predictions)


def test_quarantined_oof_is_hash_verified_and_cannot_become_rule_safe(
    tmp_path: Path,
) -> None:
    source = _quarantined(tmp_path / "xgb.zip")
    loaded = load_artifacts(
        [source], expected_roles=(ArtifactRole.QUARANTINED_XGBOOST,)
    )
    assert {row.trust for row in loaded.predictions} == {
        TrustClass.QUARANTINED_DIAGNOSTIC
    }

    with ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["predictions/xgb__tr2023__va2024.csv"] += b"\n"
    _write(source, members)
    with pytest.raises(AuditArtifactError, match="hash"):
        load_artifacts([source], expected_roles=(ArtifactRole.QUARANTINED_XGBOOST,))


def test_unknown_submission_and_absent_roles_are_missing_evidence(tmp_path: Path) -> None:
    submission = _write(
        tmp_path / "submit.zip", {"script.py": b"print('submission')", "model/m.bin": b"x"}
    )
    loaded = load_artifacts(
        [submission],
        expected_roles=(ArtifactRole.STAGE_C_TABM, ArtifactRole.ROW_FEATURE),
    )
    assert loaded.predictions == ()
    assert {row.status for row in loaded.inventory} == {"missing_evidence"}
    assert {row.role for row in loaded.inventory if row.path.name != "submit.zip"} == {
        ArtifactRole.STAGE_C_TABM,
        ArtifactRole.ROW_FEATURE,
    }


def test_conflicting_duplicate_model_fold_is_rejected(
    tmp_path: Path,
) -> None:
    first = _quarantined(tmp_path / "first.zip", probability=0.4)
    second = _quarantined(tmp_path / "second.zip", probability=0.6)
    with pytest.raises(AuditArtifactError, match="conflicting prediction evidence"):
        load_artifacts(
            [first, second], expected_roles=(ArtifactRole.QUARANTINED_XGBOOST,)
        )
