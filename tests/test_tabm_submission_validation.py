from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from competition_rules.evidence_gate import (
    AuditIdentity,
    RulesEvidenceError,
    run_phased_independence_audit,
    validate_full_audit,
)
from submission.tabm_validation import (
    TabMValidationError,
    build_validation_bundles,
    build_capacity_frames,
    compare_probe_predictions,
    validate_exact_probe,
    validate_report,
)


def _identity() -> AuditIdentity:
    return AuditIdentity(
        policy_version="dacon-236743-2026-08-15",
        candidate_id="fixture",
        data_sha256="1" * 64,
        code_sha256="2" * 64,
        config_sha256="3" * 64,
        preprocessing_sha256="4" * 64,
        model_sha256="5" * 64,
        adapter_sha256="6" * 64,
        runtime_sha256="7" * 64,
    )


def _frame(count: int = 19) -> pd.DataFrame:
    return pd.DataFrame(
        {"row_id": [f"r-{index:03d}" for index in range(count)], "x": np.arange(count)}
    )


class _Predictor:
    def __init__(self, *, batch_dependent: bool = False) -> None:
        self.batch_dependent = batch_dependent

    def state_digest(self) -> str:
        return "a" * 64

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame[["x"]].to_numpy(dtype="float32")
        return x, np.zeros((len(frame), 1), dtype="int64")

    def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 2048) -> np.ndarray:
        del batch_size
        if self.batch_dependent:
            return np.full(len(frame), len(frame) / 100.0)
        return frame["x"].to_numpy(dtype="float64") / 100.0


def test_phased_audit_resumes_verified_phases(tmp_path: Path) -> None:
    first = run_phased_independence_audit(
        frame=_frame(5),
        output_dir=tmp_path / "audit",
        identity=_identity(),
        load_predictor=_Predictor,
        singleton_count=5,
        stop_after_phases=2,
    )
    assert first.status == "incomplete"

    second = run_phased_independence_audit(
        frame=_frame(5),
        output_dir=tmp_path / "audit",
        identity=_identity(),
        load_predictor=_Predictor,
        singleton_count=5,
    )

    assert second.status == "passed"
    assert second.reused_phases == 2
    manifest = validate_full_audit(
        second.manifest_path, expected_identity=_identity()
    )
    assert manifest["schema_version"] == 2
    assert manifest["audit_scope"] == "official_sample_plus_synthetic_scale"


def test_phased_audit_rejects_tampering_and_batch_dependence(tmp_path: Path) -> None:
    with pytest.raises(RulesEvidenceError, match="row independence mismatch"):
        run_phased_independence_audit(
            frame=_frame(5),
            output_dir=tmp_path / "dependent",
            identity=_identity(),
            load_predictor=lambda: _Predictor(batch_dependent=True),
            singleton_count=5,
        )

    result = run_phased_independence_audit(
        frame=_frame(5),
        output_dir=tmp_path / "tampered",
        identity=_identity(),
        load_predictor=_Predictor,
        singleton_count=5,
    )
    payload = json.loads(result.manifest_path.read_text())
    phase = result.manifest_path.parent / payload["phases"][0]["path"]
    phase.write_text("{}", encoding="utf-8")
    with pytest.raises(RulesEvidenceError, match="phase"):
        validate_full_audit(result.manifest_path, expected_identity=_identity())


def _probe(**changes: object) -> dict[str, object]:
    probe: dict[str, object] = {
        "status": "passed",
        "python": "3.11.15",
        "torch": "2.7.1+cpu",
        "pandas": "2.0.3",
        "numpy": "1.26.4",
        "tabm": "0.0.3",
        "rtdl_num_embeddings": "0.0.12",
        "probabilities": [0.4, 0.5],
    }
    probe.update(changes)
    return probe


def test_exact_probe_requires_official_versions() -> None:
    validate_exact_probe(_probe())
    with pytest.raises(TabMValidationError, match="Python 3.11.15"):
        validate_exact_probe(_probe(python="3.12.13"))
    with pytest.raises(TabMValidationError, match="PyTorch 2.7.1"):
        validate_exact_probe(_probe(torch="2.11.0+cpu"))


def test_exact_and_gpu_predictions_must_agree() -> None:
    compare_probe_predictions(_probe(), _probe(torch="2.11.0+cu128"))
    with pytest.raises(TabMValidationError, match="probe predictions"):
        compare_probe_predictions(
            _probe(), _probe(torch="2.11.0+cu128", probabilities=[0.4, 0.5001])
        )


def test_capacity_frame_is_deterministic_and_row_local() -> None:
    test = pd.DataFrame({"row_id": ["a", "b"], "x": [10, 20]})
    sample = pd.DataFrame({"row_id": ["b", "a"], "control_success": [0.5, 0.5]})

    scale_test, scale_sample = build_capacity_frames(test, sample, row_count=7)

    assert scale_test["row_id"].tolist() == [f"SCALE_{index:06d}" for index in range(7)]
    assert scale_test["x"].tolist() == [10, 20, 10, 20, 10, 20, 10]
    assert scale_sample["row_id"].tolist() == scale_test["row_id"].tolist()
    assert scale_sample["control_success"].eq(0.5).all()


def test_validation_report_rejects_wrong_scope_or_limits() -> None:
    report = {
        "schema_version": 1,
        "status": "validation_passed",
        "audit_scope": "official_sample_plus_synthetic_scale",
        "identity": asdict(_identity()),
        "install_seconds": 2.0,
        "inference_seconds": 30.0,
        "peak_ram_bytes": 4 * 1024**3,
        "peak_vram_bytes": 2 * 1024**3,
        "package_bytes": 11_000_000,
        "extracted_bytes": 11_000_000,
    }
    validate_report(report)

    wrong = dict(report, audit_scope="hidden_test")
    with pytest.raises(TabMValidationError, match="audit scope"):
        validate_report(wrong)
    slow = dict(report, inference_seconds=481.0)
    with pytest.raises(TabMValidationError, match="runtime safety"):
        validate_report(slow)


def test_validation_bundles_are_deterministic_and_exclude_model(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    audit = evidence / "full_audit"
    audit.mkdir(parents=True)
    (audit / "full_audit_manifest.json").write_text(
        json.dumps({"schema_version": 2}), encoding="utf-8"
    )
    (audit / "phase-baseline.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    (evidence / "environment.json").write_text("{}", encoding="utf-8")
    (evidence / "install.log").write_text("installed\n", encoding="utf-8")
    (evidence / "validation.log").write_text("validated\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "status": "validation_passed",
        "audit_scope": "official_sample_plus_synthetic_scale",
        "identity": asdict(_identity()),
        "install_seconds": 2.0,
        "inference_seconds": 30.0,
        "peak_ram_bytes": 4 * 1024**3,
        "peak_vram_bytes": 2 * 1024**3,
        "package_bytes": 11_000_000,
        "extracted_bytes": 11_000_000,
    }

    first = build_validation_bundles(evidence, report, tmp_path / "one")
    second = build_validation_bundles(evidence, report, tmp_path / "two")

    assert first.review_path.read_bytes() == second.review_path.read_bytes()
    assert first.resume_path.read_bytes() == second.resume_path.read_bytes()
    import zipfile

    with zipfile.ZipFile(first.review_path) as archive:
        names = archive.namelist()
    assert "validation_report.json" in names
    assert "manifest.json" in names
    assert not any(name.startswith("model/") or name.endswith(".pt") for name in names)
