from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tree_expert.e2_artifacts import file_sha256, write_e2_bundles
from experiments.tree_expert.e2_production import (
    E2ProductionError,
    restore_resume_for_current_runtime,
)


def _bindings(code: str) -> dict[str, str]:
    return {
        "contract_sha256": "1" * 64,
        "code_sha256": code,
        "input_sha256": "3" * 64,
        "train_sha256": "4" * 64,
        "history_sha256": "5" * 64,
    }


def _failed_audit_resume(tmp_path: Path, *, audit_error: str) -> tuple[Path, str]:
    old_code = "9" * 64
    source = tmp_path / "source"
    source.mkdir()
    acceptance = source / "decisions/acceptance.json"
    acceptance.parent.mkdir(parents=True)
    acceptance.write_text(json.dumps({"status": "accepted"}), encoding="utf-8")
    token = source / "decisions/accepted_token.json"
    token.write_text(
        json.dumps(
            {
                "candidate_id": "c1_anchor_residual",
                "predictor": "catboost",
                "seeds": [42, 2026, 3407],
                "iterations": {"42": 50, "2026": 51, "3407": 52},
                "decision_sha256": file_sha256(acceptance),
            }
        ),
        encoding="utf-8",
    )
    models: dict[str, Path] = {}
    model_hashes: dict[str, str] = {}
    for seed in (42, 2026, 3407):
        path = source / f"full_fit/models/catboost_seed_{seed}.cbm"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"model-{seed}".encode())
        models[f"full_fit/models/catboost_seed_{seed}.cbm"] = path
        model_hashes[str(seed)] = file_sha256(path)
    full_fit = source / "full_fit/full_fit_manifest.json"
    full_fit.write_text(
        json.dumps(
            {
                "candidate_id": "c1_anchor_residual",
                "predictor": "catboost",
                "model_sha256": model_hashes,
            }
        ),
        encoding="utf-8",
    )
    frozen = source / "full_fit/frozen_state/feature_state.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}", encoding="utf-8")
    log = source / "tree_expert_e2.log"
    log.write_text("TREE_E2_PHASE_START phase=AUDIT\n", encoding="utf-8")
    state = source / "stage_state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "tree_expert_e2_v1",
                "status": "failed",
                "phase": "TERMINAL",
                "bindings": _bindings(old_code),
                "completed": ["full_fit_s42", "full_fit_s2026", "full_fit_s3407"],
                "skipped": [],
                "failed": [],
                "active": {},
                "decisions": {
                    "ACCEPTANCE": {"status": "accepted", "predictor": "catboost"},
                    "FULL_FIT": {"candidate_id": "c1_anchor_residual"},
                    "AUDIT": {"status": "failed", "error": audit_error},
                },
                "artifact_paths": {
                    "acceptance_decision": "decisions/acceptance.json",
                    "full_fit": "full_fit/full_fit_manifest.json",
                },
            }
        ),
        encoding="utf-8",
    )
    resume_sources = {
        "decisions/acceptance.json": acceptance,
        "decisions/accepted_token.json": token,
        "full_fit/full_fit_manifest.json": full_fit,
        "full_fit/frozen_state/feature_state.json": frozen,
        **models,
    }
    bundles = write_e2_bundles(
        output_dir=tmp_path / "bundles",
        bindings=_bindings(old_code),
        state_path=state,
        log_path=log,
        review_sources={"decisions/acceptance.json": acceptance},
        resume_sources=resume_sources,
    )
    return bundles.resume, old_code


def test_known_audit_failure_migrates_to_audit_only_resume(tmp_path: Path) -> None:
    resume, old_code = _failed_audit_resume(
        tmp_path,
        audit_error="TreeFeatureError: S1 transform season differs from valid_year",
    )
    current = _bindings("2" * 64)
    state_path = restore_resume_for_current_runtime(
        resume,
        tmp_path / "restored",
        current,
        legacy_audit_code_sha256=old_code,
        legacy_audit_resume_sha256=file_sha256(resume),
    )
    state = json.loads(state_path.read_text())
    assert state["status"] == "running"
    assert state["phase"] == "AUDIT"
    assert state["bindings"] == current
    assert "AUDIT" not in state["decisions"]
    assert (tmp_path / "restored/full_fit/models/catboost_seed_3407.cbm").is_file()


def test_unrelated_terminal_failure_cannot_be_migrated(tmp_path: Path) -> None:
    resume, old_code = _failed_audit_resume(
        tmp_path,
        audit_error="RuntimeError: unrelated failure",
    )
    with pytest.raises(E2ProductionError, match="not the registered audit failure"):
        restore_resume_for_current_runtime(
            resume,
            tmp_path / "restored",
            _bindings("2" * 64),
            legacy_audit_code_sha256=old_code,
            legacy_audit_resume_sha256=file_sha256(resume),
        )


def test_repacked_legacy_resume_cannot_be_migrated(tmp_path: Path) -> None:
    resume, old_code = _failed_audit_resume(
        tmp_path,
        audit_error="TreeFeatureError: S1 transform season differs from valid_year",
    )
    with pytest.raises(E2ProductionError, match="resume SHA-256 differs"):
        restore_resume_for_current_runtime(
            resume,
            tmp_path / "restored",
            _bindings("2" * 64),
            legacy_audit_code_sha256=old_code,
            legacy_audit_resume_sha256="7" * 64,
        )


def test_current_binding_resume_is_restored_without_migration(tmp_path: Path) -> None:
    resume, code = _failed_audit_resume(
        tmp_path,
        audit_error="TreeFeatureError: S1 transform season differs from valid_year",
    )
    state_path = restore_resume_for_current_runtime(
        resume,
        tmp_path / "restored",
        _bindings(code),
        legacy_audit_code_sha256="8" * 64,
    )

    state = json.loads(state_path.read_text())
    assert state["status"] == "failed"
    assert state["phase"] == "TERMINAL"
    assert state["decisions"]["AUDIT"]["status"] == "failed"
