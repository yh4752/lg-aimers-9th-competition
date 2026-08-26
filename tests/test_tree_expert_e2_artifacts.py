from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.tree_expert.e2_artifacts import (
    E2ArtifactError,
    E2DeliverySources,
    file_sha256,
    verify_e2_handoff,
    verify_e2_resume,
    write_e2_bundles,
    write_e2_delivery,
)
from experiments.tree_expert.e2_full_fit import AcceptedForFullFit


def _bindings() -> dict[str, str]:
    return {
        "contract_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "input_sha256": "3" * 64,
        "train_sha256": "4" * 64,
        "history_sha256": "5" * 64,
    }


def _token(predictor: str = "catboost", decision_sha: str = "a" * 64) -> AcceptedForFullFit:
    return AcceptedForFullFit(
        candidate_id="c1_anchor_residual",
        predictor=predictor,
        seeds=(42, 2026, 3407),
        iterations=MappingProxyType({42: 50, 2026: 51, 3407: 52}),
        decision_sha256=decision_sha,
    )


def _campaign(tmp_path: Path, status: str) -> dict[str, object]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    state = source / "stage_state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "tree_expert_e2_v1",
                "status": status,
                "phase": "TERMINAL",
                "bindings": _bindings(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    log = source / "tree_expert_e2.log"
    log.write_text("TREE_E2_TERMINAL\n", encoding="utf-8")
    decision = source / "acceptance_decision.json"
    decision.write_text(json.dumps({"status": status}, sort_keys=True), encoding="utf-8")
    checkpoint = source / "checkpoint.cbm"
    checkpoint.write_bytes(b"checkpoint")
    return {
        "output_dir": tmp_path / "bundles",
        "bindings": _bindings(),
        "state_path": state,
        "log_path": log,
        "review_sources": {"decisions/acceptance.json": decision},
        "resume_sources": {"jobs/example/checkpoint.cbm": checkpoint},
    }


def _delivery_sources(tmp_path: Path, token: AcceptedForFullFit) -> E2DeliverySources:
    root = tmp_path / "delivery_sources"
    root.mkdir(parents=True)
    models: dict[int, Path] = {}
    model_hashes: dict[str, str] = {}
    for seed in token.seeds:
        path = root / f"catboost_seed_{seed}.cbm"
        path.write_bytes(f"model-{seed}".encode())
        models[seed] = path
        model_hashes[str(seed)] = file_sha256(path)
    state = root / "frozen_state"
    state.mkdir()
    (state / "feature_state.json").write_text('{"state":1}', encoding="utf-8")
    acceptance = root / "acceptance_decision.json"
    acceptance.write_bytes(b"accepted-decision")
    token = _token(token.predictor, file_sha256(acceptance))
    full_fit = root / "full_fit_manifest.json"
    full_fit.write_text(
        json.dumps(
            {
                "candidate_id": token.candidate_id,
                "predictor": token.predictor,
                "model_sha256": model_hashes,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    audit = root / "inference_audit.json"
    audit.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    return E2DeliverySources(
        token=token,
        models=MappingProxyType(models),
        frozen_state=state,
        full_fit_manifest=full_fit,
        acceptance_decision=acceptance,
        inference_audit=audit,
        tabm_root=None,
    )


def _accepted_campaign(tmp_path: Path) -> dict[str, object]:
    campaign = _campaign(tmp_path, "accepted")
    sources = _delivery_sources(tmp_path, _token())
    campaign["delivery_sources"] = sources
    return campaign


def test_rejected_handoff_has_review_resume_and_no_delivery(tmp_path: Path) -> None:
    paths = write_e2_bundles(**_campaign(tmp_path, "rejected_structure"))
    with ZipFile(paths.handoff) as archive:
        assert set(archive.namelist()) == {
            "handoff_manifest.json",
            "tree_expert_e2_review.zip",
            "tree_expert_e2_resume.zip",
            "tree_expert_e2.log",
        }
    assert paths.delivery is None


def test_accepted_delivery_is_model_only_not_submission(tmp_path: Path) -> None:
    paths = write_e2_bundles(**_accepted_campaign(tmp_path))
    assert paths.delivery is not None
    with ZipFile(paths.delivery) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["review_only"] is False
        assert manifest["submission_package"] is False
        assert "script.py" not in archive.namelist()
        assert "submission.csv" not in archive.namelist()


def test_delivery_writer_requires_accepted_token(tmp_path: Path) -> None:
    sources = _delivery_sources(tmp_path, _token())
    with pytest.raises(TypeError, match="AcceptedForFullFit"):
        write_e2_delivery(object(), sources, tmp_path / "delivery.zip")


def test_bundle_is_reproducible_and_tamper_evident(tmp_path: Path) -> None:
    first = write_e2_bundles(**_accepted_campaign(tmp_path / "first"))
    second = write_e2_bundles(**_accepted_campaign(tmp_path / "second"))
    assert file_sha256(first.handoff) == file_sha256(second.handoff)

    changed = tmp_path / "changed.zip"
    with ZipFile(first.handoff) as source, ZipFile(changed, "w", ZIP_DEFLATED) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "tree_expert_e2_review.zip":
                payload += b"changed"
            target.writestr(info, payload)
    with pytest.raises(E2ArtifactError):
        verify_e2_handoff(changed)


@pytest.mark.parametrize(
    "binding", ["code_sha256", "contract_sha256", "train_sha256", "input_sha256"]
)
def test_resume_rejects_changed_binding(tmp_path: Path, binding: str) -> None:
    paths = write_e2_bundles(**_campaign(tmp_path, "budget_inconclusive"))
    changed = _bindings()
    changed[binding] = "0" * 64
    with pytest.raises(E2ArtifactError, match="binding"):
        verify_e2_resume(paths.resume, changed)


def test_verifier_rejects_traversal_member(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    info = ZipInfo("../escape")
    with ZipFile(path, "w") as archive:
        archive.writestr(info, b"x")
    with pytest.raises(E2ArtifactError, match="unsafe"):
        verify_e2_handoff(path)
