from __future__ import annotations

from pathlib import Path

import pytest

from experiments.gated_residual_final.artifacts import (
    ArtifactBindings,
    FinalArtifactError,
    create_delivery,
    create_handoff,
    extract_bundle,
    verify_bundle,
)


def _bindings(code: str = "a" * 64) -> ArtifactBindings:
    return ArtifactBindings(
        code_sha256=code,
        contract_sha256="b" * 64,
        input_manifest_sha256="c" * 64,
        train_sha256="d" * 64,
        history_sha256="e" * 64,
    )


def test_rejected_campaign_cannot_create_delivery(tmp_path: Path) -> None:
    payload = tmp_path / "model.bin"
    payload.write_bytes(b"model")

    with pytest.raises(FinalArtifactError, match="accepted evidence is required"):
        create_delivery(
            tmp_path / "delivery.zip", bindings=_bindings(), payloads={"model/model.bin": payload},
            acceptance_evidence={"status": "rejected"},
        )


def test_handoff_rejects_different_code_binding(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text("{}")
    archive = create_handoff(
        tmp_path / "handoff.zip", bindings=_bindings(), payloads={"campaign_state.json": state}
    )

    with pytest.raises(FinalArtifactError, match="artifact bindings differ"):
        verify_bundle(archive, kind="handoff", expected_bindings=_bindings("f" * 64))


def test_accepted_delivery_round_trip_verifies_every_member(tmp_path: Path) -> None:
    payload = tmp_path / "model.bin"
    payload.write_bytes(b"model")
    archive = create_delivery(
        tmp_path / "delivery.zip", bindings=_bindings(), payloads={"model/model.bin": payload},
        acceptance_evidence={"status": "accepted", "candidate_id": "G1_x"},
    )

    verified = verify_bundle(archive, kind="delivery", expected_bindings=_bindings())

    assert verified["acceptance_evidence"]["status"] == "accepted"
    assert "model/model.bin" in verified["members"]


def test_verified_handoff_extracts_only_declared_members(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text("{}")
    archive = create_handoff(
        tmp_path / "handoff.zip", bindings=_bindings(), payloads={"campaign_state.json": state}
    )

    root = extract_bundle(
        archive, tmp_path / "restored", kind="handoff", expected_bindings=_bindings()
    )

    assert (root / "campaign_state.json").read_text() == "{}"
    assert not (root / "manifest.json").exists()
