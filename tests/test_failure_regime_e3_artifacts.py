from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from experiments.failure_regime_e3.artifacts import (
    E3ArtifactError,
    E3Bindings,
    create_bundle,
    extract_bundle,
    verify_bundle,
)


def _bindings(**changes: str) -> E3Bindings:
    values = {
        "contract_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "train_sha256": "d" * 64,
        "history_sha256": "e" * 64,
        "e2_submission_sha256": "f" * 64,
    }
    values.update(changes)
    return E3Bindings(**values)


def test_review_bundle_is_deterministic_and_directory_equivalent(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    evidence = tmp_path / "decision.json"
    state.write_text('{"phase":"decision"}', encoding="utf-8")
    evidence.write_text('{"accepted":false}', encoding="utf-8")
    payloads = {"campaign/state.json": state, "evidence/decision.json": evidence}

    first = create_bundle("review", payloads, tmp_path / "first.zip", _bindings())
    second = create_bundle("review", payloads, tmp_path / "second.zip", _bindings())
    assert first.read_bytes() == second.read_bytes()
    manifest = verify_bundle(first, "review", _bindings())
    extracted = extract_bundle(first, tmp_path / "expanded", "review", _bindings())
    assert verify_bundle(extracted, "review", _bindings()) == manifest
    assert manifest["submission_package"] is False


def test_handoff_rejects_stale_binding_and_unsafe_or_extra_members(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    bundle = create_bundle("handoff", {"campaign/state.json": state}, tmp_path / "handoff.zip", _bindings())
    with pytest.raises(E3ArtifactError, match="bindings"):
        verify_bundle(bundle, "handoff", _bindings(code_sha256="1" * 64))

    with zipfile.ZipFile(tmp_path / "unsafe.zip", "w") as archive:
        archive.writestr("../state.json", b"{}")
        archive.writestr("manifest.json", json.dumps({"artifact_kind": "failure_regime_e3_handoff_v1"}))
    with pytest.raises(E3ArtifactError, match="unsafe"):
        verify_bundle(tmp_path / "unsafe.zip", "handoff", _bindings())

    extracted = extract_bundle(bundle, tmp_path / "dir", "handoff", _bindings())
    (extracted / "extra.bin").write_bytes(b"x")
    with pytest.raises(E3ArtifactError, match="member set"):
        verify_bundle(extracted, "handoff", _bindings())


def test_bundle_kind_is_limited_to_review_and_handoff(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    with pytest.raises(E3ArtifactError, match="kind"):
        create_bundle("delivery", {"state.json": state}, tmp_path / "delivery.zip", _bindings())
