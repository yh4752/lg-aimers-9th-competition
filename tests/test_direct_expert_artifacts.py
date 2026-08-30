from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from experiments.direct_expert.artifacts import (
    DirectExpertArtifactError,
    DirectExpertBindings,
    create_bundle,
    extract_bundle,
    verify_bundle,
)


def bindings() -> DirectExpertBindings:
    return DirectExpertBindings(*[str(i) * 64 for i in range(1, 7)])


def test_rejected_campaign_has_no_delivery(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "review.json").write_text('{"status":"rejected"}', encoding="utf-8")
    review = create_bundle("review", {"review.json": source / "review.json"}, tmp_path / "review.zip", bindings())
    assert verify_bundle(review, "review", bindings())["artifact_kind"] == "direct_expert_review_v1"
    with pytest.raises(DirectExpertArtifactError, match="accepted token"):
        create_bundle("delivery", {"review.json": source / "review.json"}, tmp_path / "delivery.zip", bindings())


def test_changed_binding_rejects_resume(tmp_path: Path) -> None:
    payload = tmp_path / "state.json"
    payload.write_text("{}", encoding="utf-8")
    resume = create_bundle("resume", {"state.json": payload}, tmp_path / "resume.zip", bindings())
    with pytest.raises(DirectExpertArtifactError, match="artifact bindings differ"):
        verify_bundle(resume, "resume", replace(bindings(), contract_sha256="f" * 64))


def test_expanded_bundle_verifies_and_restores_without_zip_assumption(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('{"ok":true}', encoding="utf-8")
    bundle = create_bundle("resume", {"state/payload.json": payload}, tmp_path / "resume.zip", bindings())
    expanded = tmp_path / "expanded"
    shutil.unpack_archive(bundle, expanded)
    assert verify_bundle(expanded, "resume", bindings())["artifact_kind"] == "direct_expert_resume_v1"
    restored = extract_bundle(expanded, tmp_path / "restored", "resume", bindings())
    assert (restored / "state/payload.json").read_bytes() == payload.read_bytes()
