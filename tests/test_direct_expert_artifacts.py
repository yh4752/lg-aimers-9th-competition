from dataclasses import replace
from pathlib import Path

import pytest

from experiments.direct_expert.artifacts import (
    DirectExpertArtifactError,
    DirectExpertBindings,
    create_bundle,
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
