from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from experiments.tree_privileged.artifacts import CampaignArtifactState, verify_delivery, write_campaign_bundles


def _state(tmp_path: Path, status: str) -> CampaignArtifactState:
    source = tmp_path / "source"; source.mkdir()
    review = source / "decision.json"; review.write_text('{"status":"review"}')
    checkpoint = source / "checkpoint.bin"; checkpoint.write_bytes(b"checkpoint")
    log = source / "campaign.log"; log.write_text("DONE\n")
    delivery = None
    if status == "accepted":
        delivery = source / "delivery"; delivery.mkdir(); (delivery / "model.cbm").write_bytes(b"model")
    return CampaignArtifactState(status, MappingProxyType({"contract_sha256": "1" * 64}),
                                 MappingProxyType({"decision.json": review}),
                                 MappingProxyType({"jobs/checkpoint.bin": checkpoint}), delivery, log)


def test_rejected_campaign_has_review_resume_handoff_but_no_delivery(tmp_path: Path) -> None:
    result = write_campaign_bundles(_state(tmp_path, "rejected"), tmp_path / "bundles")
    assert result.review.is_file() and result.resume.is_file() and result.handoff.is_file()
    assert result.delivery is None


def test_accepted_campaign_delivery_is_hash_bound(tmp_path: Path) -> None:
    state = _state(tmp_path, "accepted")
    result = write_campaign_bundles(state, tmp_path / "bundles")
    verified = verify_delivery(result.delivery, state.bindings)
    assert verified["artifact_kind"] == "tree_privileged_delivery_v1"
