from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.direct_expert.inputs import file_sha256
from experiments.direct_expert.stage_a import run_stage_a
from experiments.direct_expert.stage_b import run_stage_b
from tests.test_direct_expert_stage_a import FakeRuntime as StageARuntime
from tests.test_direct_expert_stage_b import FakeRuntime as StageBRuntime


@dataclass(frozen=True)
class FixtureCampaign:
    stage_a_sha256: str
    review_sha256: str
    status: str
    delivery: Path | None


def _run_fixture_campaign(root: Path) -> FixtureCampaign:
    stage_a = run_stage_a(StageARuntime(), root / "stage_a", absolute_deadline=20_000)
    stage_b = run_stage_b(
        StageBRuntime(accepted=False),
        stage_a.handoff,
        root / "stage_b",
        absolute_deadline=20_000,
    )
    return FixtureCampaign(
        file_sha256(stage_a.handoff),
        file_sha256(stage_b.review),
        stage_b.status,
        stage_b.delivery,
    )


def test_two_stage_fixture_campaign_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    first = _run_fixture_campaign(tmp_path / "first")
    second = _run_fixture_campaign(tmp_path / "second")
    assert first.stage_a_sha256 == second.stage_a_sha256
    assert first.review_sha256 == second.review_sha256
    assert first.status == "rejected"
    assert first.delivery is None
    assert not tuple(tmp_path.rglob("submission.zip"))
