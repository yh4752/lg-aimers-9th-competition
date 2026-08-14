from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from competition_rules.contract import (
    RulesContractError,
    load_policy,
    load_policy_review,
    policy_digest,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "competition_rules/policy.json"


def test_checked_in_policy_matches_official_rules() -> None:
    policy = load_policy(POLICY, project_root=ROOT)

    assert policy["policy_version"] == "dacon-236743-2026-08-15"
    assert policy["competition_id"] == "236743"
    assert policy["allowed_data_sources"] == [
        "official_train",
        "official_trackman",
    ]
    assert policy["limits"] == {
        "install_seconds": 600,
        "inference_seconds": 600,
        "package_bytes": 10_000_000_000,
        "extracted_bytes": 32_000_000_000,
        "evaluation_rows": 245789,
    }
    assert policy["manual_rules"] == {
        "team_max_members": 5,
        "duplicate_registration_allowed": False,
        "daily_submission_limit": 5,
    }
    assert len(policy["official_sources"]) == 7
    assert all(
        item["url"].startswith("https://dacon.io/")
        for item in policy["official_sources"]
    )
    assert any(
        item["url"].endswith("/talkboard/417082?page=1&dtype=recent")
        for item in policy["official_sources"]
    )
    assert policy["leaderboard_selection"] == "highest_compliant_submission"


def test_old_policy_version_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    payload["policy_version"] = "dacon-236743-2026-08-13"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RulesContractError, match="unknown policy_version"):
        load_policy(path, project_root=tmp_path)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}'])
def test_policy_rejects_ambiguous_json(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "policy.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(RulesContractError):
        load_policy(path, project_root=tmp_path)


def test_policy_rejects_symlinked_file(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "policy.json"
    link.symlink_to(target)

    with pytest.raises(RulesContractError, match="symlink"):
        load_policy(link, project_root=tmp_path)


def test_review_must_match_policy_and_package_date(tmp_path: Path) -> None:
    policy = load_policy(POLICY, project_root=ROOT)
    now = datetime(2026, 8, 13, 23, 59, tzinfo=ZoneInfo("Asia/Seoul"))
    review = {
        "schema_version": 1,
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_digest(policy),
        "reviewed_at": "2026-08-13T15:20:00+09:00",
        "sources": policy["official_sources"],
        "verdict": "unchanged",
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")

    assert load_policy_review(
        path, policy=policy, package_time=now
    )["verdict"] == "unchanged"

    review["reviewed_at"] = "2026-08-12T23:59:59+09:00"
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(RulesContractError, match="same KST date"):
        load_policy_review(path, policy=policy, package_time=now)


def test_policy_digest_is_canonical() -> None:
    left = {"b": [2, 1], "a": {"x": True}}
    right = {"a": {"x": True}, "b": [2, 1]}

    assert policy_digest(left) == policy_digest(right)
