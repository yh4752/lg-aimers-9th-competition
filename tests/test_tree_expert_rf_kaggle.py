from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from zipfile import ZipFile

import pytest

from experiments.tree_expert.rf_kaggle import (
    RFKaggleError,
    discover_rf_inputs,
    run_supervised_rf_campaign,
    runtime_archive,
    runtime_member_names,
)


def _manifest(kind: str, identity: str = "same") -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": kind,
        "campaign_id": "tree_expert_rf_v1",
        "identity": identity,
        "members": {},
    }


def _expanded(root: Path, kind: str, identity: str = "same") -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(_manifest(kind, identity), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return root


def _archive(path: Path, kind: str, identity: str = "same") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(_manifest(kind, identity), sort_keys=True, separators=(",", ":")),
        )
    return path


def test_rf_runtime_archive_contains_direct_dependencies() -> None:
    members = runtime_member_names()
    assert "experiments/tree_expert/rf_runner.py" in members
    assert "experiments/tree_expert/rf_state.py" in members
    assert "experiments/tree_expert/features.py" in members
    assert "experiments/temporal_portfolio/seasonal_features.py" in members
    assert "experiments/independent_dl/feature_sources/trackman.py" in members
    assert not any("e2_" in name for name in members)


def test_rf_runtime_archive_imports_in_isolation(tmp_path: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(runtime_archive()), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experiments.tree_expert.rf_kaggle; "
            "import experiments.tree_expert.rf_runner; "
            "import experiments.tree_expert.rf_inference; "
            "experiments.tree_expert.rf_runner.code_sha256()",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_discovery_accepts_expanded_compact_input(tmp_path: Path) -> None:
    expanded = _expanded(tmp_path / "tree_expert_rf_input", "tree_expert_rf_input_v1")

    found = discover_rf_inputs(tmp_path)

    assert found.rf_input == expanded
    assert found.resume is None


def test_discovery_deduplicates_zip_and_expanded_copy(tmp_path: Path) -> None:
    expanded = _expanded(tmp_path / "input", "tree_expert_rf_input_v1")
    _archive(tmp_path / "input.zip", "tree_expert_rf_input_v1")

    found = discover_rf_inputs(tmp_path)

    assert found.rf_input == expanded


def test_discovery_rejects_two_distinct_resumes_even_if_nested(tmp_path: Path) -> None:
    _expanded(tmp_path / "input", "tree_expert_rf_input_v1")
    _expanded(tmp_path / "a" / "tree_expert_rf_resume", "tree_expert_rf_resume_v1", "first")
    _archive(tmp_path / "b" / "tree_expert_rf_resume.zip", "tree_expert_rf_resume_v1", "second")

    with pytest.raises(RFKaggleError, match="resume count must be zero or one"):
        discover_rf_inputs(tmp_path)


@dataclass(frozen=True)
class _CampaignResult:
    status: str
    resume_bundle: Path
    review_bundle: Path
    delivery_bundle: Path | None


def test_supervisor_publishes_only_final_stable_resume(tmp_path: Path) -> None:
    resume = tmp_path / "resume.zip"
    review = tmp_path / "review.zip"
    resume.write_bytes(b"resume")
    review.write_bytes(b"review")
    published = []

    result = run_supervised_rf_campaign(
        run_campaign=lambda: _CampaignResult("paused", resume, review, None),
        publish=published.append,
    )

    assert result.status == "paused"
    assert [(item.kind, item.path) for item in published] == [
        ("resume", resume),
        ("review", review),
    ]


def test_accepted_supervisor_publishes_each_artifact_once(tmp_path: Path) -> None:
    resume = tmp_path / "resume.zip"
    review = tmp_path / "review.zip"
    delivery = tmp_path / "delivery.zip"
    for path in (resume, review, delivery):
        path.write_bytes(path.name.encode())
    published = []

    run_supervised_rf_campaign(
        run_campaign=lambda: _CampaignResult("accepted", resume, review, delivery),
        publish=published.append,
    )

    assert [item.kind for item in published] == ["resume", "review", "delivery"]
