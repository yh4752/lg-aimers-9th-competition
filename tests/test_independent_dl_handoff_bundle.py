from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import zipfile

import pytest

from experiments.independent_dl.handoff import HandoffError, write_candidate_handoff


CANDIDATE_ID = "tabm__raw_typed__p3__s42"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "campaign"
    candidate = root / "candidates" / CANDIDATE_ID
    candidate.mkdir(parents=True)
    metrics = candidate / "metrics.json"
    predictions = candidate / "predictions.csv"
    metrics.write_text(
        json.dumps(
            {
                "candidate_id": CANDIDATE_ID,
                "brier": 0.247,
                "hardware": {"devices": [{"name": "T4", "vram_bytes": 16}]},
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        "row_id,fold,season,game_type,target,probability,candidate_id\n"
        f"r1,valid_2024,2024,R,1,0.7,{CANDIDATE_ID}\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "campaign_id": "independent_dl_campaign_v1",
        "protocol": "yearly_transition_independent_dl_v1",
        "candidates": {
            CANDIDATE_ID: {
                "candidate": {
                    "candidate_id": CANDIDATE_ID,
                    "family": "tabm",
                    "feature_view": "raw_typed",
                    "seed": 42,
                    "epochs": 240,
                },
                "state": "completed",
                "metrics_path": str(metrics.relative_to(root)),
                "predictions_path": str(predictions.relative_to(root)),
                "metrics_sha256": _digest(metrics),
                "predictions_sha256": _digest(predictions),
                "best_brier": 0.247,
                "failure_reason": None,
            }
        },
    }
    (root / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    requirements = tmp_path / "requirements-colab.txt"
    requirements.write_text("tabm==0.0.3\n", encoding="utf-8")
    environment = tmp_path / "environment.json"
    environment.write_text(
        json.dumps({"python": "3.11", "torch": "2.7.1", "gpu": "T4"}),
        encoding="utf-8",
    )
    return root, requirements, environment


def _write(tmp_path: Path):
    root, requirements, environment = _fixture(tmp_path)
    return write_candidate_handoff(
        campaign_root=root,
        candidate_id=CANDIDATE_ID,
        result_path=tmp_path / "candidate_handoff.zip",
        runtime_sha256="a" * 64,
        requirements_path=requirements,
        environment_path=environment,
    )


def test_handoff_contains_only_analysis_artifacts(tmp_path: Path) -> None:
    result = _write(tmp_path)

    assert result.path.is_file()
    assert result.sha256 == _digest(result.path)
    assert result.size_bytes == result.path.stat().st_size
    with zipfile.ZipFile(result.path) as bundle:
        assert set(bundle.namelist()) == {
            "handoff_manifest.json",
            "campaign_entry.json",
            "metrics.json",
            "predictions.csv",
            "requirements-colab.txt",
            "environment.json",
        }
        handoff = json.loads(bundle.read("handoff_manifest.json"))
        assert handoff["candidate_id"] == CANDIDATE_ID
        assert handoff["campaign_id"] == "independent_dl_campaign_v1"
        assert handoff["runtime_sha256"] == "a" * 64
        assert handoff["hardware"]["devices"][0]["name"] == "T4"
        assert set(handoff["members"]) == {
            "campaign_entry.json",
            "metrics.json",
            "predictions.csv",
            "requirements-colab.txt",
            "environment.json",
        }
        assert all(
            set(metadata) == {"sha256", "size_bytes"}
            for metadata in handoff["members"].values()
        )


@pytest.mark.parametrize("state", ["pending", "failed", "running"])
def test_handoff_rejects_noncompleted_candidate(tmp_path: Path, state: str) -> None:
    root, requirements, environment = _fixture(tmp_path)
    manifest_path = root / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidates"][CANDIDATE_ID]["state"] = state
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HandoffError, match="completed"):
        write_candidate_handoff(
            root, CANDIDATE_ID, tmp_path / "out.zip", "a" * 64,
            requirements, environment,
        )


def test_handoff_rejects_changed_prediction_hash(tmp_path: Path) -> None:
    root, requirements, environment = _fixture(tmp_path)
    (root / "candidates" / CANDIDATE_ID / "predictions.csv").write_text(
        "changed", encoding="utf-8"
    )

    with pytest.raises(HandoffError, match="SHA-256"):
        write_candidate_handoff(
            root, CANDIDATE_ID, tmp_path / "out.zip", "a" * 64,
            requirements, environment,
        )


def test_handoff_rejects_path_escape_and_symlink(tmp_path: Path) -> None:
    root, requirements, environment = _fixture(tmp_path)
    manifest_path = root / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    outside = tmp_path / "outside.csv"
    outside.write_text("outside", encoding="utf-8")
    manifest["candidates"][CANDIDATE_ID]["predictions_path"] = "../outside.csv"
    manifest["candidates"][CANDIDATE_ID]["predictions_sha256"] = _digest(outside)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HandoffError, match="campaign root"):
        write_candidate_handoff(
            root, CANDIDATE_ID, tmp_path / "escape.zip", "a" * 64,
            requirements, environment,
        )

    manifest["candidates"][CANDIDATE_ID]["predictions_path"] = "link.csv"
    link = root / "link.csv"
    link.symlink_to(outside)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HandoffError, match="regular file"):
        write_candidate_handoff(
            root, CANDIDATE_ID, tmp_path / "link.zip", "a" * 64,
            requirements, environment,
        )


def test_handoff_rejects_invalid_digest_and_existing_output(tmp_path: Path) -> None:
    root, requirements, environment = _fixture(tmp_path)
    with pytest.raises(HandoffError, match="runtime_sha256"):
        write_candidate_handoff(
            root, CANDIDATE_ID, tmp_path / "bad.zip", "ABC",
            requirements, environment,
        )

    output = tmp_path / "existing.zip"
    output.write_bytes(b"keep")
    with pytest.raises(HandoffError, match="already exists"):
        write_candidate_handoff(
            root, CANDIDATE_ID, output, "a" * 64, requirements, environment,
        )
    assert output.read_bytes() == b"keep"


def test_handoff_rejects_missing_candidate(tmp_path: Path) -> None:
    root, requirements, environment = _fixture(tmp_path)
    with pytest.raises(HandoffError, match="candidate_id"):
        write_candidate_handoff(
            root, "missing", tmp_path / "out.zip", "a" * 64,
            requirements, environment,
        )
