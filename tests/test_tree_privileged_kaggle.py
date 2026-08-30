from __future__ import annotations

import json
from pathlib import Path

from experiments.tree_privileged.kaggle import discover_inputs


def _official(root: Path) -> tuple[Path, dict[str, str]]:
    from hashlib import sha256
    official = root / "official"; official.mkdir(parents=True)
    for name, payload in {"train.csv": b"train", "trackman_history.csv": b"history",
                          "test.csv": b"test", "sample_submission.csv": b"sample"}.items():
        (official / name).write_bytes(payload)
    hashes = {"official_train_sha256": sha256(b"train").hexdigest(),
              "official_history_sha256": sha256(b"history").hexdigest(),
              "e2_handoff_sha256": "3" * 64}
    return official, hashes


def _artifact(root: Path, name: str, kind: str) -> Path:
    path = root / name; path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"artifact_kind": kind}, sort_keys=True))
    return path


def test_discovery_accepts_one_official_one_input_and_optional_handoff(tmp_path: Path) -> None:
    official, hashes = _official(tmp_path)
    compact = _artifact(tmp_path, "compact", "tree_privileged_input_v1")
    handoff = _artifact(tmp_path, "handoff", "tree_privileged_handoff_v1")
    found = discover_inputs(tmp_path, official_hashes=hashes)
    assert found.official_data == official
    assert found.campaign_input == compact
    assert found.previous_handoff == handoff


def test_nested_resume_inside_handoff_is_not_a_second_handoff(tmp_path: Path) -> None:
    _, hashes = _official(tmp_path)
    _artifact(tmp_path, "compact", "tree_privileged_input_v1")
    handoff = _artifact(tmp_path, "handoff", "tree_privileged_handoff_v1")
    _artifact(handoff, "resume", "tree_privileged_resume_v1")
    found = discover_inputs(tmp_path, official_hashes=hashes)
    assert found.previous_handoff == handoff

