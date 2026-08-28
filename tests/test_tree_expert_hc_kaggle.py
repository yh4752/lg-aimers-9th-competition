from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.tree_expert.hc_kaggle import HCKaggleError, discover_inputs, verify_gpu


def _official(root: Path) -> tuple[str, str]:
    data = root / "official"
    data.mkdir(parents=True)
    train = data / "train.csv"
    history = data / "trackman_history.csv"
    train.write_bytes(b"train")
    history.write_bytes(b"history")
    return sha256(b"train").hexdigest(), sha256(b"history").hexdigest()


def _artifact(root: Path, name: str, kind: str, identity: str) -> Path:
    path = root / name
    path.mkdir()
    (path / ("handoff_manifest.json" if "handoff" in kind else "manifest.json")).write_text(
        json.dumps({"artifact_kind": kind, "identity": identity})
    )
    return path


def test_discovery_finds_official_hc_input_and_optional_handoff(tmp_path: Path):
    train_sha, history_sha = _official(tmp_path)
    compact = _artifact(tmp_path, "input", "tree_hierarchical_input_v1", "input-a")
    handoff = _artifact(tmp_path, "handoff", "tree_hierarchical_handoff_v1", "resume-a")
    found = discover_inputs(
        tmp_path,
        official_hashes={
            "official_train_sha256": train_sha,
            "official_history_sha256": history_sha,
        },
    )
    assert found.hc_input == compact
    assert found.previous_handoff == handoff


def test_discovery_rejects_distinct_previous_handoffs(tmp_path: Path):
    train_sha, history_sha = _official(tmp_path)
    _artifact(tmp_path, "input", "tree_hierarchical_input_v1", "input-a")
    _artifact(tmp_path, "handoff-a", "tree_hierarchical_handoff_v1", "resume-a")
    _artifact(tmp_path, "handoff-b", "tree_hierarchical_handoff_v1", "resume-b")
    with pytest.raises(HCKaggleError, match="handoff count"):
        discover_inputs(
            tmp_path,
            official_hashes={
                "official_train_sha256": train_sha,
                "official_history_sha256": history_sha,
            },
        )


def test_gpu_requires_one_or_two_t4_devices():
    class Cuda:
        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            return "Tesla T4"

    assert verify_gpu(type("Torch", (), {"cuda": Cuda})) == ("Tesla T4", "Tesla T4")

