from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tree_expert.e2_kaggle import (
    E2KaggleError,
    discover_inputs,
    runtime_identity_sha256,
    runtime_member_names,
    verify_gpu,
)


def _write_inputs(root: Path) -> tuple[dict[str, str], Path]:
    official = root / "official"
    official.mkdir(parents=True)
    train = official / "train.csv"
    history = official / "trackman_history.csv"
    train.write_bytes(b"train")
    history.write_bytes(b"history")
    compact = root / "compact"
    compact.mkdir()
    (compact / "manifest.json").write_text(
        json.dumps({"artifact_kind": "tree_expert_e2_input_v1"}), encoding="utf-8"
    )
    return {
        "official_train_sha256": sha256(train.read_bytes()).hexdigest(),
        "official_history_sha256": sha256(history.read_bytes()).hexdigest(),
    }, compact


def test_discovers_one_official_data_and_one_e2_input(tmp_path: Path) -> None:
    hashes, _ = _write_inputs(tmp_path)
    discovered = discover_inputs(tmp_path, official_hashes=hashes)
    assert discovered.official_data.name == "official"
    assert discovered.e2_input.name == "compact"
    assert discovered.resume is None


def test_optional_resume_rejects_ambiguous_sources(tmp_path: Path) -> None:
    hashes, _ = _write_inputs(tmp_path)
    for name in ("first", "second"):
        resume = tmp_path / name
        resume.mkdir()
        (resume / "manifest.json").write_text(
            json.dumps({"artifact_kind": "tree_expert_e2_resume_v1"}), encoding="utf-8"
        )
    with pytest.raises(E2KaggleError, match="resume count"):
        discover_inputs(tmp_path, official_hashes=hashes)


def test_requires_two_cuda_devices() -> None:
    fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
    with pytest.raises(E2KaggleError, match="two CUDA"):
        verify_gpu(fake)


def test_runtime_identity_hash_changes(tmp_path: Path) -> None:
    for name in runtime_member_names():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    before = runtime_identity_sha256(tmp_path)
    first = tmp_path / runtime_member_names()[0]
    first.write_text("changed", encoding="utf-8")
    assert runtime_identity_sha256(tmp_path) != before


def test_runtime_inventory_has_e2_and_no_generated_submission() -> None:
    names = runtime_member_names()
    assert "experiments/tree_expert/e2_runner.py" in names
    assert "experiments/tree_expert/e2_kaggle.py" in names
    assert "experiments/tree_expert/e2_production.py" in names
    assert all("KAGGLE_E2_CELL" not in name for name in names)
    assert all("submission" not in Path(name).name.lower() for name in names)
