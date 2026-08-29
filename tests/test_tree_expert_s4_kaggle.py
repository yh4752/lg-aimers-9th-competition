from pathlib import Path
from hashlib import sha256
import json

import pytest

from experiments.tree_expert.s4_kaggle import (
    S4KaggleError,
    build_s4_kaggle_cell,
    discover_s4_inputs,
    runtime_member_names,
    runtime_identity_sha256,
    verify_gpu,
)


class Cuda:
    def device_count(self): return 2
    def get_device_name(self, index): return "Tesla T4"


class Torch:
    cuda = Cuda()


def test_gpu_contract_requires_exactly_two_t4_devices():
    assert verify_gpu(Torch()) == ("Tesla T4", "Tesla T4")
    Torch.cuda.device_count = lambda: 1
    with pytest.raises(S4KaggleError):
        verify_gpu(Torch())


def test_rendered_cell_is_deterministic_small_and_has_no_submission(tmp_path):
    first = build_s4_kaggle_cell(tmp_path / "first.py")
    second = build_s4_kaggle_cell(tmp_path / "second.py")
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 1_000_000
    text = first.read_text()
    assert "S4_HANDOFF_READY" in text and "S4_SUCCESS" in text
    assert "S4_RECOVERY_READY" in text
    assert "files.download" not in text
    assert "submission/package.py" not in text
    assert len(runtime_identity_sha256(Path(__file__).resolve().parents[1])) == 64


def test_committed_cell_matches_renderer(tmp_path):
    rendered = build_s4_kaggle_cell(tmp_path / "rendered.py")
    committed = Path("experiments/tree_expert/KAGGLE_S4_CELL.py")
    assert rendered.read_bytes() == committed.read_bytes()


def test_runtime_inventory_is_s4_scoped():
    members = runtime_member_names()
    assert "experiments/tree_expert/s4_production.py" in members
    assert "experiments/tree_expert/s4_recovery.py" in members
    assert "experiments/tree_expert/s4_recovery_contract.json" in members
    assert not any(name.startswith("experiments/tabm_campaign/") for name in members)
    assert not any("/e2_" in name for name in members)


def _manifest(root: Path, kind: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps({"artifact_kind": kind}, sort_keys=True), encoding="utf-8"
    )


def _official(root: Path) -> dict[str, str]:
    root.mkdir(parents=True)
    train = root / "train.csv"
    history = root / "trackman_history.csv"
    train.write_bytes(b"train")
    history.write_bytes(b"history")
    return {
        "official_train_sha256": sha256(train.read_bytes()).hexdigest(),
        "official_history_sha256": sha256(history.read_bytes()).hexdigest(),
    }


def test_discovery_accepts_one_compact_recovery_input(tmp_path):
    hashes = _official(tmp_path / "official")
    _manifest(tmp_path / "base", "tree_s4_input_v1")
    recovery = tmp_path / "recovery"
    _manifest(recovery, "tree_s4_recovery_input_v1")

    found = discover_s4_inputs(tmp_path, official_hashes=hashes)

    assert found.recovery_input == recovery
    assert found.previous_handoff is None


def test_discovery_rejects_recovery_and_previous_handoff(tmp_path):
    hashes = _official(tmp_path / "official")
    _manifest(tmp_path / "base", "tree_s4_input_v1")
    _manifest(tmp_path / "recovery", "tree_s4_recovery_input_v1")
    _manifest(tmp_path / "previous", "tree_s4_handoff_v1")

    with pytest.raises(S4KaggleError, match="mutually exclusive"):
        discover_s4_inputs(tmp_path, official_hashes=hashes)
