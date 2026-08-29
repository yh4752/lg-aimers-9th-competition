from pathlib import Path

import pytest

from experiments.tree_expert.s4_kaggle import (
    S4KaggleError,
    build_s4_kaggle_cell,
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
    assert "files.download" not in text
    assert "submission/package.py" not in text
    assert len(runtime_identity_sha256(Path(__file__).resolve().parents[1])) == 64


def test_committed_cell_matches_renderer(tmp_path):
    rendered = build_s4_kaggle_cell(tmp_path / "rendered.py")
    committed = Path("experiments/tree_expert/KAGGLE_S4_CELL.py")
    assert rendered.read_bytes() == committed.read_bytes()
