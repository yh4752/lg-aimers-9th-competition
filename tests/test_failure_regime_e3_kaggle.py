from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from competition_rules.code_gate import inspect_inference_source
from experiments.failure_regime_e3.kaggle import (
    E3KaggleError,
    discover_inputs,
    inference_source_paths,
    verify_t4x2,
)


def _manifest(root: Path, name: str, kind: str, token: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"artifact_kind": kind, "token": token}), encoding="utf-8")
    return path


def test_discovery_accepts_expanded_inputs_and_deduplicates_same_zip_identity(tmp_path: Path) -> None:
    official = tmp_path / "official"
    official.mkdir()
    (official / "train.csv").write_text("row_id\n1\n", encoding="utf-8")
    (official / "trackman_history.csv").write_text("x\n1\n", encoding="utf-8")
    direct = _manifest(tmp_path, "direct", "direct_expert_input_v1", "a")
    with ZipFile(tmp_path / "direct.zip", "w") as archive:
        archive.write(direct / "manifest.json", "manifest.json")
    handoff = _manifest(tmp_path, "handoff", "failure_regime_e3_handoff_v1", "b")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")

    found = discover_inputs(tmp_path)
    assert found.official_data == official
    assert found.campaign_input in {direct, tmp_path / "direct.zip"}
    assert found.previous_handoff == handoff


def test_discovery_rejects_two_logically_different_handoffs(tmp_path: Path) -> None:
    official = tmp_path / "official"; official.mkdir()
    (official / "train.csv").write_text("x", encoding="utf-8")
    (official / "trackman_history.csv").write_text("x", encoding="utf-8")
    _manifest(tmp_path, "direct", "direct_expert_input_v1", "a")
    _manifest(tmp_path, "handoff1", "failure_regime_e3_handoff_v1", "b")
    _manifest(tmp_path, "handoff2", "failure_regime_e3_handoff_v1", "c")
    with pytest.raises(E3KaggleError, match="handoff count.*2"):
        discover_inputs(tmp_path)


class _Cuda:
    @staticmethod
    def device_count() -> int:
        return 2

    @staticmethod
    def get_device_name(index: int) -> str:
        return ("Tesla T4", "Tesla T4")[index]


class _Torch:
    cuda = _Cuda()


def test_gpu_gate_requires_exactly_two_t4_devices() -> None:
    assert verify_t4x2(_Torch) == ("Tesla T4", "Tesla T4")
    _Torch.cuda.device_count = staticmethod(lambda: 1)
    with pytest.raises(E3KaggleError, match="two Tesla T4"):
        verify_t4x2(_Torch)


def test_full_inference_source_closure_passes_policy_gate() -> None:
    root = Path.cwd()
    paths = inference_source_paths(root)

    assert len(paths) == 10
    assert all(path.is_file() for path in paths)
    assert inspect_inference_source(paths, project_root=root)["status"] == "passed"
