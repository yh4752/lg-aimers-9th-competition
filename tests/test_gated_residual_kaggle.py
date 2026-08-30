from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from experiments.gated_residual_final.kaggle import FinalKaggleError, discover_inputs, verify_t4x2


def _official(root: Path) -> Path:
    path = root / "official"
    path.mkdir()
    (path / "train.csv").write_text("x\n1\n")
    (path / "trackman_history.csv").write_text("x\n1\n")
    return path


def _artifact(root: Path, name: str, kind: str) -> Path:
    path = root / name
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"artifact_kind": kind}))
    return path


def test_discovery_accepts_expanded_final_input_and_one_handoff(tmp_path: Path) -> None:
    official = _official(tmp_path)
    final_input = _artifact(tmp_path, "final", "gated_residual_final_input_v1")
    handoff = _artifact(tmp_path, "handoff", "gated_residual_final_handoff_v1")

    found = discover_inputs(tmp_path)

    assert found.official_data == official
    assert found.final_input == final_input
    assert found.previous_handoff == handoff


def test_duplicate_logical_handoff_zip_and_directory_is_deduplicated(tmp_path: Path) -> None:
    _official(tmp_path)
    _artifact(tmp_path, "final", "gated_residual_final_input_v1")
    payload = {"artifact_kind": "gated_residual_final_handoff_v1", "identity": "same"}
    expanded = tmp_path / "handoff"
    expanded.mkdir()
    (expanded / "manifest.json").write_text(json.dumps(payload))
    with ZipFile(tmp_path / "handoff.zip", "w", ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(payload))

    assert discover_inputs(tmp_path).previous_handoff in {expanded, tmp_path / "handoff.zip"}


def test_more_than_one_distinct_handoff_fails_closed(tmp_path: Path) -> None:
    _official(tmp_path)
    _artifact(tmp_path, "final", "gated_residual_final_input_v1")
    _artifact(tmp_path, "h1", "gated_residual_final_handoff_v1")
    second = _artifact(tmp_path, "h2", "gated_residual_final_handoff_v1")
    (second / "manifest.json").write_text(json.dumps({"artifact_kind": "gated_residual_final_handoff_v1", "x": 2}))

    with pytest.raises(FinalKaggleError, match="handoff count"):
        discover_inputs(tmp_path)


def test_gpu_preflight_requires_exactly_two_t4s() -> None:
    class Cuda:
        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            return ("Tesla T4", "Tesla T4")[index]

    class Torch:
        cuda = Cuda()

    assert verify_t4x2(Torch) == ("Tesla T4", "Tesla T4")

    Cuda.device_count = staticmethod(lambda: 1)
    with pytest.raises(FinalKaggleError, match="two Tesla T4"):
        verify_t4x2(Torch)
