from __future__ import annotations

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.tree_expert import e2_kaggle
from experiments.tree_expert.e2_kaggle import (
    _runtime_archive,
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


def _write_deterministic_zip(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload in sorted(members.items()):
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return path


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


def test_expanded_handoff_and_nested_resume_are_one_source(tmp_path: Path) -> None:
    hashes, _ = _write_inputs(tmp_path)
    handoff = tmp_path / "tree_expert_e2_handoff"
    handoff.mkdir()
    (handoff / "handoff_manifest.json").write_text(
        json.dumps({"artifact_kind": "tree_expert_e2_handoff_v1"}),
        encoding="utf-8",
    )
    resume = handoff / "tree_expert_e2_resume"
    resume.mkdir()
    (resume / "manifest.json").write_text(
        json.dumps({"artifact_kind": "tree_expert_e2_resume_v1"}),
        encoding="utf-8",
    )

    discovered = discover_inputs(tmp_path, official_hashes=hashes)

    assert discovered.resume == handoff


def test_expanded_handoff_rebuilds_exact_resume_archive(tmp_path: Path) -> None:
    original = _write_deterministic_zip(
        tmp_path / "original_resume.zip",
        {
            "manifest.json": json.dumps(
                {"artifact_kind": "tree_expert_e2_resume_v1"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            "state/stage_state.json": b"{}",
        },
    )
    handoff = tmp_path / "tree_expert_e2_handoff"
    handoff.mkdir()
    (handoff / "handoff_manifest.json").write_text(
        json.dumps({"artifact_kind": "tree_expert_e2_handoff_v1"}),
        encoding="utf-8",
    )
    expanded_resume = handoff / "tree_expert_e2_resume"
    with ZipFile(original) as archive:
        archive.extractall(expanded_resume)

    rebuilt = e2_kaggle.materialize_resume_source(
        handoff,
        tmp_path / "rebuilt_resume.zip",
    )

    assert sha256(rebuilt.read_bytes()).hexdigest() == sha256(original.read_bytes()).hexdigest()


def test_expanded_handoff_rejects_multiple_nested_resumes(tmp_path: Path) -> None:
    handoff = tmp_path / "tree_expert_e2_handoff"
    handoff.mkdir()
    (handoff / "handoff_manifest.json").write_text(
        json.dumps({"artifact_kind": "tree_expert_e2_handoff_v1"}),
        encoding="utf-8",
    )
    for name in ("first", "second"):
        resume = handoff / name
        resume.mkdir()
        (resume / "manifest.json").write_text(
            json.dumps({"artifact_kind": "tree_expert_e2_resume_v1"}),
            encoding="utf-8",
        )

    with pytest.raises(E2KaggleError, match="nested resume count"):
        e2_kaggle.materialize_resume_source(
            handoff,
            tmp_path / "rebuilt_resume.zip",
        )


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


def test_embedded_runtime_imports_without_repository_on_pythonpath(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    extracted = tmp_path / "runtime"
    extracted.mkdir()
    with tarfile.open(
        fileobj=io.BytesIO(_runtime_archive(repository)), mode="r:gz"
    ) as archive:
        archive.extractall(extracted, filter="data")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(extracted)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.tree_expert import e2_production; print(e2_production.__file__)",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().startswith(str(extracted))
