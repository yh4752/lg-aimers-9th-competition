from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

import experiments.tree_privileged.inputs as input_module
from experiments.tree_privileged.inputs import PrivilegedInputError, file_sha256, prepare_input, verify_and_extract_input


def _nested_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _handoff(path: Path) -> Path:
    review = _nested_zip({
        "ensembles/2021_2022.csv": b"row_id,probability\na,0.4\n",
        "ensembles/2022_2023.csv": b"row_id,probability\nb,0.5\n",
        "ensembles/2023_2024.csv": b"row_id,probability\nc,0.6\n",
    })
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("tree_expert_e2_review.zip", review)
        archive.writestr("tree_expert_e2_model_delivery.zip", _nested_zip({"manifest.json": b"{}"}))
    return path


def test_prepare_and_verify_input_keeps_only_accepted_e2(monkeypatch, tmp_path: Path) -> None:
    handoff = _handoff(tmp_path / "handoff.zip")
    digest = file_sha256(handoff)
    monkeypatch.setattr(input_module, "verify_e2_handoff", lambda _: SimpleNamespace(status="accepted", delivery=True))
    archive = prepare_input(handoff, tmp_path / "tree_privileged_input.zip", expected_e2_sha256=digest)
    verified = verify_and_extract_input(archive, tmp_path / "verified", expected_e2_sha256=digest)
    assert verified.e2_handoff_sha256 == digest
    assert verified.e2_delivery.is_file()
    assert {path.name for path in verified.e2_oof_root.iterdir()} == {"2022.csv", "2023.csv", "2024.csv"}


def test_input_rejects_an_extra_or_modified_member(monkeypatch, tmp_path: Path) -> None:
    handoff = _handoff(tmp_path / "handoff.zip")
    digest = file_sha256(handoff)
    monkeypatch.setattr(input_module, "verify_e2_handoff", lambda _: SimpleNamespace(status="accepted", delivery=True))
    source = prepare_input(handoff, tmp_path / "source.zip", expected_e2_sha256=digest)
    changed = tmp_path / "changed.zip"
    with ZipFile(source) as reader, ZipFile(changed, "w", compression=ZIP_DEFLATED) as writer:
        for info in reader.infolist():
            writer.writestr(info, reader.read(info.filename))
        writer.writestr("extra.bin", b"not authorized")
    with pytest.raises(PrivilegedInputError, match="member set differs"):
        verify_and_extract_input(changed, tmp_path / "changed", expected_e2_sha256=digest)


def test_default_preparer_requires_the_sealed_e2_hash(tmp_path: Path) -> None:
    handoff = _handoff(tmp_path / "handoff.zip")
    with pytest.raises(PrivilegedInputError, match="SHA-256 differs"):
        prepare_input(handoff, tmp_path / "output.zip")
