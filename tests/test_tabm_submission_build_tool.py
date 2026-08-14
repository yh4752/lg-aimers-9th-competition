from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from submission.tabm_existing_evidence import ExistingEvidenceError
from tools import build_tabm_submission_from_existing_evidence as builder


class _VersionInfo(tuple):
    @property
    def major(self) -> int:
        return int(self[0])

    @property
    def minor(self) -> int:
        return int(self[1])


def _arguments(tmp_path: Path) -> Namespace:
    for name in ("stage_c.zip", "stage_d.zip"):
        (tmp_path / name).write_bytes(b"fixture")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "test.csv").write_text(
        "row_id,x\nr0,0\nr1,1\nr2,2\nr3,3\nr4,4\n", encoding="utf-8"
    )
    (data / "sample_submission.csv").write_text(
        "row_id,control_success\nr0,0\nr1,0\nr2,0\nr3,0\nr4,0\n",
        encoding="utf-8",
    )
    return Namespace(
        stage_c_delivery=tmp_path / "stage_c.zip",
        stage_d_delivery=tmp_path / "stage_d.zip",
        candidate_root=candidate,
        official_data=data,
        output_dir=tmp_path / "output",
    )


def test_run_build_stops_before_output_when_evidence_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "__file__", str(tmp_path / "tools/build.py"))
    arguments = _arguments(tmp_path)
    monkeypatch.setattr(
        builder,
        "load_imported_candidate",
        lambda path: (_ for _ in ()).throw(ExistingEvidenceError("bad candidate")),
    )

    with pytest.raises(ExistingEvidenceError, match="bad candidate"):
        builder.run_build(arguments)

    assert not arguments.output_dir.exists()


def test_run_build_rejects_output_that_escapes_through_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(builder, "__file__", str(project / "tools/build.py"))
    arguments = _arguments(project)
    arguments.output_dir = project / "linked/output"

    with pytest.raises(builder.SubmissionBuildError, match="outside project root"):
        builder.run_build(arguments)


def test_created_package_verifier_requires_exact_runtime_and_layout(
    tmp_path: Path,
) -> None:
    import zipfile

    archive = tmp_path / "submit.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("script.py", b"runtime")
        output.writestr("requirements.txt", b"tabm==0.0.3\n")
        output.writestr("model/weights.pt", b"weights")

    builder.verify_created_package(
        archive,
        expected_runtime=b"runtime",
        expected_requirements=b"tabm==0.0.3\n",
        expected_model={"weights.pt": b"weights"},
    )

    with pytest.raises(builder.SubmissionBuildError, match="script.py"):
        builder.verify_created_package(
            archive,
            expected_runtime=b"changed",
            expected_requirements=b"tabm==0.0.3\n",
            expected_model={"weights.pt": b"weights"},
        )


def test_final_builder_requires_exact_official_python_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder.sys, "version_info", _VersionInfo((3, 11, 14)))
    monkeypatch.setattr(
        builder.importlib.metadata,
        "version",
        lambda name: {
            "tabm": "0.0.3",
            "rtdl-num-embeddings": "0.0.12",
            "torch": "2.7.1",
            "pandas": "2.0.3",
            "numpy": "1.26.4",
        }[name],
    )

    with pytest.raises(builder.SubmissionBuildError, match="3.11.15"):
        builder.require_python311_environment()


def test_final_builder_requires_official_base_library_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder.sys, "version_info", _VersionInfo((3, 11, 15)))
    versions = {
        "tabm": "0.0.3",
        "rtdl-num-embeddings": "0.0.12",
        "torch": "2.7.1+cpu",
        "pandas": "2.1.0",
        "numpy": "1.26.4",
    }
    monkeypatch.setattr(builder.importlib.metadata, "version", versions.__getitem__)

    with pytest.raises(builder.SubmissionBuildError, match="base library"):
        builder.require_python311_environment()
