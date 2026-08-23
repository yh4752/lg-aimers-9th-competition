from __future__ import annotations

import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import errno
import subprocess
import sys
from zipfile import ZipFile

import pytest

from experiments.temporal_portfolio.inputs import (
    PortfolioInputError,
    prepare_input_archive,
    verify_official_data,
)


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


@pytest.fixture
def tiny_official_dir(tmp_path: Path) -> Path:
    root = tmp_path / "official"
    root.mkdir()
    _write_csv(
        root / "train.csv",
        [["row_id", "feature", "control_success"], ["train-1", "a", "1"], ["train-2", "b", "0"], ["train-3", "c", "1"]],
    )
    _write_csv(root / "test.csv", [["row_id", "feature"], ["test-1", "x"], ["test-2", "y"]])
    _write_csv(root / "trackman_history.csv", [["pitcher_id", "velocity"], ["p1", "90"]])
    _write_csv(root / "sample_submission.csv", [["row_id", "control_success"], ["test-1", "0.5"], ["test-2", "0.5"]])
    return root


def test_verify_data_requires_exact_official_members(tiny_official_dir: Path) -> None:
    verified = verify_official_data(tiny_official_dir)
    assert verified.train.name == "train.csv"
    assert verified.test.name == "test.csv"
    assert verified.history.name == "trackman_history.csv"
    assert verified.train_rows > verified.test_rows > 0


def test_prepare_input_archive_contains_data_identity_not_submission(
    tmp_path: Path, tiny_official_dir: Path
) -> None:
    result = prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")
    with ZipFile(result.path) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "data/train.csv",
            "data/test.csv",
            "data/trackman_history.csv",
            "data/sample_submission.csv",
        }
        assert "submission.csv" not in archive.namelist()


def test_verify_rejects_unrelated_top_level_member(tiny_official_dir: Path) -> None:
    (tiny_official_dir / "notes.txt").write_text("not official", encoding="utf-8")

    with pytest.raises(PortfolioInputError, match="top-level members"):
        verify_official_data(tiny_official_dir)


@pytest.mark.parametrize(
    ("name", "rows", "message"),
    [
        ("train.csv", [["row_id", "feature"], ["train-1", "a"]], "control_success"),
        (
            "test.csv",
            [["row_id", "feature", "control_success"], ["test-1", "x", "0"]],
            "must not contain control_success",
        ),
    ],
)
def test_verify_enforces_train_test_target_boundary(
    tiny_official_dir: Path, name: str, rows: list[list[str]], message: str
) -> None:
    _write_csv(tiny_official_dir / name, rows)

    with pytest.raises(PortfolioInputError, match=message):
        verify_official_data(tiny_official_dir)


def test_verify_wraps_missing_row_id_as_portfolio_input_error(
    tiny_official_dir: Path,
) -> None:
    _write_csv(tiny_official_dir / "test.csv", [["feature"], ["x"], ["y"]])
    _write_csv(
        tiny_official_dir / "sample_submission.csv",
        [["row_id", "control_success"], ["test-1", "0.5"], ["test-2", "0.5"]],
    )

    with pytest.raises(PortfolioInputError, match="row_id"):
        verify_official_data(tiny_official_dir)


def test_verify_requires_sample_row_ids_to_match_test(tiny_official_dir: Path) -> None:
    _write_csv(
        tiny_official_dir / "sample_submission.csv",
        [["row_id", "control_success"], ["test-2", "0.5"], ["test-1", "0.5"]],
    )

    with pytest.raises(PortfolioInputError, match="row_id sequence"):
        verify_official_data(tiny_official_dir)


@pytest.mark.parametrize("name", ["train.csv", "test.csv", "sample_submission.csv"])
def test_verify_rejects_duplicate_row_id(tiny_official_dir: Path, name: str) -> None:
    if name == "train.csv":
        _write_csv(
            tiny_official_dir / name,
            [["row_id", "feature", "control_success"], ["same", "a", "1"], ["same", "b", "0"], ["third", "c", "1"]],
        )
    elif name == "test.csv":
        _write_csv(tiny_official_dir / name, [["row_id", "feature"], ["same", "x"], ["same", "y"]])
        _write_csv(tiny_official_dir / "sample_submission.csv", [["row_id", "control_success"], ["same", "0.5"], ["same", "0.5"]])
    else:
        _write_csv(tiny_official_dir / name, [["row_id", "control_success"], ["test-1", "0.5"], ["test-1", "0.5"]])

    with pytest.raises(PortfolioInputError, match="duplicate row_id"):
        verify_official_data(tiny_official_dir)


def test_verify_rejects_malformed_row_and_nonfinite_sample_placeholder(
    tiny_official_dir: Path,
) -> None:
    _write_csv(tiny_official_dir / "trackman_history.csv", [["pitcher_id", "velocity"], ["p1"]])
    with pytest.raises(PortfolioInputError, match="malformed row"):
        verify_official_data(tiny_official_dir)

    _write_csv(tiny_official_dir / "trackman_history.csv", [["pitcher_id", "velocity"], ["p1", "90"]])
    _write_csv(tiny_official_dir / "sample_submission.csv", [["row_id", "control_success"], ["test-1", "nan"], ["test-2", "0.5"]])
    with pytest.raises(PortfolioInputError, match="finite numeric"):
        verify_official_data(tiny_official_dir)


def test_verify_rejects_symlink_source(tiny_official_dir: Path, tmp_path: Path) -> None:
    linked = tmp_path / "linked-official"
    linked.symlink_to(tiny_official_dir, target_is_directory=True)

    with pytest.raises(PortfolioInputError, match="symlink"):
        verify_official_data(linked)


def test_prepare_is_deterministic_and_manifest_binds_exact_members(
    tmp_path: Path, tiny_official_dir: Path
) -> None:
    first = prepare_input_archive(tiny_official_dir, tmp_path / "first.zip")
    second = prepare_input_archive(tiny_official_dir, tmp_path / "second.zip")

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.size_bytes == len(first.path.read_bytes())
    with ZipFile(first.path) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "data/train.csv",
            "data/test.csv",
            "data/trackman_history.csv",
            "data/sample_submission.csv",
        ]
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema_version"] == 1
        assert manifest["artifact_kind"] == "temporal_portfolio_input_v1"
        assert manifest["campaign_id"] == "temporal_portfolio_v1"
        assert manifest["submission_package"] is False
        assert manifest["train_rows"] == 3
        assert manifest["test_rows"] == 2
        assert manifest["contract_sha256"] == sha256(
            (Path(__file__).parents[1] / "experiments/temporal_portfolio/contract.json").read_bytes()
        ).hexdigest()
        for name, evidence in manifest["members"].items():
            assert isinstance(evidence["size"], int)
            assert evidence["size"] == archive.getinfo(name).file_size
            assert evidence["sha256"] == sha256(archive.read(name)).hexdigest()


def test_prepare_rejects_existing_output_without_replace(
    tmp_path: Path, tiny_official_dir: Path
) -> None:
    output = tmp_path / "input.zip"
    output.write_bytes(b"existing")

    with pytest.raises(PortfolioInputError, match="output already exists"):
        prepare_input_archive(tiny_official_dir, output)

    prepared = prepare_input_archive(tiny_official_dir, output, replace=True)
    assert prepared.path == output.absolute()
    assert prepared.path.read_bytes() != b"existing"


def test_prepare_detects_source_mutation_after_copy(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    original = module._copy_source

    def mutate_after_copy(
        source: Path, destination: object, **kwargs: object
    ) -> None:
        original(source, destination, **kwargs)
        if source.name == "train.csv":
            source.write_text(
                "row_id,feature,control_success\\nchanged,a,1\\nchanged-2,b,0\\nchanged-3,c,1\\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(module, "_copy_source", mutate_after_copy)
    with pytest.raises(PortfolioInputError, match="changed"):
        prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")


def test_prepare_cli_help() -> None:
    command = [
        sys.executable,
        "tools/prepare_temporal_portfolio_input.py",
        "--help",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    assert "--data-dir" in completed.stdout
    assert "--output" in completed.stdout


def test_verify_rejects_unterminated_quoted_csv_field(tiny_official_dir: Path) -> None:
    (tiny_official_dir / "trackman_history.csv").write_text(
        'pitcher_id\n"unterminated', encoding="utf-8"
    )

    with pytest.raises(PortfolioInputError, match="trackman_history.csv"):
        verify_official_data(tiny_official_dir)


def test_prepare_binds_train_baseline_to_the_parsed_file_version(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    original = module._read_csv

    def replace_after_parse(source: object, *args: object, **kwargs: object) -> object:
        parsed = original(source, *args, **kwargs)
        path = getattr(source, "path", source)
        if Path(path).name == "train.csv":
            _write_csv(
                Path(path),
                [
                    ["row_id", "feature", "control_success"],
                    ["replacement-1", "a", "1"],
                    ["replacement-2", "b", "0"],
                    ["replacement-3", "c", "1"],
                ],
            )
        return parsed

    monkeypatch.setattr(module, "_read_csv", replace_after_parse)
    with pytest.raises(PortfolioInputError, match="changed"):
        prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")


def test_prepare_rejects_same_content_symlink_swap_after_copy(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    original = module._copy_source
    original_train = (tiny_official_dir / "train.csv").read_bytes()
    replacement = tmp_path / "same-train.csv"
    replacement.write_bytes(original_train)

    def swap_after_copy(source: Path, destination: object, **kwargs: object) -> None:
        original(source, destination, **kwargs)
        if source.name == "train.csv":
            source.unlink()
            source.symlink_to(replacement)

    monkeypatch.setattr(module, "_copy_source", swap_after_copy)
    with pytest.raises(PortfolioInputError, match="symlink"):
        prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")


def test_prepare_rejects_contract_bytes_that_fail_sealed_validation(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    original = Path.read_bytes

    def altered_contract_bytes(path: Path) -> bytes:
        if path == Path(module.__file__).with_name("contract.json"):
            return b"{}"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", altered_contract_bytes)
    with pytest.raises(PortfolioInputError, match="contract"):
        prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")


def test_prepare_wraps_staging_creation_failure(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    def broken_mkdtemp(**kwargs: object) -> str:
        raise OSError("no staging directory")

    monkeypatch.setattr(module, "mkdtemp", broken_mkdtemp)
    output = tmp_path / "input.zip"
    with pytest.raises(PortfolioInputError, match="staging"):
        prepare_input_archive(tiny_official_dir, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".temporal-portfolio-stage-*"))


@pytest.mark.parametrize(
    ("name", "rows", "message"),
    [
        ("train.csv", [["row_id", "", "control_success"], ["a", "x", "1"]], "blank header"),
        ("test.csv", [["row_id", "row_id"], ["test-1", "x"], ["test-2", "y"]], "duplicate headers"),
        ("trackman_history.csv", [["pitcher_id"]], "no data rows"),
    ],
)
def test_verify_rejects_csv_header_and_history_boundaries(
    tiny_official_dir: Path, name: str, rows: list[list[str]], message: str
) -> None:
    _write_csv(tiny_official_dir / name, rows)

    with pytest.raises(PortfolioInputError, match=message):
        verify_official_data(tiny_official_dir)


def test_verify_rejects_train_count_not_greater_than_test(tiny_official_dir: Path) -> None:
    _write_csv(
        tiny_official_dir / "train.csv",
        [["row_id", "feature", "control_success"], ["train-1", "a", "1"], ["train-2", "b", "0"]],
    )

    with pytest.raises(PortfolioInputError, match="row count"):
        verify_official_data(tiny_official_dir)


def test_prepare_uses_fixed_zip_metadata_and_cleans_atomic_failure(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")
    with ZipFile(result.path) as archive:
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert (info.external_attr >> 16) & 0o777 == 0o644

    import experiments.temporal_portfolio.inputs as module

    def fail_copy(source: Path, destination: object, **kwargs: object) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(module, "_copy_source", fail_copy)
    failed = tmp_path / "failed.zip"
    with pytest.raises(PortfolioInputError, match="copy failed"):
        prepare_input_archive(tiny_official_dir, failed)
    assert not failed.exists()
    assert not list(tmp_path.glob(".temporal-portfolio-input-*"))


def test_prepare_cli_succeeds_from_outside_repository(
    tmp_path: Path, tiny_official_dir: Path
) -> None:
    script = Path(__file__).parents[1] / "tools/prepare_temporal_portfolio_input.py"
    output = tmp_path / "input.zip"
    completed = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tiny_official_dir), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert re.fullmatch(
        rf"TEMPORAL_INPUT_READY path={re.escape(str(output.absolute()))} "
        r"sha256=[0-9a-f]{64} size_bytes=[0-9]+\n",
        completed.stdout,
    )


def test_prepare_rejects_same_content_new_train_inode_after_baseline(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    original = module._canonical_manifest
    train = tiny_official_dir / "train.csv"
    replacement = tmp_path / "replacement-train.csv"

    def replace_after_baseline(verified: object) -> bytes:
        manifest = original(verified)
        replacement.write_bytes(train.read_bytes())
        os.replace(replacement, train)
        return manifest

    monkeypatch.setattr(module, "_canonical_manifest", replace_after_baseline)
    with pytest.raises(PortfolioInputError, match="changed"):
        prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")


def test_prepare_rejects_temp_path_swap_after_archive_hash(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    original = module.file_sha256

    def swap_after_hash(path: Path, **kwargs: object) -> str:
        digest = original(path, **kwargs)
        value = Path(path)
        if value.name.startswith(".temporal-portfolio-input-"):
            value.unlink()
            value.write_bytes(b"arbitrary replacement")
        return digest

    monkeypatch.setattr(module, "file_sha256", swap_after_hash)
    output = tmp_path / "input.zip"
    with pytest.raises(PortfolioInputError, match="temporary|publication|changed"):
        prepare_input_archive(tiny_official_dir, output)
    assert not output.exists()


def test_prepare_preserves_concurrently_created_output_without_replace(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    output = tmp_path / "input.zip"
    original = module._canonical_manifest

    def create_concurrent_output(verified: object) -> bytes:
        manifest = original(verified)
        output.write_bytes(b"concurrent owner")
        return manifest

    monkeypatch.setattr(module, "_canonical_manifest", create_concurrent_output)
    with pytest.raises(PortfolioInputError, match="output|publish"):
        prepare_input_archive(tiny_official_dir, output)
    assert output.read_bytes() == b"concurrent owner"


def test_verify_rejects_blank_row_id_and_sample_count_mismatch(
    tiny_official_dir: Path,
) -> None:
    _write_csv(
        tiny_official_dir / "test.csv",
        [["row_id", "feature"], ["", "x"], ["test-2", "y"]],
    )
    with pytest.raises(PortfolioInputError, match="blank row_id"):
        verify_official_data(tiny_official_dir)

    _write_csv(tiny_official_dir / "test.csv", [["row_id", "feature"], ["test-1", "x"], ["test-2", "y"]])
    _write_csv(tiny_official_dir / "sample_submission.csv", [["row_id", "control_success"], ["test-1", "0.5"]])
    with pytest.raises(PortfolioInputError, match="row_id sequence"):
        verify_official_data(tiny_official_dir)


def test_archive_verifier_rejects_duplicate_tampered_member(
    tmp_path: Path, tiny_official_dir: Path
) -> None:
    import experiments.temporal_portfolio.inputs as module

    result = prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")
    verified = verify_official_data(tiny_official_dir)
    manifest = module._canonical_manifest(verified)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(result.path, "a") as archive:
            archive.writestr("data/train.csv", b"tampered")

    with pytest.raises(PortfolioInputError, match="members|evidence"):
        module._verify_archive(result.path, manifest, verified)


@pytest.mark.parametrize(
    "name", ["train.csv", "test.csv", "trackman_history.csv", "sample_submission.csv"]
)
def test_prepare_rejects_output_equal_to_official_source_even_with_replace(
    tiny_official_dir: Path, name: str
) -> None:
    output = tiny_official_dir / name
    original = output.read_bytes()

    with pytest.raises(PortfolioInputError, match="official data"):
        prepare_input_archive(tiny_official_dir, output, replace=True)
    assert output.read_bytes() == original


def test_prepare_rejects_output_nested_inside_official_data_root(
    tiny_official_dir: Path,
) -> None:
    output = tiny_official_dir / "nested" / "input.zip"

    with pytest.raises(PortfolioInputError, match="official data"):
        prepare_input_archive(tiny_official_dir, output)
    assert not output.exists()


def test_verify_accepts_intermediate_symlink_alias_but_rejects_root_leaf_symlink(
    tmp_path: Path, tiny_official_dir: Path
) -> None:
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(tmp_path, target_is_directory=True)
    via_intermediate_alias = alias_parent / tiny_official_dir.name

    assert verify_official_data(via_intermediate_alias).root == tiny_official_dir.resolve()

    leaf_alias = tmp_path / "official-leaf-alias"
    leaf_alias.symlink_to(tiny_official_dir, target_is_directory=True)
    with pytest.raises(PortfolioInputError, match="symlink"):
        verify_official_data(leaf_alias)


@pytest.mark.parametrize("replace", [False, True])
def test_post_commit_staging_cleanup_failure_does_not_change_success(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch, replace: bool
) -> None:
    import experiments.temporal_portfolio.inputs as module

    output = tmp_path / "input.zip"
    if replace:
        output.write_bytes(b"old output")

    def fail_cleanup(staging: Path) -> None:
        raise OSError("cleanup denied")

    monkeypatch.setattr(module, "_cleanup_staging", fail_cleanup)
    prepared = prepare_input_archive(tiny_official_dir, output, replace=replace)
    assert prepared.path == output.absolute()
    with ZipFile(output) as archive:
        assert archive.namelist()[0] == "manifest.json"


def test_no_file_validation_runs_after_atomic_commit(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    committed = False
    original_link = module.os.link
    original_hash = module.file_sha256

    def mark_commit(*args: object, **kwargs: object) -> None:
        nonlocal committed
        original_link(*args, **kwargs)
        committed = True

    def reject_post_commit_hash(path: Path, **kwargs: object) -> str:
        if committed:
            pytest.fail("no file validation may run after publication")
        return original_hash(path, **kwargs)

    monkeypatch.setattr(module.os, "link", mark_commit)
    monkeypatch.setattr(module, "file_sha256", reject_post_commit_hash)
    prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")


def test_csv_data_does_not_retain_row_id_tuples_and_still_checks_alignment(
    tiny_official_dir: Path,
) -> None:
    import experiments.temporal_portfolio.inputs as module

    assert "row_ids" not in module._CsvData.__dataclass_fields__
    _write_csv(
        tiny_official_dir / "sample_submission.csv",
        [["row_id", "control_success"], ["test-2", "0.5"], ["test-1", "0.5"]],
    )
    with pytest.raises(PortfolioInputError, match="row_id sequence"):
        verify_official_data(tiny_official_dir)


def test_prepare_does_not_call_testzip(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    def testzip_must_not_run(self: ZipFile) -> str:
        pytest.fail("testzip must not run")

    monkeypatch.setattr(module.ZipFile, "testzip", testzip_must_not_run)
    prepare_input_archive(tiny_official_dir, tmp_path / "input.zip")


def test_unsupported_hardlink_fails_before_no_replace_commit(
    tmp_path: Path, tiny_official_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.temporal_portfolio.inputs as module

    def unsupported_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOTSUP, "hardlink unsupported")

    monkeypatch.setattr(module.os, "link", unsupported_link)
    output = tmp_path / "input.zip"
    with pytest.raises(PortfolioInputError, match="publish"):
        prepare_input_archive(tiny_official_dir, output)
    assert not output.exists()
