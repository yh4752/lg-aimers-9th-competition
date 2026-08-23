"""Fail-closed preparation of the temporal portfolio's official inputs."""
from __future__ import annotations

from dataclasses import dataclass
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from tempfile import mkstemp
from types import MappingProxyType
from typing import BinaryIO, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo


class PortfolioInputError(ValueError):
    """Raised when an official data input or prepared archive is invalid."""


@dataclass(frozen=True)
class VerifiedOfficialData:
    root: Path
    train: Path
    test: Path
    history: Path
    sample_submission: Path
    member_sha256: Mapping[str, str]
    train_rows: int
    test_rows: int


@dataclass(frozen=True)
class PreparedInputArchive:
    path: Path
    sha256: str
    size_bytes: int


_OFFICIAL_NAMES = (
    "train.csv",
    "test.csv",
    "trackman_history.csv",
    "sample_submission.csv",
)
_ARCHIVE_SOURCES = (
    ("data/train.csv", "train.csv"),
    ("data/test.csv", "test.csv"),
    ("data/trackman_history.csv", "trackman_history.csv"),
    ("data/sample_submission.csv", "sample_submission.csv"),
)
_ARCHIVE_NAMES = ("manifest.json", *(name for name, _ in _ARCHIVE_SOURCES))
_CHUNK_SIZE = 1024 * 1024
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class _CsvData:
    header: tuple[str, ...]
    row_ids: tuple[str, ...]
    row_count: int


def _absolute_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path.cwd() / value


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    try:
        for candidate in (path, *path.parents):
            if candidate.is_symlink():
                raise PortfolioInputError(f"{label} has a symlink ancestor")
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect {label}: {error}") from error


def _require_regular_file(path: Path, label: str) -> None:
    try:
        if path.is_symlink():
            raise PortfolioInputError(f"{label} must not be a symlink")
        mode = path.lstat().st_mode
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISREG(mode):
        raise PortfolioInputError(f"{label} must be a regular file")


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 for a regular file."""
    value = Path(path)
    try:
        digest = sha256()
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise PortfolioInputError(f"cannot hash {value}: {error}") from error


def _read_csv(path: Path, label: str, *, collect_row_ids: bool) -> _CsvData:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as error:
                raise PortfolioInputError(f"{label} is empty") from error
            if not header or any(not cell.strip() for cell in header):
                raise PortfolioInputError(f"{label} has a blank header")
            if len(set(header)) != len(header):
                raise PortfolioInputError(f"{label} has duplicate headers")
            if collect_row_ids and "row_id" not in header:
                raise PortfolioInputError(f"{label} must contain row_id")
            row_id_index = header.index("row_id") if collect_row_ids else None
            row_ids: list[str] = []
            row_count = 0
            for row in reader:
                if not row or len(row) != len(header):
                    raise PortfolioInputError(f"{label} has a malformed row")
                row_count += 1
                if row_id_index is not None:
                    row_id = row[row_id_index]
                    if not row_id.strip():
                        raise PortfolioInputError(f"{label} has a blank row_id")
                    row_ids.append(row_id)
            if row_count == 0:
                raise PortfolioInputError(f"{label} has no data rows")
            return _CsvData(tuple(header), tuple(row_ids), row_count)
    except PortfolioInputError:
        raise
    except (csv.Error, UnicodeError, OSError) as error:
        raise PortfolioInputError(f"cannot read {label}: {error}") from error


def _require_unique_row_ids(data: _CsvData, label: str) -> None:
    if len(set(data.row_ids)) != len(data.row_ids):
        raise PortfolioInputError(f"{label} has a duplicate row_id")


def _validate_sample_values(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            target_index = header.index("control_success")
            for row in reader:
                try:
                    value = float(row[target_index])
                except (ValueError, OverflowError) as error:
                    raise PortfolioInputError(
                        "sample_submission control_success must be finite numeric"
                    ) from error
                if not math.isfinite(value):
                    raise PortfolioInputError(
                        "sample_submission control_success must be finite numeric"
                    )
    except PortfolioInputError:
        raise
    except (csv.Error, UnicodeError, OSError, StopIteration, ValueError) as error:
        raise PortfolioInputError(f"cannot read sample_submission.csv: {error}") from error


def verify_official_data(root: str | Path) -> VerifiedOfficialData:
    """Validate the exact, top-level official data directory without discovery."""
    data_root = _absolute_path(root)
    _require_no_symlink_ancestors(data_root, "official data directory")
    try:
        if not stat.S_ISDIR(data_root.lstat().st_mode):
            raise PortfolioInputError("official data directory must be a real directory")
        entries = list(data_root.iterdir())
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect official data directory: {error}") from error
    if {entry.name for entry in entries} != set(_OFFICIAL_NAMES) or len(entries) != len(
        _OFFICIAL_NAMES
    ):
        raise PortfolioInputError("official data directory top-level members differ")

    members = {name: data_root / name for name in _OFFICIAL_NAMES}
    for name, path in members.items():
        _require_regular_file(path, name)

    train = _read_csv(members["train.csv"], "train.csv", collect_row_ids=True)
    test = _read_csv(members["test.csv"], "test.csv", collect_row_ids=True)
    history = _read_csv(
        members["trackman_history.csv"], "trackman_history.csv", collect_row_ids=False
    )
    sample = _read_csv(
        members["sample_submission.csv"], "sample_submission.csv", collect_row_ids=True
    )
    if "row_id" not in train.header or "control_success" not in train.header:
        raise PortfolioInputError("train.csv must contain row_id and control_success")
    if "row_id" not in test.header:
        raise PortfolioInputError("test.csv must contain row_id")
    if "control_success" in test.header:
        raise PortfolioInputError("test.csv must not contain control_success")
    if sample.header != ("row_id", "control_success"):
        raise PortfolioInputError("sample_submission.csv header differs")
    for data, label in ((train, "train.csv"), (test, "test.csv"), (sample, "sample_submission.csv")):
        _require_unique_row_ids(data, label)
    if train.row_count <= test.row_count:
        raise PortfolioInputError("train.csv row count must exceed test.csv row count")
    if sample.row_count != test.row_count or sample.row_ids != test.row_ids:
        raise PortfolioInputError("sample_submission.csv row_id sequence differs from test.csv")
    _validate_sample_values(members["sample_submission.csv"])

    hashes = {name: file_sha256(path) for name, path in members.items()}
    return VerifiedOfficialData(
        root=data_root,
        train=members["train.csv"],
        test=members["test.csv"],
        history=members["trackman_history.csv"],
        sample_submission=members["sample_submission.csv"],
        member_sha256=MappingProxyType(hashes),
        train_rows=train.row_count,
        test_rows=test.row_count,
    )


def _copy_source(source: Path, destination: BinaryIO) -> None:
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            destination.write(chunk)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_DATE)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _canonical_manifest(verified: VerifiedOfficialData, member_sizes: Mapping[str, int]) -> bytes:
    contract_path = Path(__file__).with_name("contract.json")
    try:
        contract_bytes = contract_path.read_bytes()
    except OSError as error:
        raise PortfolioInputError(f"cannot read contract.json: {error}") from error
    members = {
        archive_name: {
            "size": member_sizes[archive_name],
            "sha256": verified.member_sha256[source_name],
        }
        for archive_name, source_name in _ARCHIVE_SOURCES
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "temporal_portfolio_input_v1",
        "campaign_id": "temporal_portfolio_v1",
        "submission_package": False,
        "contract_sha256": sha256(contract_bytes).hexdigest(),
        "train_rows": verified.train_rows,
        "test_rows": verified.test_rows,
        "members": members,
    }
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _verify_archive(path: Path, manifest_bytes: bytes, member_sizes: Mapping[str, int]) -> None:
    try:
        with ZipFile(path) as archive:
            if archive.namelist() != list(_ARCHIVE_NAMES):
                raise PortfolioInputError("prepared ZIP members differ")
            if archive.testzip() is not None:
                raise PortfolioInputError("prepared ZIP has an invalid CRC")
            if archive.read("manifest.json") != manifest_bytes:
                raise PortfolioInputError("prepared ZIP manifest bytes differ")
            manifest = json.loads(manifest_bytes)
            if (
                type(manifest) is not dict
                or manifest.get("schema_version") != 1
                or manifest.get("artifact_kind") != "temporal_portfolio_input_v1"
                or manifest.get("campaign_id") != "temporal_portfolio_v1"
                or manifest.get("submission_package") is not False
                or type(manifest.get("members")) is not dict
            ):
                raise PortfolioInputError("prepared ZIP manifest schema differs")
            for archive_name, source_name in _ARCHIVE_SOURCES:
                info = archive.getinfo(archive_name)
                if info.file_size != member_sizes[archive_name]:
                    raise PortfolioInputError("prepared ZIP member size differs")
                digest = sha256()
                with archive.open(info) as handle:
                    for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                        digest.update(chunk)
                evidence = manifest["members"].get(archive_name)
                if (
                    type(evidence) is not dict
                    or evidence.get("size") != info.file_size
                    or evidence.get("sha256") != digest.hexdigest()
                ):
                    raise PortfolioInputError("prepared ZIP member evidence differs")
    except PortfolioInputError:
        raise
    except (BadZipFile, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise PortfolioInputError(f"cannot verify prepared ZIP: {error}") from error


def _validate_output_path(output: str | Path, replace: bool) -> Path:
    destination = _absolute_path(output)
    parent = destination.parent
    _require_no_symlink_ancestors(parent, "output directory")
    try:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise PortfolioInputError("output parent must be a real directory")
        exists = os.path.lexists(destination)
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect output path: {error}") from error
    if exists and not replace:
        raise PortfolioInputError("output already exists; pass replace=True to replace it")
    return destination


def prepare_input_archive(
    data_dir: str | Path, output: str | Path, *, replace: bool = False
) -> PreparedInputArchive:
    """Create a deterministic, verified input-only archive from official data."""
    verified = verify_official_data(data_dir)
    destination = _validate_output_path(output, replace)
    sources = {
        "train.csv": verified.train,
        "test.csv": verified.test,
        "trackman_history.csv": verified.history,
        "sample_submission.csv": verified.sample_submission,
    }
    try:
        member_sizes = {
            archive_name: sources[source_name].stat().st_size
            for archive_name, source_name in _ARCHIVE_SOURCES
        }
    except OSError as error:
        raise PortfolioInputError(f"cannot stat official data member: {error}") from error
    manifest_bytes = _canonical_manifest(verified, member_sizes)
    descriptor, temporary_name = mkstemp(
        prefix=".temporal-portfolio-input-", suffix=".zip", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            with archive.open(_zip_info("manifest.json"), "w") as target:
                target.write(manifest_bytes)
            for archive_name, source_name in _ARCHIVE_SOURCES:
                with archive.open(_zip_info(archive_name), "w") as target:
                    _copy_source(sources[source_name], target)
        _verify_archive(temporary, manifest_bytes, member_sizes)
        for source_name, source in sources.items():
            if file_sha256(source) != verified.member_sha256[source_name]:
                raise PortfolioInputError("official data source changed during archive preparation")
        os.replace(temporary, destination)
        final_size = destination.stat().st_size
        return PreparedInputArchive(destination, file_sha256(destination), final_size)
    except PortfolioInputError:
        raise
    except (BadZipFile, OSError) as error:
        raise PortfolioInputError(f"cannot prepare input archive: {error}") from error
    finally:
        try:
            if os.path.lexists(temporary):
                temporary.unlink()
        except OSError:
            pass
