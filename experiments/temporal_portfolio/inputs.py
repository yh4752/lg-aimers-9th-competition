"""Fail-closed preparation of the temporal portfolio's official inputs."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import csv
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
from tempfile import mkdtemp
from types import MappingProxyType
from typing import BinaryIO, Iterator, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo
import zlib

from .contracts import PortfolioContractError, _SEALED_CONTRACT_SHA256, load_contract


class PortfolioInputError(ValueError):
    """Raised when an official data input or prepared archive is invalid."""


@dataclass(frozen=True)
class _FileVersion:
    dev: int
    ino: int
    size_bytes: int
    mtime_ns: int
    sha256: str


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
    member_sizes: Mapping[str, int] = field(repr=False, compare=False)
    member_versions: Mapping[str, _FileVersion] = field(repr=False, compare=False)


@dataclass(frozen=True)
class PreparedInputArchive:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _CsvData:
    header: tuple[str, ...]
    row_count: int
    row_id_sequence_sha256: str | None
    sha256: str
    size_bytes: int
    version: _FileVersion


@dataclass(frozen=True)
class _StagingDirectory:
    path: Path
    dev: int
    ino: int


@dataclass
class _SafeSource:
    path: Path
    handle: BinaryIO
    initial_stat: os.stat_result

    def require_unchanged(self, label: str) -> None:
        try:
            current = os.fstat(self.handle.fileno())
        except OSError as error:
            raise PortfolioInputError(f"cannot inspect {label}: {error}") from error
        if not _same_file_stat(self.initial_stat, current):
            raise PortfolioInputError(f"{label} changed while it was being read")


class _HashingReader(io.RawIOBase):
    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self.digest = sha256()
        self.size_bytes = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        data = self._source.read(len(buffer))
        if not data:
            return 0
        size = len(data)
        buffer[:size] = data
        self.digest.update(data)
        self.size_bytes += size
        return size


class _HashingWriter:
    def __init__(self, destination: BinaryIO) -> None:
        self._destination = destination
        self.digest = sha256()
        self.size_bytes = 0

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.size_bytes += len(data)
        return self._destination.write(data)


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


def _same_file_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISREG(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _matches_version(metadata: os.stat_result, version: _FileVersion) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_dev == version.dev
        and metadata.st_ino == version.ino
        and metadata.st_size == version.size_bytes
        and metadata.st_mtime_ns == version.mtime_ns
    )


def _version_from_stat(metadata: os.stat_result, digest: str) -> _FileVersion:
    return _FileVersion(
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        digest,
    )


def _canonical_data_root(root: str | Path) -> Path:
    requested = Path(root)
    try:
        if requested.is_symlink():
            raise PortfolioInputError("official data directory must not be a symlink")
        canonical = requested.resolve(strict=True)
        if not stat.S_ISDIR(canonical.lstat().st_mode):
            raise PortfolioInputError("official data directory must be a real directory")
        return canonical
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect official data directory: {error}") from error


def _canonical_destination(output: str | Path) -> Path:
    requested = Path(output)
    try:
        if not requested.name or requested.name in (".", ".."):
            raise PortfolioInputError("output must name a file")
        return requested.resolve(strict=False)
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot canonicalize output path: {error}") from error


def _require_output_outside_data(destination: Path, verified: VerifiedOfficialData) -> None:
    if destination == verified.root or destination.is_relative_to(verified.root):
        raise PortfolioInputError("output must be outside the official data directory")


@contextmanager
def _safe_source(
    path: Path, label: str, *, expected: _FileVersion | None = None
) -> Iterator[_SafeSource]:
    descriptor: int | None = None
    handle: BinaryIO | None = None
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            raise PortfolioInputError(f"{label} must be a regular non-symlink file")
        if expected is not None and not _matches_version(before, expected):
            raise PortfolioInputError(f"{label} changed before it could be opened")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if getattr(os, "O_NOFOLLOW", 0):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not _same_file_stat(before, opened) or (
            expected is not None and not _matches_version(opened, expected)
        ):
            raise PortfolioInputError(f"{label} changed before it could be opened")
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = None
        yield _SafeSource(path, handle, opened)
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot safely open {label}: {error}") from error
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError as error:
                raise PortfolioInputError(f"cannot close {label}: {error}") from error
        elif descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def file_sha256(path: str | Path, *, expected: _FileVersion | None = None) -> str:
    source_path = Path(path)
    with _safe_source(source_path, str(source_path), expected=expected) as source:
        digest = sha256()
        for chunk in iter(lambda: source.handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
        source.require_unchanged(str(source_path))
        value = digest.hexdigest()
        if expected is not None and value != expected.sha256:
            raise PortfolioInputError(f"{source_path} content differs from verified bytes")
        return value


def _update_row_id_digest(digest: object, row_id: str) -> None:
    encoded = row_id.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _read_csv(
    source: _SafeSource,
    label: str,
    *,
    collect_row_ids: bool,
    validate_sample_values: bool = False,
) -> _CsvData:
    hashing_reader = _HashingReader(source.handle)
    text: io.TextIOWrapper | None = None
    try:
        text = io.TextIOWrapper(io.BufferedReader(hashing_reader), encoding="utf-8-sig", newline="")
        reader = csv.reader(text, strict=True)
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
        if validate_sample_values and tuple(header) != ("row_id", "control_success"):
            raise PortfolioInputError("sample_submission.csv header differs")
        row_id_index = header.index("row_id") if collect_row_ids else None
        target_index = header.index("control_success") if validate_sample_values else None
        seen: set[str] | None = set() if collect_row_ids else None
        sequence = sha256() if collect_row_ids else None
        row_count = 0
        for row in reader:
            if not row or len(row) != len(header):
                raise PortfolioInputError(f"{label} has a malformed row")
            row_count += 1
            if row_id_index is not None:
                row_id = row[row_id_index]
                if not row_id.strip():
                    raise PortfolioInputError(f"{label} has a blank row_id")
                if row_id in seen:
                    raise PortfolioInputError(f"{label} has a duplicate row_id")
                seen.add(row_id)
                _update_row_id_digest(sequence, row_id)
            if target_index is not None:
                try:
                    value = float(row[target_index])
                except (ValueError, OverflowError) as error:
                    raise PortfolioInputError("sample_submission control_success must be finite numeric") from error
                if not math.isfinite(value):
                    raise PortfolioInputError("sample_submission control_success must be finite numeric")
        if row_count == 0:
            raise PortfolioInputError(f"{label} has no data rows")
        text.close()
        text = None
        if hashing_reader.size_bytes != source.initial_stat.st_size:
            raise PortfolioInputError(f"{label} changed while it was being parsed")
        digest = hashing_reader.digest.hexdigest()
        return _CsvData(
            tuple(header), row_count, sequence.hexdigest() if sequence else None,
            digest, hashing_reader.size_bytes, _version_from_stat(source.initial_stat, digest)
        )
    except PortfolioInputError:
        raise
    except (csv.Error, UnicodeError, OSError, ValueError) as error:
        raise PortfolioInputError(f"cannot read {label}: {error}") from error
    finally:
        if text is not None:
            try:
                text.close()
            except (OSError, UnicodeError) as error:
                raise PortfolioInputError(f"cannot close {label}: {error}") from error


def _parse_csv_member(
    path: Path, label: str, *, collect_row_ids: bool, validate_sample_values: bool = False
) -> _CsvData:
    with _safe_source(path, label) as source:
        data = _read_csv(source, label, collect_row_ids=collect_row_ids, validate_sample_values=validate_sample_values)
        source.require_unchanged(label)
        return data


def verify_official_data(root: str | Path) -> VerifiedOfficialData:
    data_root = _canonical_data_root(root)
    try:
        entries = list(data_root.iterdir())
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect official data directory: {error}") from error
    if {entry.name for entry in entries} != set(_OFFICIAL_NAMES) or len(entries) != len(_OFFICIAL_NAMES):
        raise PortfolioInputError("official data directory top-level members differ")
    members = {name: data_root / name for name in _OFFICIAL_NAMES}
    train = _parse_csv_member(members["train.csv"], "train.csv", collect_row_ids=True)
    test = _parse_csv_member(members["test.csv"], "test.csv", collect_row_ids=True)
    history = _parse_csv_member(members["trackman_history.csv"], "trackman_history.csv", collect_row_ids=False)
    sample = _parse_csv_member(members["sample_submission.csv"], "sample_submission.csv", collect_row_ids=True, validate_sample_values=True)
    if "control_success" not in train.header:
        raise PortfolioInputError("train.csv must contain row_id and control_success")
    if "control_success" in test.header:
        raise PortfolioInputError("test.csv must not contain control_success")
    if train.row_count <= test.row_count:
        raise PortfolioInputError("train.csv row count must exceed test.csv row count")
    if sample.row_count != test.row_count or sample.row_id_sequence_sha256 != test.row_id_sequence_sha256:
        raise PortfolioInputError("sample_submission.csv row_id sequence differs from test.csv")
    data_by_name = {"train.csv": train, "test.csv": test, "trackman_history.csv": history, "sample_submission.csv": sample}
    return VerifiedOfficialData(
        data_root, members["train.csv"], members["test.csv"], members["trackman_history.csv"],
        members["sample_submission.csv"],
        MappingProxyType({name: data.sha256 for name, data in data_by_name.items()}),
        train.row_count, test.row_count,
        MappingProxyType({name: data.size_bytes for name, data in data_by_name.items()}),
        MappingProxyType({name: data.version for name, data in data_by_name.items()}),
    )


def _copy_source(source: Path, destination: BinaryIO, *, expected: _FileVersion | None = None) -> None:
    with _safe_source(source, source.name, expected=expected) as opened:
        for chunk in iter(lambda: opened.handle.read(_CHUNK_SIZE), b""):
            destination.write(chunk)
        opened.require_unchanged(source.name)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_DATE)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _sealed_contract_bytes() -> bytes:
    path = Path(__file__).with_name("contract.json")
    try:
        load_contract(path)
        raw = path.read_bytes()
    except PortfolioContractError as error:
        raise PortfolioInputError(f"temporal portfolio contract is invalid: {error}") from error
    except OSError as error:
        raise PortfolioInputError(f"cannot read contract.json: {error}") from error
    if sha256(raw).hexdigest() != _SEALED_CONTRACT_SHA256:
        raise PortfolioInputError("temporal portfolio contract bytes differ")
    return raw


def _canonical_manifest(verified: VerifiedOfficialData) -> bytes:
    members = {
        archive: {"size": verified.member_sizes[source], "sha256": verified.member_sha256[source]}
        for archive, source in _ARCHIVE_SOURCES
    }
    payload = {
        "schema_version": 1, "artifact_kind": "temporal_portfolio_input_v1",
        "campaign_id": "temporal_portfolio_v1", "submission_package": False,
        "contract_sha256": sha256(_sealed_contract_bytes()).hexdigest(),
        "train_rows": verified.train_rows, "test_rows": verified.test_rows, "members": members,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _verify_archive(path: Path, manifest_bytes: bytes, verified: VerifiedOfficialData, *, expected: _FileVersion | None = None) -> None:
    try:
        with _safe_source(path, "prepared staging ZIP", expected=expected) as source:
            with ZipFile(source.handle) as archive:
                if archive.namelist() != list(_ARCHIVE_NAMES):
                    raise PortfolioInputError("prepared ZIP members differ")
                if archive.read("manifest.json") != manifest_bytes:
                    raise PortfolioInputError("prepared ZIP manifest bytes differ")
                manifest = json.loads(manifest_bytes)
                if type(manifest) is not dict or type(manifest.get("members")) is not dict:
                    raise PortfolioInputError("prepared ZIP manifest schema differs")
                for archive_name, source_name in _ARCHIVE_SOURCES:
                    digest = sha256()
                    info = archive.getinfo(archive_name)
                    with archive.open(info) as member:
                        for chunk in iter(lambda: member.read(_CHUNK_SIZE), b""):
                            digest.update(chunk)
                    evidence = manifest["members"].get(archive_name)
                    if (
                        info.file_size != verified.member_sizes[source_name]
                        or type(evidence) is not dict
                        or evidence.get("size") != info.file_size
                        or evidence.get("sha256") != digest.hexdigest()
                        or digest.hexdigest() != verified.member_sha256[source_name]
                    ):
                        raise PortfolioInputError("prepared ZIP member evidence differs")
            source.require_unchanged("prepared staging ZIP")
    except PortfolioInputError:
        raise
    except (
        BadZipFile,
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        zlib.error,
    ) as error:
        raise PortfolioInputError(f"cannot verify prepared ZIP: {error}") from error


def _version_from_file(path: Path, label: str) -> _FileVersion:
    with _safe_source(path, label) as source:
        digest = sha256()
        for chunk in iter(lambda: source.handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
        source.require_unchanged(label)
        return _version_from_stat(source.initial_stat, digest.hexdigest())


def _require_path_version(path: Path, version: _FileVersion, label: str) -> None:
    try:
        if not _matches_version(os.lstat(path), version):
            raise PortfolioInputError(f"{label} changed after verification")
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect {label}: {error}") from error


def _create_staging(parent: Path) -> _StagingDirectory:
    try:
        staging = Path(mkdtemp(prefix=".temporal-portfolio-stage-", dir=parent))
        metadata = staging.lstat()
    except OSError as error:
        raise PortfolioInputError(f"cannot create staging directory: {error}") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        _cleanup_staging(staging)
        raise PortfolioInputError("staging directory permissions differ")
    return _StagingDirectory(staging, metadata.st_dev, metadata.st_ino)


def _require_staging(staging: _StagingDirectory) -> Path:
    try:
        metadata = staging.path.lstat()
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect staging directory: {error}") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (metadata.st_dev, metadata.st_ino) != (staging.dev, staging.ino)
    ):
        raise PortfolioInputError("staging directory identity differs")
    return staging.path


def _cleanup_staging(staging: Path) -> None:
    try:
        shutil.rmtree(staging)
    except OSError:
        pass


def _best_effort_cleanup(staging: _StagingDirectory | None) -> None:
    if staging is None:
        return
    try:
        _cleanup_staging(staging.path)
    except OSError:
        pass


def _canonical_output_parent(destination: Path) -> Path:
    try:
        parent = destination.parent.resolve(strict=True)
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise PortfolioInputError("output parent must be a real directory")
        return parent
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect output path: {error}") from error


def prepare_input_archive(data_dir: str | Path, output: str | Path, *, replace: bool = False) -> PreparedInputArchive:
    verified = verify_official_data(data_dir)
    destination = _canonical_destination(output)
    _require_output_outside_data(destination, verified)
    parent = _canonical_output_parent(destination)
    destination = parent / destination.name
    try:
        if os.path.lexists(destination) and destination.is_symlink():
            raise PortfolioInputError("output must not be a symlink")
        if os.path.lexists(destination) and not replace:
            raise PortfolioInputError("output already exists; pass replace=True to replace it")
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect output path: {error}") from error

    staging: _StagingDirectory | None = _create_staging(parent)
    committed = False
    try:
        archive_path = _require_staging(staging) / ".temporal-portfolio-input-archive.zip"
        descriptor = os.open(archive_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w+b", buffering=0) as raw:
            with ZipFile(raw, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
                manifest_bytes = _canonical_manifest(verified)
                with archive.open(_zip_info("manifest.json"), "w") as target:
                    target.write(manifest_bytes)
                sources = {"train.csv": verified.train, "test.csv": verified.test, "trackman_history.csv": verified.history, "sample_submission.csv": verified.sample_submission}
                for archive_name, source_name in _ARCHIVE_SOURCES:
                    with archive.open(_zip_info(archive_name), "w") as target:
                        writer = _HashingWriter(target)
                        _copy_source(sources[source_name], writer, expected=verified.member_versions[source_name])
                        if writer.size_bytes != verified.member_sizes[source_name] or writer.digest.hexdigest() != verified.member_sha256[source_name]:
                            raise PortfolioInputError(f"official data source changed before ZIP copy: {source_name}")
            raw.flush()
            try:
                os.fsync(raw.fileno())
            except OSError:
                pass
        archive_version = _version_from_file(archive_path, "prepared staging ZIP")
        _verify_archive(archive_path, manifest_bytes, verified, expected=archive_version)
        for name, source in sources.items():
            file_sha256(source, expected=verified.member_versions[name])
        archive_sha256 = file_sha256(archive_path, expected=archive_version)
        _require_path_version(archive_path, archive_version, "prepared staging ZIP")
        prepared = PreparedInputArchive(destination, archive_sha256, archive_version.size_bytes)
        if replace:
            os.replace(archive_path, destination)
        else:
            try:
                os.link(archive_path, destination, follow_symlinks=False)
            except FileExistsError as error:
                raise PortfolioInputError("output already exists during publication") from error
            except OSError as error:
                raise PortfolioInputError(f"cannot publish input archive: {error}") from error
        committed = True
    except PortfolioInputError:
        raise
    except (BadZipFile, OSError) as error:
        raise PortfolioInputError(f"cannot prepare input archive: {error}") from error
    finally:
        if not committed:
            _best_effort_cleanup(staging)
    _best_effort_cleanup(staging)
    return prepared
