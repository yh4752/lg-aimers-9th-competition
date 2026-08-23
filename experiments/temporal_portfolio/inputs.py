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
import stat
from tempfile import mkstemp
from types import MappingProxyType
from typing import BinaryIO, Iterator, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .contracts import PortfolioContractError, _SEALED_CONTRACT_SHA256, load_contract


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
    member_sizes: Mapping[str, int] = field(repr=False, compare=False)
    member_versions: Mapping[str, "_FileVersion"] = field(repr=False, compare=False)


@dataclass(frozen=True)
class PreparedInputArchive:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _FileVersion:
    dev: int
    ino: int
    size_bytes: int
    mtime_ns: int
    sha256: str


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
    sha256: str
    size_bytes: int
    version: _FileVersion


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
        if not _same_file_version(self.initial_stat, current):
            raise PortfolioInputError(f"{label} changed while it was being read")


class _HashingReader(io.RawIOBase):
    """A non-closing reader that accounts for every raw byte consumed by CSV."""

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
    """A narrow writer used to bind ZIP bytes to a source baseline digest."""

    def __init__(self, destination: BinaryIO) -> None:
        self._destination = destination
        self.digest = sha256()
        self.size_bytes = 0

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.size_bytes += len(data)
        return self._destination.write(data)


def _absolute_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path.cwd() / value


def _same_file_version(left: os.stat_result, right: os.stat_result) -> bool:
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
        dev=metadata.st_dev,
        ino=metadata.st_ino,
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        sha256=digest,
    )


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    try:
        for candidate in (path, *path.parents):
            if candidate.is_symlink():
                raise PortfolioInputError(f"{label} has a symlink ancestor")
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect {label}: {error}") from error


@contextmanager
def _safe_source(
    path: Path, label: str, *, expected: _FileVersion | None = None
) -> Iterator[_SafeSource]:
    """Open one non-symlink regular file and bind it to its opened identity."""
    _require_no_symlink_ancestors(path.parent, label)
    descriptor: int | None = None
    handle: BinaryIO | None = None
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            raise PortfolioInputError(f"{label} must be a regular non-symlink file")
        if expected is not None and not _matches_version(before, expected):
            raise PortfolioInputError(f"{label} changed before it could be opened")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if nofollow:
            flags |= nofollow
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not _same_file_version(before, opened):
            raise PortfolioInputError(f"{label} changed before it could be opened")
        if expected is not None and not _matches_version(opened, expected):
            raise PortfolioInputError(f"{label} changed before it could be opened")
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = None
        yield _SafeSource(path=path, handle=handle, initial_stat=opened)
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
    """Return a streaming SHA-256 from one safely opened file descriptor."""
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
        text = io.TextIOWrapper(
            io.BufferedReader(hashing_reader), encoding="utf-8-sig", newline=""
        )
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
            if target_index is not None:
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
        if row_count == 0:
            raise PortfolioInputError(f"{label} has no data rows")
        text.close()
        text = None
        if hashing_reader.size_bytes != source.initial_stat.st_size:
            raise PortfolioInputError(f"{label} changed while it was being parsed")
        return _CsvData(
            header=tuple(header),
            row_ids=tuple(row_ids),
            row_count=row_count,
            sha256=hashing_reader.digest.hexdigest(),
            size_bytes=hashing_reader.size_bytes,
            version=_version_from_stat(
                source.initial_stat, hashing_reader.digest.hexdigest()
            ),
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
    path: Path,
    label: str,
    *,
    collect_row_ids: bool,
    validate_sample_values: bool = False,
) -> _CsvData:
    with _safe_source(path, label) as source:
        data = _read_csv(
            source,
            label,
            collect_row_ids=collect_row_ids,
            validate_sample_values=validate_sample_values,
        )
        source.require_unchanged(label)
        return data


def _require_unique_row_ids(data: _CsvData, label: str) -> None:
    if len(set(data.row_ids)) != len(data.row_ids):
        raise PortfolioInputError(f"{label} has a duplicate row_id")


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
    train = _parse_csv_member(members["train.csv"], "train.csv", collect_row_ids=True)
    test = _parse_csv_member(members["test.csv"], "test.csv", collect_row_ids=True)
    history = _parse_csv_member(
        members["trackman_history.csv"], "trackman_history.csv", collect_row_ids=False
    )
    sample = _parse_csv_member(
        members["sample_submission.csv"],
        "sample_submission.csv",
        collect_row_ids=True,
        validate_sample_values=True,
    )
    if "control_success" not in train.header:
        raise PortfolioInputError("train.csv must contain row_id and control_success")
    if "control_success" in test.header:
        raise PortfolioInputError("test.csv must not contain control_success")
    for data, label in ((train, "train.csv"), (test, "test.csv"), (sample, "sample_submission.csv")):
        _require_unique_row_ids(data, label)
    if train.row_count <= test.row_count:
        raise PortfolioInputError("train.csv row count must exceed test.csv row count")
    if sample.row_count != test.row_count or sample.row_ids != test.row_ids:
        raise PortfolioInputError("sample_submission.csv row_id sequence differs from test.csv")

    hashes = {
        "train.csv": train.sha256,
        "test.csv": test.sha256,
        "trackman_history.csv": history.sha256,
        "sample_submission.csv": sample.sha256,
    }
    sizes = {
        "train.csv": train.size_bytes,
        "test.csv": test.size_bytes,
        "trackman_history.csv": history.size_bytes,
        "sample_submission.csv": sample.size_bytes,
    }
    versions = {
        "train.csv": train.version,
        "test.csv": test.version,
        "trackman_history.csv": history.version,
        "sample_submission.csv": sample.version,
    }
    return VerifiedOfficialData(
        root=data_root,
        train=members["train.csv"],
        test=members["test.csv"],
        history=members["trackman_history.csv"],
        sample_submission=members["sample_submission.csv"],
        member_sha256=MappingProxyType(hashes),
        train_rows=train.row_count,
        test_rows=test.row_count,
        member_sizes=MappingProxyType(sizes),
        member_versions=MappingProxyType(versions),
    )


def _copy_source(
    source: Path, destination: BinaryIO, *, expected: _FileVersion | None = None
) -> None:
    """Stream one safely opened source into a destination without path reopens."""
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


def _contract_path() -> Path:
    return Path(__file__).with_name("contract.json")


def _sealed_contract_bytes() -> bytes:
    contract_path = _contract_path()
    try:
        load_contract(contract_path)
        contract_bytes = contract_path.read_bytes()
    except PortfolioContractError as error:
        raise PortfolioInputError(f"temporal portfolio contract is invalid: {error}") from error
    except OSError as error:
        raise PortfolioInputError(f"cannot read contract.json: {error}") from error
    if sha256(contract_bytes).hexdigest() != _SEALED_CONTRACT_SHA256:
        raise PortfolioInputError("temporal portfolio contract bytes differ")
    return contract_bytes


def _canonical_manifest(verified: VerifiedOfficialData) -> bytes:
    contract_bytes = _sealed_contract_bytes()
    members = {
        archive_name: {
            "size": verified.member_sizes[source_name],
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


def _verify_archive(
    path: Path,
    manifest_bytes: bytes,
    verified: VerifiedOfficialData,
    *,
    expected: _FileVersion | None = None,
) -> None:
    try:
        with _safe_source(path, "prepared temporary ZIP", expected=expected) as source:
            with ZipFile(source.handle) as archive:
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
                    digest = sha256()
                    with archive.open(info) as handle:
                        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
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
            source.require_unchanged("prepared temporary ZIP")
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


def _version_from_open_file(handle: BinaryIO, label: str) -> _FileVersion:
    try:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PortfolioInputError(f"{label} must be a regular file")
        handle.seek(0)
        digest = sha256()
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot verify {label}: {error}") from error
    if not _same_file_version(before, after):
        raise PortfolioInputError(f"{label} changed while it was being verified")
    return _version_from_stat(after, digest.hexdigest())


def _require_path_version(path: Path, version: _FileVersion, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect {label}: {error}") from error
    if not _matches_version(metadata, version):
        raise PortfolioInputError(f"{label} changed after verification")


def _unlink_owned(
    path: Path | None,
    version: _FileVersion | None,
    *,
    identity: tuple[int, int] | None = None,
) -> None:
    if path is None:
        return
    try:
        metadata = os.lstat(path)
        owned = (
            _matches_version(metadata, version)
            if version is not None
            else identity is not None
            and (metadata.st_dev, metadata.st_ino) == identity
        )
        if owned:
            path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _publish_without_replace(
    temporary: Path, destination: Path, version: _FileVersion
) -> None:
    _require_path_version(temporary, version, "prepared temporary ZIP")
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise PortfolioInputError("output already exists during publication") from error
    except OSError as error:
        raise PortfolioInputError(f"cannot publish input archive: {error}") from error
    _require_path_version(destination, version, "published input archive")
    file_sha256(destination, expected=version)


def _publish_with_replace(
    temporary: Path, destination: Path, version: _FileVersion
) -> None:
    candidate: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, candidate_name = mkstemp(
            prefix=".temporal-portfolio-publish-", suffix=".zip", dir=destination.parent
        )
        candidate = Path(candidate_name)
        os.close(descriptor)
        descriptor = None
        candidate.unlink()
        _require_path_version(temporary, version, "prepared temporary ZIP")
        os.link(temporary, candidate, follow_symlinks=False)
        _require_path_version(candidate, version, "publication staging archive")
        os.replace(candidate, destination)
        candidate = None
        _require_path_version(destination, version, "published input archive")
        file_sha256(destination, expected=version)
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot publish replacement input archive: {error}") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _unlink_owned(candidate, version)


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
    manifest_bytes = _canonical_manifest(verified)
    descriptor: int | None = None
    temporary: Path | None = None
    temporary_handle: BinaryIO | None = None
    temporary_version: _FileVersion | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        descriptor, temporary_name = mkstemp(
            prefix=".temporal-portfolio-input-", suffix=".zip", dir=destination.parent
        )
        temporary = Path(temporary_name)
        temporary_stat = os.fstat(descriptor)
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        temporary_handle = os.fdopen(descriptor, "w+b", buffering=0)
        descriptor = None
        with ZipFile(
            temporary_handle, "w", compression=ZIP_DEFLATED, compresslevel=9
        ) as archive:
            with archive.open(_zip_info("manifest.json"), "w") as target:
                target.write(manifest_bytes)
            for archive_name, source_name in _ARCHIVE_SOURCES:
                with archive.open(_zip_info(archive_name), "w") as target:
                    writer = _HashingWriter(target)
                    _copy_source(
                        sources[source_name],
                        writer,
                        expected=verified.member_versions[source_name],
                    )
                    expected_sha256 = verified.member_sha256[source_name]
                    expected_size = verified.member_sizes[source_name]
                    if (
                        writer.size_bytes != expected_size
                        or writer.digest.hexdigest() != expected_sha256
                    ):
                        raise PortfolioInputError(
                            f"official data source changed before ZIP copy: {source_name}"
                        )
        temporary_handle.flush()
        temporary_version = _version_from_open_file(
            temporary_handle, "prepared temporary ZIP"
        )
        _verify_archive(
            temporary, manifest_bytes, verified, expected=temporary_version
        )
        for source_name, source in sources.items():
            if (
                file_sha256(
                    source, expected=verified.member_versions[source_name]
                )
                != verified.member_sha256[source_name]
            ):
                raise PortfolioInputError("official data source changed during archive preparation")
        archive_sha256 = file_sha256(temporary, expected=temporary_version)
        _require_path_version(temporary, temporary_version, "prepared temporary ZIP")
        archive_size = temporary_version.size_bytes
        if replace:
            _publish_with_replace(temporary, destination, temporary_version)
        else:
            _publish_without_replace(temporary, destination, temporary_version)
        _unlink_owned(temporary, temporary_version)
        temporary = None
        return PreparedInputArchive(destination, archive_sha256, archive_size)
    except PortfolioInputError:
        raise
    except (BadZipFile, OSError) as error:
        raise PortfolioInputError(f"cannot prepare temporary input archive: {error}") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_handle is not None:
            try:
                temporary_handle.close()
            except OSError:
                pass
        if temporary is not None:
            _unlink_owned(
                temporary,
                temporary_version,
                identity=temporary_identity,
            )
