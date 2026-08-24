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
from pathlib import PurePosixPath
import shutil
import stat
import struct
from tempfile import mkdtemp
from types import MappingProxyType
from typing import BinaryIO, Iterator, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo
import zlib

from .contracts import PortfolioContractError, _SEALED_CONTRACT_SHA256, load_contract
from .state import Bindings


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
class VerifiedPreparedInputArchive:
    path: Path
    archive_sha256: str
    size_bytes: int
    manifest_sha256: str
    bindings: Bindings
    train_rows: int
    test_rows: int
    member_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "member_sha256",
            MappingProxyType(dict(self.member_sha256)),
        )

    @property
    def sha256(self) -> str:
        """Backward-compatible name for the bound outer archive digest."""

        return self.archive_sha256


@dataclass(frozen=True)
class PreparedInputArchive(VerifiedPreparedInputArchive):
    """Creation-facing name for the verified prepared-archive evidence."""


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
_MAX_PREPARED_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_PREPARED_MANIFEST_BYTES = 1024 * 1024
_MAX_PREPARED_MEMBER_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_PREPARED_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_MAX_PREPARED_MEMBER_COMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_PREPARED_TOTAL_COMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_MAX_PREPARED_COMPRESSION_RATIO = 200.0
_MAX_PREPARED_CENTRAL_DIRECTORY_BYTES = 1024 * 1024


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
        absolute = requested if requested.is_absolute() else Path.cwd() / requested
        canonical_parent = absolute.parent.resolve(strict=False)
        return canonical_parent / absolute.name
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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PortfolioInputError(f"prepared ZIP manifest has duplicate key: {key}")
        result[key] = value
    return result


def _reject_manifest_constant(value: str) -> None:
    raise PortfolioInputError(f"prepared ZIP manifest has non-finite value: {value}")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PortfolioInputError("prepared ZIP manifest cannot be canonicalized") from error


def _preflight_prepared_zip(source: BinaryIO, size_bytes: int) -> None:
    """Validate bounded classic single-disk ZIP structure before ``ZipFile``."""

    if type(size_bytes) is not int or size_bytes < 22:
        raise BadZipFile("prepared ZIP has no end record")
    original_offset = source.tell()
    try:
        tail_size = min(size_bytes, 22 + 0xFFFF + 20)
        tail_offset = size_bytes - tail_size
        source.seek(tail_offset)
        tail = source.read(tail_size)
        marker = tail.rfind(b"PK\x05\x06")
        if marker < 0 or len(tail) - marker < 22:
            raise BadZipFile("prepared ZIP has no end record")
        eocd_offset = tail_offset + marker
        (
            signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            comment_size,
        ) = struct.unpack_from("<4s4H2LH", tail, marker)
        if signature != b"PK\x05\x06" or eocd_offset + 22 + comment_size != size_bytes:
            raise BadZipFile("prepared ZIP end record is malformed")
        if (
            disk_entries == 0xFFFF
            or total_entries == 0xFFFF
            or central_size == 0xFFFFFFFF
            or central_offset == 0xFFFFFFFF
        ):
            raise PortfolioInputError("prepared ZIP64 archives are not supported")
        if eocd_offset >= 20:
            source.seek(eocd_offset - 20)
            if source.read(4) == b"PK\x06\x07":
                raise PortfolioInputError("prepared ZIP64 archives are not supported")
        if disk_number != 0 or central_disk != 0 or disk_entries != total_entries:
            raise PortfolioInputError("prepared multi-disk ZIP archives are not supported")
        if total_entries > len(_ARCHIVE_NAMES):
            raise PortfolioInputError("prepared ZIP entry count exceeds the limit")
        if central_size > _MAX_PREPARED_CENTRAL_DIRECTORY_BYTES:
            raise PortfolioInputError("prepared ZIP central directory exceeds the limit")
        if central_offset + central_size != eocd_offset:
            raise PortfolioInputError("prepared ZIP central directory bounds are invalid")

        central_end = central_offset + central_size
        cursor = central_offset
        actual_count = 0
        while cursor < central_end:
            if central_end - cursor < 46:
                raise PortfolioInputError("prepared ZIP central record is truncated")
            source.seek(cursor)
            header = source.read(46)
            if len(header) != 46:
                raise PortfolioInputError("prepared ZIP central record is truncated")
            fields = struct.unpack("<4s6H3L5H2L", header)
            if fields[0] != b"PK\x01\x02":
                raise PortfolioInputError("prepared ZIP central record signature differs")
            compressed_size = fields[8]
            uncompressed_size = fields[9]
            filename_size = fields[10]
            extra_size = fields[11]
            member_comment_size = fields[12]
            disk_start = fields[13]
            local_header_offset = fields[16]
            if (
                compressed_size == 0xFFFFFFFF
                or uncompressed_size == 0xFFFFFFFF
                or disk_start == 0xFFFF
                or local_header_offset == 0xFFFFFFFF
            ):
                raise PortfolioInputError("prepared ZIP64 member fields are not supported")
            if disk_start != 0:
                raise PortfolioInputError("prepared multi-disk member is not supported")
            record_end = (
                cursor
                + 46
                + filename_size
                + extra_size
                + member_comment_size
            )
            if record_end > central_end:
                raise PortfolioInputError("prepared ZIP central record exceeds its bounds")
            actual_count += 1
            if actual_count > len(_ARCHIVE_NAMES):
                raise PortfolioInputError("prepared ZIP entry count exceeds the limit")
            cursor = record_end
        if cursor != central_end or actual_count != total_entries:
            raise PortfolioInputError("prepared ZIP central directory count differs")
    finally:
        source.seek(original_offset)


def _validate_prepared_member_name(info: ZipInfo) -> None:
    name = info.filename
    original_name = getattr(info, "orig_filename", name)
    if original_name != name:
        raise PortfolioInputError("prepared ZIP has a NUL-truncated member path")
    if (
        type(name) is not str
        or not name
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise PortfolioInputError("prepared ZIP has an unsafe member path")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or name.startswith("/")
        or name.endswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise PortfolioInputError("prepared ZIP has an unsafe member path")


def _inspect_prepared_metadata(archive: ZipFile) -> dict[str, ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise PortfolioInputError("prepared ZIP has duplicate members")
    if len(names) != len(_ARCHIVE_NAMES) or set(names) != set(_ARCHIVE_NAMES):
        raise PortfolioInputError("prepared ZIP members differ")
    total_uncompressed = 0
    total_compressed = 0
    result: dict[str, ZipInfo] = {}
    for info in infos:
        _validate_prepared_member_name(info)
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if info.is_dir() or stat.S_ISLNK(mode) or file_type not in {0, stat.S_IFREG}:
            raise PortfolioInputError(f"prepared ZIP member is not regular: {info.filename}")
        if info.flag_bits & 1:
            raise PortfolioInputError("prepared ZIP contains an encrypted member")
        if type(info.file_size) is not int or info.file_size < 0:
            raise PortfolioInputError("prepared ZIP member size is invalid")
        if type(info.compress_size) is not int or info.compress_size < 0:
            raise PortfolioInputError("prepared ZIP compressed size is invalid")
        if info.file_size > _MAX_PREPARED_MEMBER_UNCOMPRESSED_BYTES:
            raise PortfolioInputError("prepared ZIP member exceeds the size limit")
        if info.compress_size > _MAX_PREPARED_MEMBER_COMPRESSED_BYTES:
            raise PortfolioInputError("prepared ZIP compressed member exceeds the size limit")
        total_uncompressed += info.file_size
        total_compressed += info.compress_size
        if total_uncompressed > _MAX_PREPARED_TOTAL_UNCOMPRESSED_BYTES:
            raise PortfolioInputError("prepared ZIP total uncompressed size exceeds the limit")
        if total_compressed > _MAX_PREPARED_TOTAL_COMPRESSED_BYTES:
            raise PortfolioInputError("prepared ZIP total compressed size exceeds the limit")
        if info.file_size > 0 and (
            info.compress_size == 0
            or info.file_size / info.compress_size > _MAX_PREPARED_COMPRESSION_RATIO
        ):
            raise PortfolioInputError("prepared ZIP member compression ratio exceeds the limit")
        if info.filename == "manifest.json" and info.file_size > _MAX_PREPARED_MANIFEST_BYTES:
            raise PortfolioInputError("prepared ZIP manifest exceeds the size limit")
        result[info.filename] = info
    return result


def _read_prepared_member(archive: ZipFile, info: ZipInfo) -> tuple[str, int, bytes | None]:
    digest = sha256()
    size = 0
    chunks: list[bytes] | None = [] if info.filename == "manifest.json" else None
    with archive.open(info, "r") as member:
        while chunk := member.read(_CHUNK_SIZE):
            size += len(chunk)
            if size > info.file_size or size > _MAX_PREPARED_MEMBER_UNCOMPRESSED_BYTES:
                raise PortfolioInputError("prepared ZIP member decompressed beyond its size")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
    if size != info.file_size:
        raise PortfolioInputError("prepared ZIP member decompressed size differs")
    return digest.hexdigest(), size, b"".join(chunks) if chunks is not None else None


def _parse_prepared_manifest(raw: bytes) -> dict[str, object]:
    try:
        manifest = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_manifest_object,
            parse_constant=_reject_manifest_constant,
        )
    except PortfolioInputError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortfolioInputError(f"cannot parse prepared ZIP manifest: {error}") from error
    if type(manifest) is not dict or raw != _canonical_json(manifest):
        raise PortfolioInputError("prepared ZIP manifest is not canonical JSON")
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "campaign_id",
        "submission_package",
        "contract_sha256",
        "train_rows",
        "test_rows",
        "members",
    }
    if set(manifest) != expected_keys:
        raise PortfolioInputError("prepared ZIP manifest schema differs")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or type(manifest["artifact_kind"]) is not str
        or manifest["artifact_kind"] != "temporal_portfolio_input_v1"
        or type(manifest["campaign_id"]) is not str
        or manifest["campaign_id"] != "temporal_portfolio_v1"
        or manifest["submission_package"] is not False
        or not _is_sha256(manifest["contract_sha256"])
        or manifest["contract_sha256"] != sha256(_sealed_contract_bytes()).hexdigest()
        or type(manifest["train_rows"]) is not int
        or manifest["train_rows"] <= 0
        or type(manifest["test_rows"]) is not int
        or manifest["test_rows"] <= 0
        or manifest["train_rows"] <= manifest["test_rows"]
    ):
        raise PortfolioInputError("prepared ZIP manifest identity or types differ")
    records = manifest["members"]
    if type(records) is not dict or set(records) != set(_ARCHIVE_NAMES[1:]):
        raise PortfolioInputError("prepared ZIP manifest member records differ")
    for name, record in records.items():
        if (
            type(name) is not str
            or type(record) is not dict
            or set(record) != {"size", "sha256"}
            or type(record["size"]) is not int
            or record["size"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise PortfolioInputError("prepared ZIP manifest member record is invalid")
    return manifest


def verify_prepared_input_archive(
    path: str | Path, *, expected_archive_sha256: str
) -> PreparedInputArchive:
    """Verify an uploaded prepared input against a caller-trusted outer digest."""

    if not _is_sha256(expected_archive_sha256):
        raise PortfolioInputError("expected archive SHA-256 must be lowercase hexadecimal")
    try:
        requested = Path(path)
        with _safe_source(requested, "prepared input ZIP") as source:
            if source.initial_stat.st_size > _MAX_PREPARED_ARCHIVE_BYTES:
                raise PortfolioInputError("prepared ZIP exceeds the archive size limit")
            archive_digest = sha256()
            for chunk in iter(lambda: source.handle.read(_CHUNK_SIZE), b""):
                archive_digest.update(chunk)
            observed_archive_sha256 = archive_digest.hexdigest()
            if observed_archive_sha256 != expected_archive_sha256:
                raise PortfolioInputError("prepared archive SHA-256 differs from expected")
            source.require_unchanged("prepared input ZIP")
            source.handle.seek(0)
            _preflight_prepared_zip(source.handle, source.initial_stat.st_size)
            source.handle.seek(0)
            with ZipFile(source.handle) as archive:
                infos = _inspect_prepared_metadata(archive)
                manifest_digest, _, manifest_bytes = _read_prepared_member(
                    archive, infos["manifest.json"]
                )
                if manifest_bytes is None:
                    raise PortfolioInputError("prepared ZIP manifest is unavailable")
                manifest = _parse_prepared_manifest(manifest_bytes)
                records = manifest["members"]
                observed_members: dict[str, str] = {}
                for name in _ARCHIVE_NAMES[1:]:
                    digest, size, _ = _read_prepared_member(archive, infos[name])
                    record = records[name]
                    if record["size"] != size or record["sha256"] != digest:
                        raise PortfolioInputError("prepared ZIP member evidence differs")
                    observed_members[name] = digest
            source.require_unchanged("prepared input ZIP")
            canonical_path = Path(os.path.realpath(source.path))
            contract_sha256 = manifest["contract_sha256"]
            campaign_id = manifest["campaign_id"]
            return PreparedInputArchive(
                canonical_path,
                observed_archive_sha256,
                source.initial_stat.st_size,
                manifest_digest,
                Bindings(campaign_id, contract_sha256, manifest_digest),
                manifest["train_rows"],
                manifest["test_rows"],
                MappingProxyType(observed_members),
            )
    except PortfolioInputError:
        raise
    except (
        BadZipFile,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        EOFError,
        json.JSONDecodeError,
        UnicodeError,
        zlib.error,
    ) as error:
        raise PortfolioInputError(f"cannot verify prepared ZIP: {error}") from error


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
    try:
        exists = os.path.lexists(destination)
        if exists and stat.S_ISLNK(os.lstat(destination).st_mode):
            raise PortfolioInputError("output must not be a symlink")
    except PortfolioInputError:
        raise
    except OSError as error:
        raise PortfolioInputError(f"cannot inspect output path: {error}") from error
    _require_output_outside_data(destination, verified)
    parent = _canonical_output_parent(destination)
    destination = parent / destination.name
    try:
        if exists and not replace:
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
        manifest_sha256 = sha256(manifest_bytes).hexdigest()
        archive_member_sha256 = MappingProxyType(
            {
                archive_name: verified.member_sha256[source_name]
                for archive_name, source_name in _ARCHIVE_SOURCES
            }
        )
        prepared = PreparedInputArchive(
            destination,
            archive_sha256,
            archive_version.size_bytes,
            manifest_sha256,
            Bindings(
                "temporal_portfolio_v1",
                sha256(_sealed_contract_bytes()).hexdigest(),
                manifest_sha256,
            ),
            verified.train_rows,
            verified.test_rows,
            archive_member_sha256,
        )
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
