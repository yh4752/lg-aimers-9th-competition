"""Deterministic, fail-closed handoffs for the temporal portfolio.

Discovery deliberately searches at most ``_MAX_DISCOVERY_DEPTH`` directory levels,
``_MAX_DISCOVERY_ENTRIES`` entries, and ``_MAX_DISCOVERY_INSPECTED_BYTES`` bytes of
regular-file candidates.  It never extracts archives and treats every manifest-
bearing candidate as untrusted until all candidates have been verified.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from types import MappingProxyType
from typing import BinaryIO
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo
import zlib

from .state import ALLOWED_STAGE_ORDER, Bindings, Lineage


class PortfolioArtifactError(ValueError):
    """Raised when temporal portfolio evidence cannot be trusted."""


_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_OUTER_MEMBERS = frozenset(
    {
        "review.zip",
        "resume.zip",
        "run.log",
        "stage_summary.json",
        "handoff_manifest.json",
    }
)
_PAYLOAD_MEMBERS = frozenset(_OUTER_MEMBERS - {"handoff_manifest.json"})
_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_MEMBER_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_MAX_MEMBER_COMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_COMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200.0
_MAX_DISCOVERY_ROOTS = 32
_MAX_DISCOVERY_DEPTH = 4
_MAX_DISCOVERY_ENTRIES = 4096
_MAX_DISCOVERY_INSPECTED_BYTES = 2 * 1024 * 1024 * 1024
_MAX_DISCOVERY_CANDIDATES = 128
_MAX_DISCOVERY_ZIP_MEMBERS = 16384
_CHUNK_SIZE = 1024 * 1024
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class BoundFile:
    """An immutable file binding that is rechecked while being consumed."""

    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        try:
            path = Path(os.path.abspath(os.fspath(self.path)))
        except (TypeError, ValueError, OSError) as error:
            raise PortfolioArtifactError("bound file path is invalid") from error
        if not _is_sha256(self.sha256):
            raise PortfolioArtifactError("bound file SHA-256 is invalid")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise PortfolioArtifactError("bound file size is invalid")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class StageEvidence:
    stage: str
    bindings: Bindings
    lineage: Lineage
    review_members: Mapping[str, bytes | BoundFile]
    resume_members: Mapping[str, bytes | BoundFile]
    run_log: bytes
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.bindings) is not Bindings:
            raise PortfolioArtifactError("evidence bindings type is invalid")
        if type(self.lineage) is not Lineage:
            raise PortfolioArtifactError("evidence lineage type is invalid")
        if type(self.stage) is not str or self.stage != self.lineage.stage:
            raise PortfolioArtifactError("evidence stage differs from its lineage")
        if self.bindings.campaign_id != self.lineage.campaign_id:
            raise PortfolioArtifactError("evidence campaign bindings differ")
        if type(self.run_log) is not bytes:
            raise PortfolioArtifactError("run log must be immutable bytes")
        object.__setattr__(
            self,
            "review_members",
            _snapshot_member_mapping(self.review_members, "review"),
        )
        object.__setattr__(
            self,
            "resume_members",
            _snapshot_member_mapping(self.resume_members, "resume"),
        )
        summary = _snapshot_json(self.summary, "stage summary")
        if not isinstance(summary, Mapping):
            raise PortfolioArtifactError("stage summary must be a mapping")
        object.__setattr__(self, "summary", summary)


@dataclass(frozen=True)
class HandoffPath:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        try:
            path = Path(os.path.abspath(os.fspath(self.path)))
        except (TypeError, ValueError, OSError) as error:
            raise PortfolioArtifactError("handoff path is invalid") from error
        if not _is_sha256(self.sha256):
            raise PortfolioArtifactError("handoff SHA-256 is invalid")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class VerifiedHandoff:
    path: Path
    sha256: str | None
    manifest_sha256: str
    stage: str
    sequence: int
    bindings: Bindings
    lineage: Lineage
    members: Mapping[str, str]
    review_members: Mapping[str, str] = field(repr=False)
    resume_members: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "members", MappingProxyType(dict(self.members)))
        object.__setattr__(
            self, "review_members", MappingProxyType(dict(self.review_members))
        )
        object.__setattr__(
            self, "resume_members", MappingProxyType(dict(self.resume_members))
        )


@dataclass
class _SafeFile:
    path: Path
    handle: BinaryIO
    initial_stat: os.stat_result

    def require_unchanged(self) -> None:
        current = os.fstat(self.handle.fileno())
        if not _same_file(self.initial_stat, current):
            raise PortfolioArtifactError(f"file changed while being read: {self.path}")


@dataclass
class _DiscoveryBudget:
    entries: int = 0
    inspected_bytes: int = 0
    candidates: int = 0

    def add_entries(self, count: int) -> None:
        self.entries += count
        if self.entries > _MAX_DISCOVERY_ENTRIES:
            raise PortfolioArtifactError("discovery entry limit exceeded")

    def add_inspected_file(self, size_bytes: int) -> None:
        self.inspected_bytes += size_bytes
        if self.inspected_bytes > _MAX_DISCOVERY_INSPECTED_BYTES:
            raise PortfolioArtifactError("discovery inspection byte limit exceeded")

    def add_candidate(self) -> None:
        self.candidates += 1
        if self.candidates > _MAX_DISCOVERY_CANDIDATES:
            raise PortfolioArtifactError("discovery candidate limit exceeded")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _snapshot_member_mapping(
    value: object, label: str
) -> Mapping[str, bytes | BoundFile]:
    if not isinstance(value, Mapping):
        raise PortfolioArtifactError(f"{label} members must be a mapping")
    snapshot: dict[str, bytes | BoundFile] = {}
    try:
        items = value.items()
        for name, member in items:
            _validate_member_name(name, reserved=frozenset({"manifest.json"}))
            if type(member) is not bytes and type(member) is not BoundFile:
                raise PortfolioArtifactError(
                    f"{label} member must be immutable bytes or BoundFile: {name}"
                )
            if name in snapshot:
                raise PortfolioArtifactError(f"{label} member is duplicated: {name}")
            snapshot[name] = member
    except PortfolioArtifactError:
        raise
    except Exception as error:
        raise PortfolioArtifactError(f"cannot snapshot {label} members") from error
    return MappingProxyType(dict(sorted(snapshot.items())))


def _snapshot_json(value: object, label: str) -> object:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PortfolioArtifactError(f"{label} contains a non-finite number")
        return value
    if type(value) in {list, tuple}:
        return tuple(_snapshot_json(item, label) for item in value)
    if isinstance(value, Mapping):
        snapshot: dict[str, object] = {}
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise PortfolioArtifactError(f"{label} keys must be strings")
                snapshot[key] = _snapshot_json(item, label)
        except PortfolioArtifactError:
            raise
        except Exception as error:
            raise PortfolioArtifactError(f"cannot snapshot {label}") from error
        return MappingProxyType(dict(sorted(snapshot.items())))
    raise PortfolioArtifactError(f"{label} contains an unsupported JSON value")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain_json(item) for item in value]
    return value


def canonical_json(value: object) -> bytes:
    """Return the one accepted UTF-8 JSON encoding."""

    try:
        return json.dumps(
            _plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PortfolioArtifactError("value cannot be encoded as canonical JSON") from error


def _validate_member_name(name: object, *, reserved: frozenset[str]) -> None:
    if type(name) is not str or not name:
        raise PortfolioArtifactError("unsafe ZIP member path")
    if name in reserved or "\\" in name or any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise PortfolioArtifactError(f"unsafe or reserved ZIP member path: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or name.startswith("/")
        or name.endswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise PortfolioArtifactError(f"unsafe or reserved ZIP member path: {name!r}")


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISREG(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, *, include_final: bool = True) -> None:
    absolute = _absolute_path(path)
    parts = absolute.parts
    current = Path(parts[0])
    stop = len(parts) if include_final else len(parts) - 1
    for part in parts[1:stop]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise PortfolioArtifactError(f"symlink path component is not allowed: {current}")


def _open_safe_file(path: Path, label: str) -> _SafeFile:
    absolute = _absolute_path(path)
    _reject_symlink_components(absolute)
    before = os.lstat(absolute)
    if not stat.S_ISREG(before.st_mode):
        raise PortfolioArtifactError(f"{label} must be a regular non-symlink file")
    if before.st_size > _MAX_ARCHIVE_BYTES and label == "handoff":
        raise PortfolioArtifactError("handoff exceeds the archive size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_file(before, opened):
            raise PortfolioArtifactError(f"{label} changed before it could be opened")
        return _SafeFile(absolute, os.fdopen(descriptor, "rb", buffering=0), opened)
    except Exception:
        os.close(descriptor)
        raise


def _read_safe_file(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    maximum_size: int = _MAX_MEMBER_UNCOMPRESSED_BYTES,
) -> bytes:
    source = _open_safe_file(path, label)
    try:
        if source.initial_stat.st_size > maximum_size:
            raise PortfolioArtifactError(f"{label} exceeds the size limit")
        if expected_size is not None and source.initial_stat.st_size != expected_size:
            raise PortfolioArtifactError(f"{label} size differs from its binding")
        digest = sha256()
        chunks: list[bytes] = []
        total = 0
        while chunk := source.handle.read(_CHUNK_SIZE):
            total += len(chunk)
            if total > maximum_size:
                raise PortfolioArtifactError(f"{label} exceeds the size limit")
            digest.update(chunk)
            chunks.append(chunk)
        source.require_unchanged()
        if expected_size is not None and total != expected_size:
            raise PortfolioArtifactError(f"{label} size differs from its binding")
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise PortfolioArtifactError(f"{label} content differs from its SHA-256 binding")
        return b"".join(chunks)
    finally:
        source.handle.close()


def bind_file(path: str | Path) -> BoundFile:
    """Bind one regular file to its current bytes for later fail-closed use."""

    try:
        source_path = _absolute_path(path)
        source = _open_safe_file(source_path, "bound file")
        try:
            digest = sha256()
            size = 0
            while chunk := source.handle.read(_CHUNK_SIZE):
                size += len(chunk)
                if size > _MAX_MEMBER_UNCOMPRESSED_BYTES:
                    raise PortfolioArtifactError("bound file exceeds the member size limit")
                digest.update(chunk)
            source.require_unchanged()
        finally:
            source.handle.close()
        return BoundFile(source_path, digest.hexdigest(), size)
    except PortfolioArtifactError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise PortfolioArtifactError("cannot safely bind file") from error


def _materialize_members(
    members: Mapping[str, bytes | BoundFile], label: str
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    total = 0
    for name, value in members.items():
        if type(value) is bytes:
            data = value
        else:
            data = _read_safe_file(
                value.path,
                f"{label} member {name}",
                expected_sha256=value.sha256,
                expected_size=value.size_bytes,
            )
        if len(data) > _MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise PortfolioArtifactError(f"{label} member exceeds the uncompressed size limit: {name}")
        total += len(data)
        if total > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise PortfolioArtifactError(f"{label} exceeds the total uncompressed size limit")
        result[name] = data
    return result


def _bindings_dict(bindings: Bindings) -> dict[str, object]:
    return {
        "campaign_id": bindings.campaign_id,
        "contract_sha256": bindings.contract_sha256,
        "input_manifest_sha256": bindings.input_manifest_sha256,
    }


def _lineage_dict(lineage: Lineage) -> dict[str, object]:
    return {
        "campaign_id": lineage.campaign_id,
        "stage": lineage.stage,
        "sequence": lineage.sequence,
        "parent_manifest_sha256": lineage.parent_manifest_sha256,
        "training_sha256": lineage.training_sha256,
        "state_schema_version": lineage.state_schema_version,
        "runtime_sha256": lineage.runtime_sha256,
    }


def _member_records(members: Mapping[str, bytes]) -> dict[str, dict[str, object]]:
    return {
        name: {"sha256": sha256(value).hexdigest(), "size_bytes": len(value)}
        for name, value in sorted(members.items())
    }


def _nested_manifest(
    kind: str, members: Mapping[str, bytes], bindings: Bindings
) -> bytes:
    return canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": kind,
            "submission_package": False,
            "bindings": _bindings_dict(bindings),
            "members": _member_records(members),
        }
    )


def _handoff_manifest(members: Mapping[str, bytes], evidence: StageEvidence) -> bytes:
    return canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "temporal_handoff_v1",
            "submission_package": False,
            "bindings": _bindings_dict(evidence.bindings),
            "lineage": _lineage_dict(evidence.lineage),
            "members": _member_records(members),
        }
    )


def _deterministic_zip_bytes(members: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(members.items()):
            info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value, compress_type=ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _build_nested_zip(
    kind: str, members: Mapping[str, bytes | BoundFile], bindings: Bindings
) -> bytes:
    materialized = _materialize_members(members, kind)
    manifest = _nested_manifest(kind, materialized, bindings)
    return _deterministic_zip_bytes({**materialized, "manifest.json": manifest})


def _prepare_output_root(root: str | Path) -> Path:
    try:
        absolute = _absolute_path(root)
    except (TypeError, ValueError, OSError) as error:
        raise PortfolioArtifactError("handoff output root is invalid") from error
    _reject_symlink_components(absolute)
    try:
        if absolute.exists():
            metadata = absolute.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise PortfolioArtifactError("handoff output root must not be a symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                raise PortfolioArtifactError("handoff output root must be a directory")
        else:
            _reject_symlink_components(absolute, include_final=False)
            absolute.mkdir(parents=True, exist_ok=False)
        return absolute
    except PortfolioArtifactError:
        raise
    except OSError as error:
        raise PortfolioArtifactError("cannot create handoff output root") from error


def _atomic_publish(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PortfolioArtifactError("handoff output must not be a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise PortfolioArtifactError("handoff output must be a regular file")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_handoff_file(temporary)
        if path.exists() or path.is_symlink():
            current = path.lstat()
            if stat.S_ISLNK(current.st_mode):
                raise PortfolioArtifactError("handoff output became a symlink")
            if not stat.S_ISREG(current.st_mode):
                raise PortfolioArtifactError("handoff output is not a regular file")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _verify_handoff_file(path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_handoff(root: str | Path, evidence: StageEvidence) -> HandoffPath:
    """Atomically publish and self-verify one deterministic stage handoff."""

    try:
        if type(evidence) is not StageEvidence:
            raise PortfolioArtifactError("stage evidence type is invalid")
        review = _build_nested_zip(
            "temporal_review_v1", evidence.review_members, evidence.bindings
        )
        resume = _build_nested_zip(
            "temporal_resume_v1", evidence.resume_members, evidence.bindings
        )
        members = {
            "review.zip": review,
            "resume.zip": resume,
            "run.log": evidence.run_log,
            "stage_summary.json": canonical_json(evidence.summary),
        }
        manifest = _handoff_manifest(members, evidence)
        payload = _deterministic_zip_bytes(
            {**members, "handoff_manifest.json": manifest}
        )
        output_root = _prepare_output_root(root)
        path = output_root / f"temporal_portfolio_stage_{evidence.stage}_handoff.zip"
        _atomic_publish(path, payload)
        verified = _verify_handoff_file(path)
        if verified.sha256 is None:
            raise PortfolioArtifactError("published handoff has no file SHA-256")
        return HandoffPath(path, verified.sha256)
    except PortfolioArtifactError:
        raise
    except (BadZipFile, OSError, zlib.error, json.JSONDecodeError, UnicodeError, TypeError, ValueError, RuntimeError) as error:
        raise PortfolioArtifactError("cannot publish temporal portfolio handoff") from error


def _inspect_zip_metadata(
    archive: ZipFile,
    *,
    label: str,
    required_manifest: str | None,
    exact_names: frozenset[str] | None = None,
) -> dict[str, ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise PortfolioArtifactError(f"{label} has duplicate ZIP members")
    if required_manifest is not None and required_manifest not in names:
        raise PortfolioArtifactError(f"{label} has no required manifest")
    total_uncompressed = 0
    total_compressed = 0
    result: dict[str, ZipInfo] = {}
    for info in infos:
        original_name = getattr(info, "orig_filename", info.filename)
        if original_name != info.filename:
            raise PortfolioArtifactError(f"{label} has an unsafe NUL-truncated member path")
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if info.is_dir() or stat.S_ISLNK(mode) or file_type not in {0, stat.S_IFREG}:
            raise PortfolioArtifactError(
                f"{label} member is not a regular file: {info.filename}"
            )
        _validate_member_name(info.filename, reserved=frozenset())
        if info.flag_bits & 1:
            raise PortfolioArtifactError(f"{label} contains an encrypted member")
        if type(info.file_size) is not int or info.file_size < 0:
            raise PortfolioArtifactError(f"{label} member size is invalid")
        if type(info.compress_size) is not int or info.compress_size < 0:
            raise PortfolioArtifactError(f"{label} compressed member size is invalid")
        if info.file_size > _MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise PortfolioArtifactError(
                f"{label} member exceeds the uncompressed size limit: {info.filename}"
            )
        if info.compress_size > _MAX_MEMBER_COMPRESSED_BYTES:
            raise PortfolioArtifactError(
                f"{label} member exceeds the compressed size limit: {info.filename}"
            )
        total_uncompressed += info.file_size
        total_compressed += info.compress_size
        if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise PortfolioArtifactError(f"{label} exceeds the total uncompressed size limit")
        if total_compressed > _MAX_TOTAL_COMPRESSED_BYTES:
            raise PortfolioArtifactError(f"{label} exceeds the total compressed size limit")
        if info.file_size > 0:
            if info.compress_size == 0:
                raise PortfolioArtifactError(
                    f"{label} member has an invalid compression ratio: {info.filename}"
                )
            if info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO:
                raise PortfolioArtifactError(
                    f"{label} member exceeds the compression ratio limit: {info.filename}"
                )
        if required_manifest == info.filename and info.file_size > _MAX_MANIFEST_BYTES:
            raise PortfolioArtifactError(f"{label} manifest exceeds the size limit")
        result[info.filename] = info
    if exact_names is not None and set(names) != set(exact_names):
        raise PortfolioArtifactError(f"{label} has unexpected or missing members")
    return result


def _read_zip_member(archive: ZipFile, info: ZipInfo, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    with archive.open(info, "r") as source:
        while chunk := source.read(_CHUNK_SIZE):
            total += len(chunk)
            if total > info.file_size or total > _MAX_MEMBER_UNCOMPRESSED_BYTES:
                raise PortfolioArtifactError(f"{label} decompressed beyond its declared size")
            chunks.append(chunk)
    if total != info.file_size:
        raise PortfolioArtifactError(f"{label} decompressed size differs")
    return b"".join(chunks)


def _decode_manifest(value: bytes, label: str) -> dict[str, object]:
    if len(value) > _MAX_MANIFEST_BYTES:
        raise PortfolioArtifactError(f"{label} manifest exceeds the size limit")
    parsed = json.loads(value.decode("utf-8"))
    if type(parsed) is not dict:
        raise PortfolioArtifactError(f"{label} manifest must be an object")
    if value != canonical_json(parsed):
        raise PortfolioArtifactError(f"{label} manifest is not canonical JSON")
    return parsed


def _exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(expected):
        raise PortfolioArtifactError(f"{label} has invalid keys")
    return value


def _parse_bindings(value: object, label: str) -> Bindings:
    mapping = _exact_keys(
        value,
        frozenset({"campaign_id", "contract_sha256", "input_manifest_sha256"}),
        f"{label} bindings",
    )
    if type(mapping["campaign_id"]) is not str:
        raise PortfolioArtifactError(f"{label} campaign ID type is invalid")
    if not _is_sha256(mapping["contract_sha256"]):
        raise PortfolioArtifactError(f"{label} contract SHA-256 is invalid")
    if not _is_sha256(mapping["input_manifest_sha256"]):
        raise PortfolioArtifactError(f"{label} input manifest SHA-256 is invalid")
    try:
        return Bindings(
            mapping["campaign_id"],
            mapping["contract_sha256"],
            mapping["input_manifest_sha256"],
        )
    except Exception as error:
        raise PortfolioArtifactError(f"{label} bindings are invalid") from error


def _parse_lineage(value: object, label: str) -> Lineage:
    mapping = _exact_keys(
        value,
        frozenset(
            {
                "campaign_id",
                "stage",
                "sequence",
                "parent_manifest_sha256",
                "training_sha256",
                "state_schema_version",
                "runtime_sha256",
            }
        ),
        f"{label} lineage",
    )
    if type(mapping["campaign_id"]) is not str or type(mapping["stage"]) is not str:
        raise PortfolioArtifactError(f"{label} lineage strings have invalid types")
    if type(mapping["sequence"]) is not int or mapping["sequence"] <= 0:
        raise PortfolioArtifactError(f"{label} sequence is invalid")
    if type(mapping["state_schema_version"]) is not int or mapping["state_schema_version"] <= 0:
        raise PortfolioArtifactError(f"{label} state schema version is invalid")
    for field_name in (
        "parent_manifest_sha256",
        "training_sha256",
        "runtime_sha256",
    ):
        if not _is_sha256(mapping[field_name]):
            raise PortfolioArtifactError(f"{label} lineage SHA-256 is invalid")
    try:
        return Lineage(
            mapping["campaign_id"],
            mapping["stage"],
            mapping["sequence"],
            mapping["parent_manifest_sha256"],
            mapping["training_sha256"],
            mapping["state_schema_version"],
            mapping["runtime_sha256"],
        )
    except Exception as error:
        raise PortfolioArtifactError(f"{label} lineage is invalid") from error


def _parse_member_records(
    value: object,
    *,
    label: str,
    reserved: frozenset[str],
) -> dict[str, tuple[str, int]]:
    if type(value) is not dict:
        raise PortfolioArtifactError(f"{label} members must be an object")
    records: dict[str, tuple[str, int]] = {}
    for name, record_value in value.items():
        _validate_member_name(name, reserved=reserved)
        record = _exact_keys(
            record_value, frozenset({"sha256", "size_bytes"}), f"{label} member"
        )
        digest = record["sha256"]
        size = record["size_bytes"]
        if not _is_sha256(digest):
            raise PortfolioArtifactError(f"{label} member SHA-256 is invalid")
        if type(size) is not int or size < 0:
            raise PortfolioArtifactError(f"{label} member size is invalid")
        if size > _MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise PortfolioArtifactError(f"{label} member exceeds the size limit")
        records[name] = (digest, size)
    return records


def _verify_member_values(
    members: Mapping[str, bytes], records: Mapping[str, tuple[str, int]], label: str
) -> Mapping[str, str]:
    if set(members) != set(records):
        raise PortfolioArtifactError(f"{label} manifest members differ from the archive")
    result: dict[str, str] = {}
    for name, value in members.items():
        expected_digest, expected_size = records[name]
        if len(value) != expected_size:
            raise PortfolioArtifactError(f"{label} member size differs: {name}")
        observed = sha256(value).hexdigest()
        if observed != expected_digest:
            raise PortfolioArtifactError(f"{label} member SHA-256 differs: {name}")
        result[name] = observed
    return MappingProxyType(dict(sorted(result.items())))


def _verify_nested_bytes(
    value: bytes, *, kind: str, bindings: Bindings
) -> Mapping[str, str]:
    label = kind.removesuffix("_v1")
    with ZipFile(io.BytesIO(value), "r") as archive:
        infos = _inspect_zip_metadata(
            archive, label=label, required_manifest="manifest.json"
        )
        manifest_bytes = _read_zip_member(archive, infos["manifest.json"], label)
        manifest = _decode_manifest(manifest_bytes, label)
        mapping = _exact_keys(
            manifest,
            frozenset(
                {
                    "schema_version",
                    "artifact_kind",
                    "submission_package",
                    "bindings",
                    "members",
                }
            ),
            f"{label} manifest",
        )
        if type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1:
            raise PortfolioArtifactError(f"{label} schema version is invalid")
        if type(mapping["artifact_kind"]) is not str or mapping["artifact_kind"] != kind:
            raise PortfolioArtifactError(f"{label} artifact kind is invalid")
        if type(mapping["submission_package"]) is not bool or mapping["submission_package"] is not False:
            raise PortfolioArtifactError(f"{label} submission flag is invalid")
        nested_bindings = _parse_bindings(mapping["bindings"], label)
        if nested_bindings != bindings:
            raise PortfolioArtifactError(f"{label} bindings differ from the handoff")
        records = _parse_member_records(
            mapping["members"], label=label, reserved=frozenset({"manifest.json"})
        )
        expected_names = {*records, "manifest.json"}
        if set(infos) != expected_names:
            raise PortfolioArtifactError(f"{label} manifest members differ from the archive")
        member_values = {
            name: _read_zip_member(archive, info, label)
            for name, info in infos.items()
            if name != "manifest.json"
        }
        return _verify_member_values(member_values, records, label)


def _verify_handoff_values(
    members: Mapping[str, bytes], *, path: Path, archive_sha256: str | None
) -> VerifiedHandoff:
    manifest_bytes = members["handoff_manifest.json"]
    manifest_sha256 = sha256(manifest_bytes).hexdigest()
    manifest = _decode_manifest(manifest_bytes, "handoff")
    mapping = _exact_keys(
        manifest,
        frozenset(
            {
                "schema_version",
                "artifact_kind",
                "submission_package",
                "bindings",
                "lineage",
                "members",
            }
        ),
        "handoff manifest",
    )
    if type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1:
        raise PortfolioArtifactError("handoff schema version is invalid")
    if type(mapping["artifact_kind"]) is not str or mapping["artifact_kind"] != "temporal_handoff_v1":
        raise PortfolioArtifactError("handoff artifact kind is invalid")
    if type(mapping["submission_package"]) is not bool or mapping["submission_package"] is not False:
        raise PortfolioArtifactError("handoff submission flag is invalid")
    bindings = _parse_bindings(mapping["bindings"], "handoff")
    lineage = _parse_lineage(mapping["lineage"], "handoff")
    if bindings.campaign_id != lineage.campaign_id:
        raise PortfolioArtifactError("handoff campaign bindings and lineage differ")
    records = _parse_member_records(
        mapping["members"],
        label="handoff",
        reserved=frozenset({"handoff_manifest.json"}),
    )
    if set(records) != set(_PAYLOAD_MEMBERS):
        raise PortfolioArtifactError("handoff manifest members are not exact")
    payloads = {
        name: value for name, value in members.items() if name != "handoff_manifest.json"
    }
    verified_members = _verify_member_values(payloads, records, "handoff")
    review_members = _verify_nested_bytes(
        payloads["review.zip"], kind="temporal_review_v1", bindings=bindings
    )
    resume_members = _verify_nested_bytes(
        payloads["resume.zip"], kind="temporal_resume_v1", bindings=bindings
    )
    return VerifiedHandoff(
        path,
        archive_sha256,
        manifest_sha256,
        lineage.stage,
        lineage.sequence,
        bindings,
        lineage,
        verified_members,
        review_members,
        resume_members,
    )


def _verify_handoff_file(path: Path) -> VerifiedHandoff:
    source = _open_safe_file(Path(path), "handoff")
    try:
        digest = sha256()
        while chunk := source.handle.read(_CHUNK_SIZE):
            digest.update(chunk)
        source.require_unchanged()
        source.handle.seek(0)
        with ZipFile(source.handle, "r") as archive:
            infos = _inspect_zip_metadata(
                archive,
                label="handoff",
                required_manifest="handoff_manifest.json",
                exact_names=_OUTER_MEMBERS,
            )
            values = {
                name: _read_zip_member(archive, info, "handoff")
                for name, info in infos.items()
            }
        source.require_unchanged()
        return _verify_handoff_values(
            values, path=source.path, archive_sha256=digest.hexdigest()
        )
    finally:
        source.handle.close()


def verify_handoff(path: str | Path) -> VerifiedHandoff:
    """Verify a handoff ZIP without extracting or trusting its filename."""

    try:
        return _verify_handoff_file(Path(path))
    except PortfolioArtifactError:
        raise
    except (BadZipFile, OSError, zlib.error, json.JSONDecodeError, UnicodeError, TypeError, ValueError, RuntimeError) as error:
        raise PortfolioArtifactError("temporal portfolio handoff verification failed") from error


def _verify_handoff_directory(path: Path) -> VerifiedHandoff:
    absolute = _absolute_path(path)
    _reject_symlink_components(absolute)
    metadata = absolute.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise PortfolioArtifactError("extracted handoff must be a real directory")
    entries = list(os.scandir(absolute))
    if len(entries) > _MAX_DISCOVERY_ENTRIES:
        raise PortfolioArtifactError("discovery entry limit exceeded")
    if {entry.name for entry in entries} != set(_OUTER_MEMBERS):
        raise PortfolioArtifactError("extracted handoff has unexpected or missing members")
    sizes: dict[str, int] = {}
    for entry in entries:
        entry_stat = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or not stat.S_ISREG(entry_stat.st_mode):
            raise PortfolioArtifactError(
                f"extracted handoff member must be a regular non-symlink file: {entry.name}"
            )
        if entry_stat.st_size > _MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise PortfolioArtifactError("extracted handoff member exceeds the size limit")
        sizes[entry.name] = entry_stat.st_size
    if sum(sizes.values()) > _MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise PortfolioArtifactError("extracted handoff exceeds the total uncompressed size limit")
    values = {
        name: _read_safe_file(
            absolute / name,
            f"extracted handoff member {name}",
            maximum_size=_MAX_MANIFEST_BYTES
            if name == "handoff_manifest.json"
            else _MAX_MEMBER_UNCOMPRESSED_BYTES,
        )
        for name in sorted(sizes)
    }
    return _verify_handoff_values(values, path=absolute, archive_sha256=None)


def _zip_declares_handoff_manifest(
    path: Path, expected_stat: os.stat_result
) -> bool:
    """Inspect central metadata only; never decompress a prospective member."""

    source = _open_safe_file(path, "discovery archive")
    try:
        if (
            not _same_file(expected_stat, source.initial_stat)
            or source.initial_stat.st_size > _MAX_ARCHIVE_BYTES
        ):
            raise PortfolioArtifactError(
                "discovery archive changed before metadata inspection"
            )
        try:
            with ZipFile(source.handle, "r") as archive:
                infos = archive.infolist()
                if len(infos) > _MAX_DISCOVERY_ZIP_MEMBERS:
                    raise PortfolioArtifactError(
                        "discovery ZIP member metadata limit exceeded"
                    )
                declares_manifest = any(
                    info.filename == "handoff_manifest.json" for info in infos
                )
        except BadZipFile:
            declares_manifest = False
        source.require_unchanged()
        return declares_manifest
    finally:
        source.handle.close()


def _discover_candidates(root: Path, budget: _DiscoveryBudget) -> list[Path]:
    absolute = _absolute_path(root)
    _reject_symlink_components(absolute)
    root_stat = absolute.lstat()
    if stat.S_ISLNK(root_stat.st_mode):
        raise PortfolioArtifactError(f"discovery root must not be a symlink: {root}")
    if stat.S_ISREG(root_stat.st_mode):
        budget.add_candidate()
        return [absolute]
    if not stat.S_ISDIR(root_stat.st_mode):
        raise PortfolioArtifactError(f"discovery root is not a file or directory: {root}")

    candidates: list[Path] = []
    queue: list[tuple[Path, int]] = [(absolute, 0)]
    seen_directories: set[tuple[int, int]] = set()
    while queue:
        directory, depth = queue.pop(0)
        directory_stat = directory.lstat()
        identity = (directory_stat.st_dev, directory_stat.st_ino)
        if identity in seen_directories:
            raise PortfolioArtifactError("discovery encountered a directory alias")
        seen_directories.add(identity)
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        budget.add_entries(len(entries))
        names = {entry.name for entry in entries}
        if "handoff_manifest.json" in names:
            budget.add_candidate()
            candidates.append(directory)
            continue
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            if entry.is_symlink() or stat.S_ISLNK(metadata.st_mode):
                raise PortfolioArtifactError(
                    f"discovery does not follow symlink aliases: {entry.path}"
                )
            child = Path(entry.path)
            if stat.S_ISREG(metadata.st_mode):
                if metadata.st_size <= _MAX_ARCHIVE_BYTES:
                    budget.add_inspected_file(metadata.st_size)
                    if _zip_declares_handoff_manifest(child, metadata):
                        budget.add_candidate()
                        candidates.append(child)
            elif stat.S_ISDIR(metadata.st_mode) and depth < _MAX_DISCOVERY_DEPTH:
                queue.append((child, depth + 1))
    return candidates


def _lineage_identity(value: VerifiedHandoff) -> tuple[object, ...]:
    return (
        value.lineage.campaign_id,
        value.lineage.training_sha256,
        value.lineage.state_schema_version,
        value.lineage.runtime_sha256,
    )


def discover_handoffs(roots: Iterable[str | Path]) -> tuple[VerifiedHandoff, ...]:
    """Verify all bounded candidates and return the highest one on a linear chain."""

    try:
        if isinstance(roots, (str, bytes, os.PathLike)):
            raise PortfolioArtifactError("discovery roots must be an iterable of paths")
        root_snapshot = tuple(roots)
        if not root_snapshot:
            raise PortfolioArtifactError("no discovery roots were provided")
        if len(root_snapshot) > _MAX_DISCOVERY_ROOTS:
            raise PortfolioArtifactError("discovery root limit exceeded")
        budget = _DiscoveryBudget()
        candidates: list[Path] = []
        for root in root_snapshot:
            candidates.extend(_discover_candidates(Path(root), budget))
        if not candidates:
            raise PortfolioArtifactError("no temporal portfolio handoff was found")

        verified: list[VerifiedHandoff] = []
        seen_objects: dict[tuple[int, int], Path] = {}
        for candidate in sorted(candidates, key=lambda item: os.fspath(item)):
            metadata = candidate.lstat()
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in seen_objects:
                raise PortfolioArtifactError("discovery encountered a candidate alias")
            seen_objects[identity] = candidate
            if stat.S_ISDIR(metadata.st_mode):
                verified.append(_verify_handoff_directory(candidate))
            else:
                verified.append(_verify_handoff_file(candidate))

        first = verified[0]
        if any(item.bindings != first.bindings for item in verified[1:]):
            raise PortfolioArtifactError("discovered handoff bindings conflict")
        identity = _lineage_identity(first)
        if any(_lineage_identity(item) != identity for item in verified[1:]):
            raise PortfolioArtifactError("discovered handoff lineage identities conflict")

        by_sequence: dict[int, list[VerifiedHandoff]] = {}
        for item in verified:
            by_sequence.setdefault(item.sequence, []).append(item)
        chain: list[VerifiedHandoff] = []
        for sequence in sorted(by_sequence):
            group = by_sequence[sequence]
            hashes = {item.manifest_sha256 for item in group}
            if len(hashes) != 1:
                raise PortfolioArtifactError(
                    "discovered handoffs have equal sequence and different manifest hashes"
                )
            chain.append(min(group, key=lambda item: os.fspath(item.path)))
        stage_rank = {stage: index for index, stage in enumerate(ALLOWED_STAGE_ORDER)}
        for previous, current in zip(chain, chain[1:]):
            if (
                current.sequence != previous.sequence + 1
                or current.lineage.parent_manifest_sha256
                != previous.manifest_sha256
                or stage_rank[current.stage] < stage_rank[previous.stage]
            ):
                raise PortfolioArtifactError(
                    "discovered handoffs form a fork or disconnected parent chain"
                )
        return (chain[-1],)
    except PortfolioArtifactError:
        raise
    except (BadZipFile, OSError, zlib.error, json.JSONDecodeError, UnicodeError, TypeError, ValueError, RuntimeError) as error:
        raise PortfolioArtifactError("temporal portfolio handoff discovery failed") from error


__all__ = [
    "BoundFile",
    "HandoffPath",
    "PortfolioArtifactError",
    "StageEvidence",
    "VerifiedHandoff",
    "bind_file",
    "canonical_json",
    "discover_handoffs",
    "verify_handoff",
    "write_handoff",
]
