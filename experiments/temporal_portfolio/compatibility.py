"""Fail-closed checkpoint runtime compatibility validation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


class CompatibilityError(ValueError):
    """Raised when checkpoint and current runtime identities are incompatible."""


STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointIdentity:
    training_sha256: str
    runtime_sha256: str
    state_schema_version: int

    def __post_init__(self) -> None:
        _validate_checkpoint(self)


COMPATIBLE_RUNTIME_MIGRATIONS: Mapping[tuple[str, str], int] = MappingProxyType({})


def validate_runtime(
    checkpoint: CheckpointIdentity,
    *,
    current_training_sha256: str,
    current_runtime_sha256: str,
) -> None:
    _validate_runtime_with_migrations(
        checkpoint,
        current_training_sha256=current_training_sha256,
        current_runtime_sha256=current_runtime_sha256,
        migrations=COMPATIBLE_RUNTIME_MIGRATIONS,
    )


def _validate_runtime_with_migrations(
    checkpoint: CheckpointIdentity,
    *,
    current_training_sha256: str,
    current_runtime_sha256: str,
    migrations: Mapping[tuple[str, str], int],
) -> None:
    if type(checkpoint) is not CheckpointIdentity:
        raise CompatibilityError("checkpoint identity has an invalid type")
    _validate_checkpoint(checkpoint)
    if not _is_sha256(current_training_sha256):
        raise CompatibilityError("current training SHA-256 is invalid")
    if not _is_sha256(current_runtime_sha256):
        raise CompatibilityError("current runtime SHA-256 is invalid")
    if checkpoint.training_sha256 != current_training_sha256:
        raise CompatibilityError("checkpoint training identity differs")
    if checkpoint.state_schema_version != STATE_SCHEMA_VERSION:
        raise CompatibilityError("checkpoint state schema version is unsupported")
    migration_snapshot = _snapshot_migrations(migrations)
    if checkpoint.runtime_sha256 == current_runtime_sha256:
        return
    key = (checkpoint.runtime_sha256, current_runtime_sha256)
    if migration_snapshot.get(key) != STATE_SCHEMA_VERSION:
        raise CompatibilityError(
            "checkpoint runtime differs without an exact tested migration"
        )


def _validate_checkpoint(value: object) -> None:
    if type(value) is not CheckpointIdentity:
        raise CompatibilityError("checkpoint identity has an invalid type")
    try:
        training_sha256 = value.training_sha256
        runtime_sha256 = value.runtime_sha256
        state_schema_version = value.state_schema_version
    except AttributeError as error:
        raise CompatibilityError("checkpoint identity is malformed") from error
    if not _is_sha256(training_sha256):
        raise CompatibilityError("checkpoint training SHA-256 is invalid")
    if not _is_sha256(runtime_sha256):
        raise CompatibilityError("checkpoint runtime SHA-256 is invalid")
    if type(state_schema_version) is not int or state_schema_version <= 0:
        raise CompatibilityError("checkpoint state schema version must be positive")


def _snapshot_migrations(value: object) -> dict[tuple[str, str], int]:
    if not isinstance(value, Mapping):
        raise CompatibilityError("runtime migrations must be a mapping")
    snapshot: dict[tuple[str, str], int] = {}
    try:
        for item in value.items():
            if type(item) is not tuple or len(item) != 2:
                raise CompatibilityError("runtime migration has a malformed item")
            key, schema_version = item
            if (
                type(key) is not tuple
                or len(key) != 2
                or not _is_sha256(key[0])
                or not _is_sha256(key[1])
            ):
                raise CompatibilityError("runtime migration key is invalid")
            if type(schema_version) is not int or schema_version <= 0:
                raise CompatibilityError("runtime migration schema version is invalid")
            if key in snapshot:
                raise CompatibilityError("runtime migration key is duplicated")
            snapshot[key] = schema_version
    except CompatibilityError:
        raise
    except Exception as error:
        raise CompatibilityError("runtime migration mapping snapshot failed") from error
    return snapshot


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
