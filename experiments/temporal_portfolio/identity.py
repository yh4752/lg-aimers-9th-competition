from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping


class PortfolioIdentityError(ValueError):
    """Raised when a training identity cannot be represented canonically."""


_ROOT_FIELDS = frozenset(
    {
        "data_rows",
        "train_seasons",
        "valid_year",
        "decay",
        "features",
        "model",
        "loss",
        "seed",
    }
)
_DECAYS = frozenset({"0.40", "0.55", "0.70", "1.00"})
_LOSSES = frozenset({"bce", "brier"})


class _FrozenList(tuple[object, ...]):
    """Private marker for list values frozen by this module.

    A normal tuple is deliberately not accepted as JSON input.  This marker
    allows ``dict(identity.payload)`` to be fed back to ``from_payload``.
    """


@dataclass(frozen=True, init=False)
class TrainingIdentity:
    payload: Mapping[str, object]
    sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("TrainingIdentity instances must be created with from_payload()")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> TrainingIdentity:
        normalized = _normalize_payload(payload)
        canonical_bytes = _canonical_json_bytes(normalized)
        identity = object.__new__(cls)
        object.__setattr__(identity, "payload", _freeze_json(normalized))
        object.__setattr__(identity, "sha256", hashlib.sha256(canonical_bytes).hexdigest())
        return identity


def audit_duplicate(identity: TrainingIdentity, completed: Mapping[str, str]) -> str | None:
    """Return a prior completed path for the exact identity, if one exists."""
    if not isinstance(identity, TrainingIdentity):
        raise PortfolioIdentityError("identity has an invalid type")
    try:
        supplied_payload = identity.payload
        supplied_sha256 = identity.sha256
        verified = TrainingIdentity.from_payload(supplied_payload)
    except (AttributeError, PortfolioIdentityError) as error:
        raise PortfolioIdentityError("identity has an invalid payload") from error
    if not _same_frozen_json(supplied_payload, verified.payload):
        raise PortfolioIdentityError("identity payload is not the frozen representation")
    if type(supplied_sha256) is not str or supplied_sha256 != verified.sha256:
        raise PortfolioIdentityError("identity payload and SHA-256 differ")
    completed_snapshot = _snapshot_mapping(completed, "completed jobs")
    for digest, path in completed_snapshot.items():
        if not _is_sha256(digest) or type(path) is not str or not path.strip():
            raise PortfolioIdentityError("completed jobs contain an invalid hash or path")
    path = completed_snapshot.get(supplied_sha256)
    return None if path is None else str(path)


def _normalize_payload(payload: Mapping[str, object]) -> dict[str, object]:
    root = _snapshot_mapping(payload, "training identity payload")
    if set(root) != _ROOT_FIELDS:
        raise PortfolioIdentityError("training identity fields differ")

    data_rows = root["data_rows"]
    if not _is_sha256(data_rows):
        raise PortfolioIdentityError("data_rows must be a lowercase SHA-256")

    train_seasons = _list(root["train_seasons"], "train_seasons")
    if not train_seasons or any(type(year) is not int or not _is_year(year) for year in train_seasons):
        raise PortfolioIdentityError("train_seasons must contain four-digit integer years")
    if any(left >= right for left, right in zip(train_seasons, train_seasons[1:])):
        raise PortfolioIdentityError("train_seasons must be strictly increasing")

    valid_year = root["valid_year"]
    if type(valid_year) is not int or not _is_year(valid_year) or valid_year <= train_seasons[-1]:
        raise PortfolioIdentityError("valid_year must follow all training seasons")

    decay = root["decay"]
    if decay is not None and (type(decay) is not str or decay not in _DECAYS):
        raise PortfolioIdentityError("decay must be an approved decimal string or null")

    features = _list(root["features"], "features")
    if (
        not features
        or any(type(feature) is not str or not feature for feature in features)
        or len(set(features)) != len(features)
    ):
        raise PortfolioIdentityError("features must be unique non-empty strings")

    model = _normalize_json(root["model"], "model")
    if type(model) is not dict or not model:
        raise PortfolioIdentityError("model must be a non-empty object")

    loss = root["loss"]
    if type(loss) is not str or loss not in _LOSSES:
        raise PortfolioIdentityError("loss is not approved")

    seed = root["seed"]
    if type(seed) is not int or seed < 0:
        raise PortfolioIdentityError("seed must be a non-negative integer")

    return {
        "data_rows": data_rows,
        "train_seasons": train_seasons,
        "valid_year": valid_year,
        "decay": decay,
        "features": features,
        "model": model,
        "loss": loss,
        "seed": seed,
    }


def _list(value: object, label: str) -> list[object]:
    if type(value) is list:
        return list(value)
    if isinstance(value, _FrozenList):
        return list(value)
    raise PortfolioIdentityError(f"{label} must be a JSON array")


def _normalize_json(value: object, label: str) -> object:
    if value is None or type(value) is bool or type(value) is str or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PortfolioIdentityError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        snapshot = _snapshot_mapping(value, label)
        normalized: dict[str, object] = {}
        for key, nested in snapshot.items():
            normalized[key] = _normalize_json(nested, f"{label}.{key}")
        return normalized
    if type(value) is list or isinstance(value, _FrozenList):
        return [_normalize_json(item, label) for item in value]
    raise PortfolioIdentityError(f"{label} contains an unsupported JSON value")


def _snapshot_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PortfolioIdentityError(f"{label} must be a mapping")
    snapshot: dict[str, object] = {}
    try:
        for item in value.items():
            key, nested = item
            if type(key) is not str:
                raise PortfolioIdentityError(f"{label} has a non-string object key")
            if key in snapshot:
                raise PortfolioIdentityError(f"{label} has a duplicate object key")
            snapshot[key] = nested
    except PortfolioIdentityError:
        raise
    except Exception as error:
        raise PortfolioIdentityError(f"{label} mapping snapshot failed") from error
    return snapshot


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(nested) for key, nested in value.items()})
    if type(value) is list:
        return _FrozenList(_freeze_json(item) for item in value)
    return value


def _same_frozen_json(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is MappingProxyType:
        return left.keys() == right.keys() and all(
            _same_frozen_json(left[key], right[key]) for key in left
        )
    if type(left) is _FrozenList:
        return len(left) == len(right) and all(
            _same_frozen_json(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PortfolioIdentityError("payload cannot be represented as canonical JSON") from error


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_year(value: int) -> bool:
    return 1900 <= value <= 2100
