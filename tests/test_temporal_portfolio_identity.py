from collections.abc import Mapping
from types import MappingProxyType

import pytest

from experiments.temporal_portfolio.identity import TrainingIdentity, audit_duplicate


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "data_rows": "a" * 64,
        "train_seasons": [2021],
        "valid_year": 2022,
        "decay": None,
        "features": ["base"],
        "model": {"profile": "p2", "config": {"alpha": 1, "beta": [True]}},
        "loss": "bce",
        "seed": 3407,
    }
    payload.update(overrides)
    return payload


class _ItemsSnapshotMapping(Mapping[str, object]):
    def __init__(self, items: tuple[tuple[str, object], ...], getitem_value: object) -> None:
        self._items = items
        self._getitem_value = getitem_value

    def __getitem__(self, key: str) -> object:
        if isinstance(self._getitem_value, Exception):
            raise self._getitem_value
        return self._getitem_value

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def items(self):
        return self._items


def test_training_identity_changes_only_for_semantic_changes() -> None:
    base = TrainingIdentity.from_payload({
        "data_rows": "a" * 64, "train_seasons": [2021], "valid_year": 2022,
        "decay": None, "features": ["base"], "model": {"profile": "p2"},
        "loss": "bce", "seed": 3407,
    })
    same = TrainingIdentity.from_payload(dict(base.payload))
    changed = TrainingIdentity.from_payload({**dict(base.payload), "decay": "0.55"})
    assert base.sha256 == same.sha256
    assert base.sha256 != changed.sha256
    assert audit_duplicate(base, {base.sha256: "jobs/old"}) == "jobs/old"


def test_training_identity_cannot_be_directly_constructed_with_unbound_state() -> None:
    with pytest.raises(TypeError):
        TrainingIdentity(payload={"mutable": []}, sha256="a" * 64)


def test_training_identity_requires_from_payload_factory() -> None:
    with pytest.raises(TypeError, match="from_payload"):
        TrainingIdentity()


def test_audit_duplicate_rejects_forged_identity_with_mismatched_payload_and_digest() -> None:
    forged = object.__new__(TrainingIdentity)
    object.__setattr__(forged, "payload", _payload())
    object.__setattr__(forged, "sha256", "b" * 64)

    with pytest.raises(ValueError):
        audit_duplicate(forged, {"b" * 64: "jobs/unrelated"})


def test_audit_duplicate_rejects_mutable_forged_payload_with_correct_digest() -> None:
    verified = TrainingIdentity.from_payload(_payload(model={"layers": [32]}))
    forged = object.__new__(TrainingIdentity)
    object.__setattr__(forged, "payload", _payload(model={"layers": [32]}))
    object.__setattr__(forged, "sha256", verified.sha256)

    with pytest.raises(ValueError, match="frozen representation"):
        audit_duplicate(forged, {verified.sha256: "jobs/unrelated"})


@pytest.mark.parametrize(
    "payload",
    (
        {key: value for key, value in _payload().items() if key != "seed"},
        _payload(unexpected="value"),
    ),
)
def test_training_identity_requires_exact_root_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainingIdentity.from_payload(payload)


@pytest.mark.parametrize("data_rows", ("A" * 64, "g" * 64, "a" * 63))
def test_training_identity_requires_lowercase_sha256_source_hash(data_rows: str) -> None:
    with pytest.raises(ValueError):
        TrainingIdentity.from_payload(_payload(data_rows=data_rows))


@pytest.mark.parametrize("field", ("valid_year", "seed"))
def test_training_identity_rejects_booleans_for_integer_fields(field: str) -> None:
    with pytest.raises(ValueError):
        TrainingIdentity.from_payload(_payload(**{field: True}))


@pytest.mark.parametrize(
    "model",
    (
        {"weight": float("nan")},
        {"weight": float("inf")},
        {"weight": float("-inf")},
        {"nested": {"weight": float("nan")}},
    ),
)
def test_training_identity_rejects_nonfinite_nested_model_values(model: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainingIdentity.from_payload(_payload(model=model))


@pytest.mark.parametrize(
    "field,value",
    (
        ("features", {"base"}),
        ("features", ("base",)),
        ("model", {"layers": (32, 16)}),
        ("model", {"layers": {32, 16}}),
    ),
)
def test_training_identity_rejects_external_sets_and_tuples(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        TrainingIdentity.from_payload(_payload(**{field: value}))


def test_training_identity_preserves_feature_order_in_hash() -> None:
    first = TrainingIdentity.from_payload(_payload(features=["base", "recent"]))
    second = TrainingIdentity.from_payload(_payload(features=["recent", "base"]))

    assert first.sha256 != second.sha256


def test_training_identity_freezes_nested_values_and_accepts_its_own_frozen_payload() -> None:
    identity = TrainingIdentity.from_payload(
        _payload(model={"profile": "p2", "config": {"layers": [32, True]}})
    )

    assert isinstance(identity.payload, MappingProxyType)
    assert isinstance(identity.payload["model"], MappingProxyType)
    assert identity.payload["model"]["config"]["layers"] == (32, True)
    with pytest.raises(TypeError):
        identity.payload["model"]["config"]["new"] = "value"
    with pytest.raises(AttributeError):
        identity.payload["model"]["config"]["layers"].append(64)
    assert TrainingIdentity.from_payload(dict(identity.payload)).sha256 == identity.sha256


@pytest.mark.parametrize(
    "completed",
    (
        {"not-a-hash": "jobs/old"},
        {"a" * 64: 7},
    ),
)
def test_audit_duplicate_rejects_malformed_completed_entries(completed: dict[str, object]) -> None:
    identity = TrainingIdentity.from_payload(_payload())

    with pytest.raises(ValueError):
        audit_duplicate(identity, completed)


def test_audit_duplicate_returns_none_when_no_identity_matches() -> None:
    identity = TrainingIdentity.from_payload(_payload())

    assert audit_duplicate(identity, {"b" * 64: "jobs/other"}) is None


def test_audit_duplicate_uses_the_validated_completed_snapshot() -> None:
    identity = TrainingIdentity.from_payload(_payload())
    completed = _ItemsSnapshotMapping(((identity.sha256, "jobs/old"),), 7)

    assert audit_duplicate(identity, completed) == "jobs/old"


def test_training_identity_uses_root_items_snapshot_without_getitem() -> None:
    expected = TrainingIdentity.from_payload(_payload())
    volatile = _ItemsSnapshotMapping(tuple(_payload().items()), KeyError("volatile"))

    assert TrainingIdentity.from_payload(volatile).sha256 == expected.sha256


def test_training_identity_rejects_duplicate_nested_mapping_items() -> None:
    duplicate_model = _ItemsSnapshotMapping(
        (("profile", "p2"), ("profile", "p3")), "not-the-snapshot"
    )

    with pytest.raises(ValueError):
        TrainingIdentity.from_payload(_payload(model=duplicate_model))


@pytest.mark.parametrize("path", ("", " ", "\t"))
def test_audit_duplicate_rejects_empty_or_whitespace_completed_paths(path: str) -> None:
    identity = TrainingIdentity.from_payload(_payload())

    with pytest.raises(ValueError):
        audit_duplicate(identity, {identity.sha256: path})


def test_training_identity_uses_a_pinned_canonical_digest_for_reordered_mappings() -> None:
    expected_sha256 = "347604821446076c9e3c368c495a4b70ac190a933c3ef2ef8b4ace322bcb29bb"
    standard = _payload()
    reordered_root = dict(reversed(tuple(standard.items())))
    reordered_nested = _payload(model={"config": {"beta": [True], "alpha": 1}, "profile": "p2"})

    assert TrainingIdentity.from_payload(standard).sha256 == expected_sha256
    assert TrainingIdentity.from_payload(reordered_root).sha256 == expected_sha256
    assert TrainingIdentity.from_payload(reordered_nested).sha256 == expected_sha256


@pytest.mark.parametrize(
    "field,value",
    (
        ("train_seasons", []),
        ("train_seasons", [True]),
        ("train_seasons", [2021, 2021]),
        ("train_seasons", [2022, 2021]),
        ("train_seasons", [1899]),
        ("valid_year", True),
        ("valid_year", 2101),
        ("valid_year", 2021),
        ("decay", 0.55),
        ("decay", "0.99"),
        ("decay", "0.5"),
        ("features", []),
        ("features", ["base", "base"]),
        ("features", [""]),
        ("features", [1]),
        ("model", {}),
        ("model", {"blob": b"x"}),
        ("model", {"values": frozenset({1})}),
        ("model", {1: "value"}),
        ("loss", "logloss"),
        ("seed", -1),
    ),
)
def test_training_identity_rejects_invalid_spec_boundary(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        TrainingIdentity.from_payload(_payload(**{field: value}))


def test_audit_duplicate_requires_a_mapping_for_completed_jobs() -> None:
    identity = TrainingIdentity.from_payload(_payload())

    with pytest.raises(ValueError):
        audit_duplicate(identity, [(identity.sha256, "jobs/old")])
