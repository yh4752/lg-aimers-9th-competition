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
        "model": {"profile": "p2"},
        "loss": "bce",
        "seed": 3407,
    }
    payload.update(overrides)
    return payload


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


def test_audit_duplicate_rejects_forged_identity_with_mismatched_payload_and_digest() -> None:
    forged = object.__new__(TrainingIdentity)
    object.__setattr__(forged, "payload", _payload())
    object.__setattr__(forged, "sha256", "b" * 64)

    with pytest.raises(ValueError):
        audit_duplicate(forged, {"b" * 64: "jobs/unrelated"})


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
