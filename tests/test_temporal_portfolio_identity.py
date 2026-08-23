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
