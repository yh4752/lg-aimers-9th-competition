from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from experiments.temporal_portfolio.compatibility import (
    COMPATIBLE_RUNTIME_MIGRATIONS,
    CheckpointIdentity,
    CompatibilityError,
    validate_runtime,
)
from experiments.temporal_portfolio.state import (
    ALLOWED_STAGE_ORDER,
    ALLOWED_STATUS,
    Bindings,
    CampaignState,
    Lineage,
    PortfolioStateError,
    validate_transition,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


@pytest.fixture
def bindings() -> Bindings:
    return Bindings("temporal_portfolio_v1", SHA_A, SHA_B)


class _OneViewMapping(Mapping[tuple[str, str], int]):
    def __init__(self, items: tuple[tuple[object, object], ...]) -> None:
        self._items = items
        self.read_count = 0

    def __getitem__(self, key: tuple[str, str]) -> int:
        raise RuntimeError("source access")

    def __iter__(self):
        raise RuntimeError("source access")

    def __len__(self) -> int:
        raise RuntimeError("source access")

    def items(self):
        self.read_count += 1
        if self.read_count > 1:
            raise RuntimeError("second source access")
        return self._items


class _MalformedItemsMapping(Mapping[tuple[str, str], int]):
    def __init__(self, item: object) -> None:
        self._item = item

    def __getitem__(self, key: tuple[str, str]) -> int:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self):
        return (self._item,)


class _ExplodingItemsMapping(Mapping[tuple[str, str], int]):
    def __getitem__(self, key: tuple[str, str]) -> int:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self):
        raise RuntimeError("boom")


def test_allowed_state_values_are_exactly_sealed() -> None:
    assert ALLOWED_STAGE_ORDER == (
        "fresh", "T1", "T2A", "T2B", "T3A", "T3BT4", "T5A", "T5B"
    )
    assert ALLOWED_STATUS == (
        "fresh", "active", "completed", "budget_inconclusive", "failed", "rule_blocked"
    )


def test_fresh_state_has_zero_lineage_and_preserves_bindings(bindings: Bindings) -> None:
    state = CampaignState.fresh(bindings)

    assert state == CampaignState("fresh", "fresh", 0, None, bindings)
    assert state.lineage == Lineage("temporal_portfolio_v1", 0, None)
    assert state.bindings is bindings


def test_state_sequence_and_parent_are_monotonic(bindings: Bindings) -> None:
    fresh = CampaignState.fresh(bindings)
    next_state = fresh.advance(
        stage="T1", status="active", parent_manifest_sha256=SHA_C
    )
    assert next_state.sequence == 1
    assert next_state.parent_manifest_sha256 == SHA_C
    with pytest.raises(PortfolioStateError, match="sequence"):
        validate_transition(next_state, replace(next_state, sequence=0))


def test_state_can_progress_within_a_stage_and_skip_forward(bindings: Bindings) -> None:
    active = CampaignState.fresh(bindings).advance(
        stage="T1", status="active", parent_manifest_sha256=SHA_C
    )
    completed = active.advance(
        stage="T1", status="completed", parent_manifest_sha256=SHA_D
    )
    skipped = completed.advance(
        stage="T3A", status="active", parent_manifest_sha256=SHA_A
    )

    assert completed.sequence == 2
    assert skipped.stage == "T3A"
    assert skipped.sequence == 3


def test_state_rejects_backtracking(bindings: Bindings) -> None:
    later = CampaignState("T3A", "active", 3, SHA_C, bindings)

    with pytest.raises(PortfolioStateError, match="backward"):
        later.advance(stage="T2B", status="active", parent_manifest_sha256=SHA_D)


def test_transition_rejects_binding_drift(bindings: Bindings) -> None:
    old = CampaignState.fresh(bindings)
    drifted = Bindings("temporal_portfolio_v1", SHA_C, SHA_B)
    new = CampaignState("T1", "active", 1, SHA_D, drifted)

    with pytest.raises(PortfolioStateError, match="bindings"):
        validate_transition(old, new)


@pytest.mark.parametrize(
    "field,value",
    (
        ("campaign_id", "other"),
        ("campaign_id", 7),
        ("contract_sha256", "A" * 64),
        ("contract_sha256", "a" * 63),
        ("input_manifest_sha256", "g" * 64),
        ("input_manifest_sha256", b"b" * 64),
    ),
)
def test_bindings_reject_invalid_identity_fields(field: str, value: object) -> None:
    values: dict[str, object] = {
        "campaign_id": "temporal_portfolio_v1",
        "contract_sha256": SHA_A,
        "input_manifest_sha256": SHA_B,
    }
    values[field] = value

    with pytest.raises(PortfolioStateError):
        Bindings(**values)


@pytest.mark.parametrize(
    "args",
    (
        (7, "active", 1, SHA_C),
        ("T1", 7, 1, SHA_C),
        ("unknown", "active", 1, SHA_C),
        ("T1", "unknown", 1, SHA_C),
        ("T1", "active", True, SHA_C),
        ("T1", "active", 0, SHA_C),
        ("T1", "active", 1, "C" * 64),
        ("T1", "active", 1, None),
    ),
)
def test_campaign_state_rejects_invalid_nonfresh_values(
    bindings: Bindings, args: tuple[object, object, object, object]
) -> None:
    with pytest.raises(PortfolioStateError):
        CampaignState(*args, bindings)


@pytest.mark.parametrize(
    "args",
    (
        ("fresh", "active", 0, None),
        ("T1", "fresh", 1, SHA_C),
        ("fresh", "fresh", 1, None),
        ("fresh", "fresh", 0, SHA_C),
    ),
)
def test_campaign_state_rejects_incoherent_stage_status_and_lineage(
    bindings: Bindings, args: tuple[object, object, object, object]
) -> None:
    with pytest.raises(PortfolioStateError):
        CampaignState(*args, bindings)


def test_campaign_state_rejects_constructor_and_transition_type_misuse(
    bindings: Bindings,
) -> None:
    with pytest.raises(PortfolioStateError, match="bindings"):
        CampaignState("fresh", "fresh", 0, None, object())
    with pytest.raises(PortfolioStateError, match="bindings"):
        CampaignState.fresh(object())
    with pytest.raises(PortfolioStateError, match="state type"):
        validate_transition(object(), CampaignState.fresh(bindings))
    with pytest.raises(PortfolioStateError, match="state type"):
        validate_transition(CampaignState.fresh(bindings), object())


def test_invalid_stage_is_wrapped_as_domain_error_not_value_error(
    bindings: Bindings,
) -> None:
    forged = object.__new__(CampaignState)
    object.__setattr__(forged, "stage", "unknown")
    object.__setattr__(forged, "status", "active")
    object.__setattr__(forged, "sequence", 1)
    object.__setattr__(forged, "parent_manifest_sha256", SHA_C)
    object.__setattr__(forged, "bindings", bindings)

    with pytest.raises(PortfolioStateError, match="stage"):
        validate_transition(CampaignState.fresh(bindings), forged)


def test_advance_validates_forged_receiver_before_sequence_arithmetic(
    bindings: Bindings,
) -> None:
    forged = object.__new__(CampaignState)
    object.__setattr__(forged, "stage", "T1")
    object.__setattr__(forged, "status", "active")
    object.__setattr__(forged, "sequence", "1")
    object.__setattr__(forged, "parent_manifest_sha256", SHA_C)
    object.__setattr__(forged, "bindings", bindings)

    with pytest.raises(PortfolioStateError, match="sequence"):
        forged.advance(stage="T2A", status="active", parent_manifest_sha256=SHA_D)


@pytest.mark.parametrize(
    "args",
    (
        ("other", 1, SHA_A),
        (7, 1, SHA_A),
        ("temporal_portfolio_v1", True, SHA_A),
        ("temporal_portfolio_v1", -1, None),
        ("temporal_portfolio_v1", 0, SHA_A),
        ("temporal_portfolio_v1", 1, None),
        ("temporal_portfolio_v1", 1, "A" * 64),
    ),
)
def test_lineage_rejects_invalid_or_incoherent_values(
    args: tuple[object, object, object]
) -> None:
    with pytest.raises(PortfolioStateError):
        Lineage(*args)


def test_state_value_objects_are_immutable(bindings: Bindings) -> None:
    state = CampaignState.fresh(bindings)
    lineage = state.lineage

    for value, field in (
        (bindings, "campaign_id"),
        (state, "sequence"),
        (lineage, "parent_manifest_sha256"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, "changed")


@pytest.mark.parametrize(
    "args",
    (
        ("A" * 64, SHA_B, 1),
        (SHA_A, "b" * 63, 1),
        (b"a" * 64, SHA_B, 1),
        (SHA_A, SHA_B, True),
        (SHA_A, SHA_B, 0),
        (SHA_A, SHA_B, -1),
    ),
)
def test_checkpoint_identity_rejects_invalid_fields(
    args: tuple[object, object, object]
) -> None:
    with pytest.raises(CompatibilityError):
        CheckpointIdentity(*args)


def test_runtime_same_hash_needs_no_migration() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    validate_runtime(checkpoint, current_runtime_sha256=SHA_B, migrations={})
    validate_runtime(checkpoint, current_runtime_sha256=SHA_B)


def test_runtime_change_needs_explicit_compatibility() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)
    with pytest.raises(CompatibilityError, match="runtime"):
        validate_runtime(checkpoint, current_runtime_sha256=SHA_C, migrations={})
    validate_runtime(
        checkpoint,
        current_runtime_sha256=SHA_C,
        migrations={(SHA_B, SHA_C): 1},
    )


@pytest.mark.parametrize(
    "migrations",
    (
        {(SHA_B, SHA_D): 1},
        {(SHA_B, SHA_C): 2},
    ),
)
def test_runtime_rejects_migration_pair_or_schema_mismatch(
    migrations: Mapping[tuple[str, str], int],
) -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="runtime"):
        validate_runtime(
            checkpoint, current_runtime_sha256=SHA_C, migrations=migrations
        )


def test_runtime_snapshots_caller_mapping_exactly_once() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)
    migrations = _OneViewMapping((((SHA_B, SHA_C), 1),))

    validate_runtime(
        checkpoint, current_runtime_sha256=SHA_C, migrations=migrations
    )

    assert migrations.read_count == 1


@pytest.mark.parametrize(
    "items",
    (
        (((SHA_B, SHA_C), True),),
        (((SHA_B, SHA_C), 0),),
        (((SHA_B.upper(), SHA_C), 1),),
        (((SHA_B, SHA_C.encode()), 1),),
        (((SHA_B,), 1),),
        (([SHA_B, SHA_C], 1),),
        (((SHA_B, SHA_C), 1), ((SHA_B, SHA_C), 1)),
    ),
)
def test_runtime_rejects_invalid_migration_entries(
    items: tuple[tuple[object, object], ...],
) -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="migration"):
        validate_runtime(
            checkpoint,
            current_runtime_sha256=SHA_C,
            migrations=_OneViewMapping(items),
        )


@pytest.mark.parametrize("item", ("ab", ["a", "b"], ("only-one",)))
def test_runtime_rejects_malformed_mapping_items(item: object) -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="migration"):
        validate_runtime(
            checkpoint,
            current_runtime_sha256=SHA_C,
            migrations=_MalformedItemsMapping(item),
        )


def test_runtime_rejects_mapping_snapshot_failures() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="snapshot"):
        validate_runtime(
            checkpoint,
            current_runtime_sha256=SHA_C,
            migrations=_ExplodingItemsMapping(),
        )


def test_runtime_validates_migrations_even_when_runtime_matches() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="migration"):
        validate_runtime(
            checkpoint,
            current_runtime_sha256=SHA_B,
            migrations={(SHA_B, SHA_C): True},
        )


def test_runtime_rejects_invalid_call_types() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="checkpoint"):
        validate_runtime(object(), current_runtime_sha256=SHA_B, migrations={})
    with pytest.raises(CompatibilityError, match="current runtime"):
        validate_runtime(checkpoint, current_runtime_sha256=b"b" * 64, migrations={})
    with pytest.raises(CompatibilityError, match="mapping"):
        validate_runtime(checkpoint, current_runtime_sha256=SHA_B, migrations=[])


def test_production_runtime_migration_registry_is_empty_and_immutable() -> None:
    assert isinstance(COMPATIBLE_RUNTIME_MIGRATIONS, MappingProxyType)
    assert dict(COMPATIBLE_RUNTIME_MIGRATIONS) == {}
    with pytest.raises(TypeError):
        COMPATIBLE_RUNTIME_MIGRATIONS[(SHA_B, SHA_C)] = 1


def test_checkpoint_identity_is_immutable() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(FrozenInstanceError):
        checkpoint.state_schema_version = 2
