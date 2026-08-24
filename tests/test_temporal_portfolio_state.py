from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from experiments.temporal_portfolio.compatibility import (
    COMPATIBLE_RUNTIME_MIGRATIONS,
    STATE_SCHEMA_VERSION,
    CheckpointIdentity,
    CompatibilityError,
    _validate_runtime_with_migrations,
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
    assert STATE_SCHEMA_VERSION == 1


def test_fresh_state_has_zero_sequence_and_preserves_bindings(bindings: Bindings) -> None:
    state = CampaignState.fresh(bindings)

    assert state == CampaignState("fresh", "fresh", 0, None, bindings)
    assert state.bindings is bindings


def test_state_sequence_and_parent_are_monotonic(bindings: Bindings) -> None:
    fresh = CampaignState.fresh(bindings)
    next_state = fresh.advance(
        stage="T1",
        status="active",
        verified_parent_manifest_sha256=SHA_B,
    )
    assert next_state.sequence == 1
    assert next_state.parent_manifest_sha256 == bindings.input_manifest_sha256
    candidate = CampaignState("T1", "completed", 1, SHA_C, bindings)
    with pytest.raises(PortfolioStateError, match="sequence"):
        validate_transition(
            next_state, candidate, expected_parent_manifest_sha256=SHA_C
        )


def test_state_can_progress_within_a_stage_and_skip_forward(bindings: Bindings) -> None:
    active = CampaignState.fresh(bindings).advance(
        stage="T1",
        status="active",
        verified_parent_manifest_sha256=SHA_B,
    )
    completed = active.advance(
        stage="T1",
        status="completed",
        verified_parent_manifest_sha256=SHA_D,
    )
    skipped = completed.advance(
        stage="T3A",
        status="active",
        verified_parent_manifest_sha256=SHA_A,
    )

    assert completed.sequence == 2
    assert skipped.stage == "T3A"
    assert skipped.sequence == 3


@pytest.mark.parametrize(
    "status", ("active", "completed", "budget_inconclusive", "failed", "rule_blocked")
)
def test_fresh_state_enters_t1_with_active_or_immediate_terminal_status(
    bindings: Bindings, status: str
) -> None:
    state = CampaignState.fresh(bindings).advance(
        stage="T1",
        status=status,
        verified_parent_manifest_sha256=bindings.input_manifest_sha256,
    )

    assert (state.stage, state.status) == ("T1", status)


def test_fresh_state_cannot_skip_t1(bindings: Bindings) -> None:
    with pytest.raises(PortfolioStateError, match="T1"):
        CampaignState.fresh(bindings).advance(
            stage="T2A",
            status="active",
            verified_parent_manifest_sha256=bindings.input_manifest_sha256,
        )


@pytest.mark.parametrize(
    "status", ("active", "completed", "budget_inconclusive", "failed", "rule_blocked")
)
def test_active_state_may_stay_active_or_finish_same_stage(
    bindings: Bindings, status: str
) -> None:
    active = CampaignState("T1", "active", 1, SHA_B, bindings)

    successor = active.advance(
        stage="T1", status=status, verified_parent_manifest_sha256=SHA_C
    )

    assert successor.status == status


@pytest.mark.parametrize("retry_status", ("budget_inconclusive", "failed"))
def test_inconclusive_or_failed_state_retries_same_stage_before_finishing(
    bindings: Bindings, retry_status: str
) -> None:
    stopped = CampaignState("T2A", retry_status, 2, SHA_B, bindings)

    active = stopped.advance(
        stage="T2A", status="active", verified_parent_manifest_sha256=SHA_C
    )
    completed = active.advance(
        stage="T2A", status="completed", verified_parent_manifest_sha256=SHA_D
    )

    assert (active.status, completed.status) == ("active", "completed")


@pytest.mark.parametrize("retry_status", ("budget_inconclusive", "failed"))
@pytest.mark.parametrize(
    "next_status", ("completed", "budget_inconclusive", "failed", "rule_blocked")
)
def test_retryable_state_cannot_jump_directly_to_a_terminal_status(
    bindings: Bindings, retry_status: str, next_status: str
) -> None:
    stopped = CampaignState("T2A", retry_status, 2, SHA_B, bindings)

    with pytest.raises(PortfolioStateError, match="retry"):
        stopped.advance(
            stage="T2A",
            status=next_status,
            verified_parent_manifest_sha256=SHA_C,
        )


def test_completed_state_may_only_remain_completed_on_same_stage(
    bindings: Bindings,
) -> None:
    completed = CampaignState("T2A", "completed", 2, SHA_B, bindings)

    repeated = completed.advance(
        stage="T2A", status="completed", verified_parent_manifest_sha256=SHA_C
    )
    assert repeated.status == "completed"
    with pytest.raises(PortfolioStateError, match="completed"):
        completed.advance(
            stage="T2A", status="active", verified_parent_manifest_sha256=SHA_C
        )


@pytest.mark.parametrize(
    "status", ("active", "completed", "budget_inconclusive", "failed", "rule_blocked")
)
def test_completed_state_may_advance_to_immediate_outcome(
    bindings: Bindings, status: str
) -> None:
    completed = CampaignState("T1", "completed", 2, SHA_B, bindings)

    successor = completed.advance(
        stage="T3A", status=status, verified_parent_manifest_sha256=SHA_C
    )

    assert (successor.stage, successor.status) == ("T3A", status)


@pytest.mark.parametrize("status", ("active", "budget_inconclusive", "failed"))
def test_noncompleted_state_cannot_move_forward(
    bindings: Bindings, status: str
) -> None:
    state = CampaignState("T1", status, 1, SHA_B, bindings)

    with pytest.raises(PortfolioStateError, match="completed"):
        state.advance(
            stage="T2A", status="active", verified_parent_manifest_sha256=SHA_C
        )


def test_rule_blocked_state_has_no_successor(bindings: Bindings) -> None:
    blocked = CampaignState("T1", "rule_blocked", 1, SHA_B, bindings)

    for stage in ("T1", "T2A"):
        with pytest.raises(PortfolioStateError, match="rule_blocked"):
            blocked.advance(
                stage=stage,
                status="completed",
                verified_parent_manifest_sha256=SHA_C,
            )


def test_state_rejects_backtracking(bindings: Bindings) -> None:
    later = CampaignState("T3A", "active", 3, SHA_C, bindings)

    with pytest.raises(PortfolioStateError, match="backward"):
        later.advance(
            stage="T2B",
            status="active",
            verified_parent_manifest_sha256=SHA_D,
        )


def test_transition_rejects_binding_drift(bindings: Bindings) -> None:
    old = CampaignState.fresh(bindings)
    drifted = Bindings("temporal_portfolio_v1", SHA_C, SHA_B)
    new = CampaignState("T1", "active", 1, SHA_D, drifted)

    with pytest.raises(PortfolioStateError, match="bindings"):
        validate_transition(old, new, expected_parent_manifest_sha256=SHA_D)


def test_transition_rejects_parent_unrelated_to_verified_manifest(
    bindings: Bindings,
) -> None:
    fresh = CampaignState.fresh(bindings)
    unrelated = CampaignState("T1", "active", 1, SHA_C, bindings)

    with pytest.raises(PortfolioStateError, match="parent"):
        validate_transition(
            fresh,
            unrelated,
            expected_parent_manifest_sha256=bindings.input_manifest_sha256,
        )
    with pytest.raises(PortfolioStateError, match="parent"):
        fresh.advance(
            stage="T1",
            status="active",
            verified_parent_manifest_sha256=SHA_C,
        )


@pytest.mark.parametrize("expected", (None, b"b" * 64, "B" * 64, "b" * 63))
def test_transition_rejects_invalid_expected_parent_hash(
    bindings: Bindings, expected: object
) -> None:
    next_state = CampaignState("T1", "active", 1, SHA_B, bindings)

    with pytest.raises(PortfolioStateError, match="expected parent"):
        validate_transition(
            CampaignState.fresh(bindings),
            next_state,
            expected_parent_manifest_sha256=expected,
        )


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
        validate_transition(
            object(),
            CampaignState.fresh(bindings),
            expected_parent_manifest_sha256=SHA_B,
        )
    with pytest.raises(PortfolioStateError, match="state type"):
        validate_transition(
            CampaignState.fresh(bindings),
            object(),
            expected_parent_manifest_sha256=SHA_B,
        )


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
        validate_transition(
            CampaignState.fresh(bindings),
            forged,
            expected_parent_manifest_sha256=SHA_C,
        )


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
        forged.advance(
            stage="T2A",
            status="active",
            verified_parent_manifest_sha256=SHA_D,
        )


@pytest.mark.parametrize(
    "args",
    (
        ("other", "T1", 1, SHA_A, SHA_B, 1, SHA_C),
        (7, "T1", 1, SHA_A, SHA_B, 1, SHA_C),
        ("temporal_portfolio_v1", "fresh", 1, SHA_A, SHA_B, 1, SHA_C),
        ("temporal_portfolio_v1", "unknown", 1, SHA_A, SHA_B, 1, SHA_C),
        ("temporal_portfolio_v1", "T1", True, SHA_A, SHA_B, 1, SHA_C),
        ("temporal_portfolio_v1", "T1", 0, SHA_A, SHA_B, 1, SHA_C),
        ("temporal_portfolio_v1", "T1", 1, None, SHA_B, 1, SHA_C),
        ("temporal_portfolio_v1", "T1", 1, "A" * 64, SHA_B, 1, SHA_C),
        ("temporal_portfolio_v1", "T1", 1, SHA_A, "B" * 64, 1, SHA_C),
        ("temporal_portfolio_v1", "T1", 1, SHA_A, SHA_B, True, SHA_C),
        ("temporal_portfolio_v1", "T1", 1, SHA_A, SHA_B, 0, SHA_C),
        ("temporal_portfolio_v1", "T1", 1, SHA_A, SHA_B, 1, "C" * 64),
    ),
)
def test_lineage_rejects_invalid_or_incoherent_values(
    args: tuple[object, object, object, object, object, object, object]
) -> None:
    with pytest.raises(PortfolioStateError):
        Lineage(*args)


def test_nonfresh_lineage_represents_all_required_manifest_identity_fields(
    bindings: Bindings,
) -> None:
    checkpoint = CheckpointIdentity(SHA_C, SHA_D, 1)
    state = CampaignState.fresh(bindings).advance(
        stage="T1",
        status="active",
        verified_parent_manifest_sha256=SHA_B,
    )

    lineage = state.lineage(checkpoint)

    assert lineage == Lineage(
        "temporal_portfolio_v1", "T1", 1, SHA_B, SHA_C, 1, SHA_D
    )
    assert (
        lineage.campaign_id,
        lineage.stage,
        lineage.sequence,
        lineage.parent_manifest_sha256,
        lineage.training_sha256,
        lineage.state_schema_version,
        lineage.runtime_sha256,
    ) == (
        "temporal_portfolio_v1",
        "T1",
        1,
        SHA_B,
        SHA_C,
        1,
        SHA_D,
    )


def test_fresh_state_has_no_checkpoint_lineage(bindings: Bindings) -> None:
    with pytest.raises(PortfolioStateError, match="fresh"):
        CampaignState.fresh(bindings).lineage(CheckpointIdentity(SHA_A, SHA_C, 1))


def test_state_lineage_requires_exact_checkpoint_identity(bindings: Bindings) -> None:
    state = CampaignState("T1", "active", 1, SHA_B, bindings)

    with pytest.raises(PortfolioStateError, match="checkpoint"):
        state.lineage(object())


def test_state_value_objects_are_immutable(bindings: Bindings) -> None:
    state = CampaignState("T1", "active", 1, SHA_B, bindings)
    lineage = state.lineage(CheckpointIdentity(SHA_C, SHA_D, 1))

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

    validate_runtime(
        checkpoint,
        current_training_sha256=SHA_A,
        current_runtime_sha256=SHA_B,
    )


def test_runtime_rejects_changed_training_identity_even_when_runtime_matches() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="training"):
        validate_runtime(
            checkpoint,
            current_training_sha256=SHA_C,
            current_runtime_sha256=SHA_B,
        )


def test_checkpoint_parses_positive_but_production_rejects_unsupported_schema() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 2)

    assert checkpoint.state_schema_version == 2
    with pytest.raises(CompatibilityError, match="schema"):
        validate_runtime(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_B,
        )
    with pytest.raises(CompatibilityError, match="schema"):
        _validate_runtime_with_migrations(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_C,
            migrations={(SHA_B, SHA_C): 2},
        )


def test_runtime_change_needs_explicit_compatibility() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)
    with pytest.raises(CompatibilityError, match="runtime"):
        validate_runtime(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_C,
        )
    _validate_runtime_with_migrations(
        checkpoint,
        current_training_sha256=SHA_A,
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
        _validate_runtime_with_migrations(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_C,
            migrations=migrations,
        )


def test_runtime_snapshots_caller_mapping_exactly_once() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)
    migrations = _OneViewMapping((((SHA_B, SHA_C), 1),))

    _validate_runtime_with_migrations(
        checkpoint,
        current_training_sha256=SHA_A,
        current_runtime_sha256=SHA_C,
        migrations=migrations,
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
        _validate_runtime_with_migrations(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_C,
            migrations=_OneViewMapping(items),
        )


@pytest.mark.parametrize("item", ("ab", ["a", "b"], ("only-one",)))
def test_runtime_rejects_malformed_mapping_items(item: object) -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="migration"):
        _validate_runtime_with_migrations(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_C,
            migrations=_MalformedItemsMapping(item),
        )


def test_runtime_rejects_mapping_snapshot_failures() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="snapshot"):
        _validate_runtime_with_migrations(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_C,
            migrations=_ExplodingItemsMapping(),
        )


def test_runtime_validates_migrations_even_when_runtime_matches() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="migration"):
        _validate_runtime_with_migrations(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_B,
            migrations={(SHA_B, SHA_C): True},
        )


def test_runtime_rejects_invalid_call_types() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(CompatibilityError, match="checkpoint"):
        validate_runtime(
            object(),
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_B,
        )
    with pytest.raises(CompatibilityError, match="current training"):
        validate_runtime(
            checkpoint,
            current_training_sha256=b"a" * 64,
            current_runtime_sha256=SHA_B,
        )
    with pytest.raises(CompatibilityError, match="current runtime"):
        validate_runtime(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=b"b" * 64,
        )


def test_public_runtime_validator_does_not_accept_caller_migrations() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, STATE_SCHEMA_VERSION)

    with pytest.raises(TypeError, match="migrations"):
        validate_runtime(
            checkpoint,
            current_training_sha256=SHA_A,
            current_runtime_sha256=SHA_C,
            migrations={(SHA_B, SHA_C): STATE_SCHEMA_VERSION},
        )


def test_production_runtime_migration_registry_is_empty_and_immutable() -> None:
    assert isinstance(COMPATIBLE_RUNTIME_MIGRATIONS, MappingProxyType)
    assert dict(COMPATIBLE_RUNTIME_MIGRATIONS) == {}
    with pytest.raises(TypeError):
        COMPATIBLE_RUNTIME_MIGRATIONS[(SHA_B, SHA_C)] = 1


def test_checkpoint_identity_is_immutable() -> None:
    checkpoint = CheckpointIdentity(SHA_A, SHA_B, 1)

    with pytest.raises(FrozenInstanceError):
        checkpoint.state_schema_version = 2
