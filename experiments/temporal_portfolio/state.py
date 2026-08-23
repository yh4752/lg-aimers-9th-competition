"""Immutable campaign state and lineage for the temporal portfolio."""
from __future__ import annotations

from dataclasses import dataclass

from .compatibility import CheckpointIdentity


class PortfolioStateError(ValueError):
    """Raised when campaign state or lineage is invalid."""


ALLOWED_STAGE_ORDER = (
    "fresh",
    "T1",
    "T2A",
    "T2B",
    "T3A",
    "T3BT4",
    "T5A",
    "T5B",
)
ALLOWED_STATUS = (
    "fresh",
    "active",
    "completed",
    "budget_inconclusive",
    "failed",
    "rule_blocked",
)

_CAMPAIGN_ID = "temporal_portfolio_v1"
_STAGE_RANK = {stage: rank for rank, stage in enumerate(ALLOWED_STAGE_ORDER)}
_RETRYABLE_STATUS = frozenset({"budget_inconclusive", "failed"})


@dataclass(frozen=True)
class Bindings:
    campaign_id: str
    contract_sha256: str
    input_manifest_sha256: str

    def __post_init__(self) -> None:
        _validate_bindings(self)


@dataclass(frozen=True)
class Lineage:
    campaign_id: str
    stage: str
    sequence: int
    parent_manifest_sha256: str
    training_sha256: str
    state_schema_version: int
    runtime_sha256: str

    def __post_init__(self) -> None:
        _validate_lineage(self)


@dataclass(frozen=True)
class CampaignState:
    stage: str
    status: str
    sequence: int
    parent_manifest_sha256: str | None
    bindings: Bindings

    def __post_init__(self) -> None:
        _validate_state(self)

    @classmethod
    def fresh(cls, bindings: Bindings) -> CampaignState:
        return cls("fresh", "fresh", 0, None, bindings)

    def lineage(self, checkpoint: CheckpointIdentity) -> Lineage:
        _validate_state(self)
        if self.stage == "fresh":
            raise PortfolioStateError("fresh state has no checkpoint lineage")
        if type(checkpoint) is not CheckpointIdentity:
            raise PortfolioStateError("lineage checkpoint identity has an invalid type")
        try:
            return Lineage(
                self.bindings.campaign_id,
                self.stage,
                self.sequence,
                self.parent_manifest_sha256,
                checkpoint.training_sha256,
                checkpoint.state_schema_version,
                checkpoint.runtime_sha256,
            )
        except AttributeError as error:
            raise PortfolioStateError("lineage checkpoint identity is malformed") from error

    def advance(
        self,
        *,
        stage: str,
        status: str,
        verified_parent_manifest_sha256: str,
    ) -> CampaignState:
        _validate_state(self)
        candidate = CampaignState(
            stage,
            status,
            self.sequence + 1,
            verified_parent_manifest_sha256,
            self.bindings,
        )
        validate_transition(
            self,
            candidate,
            expected_parent_manifest_sha256=verified_parent_manifest_sha256,
        )
        return candidate


def validate_transition(
    old: CampaignState,
    new: CampaignState,
    *,
    expected_parent_manifest_sha256: str,
) -> None:
    if type(old) is not CampaignState or type(new) is not CampaignState:
        raise PortfolioStateError("state type is invalid")
    _validate_state(old)
    _validate_state(new)
    if not _is_sha256(expected_parent_manifest_sha256):
        raise PortfolioStateError("expected parent manifest SHA-256 is invalid")
    if old.bindings != new.bindings:
        raise PortfolioStateError("state bindings differ")
    if (
        old.stage == "fresh"
        and expected_parent_manifest_sha256
        != old.bindings.input_manifest_sha256
    ):
        raise PortfolioStateError("fresh state parent differs from the input manifest")
    if new.parent_manifest_sha256 != expected_parent_manifest_sha256:
        raise PortfolioStateError("state parent differs from the verified manifest")
    if new.sequence != old.sequence + 1:
        raise PortfolioStateError("state sequence must increase by exactly one")
    if old.status == "rule_blocked":
        raise PortfolioStateError("rule_blocked state has no successor")

    old_rank = _STAGE_RANK[old.stage]
    new_rank = _STAGE_RANK[new.stage]
    if new_rank < old_rank:
        raise PortfolioStateError("state stage moved backward")
    if old.stage == "fresh":
        if new.stage != "T1":
            raise PortfolioStateError("fresh state must enter stage T1")
        return
    if new_rank == old_rank:
        if old.status == "active":
            return
        if old.status in _RETRYABLE_STATUS:
            if new.status != "active":
                raise PortfolioStateError(
                    "retryable state must become active before another terminal status"
                )
            return
        if old.status == "completed":
            if new.status != "completed":
                raise PortfolioStateError(
                    "completed state may only remain completed on the same stage"
                )
            return
        raise PortfolioStateError("state status transition is not allowed")
    if old.status != "completed":
        raise PortfolioStateError("only a completed state may move to a later stage")


def _validate_bindings(value: object) -> None:
    if type(value) is not Bindings:
        raise PortfolioStateError("state bindings have an invalid type")
    try:
        campaign_id = value.campaign_id
        contract_sha256 = value.contract_sha256
        input_manifest_sha256 = value.input_manifest_sha256
    except AttributeError as error:
        raise PortfolioStateError("state bindings are malformed") from error
    if type(campaign_id) is not str or campaign_id != _CAMPAIGN_ID:
        raise PortfolioStateError("state bindings campaign_id is not approved")
    if not _is_sha256(contract_sha256):
        raise PortfolioStateError("state bindings contract SHA-256 is invalid")
    if not _is_sha256(input_manifest_sha256):
        raise PortfolioStateError("state bindings input manifest SHA-256 is invalid")


def _validate_lineage(value: object) -> None:
    if type(value) is not Lineage:
        raise PortfolioStateError("lineage has an invalid type")
    try:
        campaign_id = value.campaign_id
        stage = value.stage
        sequence = value.sequence
        parent_manifest_sha256 = value.parent_manifest_sha256
        training_sha256 = value.training_sha256
        state_schema_version = value.state_schema_version
        runtime_sha256 = value.runtime_sha256
    except AttributeError as error:
        raise PortfolioStateError("lineage is malformed") from error
    if type(campaign_id) is not str or campaign_id != _CAMPAIGN_ID:
        raise PortfolioStateError("lineage campaign_id is not approved")
    if type(stage) is not str or stage not in _STAGE_RANK or stage == "fresh":
        raise PortfolioStateError("lineage stage must be a non-fresh campaign stage")
    if type(sequence) is not int or sequence <= 0:
        raise PortfolioStateError("lineage sequence must be a positive integer")
    if not _is_sha256(parent_manifest_sha256):
        raise PortfolioStateError("lineage parent manifest SHA-256 is invalid")
    if not _is_sha256(training_sha256):
        raise PortfolioStateError("lineage training SHA-256 is invalid")
    if type(state_schema_version) is not int or state_schema_version <= 0:
        raise PortfolioStateError("lineage state schema version must be positive")
    if not _is_sha256(runtime_sha256):
        raise PortfolioStateError("lineage runtime SHA-256 is invalid")


def _validate_state(value: object) -> None:
    if type(value) is not CampaignState:
        raise PortfolioStateError("state type is invalid")
    try:
        stage = value.stage
        status = value.status
        sequence = value.sequence
        parent_manifest_sha256 = value.parent_manifest_sha256
        bindings = value.bindings
    except AttributeError as error:
        raise PortfolioStateError("state is malformed") from error
    _validate_bindings(bindings)
    if type(stage) is not str or stage not in _STAGE_RANK:
        raise PortfolioStateError("state stage is not allowed")
    if type(status) is not str or status not in ALLOWED_STATUS:
        raise PortfolioStateError("state status is not allowed")
    if (stage == "fresh") != (status == "fresh"):
        raise PortfolioStateError("state stage and status are incoherent")
    if type(sequence) is not int or sequence < 0:
        raise PortfolioStateError("state sequence must be a non-negative integer")
    if stage == "fresh":
        if sequence != 0 or parent_manifest_sha256 is not None:
            raise PortfolioStateError("fresh state lineage is invalid")
    else:
        if sequence == 0:
            raise PortfolioStateError("non-fresh state sequence must be positive")
        if not _is_sha256(parent_manifest_sha256):
            raise PortfolioStateError("non-fresh state needs a parent manifest SHA-256")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
