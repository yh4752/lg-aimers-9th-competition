"""Immutable campaign state and lineage for the temporal portfolio."""
from __future__ import annotations

from dataclasses import dataclass


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
    sequence: int
    parent_manifest_sha256: str | None

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

    @property
    def lineage(self) -> Lineage:
        return Lineage(
            self.bindings.campaign_id,
            self.sequence,
            self.parent_manifest_sha256,
        )

    def advance(
        self, *, stage: str, status: str, parent_manifest_sha256: str
    ) -> CampaignState:
        _validate_state(self)
        candidate = CampaignState(
            stage,
            status,
            self.sequence + 1,
            parent_manifest_sha256,
            self.bindings,
        )
        validate_transition(self, candidate)
        return candidate


def validate_transition(old: CampaignState, new: CampaignState) -> None:
    if type(old) is not CampaignState or type(new) is not CampaignState:
        raise PortfolioStateError("state type is invalid")
    _validate_state(old)
    _validate_state(new)
    if old.bindings != new.bindings:
        raise PortfolioStateError("state bindings differ")
    if new.sequence != old.sequence + 1:
        raise PortfolioStateError("state sequence must increase by exactly one")
    if _STAGE_RANK[new.stage] < _STAGE_RANK[old.stage]:
        raise PortfolioStateError("state stage moved backward")


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
        sequence = value.sequence
        parent_manifest_sha256 = value.parent_manifest_sha256
    except AttributeError as error:
        raise PortfolioStateError("lineage is malformed") from error
    if type(campaign_id) is not str or campaign_id != _CAMPAIGN_ID:
        raise PortfolioStateError("lineage campaign_id is not approved")
    if type(sequence) is not int or sequence < 0:
        raise PortfolioStateError("lineage sequence must be a non-negative integer")
    if sequence == 0:
        if parent_manifest_sha256 is not None:
            raise PortfolioStateError("fresh lineage must not have a parent manifest")
    elif not _is_sha256(parent_manifest_sha256):
        raise PortfolioStateError("non-fresh lineage needs a parent manifest SHA-256")


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
