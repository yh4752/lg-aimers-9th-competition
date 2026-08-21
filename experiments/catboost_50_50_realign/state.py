from __future__ import annotations

from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Mapping


class RealignStateError(ValueError):
    """Raised when campaign state is malformed or moves backward."""


@dataclass(frozen=True)
class Bindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str


@dataclass(frozen=True)
class CampaignState:
    status: str
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None
    decision_sha256: str | None
    selected_tree_count: int | None
    bindings: Bindings

    def validate(self) -> None:
        _validate_state(self)


ALLOWED_STATUS = (
    "fresh",
    "f1_active",
    "f1_complete",
    "deployment_blocked",
    "full_fit_active",
    "completed",
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_F1 = ("tabm_f1_2022", "catboost_f1_2022")
_ALL = (*_F1, "catboost_full_2024")
_TREE_COUNTS = (4, 8, 12, 16, 20, 24, 28, 32)


def bindings_payload(bindings: Bindings) -> dict[str, str]:
    values = {
        "contract_sha256": bindings.contract_sha256,
        "code_sha256": bindings.code_sha256,
        "input_manifest_sha256": bindings.input_manifest_sha256,
        "train_sha256": bindings.train_sha256,
        "history_sha256": bindings.history_sha256,
    }
    if any(_SHA_RE.fullmatch(value) is None for value in values.values()):
        raise RealignStateError("state bindings differ")
    return values


def _validate_state(state: CampaignState) -> None:
    bindings_payload(state.bindings)
    if state.status not in ALLOWED_STATUS:
        raise RealignStateError("state status differs")
    if len(state.completed_job_ids) != len(set(state.completed_job_ids)):
        raise RealignStateError("completed job list differs")
    if state.decision_sha256 is not None and _SHA_RE.fullmatch(state.decision_sha256) is None:
        raise RealignStateError("decision SHA-256 differs")
    valid = False
    if state.status == "fresh":
        valid = (
            state.completed_job_ids == ()
            and state.active_job_id is None
            and state.decision_sha256 is None
            and state.selected_tree_count is None
        )
    elif state.status == "f1_active":
        valid = (
            (state.completed_job_ids, state.active_job_id)
            in (((), "tabm_f1_2022"), (_F1[:1], "catboost_f1_2022"))
            and state.decision_sha256 is None
            and state.selected_tree_count is None
        )
    elif state.status == "f1_complete":
        valid = (
            state.completed_job_ids == _F1
            and state.active_job_id is None
            and state.decision_sha256 is None
            and state.selected_tree_count is None
        )
    elif state.status == "deployment_blocked":
        valid = (
            state.completed_job_ids == _F1
            and state.active_job_id is None
            and state.decision_sha256 is not None
            and state.selected_tree_count is None
        )
    elif state.status == "full_fit_active":
        valid = (
            state.completed_job_ids == _F1
            and state.active_job_id == "catboost_full_2024"
            and state.decision_sha256 is not None
            and state.selected_tree_count in _TREE_COUNTS
        )
    elif state.status == "completed":
        if state.completed_job_ids != _ALL:
            raise RealignStateError("completed state requires the full model")
        valid = (
            state.active_job_id is None
            and state.decision_sha256 is not None
            and state.selected_tree_count in _TREE_COUNTS
        )
    if not valid:
        raise RealignStateError("state fields differ")


def serialize_state(state: CampaignState) -> bytes:
    state.validate()
    return json.dumps(
        {
            "schema_version": 1,
            "campaign_id": "catboost_50_50_realign_v2",
            "status": state.status,
            "completed_job_ids": list(state.completed_job_ids),
            "active_job_id": state.active_job_id,
            "decision_sha256": state.decision_sha256,
            "selected_tree_count": state.selected_tree_count,
            "bindings": bindings_payload(state.bindings),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def deserialize_state(payload: bytes) -> CampaignState:
    try:
        root = json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise RealignStateError("state JSON is unreadable") from error
    expected = {
        "schema_version",
        "campaign_id",
        "status",
        "completed_job_ids",
        "active_job_id",
        "decision_sha256",
        "selected_tree_count",
        "bindings",
    }
    if (
        type(root) is not dict
        or set(root) != expected
        or root["schema_version"] != 1
        or root["campaign_id"] != "catboost_50_50_realign_v2"
        or type(root["status"]) is not str
        or type(root["completed_job_ids"]) is not list
        or any(type(item) is not str for item in root["completed_job_ids"])
        or (root["active_job_id"] is not None and type(root["active_job_id"]) is not str)
        or type(root["bindings"]) is not dict
        or set(root["bindings"])
        != {
            "contract_sha256",
            "code_sha256",
            "input_manifest_sha256",
            "train_sha256",
            "history_sha256",
        }
    ):
        raise RealignStateError("state schema differs")
    bindings = Bindings(**root["bindings"])
    state = CampaignState(
        status=root["status"],
        completed_job_ids=tuple(root["completed_job_ids"]),
        active_job_id=root["active_job_id"],
        decision_sha256=root["decision_sha256"],
        selected_tree_count=root["selected_tree_count"],
        bindings=bindings,
    )
    state.validate()
    if serialize_state(state) != payload:
        raise RealignStateError("state JSON is not canonical")
    return state


def validate_transition(previous: CampaignState, current: CampaignState) -> None:
    previous.validate()
    current.validate()
    if previous.bindings != current.bindings:
        raise RealignStateError("state transition changed bindings")
    allowed: Mapping[str, set[str]] = MappingProxyType(
        {
            "fresh": {"fresh", "f1_active"},
            "f1_active": {"f1_active", "f1_complete"},
            "f1_complete": {
                "f1_complete",
                "deployment_blocked",
                "full_fit_active",
            },
            "deployment_blocked": {"deployment_blocked"},
            "full_fit_active": {"full_fit_active", "completed"},
            "completed": {"completed"},
        }
    )
    if current.status not in allowed[previous.status]:
        raise RealignStateError("state transition is not allowed")
    if not set(previous.completed_job_ids).issubset(current.completed_job_ids):
        raise RealignStateError("state transition removed completed work")
